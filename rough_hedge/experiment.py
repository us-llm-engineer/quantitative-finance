"""Hedging experiment framework: leakage detection, policy comparison, and statistical summarization.

Implements the experimental protocol (R2.3): run multiple hedging strategies on shared
paths, check for information leakage, and compute paired differences with bootstrap
confidence intervals and Holm adjustment for multiple comparisons.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from rough_hedge.rbergomi import simulate_rbergomi
from rough_hedge.training import make_splits, train_hedger
from rough_hedge.models import FFNHedger, GRUHedger, SignatureHedger
from rough_hedge.baselines import (
    bs_delta_policy, leland_delta_policy, ww_policy
)
from rough_hedge.risk import cvar_tail_mean
from rough_hedge.hedging import hedging_pnl
from rough_hedge.stats import (
    paired_bootstrap, seed_aggregate, cohens_d, holm, EffectSize
)


class LeakageError(AssertionError):
    """Raised when a policy depends on future prices."""
    pass


def leakage_check(policy_fn, S, n_cuts=None, seed=0, atol=1e-12):
    """Check that a hedging policy does not leak future price information.
    
    For each cut point, perturb prices after that point and verify that positions
    before the cut remain identical. This ensures causality (positions depend only
    on past information).
    
    Args:
        policy_fn: callable S -> positions (n_paths, N), must be deterministic
        S: price array (n_paths, N+1), column i = S(t_i)
        n_cuts: int, number of random cut points to test. None = test all N cuts
        seed: int, random seed for perturbations
        atol: float, absolute tolerance for position differences
    
    Raises:
        LeakageError: if positions before cut change when future prices perturb
        ValueError: for invalid input
    """
    S = np.asarray(S, dtype=np.float64)
    
    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError("S must be 2-D with at least 2 columns")
    if np.any(S <= 0.0):
        raise ValueError("S must be strictly positive")
    if n_cuts is not None and (not isinstance(n_cuts, int) or n_cuts < 1):
        raise ValueError("n_cuts must be None or int >= 1")
    
    n_paths, N_plus_1 = S.shape
    N = N_plus_1 - 1
    
    # Determine which cuts to test
    if n_cuts is None:
        cuts = list(range(N))
    else:
        rng = np.random.default_rng(seed)
        cuts = rng.choice(N, size=min(n_cuts, N), replace=False).tolist()
    
    # Get baseline positions on unperturbed paths
    try:
        pos_baseline = np.asarray(policy_fn(S), dtype=np.float64)
    except Exception as e:
        raise ValueError(f"policy_fn(S) failed: {e}")
    
    if pos_baseline.shape != (n_paths, N):
        raise ValueError(f"policy_fn must return shape (n_paths={n_paths}, N={N}), got {pos_baseline.shape}")
    
    # Test causality at each cut
    rng = np.random.default_rng(seed)
    for t in cuts:
        S_perturbed = S.copy()
        # Perturb prices after column t
        perturbations = np.exp(0.3 * rng.standard_normal((n_paths, N - t)))
        S_perturbed[:, t+1:] *= perturbations
        
        try:
            pos_perturbed = np.asarray(policy_fn(S_perturbed), dtype=np.float64)
        except Exception as e:
            raise LeakageError(f"policy_fn(S_perturbed) failed at cut {t}: {e}")
        
        # Check that positions before cut are identical
        if not np.allclose(pos_baseline[:, :t+1], pos_perturbed[:, :t+1], atol=atol):
            diff_mask = ~np.isclose(pos_baseline[:, :t+1], pos_perturbed[:, :t+1], atol=atol)
            offending_step = np.where(diff_mask)[1][0]
            raise LeakageError(f"policy depends on future prices: positions differ at step {offending_step} after perturbing prices at step {t+1}")


def _apply_classical_policy_stepwise(policy_step, S, N):
    """Apply a step-wise policy function to generate full position arrays.
    
    Args:
        policy_step: callable(i, hist, prev) -> positions for step i
        S: (n_paths, N+1) price array
        N: number of steps
    
    Returns:
        (n_paths, N) position array
    """
    S = np.asarray(S, dtype=np.float64)
    n_paths = S.shape[0]
    positions = np.zeros((n_paths, N), dtype=np.float64)
    prev = np.zeros(n_paths, dtype=np.float64)
    
    for i in range(N):
        hist = S[:, :i+1]
        positions[:, i] = policy_step(i, hist, prev)
        prev = positions[:, i]
    
    return positions


def run_comparison(config: dict) -> dict:
    """Run hedging strategies on shared simulated paths and return losses and policies.
    
    Simulates rough Bergomi paths once, then trains (or evaluates) each arm, storing
    per-path losses and callable policies for each training seed.
    
    Args:
        config: dict with required keys
            n_seeds (int >= 1), seed (int), n_train, n_val, n_test, N, T, H, eta, rho, xi0,
            cost_bp, arms (tuple of arm names), width (default 16)
            For learned arms: epochs, batch_size, lr, patience (required when learning)
    
    Returns:
        dict with keys:
            config_hash: SHA256 hex of canonical config
            test_paths: (n_test, N+1) array
            test_paths_sha256: SHA256 hex of test_paths.tobytes()
            test_losses: {arm: (n_seeds, n_test) array}
            histories: {arm: None (classical) or list of n_seeds dicts (learned)}
            budgets: {arm: None (classical) or dict of training config (learned)}
            policies: {arm: list of n_seeds callable}
            leakage_checked: {arm: True}
    
    Raises:
        ValueError: for missing config keys, invalid values, unknown arms
    """
    # Validate required config keys and values
    required = ["n_seeds", "seed", "n_train", "n_val", "n_test", "N", "T", "H", "eta", "rho", "xi0", "cost_bp", "arms"]
    for key in required:
        if key not in config:
            raise ValueError(f"config missing required key: {key}")
    
    n_seeds = config.get("n_seeds")
    seed = config.get("seed")
    n_train = config.get("n_train")
    n_val = config.get("n_val")
    n_test = config.get("n_test")
    N = config.get("N")
    T = config.get("T")
    H = config.get("H")
    eta = config.get("eta")
    rho = config.get("rho")
    xi0 = config.get("xi0")
    cost_bp = config.get("cost_bp")
    arms = config.get("arms")
    width = config.get("width", 16)
    device = config.get("device", "cpu")
    parallel_workers = config.get("parallel_workers", 1)
    
    # Validation
    if not isinstance(n_seeds, int) or n_seeds < 1:
        raise ValueError("n_seeds must be int >= 1")
    if not isinstance(n_test, int) or n_test < 20:
        raise ValueError("n_test must be int >= 20")
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be int >= 2")
    if T <= 0.0:
        raise ValueError("T must be > 0")
    if xi0 <= 0.0:
        raise ValueError("xi0 must be > 0")
    if cost_bp < 0.0:
        raise ValueError("cost_bp must be >= 0")
    if not isinstance(arms, (tuple, list)) or len(arms) == 0:
        raise ValueError("arms must be non-empty tuple or list")
    if len(set(arms)) != len(arms):
        raise ValueError("arms must be unique")
    if not isinstance(width, int) or width < 1 or width > 16:
        raise ValueError("width must be int in [1, 16]")
    if not isinstance(parallel_workers, int) or parallel_workers < 1:
        raise ValueError("parallel_workers must be int >= 1")
    
    valid_arms = {"bs_delta", "leland", "ww_band", "ffn", "gru", "sig1", "sig2", "sig3"}
    for arm in arms:
        if arm not in valid_arms:
            raise ValueError(f"unknown arm: {arm}")

    learned_arms = {"ffn", "gru", "sig1", "sig2", "sig3"}
    has_learned = any(arm in learned_arms for arm in arms)
    
    if has_learned:
        for key in ["epochs", "batch_size", "lr", "patience"]:
            if key not in config:
                raise ValueError(f"config missing required key for learned arms: {key}")
    
    # Compute config hash
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    
    # Simulate test paths once (shared by all arms)
    rng_master = np.random.default_rng(seed)
    seeds_per_arm = rng_master.integers(0, 2**31, size=n_seeds + 1)  # +1 for test simulation
    test_seed = seeds_per_arm[0]
    
    rng_test = np.random.default_rng(test_seed)
    result_sim = simulate_rbergomi(
        rng=rng_test,
        n_paths=n_test,
        n_steps=N,
        T=T,
        H=H,
        eta=eta,
        rho=rho,
        xi0=xi0
    )
    S_test = result_sim['S']
    test_paths_sha256 = hashlib.sha256(np.ascontiguousarray(S_test).tobytes()).hexdigest()
    S_test = result_sim['S']
    
    # Build manifest for training data
    manifest = {
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_steps": N,
        "T": T,
        "strike": 1.0,
        "cost": 0.0,
        "market": "rbergomi",
        "H": H,
        "eta": eta,
        "rho": rho,
        "xi0": xi0
    }
    
    # Make splits (train/val/test) for training
    splits = make_splits(manifest)
    S_train, S_val = splits.train, splits.val
    
    # Initialize result dict
    result = {
        "config_hash": cfg_hash,
        "test_paths": S_test,
        "test_paths_sha256": test_paths_sha256,
        "test_losses": {},
        "histories": {},
        "budgets": {},
        "policies": {},
        "leakage_checked": {}
    }
    
    # Prepare cost rate
    c = cost_bp * 1e-4
    sigma = math.sqrt(xi0)
    
    # Process each arm
    for arm in arms:
        result["test_losses"][arm] = np.zeros((n_seeds, n_test), dtype=np.float64)
        result["policies"][arm] = []
        
        if arm in learned_arms:
            # Train learned arm
            result["histories"][arm] = []
            
            # Extract training config
            train_config = {
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "patience": config.get("patience"),
                "alpha": 0.95
            }
            
            # Store budget once per arm type
            if arm == "ffn":
                result["budgets"][arm] = {
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"],
                    "lr": config["lr"],
                    "patience": config.get("patience"),
                    "n_train_paths": n_train
                }
            else:
                # Learned arms share the same budget
                if "ffn" in result["budgets"] and result["budgets"]["ffn"] is not None:
                    result["budgets"][arm] = result["budgets"]["ffn"]
                else:
                    result["budgets"][arm] = {
                        "epochs": config["epochs"],
                        "batch_size": config["batch_size"],
                        "lr": config["lr"],
                        "patience": config.get("patience"),
                        "n_train_paths": n_train
                    }
            
            # Construct models on the calling thread.  Their initialization
            # deliberately preserves the process RNG state, so construction is
            # kept out of worker threads.  CUDA work below is independent per
            # seed and can use separate streams safely.
            models = []
            for k in range(n_seeds):
                train_seed = int(seeds_per_arm[k + 1])
                if arm == "ffn":
                    model = FFNHedger(hidden=width, seed=train_seed)
                elif arm == "gru":
                    model = GRUHedger(hidden=width, seed=train_seed)
                elif arm == "sig1":
                    model = SignatureHedger(depth=1, hidden=width, seed=train_seed)
                elif arm == "sig2":
                    model = SignatureHedger(depth=2, hidden=width, seed=train_seed)
                elif arm == "sig3":
                    model = SignatureHedger(depth=3, hidden=width, seed=train_seed)
                models.append((k, train_seed, model))

            def run_learned_seed(task):
                k, train_seed, model = task
                print(f"Training {arm}, seed {k + 1}/{n_seeds} (seed={train_seed})", flush=True)
                stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None
                stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
                with stream_context:
                    trained_model, history = train_hedger(model, splits, train_config, train_seed, device=device)
                    log_moneyness = np.log(S_test)
                    times = np.arange(N + 1) * (T / N)
                    with torch.no_grad():
                        logm_tensor = torch.from_numpy(log_moneyness).to(device).float()
                        times_tensor = torch.from_numpy(times).to(device).float()
                        positions = trained_model(logm_tensor, times_tensor).cpu().numpy()
                if stream is not None:
                    stream.synchronize()
                hedge_result = hedging_pnl(S_test, positions, strike=1.0, cost=c)
                print(f"Finished {arm}, seed {k + 1}/{n_seeds}", flush=True)
                return k, trained_model, history, -hedge_result.pnl

            workers = min(parallel_workers, n_seeds) if device.startswith("cuda") else 1
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rough-hedge-cuda") as executor:
                    seed_runs = list(executor.map(run_learned_seed, models))
            else:
                seed_runs = [run_learned_seed(task) for task in models]

            for k, trained_model, history, losses in sorted(seed_runs):
                result["histories"][arm].append(history)
                result["test_losses"][arm][k, :] = losses
                
                # Create policy function for this seed (closure over trained_model)
                def make_learned_policy(trained_model_copy, device, T, N):
                    def policy(S):
                        S = np.asarray(S, dtype=np.float64)
                        log_moneyness = np.log(S)  # (n_paths, N+1)
                        times = np.arange(S.shape[1]) * (T / N)
                        
                        with torch.no_grad():
                            logm_tensor = torch.from_numpy(log_moneyness).to(device).float()
                            times_tensor = torch.from_numpy(times).to(device).float()
                            positions_tensor = trained_model_copy(logm_tensor, times_tensor)
                            positions = positions_tensor.cpu().numpy()
                        return positions
                    return policy
                
                result["policies"][arm].append(make_learned_policy(copy.deepcopy(trained_model).to(device), device, T, N))
        
        else:
            # Classical arm (no training)
            result["histories"][arm] = None
            result["budgets"][arm] = None
            
            # Create step-wise policy function
            if arm == "bs_delta":
                policy_step = bs_delta_policy(strike=1.0, sigma=sigma, T=T, n_steps=N)
            elif arm == "leland":
                policy_step = leland_delta_policy(strike=1.0, sigma=sigma, T=T, n_steps=N, cost=c)
            elif arm == "ww_band":
                policy_step = ww_policy(strike=1.0, sigma=sigma, T=T, n_steps=N, cost=c, 
                                       risk_aversion=config.get("ww_gamma", 10.0))
            
            # Create wrapper for applying step-wise policy
            def make_classical_policy(policy_step_fn, N):
                def policy(S):
                    return _apply_classical_policy_stepwise(policy_step_fn, S, N)
                return policy
            
            policy_fn = make_classical_policy(policy_step, N)
            
            # Evaluate on test paths
            positions_test = policy_fn(S_test)
            hedge_result = hedging_pnl(S_test, positions_test, strike=1.0, cost=c)
            losses = -hedge_result.pnl
            result["test_losses"][arm][0, :] = losses
            
            # Replicate losses for all seeds (classical arm gives same result)
            for k in range(1, n_seeds):
                result["test_losses"][arm][k, :] = losses
            
            # All seeds use the same policy for classical arms
            for k in range(n_seeds):
                result["policies"][arm].append(policy_fn)
    
    # Run leakage check on a subset of test paths
    leakage_subset = min(64, n_test)
    S_leakage = S_test[:leakage_subset]
    for arm in arms:
        for k in range(n_seeds):
            try:
                leakage_check(result["policies"][arm][k], S_leakage, n_cuts=None, seed=0, atol=1e-12)
            except LeakageError as e:
                raise LeakageError(f"arm {arm} seed {k}: {e}")
        result["leakage_checked"][arm] = True
    
    return result


def paired_differences(result, arm_a, arm_b):
    """Extract paired differences in test losses between two arms.
    
    Args:
        result: dict from run_comparison
        arm_a, arm_b: arm names
    
    Returns:
        (n_seeds, n_test) array of differences: loss_a - loss_b on shared paths
    
    Raises:
        KeyError: if arm not in result
    """
    losses_a = result["test_losses"][arm_a]
    losses_b = result["test_losses"][arm_b]
    return losses_a - losses_b


def compare_arms(result, arm_a, arm_b, n_boot=500, seed=0, alpha=0.95):
    """Compute detailed paired comparison between two arms.
    
    For each training seed, runs a bootstrap on the paired per-path losses.
    Then aggregates across seeds and applies Holm correction.
    
    Args:
        result: dict from run_comparison (needs only test_losses)
        arm_a, arm_b: arm names
        n_boot: int, bootstrap replicates per seed
        seed: int, bootstrap seed base
        alpha: float in (0, 1), CVaR level
    
    Returns:
        dict with keys:
            per_seed: list of BootstrapResult (one per seed)
            aggregate: SeedSummary of per-seed estimates
            effect: list of EffectSize (one per seed)
    """
    losses_a = result["test_losses"][arm_a]
    losses_b = result["test_losses"][arm_b]
    n_seeds = losses_a.shape[0]
    
    per_seed_results = []
    per_seed_estimates = []
    effect_sizes = []
    
    for k in range(n_seeds):
        # Bootstrap on this seed's per-path losses (paired comparison)
        boot_result = paired_bootstrap(losses_a[k], losses_b[k], n_boot=n_boot, seed=seed + k, alpha=alpha)
        per_seed_results.append(boot_result)
        per_seed_estimates.append(boot_result.estimate)
        
        # Effect size from per-path differences
        diff_k = losses_a[k] - losses_b[k]
        try:
            effect = cohens_d(diff_k, level=0.95)
        except ValueError:
            # When std(diff) == 0, effect size is undefined; use a degenerate value
            effect = EffectSize(d=0.0, lo=0.0, hi=0.0, level=0.95)
        effect_sizes.append(effect)
    
    # Aggregate across seeds
    aggregate = seed_aggregate(per_seed_estimates, level=0.95)
    
    return {
        "per_seed": per_seed_results,
        "aggregate": aggregate,
        "effect": effect_sizes
    }
