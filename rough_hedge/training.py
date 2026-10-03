"""Hedger training, data splits, and loss functions for rough volatility hedging.

Implements the training protocol (R2.2): manifest generation, data splitting with
separate random streams, CVaR loss computation, and training with early stopping.
API: tests/api/R2_2.md.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from collections import namedtuple
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import psutil
import os

import threading
import time

from rough_hedge.rbergomi import simulate_rbergomi
from rough_hedge.hedging import hedging_pnl
from rough_hedge.risk import ru_objective, cvar_tail_mean

# Splits NamedTuple for returning train/val/test data with manifest
Splits = namedtuple("Splits", ["train", "val", "test", "manifest", "sha256"])


def manifest_sha256(manifest: dict) -> str:
    """Compute SHA256 hash of the canonical JSON representation of the manifest.
    
    The manifest must contain required fields for the market type and is hashed
    with sorted keys and no spaces.
    """
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def make_splits(manifest: dict) -> Splits:
    """Generate train/val/test splits from the manifest specification.
    
    Uses separate random streams (via SeedSequence.spawn) for train, val, test
    to ensure non-overlapping paths. Market can be "bs" (Black-Scholes) or
    "rbergomi" (rough Bergomi).
    
    Args:
        manifest: dict with keys
            seed (int >= 0), n_train, n_val, n_test (ints >= 1),
            n_steps (N >= 1), T > 0, strike > 0, cost >= 0,
            market ("bs" or "rbergomi")
            If market == "bs": sigma > 0
            If market == "rbergomi": H in (0, 0.5), eta >= 0, rho in [-1,1], xi0 > 0
    
    Returns:
        Splits: train, val, test (float64 ndarrays, shape (n, N+1)),
                manifest, sha256 (str)
    """
    # Validate manifest
    for key in ("seed", "n_train", "n_val", "n_test", "n_steps", "T", "strike", "cost", "market"):
        if key not in manifest:
            raise ValueError(f"Missing required key: {key}")
    
    seed = manifest["seed"]
    n_train = manifest["n_train"]
    n_val = manifest["n_val"]
    n_test = manifest["n_test"]
    N = manifest["n_steps"]
    T = manifest["T"]
    strike = manifest["strike"]
    cost = manifest["cost"]
    market = manifest["market"]
    
    # Type checks
    if not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(n_train, numbers.Integral) or n_train < 1:
        raise ValueError("n_train must be an integer >= 1")
    if not isinstance(n_val, numbers.Integral) or n_val < 1:
        raise ValueError("n_val must be an integer >= 1")
    if not isinstance(n_test, numbers.Integral) or n_test < 1:
        raise ValueError("n_test must be an integer >= 1")
    if not isinstance(N, numbers.Integral) or N < 1:
        raise ValueError("n_steps must be an integer >= 1")
    if not isinstance(T, (int, float)) or T <= 0:
        raise ValueError("T must be > 0")
    if not isinstance(strike, (int, float)) or strike <= 0:
        raise ValueError("strike must be > 0")
    if not isinstance(cost, (int, float)) or cost < 0:
        raise ValueError("cost must be >= 0")
    if not isinstance(market, str) or market not in ("bs", "rbergomi"):
        raise ValueError("market must be 'bs' or 'rbergomi'")
    
    if market == "bs":
        if "sigma" not in manifest:
            raise ValueError("market 'bs' requires sigma")
        sigma = manifest["sigma"]
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be > 0")
    elif market == "rbergomi":
        for key in ("H", "eta", "rho", "xi0"):
            if key not in manifest:
                raise ValueError(f"market 'rbergomi' requires {key}")
        H = manifest["H"]
        eta = manifest["eta"]
        rho = manifest["rho"]
        xi0 = manifest["xi0"]
        if not isinstance(H, (int, float)) or not (0 < H < 0.5):
            raise ValueError("H must be in (0, 0.5)")
        if not isinstance(eta, (int, float)) or eta < 0:
            raise ValueError("eta must be >= 0")
        if not isinstance(rho, (int, float)) or not (-1 <= rho <= 1):
            raise ValueError("rho must be in [-1, 1]")
        if not isinstance(xi0, (int, float)) or xi0 <= 0:
            raise ValueError("xi0 must be > 0")
    
    # Create separate streams via SeedSequence.spawn
    children = np.random.SeedSequence(seed).spawn(3)
    rngs = [np.random.default_rng(child) for child in children]
    
    # Generate paths for each role (train, val, test)
    splits = []
    for rng, n in zip(rngs, (n_train, n_val, n_test)):
        if market == "bs":
            dt = T / N
            z = rng.standard_normal((n, N))
            inc = -0.5 * sigma**2 * dt + sigma * math.sqrt(dt) * z
            S = np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1))
        else:  # rbergomi
            S = simulate_rbergomi(n, N, T, H, eta, rho, xi0, kappa=1, rng=rng)["S"]
        splits.append(S.astype(np.float64))
    
    sha256 = manifest_sha256(manifest)
    return Splits(train=splits[0], val=splits[1], test=splits[2], manifest=manifest, sha256=sha256)


def hedging_loss_torch(S: torch.Tensor, positions: torch.Tensor, strike: float, cost: float) -> torch.Tensor:
    """Compute hedging losses (negative PnL) from positions and price paths.
    
    Implements trade-by-trade cash ledger with unwind at final time.
    Differentiable in positions tensor.
    
    Args:
        S: (B, N+1) price paths, float32 or float64
        positions: (B, N) hedge positions, float32 or float64
        strike: option strike
        cost: proportional trading cost rate
    
    Returns:
        (B,) tensor of losses (negative of pnl)
    """
    B, N_plus_1 = S.shape
    N = N_plus_1 - 1
    device = S.device
    dtype = S.dtype
    
    # Vectorized cash ledger computation
    # trades[b, i] = positions[b, i] - positions[b, i-1] (with positions[b, -1] = 0)
    pos_prev = torch.cat([torch.zeros((B, 1), dtype=dtype, device=device), positions[:, :-1]], dim=1)
    trades = positions - pos_prev  # (B, N)
    
    # cost for each trade: cost * |trade| * S[i]
    trade_costs = cost * torch.abs(trades) * S[:, :N]  # (B, N)
    
    # cash change at each step: -trade * S - trade_cost
    cash_change = -trades * S[:, :N] - trade_costs  # (B, N)
    
    # total cash before final unwind
    cash_before_final = torch.sum(cash_change, dim=1)  # (B,)
    
    # final unwind: sell remaining positions at S[N]
    final_pos = positions[:, -1]  # (B,)
    final_unwind = final_pos * S[:, N] - cost * torch.abs(final_pos) * S[:, N]  # (B,)
    
    # total cash after final unwind
    cash_total = cash_before_final + final_unwind  # (B,)
    
    # payoff
    payoff = torch.clamp(S[:, N] - strike, min=0.0)  # (B,)
    
    # Loss = negative of the ledger PnL
    loss = -(cash_total - payoff)  # (B,)
    
    return loss


def cvar_ru_loss(losses: torch.Tensor, v: float | torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute CVaR (Conditional Value-at-Risk) via Rockafellar-Uryasev formula.
    
    CVaR = v + E[max(loss - v, 0)] / (1 - alpha)
    
    This equals the tail mean of the loss distribution at the alpha quantile.
    
    Args:
        losses: (B,) or (n,) tensor of losses
        v: threshold parameter (learnable scalar), float or 0-d tensor
        alpha: CVaR level in (0, 1)
    
    Returns:
        0-d tensor, scalar loss value
    """
    if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    
    # Ensure v is a tensor
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=losses.dtype)
    else:
        v = v.to(dtype=losses.dtype)
    
    # CVaR formula: v + E[ReLU(L - v)] / (1 - alpha)
    cvar = v + torch.relu(losses - v).mean() / (1 - alpha)
    return cvar


def evaluate_losses(
    model: nn.Module, 
    S: np.ndarray, 
    manifest: dict, 
    batch_size: int = 4096
) -> np.ndarray:
    """Evaluate hedging losses for a model on price paths S.
    
    Runs model in eval mode under no_grad to get positions, then computes
    per-path losses using the strike and cost from the manifest.
    
    Args:
        model: torch.nn.Module with forward(logm, times) -> positions
        S: (n, N+1) price paths, float64
        manifest: dict with "strike", "cost", "T", "n_steps"
        batch_size: batch size for processing (default 4096)
    
    Returns:
        (n,) float64 ndarray of losses
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    strike = manifest["strike"]
    cost = manifest["cost"]
    N = manifest["n_steps"]
    T = manifest["T"]
    n = S.shape[0]
    
    losses = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            S_batch = S[i:j]
            
            # Prepare inputs
            logm = np.log(S_batch / S_batch[:, :1])
            times = np.arange(N + 1, dtype=np.float64) * T / N
            
            # Convert to torch and move to device
            logm_t = torch.tensor(logm, dtype=dtype, device=device)
            times_t = torch.tensor(times, dtype=dtype, device=device)
            
            # Get positions
            positions = model(logm_t, times_t).detach()  # (j-i, N)
            
            # Compute losses
            S_batch_t = torch.tensor(S_batch, dtype=dtype, device=device)
            loss_batch = hedging_loss_torch(S_batch_t, positions, strike, cost)
            losses.append(loss_batch.cpu().numpy().astype(np.float64))
    
    return np.concatenate(losses)


def train_hedger(
    model: nn.Module,
    splits: Splits,
    config: dict,
    seed: int,
    device: str = "cpu",
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, dict]:
    """Train a hedger model using CVaR loss with early stopping.
    
    Jointly optimizes the model weights and a learnable threshold v on
    cvar_ru_loss over minibatches of the training set.
    
    Args:
        model: torch.nn.Module, untouched after training (returns a deep copy)
        splits: Splits object with train/val/test ndarrays and manifest
        config: dict with required keys (epochs, batch_size, lr) and optional
            (alpha, optimizer, lr_schedule, patience, min_delta)
        seed: random seed for epoch shuffling
        device: "cpu" or "cuda*"
    
    Returns:
        (trained_model, history) where trained_model has the best epoch's weights
        and history is a dict with train_cvar, val_cvar, lr, epochs_run, steps_taken,
        best_epoch, stopped_early, v_final, budget
    """
    # Validate config
    for key in ("epochs", "batch_size", "lr"):
        if key not in config:
            raise ValueError(f"config missing required key: {key}")
    
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    lr = config.get("lr")
    alpha = config.get("alpha", 0.95)
    optimizer = config.get("optimizer", "adam")
    lr_schedule = config.get("lr_schedule", "constant")
    patience = config.get("patience", None)
    min_delta = config.get("min_delta", 0.0)
    
    # Validate config values
    if not isinstance(epochs, numbers.Integral) or epochs < 1:
        raise ValueError("epochs must be integer >= 1")
    if not isinstance(batch_size, numbers.Integral) or batch_size < 1:
        raise ValueError("batch_size must be integer >= 1")
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError("lr must be > 0")
    if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if optimizer not in ("adam", "sgd"):
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError("lr_schedule must be 'constant' or 'cosine'")
    if patience is not None and (not isinstance(patience, numbers.Integral) or patience < 1):
        raise ValueError("patience must be None or integer >= 1")
    if not isinstance(min_delta, (int, float)) or min_delta < 0:
        raise ValueError("min_delta must be >= 0")
    for key in config:
        if key not in ("epochs", "batch_size", "lr", "alpha", "optimizer", "lr_schedule", "patience", "min_delta"):
            raise ValueError(f"Unknown config key: {key}")
    
    # Validate device
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda*', got {device}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")
    
    # Copy model to device
    model = copy.deepcopy(model).to(device)
    dtype = next(model.parameters()).dtype
    
    # Get training data
    S_train = splits.train  # (n_train, N+1)
    S_val = splits.val
    manifest = splits.manifest
    strike = manifest["strike"]
    cost = manifest["cost"]
    N = manifest["n_steps"]
    T = manifest["T"]
    n_train = S_train.shape[0]
    
    # Compute steps per epoch
    steps_per_epoch = math.ceil(n_train / batch_size)
    steps_planned = epochs * steps_per_epoch
    
    # Initialize learnable threshold
    v = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    
    # Setup optimizer
    if optimizer == "adam":
        opt = torch.optim.Adam([{"params": model.parameters()}, {"params": [v]}], lr=lr)
    else:  # sgd
        opt = torch.optim.SGD([{"params": model.parameters()}, {"params": [v]}], lr=lr)
    
    # Training loop
    history = {
        "train_cvar": [],
        "val_cvar": [],
        "lr": [],
        "epochs_run": 0,
        "steps_taken": 0,
        "best_epoch": 0,
        "stopped_early": False,
        "v_final": 0.0,
        "budget": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_schedule": lr_schedule,
            "alpha": alpha,
            "optimizer": optimizer,
            "steps_per_epoch": steps_per_epoch,
            "steps_planned": steps_planned,
            "n_train": n_train,
            "patience": patience,
            "min_delta": min_delta,
        }
    }
    
    best_val_cvar = float('inf')
    best_epoch_weights = None
    epochs_no_improve = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    
    for epoch in range(epochs):
        # Shuffle training indices
        indices = torch.randperm(n_train, generator=rng).numpy()
        S_shuffled = S_train[indices]
        
        # Record learning rate at start of epoch (for the first step)
        if lr_schedule == "constant":
            current_lr = lr
        else:  # cosine
            t = epoch * steps_per_epoch
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * t / steps_planned))
        history["lr"].append(current_lr)
        
        # Training minibatches
        epoch_losses = []
        for step in range(steps_per_epoch):
            # Get batch indices
            start = step * batch_size
            end = min(start + batch_size, n_train)
            S_batch = S_shuffled[start:end]
            
            # Forward pass
            logm = np.log(S_batch / S_batch[:, :1])
            times = np.arange(N + 1, dtype=np.float64) * T / N
            logm_t = torch.tensor(logm, dtype=dtype, device=device)
            times_t = torch.tensor(times, dtype=dtype, device=device)
            
            positions = model(logm_t, times_t)  # (b, N)
            S_batch_t = torch.tensor(S_batch, dtype=dtype, device=device)
            losses = hedging_loss_torch(S_batch_t, positions, strike, cost)
            
            # Compute CVaR loss
            cvar = cvar_ru_loss(losses, v, alpha)
            
            # Backward pass
            opt.zero_grad()
            cvar.backward()
            opt.step()
            
            epoch_losses.append(losses.detach().cpu().numpy())
            
            # Learning rate schedule update (per-step for cosine)
            if lr_schedule == "cosine":
                t = epoch * steps_per_epoch + step + 1
                if t < steps_planned:
                    lr_t = lr * 0.5 * (1 + math.cos(math.pi * t / steps_planned))
                    for param_group in opt.param_groups:
                        param_group["lr"] = lr_t
            
            history["steps_taken"] += 1
        
        # Evaluate on training set (all batches)
        L_train_np = np.concatenate(epoch_losses)
        train_cvar = float(cvar_tail_mean(L_train_np, alpha))
        history["train_cvar"].append(train_cvar)
        
        # Evaluate on validation set (recompute with current model)
        L_val_np = evaluate_losses(model, S_val, manifest, batch_size=4096)
        val_cvar = float(cvar_tail_mean(L_val_np, alpha))
        history["val_cvar"].append(val_cvar)
        
        history["epochs_run"] += 1
        if progress_callback is not None:
            progress_callback({
                "epoch": epoch + 1,
                "epochs_planned": epochs,
                "train_cvar95": train_cvar,
                "val_cvar95": val_cvar,
                "lr": current_lr,
                "steps_taken": history["steps_taken"],
            })
        
        # Early stopping logic
        if epoch == 0:
            best_val_cvar = val_cvar
            best_epoch_weights = copy.deepcopy(model.state_dict())
            history["best_epoch"] = 0
            best_v = float(v.detach().cpu())
        else:
            if val_cvar < best_val_cvar - min_delta:
                best_val_cvar = val_cvar
                best_epoch_weights = copy.deepcopy(model.state_dict())
                history["best_epoch"] = epoch
                best_v = float(v.detach().cpu())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if patience is not None and epochs_no_improve >= patience:
                    history["stopped_early"] = True
                    break
    
    # Restore best weights
    model.load_state_dict(best_epoch_weights)
    history["v_final"] = best_v
    
    # Move back to original device (the model is returned in the training device)
    return model, history


def peak_memory_bytes(fn: Callable, device: str = "cpu") -> tuple[Any, int]:
    """Measure peak memory usage of a function call.
    
    On CPU, measures peak resident set size growth in bytes.
    On CUDA, measures peak GPU memory allocation in bytes.
    
    Args:
        fn: callable that returns a result
        device: "cpu" or "cuda*"
    
    Returns:
        (result, peak_bytes) where peak_bytes is the peak memory usage in bytes
    """
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda*', got {device}")
    
    if device == "cpu":
        # Measure CPU memory growth using RSS
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss
        peak_rss = mem_start
        
        # Polling approach: monitor RSS during execution
        stop_polling = [False]  # Use list for mutable reference in nested function
        
        def poll_memory():
            nonlocal peak_rss
            while not stop_polling[0]:
                current_rss = process.memory_info().rss
                peak_rss = max(peak_rss, current_rss)
                time.sleep(0.001)  # Poll every ms
        
        # Start polling thread
        poll_thread = threading.Thread(target=poll_memory, daemon=True)
        poll_thread.start()
        
        try:
            result = fn()
        finally:
            stop_polling[0] = True
            poll_thread.join(timeout=1.0)
        
        # Final measurement to ensure we got the peak
        peak_rss = max(peak_rss, process.memory_info().rss)
        peak = max(0, peak_rss - mem_start)
        return result, peak
    else:
        # Measure GPU memory
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        peak = torch.cuda.max_memory_allocated()
        return result, int(peak)
