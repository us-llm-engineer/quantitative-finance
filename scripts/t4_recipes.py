#!/usr/bin/env python3
"""T4 recipe study: train 75 units (5 recipes × 3 arms × 5 seeds) with rough_hedge.

CLI: python3 scripts/t4_recipes.py unit --recipe R --arm A --seed K --out DIR [--quick]
     python3 scripts/t4_recipes.py classical --out DIR [--quick]
     python3 scripts/t4_recipes.py collect --out DIR

Produces per-unit JSON and aggregates into recipes.json with across-seed stats.
"""
import sys
import os
import argparse
import json
import hashlib
import time
import base64
import math
import subprocess
import copy
import numbers
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
import torch.optim
from datetime import timezone

proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))

def get_iso_now():
    """Get current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

from rough_hedge import training, models, baselines, risk, stats, hedging
from rough_hedge.rbergomi import simulate_rbergomi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_monitor import EpochMonitor

# Device detection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}", flush=True)


def get_gpu_name() -> str:
    """Get GPU model name if available."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def make_config_hash(config: dict) -> str:
    """Compute SHA256 of canonical config JSON."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_array_hash(arr: np.ndarray) -> str:
    """Compute SHA256 of array bytes."""
    return hashlib.sha256(arr.astype(np.float64).tobytes()).hexdigest()


def get_recipe_config(recipe: str, quick: bool = False) -> Dict[str, Any]:
    """Get training config for a recipe.

    Recipes from addendum-A2-recipes-L4.md grounded on paper full text.
    """
    if quick:
        return {
            "optimizer": "adam",
            "lr": 1e-2,
            "batch_size": 64,
            "epochs": 2,
            "lr_schedule": "constant",
            "alpha": 0.95,
            "patience": None,
            "min_delta": 0.0,
            "weight_decay": 0.0,
            "fresh_batch_mode": False,
        }

    if recipe == "R-DH":
        return {
            "optimizer": "adam",
            "lr": 5e-3,
            "batch_size": 256,
            "epochs": 100,
            "lr_schedule": "constant",
            "alpha": 0.95,
            "patience": 10,
            "min_delta": 0.0,
            "weight_decay": 0.0,
            "fresh_batch_mode": False,
        }
    elif recipe == "R-SIG":
        return {
            "optimizer": "adamw",
            "lr": 1e-2,
            "batch_size": 64,
            "epochs": 64,
            "lr_schedule": "cosine",
            "cosine_floor": 1e-3,
            "alpha": 0.95,
            "patience": None,
            "min_delta": 0.0,
            "weight_decay": 0.01,
            "fresh_batch_mode": False,
        }
    elif recipe == "R-NTBN":
        return {
            "optimizer": "adam",
            "lr": 1e-2,
            "batch_size": int(65536 * PATH_SCALE),
            "epochs": 120,
            "lr_schedule": "constant",
            "alpha": 0.95,
            "patience": None,
            "min_delta": 0.0,
            "weight_decay": 0.0,
            "fresh_batch_mode": True,
        }
    elif recipe == "R-HOR":
        return {
            "optimizer": "adam",
            "lr": 5e-3,
            "batch_size": 256,
            "epochs": 75,
            "lr_schedule": "constant",
            "alpha": 0.95,
            "patience": None,
            "min_delta": 0.0,
            "weight_decay": 0.0,
            "fresh_batch_mode": False,
        }
    elif recipe == "R-GPU":
        return {
            "optimizer": "adam",
            "lr": 5e-3,
            "batch_size": 1024,
            "epochs": 50,
            "lr_schedule": "constant",
            "alpha": 0.95,
            "patience": 10,
            "min_delta": 0.0,
            "weight_decay": 0.0,
            "fresh_batch_mode": False,
        }
    else:
        raise ValueError(f"Unknown recipe: {recipe}")



RECIPE_ORDER = ["R-GPU", "R-NTBN", "R-SIG", "R-DH", "R-HOR"]   # cheap first (addendum A2)
ARMS = ["FFN", "GRU", "sig2"]
SEEDS = [0, 1, 2]
PATH_SCALE = 0.1   # small-scale study: training paths (and R-NTBN full batch) x0.1 of the paper sizes


def get_train_test_sizes(recipe: str, quick: bool = False) -> tuple[int, int]:
    """n_train, n_test per recipe (addendum A2): R-SIG 10,000 [2508.02759]; R-HOR 1e5 [2102.01962]."""
    if quick:
        return 1024, 2000
    return int({"R-SIG": 10000, "R-HOR": 100000}.get(recipe, 65536) * PATH_SCALE), 20000


def planned_steps(recipe: str) -> tuple[int, int]:
    """(optimizer steps, epochs) of one full unit, before any early stop."""
    cfg = get_recipe_config(recipe)
    n_train, _ = get_train_test_sizes(recipe)
    per_epoch = 1 if cfg.get("fresh_batch_mode") else math.ceil(n_train / cfg["batch_size"])
    return per_epoch * cfg["epochs"], cfg["epochs"]


def _gpu_logger(path: Path):
    """Background nvidia-smi timeline (addendum A3 E); returns the Popen or None."""
    try:
        return subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
             "--format=csv", "-l", "2"], stdout=open(path, "w"), stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None


def recipe_workers(out_dir: Path, recipe: str, workers: int) -> int:
    """Concurrency capped by GPU memory: 85% of device memory / measured peak per unit (timing.json)."""
    tj = Path(out_dir) / "timing.json"
    if not tj.exists() or not torch.cuda.is_available():
        return workers
    units = json.loads(tj.read_text())["units"]
    peak = max(units[f"{recipe}/{a}"]["on"]["peak_memory_bytes"] for a in ARMS if f"{recipe}/{a}" in units)
    total = torch.cuda.get_device_properties(0).total_memory
    return max(1, min(workers, int(0.85 * total // max(peak, 1))))


def batch_run(out_dir: Path, workers: int = 8, quick: bool = False, threads: int = 1) -> None:
    """Recipes run sequentially; the 15 units of one recipe run as concurrent processes.

    Started detached by the notebook launch cell; per-unit stdout goes to <unit>.log.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"batch start {get_iso_now()} workers={workers} quick={quick}", flush=True)
    for recipe in RECIPE_ORDER:
        jobs = [(recipe, arm, seed, out_dir, quick, out_dir / f"{recipe}_{arm}_{seed}.log")
                for arm in ARMS for seed in SEEDS
                if not (out_dir / f"{recipe}_{arm}_{seed}.json").exists()]   # resumable
        w = recipe_workers(out_dir, recipe, workers)
        thr = threads
        if recipe == "R-NTBN":
            # full-batch 65,536 paths: ~7 GB per GRU unit measured locally (OOM on 4 GiB) -> 2 lanes on 22 GiB
            w, thr = min(w, 2), 1
        logger = _gpu_logger(out_dir / f"gpu_{recipe}.csv")
        t0 = time.time()
        print(f"[{recipe}] {len(jobs)} units, {w} concurrent, start {get_iso_now()}", flush=True)
        if thr <= 1:
            with Pool(processes=w) as pool:
                for job_name, rc in pool.imap_unordered(run_unit_job, jobs):
                    print(f"  {job_name}: rc={rc} t+{time.time() - t0:.0f}s", flush=True)
        else:
            # w processes, each running its share of units on `threads` threads
            shares = [jobs[i::w] for i in range(w) if jobs[i::w]]
            procs = [subprocess.Popen([sys.executable, str(Path(__file__)), "units", "--out", str(out_dir),
                                       "--threads", str(thr),
                                       "--jobs", ",".join(f"{r}:{a}:{s}" for r, a, s, *_ in share)]
                                      + (["--quick"] if quick else []),
                                      stdout=open(out_dir / f"proc_{recipe}_{i}.log", "w"), stderr=subprocess.STDOUT)
                     for i, share in enumerate(shares)]
            for p in procs:
                p.wait()
            print(f"  {len(shares)} processes x {thr} threads rc={[p.returncode for p in procs]} "
                  f"t+{time.time() - t0:.0f}s", flush=True)
        if logger is not None:
            logger.terminate()
        print(f"[{recipe}] done in {time.time() - t0:.0f}s", flush=True)
    if not (out_dir / "classical.json").exists():
        print("classical:", run_classical_job((out_dir, quick, out_dir / "classical.log")), flush=True)
    print(f"Batch run complete {get_iso_now()}", flush=True)


class _ThreadStdout:
    """sys.stdout proxy: each worker thread writes to its own log file (per-unit logs under threads)."""
    def __init__(self, default):
        import threading
        self.default, self.local = default, threading.local()

    def write(self, text):
        return getattr(self.local, "fh", self.default).write(text)

    def flush(self):
        getattr(self.local, "fh", self.default).flush()


def run_units_threaded(jobs: list, out_dir: Path, quick: bool, threads: int) -> None:
    """Run several units in ONE process on K threads, each on its own CUDA stream.

    Model init, data and shuffling use private generators (rough_hedge/models.py, training.py),
    so threads do not race on a global seed. Peak memory under threads is per process.
    """
    from concurrent.futures import ThreadPoolExecutor
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    proxy = _ThreadStdout(sys.stdout)
    sys.stdout = proxy

    def one(job):
        recipe, arm, seed = job
        with open(Path(out_dir) / f"{recipe}_{arm}_{seed}.log", "w") as fh:
            proxy.local.fh = fh
            try:
                with torch.cuda.stream(torch.cuda.Stream()):
                    train_unit(recipe, arm, seed, Path(out_dir), quick)
                return f"{recipe}_{arm}_{seed}", 0
            except Exception as exc:
                import traceback
                traceback.print_exc(file=fh)
                return f"{recipe}_{arm}_{seed}", 1
            finally:
                del proxy.local.fh

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for name, rc in ex.map(one, jobs):
            proxy.default.write(f"  {name}: rc={rc}\n"); proxy.default.flush()


def time_one(recipe: str, arm: str, steps: int, monitor: bool) -> dict:
    """Run train_recipe for ~`steps` real-batch optimizer steps; return honest timings."""
    cfg = dict(get_recipe_config(recipe))
    bs = cfg["batch_size"]
    if cfg.get("fresh_batch_mode"):
        steps = min(steps, 10)          # each fresh epoch simulates a full batch on the CPU
        cfg["epochs"], n_train = steps, bs
    else:
        n_train, cfg["epochs"] = bs * steps, 1
    cfg["patience"] = None
    manifest = {"seed": 0, "n_train": n_train, "n_val": 4096, "n_test": 512, "n_steps": 50,
                "T": 30 / 365, "strike": 1.0, "cost": 0.001, "market": "rbergomi",
                "H": 0.1, "eta": 1.0, "rho": -0.5, "xi0": 0.1}
    splits = training.make_splits(manifest)
    t0 = time.time()
    _, h = train_recipe(create_arm_model(arm, 0), splits, cfg, 0, device=DEVICE, enable_monitor=monitor)
    wall = time.time() - t0
    med = h["ms_per_step_median"]
    step_ms = float(np.median(med[1:] if len(med) > 1 else med)) if cfg.get("fresh_batch_mode") else float(med[0])
    epoch_overhead_s = max(0.0, (sum(h["wall_s"]) - step_ms * h["steps_taken"] / 1000) / max(h["epochs_run"], 1))
    return {"ms_per_step": step_ms, "epoch_overhead_s": epoch_overhead_s, "wall_s": wall,
            "steps": h["steps_taken"], "peak_memory_bytes": int(max(h["peak_memory"]))}


def time_recipes(out_dir: Path, steps: int = 200, quick: bool = False) -> None:
    """Measure ms/step (monitor off and on) for every recipe x arm, then project the study.

    Projection per unit = planned_steps * ms_step + epochs * epoch_overhead; a recipe's 15 units
    are divided over C concurrent processes, scaled by the measured concurrency efficiency.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"device": get_gpu_name(), "steps_timed": steps, "units": {}}
    for recipe in RECIPE_ORDER:
        for arm in ARMS:
            off = time_one(recipe, arm, steps, monitor=False)
            on = time_one(recipe, arm, steps, monitor=True)
            ps, ep = planned_steps(recipe)
            unit_s = ps * on["ms_per_step"] / 1000 + ep * on["epoch_overhead_s"]
            res["units"][f"{recipe}/{arm}"] = {"off": off, "on": on, "planned_steps": ps, "epochs": ep,
                                               "monitor_overhead": on["ms_per_step"] / off["ms_per_step"] - 1,
                                               "projected_unit_s": unit_s}
            print(f"{recipe:7s} {arm:5s} ms/step off {off['ms_per_step']:7.2f} on {on['ms_per_step']:7.2f} "
                  f"(+{100 * (on['ms_per_step'] / off['ms_per_step'] - 1):4.1f}%) epoch-overhead {on['epoch_overhead_s']:.2f}s "
                  f"peak {on['peak_memory_bytes'] / 2**20:6.0f} MiB -> unit {unit_s / 60:6.1f} min", flush=True)
    print("\nProjected study wall time (5 seeds per arm; C concurrent processes; ideal scaling, "
          "concurrency efficiency is measured by the `scale` command):", flush=True)
    res["projection_hours"] = {}
    for C in (1, 2, 4, 8, 15):
        per_recipe = {r: sum(5 * res["units"][f"{r}/{a}"]["projected_unit_s"] for a in ARMS) / C / 3600
                      for r in RECIPE_ORDER}
        res["projection_hours"][C] = per_recipe
        print(f"  C={C:2d}: " + "  ".join(f"{r} {h:5.2f}h" for r, h in per_recipe.items())
              + f"  total {sum(per_recipe.values()):6.2f}h", flush=True)
    (out_dir / "timing.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"Wrote {out_dir / 'timing.json'}", flush=True)


def scale_probe(out_dir: Path, recipe: str, arm: str, steps: int, levels=(1, 2, 4, 8)) -> None:
    """Concurrency efficiency: C identical `time_one` processes; aggregate steps/s vs C=1."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for C in levels:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, str(Path(__file__)), "time-one", "--recipe", recipe,
                                   "--arm", arm, "--steps", str(steps), "--out", str(out_dir / f"scale_{C}_{i}.json")],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for i in range(C)]
        rcs = [p.wait() for p in procs]
        wall = time.time() - t0
        rows[C] = {"wall_s": wall, "agg_steps_per_s": C * steps / wall, "rcs": rcs}
        print(f"C={C:2d} wall {wall:6.1f}s aggregate {C * steps / wall:7.1f} steps/s "
              f"efficiency {rows[C]['agg_steps_per_s'] / rows[levels[0]]['agg_steps_per_s'] / C:.2f}", flush=True)
    (out_dir / f"scale_{recipe}_{arm}.json").write_text(json.dumps(rows, indent=2))


def create_arm_model(arm: str, seed: int) -> nn.Module:
    """Create a fresh hedger model for an arm."""
    if arm == "FFN":
        return models.FFNHedger(hidden=64, layers=2, seed=seed, dtype=torch.float32)
    elif arm == "GRU":
        return models.GRUHedger(hidden=64, seed=seed, dtype=torch.float32)
    elif arm == "sig2":
        return models.SignatureHedger(depth=2, hidden=64, seed=seed, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown arm: {arm}")


def generate_test_paths(seed_offset: int, n_test: int = 20000) -> np.ndarray:
    """Generate deterministic test paths (rBergomi with H=0.1).

    Returns (n_test, N+1) array.
    """
    H = 0.1
    N = 50
    T = 30 / 365
    eta = 1.0
    rho = -0.5
    xi0 = 0.1

    rng = np.random.default_rng(seed_offset)
    result = simulate_rbergomi(n_test, N, T, H, eta, rho, xi0, kappa=1, rng=rng)
    return result["S"].astype(np.float64)


def shared_test_paths(n_test: int = 20000) -> np.ndarray:
    """Generate and normalize test paths (rBergomi with H=0.1).

    Returns (n_test, N+1) array normalized so S/S0 starts at 1.0.
    """
    S_test = generate_test_paths(seed_offset=42, n_test=n_test)
    # Normalize to start at 1.0
    S_test = S_test / S_test[:, :1]
    return S_test


def shared_probe_paths(n_probe: int = 4096) -> np.ndarray:
    """Fixed monitor probe set: same 4096 paths for every unit (seed offset 7), S/S0 = 1."""
    S = generate_test_paths(seed_offset=7, n_test=n_probe)
    return S / S[:, :1]


def train_recipe(
    model: nn.Module, 
    splits: training.Splits,
    config: Dict[str, Any],
    seed: int,
    device: str = "cpu",
    enable_monitor: bool = True
) -> tuple[nn.Module, dict]:
    """Train a hedger model using recipe config (A3: device-resident, honest timing).
    
    Mirrors training.train_hedger but adds:
    - AdamW optimizer with weight_decay
    - Cosine schedule with floor (min lr)
    - Fresh-batch mode for R-NTBN (one new batch per epoch)
    - A3 Priority A: device-resident data (no .cpu() in step loop, torch.cuda.Event timing)
    """
    # Config
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    lr = config.get("lr")
    alpha = config.get("alpha", 0.95)
    optimizer = config.get("optimizer", "adam")
    lr_schedule = config.get("lr_schedule", "constant")
    patience = config.get("patience", None)
    min_delta = config.get("min_delta", 0.0)
    weight_decay = config.get("weight_decay", 0.0)
    cosine_floor = config.get("cosine_floor", 0.0)
    fresh_batch_mode = config.get("fresh_batch_mode", False)
    
    model = copy.deepcopy(model).to(device)
    dtype = next(model.parameters()).dtype
    
    S_train = splits.train
    S_val = splits.val
    manifest = splits.manifest
    strike = manifest["strike"]
    cost = manifest["cost"]
    N = manifest["n_steps"]
    T = manifest["T"]
    n_train = S_train.shape[0]
    
    if fresh_batch_mode:
        steps_per_epoch = 1
    else:
        steps_per_epoch = math.ceil(n_train / batch_size)
    steps_planned = epochs * steps_per_epoch
    
    v = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    
    if optimizer == "adam":
        opt = torch.optim.Adam([{"params": model.parameters()}, {"params": [v]}], lr=lr)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW([{"params": model.parameters()}, {"params": [v]}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD([{"params": model.parameters()}, {"params": [v]}], lr=lr)
    else:
        raise ValueError(f"optimizer must be 'adam', 'adamw', or 'sgd', got {optimizer}")
    
    # A3 Priority A: move data to device ONCE
    S_train_np = np.asarray(S_train, dtype=np.float64)
    S_val_np = np.asarray(S_val, dtype=np.float64)
    S_train_t = torch.tensor(S_train_np, dtype=dtype, device=device)
    S_val_t = torch.tensor(S_val_np, dtype=dtype, device=device)
    times_fixed = torch.tensor(np.arange(N + 1, dtype=np.float64) * T / N, dtype=dtype, device=device)
    
    history = {
        "train_cvar": [],
        "val_cvar": [],
        "lr": [],
        "wall_s": [],
        "ms_per_step": [],
        "peak_memory": [],
        "epochs_run": 0,
        "steps_taken": 0,
        "best_epoch": 0,
        "stopped_early": False,
        "v_final": 0.0,
        "monitor": [],  # A3 monitoring
    }
    
    best_val_cvar = float('inf')
    best_epoch_weights = None
    epochs_no_improve = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    probe_S_t = torch.tensor(shared_probe_paths(), dtype=dtype, device=device)
    monitor = EpochMonitor(model, v, probe_S_t, times_fixed, strike, cost,
                           math.sqrt(manifest.get("xi0", 0.1)), T, alpha,
                           batch_size, epochs, enabled=enable_monitor)

    tag = config.get("tag", "")
    t_train0 = time.time()
    print(f"[{tag}] start: {epochs} epochs x {steps_per_epoch} steps = {steps_planned} steps, "
          f"batch {batch_size}, n_train {n_train}, opt {optimizer}, lr {lr}, schedule {lr_schedule}", flush=True)
    for epoch in range(epochs):
        monitor.start_epoch()
        # A3 Priority A: torch.cuda.Event timing
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device=device)
            epoch_start_event = torch.cuda.Event(enable_timing=True)
            epoch_end_event = torch.cuda.Event(enable_timing=True)
            epoch_start_event.record()

        # Generate or shuffle training data
        if fresh_batch_mode:
            fresh_seed = seed * 1000 + epoch
            rng_fresh = np.random.default_rng(fresh_seed)
            result = simulate_rbergomi(batch_size, N, T, 0.1, 1.0, -0.5, 0.1, kappa=1, rng=rng_fresh)
            S_train_batch_t = torch.tensor(result["S"].astype(np.float64), dtype=dtype, device=device)
            S_shuffled = S_train_batch_t
        else:
            indices = torch.randperm(n_train, generator=rng).to(device)
            S_shuffled = S_train_t[indices]
        
        if lr_schedule == "constant":
            current_lr = lr
        else:
            t = epoch * steps_per_epoch
            current_lr = cosine_floor + (lr - cosine_floor) * 0.5 * (1 + math.cos(math.pi * t / steps_planned))
        history["lr"].append(current_lr)
        
        # Training steps (A3: NO .cpu() in loop, H1: no sync in loop)
        epoch_losses_list = []
        step_events = []  # H1: collect events, sync once after loop
        
        for step in range(steps_per_epoch):
            step_events_pair = None
            if device.startswith("cuda"):
                step_start = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)
                step_start.record()
                step_events_pair = (step_start, step_end)
            
            # Get batch on device
            if fresh_batch_mode:
                S_batch_t = S_train_batch_t
            else:
                start = step * batch_size
                end = min(start + batch_size, n_train)
                S_batch_t = S_shuffled[start:end]
            
            # Forward (all on device, no .cpu())
            logm = torch.log(S_batch_t / S_batch_t[:, :1])
            positions = model(logm, times_fixed[:S_batch_t.shape[1]])
            losses = training.hedging_loss_torch(S_batch_t, positions, strike, cost)
            cvar = training.cvar_ru_loss(losses, v, alpha)
            
            # Backward
            opt.zero_grad()
            cvar.backward()
            
            monitor.after_backward()
            
            opt.step()
            
            # Keep losses on device
            epoch_losses_list.append(losses)
            
            # H2: Fixed cosine formula: lr(0)=lr, lr(T)=cosine_floor
            if lr_schedule == "cosine":
                t = epoch * steps_per_epoch + step + 1
                if t < steps_planned:
                    cos_val = math.cos(math.pi * t / steps_planned)
                    lr_t = cosine_floor + (lr - cosine_floor) * 0.5 * (1 + cos_val)
                    for param_group in opt.param_groups:
                        param_group["lr"] = lr_t
            
            history["steps_taken"] += 1

            if step_events_pair:
                # Record end event before collecting pair
                step_start, step_end = step_events_pair
                step_end.record()
                step_events.append((step_start, step_end))
        
        # H1: synchronize once after loop, then read timings
        step_times = []
        if device.startswith("cuda") and step_events:
            torch.cuda.current_stream().synchronize()
            for start_ev, end_ev in step_events:
                step_times.append(start_ev.elapsed_time(end_ev))
        
        # Reduce losses at epoch end (single .cpu() call, detach first)
        L_train_t = torch.cat(epoch_losses_list, dim=0)
        L_train_np = L_train_t.detach().cpu().numpy()
        train_cvar = float(risk.cvar_tail_mean(L_train_np, alpha))
        history["train_cvar"].append(train_cvar)
        
        # Validation (on device)
        logm_val = torch.log(S_val_t / S_val_t[:, :1])
        with torch.no_grad():
            positions_val = model(logm_val, times_fixed[:S_val_t.shape[1]])
            losses_val = training.hedging_loss_torch(S_val_t, positions_val, strike, cost)
        L_val_np = losses_val.detach().cpu().numpy()
        val_cvar = float(risk.cvar_tail_mean(L_val_np, alpha))
        history["val_cvar"].append(val_cvar)
        
        history["epochs_run"] += 1
        peak_memory = torch.cuda.max_memory_allocated(device=device) if device.startswith("cuda") else 0
        history["peak_memory"].append(int(peak_memory))
        
        # Timing
        if device.startswith("cuda"):
            epoch_end_event.record()
            torch.cuda.current_stream().synchronize()
            epoch_ms = epoch_start_event.elapsed_time(epoch_end_event)
            epoch_elapsed = epoch_ms / 1000
            mean_ms = np.mean(step_times) if step_times else 0
        else:
            epoch_elapsed = 0
            mean_ms = 0
        
        history["wall_s"].append(epoch_elapsed)
        history["ms_per_step"].append(mean_ms)
        history.setdefault("ms_per_step_median", []).append(float(np.median(step_times)) if step_times else 0.0)
        
        done_frac = (epoch + 1) / epochs
        eta_s = (time.time() - t_train0) * (1 - done_frac) / done_frac
        print(f"[{tag}] Epoch {epoch+1}/{epochs} step {history['steps_taken']}/{steps_planned} "
              f"train_cvar={train_cvar:.6f} val_cvar={val_cvar:.6f} lr={current_lr:.6f} "
              f"wall_s={epoch_elapsed:.1f} ms/step={mean_ms:.2f} elapsed={time.time() - t_train0:.0f}s eta={eta_s:.0f}s", flush=True)
        rec = monitor.end_epoch(epoch, current_lr, epoch_elapsed, float(mean_ms), peak_memory, train_cvar, val_cvar)
        if rec is not None:
            print(f"  monitor: probe_cvar95={rec['probe_cvar95']:.5f} v={rec['ru_threshold_v']:.5f} "
                  f"var95={rec['probe_var95']:.5f} |d-dBS|={rec['delta_dist_mean']:.4f} "
                  f"gnorm={rec['grad_norm_mean']:.3e} B_simple={rec['grad_noise_scale_B_simple']} "
                  f"upd={rec['update_to_weight_ratio']:.2e}", flush=True)
        
        # Early stopping
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
    
    history["monitor"] = monitor.finish(history["epochs_run"] - 1) if enable_monitor else None
    model.load_state_dict(best_epoch_weights)
    history["v_final"] = best_v
    return model, history



def train_unit(recipe: str, arm: str, seed: int, out_dir: Path, quick: bool = False) -> None:
    """Train one unit: (recipe, arm, seed) and save results to JSON."""
    start_time = time.time()
    start_iso = get_iso_now()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common parameters
    H = 0.1
    N = 50
    T = 30 / 365
    strike = 1.0
    cost = 0.001  # 10 basis points

    n_train, n_test = get_train_test_sizes(recipe, quick)

    # Manifest for training data
    manifest = {
        "seed": seed,
        "n_train": n_train,
        "n_val": int(n_train * 0.2),
        "n_test": n_test,
        "n_steps": N,
        "T": T,
        "strike": strike,
        "cost": cost,
        "market": "rbergomi",
        "H": H,
        "eta": 1.0,
        "rho": -0.5,
        "xi0": 0.1,
    }

    config = {
        "recipe": recipe,
        "arm": arm,
        "seed": seed,
        "manifest": manifest,
        **get_recipe_config(recipe, quick)
    }
    config_hash = make_config_hash(config)

    # Generate test paths (shared with classical, normalized)
    S_test = shared_test_paths(n_test)
    test_paths_hash = make_array_hash(S_test)

    # Create model and move to device
    model = create_arm_model(arm, seed).to(DEVICE)

    # Generate train/val splits
    splits = training.make_splits(manifest)

    # Train hedger using recipe
    torch.cuda.reset_peak_memory_stats(device=DEVICE)
    train_config = {**get_recipe_config(recipe, quick), "tag": f"{recipe}/{arm}/s{seed}"}
    trained_model, history = train_recipe(
        model, splits, train_config, seed, device=DEVICE
    )
    peak_memory = torch.cuda.max_memory_allocated(device=DEVICE) if DEVICE.startswith("cuda") else 0

    # Evaluate on test paths
    test_losses = training.evaluate_losses(trained_model, S_test, manifest, batch_size=4096)

    # Encode losses as float32 base64
    losses_f32 = test_losses.astype(np.float32)
    losses_b64 = base64.b64encode(losses_f32.tobytes()).decode("ascii")

    # Compute statistics
    alpha_levels = [0.90, 0.95, 0.99]
    stats_dict = {}
    for alpha in alpha_levels:
        stats_dict[f"cvar_{int(alpha*100)}"] = float(risk.cvar_tail_mean(test_losses, alpha))

    var_95 = np.quantile(test_losses, 0.95)
    stats_dict["var_95"] = float(var_95)
    stats_dict["mean"] = float(test_losses.mean())
    stats_dict["sd"] = float(test_losses.std())

    # Compute turnover from model policy
    def policy_fn(S):
        with torch.no_grad():
            logm = np.log(S / S[:, :1])
            times = np.arange(N + 1, dtype=np.float64) * T / N
            logm_t = torch.tensor(logm, dtype=torch.float32, device=DEVICE)
            times_t = torch.tensor(times, dtype=torch.float32, device=DEVICE)
            positions = trained_model(logm_t, times_t).cpu().numpy()
        return positions

    positions = policy_fn(S_test)
    stats_dict["turnover"] = float(hedging.turnover(positions).mean())

    # Post-training stats from history
    stats_dict["best_epoch"] = history.get("best_epoch", 0)
    stats_dict["stopped_early"] = history.get("stopped_early", False)

    # Learning curve area: trapz of training CVaR
    train_cvar = np.array(history.get("train_cvar", []))
    if len(train_cvar) > 1:
        stats_dict["learning_curve_area"] = float(trapezoid(train_cvar))
    else:
        stats_dict["learning_curve_area"] = 0.0

    # Generalization gap: validation - training (at best epoch)
    best_epoch = history.get("best_epoch", 0)
    val_cvar_list = history.get("val_cvar", [])
    train_cvar_list = history.get("train_cvar", [])
    if best_epoch < len(val_cvar_list) and best_epoch < len(train_cvar_list):
        gap = val_cvar_list[best_epoch] - train_cvar_list[best_epoch]
        stats_dict["val_minus_train_gap"] = float(gap)
    else:
        stats_dict["val_minus_train_gap"] = 0.0

    # Assemble output
    end_time = time.time()
    end_iso = get_iso_now()

    result = {
        "config": config,
        "config_sha256": config_hash,
        "test_paths_sha256": test_paths_hash,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gpu_name": get_gpu_name(),
        "started_at": start_iso,
        "finished_at": end_iso,
        "wall_s": end_time - start_time,
        "peak_gpu_memory_bytes": int(peak_memory),
        "histories": history,
        "per_path_losses": {
            "b64": losses_b64,
            "shape": [int(n_test)]
        },
        "stats": stats_dict
    }

    # Save to JSON
    out_file = out_dir / f"{recipe}_{arm}_{seed}.json"
    out_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved {out_file}", flush=True)


def _apply_classical_policy_stepwise(policy_step, S, N):
    """Apply a step-wise policy function to generate full position arrays."""
    S = np.asarray(S, dtype=np.float64)
    n_paths = S.shape[0]
    positions = np.zeros((n_paths, N), dtype=np.float64)
    prev = np.zeros(n_paths, dtype=np.float64)

    for i in range(N):
        hist = S[:, :i+1]
        positions[:, i] = policy_step(i, hist, prev)
        prev = positions[:, i]

    return positions


def evaluate_classical(out_dir: Path, quick: bool = False) -> None:
    """Evaluate classical baselines on shared test paths.

    Evaluates bs_delta, leland, ww_band on the same test paths used by units.
    Saves to classical.json with per-path losses and stats.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    start_iso = get_iso_now()

    # Common parameters
    H = 0.1
    N = 50
    T = 30 / 365
    strike = 1.0  # Normalized to 1.0
    cost = 0.001  # 10 basis points
    xi0 = 0.1
    sigma = math.sqrt(xi0)  # Implied vol for classical

    n_test = 2000 if quick else 20000

    # Generate shared test paths (normalized)
    S_test = shared_test_paths(n_test)
    test_paths_hash = make_array_hash(S_test)

    # Create step-wise policies
    policy_steps = {
        "bs_delta": baselines.bs_delta_policy(strike=strike, sigma=sigma, T=T, n_steps=N),
        "leland": baselines.leland_delta_policy(strike=strike, sigma=sigma, T=T, n_steps=N, cost=cost),
        "ww_band": baselines.ww_policy(strike=strike, sigma=sigma, T=T, n_steps=N, cost=cost, risk_aversion=10.0),
    }

    results = {
        "test_paths_sha256": test_paths_hash,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "started_at": start_iso,
        "n_test": n_test,
        "classical_arms": {}
    }

    for name, policy_step in policy_steps.items():
        try:
            # Apply step-wise policy
            positions = _apply_classical_policy_stepwise(policy_step, S_test, N)

            # Compute losses using hedging.hedging_pnl
            hedge_result = hedging.hedging_pnl(S_test, positions, strike=strike, cost=cost)
            losses_np = -hedge_result.pnl  # Loss = -PnL

            # Encode as base64
            losses_f32 = losses_np.astype(np.float32)
            losses_b64 = base64.b64encode(losses_f32.tobytes()).decode("ascii")

            # Compute stats
            arm_stats = {
                "cvar_95": float(risk.cvar_tail_mean(losses_np, 0.95)),
                "cvar_90": float(risk.cvar_tail_mean(losses_np, 0.90)),
                "cvar_99": float(risk.cvar_tail_mean(losses_np, 0.99)),
                "var_95": float(np.quantile(losses_np, 0.95)),
                "mean": float(losses_np.mean()),
                "sd": float(losses_np.std()),
                "turnover": float(hedging.turnover(positions).mean()),
            }

            results["classical_arms"][name] = {
                "per_path_losses": {
                    "b64": losses_b64,
                    "shape": [int(n_test)]
                },
                "stats": arm_stats
            }
            print(f"  {name}: CVaR95={arm_stats['cvar_95']:.4f}", flush=True)
        except Exception as e:
            print(f"  {name}: FAILED - {e}", flush=True)

    end_time = time.time()
    end_iso = get_iso_now()
    results["finished_at"] = end_iso
    results["wall_s"] = end_time - start_time

    # Save results
    out_file = out_dir / "classical.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {out_file}", flush=True)


def collect_results(out_dir: Path) -> None:
    """Collect all unit JSONs into recipes.json with across-seed stats."""
    out_dir = Path(out_dir)

    # Load all unit results
    units = {}
    for json_file in sorted(out_dir.glob("R-*_*_*.json")):
        try:
            data = json.loads(json_file.read_text())
            recipe = data["config"]["recipe"]
            arm = data["config"]["arm"]
            seed = data["config"]["seed"]

            key = (recipe, arm)
            if key not in units:
                units[key] = []
            units[key].append(data)
        except Exception as e:
            print(f"Warning: failed to load {json_file}: {e}", flush=True)

    # Aggregate statistics
    aggregated = {
        "experiment": "t4_recipes",
        "timestamp": get_iso_now(),
        "by_recipe_arm": {},
        "ranking": {}
    }

    for (recipe, arm), results_list in sorted(units.items()):
        n_seeds = len(results_list)

        # Extract CVaR95 per seed
        cvar95_per_seed = np.array([r["stats"]["cvar_95"] for r in results_list])

        # Seed mean and CI
        mean_cvar = cvar95_per_seed.mean()
        sd_cvar = cvar95_per_seed.std()
        se_cvar = sd_cvar / np.sqrt(n_seeds)
        ci_lo = mean_cvar - 1.96 * se_cvar
        ci_hi = mean_cvar + 1.96 * se_cvar

        # Use string key for JSON serialization
        key = f"{recipe}_{arm}"
        aggregated["by_recipe_arm"][key] = {
            "recipe": recipe,
            "arm": arm,
            "n_seeds": n_seeds,
            "cvar95_mean": float(mean_cvar),
            "cvar95_sd": float(sd_cvar),
            "cvar95_se": float(se_cvar),
            "cvar95_ci_lo": float(ci_lo),
            "cvar95_ci_hi": float(ci_hi),
            "individual_cvar95": cvar95_per_seed.tolist(),
            "individual_seeds": [r["config"]["seed"] for r in results_list],
        }

        print(f"{recipe:8s} {arm:5s}: CVaR95 {mean_cvar:.4f} ± {se_cvar:.4f}", flush=True)

    # Save aggregated results
    out_file = out_dir / "recipes.json"
    out_file.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"Saved {out_file}", flush=True)


def run_unit_job(args_tuple: tuple) -> tuple[str, int]:
    """Helper to run a single unit job and return (job_name, rc)."""
    recipe, arm, seed, out_dir, quick, log_file = args_tuple
    cmd = [
        sys.executable, str(Path(__file__)), "unit",
        "--recipe", recipe, "--arm", arm, "--seed", str(seed),
        "--out", str(out_dir)
    ]
    if quick:
        cmd.append("--quick")

    with open(log_file, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return f"{recipe}_{arm}_{seed}", rc


def run_classical_job(args_tuple: tuple) -> tuple[str, int]:
    """Helper to run classical job and return (job_name, rc)."""
    out_dir, quick, log_file = args_tuple
    cmd = [sys.executable, str(Path(__file__)), "classical", "--out", str(out_dir)]
    if quick:
        cmd.append("--quick")

    with open(log_file, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return "classical", rc




def main():
    parser = argparse.ArgumentParser(
        description="T4 recipe study: train units and evaluate baselines"
    )
    parser.add_argument("command", choices=["unit", "units", "classical", "collect", "batch", "time", "time-one", "scale"],
                        help="Command to run")
    parser.add_argument("--recipe", type=str, default="R-GPU",
                        choices=["R-DH", "R-SIG", "R-NTBN", "R-HOR", "R-GPU"],
                        help="Training recipe (for unit command)")
    parser.add_argument("--arm", type=str, default="FFN",
                        choices=["FFN", "GRU", "sig2"],
                        help="Hedger architecture (for unit command)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (for unit command)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent processes (for batch command)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of optimizer steps (for time command)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Units per process run on separate threads/CUDA streams (batch, units)")
    parser.add_argument("--jobs", type=str, default="",
                        help="Comma list recipe:arm:seed (units command)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: minimal training for testing")

    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.command == "unit":
        train_unit(args.recipe, args.arm, args.seed, out_dir, args.quick)
    elif args.command == "classical":
        evaluate_classical(out_dir, args.quick)
    elif args.command == "collect":
        collect_results(out_dir)
    elif args.command == "batch":
        batch_run(out_dir, args.workers, args.quick, args.threads)
    elif args.command == "units":
        jobs = [(r, a, int(sd)) for r, a, sd in (j.split(":") for j in args.jobs.split(",") if j)]
        run_units_threaded(jobs, out_dir, args.quick, args.threads)
    elif args.command == "time":
        time_recipes(out_dir, args.steps, args.quick)
    elif args.command == "time-one":
        out_dir.write_text(json.dumps(time_one(args.recipe, args.arm, args.steps, monitor=True)))
    elif args.command == "scale":
        scale_probe(out_dir, args.recipe, args.arm, args.steps)


if __name__ == "__main__":
    main()
