#!/usr/bin/env python3
"""Notebook 03 hedging on REAL market windows with immutable run-scoped artifacts.

Commands (run from anywhere):
  python3 scripts/exp_nb03_hedging.py train --run-id ID [--quick] [--workers N]
      Train independent seeded hedgers on calibrated simulated rough-Bergomi paths.
  python3 scripts/exp_nb03_hedging.py replay --run-id ID
      Replay the completed policies on real windows into results/nb03/runs/ID/.
  python3 scripts/exp_nb03_hedging.py figures --run-id ID
      Render new figures/fig-4.5-ID.* and figures/fig-4.6-ID.* only.

Test windows are real: daily S&P 500 closes (^GSPC_1d) with VIX implied vol (^VIX_1d / 100, joined by
date), and hourly bars (^GSPC_1h). Training data is simulated, test data is never simulated.
Loss = -pnl, in notional units with S0 = 1, cost 10 bp per side. Commands never overwrite
an existing run directory or figure artifact.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES_DIR = ROOT / "results" / "nb03"
FIG_DIR = ROOT / "figures"
RUN_ID: str | None = None
RUN_DIR: Path | None = None
MODEL_DIR: Path | None = None
TRAIN_LOG: Path | None = None
HEDGING_JSON: Path | None = None
INTRADAY_JSON: Path | None = None

COST = 0.001            # 10 bp per side
COST_BP = 10
STRIKE = 1.0
WW_GAMMA = 10.0         # Whalley-Wilmott risk aversion (same default as rough_hedge.experiment)
WIDTH = 16              # hidden width of learned hedgers (same default as rough_hedge.experiment)
N_SEEDS = 5
LEARNED = ("ffn", "gru", "sig2")
CLASSICAL = ("bs_delta_vix", "leland", "ww_band")
DAILY_ARMS = CLASSICAL + LEARNED
INTRA_ARMS = ("bs_delta_vix", "ww_band", "gru")
DAILY_LEN = 21
INTRA_DAYS = 5
CVAR_ALPHA = 0.95
N_BOOT = 2000
BOOT_SEED = 12345
PAIRED_SEED = 777
VIX_THRESHOLD = 20.0
MIN_WINDOWS = 10
PAIRED_MIN = 20         # rough_hedge.stats.paired_bootstrap needs n >= 20


def configure_run(run_id: str, *, create: bool = False) -> None:
    """Bind this invocation to an immutable result directory.

    Every official run has a caller-supplied identifier.  Training refuses an
    existing directory; replay and figure rendering only consume that exact
    directory.  This keeps quick artifacts and earlier figures immutable.
    """
    global RUN_ID, RUN_DIR, MODEL_DIR, TRAIN_LOG, HEDGING_JSON, INTRADAY_JSON
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("run_id must use only lowercase letters, digits, '-' and '_'")
    run_dir = RES_DIR / "runs" / run_id
    if create:
        if run_dir.exists():
            raise FileExistsError(f"immutable run already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        (run_dir / "models").mkdir()
        (run_dir / "workers").mkdir()
    elif not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    RUN_ID = run_id
    RUN_DIR = run_dir
    MODEL_DIR = run_dir / "models"
    TRAIN_LOG = run_dir / "train_log.json"
    HEDGING_JSON = run_dir / "hedging.json"
    INTRADAY_JSON = run_dir / "intraday_rehedging.json"


def require_run_paths() -> tuple[Path, Path, Path, Path, Path]:
    if None in (RUN_DIR, MODEL_DIR, TRAIN_LOG, HEDGING_JSON, INTRADAY_JSON):
        raise RuntimeError("configure_run() must be called before executing a command")
    return RUN_DIR, MODEL_DIR, TRAIN_LOG, HEDGING_JSON, INTRADAY_JSON


def figure_paths() -> tuple[Path, Path, Path, Path, Path]:
    """Return new, run-scoped figure paths without touching legacy figures."""
    if RUN_ID is None:
        raise RuntimeError("configure_run() must be called before rendering figures")
    # ``with_suffix`` treats the ``.5-<run-id>`` portion as a suffix and
    # would collapse both figure identifiers to ``fig-4.*``.  Construct the
    # complete file names directly so each official run produces new files.
    # ``analysis`` identifies this immutable rendering revision.  It also
    # ensures a repaired rendering never rewrites a previously emitted file.
    return (FIG_DIR / f"fig-4.5-{RUN_ID}-analysis.csv", FIG_DIR / f"fig-4.5-{RUN_ID}-analysis.png",
            FIG_DIR / f"fig-4.6-{RUN_ID}-analysis.csv", FIG_DIR / f"fig-4.6-{RUN_ID}-analysis.png",
            FIG_DIR / f"fig-4.6-{RUN_ID}-analysis-failure_paths.csv")


# ----------------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def config_hash(cfg: dict) -> str:
    return sha256_bytes(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode())


def versions() -> dict:
    import scipy
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__}


def block_length(n: int) -> int:
    """Moving-block-bootstrap block length: n^(1/3) rounded, at least 1 (stated in provenance)."""
    return max(1, int(round(n ** (1.0 / 3.0))))


def cvar95(x) -> float:
    from rough_hedge.risk import cvar_tail_mean
    return float(cvar_tail_mean(np.asarray(x, dtype=np.float64), CVAR_ALPHA))


def load_calibration() -> tuple[dict, str]:
    """H, eta and rho = rho_rv_corrected from calibration.json. The clipped key `rho` is never used."""
    p = RES_DIR / "calibration.json"
    if not p.exists():
        raise FileNotFoundError(p)
    raw = p.read_bytes()
    cal = json.loads(raw)["calibration"]
    rho = cal["diagnostics"]["rho_rv_corrected"]
    out = {"H": float(cal["H"]), "eta": float(cal["eta"]), "rho": float(rho)}
    if not (0 < out["H"] < 0.5) or out["eta"] < 0 or not (-1 <= out["rho"] <= 1):
        raise ValueError(f"calibration out of range: {out}")
    return out, sha256_bytes(raw)


def load_manifest() -> dict:
    from rough_hedge import realdata
    probs = realdata.verify_manifest(ROOT / "data")
    if probs:
        raise RuntimeError(f"data manifest verification failed: {probs}")
    ents = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    return {f"{e['symbol']}_{e['interval']}": e for e in ents}


def manifest_sha(man: dict, keys: list[str]) -> list[str]:
    out = []
    for k in keys:
        if k not in man:
            raise KeyError(f"{k} not in MANIFEST")
        out.append(man[k]["sha256"])
    return out


def _naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.tz_localize(None)).normalize()


def load_daily():
    """Return (dates, close, vix_by_date, n_dropped_gspc, n_dropped_vix); vix is a decimal-vol Series."""
    from rough_hedge import realdata
    g = realdata.load_chart_json(ROOT / "data/raw/^GSPC_1d.json")
    v = realdata.load_chart_json(ROOT / "data/raw/^VIX_1d.json")
    gd = _naive_dates(g.index)
    vd = _naive_dates(v.index)
    if gd.has_duplicates or vd.has_duplicates:
        raise ValueError("duplicate dates in daily data")
    vix = pd.Series(v["close"].to_numpy() / 100.0, index=vd)
    return gd, g["close"].to_numpy(), vix, int(g.attrs["n_dropped"]), int(v.attrs["n_dropped"])


def xi0_from_vix(gd, vix) -> tuple[float, int]:
    common = gd.intersection(vix.index)
    if len(common) == 0:
        raise ValueError("no common dates between GSPC and VIX")
    return float(np.mean(vix.loc[common].to_numpy() ** 2)), int(len(common))


XI0_FORMULA = ("xi0 = mean_t (VIX_close_t / 100)^2 over all dates in ^GSPC_1d intersect ^VIX_1d "
               "(annualised variance)")


# ----------------------------------------------------------------------------------------------
# real windows
# ----------------------------------------------------------------------------------------------
def build_daily_windows():
    from rough_hedge import realdata
    gd, close, vix, g_drop, v_drop = load_daily()
    wins = realdata.non_overlapping_windows(len(close), DAILY_LEN)
    S, IV, starts, ends, regs, vix0 = [], [], [], [], [], []
    dropped_vix = dropped_gap = 0
    for a, b in wins:
        dates = gd[a:b + 1]
        if (dates[-1] - dates[0]).days > 35:
            dropped_gap += 1
            continue
        iv = vix.reindex(dates).to_numpy()
        if np.isnan(iv).any():
            dropped_vix += 1
            continue
        s = close[a:b + 1]
        S.append(s / s[0])
        IV.append(iv)
        starts.append(str(dates[0].date()))
        ends.append(str(dates[-1].date()))
        vix0.append(float(iv[0] * 100.0))
        regs.append("low" if iv[0] * 100.0 < VIX_THRESHOLD else "high")
    if not S:
        raise ValueError("no usable daily windows")
    info = {"n_candidate_windows": len(wins), "n_dropped_missing_vix": dropped_vix,
            "n_dropped_calendar_gap": dropped_gap, "n_windows": len(S),
            "gspc_rows_dropped_null": g_drop, "vix_rows_dropped_null": v_drop,
            "n_low": int(sum(r == "low" for r in regs)), "n_high": int(sum(r == "high" for r in regs))}
    return (np.array(S), np.array(IV), np.array(starts), np.array(ends), np.array(regs),
            np.array(vix0), info, gd, vix)


def build_intraday_windows():
    """5-trading-day windows from hourly bars. Returns per-window open/close matrices (w, 5, B)."""
    from rough_hedge import realdata
    h = realdata.load_chart_json(ROOT / "data/raw/^GSPC_1h.json")
    gd, _, vix, _, _ = load_daily()
    ny = h.index.tz_convert("America/New_York")
    ndate = pd.DatetimeIndex(ny.tz_localize(None)).normalize()
    cnt = pd.Series(1, index=ndate).groupby(level=0).sum()
    bars_per_day = int(cnt.mode().iloc[0])
    day_open = {d: g.to_numpy() for d, g in h["open"].groupby(ndate)}
    day_close = {d: g.to_numpy() for d, g in h["close"].groupby(ndate)}
    n_incomplete_days = int((cnt != bars_per_day).sum())
    cal = gd[(gd >= cnt.index.min()) & (gd <= cnt.index.max())]
    in_cal = set(cal)
    n_hourly_not_in_calendar = int(sum(d not in in_cal for d in cnt.index))
    comp = set(d for d in cnt.index if cnt.loc[d] == bars_per_day)
    O, C, VX, starts, ends, dts, regs = [], [], [], [], [], [], []
    dropped_incomplete = dropped_vix = 0
    for k in range(len(cal) // INTRA_DAYS):
        days = cal[k * INTRA_DAYS:(k + 1) * INTRA_DAYS]
        if any(d not in comp for d in days):
            dropped_incomplete += 1
            continue
        iv = vix.reindex(days).to_numpy()
        if np.isnan(iv).any():
            dropped_vix += 1
            continue
        O.append(np.stack([day_open[d] for d in days]))
        C.append(np.stack([day_close[d] for d in days]))
        VX.append(iv)
        starts.append(str(days[0].date()))
        ends.append(str(days[-1].date()))
        dts.append([str(d.date()) for d in days])
        regs.append("low" if iv[0] * 100.0 < VIX_THRESHOLD else "high")
    if not O:
        raise ValueError("no usable intraday windows")
    ks = [k for k in (1, 2, 4, 7) if k <= bars_per_day]
    info = {"bars_per_day": bars_per_day, "rehedges_per_day": ks, "n_calendar_blocks": len(cal) // INTRA_DAYS,
            "n_dropped_incomplete_bars": dropped_incomplete, "n_dropped_missing_vix": dropped_vix,
            "n_windows": len(O), "n_hourly_days_incomplete": n_incomplete_days,
            "n_hourly_days_not_in_gspc_calendar": n_hourly_not_in_calendar,
            "n_low": int(sum(r == "low" for r in regs)), "n_high": int(sum(r == "high" for r in regs))}
    return (np.array(O), np.array(C), np.array(VX), np.array(starts), np.array(ends), dts,
            np.array(regs), info)


def intraday_paths(O, C, VX, k: int):
    """Subsample bars within each day: bar indices floor(B*j/k), price = bar OPEN at decision time,
    plus the final close of the last bar of day 5. Returns S (w, 5k+1) normalised by S0, iv (w, 5k+1)."""
    w, nd, B = O.shape
    idx = [int(np.floor(B * j / k)) for j in range(k)]
    pts = np.concatenate([O[:, d, idx] for d in range(nd)], axis=1)
    pts = np.concatenate([pts, C[:, -1, -1:]], axis=1)
    iv = np.concatenate([np.repeat(VX[:, d:d + 1], k, axis=1) for d in range(nd)] + [VX[:, -1:]], axis=1)
    return pts / pts[:, :1], iv


# ----------------------------------------------------------------------------------------------
# training (simulated rough-Bergomi paths -> learned hedgers)
# ----------------------------------------------------------------------------------------------
def make_model(arm: str, seed: int):
    from rough_hedge.models import FFNHedger, GRUHedger, SignatureHedger
    if arm == "ffn":
        return FFNHedger(hidden=WIDTH, seed=seed)
    if arm == "gru":
        return GRUHedger(hidden=WIDTH, seed=seed)
    if arm == "sig2":
        return SignatureHedger(depth=2, hidden=WIDTH, seed=seed)
    raise ValueError(arm)


def train_config(quick: bool, cal: dict, xi0: float, n_xi: int, ks: list[int]) -> dict:
    cfg = {"experiment": "nb03_train", "H": cal["H"], "eta": cal["eta"], "rho": cal["rho"],
           "rho_source": "rho_rv_corrected", "xi0": xi0, "xi0_formula": XI0_FORMULA, "xi0_n_dates": n_xi,
           "cost": COST, "cost_bp": COST_BP, "strike": STRIKE, "width": WIDTH, "n_seeds": N_SEEDS,
           "daily": {"N": DAILY_LEN, "T": DAILY_LEN / 252, "arms": list(LEARNED)},
           "intraday": {"days": INTRA_DAYS, "T": INTRA_DAYS / 252, "arm": "gru", "rehedges_per_day": ks},
           "optimizer": "adam", "lr": 5e-3, "patience": 10, "quick": quick,
           "data_seed_rule": "daily: seed; intraday: 100*k + seed (shared across arms of one seed)"}
    if quick:
        cfg.update({"n_train": 2048, "n_val": 512, "batch": 512, "epochs": 2})
    else:
        cfg.update({"n_train": 65536, "n_val": 16384, "batch": 4096, "epochs": 50})
    return cfg


def _train_one_unit(payload: dict) -> dict:
    """Run one seeded unit in its own process and stream one record per epoch."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    from rough_hedge.training import make_splits, peak_memory_bytes, train_hedger

    unit, cfg, cal, xi0 = payload["unit"], payload["cfg"], payload["cal"], payload["xi0"]
    run_dir = Path(payload["run_dir"])
    worker_log = run_dir / "workers" / f"{unit['name']}.jsonl"
    model_dir = run_dir / "models"
    t0 = time.time()
    dseed = unit["seed"] if unit["kind"] == "daily" else 100 * unit["k"] + unit["seed"]
    with worker_log.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        print(json.dumps({"event": "started", "unit": unit["name"], "pid": os.getpid(), "at": now_iso()}), flush=True)
        splits = make_splits({
            "seed": dseed, "n_train": cfg["n_train"], "n_val": cfg["n_val"], "n_test": 1,
            "n_steps": unit["N"], "T": float(unit["T"]), "strike": STRIKE, "cost": COST,
            "market": "rbergomi", "H": cal["H"], "eta": cal["eta"], "rho": cal["rho"], "xi0": xi0,
        })
        sim_wall_s = time.time() - t0
        model = make_model(unit["arm"], unit["seed"])
        tcfg = {"epochs": cfg["epochs"], "batch_size": cfg["batch"], "lr": cfg["lr"],
                "patience": cfg["patience"], "optimizer": "adam"}

        def progress(event: dict) -> None:
            event.update({"event": "epoch", "unit": unit["name"], "at": now_iso()})
            print(json.dumps(event), flush=True)

        (model, hist), peak = peak_memory_bytes(
            lambda: train_hedger(model, splits, tcfg, seed=unit["seed"], device=payload["device"],
                                  progress_callback=progress), payload["device"])
        path = model_dir / f"{unit['name']}.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        entry = dict(unit, model_file=str(path.relative_to(ROOT)), model_sha256=sha256_file(path),
                     history=hist, wall_s=time.time() - t0, sim_wall_s=sim_wall_s,
                     device=payload["device"], peak_mem_bytes=int(peak),
                     peak_mem_kind=("cuda_max_allocated" if payload["device"] != "cpu" else "cpu_rss_growth"),
                     worker_log=str(worker_log.relative_to(ROOT)))
        print(json.dumps({"event": "finished", "unit": unit["name"], "wall_s": entry["wall_s"],
                          "best_val_cvar": min(hist["val_cvar"]), "at": now_iso()}), flush=True)
    return entry


def cmd_train(quick: bool, workers: int):
    import torch
    run_dir, _, train_log, _, _ = require_run_paths()
    if workers < 1:
        raise ValueError("workers must be at least one")
    t_start, started = time.time(), now_iso()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cal, cal_sha = load_calibration()
    man = load_manifest()
    gd, _, vix, _, _ = load_daily()
    xi0, n_xi = xi0_from_vix(gd, vix)
    ks = build_intraday_windows()[-1]["rehedges_per_day"]
    cfg = train_config(quick, cal, xi0, n_xi, ks)
    cfg_sha = config_hash(cfg)
    units = [dict(kind="daily", arm=a, seed=s, k=None, N=DAILY_LEN, T=DAILY_LEN / 252,
                  name=f"daily_{a}_{s}") for s in range(N_SEEDS) for a in LEARNED]
    units += [dict(kind="intraday", arm="gru", seed=s, k=k, N=INTRA_DAYS * k, T=INTRA_DAYS / 252,
                   name=f"intraday_gru_k{k}_{s}") for k in ks for s in range(N_SEEDS)]
    log = {"experiment": "nb03_train", "run_id": RUN_ID, "config": cfg, "config_sha256": cfg_sha,
           "quick": quick, "device": device, "workers": workers, "calibration_sha256": cal_sha,
           "finished": False, "units": [], "data_sha256": manifest_sha(man, ["^GSPC_1d", "^VIX_1d"]),
           "versions": versions(), "source_sha256": sha256_file(Path(__file__)), "started_at": started}
    if device.startswith("cuda"):
        log["gpu_name"] = torch.cuda.get_device_name(0)
    train_log.write_text(json.dumps(log, indent=2))
    print(f"train: {len(units)} independent units on {device}; workers={workers}; run_id={RUN_ID}", flush=True)
    payloads = [dict(unit=u, cfg=cfg, cal=cal, xi0=xi0, device=device, run_dir=str(run_dir)) for u in units]
    if workers == 1:
        iterator = (_train_one_unit(p) for p in payloads)
        for entry in iterator:
            log["units"].append(entry)
            log["wall_s"] = time.time() - t_start
            train_log.write_text(json.dumps(log, indent=2))
            print(f"  {entry['name']}: {entry['wall_s']:.1f}s, epochs={entry['history']['epochs_run']}", flush=True)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_train_one_unit, p) for p in payloads]
            for future in as_completed(futures):
                entry = future.result()
                log["units"].append(entry)
                log["wall_s"] = time.time() - t_start
                train_log.write_text(json.dumps(log, indent=2))
                print(f"  {entry['name']}: {entry['wall_s']:.1f}s, epochs={entry['history']['epochs_run']}", flush=True)
    log["units"].sort(key=lambda u: u["name"])
    log.update({"finished": True, "finished_at": now_iso(), "wall_s": time.time() - t_start})
    train_log.write_text(json.dumps(log, indent=2))
    print(f"train done: {len(units)} units, {log['wall_s']:.1f}s", flush=True)


# ----------------------------------------------------------------------------------------------
# replay
# ----------------------------------------------------------------------------------------------
def classical_policy(kind: str, T: float, N: int):
    from rough_hedge.baselines import (bs_call_delta, bs_call_gamma, leland_call_delta,
                                       ww_half_width, apply_no_trade_band)
    dt = T / N

    def policy(i, S_hist, iv_hist, prev):
        tau = T - i * dt
        out = np.empty(S_hist.shape[0])
        for r in range(S_hist.shape[0]):
            s = float(S_hist[r, -1])
            sig = float(iv_hist[r, -1])
            if kind == "leland":
                out[r] = np.ravel(leland_call_delta(s, STRIKE, sig, tau, COST, dt))[0]
            elif kind == "ww_band":
                centre = np.ravel(bs_call_delta(s, STRIKE, sig, tau))[0]
                gam = np.ravel(bs_call_gamma(s, STRIKE, sig, tau))[0]
                half = np.ravel(ww_half_width(s, gam, COST, WW_GAMMA))[0]
                out[r] = np.ravel(apply_no_trade_band(prev[r], centre, half))[0]
            else:
                raise ValueError(kind)
        return out
    return policy


def learned_policy(model, T: float, N: int):
    """Causal callable policy(i, S_hist, iv_hist, prev_pos): log-moneyness history padded with its last
    value to length N+1; the hedger is causal so only column i of its output is used."""
    import torch
    times = torch.tensor(np.arange(N + 1, dtype=np.float64) * T / N, dtype=model.dtype)

    def policy(i, S_hist, iv_hist, prev):
        logm = np.log(S_hist / S_hist[:, :1])
        pad = np.repeat(logm[:, -1:], N - i, axis=1)
        x = np.concatenate([logm, pad], axis=1)
        with torch.no_grad():
            pos = model(torch.tensor(x, dtype=model.dtype), times)
        return pos[:, i].double().numpy()
    return policy


def load_learned(unit: dict):
    import torch
    p = ROOT / unit["model_file"]
    if not p.exists():
        raise FileNotFoundError(p)
    if sha256_file(p) != unit["model_sha256"]:
        raise RuntimeError(f"{p} does not match train_log sha256")
    m = make_model(unit["arm"], unit["seed"])
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def replay_policy(S, iv, T, policy):
    from rough_hedge import realdata
    return realdata.replay_hedge(S, iv, STRIKE, COST, policy, T)


def leakage_probe(name: str, S, iv, T, policy, rng) -> dict:
    """Perturb prices and vols AFTER step i: positions[:, :i+1] must be unchanged (exact). Power control:
    perturbing the PAST must change positions[:, i]. Raises on any leak."""
    n, Np1 = S.shape
    N = Np1 - 1
    base = replay_policy(S, iv, T, policy).positions
    max_diff, power_ok = 0.0, []
    for i in sorted({max(1, N // 4), max(2, N // 2), max(3, (3 * N) // 4)}):
        S2, iv2 = S.copy(), iv.copy()
        S2[:, i + 1:] *= np.exp(0.2 * rng.standard_normal((n, N - i)))
        iv2[:, i + 1:] *= np.exp(0.2 * rng.standard_normal((n, N - i)))
        pos2 = replay_policy(S2, iv2, T, policy).positions
        d = float(np.max(np.abs(pos2[:, :i + 1] - base[:, :i + 1])))
        max_diff = max(max_diff, d)
        if d != 0.0:
            raise RuntimeError(f"LEAKAGE: {name}: perturbing prices after step {i} changed earlier "
                               f"positions by {d:.3e}")
        S3 = S.copy()
        S3[:, 1:i + 1] *= np.exp(0.2 * rng.standard_normal((n, i)))
        pos3 = replay_policy(S3, iv, T, policy).positions
        power_ok.append(bool(np.max(np.abs(pos3[:, i] - base[:, i])) > 0))
    if not all(power_ok):
        raise RuntimeError(f"probe has no power for {name}: past perturbation did not change positions")
    return {"policy": name, "max_abs_diff_future_perturbation": max_diff, "passed": True,
            "past_perturbation_changes_positions": True}


def pnl_path(S_row, pos_row, premium, strike=STRIKE, cost=COST):
    """Cumulative P&L at every grid point; final entry includes unwind cost and call payoff."""
    prev = np.concatenate([[0.0], pos_row[:-1]])
    step = pos_row * (S_row[1:] - S_row[:-1]) - cost * np.abs(pos_row - prev) * S_row[:-1]
    path = premium + np.concatenate([[0.0], np.cumsum(step)])
    path[-1] -= cost * abs(pos_row[-1]) * S_row[-1] + max(S_row[-1] - strike, 0.0)
    return path


# ---- statistics ------------------------------------------------------------------------------
def seedmean_cvar_boot(L: np.ndarray, block: int):
    """L: (n_seeds, n). Statistic = mean over seeds of CVaR95 on the SAME block-resampled windows."""
    from rough_hedge.realdata import moving_block_bootstrap
    n = L.shape[1]
    idx = np.arange(n, dtype=np.float64)
    vals = []

    def stat(ii):
        j = ii.astype(int)
        v = float(np.mean([cvar95(L[s, j]) for s in range(L.shape[0])]))
        vals.append(v)
        return v
    est, lo, hi = moving_block_bootstrap(idx, block, N_BOOT, BOOT_SEED, stat)
    return est, lo, hi, float(np.std(vals[1:], ddof=1))


def summarise(losses: dict, mask: np.ndarray, regime: str, bs_arm="bs_delta_vix") -> list[dict]:
    from rough_hedge.stats import paired_bootstrap, holm
    n = int(mask.sum())
    if n == 0:
        return []
    block = block_length(n)
    rows = []
    for arm, L in losses.items():
        Lm = L[:, mask]
        est, lo, hi, se = seedmean_cvar_boot(Lm, block)
        per = []
        for s in range(Lm.shape[0]):
            e, l, h, _ = seedmean_cvar_boot(Lm[s:s + 1], block)
            per.append({"cvar95": e, "lo": l, "hi": h})
        pc = [p["cvar95"] for p in per]
        rows.append({"arm": arm, "regime": regime, "n_windows": n, "block_length": block, "n_boot": N_BOOT,
                     "cvar95": est, "cvar_lo": lo, "cvar_hi": hi, "cvar_boot_se": se,
                     "per_seed": per, "seed_min": min(pc), "seed_max": max(pc),
                     "mean_loss": float(Lm.mean()), "flag_lt_10_windows": n < MIN_WINDOWS})
    bs = {r["arm"]: r for r in rows}[bs_arm]
    bs_loss = losses[bs_arm][0, mask]
    praw, names = [], []
    for r in rows:
        arm = r["arm"]
        if arm == bs_arm:
            r.update({"delta_vs_bs": 0.0, "p_raw": None, "p_holm": None, "per_seed_delta": [0.0]})
            continue
        r["delta_vs_bs"] = r["cvar95"] - bs["cvar95"]
        r["per_seed_delta"] = [p["cvar95"] - bs["cvar95"] for p in r["per_seed"]]
        if n >= PAIRED_MIN:
            res = [paired_bootstrap(losses[arm][s, mask], bs_loss, n_boot=N_BOOT, seed=PAIRED_SEED)
                   for s in range(losses[arm].shape[0])]
            r["per_seed_p"] = [x.pvalue for x in res]
            r["per_seed_delta_ci"] = [[x.lo, x.hi] for x in res]
            r["p_raw"] = float(max(x.pvalue for x in res))   # conservative: must hold for every seed
            praw.append(r["p_raw"])
            names.append(arm)
        else:
            r["p_raw"] = None
            r["p_holm"] = None
            r["p_note"] = f"n={n} < {PAIRED_MIN}: paired bootstrap not run"
    if praw:
        adj, _ = holm(np.array(praw))
        for a, p in zip(names, adj):
            [r for r in rows if r["arm"] == a][0]["p_holm"] = float(p)
    return rows


def load_train_log() -> dict:
    _, _, train_log, _, _ = require_run_paths()
    if not train_log.exists():
        raise FileNotFoundError(f"{train_log}: run `train` first")
    log = json.loads(train_log.read_text())
    if not log.get("finished"):
        raise RuntimeError("train_log.json is not finished")
    return log


def provenance(cfg: dict, data_keys: list[str], man: dict, log: dict, cal_sha: str, t0: float,
               started: str) -> dict:
    _, _, train_log, _, _ = require_run_paths()
    return {"config": cfg, "config_sha256": config_hash(cfg),
            "data_sha256": manifest_sha(man, data_keys), "data_files": data_keys,
            "calibration_sha256": cal_sha, "rho_source": "rho_rv_corrected",
            "xi0_formula": XI0_FORMULA, "xi0": log["config"]["xi0"],
            "run_id": RUN_ID, "train_log_sha256": sha256_file(train_log), "train_config_sha256": log["config_sha256"],
            "train_device": log["device"], "device": "cpu", "versions": versions(), "quick": log["quick"],
            "trained_on": "calibrated simulated rough-Bergomi paths (never on the real test windows)",
            "started_at": started, "finished_at": now_iso(), "wall_s": time.time() - t0}


def print_cvar_table(title: str, summ: list[dict]):
    print(title)
    for r in summ:
        flag = " [n<10]" if r["flag_lt_10_windows"] else ""
        ph = "" if r.get("p_holm") is None else f" p_holm={r['p_holm']:.3f}"
        print(f"  {r['regime']:5s} {r['arm']:13s} n={r['n_windows']:3d} CVaR95={r['cvar95']:.5f} "
              f"[{r['cvar_lo']:.5f},{r['cvar_hi']:.5f}] seeds[{r['seed_min']:.5f},{r['seed_max']:.5f}] "
              f"delta={r['delta_vs_bs']:+.5f}{ph}{flag}")


def cmd_replay():
    from rough_hedge.realdata import moving_block_bootstrap
    _, _, _, hedging_json, intraday_json = require_run_paths()
    t0 = time.time()
    started = now_iso()
    log = load_train_log()
    cal, cal_sha = load_calibration()
    if cal_sha != log["calibration_sha256"]:
        raise RuntimeError("calibration.json changed since training")
    man = load_manifest()
    rng = np.random.default_rng(2024)

    # ---------------- daily ----------------
    S, IV, starts, ends, regs, vix0, winfo, gd, vix = build_daily_windows()
    xi0, _ = xi0_from_vix(gd, vix)
    if abs(xi0 - log["config"]["xi0"]) > 1e-15:
        raise RuntimeError("xi0 differs from the training log")
    T, N = DAILY_LEN / 252, DAILY_LEN
    print(f"daily windows: {winfo}", flush=True)
    dres: dict = {}
    losses: dict = {}
    probes = []
    dres["bs_delta_vix"] = {"seeds": [None], "r": [replay_policy(S, IV, T, "bs_delta")]}
    for arm in ("leland", "ww_band"):
        pol = classical_policy(arm, T, N)
        probes.append(leakage_probe(arm, S, IV, T, pol, rng))
        dres[arm] = {"seeds": [None], "r": [replay_policy(S, IV, T, pol)]}
    for arm in LEARNED:
        us = sorted([u for u in log["units"] if u["kind"] == "daily" and u["arm"] == arm],
                    key=lambda u: u["seed"])
        if len(us) != N_SEEDS:
            raise RuntimeError(f"{arm}: expected {N_SEEDS} trained seeds, found {len(us)}")
        rs = []
        for u in us:
            pol = learned_policy(load_learned(u), T, N)
            probes.append(leakage_probe(f"daily_{arm}_{u['seed']}", S, IV, T, pol, rng))
            rs.append(replay_policy(S, IV, T, pol))
        dres[arm] = {"seeds": [u["seed"] for u in us], "r": rs}
    for arm in DAILY_ARMS:
        losses[arm] = np.stack([-x.pnl for x in dres[arm]["r"]])
        if not np.all(np.isfinite(losses[arm])):
            raise ValueError(f"non-finite losses for {arm}")
    summ = []
    for reg, m in (("all", np.ones(len(S), dtype=bool)), ("low", regs == "low"), ("high", regs == "high")):
        summ += summarise(losses, m, reg)
    cfg = {"experiment": "hedging", "window_len_days": DAILY_LEN, "T": T, "N": N, "strike": STRIKE,
           "cost": COST, "cost_bp": COST_BP, "vix_threshold": VIX_THRESHOLD,
           "regime_rule": "low if VIX close at window start < 20 else high",
           "arms": list(DAILY_ARMS), "learned_arms": list(LEARNED), "n_seeds": N_SEEDS,
           "ww_gamma": WW_GAMMA, "cvar_alpha": CVAR_ALPHA, "n_boot": N_BOOT,
           "bootstrap": "moving block bootstrap, block length = round(n^(1/3)) per regime subset",
           "paired_test": "paired_bootstrap per seed (iid windows); p of a learned arm = max over seeds; "
                          "Holm across the 5 non-reference arms within each regime",
           "iv_rule": "^VIX_1d close / 100 joined by date", "windows": winfo,
           "train_config_sha256": log["config_sha256"]}
    out = {"experiment": "hedging",
           "provenance": provenance(cfg, ["^GSPC_1d", "^VIX_1d"], man, log, cal_sha, t0, started),
           "windows": {"start_date": starts.tolist(), "end_date": ends.tolist(), "regime": regs.tolist(),
                       "vix_start": vix0.tolist(), **winfo},
           "arms": {a: {"learned": a in LEARNED, "seeds": dres[a]["seeds"],
                        "losses": losses[a].tolist(),
                        "costs": [x.costs.tolist() for x in dres[a]["r"]],
                        "turnover": [x.turnover.tolist() for x in dres[a]["r"]]} for a in DAILY_ARMS},
           "summary": summ, "leakage_probe": probes}
    if hedging_json.exists():
        raise FileExistsError(f"refusing to overwrite immutable replay output: {hedging_json}")
    hedging_json.write_text(json.dumps(out, indent=2))
    print_cvar_table("daily CVaR95 (loss = -pnl, S0 = 1, 10 bp/side):", summ)
    print(f"leakage probe (daily): {len(probes)} policies, max|diff| over all = "
          f"{max(p['max_abs_diff_future_perturbation'] for p in probes)} -> "
          f"passed={all(p['passed'] for p in probes)}", flush=True)

    # ---------------- intraday ----------------
    O, C, VX, istarts, iends, idts, iregs, iinfo = build_intraday_windows()
    ks = iinfo["rehedges_per_day"]
    print(f"intraday windows: {iinfo}", flush=True)
    per_k: dict = {}
    iprobes = []
    for k in ks:
        Sk, IVk = intraday_paths(O, C, VX, k)
        Nk, Tk = INTRA_DAYS * k, INTRA_DAYS / 252
        ent: dict = {"bs_delta_vix": ([None], [replay_policy(Sk, IVk, Tk, "bs_delta")])}
        pol = classical_policy("ww_band", Tk, Nk)
        iprobes.append(leakage_probe(f"intraday_ww_band_k{k}", Sk, IVk, Tk, pol, rng))
        ent["ww_band"] = ([None], [replay_policy(Sk, IVk, Tk, pol)])
        us = sorted([u for u in log["units"] if u["kind"] == "intraday" and u["k"] == k],
                    key=lambda u: u["seed"])
        if len(us) != N_SEEDS:
            raise RuntimeError(f"gru k={k}: expected {N_SEEDS} seeds, found {len(us)}")
        rs = []
        for u in us:
            gp = learned_policy(load_learned(u), Tk, Nk)
            iprobes.append(leakage_probe(u["name"], Sk, IVk, Tk, gp, rng))
            rs.append(replay_policy(Sk, IVk, Tk, gp))
        ent["gru"] = ([u["seed"] for u in us], rs)
        per_k[k] = {"S": Sk, "ent": ent}
    n_w = len(O)
    block = block_length(n_w)
    isumm = []
    prev_L: dict = {}
    for k in ks:
        for arm in INTRA_ARMS:
            seeds, rs = per_k[k]["ent"][arm]
            L = np.stack([-x.pnl for x in rs])
            est, lo, hi, se = seedmean_cvar_boot(L, block)
            row = {"arm": arm, "rehedges_per_day": k, "cvar95": est, "cvar_lo": lo, "cvar_hi": hi,
                   "cvar_boot_se": se, "per_seed_cvar": [cvar95(L[s]) for s in range(len(L))],
                   "mean_turnover": float(np.mean([x.turnover.mean() for x in rs])),
                   "mean_cost": float(np.mean([x.costs.mean() for x in rs])),
                   "n_windows": n_w, "block_length": block, "cost_bp": COST_BP,
                   "gain_from_prev": None, "gain_se": None, "saturated": 0}
            if arm in prev_L:
                Lp = prev_L[arm]
                vals = []
                idx = np.arange(n_w, dtype=np.float64)

                def stat(ii, Lp=Lp, L=L):
                    j = ii.astype(int)
                    v = float(np.mean([cvar95(Lp[s, j]) for s in range(Lp.shape[0])])
                              - np.mean([cvar95(L[s, j]) for s in range(L.shape[0])]))
                    vals.append(v)
                    return v
                gain, _, _ = moving_block_bootstrap(idx, block, N_BOOT, BOOT_SEED, stat)
                gse = float(np.std(vals[1:], ddof=1))
                row["gain_from_prev"], row["gain_se"] = gain, gse
                row["saturated"] = int(gain < gse or gain < 0)
            prev_L[arm] = L
            isumm.append(row)
    # failure cases: worst 3 windows of bs_delta_vix at the finest frequency
    kmax = ks[-1]
    ent = per_k[kmax]["ent"]
    Sk = per_k[kmax]["S"]
    worst = np.argsort(ent["bs_delta_vix"][1][0].pnl)[:3]
    fail = []
    for w in worst:
        paths = {}
        for arm in INTRA_ARMS:
            rs = ent[arm][1]
            ps = np.stack([pnl_path(Sk[w], x.positions[w], x.premium[w]) for x in rs])
            for x, p in zip(rs, ps):
                if abs(p[-1] - x.pnl[w]) > 1e-9:
                    raise RuntimeError("pnl path reconstruction mismatch")
            paths[arm] = ps.mean(axis=0).tolist()
        fail.append({"window": int(w), "start_date": str(istarts[w]), "end_date": str(iends[w]),
                     "dates": idts[w], "rehedges_per_day": kmax,
                     "loss_bs_delta_vix": float(-ent["bs_delta_vix"][1][0].pnl[w]),
                     "S_over_S0": Sk[w].tolist(), "pnl_path": paths,
                     "x_days": (np.arange(INTRA_DAYS * kmax + 1) / kmax).tolist(),
                     "gru_path_note": "mean over seeds of per-seed cumulative P&L paths"})
    icfg = {"experiment": "intraday_rehedging", "window_days": INTRA_DAYS, "T": INTRA_DAYS / 252,
            "rehedges_per_day": ks, "N_rule": "N = 5k", "strike": STRIKE, "cost": COST,
            "cost_bp": COST_BP, "arms": list(INTRA_ARMS), "n_seeds": N_SEEDS, "ww_gamma": WW_GAMMA,
            "price_rule": "decision price = bar OPEN at bar indices floor(B*j/k) of each day; final point = "
                          "close of last bar of day 5; S normalised by S0 = first open",
            "iv_rule": "^VIX_1d close / 100 of that date", "cvar_alpha": CVAR_ALPHA, "n_boot": N_BOOT,
            "bootstrap": "moving block bootstrap, block length = round(n^(1/3))", "block_length": block,
            "saturated_rule": "CVaR gain from previous frequency < its block-bootstrap SE, or negative",
            "windows": iinfo, "train_config_sha256": log["config_sha256"]}
    perk_json = {str(k): {a: {"seeds": per_k[k]["ent"][a][0],
                              "losses": [(-x.pnl).tolist() for x in per_k[k]["ent"][a][1]],
                              "costs": [x.costs.tolist() for x in per_k[k]["ent"][a][1]],
                              "turnover": [x.turnover.tolist() for x in per_k[k]["ent"][a][1]]}
                          for a in INTRA_ARMS} for k in ks}
    iout = {"experiment": "intraday_rehedging",
            "provenance": provenance(icfg, ["^GSPC_1d", "^VIX_1d", "^GSPC_1h"], man, log, cal_sha, t0,
                                     started),
            "windows": {"start_date": istarts.tolist(), "end_date": iends.tolist(), "dates": idts,
                        "regime": iregs.tolist(), **iinfo},
            "per_k": perk_json, "summary": isumm, "failure_cases": fail, "leakage_probe": iprobes}
    if intraday_json.exists():
        raise FileExistsError(f"refusing to overwrite immutable replay output: {intraday_json}")
    intraday_json.write_text(json.dumps(iout, indent=2))
    print("intraday CVaR95:")
    for r_ in isumm:
        print(f"  k={r_['rehedges_per_day']} {r_['arm']:13s} CVaR95={r_['cvar95']:.5f} "
              f"[{r_['cvar_lo']:.5f},{r_['cvar_hi']:.5f}] turnover={r_['mean_turnover']:.3f} "
              f"cost={r_['mean_cost']:.5f} saturated={r_['saturated']}")
    print(f"intraday leakage probe: {len(iprobes)} policies, max|diff| = "
          f"{max(p['max_abs_diff_future_perturbation'] for p in iprobes)}, passed", flush=True)
    print(f"replay done in {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------------------------------------
# figures (reads result JSONs only)
# ----------------------------------------------------------------------------------------------
ARM_COLOR = {"bs_delta_vix": "#0072B2", "leland": "#D55E00", "ww_band": "#009E73",
             "ffn": "#CC79A7", "gru": "#E69F00", "sig2": "#56B4E9"}


def _ids(prov: dict) -> tuple[str, str, str]:
    ch = prov["config_sha256"]
    return ch[:8], ch, ";".join(prov["data_sha256"])


def cmd_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, _, _, hedging_json, intraday_json = require_run_paths()
    fig45_csv, fig45_png, fig46_csv, fig46_png, failure_csv = figure_paths()
    destinations = (fig45_csv, fig45_png, fig46_csv, fig46_png, failure_csv)
    collisions = [str(p) for p in destinations if p.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite existing figure artifact(s): {collisions}")
    FIG_DIR.mkdir(exist_ok=True)
    seedrange = "0-%d" % (N_SEEDS - 1)
    # ================= fig-4.5 =================
    h = json.loads(hedging_json.read_text())
    prov = h["provenance"]
    run_id, chash, dsha = _ids(prov)
    tag = "[QUICK PIPELINE CHECK, not final] " if prov["quick"] else ""
    arms = list(h["arms"].keys())
    regs = np.array(h["windows"]["regime"])
    pooled = {a: np.array(h["arms"][a]["losses"]) for a in arms}
    allv = np.concatenate([v.ravel() for v in pooled.values()])
    edges = np.linspace(allv.min(), allv.max(), 41)
    n_all = len(regs)

    def seedtxt(a):
        return seedrange if h["arms"][a]["learned"] else ""
    rows = []
    for a in arms:
        dens, _ = np.histogram(pooled[a].ravel(), bins=edges, density=True)
        for i in range(40):
            rows.append({"panel": "hist", "arm": a, "regime": "all", "bin_left": edges[i],
                         "bin_right": edges[i + 1], "density": dens[i], "n_windows": n_all,
                         "seed": seedtxt(a)})
    for r in h["summary"]:
        rows.append({"panel": "cvar", "arm": r["arm"], "regime": r["regime"], "cvar95": r["cvar95"],
                     "cvar_lo": r["cvar_lo"], "cvar_hi": r["cvar_hi"], "delta_vs_bs": r["delta_vs_bs"],
                     "p_holm": r["p_holm"], "n_windows": r["n_windows"], "seed": seedtxt(r["arm"])})
    df = pd.DataFrame(rows)
    df["cost_bp"] = COST_BP
    df["run_id"], df["config_hash"], df["data_sha256"] = run_id, chash, dsha
    cols = ["panel", "arm", "regime", "bin_left", "bin_right", "density", "cvar95", "cvar_lo", "cvar_hi",
            "delta_vs_bs", "p_holm", "n_windows", "cost_bp", "run_id", "seed", "config_hash",
            "data_sha256"]
    df = df[cols]
    df.to_csv(fig45_csv, index=False)

    nl, nh = int((regs == "low").sum()), int((regs == "high").sum())
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="white")
    mids = 0.5 * (edges[:-1] + edges[1:])
    for a in arms:
        d = df[(df.panel == "hist") & (df.arm == a)]
        ax[0].step(mids, d.density.to_numpy(), where="mid", color=ARM_COLOR[a], label=a, lw=1.4)
    ax[0].set_xlabel("hedging loss L (notional units, S0 = 1)")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) loss histograms, all windows", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(color="#dddddd", lw=0.5)
    w = 0.13
    for ri, reg in enumerate(("low", "high")):
        for ai, a in enumerate(arms):
            r = df[(df.panel == "cvar") & (df.arm == a) & (df.regime == reg)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = ri + (ai - 2.5) * w
            ax[1].errorbar(x, r.cvar95, yerr=[[r.cvar95 - r.cvar_lo], [r.cvar_hi - r.cvar95]], fmt="o",
                           color=ARM_COLOR[a], capsize=3, label=a if ri == 0 else None)
            if r.n_windows < MIN_WINDOWS:
                ax[1].annotate("!", (x, r.cvar_hi), color="red", fontsize=14, ha="center",
                               fontweight="bold")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([f"low VIX (<20)\nn={nl}" + (" WARNING n<10" if nl < MIN_WINDOWS else ""),
                           f"high VIX (>=20)\nn={nh}" + (" WARNING n<10" if nh < MIN_WINDOWS else "")])
    ax[1].set_ylabel("CVaR95")
    ax[1].set_title("(b) CVaR95, block-bootstrap 95% interval", fontsize=10)
    ax[1].grid(color="#dddddd", lw=0.5)
    ax[1].legend(frameon=False, fontsize=7)
    others = [a for a in arms if a != "bs_delta_vix"]
    regl = ("all", "low", "high")
    for ri, reg in enumerate(regl):
        for ai, a in enumerate(others):
            r = df[(df.panel == "cvar") & (df.arm == a) & (df.regime == reg)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = ri + (ai - 2) * 0.15
            ax[2].plot(x, r.delta_vs_bs, "o", color=ARM_COLOR[a], label=a if ri == 0 else None)
            ptxt = "p_Holm=n/a" if pd.isna(r.p_holm) else f"p_Holm={r.p_holm:.2f}"
            ax[2].annotate(ptxt, (x, r.delta_vs_bs), xytext=(0, 6), textcoords="offset points",
                           rotation=90, fontsize=6.5, ha="center")
            if r.n_windows < MIN_WINDOWS:
                ax[2].annotate("!", (x, r.delta_vs_bs), color="red", fontsize=12, ha="center")
    ax[2].margins(y=0.3)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels([f"{r}\nn={int((regs == r).sum()) if r != 'all' else n_all}" for r in regl])
    ax[2].set_ylabel("CVaR95(arm) - CVaR95(bs_delta_vix)")
    ax[2].set_title("(c) paired difference to bs_delta_vix, Holm p", fontsize=10)
    ax[2].grid(color="#dddddd", lw=0.5)
    ax[2].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{tag}Fig 4.5  Real monthly S&P 500 windows, 10 bp per side, n_low={nl}, n_high={nh}; "
                 f"learned hedgers trained on simulated paths only", fontsize=11)
    fig.text(0.5, 0.005, "How to read this chart: each hedge was run on real one-month windows; lower and "
             "further left is better, bars show 95% bootstrap intervals.\nPanels split calm (VIX below 20) "
             "and stressed windows; learned hedgers were trained on simulated paths only. "
             f"Data: figures/{fig45_csv.name}.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(fig45_png, dpi=130)
    plt.close(fig)

    # ================= fig-4.6 =================
    j = json.loads(intraday_json.read_text())
    prov = j["provenance"]
    run_id, chash, dsha = _ids(prov)
    tag = "[QUICK PIPELINE CHECK, not final] " if prov["quick"] else ""
    rows = []
    for r in j["summary"]:
        rows.append({"arm": r["arm"], "rehedges_per_day": r["rehedges_per_day"], "cvar95": r["cvar95"],
                     "cvar_lo": r["cvar_lo"], "cvar_hi": r["cvar_hi"], "mean_turnover": r["mean_turnover"],
                     "mean_cost": r["mean_cost"], "cost_bp": COST_BP, "n_windows": r["n_windows"],
                     "saturated": r["saturated"], "run_id": run_id,
                     "seed": seedrange if r["arm"] == "gru" else "",
                     "config_hash": chash, "data_sha256": dsha})
    d6 = pd.DataFrame(rows)
    d6.to_csv(fig46_csv, index=False)
    fr = []
    for f in j["failure_cases"]:
        for arm, p in f["pnl_path"].items():
            for x, y in zip(f["x_days"], p):
                fr.append({"window": f["window"], "start_date": f["start_date"], "end_date": f["end_date"],
                           "arm": arm, "rehedges_per_day": f["rehedges_per_day"], "day": x, "cum_pnl": y,
                           "run_id": run_id, "seed": seedrange if arm == "gru" else "",
                           "config_hash": chash, "data_sha256": dsha})
    pd.DataFrame(fr).to_csv(failure_csv, index=False)

    fig = plt.figure(figsize=(15, 9.5), facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    axa = fig.add_subplot(gs[0, 0:3])
    axb = fig.add_subplot(gs[0, 3:6])
    nwin = int(d6.n_windows.iloc[0])
    kk = sorted(d6.rehedges_per_day.unique())
    for a in INTRA_ARMS:
        d = d6[d6.arm == a].sort_values("rehedges_per_day")
        x = np.log2(d.rehedges_per_day.to_numpy())
        axa.errorbar(x, d.cvar95, yerr=[d.cvar95 - d.cvar_lo, d.cvar_hi - d.cvar95], marker="o", capsize=3,
                     color=ARM_COLOR[a], label=a)
        sat = d[d.saturated == 1]
        if len(sat):
            s0 = sat.iloc[0]
            axa.plot(np.log2(s0.rehedges_per_day), s0.cvar95, marker="*", ms=16, mfc="none",
                     color=ARM_COLOR[a], mew=1.5)
            axa.annotate("saturates", (np.log2(s0.rehedges_per_day), s0.cvar95), xytext=(4, 8),
                         textcoords="offset points", fontsize=7, color=ARM_COLOR[a])
    axa.set_xticks(np.log2(kk))
    axa.set_xticklabels([str(k) for k in kk])
    axa.set_xlabel("rehedges per day (log2)")
    axa.set_ylabel("CVaR95 (notional units)")
    axa.set_title("(a) CVaR95 vs rehedging frequency (star = first saturated step)", fontsize=10)
    axa.legend(frameon=False, fontsize=8)
    axa.grid(color="#dddddd", lw=0.5)
    axb2 = axb.twinx()
    for a in INTRA_ARMS:
        d = d6[d6.arm == a].sort_values("rehedges_per_day")
        x = np.log2(d.rehedges_per_day.to_numpy())
        axb.plot(x, d.mean_turnover, "-o", color=ARM_COLOR[a], label=f"{a} turnover")
        axb2.plot(x, d.mean_cost, "--s", color=ARM_COLOR[a], alpha=0.7)
    axb.set_xticks(np.log2(kk))
    axb.set_xticklabels([str(k) for k in kk])
    axb.set_xlabel("rehedges per day (log2)")
    axb.set_ylabel("mean turnover (shares), solid")
    axb2.set_ylabel("mean cost (notional units), dashed")
    axb.set_title("(b) turnover and cost", fontsize=10)
    axb.legend(frameon=False, fontsize=7)
    axb.grid(color="#dddddd", lw=0.5)
    for c, f in enumerate(j["failure_cases"][:3]):
        axc = fig.add_subplot(gs[1, 2 * c:2 * c + 2])
        for a in INTRA_ARMS:
            axc.plot(f["x_days"], f["pnl_path"][a], color=ARM_COLOR[a], label=a)
        axc.axhline(0, color="k", lw=0.6)
        axc.set_title(f"(c{c + 1}) {f['start_date']} to {f['end_date']}  k={f['rehedges_per_day']}\n"
                      f"bs_delta_vix loss {f['loss_bs_delta_vix']:.4f}", fontsize=9)
        axc.set_xlabel("days in window")
        if c == 0:
            axc.set_ylabel("cumulative P&L (notional units)")
            axc.legend(frameon=False, fontsize=7)
        axc.grid(color="#dddddd", lw=0.5)
    fig.suptitle(f"{tag}Fig 4.6  Hourly bars, 10 bp per side, 5-day windows (n={nwin}); gru trained on "
                 f"simulated paths only", fontsize=11)
    fig.text(0.5, 0.005, "How to read this chart: moving right means hedging more often; the curve flattens "
             "or turns up once trading costs outweigh the tighter hedge.\nThe lower panel shows the three "
             f"worst real windows. Data: figures/{fig46_csv.name}.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(fig46_png, dpi=130)
    plt.close(fig)
    print(f"figures written: {fig45_png.name}, {fig46_png.name}, {failure_csv.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--quick", action="store_true")
    t.add_argument("--run-id", required=True, help="new immutable result identifier")
    t.add_argument("--workers", type=int, default=1, help="independent process workers (never threads)")
    r = sub.add_parser("replay")
    r.add_argument("--run-id", required=True, help="existing immutable result identifier")
    f = sub.add_parser("figures")
    f.add_argument("--run-id", required=True, help="existing immutable result identifier")
    a = ap.parse_args()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "train":
        configure_run(a.run_id, create=True)
        cmd_train(a.quick, a.workers)
    elif a.cmd == "replay":
        configure_run(a.run_id)
        cmd_replay()
    else:
        configure_run(a.run_id)
        cmd_figures()


if __name__ == "__main__":
    main()
