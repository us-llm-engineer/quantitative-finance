"""In-training monitor for train_recipe (contract: plans/round-03/addendum-A3-training-monitor.md).

Per-step quantities stay on the device; everything is read back once per epoch.
"""
from __future__ import annotations

import base64
import hashlib
import math
import time

import numpy as np
import torch

from rough_hedge import training, risk, hedging

PNL_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def bs_delta_torch(S, times, strike, sigma, T):
    """Black-Scholes call delta at t_0..t_{N-1} for paths S (B, N+1), r = 0."""
    tau = (T - times[:-1]).clamp_min(1e-12)                       # (N,)
    d1 = (torch.log(S[:, :-1] / strike) + 0.5 * sigma ** 2 * tau) / (sigma * torch.sqrt(tau))
    return 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))


class EpochMonitor:
    def __init__(self, model, v, probe_S, times, strike, cost, sigma, T, alpha,
                 batch_size, epochs, enabled=True):
        self.model, self.v, self.enabled = model, v, enabled
        self.probe_S, self.times = probe_S, times
        self.strike, self.cost, self.sigma, self.T, self.alpha = strike, cost, sigma, T, alpha
        self.batch_size = batch_size
        self.probe_sha256 = hashlib.sha256(probe_S.detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest()
        last = max(epochs - 1, 0)
        self.snapshot_epochs = sorted({0, round(0.25 * last), round(0.5 * last), round(0.75 * last), last})
        # group parameters by owning module path, e.g. "net.0", "net.2", "gru"
        self.groups = {}
        for name, p in model.named_parameters():
            self.groups.setdefault(name.rsplit(".", 1)[0] if "." in name else name, []).append(p)
        self.layers = list(self.groups)
        with torch.no_grad():
            self.delta_bs = bs_delta_torch(probe_S, times, strike, sigma, T)
        self.epochs, self.snapshots = [], {}

    # ---------------------------------------------------------------- per epoch
    def start_epoch(self):
        if not self.enabled:
            return
        self.t0_unix = time.time()
        self.theta0 = _flat_params(self.model)
        dev = self.theta0.device
        self.n_steps = 0
        self.sum_g = torch.zeros_like(self.theta0)
        self.sum_g2 = torch.zeros((), device=dev, dtype=self.theta0.dtype)   # sum of |g|^2
        self.norms = []                                                        # device scalars
        self.layer_sq = torch.zeros(len(self.layers), device=dev, dtype=self.theta0.dtype)

    def after_backward(self):
        """Call between loss.backward() and opt.step(). No host sync."""
        if not self.enabled:
            return
        with torch.no_grad():
            parts, layer_sq = [], []
            for params in self.groups.values():
                layer_sq.append(sum((p.grad * p.grad).sum() if p.grad is not None
                                    else self.sum_g2 * 0 for p in params))
            for p in self.model.parameters():
                parts.append((p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1))
            g = torch.cat(parts)
            n2 = (g * g).sum()
            self.sum_g += g
            self.sum_g2 += n2
            self.norms.append(torch.sqrt(n2))
            self.layer_sq += torch.stack(layer_sq)
            self.n_steps += 1

    def end_epoch(self, epoch, lr, wall_s, ms_per_step, peak_memory, train_cvar, val_cvar):
        if not self.enabled:
            return None
        with torch.no_grad():
            norms = torch.stack(self.norms).double().cpu().numpy()
            S_steps = self.n_steps
            mean_g = self.sum_g / S_steps
            G2 = float((mean_g * mean_g).sum())
            Eg2 = float(self.sum_g2) / S_steps
            b_simple = (self.batch_size * (Eg2 - G2) / G2) if (S_steps >= 2 and G2 > 0) else None
            theta1 = _flat_params(self.model)
            upd_ratio = float((theta1 - self.theta0).norm() / self.theta0.norm().clamp_min(1e-12))
            layer_share = (self.layer_sq / self.layer_sq.sum().clamp_min(1e-30)).cpu().numpy()

            logm = torch.log(self.probe_S / self.probe_S[:, :1])
            pos = self.model(logm, self.times)
            losses_t = training.hedging_loss_torch(self.probe_S, pos, self.strike, self.cost)
            dist = (pos - self.delta_bs).abs()
            losses = losses_t.double().cpu().numpy()
            pos_np = pos.double().cpu().numpy()
            dist_all, dist_t0 = float(dist.mean()), float(dist[:, 0].mean())
            v_now = float(self.v)

        pnl = -losses
        rec = {
            "epoch": epoch, "t_start_unix": self.t0_unix, "t_end_unix": time.time(),
            "lr": lr, "wall_s": wall_s, "ms_per_step": ms_per_step,
            "paths_per_s": (self.batch_size * S_steps / wall_s) if wall_s > 0 else None,
            "peak_memory_bytes": int(peak_memory), "steps": S_steps,
            "train_cvar95_pooled": train_cvar, "val_cvar95": val_cvar,
            "grad_norm_mean": float(norms.mean()), "grad_norm_sd": float(norms.std()),
            "grad_norm_max": float(norms.max()),
            "grad_noise_scale_B_simple": b_simple,
            "update_to_weight_ratio": upd_ratio,
            "layer_grad_share": {n: float(s) for n, s in zip(self.layers, layer_share)},
            "probe_cvar90": float(risk.cvar_tail_mean(losses, 0.90)),
            "probe_cvar95": float(risk.cvar_tail_mean(losses, 0.95)),
            "probe_cvar99": float(risk.cvar_tail_mean(losses, 0.99)),
            "probe_var95": float(np.quantile(losses, 0.95)),
            "probe_mean_loss": float(losses.mean()), "probe_sd_loss": float(losses.std()),
            "ru_threshold_v": v_now,
            "delta_dist_mean": dist_all, "delta_dist_t0": dist_t0,
            "turnover": float(hedging.turnover(pos_np).mean()),
            "pnl_quantiles": {str(q): float(np.quantile(pnl, q)) for q in PNL_QUANTILES},
        }
        self.epochs.append(rec)
        if epoch in self.snapshot_epochs:
            self.snapshots[str(epoch)] = base64.b64encode(losses.astype(np.float16).tobytes()).decode("ascii")
        return rec

    def finish(self, last_epoch):
        """Guarantee a final snapshot when early stopping cut the schedule."""
        if self.enabled and self.epochs and str(last_epoch) not in self.snapshots:
            logm = torch.log(self.probe_S / self.probe_S[:, :1])
            with torch.no_grad():
                pos = self.model(logm, self.times)
                losses = training.hedging_loss_torch(self.probe_S, pos, self.strike, self.cost)
            self.snapshots[str(last_epoch)] = base64.b64encode(
                losses.double().cpu().numpy().astype(np.float16).tobytes()).decode("ascii")
        return {"probe_sha256": self.probe_sha256, "n_probe": int(self.probe_S.shape[0]),
                "snapshot_dtype": "float16", "epochs": self.epochs, "snapshots": self.snapshots}
