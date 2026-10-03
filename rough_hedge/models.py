"""Torch hedging models with neural network architectures (API: tests/api/R2_1.md).

Three causal neural network architectures for position control in hedging:
- FFNHedger: feedforward network with previous position feedback
- GRUHedger: recurrent GRU cell with memory
- SignatureHedger: signature features from lead-lag path as additional input
"""
from __future__ import annotations

import numbers
import numpy as np
import torch
import torch.nn as nn
from rough_hedge.signature import lead_lag as _lead_lag, truncated_signature


def _batched_prefix_signature_features(x: torch.Tensor, depth: int) -> torch.Tensor:
    """Return lead-lag prefix signatures entirely on the current torch device."""
    B, n, d = x.shape
    work = x.to(dtype=torch.float64)
    ll = torch.empty((B, 2 * n - 1, 2 * d), dtype=work.dtype, device=work.device)
    ll[:, 0::2, :d] = work
    ll[:, 0::2, d:] = work
    ll[:, 1::2, :d] = work[:, 1:]
    ll[:, 1::2, d:] = work[:, :-1]
    deltas = ll[:, 1:] - ll[:, :-1]

    levels = [torch.zeros((B, (2 * d) ** k), dtype=work.dtype, device=work.device)
              for k in range(1, depth + 1)]
    out = torch.zeros((B, n, sum((2 * d) ** k for k in range(1, depth + 1))),
                      dtype=work.dtype, device=work.device)
    one = torch.ones((B, 1), dtype=work.dtype, device=work.device)
    for segment, delta in enumerate(deltas.unbind(dim=1)):
        segment_levels = []
        current = one
        for k in range(1, depth + 1):
            current = (current.unsqueeze(-1) * delta.unsqueeze(-2)).reshape(B, -1) / k
            segment_levels.append(current)
        left, right = [one] + levels, [one] + segment_levels
        levels = [sum((left[i].unsqueeze(-1) * right[k - i].unsqueeze(-2)).reshape(B, -1)
                      for i in range(k + 1)) for k in range(1, depth + 1)]
        if segment % 2 == 1:
            out[:, (segment + 1) // 2] = torch.cat(levels, dim=-1)
    return out


class FFNHedger(nn.Module):
    """Feedforward hedger: input (t/T, logm_j, y_{j-1}), net = Lin(3,h)-ReLU-[Lin(h,h)-ReLU]^L-Lin(h,1)."""

    def __init__(self, hidden: int = 64, layers: int = 2, seed: int | None = None, dtype=torch.float32):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, numbers.Integral) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if isinstance(layers, bool) or not isinstance(layers, numbers.Integral) or layers < 1:
            raise ValueError("layers must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")

        self.dtype = dtype
        self.h = hidden
        self.L = layers

        # Save global RNG state to restore later
        saved_rng = torch.get_rng_state()

        # Build network layers
        layers_list = [nn.Linear(3, hidden)]
        for _ in range(layers - 1):
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(hidden, hidden))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers_list)

        # Initialize with seed using private generator
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            with torch.no_grad():
                for p in self.net.parameters():
                    if p.dim() == 1:  # bias
                        p.normal_(0, 0.01, generator=g)
                    else:  # weight
                        p.normal_(0, 1.0 / np.sqrt(p.size(1)), generator=g)

        self.to(dtype=self.dtype)
        
        # Restore global RNG state
        if seed is not None:
            torch.set_rng_state(saved_rng)
        
        self.markov = True

    @property
    def n_params(self) -> int:
        """Total number of parameters: 4h + (L-1)(h^2+h) + (h+1)."""
        return 4 * self.h + (self.L - 1) * (self.h * self.h + self.h) + (self.h + 1)

    def forward(self, logm: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute hedging positions.
        
        Args:
            logm: (B, N+1) log price changes
            times: (N+1,) strictly increasing time points
            
        Returns:
            (B, N) positions y_0..y_{N-1}
        """
        logm = logm.to(dtype=self.dtype)
        times = times.to(dtype=self.dtype)
        
        # Validate inputs
        if logm.ndim != 2:
            raise ValueError("logm must be 2-D")
        B, n1 = logm.shape
        if n1 < 2:
            raise ValueError("N >= 1 required")
        if times.shape[0] != n1:
            raise ValueError("times length must match logm")
        if not torch.all(torch.isfinite(times)):
            raise ValueError("times must be finite")
        if not torch.all(times >= 0):
            raise ValueError("times must be non-negative")
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError("times must be strictly increasing")
        if B == 0:
            raise ValueError("batch cannot be empty")
        if not torch.all(torch.isfinite(logm)):
            raise ValueError("logm contains NaN or inf")

        N = n1 - 1
        t_scale = times[-1]
        positions = []
        prev = torch.zeros(B, dtype=self.dtype, device=logm.device)

        for j in range(N):
            t_norm = times[j] / t_scale
            inp = torch.stack([
                torch.full((B,), t_norm, dtype=self.dtype, device=logm.device),
                logm[:, j],
                prev
            ], dim=-1)
            prev = self.net(inp)[:, 0]
            positions.append(prev)

        return torch.stack(positions, dim=1)


class GRUHedger(nn.Module):
    """Recurrent hedger: GRUCell(3, hidden) taking (t/T, logm_j, y_{j-1}), head = Lin(hidden, 1)."""

    def __init__(self, hidden: int = 64, seed: int | None = None, dtype=torch.float32):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, numbers.Integral) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")

        self.dtype = dtype
        self.h = hidden

        # Save global RNG state
        saved_rng = torch.get_rng_state()

        self.cell = nn.GRUCell(3, hidden)
        self.head = nn.Linear(hidden, 1)

        # Initialize with seed using private generator
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            with torch.no_grad():
                for p in self.cell.parameters():
                    if p.dim() == 1:  # bias
                        p.normal_(0, 0.01, generator=g)
                    else:  # weight
                        p.normal_(0, 1.0 / np.sqrt(p.size(1)), generator=g)
                for p in self.head.parameters():
                    if p.dim() == 1:
                        p.normal_(0, 0.01, generator=g)
                    else:
                        p.normal_(0, 1.0 / np.sqrt(p.size(1)), generator=g)

        self.to(dtype=self.dtype)
        
        # Restore global RNG state
        if seed is not None:
            torch.set_rng_state(saved_rng)
        
        self.markov = False

    @property
    def n_params(self) -> int:
        """Total number of parameters: 3*h^2 + 15*h + h + 1."""
        return 3 * self.h * self.h + 15 * self.h + self.h + 1

    def forward(self, logm: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute hedging positions."""
        logm = logm.to(dtype=self.dtype)
        times = times.to(dtype=self.dtype)
        
        # Validate inputs
        if logm.ndim != 2:
            raise ValueError("logm must be 2-D")
        B, n1 = logm.shape
        if n1 < 2:
            raise ValueError("N >= 1 required")
        if times.shape[0] != n1:
            raise ValueError("times length must match logm")
        if not torch.all(torch.isfinite(times)):
            raise ValueError("times must be finite")
        if not torch.all(times >= 0):
            raise ValueError("times must be non-negative")
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError("times must be strictly increasing")
        if B == 0:
            raise ValueError("batch cannot be empty")
        if not torch.all(torch.isfinite(logm)):
            raise ValueError("logm contains NaN or inf")

        N = n1 - 1
        t_scale = times[-1]
        positions = []
        prev = torch.zeros(B, dtype=self.dtype, device=logm.device)
        h = torch.zeros(B, self.h, dtype=self.dtype, device=logm.device)

        for j in range(N):
            t_norm = times[j] / t_scale
            inp = torch.stack([
                torch.full((B,), t_norm, dtype=self.dtype, device=logm.device),
                logm[:, j],
                prev
            ], dim=-1)
            h = self.cell(inp, h)
            prev = self.head(h)[:, 0]
            positions.append(prev)

        return torch.stack(positions, dim=1)


class SignatureHedger(nn.Module):
    """Signature-based hedger: features [t/T, logm_j, y_{j-1}] ++ sig(lead_lag(X[:j+1]))."""

    def __init__(self, depth: int, hidden: int = 0, seed: int | None = None, dtype=torch.float32):
        super().__init__()
        if isinstance(depth, bool) or not isinstance(depth, numbers.Integral) or depth < 0 or depth > 3:
            raise ValueError("depth must be in {0, 1, 2, 3}")
        if isinstance(hidden, bool) or not isinstance(hidden, numbers.Integral) or hidden < 0:
            raise ValueError("hidden must be a non-negative integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")

        self.dtype = dtype
        self.depth = depth
        self.h = hidden

        # Compute n_features = 3 + sig_dim(4, depth)
        if depth == 0:
            self.n_features = 3
        else:
            self.n_features = 3 + sum(4 ** k for k in range(1, depth + 1))

        # Save global RNG state
        saved_rng = torch.get_rng_state()

        # Build head network
        if hidden == 0:
            self.head = nn.Linear(self.n_features, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

        # Initialize with seed using private generator
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            with torch.no_grad():
                for p in self.head.parameters():
                    if p.dim() == 1:  # bias
                        p.normal_(0, 0.01, generator=g)
                    else:  # weight
                        p.normal_(0, 1.0 / np.sqrt(p.size(1)), generator=g)

        self.to(dtype=self.dtype)
        
        # Restore global RNG state
        if seed is not None:
            torch.set_rng_state(saved_rng)
        
        self.markov = depth <= 1

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        if self.h == 0:
            return self.n_features + 1
        else:
            return self.n_features * self.h + self.h + self.h + 1

    def forward(self, logm: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute hedging positions."""
        logm = logm.to(dtype=self.dtype)
        times = times.to(dtype=self.dtype)
        
        # Validate inputs
        if logm.ndim != 2:
            raise ValueError("logm must be 2-D")
        B, n1 = logm.shape
        if n1 < 2:
            raise ValueError("N >= 1 required")
        if times.shape[0] != n1:
            raise ValueError("times length must match logm")
        if not torch.all(torch.isfinite(times)):
            raise ValueError("times must be finite")
        if not torch.all(times >= 0):
            raise ValueError("times must be non-negative")
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError("times must be strictly increasing")
        if B == 0:
            raise ValueError("batch cannot be empty")
        if not torch.all(torch.isfinite(logm)):
            raise ValueError("logm contains NaN or inf")

        N = n1 - 1
        t_scale = times[-1]
        positions = []
        prev = torch.zeros(B, dtype=self.dtype, device=logm.device)

        # Build the path X = (t/T, logm)
        t_path = (times / t_scale).unsqueeze(0).expand(B, -1)
        x = torch.stack([t_path, logm], dim=-1)  # (B, N+1, 2)

        # Precompute signatures if needed
        sigs = None
        if self.depth > 0:
            with torch.no_grad():
                sigs = _batched_prefix_signature_features(x, self.depth).to(dtype=self.dtype)

        for j in range(N):
            t_norm = times[j] / t_scale
            markov_inp = torch.stack([
                torch.full((B,), t_norm, dtype=self.dtype, device=logm.device),
                logm[:, j],
                prev
            ], dim=-1)

            if self.depth == 0:
                inp = markov_inp
            else:
                inp = torch.cat([markov_inp, sigs[:, j]], dim=-1)

            prev = self.head(inp)[:, 0]
            positions.append(prev)

        return torch.stack(positions, dim=1)
