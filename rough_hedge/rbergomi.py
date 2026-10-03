"""Rough Bergomi simulation with the hybrid scheme (kappa = 0 or 1) and exact-covariance references.

Convention (one fixed notation for the whole package):
    H  Hurst exponent, a = H - 1/2 in (-1/2, 0)
    W~_t = sqrt(2H) * int_0^t (t-u)^a dW_u      Riemann-Liouville (Volterra) process, Var(W~_t) = t^(2H)
    V_t  = xi0 * exp( eta * W~_t - eta^2/2 * t^(2H) )           spot variance (flat forward variance xi0)
    dS_t = S_t sqrt(V_t) ( rho dW_t + sqrt(1-rho^2) dW'_t )     risk-neutral, zero rate
so eta here equals the vol-of-vol nu of the form  V_t = xi0 * E( sqrt(2H) nu int (t-u)^a dW_u ).
"""
from __future__ import annotations
import math
import numpy as np


def b_star(n_steps: int, a: float) -> np.ndarray:
    """Optimal evaluation points b_k* (k = 1..n_steps) (Proposition 2.8).
    
    Element k-1 holds b_k* = ((k^(a+1) - (k-1)^(a+1)) / (a+1))^(1/a).
    """
    k = np.arange(1, n_steps + 1, dtype=np.float64)
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)


def kernel_weights(n_steps: int, a: float, kappa: int) -> np.ndarray:
    """Unit-horizon weights g[j] multiplying dW_{i-j} in W~_i (j = 0..n_steps-1).
    
    kappa = 1: g[0]=0 (exact Wiener integral), j >= 1 uses (b_{j+1}*)^a / n^a.
    kappa = 0: plain right-endpoint Riemann sum, weight ((j+1)/n)^a for j >= 0.
    """
    n = n_steps
    g = np.zeros(n, dtype=np.float64)
    if kappa == 0:
        g[:] = ((np.arange(n, dtype=np.float64) + 1.0) / n) ** a
    elif kappa == 1:
        b = b_star(n, a)
        g[1:] = (b[1:] / n) ** a
    else:
        raise ValueError(f"kappa must be 0 or 1, got {kappa}")
    return g


def _physical_weight(lag: int, n: int, a: float, T: float, kappa: int) -> float:
    """Weight multiplying dW_{i-lag} in W~_i for physical time with horizon T."""
    h = T / n
    if kappa == 1:
        if lag == 0:
            return 0.0
        k = lag + 1
        b = ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)
        return (b * h) ** a
    return ((lag + 1) * h) ** a


def simulate_volterra(n_paths, n_steps, T, H, kappa=1, rng=None, noise=None, 
                      return_noise=False, antithetic=False):
    """Simulate the Volterra process W~ using the hybrid scheme.
    
    Args:
        n_paths: number of sample paths
        n_steps: number of time steps
        T: time horizon
        H: Hurst exponent in (0, 1/2)
        kappa: 0 for forward sum, 1 for hybrid scheme
        rng: numpy.random.Generator (default: numpy's default)
        noise: dict with "dW" (n_paths, n_steps) and optionally "Y" (for kappa=1)
               These should have physical covariance (variance = T/n)
        return_noise: if True, return (W, noise_dict)
        antithetic: if True, paths n_paths/2..n_paths-1 are negatives of 0..n_paths/2-1
    
    Returns:
        W: ndarray (n_paths, n_steps+1) with W[:, 0] = 0
        or (W, noise_dict) if return_noise=True
    """
    # Validation
    if not (0 < H < 0.5):
        raise ValueError(f"H must be in (0, 0.5), got {H}")
    if kappa not in {0, 1}:
        raise ValueError(f"kappa must be 0 or 1, got {kappa}")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"antithetic requires even n_paths, got {n_paths}")
    
    if rng is None:
        rng = np.random.default_rng()
    
    a = H - 0.5
    n = n_steps
    h = T / n
    c = math.sqrt(2 * H)
    
    # Generate or validate noise
    if noise is None:
        # Generate Cholesky factor for (dW_i, Y_i) covariance with physical variance
        s12 = h ** (a + 1) / (a + 1)
        cov_block = np.array([[h, s12], [s12, h ** (2 * a + 1) / (2 * a + 1)]], dtype=np.float64)
        L = np.linalg.cholesky(cov_block)
        z = rng.standard_normal((n_paths, n, 2), dtype=np.float64)
        noise_gen = z @ L.T
        dW = noise_gen[..., 0]
        Y = noise_gen[..., 1]
    else:
        dW = noise["dW"].astype(np.float64).copy()
        Y = noise.get("Y", np.zeros((n_paths, n), dtype=np.float64)).astype(np.float64).copy()
        if dW.shape != (n_paths, n):
            raise ValueError(f"noise['dW'] must have shape ({n_paths}, {n}), got {dW.shape}")
        if kappa == 1 and "Y" in noise and noise["Y"].shape != (n_paths, n):
            raise ValueError(f"noise['Y'] must have shape ({n_paths}, {n}), got {noise['Y'].shape}")
    
    # Antithetic pairs
    if antithetic:
        n_half = n_paths // 2
        dW[n_half:] = -dW[:n_half]
        Y[n_half:] = -Y[:n_half]
    
    # Compute physical weights and apply FFT convolution
    # W~_i = c * (sum_{j=1}^{i} phys_weight(i-j, ...) * dW_{j} + kappa * Y_i)
    #      = c * (sum_{lag=0}^{i-1} phys_weight(lag, ...) * dW_{i-lag})  [1-indexed]
    #      = c * (sum_{lag=0}^{i-1} phys_weight(lag, ...) * dW[i-1-lag])  [0-indexed]
    # Compute weights for all lags
    weights = np.array([_physical_weight(lag, n, a, T, kappa) for lag in range(n)], dtype=np.float64)
    
    # FFT-based linear convolution
    m = 1 << (2 * n - 1).bit_length()
    dW_fft = np.fft.rfft(dW, n=m, axis=1)
    weights_fft = np.fft.rfft(weights, n=m)
    conv = np.fft.irfft(dW_fft * weights_fft[None, :], n=m, axis=1)[:, :n]
    
    # Build W~: W~[:, i] = c * (conv[:, i-1] + Y[:, i-1]) for i=1..n
    W = np.zeros((n_paths, n + 1), dtype=np.float64)
    W[:, 1:] = c * (conv + (Y if kappa == 1 else 0))
    
    # Return based on return_noise
    if return_noise:
        # Always return Y even for kappa=0 so the noise dict has consistent shape
        return_dict = {"dW": dW, "Y": Y}
        return W, return_dict
    return W


def simulate_rbergomi(n_paths, n_steps, T, H, eta, rho, xi0, kappa=1, rng=None, antithetic=False):
    """Simulate rough Bergomi paths (S, V, W~).
    
    Args:
        n_paths: number of sample paths
        n_steps: number of time steps
        T: time horizon
        H: Hurst exponent in (0, 1/2)
        eta: vol-of-vol parameter (>= 0)
        rho: correlation in [-1, 1]
        xi0: initial spot variance (> 0)
        kappa: 0 or 1 for the Volterra scheme
        rng: numpy.random.Generator
        antithetic: if True, use antithetic variance reduction
    
    Returns:
        dict with keys "W", "V", "S", "dW" (all ndarrays)
    """
    # Validation
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    if not (-1 <= rho <= 1):
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    if xi0 <= 0:
        raise ValueError(f"xi0 must be positive, got {xi0}")
    
    if rng is None:
        rng = np.random.default_rng()
    
    a = H - 0.5
    n = n_steps
    h = T / n
    
    # Simulate W~ with noise tracking
    W, noise_dict = simulate_volterra(n_paths, n, T, H, kappa=kappa, rng=rng, 
                                       return_noise=True, antithetic=antithetic)
    # dW is already in physical units (variance h = T/n)
    dW = noise_dict["dW"]
    
    # Spot variance at grid points
    t = np.arange(n + 1, dtype=np.float64) * T / n
    V = xi0 * np.exp(eta * W - 0.5 * eta ** 2 * t[None, :] ** (2 * H))
    
    # Generate independent Brownian increments for the price
    dB_indep = rng.standard_normal((n_paths, n), dtype=np.float64) * math.sqrt(h)
    dB = rho * dW + math.sqrt(1 - rho ** 2) * dB_indep
    
    # Log-spot increments and cumsum
    dlogS = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * h
    S = np.zeros((n_paths, n + 1), dtype=np.float64)
    S[:, 0] = 1.0
    S[:, 1:] = np.exp(np.cumsum(dlogS, axis=1))
    
    return {
        "W": W.astype(np.float64),
        "V": V.astype(np.float64),
        "S": S.astype(np.float64),
        "dW": dW.astype(np.float64),
    }


# Torch optional path
def simulate_volterra_torch(n_paths, n_steps, T, H, kappa=1, device="cpu", dtype=None, seed=0, chunk=None):
    """Simulate Volterra process using PyTorch (GPU-compatible).
    
    Args:
        n_paths: number of paths
        n_steps: number of time steps
        T: horizon
        H: Hurst exponent
        kappa: 0 or 1
        device: "cpu" or "cuda" (if available)
        dtype: torch.float64 or torch.float32 (default: torch.float64)
        seed: random seed
        chunk: process paths in chunks of this size (memory only)
    
    Returns:
        torch.Tensor of shape (n_paths, n_steps+1)
    """
    # Validation BEFORE any work (torch import, random number generation, device allocation)
    if not (0 < H < 0.5):
        raise ValueError(f"H must be in (0, 0.5), got {H}")
    if kappa not in {0, 1}:
        raise ValueError(f"kappa must be 0 or 1, got {kappa}")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if chunk is not None and chunk < 1:
        raise ValueError(f"chunk must be None or >= 1, got {chunk}")
    
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for simulate_volterra_torch")
    
    if dtype is None:
        dtype = torch.float64
    
    a = H - 0.5
    n = n_steps
    h = T / n
    c = math.sqrt(2 * H)
    
    # Seed generator
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Cholesky factor for (dW_i, Y_i)
    s12 = h ** (a + 1) / (a + 1)
    cov_block = torch.tensor([[h, s12], [s12, h ** (2 * a + 1) / (2 * a + 1)]], 
                               dtype=dtype, device=device)
    L = torch.linalg.cholesky(cov_block)
    
    # Compute physical weights
    weights_np = np.array([_physical_weight(lag, n, a, T, kappa) for lag in range(n)], dtype=np.float64)
    weights = torch.tensor(weights_np, dtype=dtype, device=device)
    
    # Process in chunks if specified
    if chunk is None:
        chunk = n_paths
    
    W_all = []
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        batch_size = end - start
        
        # Generate noise
        z = torch.randn(batch_size, n, 2, dtype=dtype, device=device, generator=gen)
        noise_gen = z @ L.T
        dW = noise_gen[..., 0]
        Y = noise_gen[..., 1]
        
        # FFT convolution
        m = 1 << (2 * n - 1).bit_length()
        dW_fft = torch.fft.rfft(dW, n=m, dim=1)
        weights_fft = torch.fft.rfft(weights, n=m)
        conv = torch.fft.irfft(dW_fft * weights_fft[None, :], n=m, dim=1)[:, :n]
        
        # Build W~
        W_batch = torch.zeros(batch_size, n + 1, dtype=dtype, device=device)
        W_batch[:, 1:] = c * (conv + (Y if kappa == 1 else 0))
        W_all.append(W_batch)
    
    return torch.cat(W_all, dim=0)
