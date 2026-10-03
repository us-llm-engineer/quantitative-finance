"""Roughness exponent estimation from log-volatility or log-spot series.

Functions for computing scaled absolute moments m(q, D) and recovering the Hurst exponent H
via regression of zeta_q on q, where m(q, D) = (1/N) sum |x[kD] - x[(k-1)D]|^q.
"""
from __future__ import annotations
import numpy as np


def moment_m(x, q: float, delta: int) -> float | np.ndarray:
    """Scaled absolute moment m(q, D).
    
    m(q, D) = (1/N) sum_{k=1}^{N} |x[kD] - x[(k-1)D]|^q  where N = floor((L-1)/D)
    
    Args:
        x: series of shape (L,) or (n_series, L)
        q: moment order (must be > 0)
        delta: lag (must be >= 1)
    
    Returns:
        float or ndarray (n_series,) depending on input shape
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if delta < 1:
        raise ValueError(f"delta must be >= 1, got {delta}")
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        L = len(x)
        N = (L - 1) // delta
        if N < 2:
            raise ValueError(f"Need at least {2*delta + 1} points for at least 2 observations, got {L}")
        tot = 0.0
        for k in range(1, N + 1):
            tot += np.abs(x[k * delta] - x[(k - 1) * delta]) ** q
        return float(tot / N)
    elif x.ndim == 2:
        n_series, L = x.shape
        N = (L - 1) // delta
        if N < 2:
            raise ValueError(f"Need at least {2*delta + 1} points per series for at least 2 observations, got {L}")
        result = np.zeros(n_series, dtype=np.float64)
        for i in range(n_series):
            tot = 0.0
            for k in range(1, N + 1):
                tot += np.abs(x[i, k * delta] - x[i, (k - 1) * delta]) ** q
            result[i] = tot / N
        return result
    else:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")


def zeta_exponents(x, qs=(0.5, 1.0, 1.5, 2.0, 3.0), lags=range(1, 31)) -> np.ndarray:
    """Exponents zeta_q from regression of log m(q, D) on log D.
    
    For each q, regresses log m(q, D) against log D using OLS (no intercept),
    yielding the slope zeta_q.
    
    Args:
        x: series of shape (L,) or (n_series, L)
        qs: sequence of moment orders
        lags: sequence of lag values
    
    Returns:
        ndarray of shape (len(qs),) or (n_series, len(qs))
    """
    x = np.asarray(x, dtype=np.float64)
    lags = np.asarray(lags, dtype=int)
    qs = np.asarray(qs, dtype=np.float64)
    
    log_lags = np.log(lags.astype(np.float64))
    
    if x.ndim == 1:
        zeta = np.zeros(len(qs), dtype=np.float64)
        for i, q in enumerate(qs):
            log_moments = np.log([moment_m(x, q, d) for d in lags])
            zeta[i] = np.polyfit(log_lags, log_moments, 1)[0]
        return zeta
    elif x.ndim == 2:
        n_series = x.shape[0]
        zeta = np.zeros((n_series, len(qs)), dtype=np.float64)
        for i in range(n_series):
            for j, q in enumerate(qs):
                log_moments = np.log([moment_m(x[i], q, d) for d in lags])
                zeta[i, j] = np.polyfit(log_lags, log_moments, 1)[0]
        return zeta
    else:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")


def estimate_hurst(x, qs=(0.5, 1.0, 1.5, 2.0, 3.0), lags=range(1, 31)) -> tuple:
    """Estimate Hurst exponent from zeta exponents via linear regression.
    
    Recovers H_hat as the OLS slope (with intercept) of zeta_q on q.
    
    Args:
        x: series of shape (L,) or (n_series, L)
        qs: moment orders
        lags: lag values
    
    Returns:
        (H_hat, zeta) where:
            H_hat: float or ndarray (n_series,)
            zeta: ndarray of zeta_q values
    """
    x = np.asarray(x, dtype=np.float64)
    qs_arr = np.asarray(qs, dtype=np.float64)
    
    zeta = zeta_exponents(x, qs=qs, lags=lags)
    
    if zeta.ndim == 1:
        # Single series: H_hat is float
        H_hat = float(np.polyfit(qs_arr, zeta, 1)[0])
        return H_hat, zeta
    else:
        # Multiple series: H_hat is ndarray
        n_series = zeta.shape[0]
        H_hat = np.zeros(n_series, dtype=np.float64)
        for i in range(n_series):
            H_hat[i] = np.polyfit(qs_arr, zeta[i], 1)[0]
        return H_hat, zeta


def integrated_variance_proxy(V, steps_per_obs: int, window: int) -> np.ndarray:
    """Proxy for integrated variance.
    
    Computes integrated variance averaged over a window of fine-grid steps.
    Entry k-1 of the result is the mean of V[..., k*s-window : k*s] 
    (left points of the window finest steps before observation k).
    
    Args:
        V: array (..., n_fine) on a fine grid
        steps_per_obs: s, the number of fine steps per observation
        window: width of the averaging window (must be in [1, s])
    
    Returns:
        ndarray (..., n_obs) where n_obs = n_fine // s
    """
    V = np.asarray(V, dtype=np.float64)
    s = steps_per_obs
    n_fine = V.shape[-1]
    n_obs = n_fine // s
    
    if window < 1 or window > s:
        raise ValueError(f"window must be in [1, {s}], got {window}")
    
    result = np.zeros(V.shape[:-1] + (n_obs,), dtype=np.float64)
    for k in range(1, n_obs + 1):
        start = k * s - window
        end = k * s
        result[..., k - 1] = np.mean(V[..., start:end], axis=-1)
    
    return result
