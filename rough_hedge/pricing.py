"""Pricing helpers and variance-reduced Monte Carlo estimators for rough Bergomi.

Conventions: zero rate, S0 = 1, K = e^k (log-strike), T maturity in years, 
H = a + 1/2 in (0, 1/2), sigma vol (not variance), w = sigma^2 T total variance.
rBergomi as in rbergomi.py: V_t = xi0 exp(eta W~_t - eta^2/2 t^(2H)).
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Literal

import numpy as np
from scipy.special import ndtri, erfc
from scipy.optimize import brentq

# Named tuples for return types
Estimate = namedtuple('Estimate', ['price', 'stderr', 'alpha'])
RuntimeAdjusted = namedtuple('RuntimeAdjusted', ['phi2', 'psi2'])


# ==============================================================================
# Black-Scholes helpers
# ==============================================================================

def _phi(x):
    """Standard normal CDF using scipy.special.ndtri."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _N(x):
    """Standard normal CDF."""
    return ndtri(x) if np.isscalar(x) else np.array([ndtri(float(xi)) for xi in np.asarray(x)])


def bs_call(S, K, sigma, T):
    """Black-Scholes call price.
    
    Args:
        S, K, sigma, T: scalars or ndarrays (broadcasting)
    Returns:
        float if scalar input, ndarray otherwise.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    
    is_scalar = (S.ndim == 0 and K.ndim == 0 and sigma.ndim == 0 and T.ndim == 0)
    
    w = sigma ** 2 * T
    w = np.where(w == 0, 1e-100, w)  # avoid division by zero
    d1 = (np.log(S / K) + 0.5 * w) / np.sqrt(w)
    d2 = d1 - np.sqrt(w)
    
    # Compute N(d1) and N(d2)
    N_d1 = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))
    N_d2 = 0.5 * (1.0 + np.vectorize(math.erf)(d2 / math.sqrt(2.0)))
    
    call = S * N_d1 - K * N_d2
    
    if is_scalar:
        return float(call)
    return call


def bs_put(S, K, sigma, T):
    """Black-Scholes put price."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    
    is_scalar = (S.ndim == 0 and K.ndim == 0 and sigma.ndim == 0 and T.ndim == 0)
    
    call = bs_call(S, K, sigma, T)
    put = call - S + K
    
    if is_scalar:
        return float(put)
    return put


def bs_call_total_var(S, K, w):
    """Black-Scholes call with total variance w = sigma^2 * T.
    
    For w = 0, returns intrinsic value max(S-K, 0).
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    
    is_scalar = (S.ndim == 0 and K.ndim == 0 and w.ndim == 0)
    
    # Handle zero variance case
    intrinsic = np.maximum(S - K, 0.0)
    
    # For w > 0
    sqrt_w = np.sqrt(w)
    d1 = (np.log(S / K) + 0.5 * w) / sqrt_w
    d2 = d1 - sqrt_w
    
    N_d1 = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))
    N_d2 = 0.5 * (1.0 + np.vectorize(math.erf)(d2 / math.sqrt(2.0)))
    
    call = S * N_d1 - K * N_d2
    
    # Use intrinsic for w == 0
    call = np.where(w <= 0, intrinsic, call)
    
    if is_scalar:
        return float(call)
    return call


def bs_delta(S, K, sigma, T, kind="call"):
    """Black-Scholes delta (call or put)."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    
    is_scalar = (S.ndim == 0 and K.ndim == 0 and sigma.ndim == 0 and T.ndim == 0)
    
    w = sigma ** 2 * T
    w = np.where(w == 0, 1e-100, w)
    d1 = (np.log(S / K) + 0.5 * w) / np.sqrt(w)
    
    N_d1 = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))
    
    if kind == "call":
        delta = N_d1
    elif kind == "put":
        delta = N_d1 - 1.0
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind}")
    
    if is_scalar:
        return float(delta)
    return delta


def bs_gamma(S, K, sigma, T):
    """Black-Scholes gamma (d^2C/dS^2)."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    
    is_scalar = (S.ndim == 0 and K.ndim == 0 and sigma.ndim == 0 and T.ndim == 0)
    
    w = sigma ** 2 * T
    w = np.where(w == 0, 1e-100, w)
    sqrt_w = np.sqrt(w)
    d1 = (np.log(S / K) + 0.5 * w) / sqrt_w
    
    # phi(d1) = exp(-d1^2/2) / sqrt(2*pi)
    phi_d1 = np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    gamma = phi_d1 / (S * sqrt_w)
    
    if is_scalar:
        return float(gamma)
    return gamma


def bs_vega(S, K, sigma, T):
    """Black-Scholes vega (dC/dsigma)."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    
    is_scalar = (S.ndim == 0 and K.ndim == 0 and sigma.ndim == 0 and T.ndim == 0)
    
    w = sigma ** 2 * T
    sqrt_T = np.sqrt(T)
    sqrt_w = np.sqrt(np.where(w == 0, 1e-100, w))
    d1 = (np.log(S / K) + 0.5 * w) / sqrt_w
    
    phi_d1 = np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    vega = S * phi_d1 * sqrt_T
    
    if is_scalar:
        return float(vega)
    return vega


def implied_vol(price, S, K, T, tol=1e-12):
    """Implied volatility by Brent's method with high precision.
    
    Args:
        price: float or ndarray of call prices
        S, K, T: scalar float
    Returns:
        float if scalar price, ndarray otherwise
    Raises:
        ValueError: if price is invalid or NaN
    """
    price_arr = np.asarray(price, dtype=np.float64)
    is_scalar_input = price_arr.ndim == 0
    price_arr = np.atleast_1d(price_arr)
    
    S = float(S)
    K = float(K)
    T = float(T)
    
    # Validate all prices first (don't modify input)
    intrinsic = max(S - K, 0.0)
    
    for p in price_arr:
        if np.isnan(p):
            raise ValueError("NaN price")
        if p < 0.0 or p < intrinsic or p >= S:
            raise ValueError(f"Price {p} is arbitrageable (intrinsic={intrinsic}, S={S})")
    
    # For each price, compute implied vol
    def objective(sigma, p):
        return bs_call(S, K, sigma, T) - p
    
    vols = np.zeros_like(price_arr)
    for i, p in enumerate(price_arr):
        # Use bracketing with Brent's method
        try:
            vols[i] = brentq(objective, 1e-8, 10.0, args=(p,), xtol=1e-15, rtol=1e-15)
        except ValueError:
            # If brent fails, try endpoints
            if abs(objective(1e-8, p)) < 1e-14:
                vols[i] = 1e-8
            elif abs(objective(10.0, p)) < 1e-14:
                vols[i] = 10.0
            else:
                vols[i] = 0.5  # fallback
    
    if is_scalar_input:
        return float(vols[0])
    return vols


def simulate_pricing_sample(n_paths, n_steps, T, H, eta, rho, xi0, seed, antithetic=False, kappa=1):
    """Simulate rBergomi paths for pricing: V, dW1, dW2, IV, S1, ST.
    
    Uses the hybrid scheme from rbergomi.py. Returns Monte Carlo samples for pricing estimators.
    
    Args:
        n_paths: number of Monte Carlo paths
        n_steps: number of time steps
        T: maturity (years)
        H: Hurst exponent in (0, 1/2)
        eta: vol-of-vol (eta >= 0)
        rho: correlation in [-1, 1]
        xi0: spot volatility (xi0 > 0)
        seed: random seed
        antithetic: if True, use antithetic sampling (n_paths must be even)
        kappa: 0 or 1 (hybrid scheme parameter)
    Returns:
        dict with keys V, dW1, dW2, IV, S1, ST (all float64 numpy arrays)
    Raises:
        ValueError: for invalid inputs
    """
    # Validate inputs
    if H <= 0 or H >= 0.5:
        raise ValueError(f"H must be in (0, 1/2), got {H}")
    if abs(rho) > 1:
        raise ValueError(f"|rho| must be <= 1, got {rho}")
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    if xi0 <= 0:
        raise ValueError(f"xi0 must be > 0, got {xi0}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even for antithetic sampling, got {n_paths}")
    
    # Import E1's rbergomi module
    from . import rbergomi as rb
    
    # Simulate using E1's function
    rng = np.random.default_rng(seed)
    result = rb.simulate_rbergomi(
        n_paths=n_paths, n_steps=n_steps, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
        kappa=kappa, rng=rng, antithetic=antithetic
    )
    
    V = result["V"]  # (n_paths, n_steps+1)
    dW_raw = result["dW"]  # (n_paths, n_steps) - Brownian increments driving W~
    
    # dW1 are the Brownian increments that drive the Volterra process W~
    dW1 = dW_raw  # Brownian increments driving W~ (already have variance dt)
    
    # Generate independent dW2
    dW2 = rng.standard_normal((n_paths, n_steps)) * np.sqrt(T / n_steps)
    
    # For antithetic, mirror dW2 as well
    if antithetic:
        h = n_paths // 2
        dW2_first = dW2[:h]
        dW2 = np.vstack([dW2_first, -dW2_first])
    
    # Compute IV: integrated variance (left-point rule)
    dt = T / n_steps
    IV = dt * np.sum(V[:, :-1], axis=1)
    
    # Compute S1 and ST
    sqrt_V = np.sqrt(V[:, :-1])  # V at time steps 0..n_steps-1
    
    sum_sqrt_V_dW1 = np.sum(sqrt_V * dW1, axis=1)
    sum_sqrt_V_dW2 = np.sum(sqrt_V * dW2, axis=1)
    
    S1 = np.exp(rho * sum_sqrt_V_dW1 - 0.5 * rho ** 2 * IV)
    ST = S1 * np.exp(np.sqrt(1 - rho ** 2) * sum_sqrt_V_dW2 - 0.5 * (1 - rho ** 2) * IV)
    
    return {
        "V": V,
        "dW1": dW1,
        "dW2": dW2,
        "IV": IV,
        "S1": S1,
        "ST": ST,
    }


def control_coefficient(X, Y):
    """Compute optimal control coefficient: alpha = -Cov(X,Y)/Var(Y).
    
    Returns 0.0 if Y is constant (no NaN).
    """
    X = np.asarray(X, dtype=np.float64).flatten()
    Y = np.asarray(Y, dtype=np.float64).flatten()
    
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    
    cov_XY = np.mean((X - X_mean) * (Y - Y_mean))
    var_Y = np.mean((Y - Y_mean) ** 2)
    
    if abs(var_Y) < 1e-15:
        return 0.0
    
    return -cov_XY / var_Y


def price_estimate(method, sample, strike, rho):
    """Price estimator with variance reduction.
    
    Args:
        method: "base", "antithetic", "conditional", "controlled", or "mixed"
        sample: dict with keys "IV", "S1", "ST" (and optionally "V", "dW1", "dW2")
        strike: strike price K > 0
        rho: correlation in [-1, 1]
    Returns:
        Estimate(price, stderr, alpha) namedtuple
    Raises:
        ValueError: for invalid inputs
    """
    if method not in ("base", "antithetic", "conditional", "controlled", "mixed"):
        raise ValueError(f"Unknown method: {method}")
    if abs(rho) > 1:
        raise ValueError(f"|rho| must be <= 1, got {rho}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    
    IV = np.asarray(sample["IV"], dtype=np.float64)
    S1 = np.asarray(sample["S1"], dtype=np.float64)
    ST = np.asarray(sample["ST"], dtype=np.float64)
    
    n_paths = len(IV)
    
    if method == "base":
        Z = np.maximum(ST - strike, 0.0)
        alpha = np.nan
    elif method == "antithetic":
        # Pairs as produced by the sampler: (0, n/2), (1, n/2+1), ...
        h = n_paths // 2
        Z = (np.maximum(ST[:h] - strike, 0.0) + np.maximum(ST[h:] - strike, 0.0)) / 2.0
        alpha = np.nan
        n_paths = h
    elif method == "conditional":
        Z = bs_call_total_var(S1, strike, (1 - rho ** 2) * IV)
        alpha = np.nan
    elif method == "controlled":
        # X = (ST - K)+, Y = BS(Q - IV; ST, K)
        Q = np.max(IV)
        X = np.maximum(ST - strike, 0.0)
        Y = bs_call_total_var(ST, strike, Q - IV)
        alpha = control_coefficient(X, Y)
        E_Y = bs_call_total_var(1.0, strike, Q)
        Z = X + alpha * Y
        price = np.mean(Z) - alpha * E_Y
        stderr = np.std(Z, ddof=1) / np.sqrt(n_paths)
        return Estimate(price, stderr, alpha)
    elif method == "mixed":
        # X = BS((1-rho^2) IV; S1, K), Y = BS(rho^2 (Q - IV); S1, K)
        Q = np.max(IV)
        X = bs_call_total_var(S1, strike, (1 - rho ** 2) * IV)
        Y = bs_call_total_var(S1, strike, rho ** 2 * (Q - IV))
        alpha = control_coefficient(X, Y)
        E_Y = bs_call_total_var(1.0, strike, rho ** 2 * Q)
        Z = X + alpha * Y
        price = np.mean(Z) - alpha * E_Y
        stderr = np.std(Z, ddof=1) / np.sqrt(n_paths)
        return Estimate(price, stderr, alpha)
    
    # For base, antithetic, conditional
    price = np.mean(Z)
    stderr = np.std(Z, ddof=1) / np.sqrt(n_paths)
    return Estimate(price, stderr, alpha)


# ==============================================================================
# Error measures
# ==============================================================================

def runtime_adjusted_errors(iv_estimates, iv_true, tau_ms):
    """Compute runtime-adjusted MSE (Eq 3.1).
    
    Args:
        iv_estimates: (N, m) array of N repetitions x m strikes
        iv_true: (m,) array of true implied vols
        tau_ms: runtime in milliseconds
    Returns:
        RuntimeAdjusted(phi2, psi2) namedtuple
    Raises:
        ValueError: if N < 2
    """
    iv_estimates = np.asarray(iv_estimates, dtype=np.float64)
    iv_true = np.asarray(iv_true, dtype=np.float64)
    
    N = iv_estimates.shape[0]
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    
    # sigma2_k = sum_i (est_ik - true_k)^2 / (N-1)
    sigma2_k = np.sum((iv_estimates - iv_true) ** 2, axis=0) / (N - 1)
    
    # phi2 = mean of sigma2_k
    phi2 = np.mean(sigma2_k)
    
    # psi2 = tau_ms * phi2
    psi2 = tau_ms * phi2
    
    return RuntimeAdjusted(phi2, psi2)
