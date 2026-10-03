"""Classical hedging baselines: Black-Scholes, Leland adjustment, Whalley-Wilmott band policy."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def bs_call_delta(S, strike, sigma, tau):
    """Black-Scholes call delta = N(d_1).

    Args:
        S: float or array
        strike: float > 0
        sigma: float > 0 (volatility)
        tau: float >= 0 (time to maturity)

    Returns:
        float or array, same shape as S

    Raises:
        ValueError: for invalid inputs
    """
    S = np.asarray(S, dtype=np.float64)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {sigma}")
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"strike must be positive and finite, got {strike}")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError(f"tau must be non-negative and finite, got {tau}")

    S_scalar = np.isscalar(S)
    S = np.atleast_1d(S)

    if tau == 0.0:
        # At expiration: delta is 1 if in-the-money, 0 otherwise
        delta = np.asarray(S > strike, dtype=np.float64)
    else:
        d1 = (np.log(S / strike) + 0.5 * sigma**2 * tau) / (sigma * math.sqrt(tau))
        delta = norm.cdf(d1)

    return float(delta.item()) if S_scalar else delta


def bs_call_gamma(S, strike, sigma, tau):
    """Black-Scholes call gamma = N'(d_1) / (S * sigma * sqrt(tau)).

    Args:
        S: float or array
        strike: float > 0
        sigma: float > 0
        tau: float >= 0

    Returns:
        float or array

    Raises:
        ValueError: for invalid inputs
    """
    S = np.asarray(S, dtype=np.float64)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {sigma}")
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"strike must be positive and finite, got {strike}")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError(f"tau must be non-negative and finite, got {tau}")

    S_scalar = np.isscalar(S)
    S = np.atleast_1d(S)

    if tau == 0.0:
        gamma = np.zeros_like(S, dtype=np.float64)
    else:
        d1 = (np.log(S / strike) + 0.5 * sigma**2 * tau) / (sigma * math.sqrt(tau))
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(tau))

    return float(gamma.item()) if S_scalar else gamma


def leland_sigma(sigma, cost, dt):
    """Leland adjustment: sigma_L^2 = sigma^2 (1 + sqrt(2/pi) * cost / (sigma * sqrt(dt))).

    Args:
        sigma: float > 0
        cost: float >= 0 (cost rate per side)
        dt: float > 0 (time step)

    Returns:
        float

    Raises:
        ValueError: for invalid inputs
    """
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {sigma}")
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"cost must be non-negative and finite, got {cost}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")

    adjustment = 1.0 + math.sqrt(2.0 / math.pi) * cost / (sigma * math.sqrt(dt))
    return sigma * math.sqrt(adjustment)


def leland_call_delta(S, strike, sigma, tau, cost, dt):
    """Black-Scholes delta at Leland-adjusted volatility.

    Args:
        S: float or array
        strike: float > 0
        sigma: float > 0
        tau: float >= 0
        cost: float >= 0
        dt: float > 0

    Returns:
        float or array

    Raises:
        ValueError: for invalid inputs
    """
    sigma_L = leland_sigma(sigma, cost, dt)
    return bs_call_delta(S, strike, sigma_L, tau)


def ww_half_width(S, gamma_bs, cost, risk_aversion, discount=1.0):
    """Whalley-Wilmott half-width from Eq 40: h = (3 * lambda * discount * S * Gamma^2 / (2 * gamma))^(1/3).

    Args:
        S: float or array
        gamma_bs: float or array (BS gamma)
        cost: float >= 0 (lambda in formula)
        risk_aversion: float > 0 (gamma in formula)
        discount: float > 0 (default 1.0)

    Returns:
        float or array

    Raises:
        ValueError: for invalid inputs
    """
    S = np.asarray(S, dtype=np.float64)
    gamma_bs = np.asarray(gamma_bs, dtype=np.float64)

    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"cost must be non-negative and finite, got {cost}")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError(f"risk_aversion must be positive and finite, got {risk_aversion}")
    if not np.isfinite(discount) or discount <= 0.0:
        raise ValueError(f"discount must be positive and finite, got {discount}")
    if np.any(gamma_bs < 0.0):
        raise ValueError(f"gamma_bs must be non-negative, got min {np.min(gamma_bs)}")

    S_scalar = np.isscalar(S)
    gamma_bs_scalar = np.isscalar(gamma_bs)

    S = np.atleast_1d(S)
    gamma_bs = np.atleast_1d(gamma_bs)

    # Broadcast to common shape
    S_bc, gamma_bc = np.broadcast_arrays(S, gamma_bs)

    half_width = (3.0 * cost * discount * S_bc * gamma_bc**2 / (2.0 * risk_aversion))**(1.0 / 3.0)

    if S_scalar:
        return float(half_width.item())
    else:
        return half_width


def apply_no_trade_band(prev, center, half_width):
    """Clip positions to the band [center - half_width, center + half_width] (nearest edge).

    Args:
        prev: array, previous positions
        center: float or array, band center
        half_width: float or array

    Returns:
        new array (not modifying input)

    Raises:
        ValueError: for negative half_width
    """
    prev = np.asarray(prev, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    half_width = np.asarray(half_width, dtype=np.float64)

    if np.any(half_width < 0.0):
        raise ValueError(f"half_width must be non-negative, got min {np.min(half_width)}")

    # Clip to [center - half_width, center + half_width]
    lower = center - half_width
    upper = center + half_width
    return np.clip(prev, lower, upper)


def bs_delta_policy(strike, sigma, T, n_steps):
    """Black-Scholes delta hedging policy.

    Args:
        strike: float > 0
        sigma: float > 0
        T: float > 0 (time to maturity)
        n_steps: int >= 1 (number of rebalancing steps)

    Returns:
        callable(i, hist, prev) -> (n_paths,) positions

    Raises:
        ValueError: for invalid inputs
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps}")

    dt = T / n_steps

    def policy(i, hist, prev):
        tau = T - i * dt
        S_now = hist[:, -1]
        delta = bs_call_delta(S_now, strike, sigma, tau)
        return np.asarray(delta, dtype=np.float64)

    return policy


def leland_delta_policy(strike, sigma, T, n_steps, cost):
    """Leland-adjusted BS delta hedging policy.

    Args:
        strike: float > 0
        sigma: float > 0
        T: float > 0
        n_steps: int >= 1
        cost: float >= 0

    Returns:
        callable(i, hist, prev) -> (n_paths,) positions

    Raises:
        ValueError: for invalid inputs
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps}")
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"cost must be non-negative and finite, got {cost}")

    dt = T / n_steps
    sigma_L = leland_sigma(sigma, cost, dt)

    def policy(i, hist, prev):
        tau = T - i * dt
        S_now = hist[:, -1]
        delta = bs_call_delta(S_now, strike, sigma_L, tau)
        return np.asarray(delta, dtype=np.float64)

    return policy


def ww_policy(strike, sigma, T, n_steps, cost, risk_aversion, discount=1.0):
    """Whalley-Wilmott band policy: BS delta with no-trade band.

    Args:
        strike: float > 0
        sigma: float > 0
        T: float > 0
        n_steps: int >= 1
        cost: float >= 0
        risk_aversion: float > 0 (gamma)
        discount: float > 0

    Returns:
        callable(i, hist, prev) -> (n_paths,) positions

    Raises:
        ValueError: for invalid inputs
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps}")
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"cost must be non-negative and finite, got {cost}")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError(f"risk_aversion must be positive and finite, got {risk_aversion}")

    dt = T / n_steps

    def policy(i, hist, prev):
        tau = T - i * dt
        S_now = hist[:, -1]

        # Centre: BS delta
        centre = bs_call_delta(S_now, strike, sigma, tau)

        # Half-width: WW formula
        gamma = bs_call_gamma(S_now, strike, sigma, tau)
        half = ww_half_width(S_now, gamma, cost, risk_aversion, discount=discount)

        # Apply band
        positions = apply_no_trade_band(prev, centre, half)
        return np.asarray(positions, dtype=np.float64)

    return policy
