"""Discrete hedging environment for a short European call: accounting, P&L decomposition, strategy execution.

Conventions: zero interest rate, unit notional; S has shape (n_paths, N+1) with S[:, i] = S(t_i);
rebalancing dates t_0..t_{N-1}; delta[:, i] is the position chosen at t_i, held over [t_i, t_{i+1}).
Position before t_0 is 0. Trading cost = cost * |trade| * price; charged on every change (entry, rebalancing, exit).
"""
from __future__ import annotations

import typing

import numpy as np


class HedgeResult(typing.NamedTuple):
    """Result of a hedging strategy: profit & loss decomposition and positions chosen."""
    pnl: np.ndarray  # (n_paths,), float64
    gains: np.ndarray  # (n_paths,), float64, sum of delta_i (S_{i+1} - S_i)
    costs: np.ndarray  # (n_paths,), float64, sum of trading costs
    payoff: np.ndarray  # (n_paths,), float64, max(S_N - strike, 0)
    positions: np.ndarray  # (n_paths, N), float64, copy of the positions delta


def hedging_pnl(S, delta, strike, cost, p0=0.0, unwind=True):
    """Discrete short-call hedging with proportional trading costs and P&L decomposition.

    Args:
        S: (n_paths, N+1) price array, float64
        delta: (n_paths, N) position array, float64
        strike: float, strike price > 0
        cost: float, cost rate >= 0
        p0: float, initial cash position (default 0.0)
        unwind: bool, if True charge cost on final position sale; if False mark to market (default True)

    Returns:
        HedgeResult with pnl = p0 + gains - costs - payoff and positions = copy of delta

    Raises:
        ValueError: for invalid input (cost < 0, non-finite values, shape mismatches, etc.)
    """
    # Validate inputs
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"cost must be non-negative and finite, got {cost}")

    S = np.asarray(S, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)

    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError(f"S must be 2-D with at least 2 columns, got shape {S.shape}")

    if delta.ndim != 2:
        raise ValueError(f"delta must be 2-D, got shape {delta.shape}")

    n_paths, N = delta.shape
    if S.shape[0] != n_paths or S.shape[1] != N + 1:
        raise ValueError(f"delta shape {delta.shape} incompatible with S shape {S.shape}; need (n_paths, {N}) and ({n_paths}, {N + 1})")

    # Check for non-finite or non-positive prices
    if not np.all(np.isfinite(S)):
        raise ValueError("S contains non-finite values")
    if np.any(S <= 0.0):
        raise ValueError("all prices must be positive")

    # Check for non-finite positions
    if not np.all(np.isfinite(delta)):
        raise ValueError("delta contains non-finite values")

    # Validate strike
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"strike must be positive and finite, got {strike}")

    # Compute P&L components
    gains = np.zeros(n_paths, dtype=np.float64)
    costs_arr = np.zeros(n_paths, dtype=np.float64)

    # Trade at each rebalancing date
    for i in range(N):
        # Change in position
        prev_pos = np.zeros(n_paths) if i == 0 else delta[:, i - 1]
        trade_size = np.abs(delta[:, i] - prev_pos)

        # Charge trading cost on the change
        trade_cost = cost * trade_size * S[:, i]
        costs_arr += trade_cost

        # Accrue gains from current position over [t_i, t_{i+1})
        gains += delta[:, i] * (S[:, i + 1] - S[:, i])

    # Final position settlement at t_N
    if unwind:
        final_cost = cost * np.abs(delta[:, N - 1]) * S[:, N]
        costs_arr += final_cost
    
    # Payoff of the short call
    payoff = np.maximum(S[:, N] - strike, 0.0)

    # P&L identity: pnl = p0 + gains - costs - payoff
    pnl = p0 + gains - costs_arr - payoff

    return HedgeResult(pnl=pnl, gains=gains, costs=costs_arr, payoff=payoff, positions=delta.copy())


def run_strategy(S, policy, strike, cost, p0=0.0, unwind=True):
    """Execute a hedging policy by calling it at each rebalancing date.

    Args:
        S: (n_paths, N+1) price array
        policy: callable(i, hist, prev) -> (n_paths,) positions to hold at step i
        strike: float
        cost: float
        p0: float, initial cash
        unwind: bool

    Returns:
        HedgeResult with positions = the decisions taken by the policy

    Raises:
        ValueError: if policy returns wrong shape or cost is invalid
    """
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"cost must be non-negative and finite, got {cost}")

    S = np.asarray(S, dtype=np.float64)
    n_paths, N_plus_1 = S.shape
    N = N_plus_1 - 1

    positions = np.empty((n_paths, N), dtype=np.float64)
    prev = np.zeros(n_paths, dtype=np.float64)

    for i in range(N):
        # Pass history as S[:, :i+1] and previous position
        hist = S[:, : i + 1]
        positions[:, i] = policy(i, hist, prev)

        # Validate shape
        if positions[:, i].shape != (n_paths,):
            raise ValueError(f"policy at step {i} returned shape {positions[:, i].shape}, expected ({n_paths},)")

        prev = positions[:, i]

    # Use hedging_pnl with the computed positions
    return hedging_pnl(S, positions, strike, cost, p0=p0, unwind=unwind)


def turnover(positions, unwind=True):
    """Sum of absolute position changes, including entry and optional exit.

    Args:
        positions: (n_paths, N) position array
        unwind: bool, if True include final exit; if False, don't

    Returns:
        (n_paths,) array of turnover values
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_paths, N = positions.shape

    # Entry: |delta_0 - 0|
    turn = np.abs(positions[:, 0])

    # Rebalancing: |delta_i - delta_{i-1}|
    for i in range(1, N):
        turn += np.abs(positions[:, i] - positions[:, i - 1])

    # Exit: |0 - delta_{N-1}|
    if unwind:
        turn += np.abs(positions[:, N - 1])

    return turn
