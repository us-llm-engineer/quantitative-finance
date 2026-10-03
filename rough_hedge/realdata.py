"""Real data loading, calibration, and hedging replay for rough Bergomi.

Units: variances per day (not annualised); volatilities and VIX levels DECIMAL annualised vols;
times in years; rates zero.
"""
from __future__ import annotations

import hashlib
import json
from collections import namedtuple
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm


class ChartDataError(ValueError):
    """Malformed or unusable chart payload."""
    pass


def load_chart_json(path: str | Path) -> pd.DataFrame:
    """Load Yahoo Finance chart JSON and return OHLCV DataFrame.

    Args:
        path: path to chart JSON file

    Returns:
        DataFrame with index (tz-aware UTC, ascending), columns [open, high, low, close, volume].
        df.attrs["n_dropped"] is the count of rows with any null in the five OHLCV fields.

    Raises:
        FileNotFoundError: if file does not exist
        ChartDataError: if JSON is invalid, payload malformed, or fields missing
    """
    path = Path(path)

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, ValueError) as e:
        raise ChartDataError(f"Invalid JSON: {e}") from e

    # Check chart structure
    try:
        chart = data.get("chart", {})
        if chart.get("error") is not None:
            error_info = chart["error"]
            error_code = error_info.get("code", "Unknown")
            error_msg = error_info.get("description", "Unknown error")
            raise ChartDataError(f"API error: {error_code}: {error_msg}")

        result = chart.get("result")
        if result is None:
            raise ChartDataError("Missing 'chart.result'")
        if len(result) == 0:
            raise ChartDataError("Empty 'chart.result'")

        bar_data = result[0]
        timestamps = bar_data.get("timestamp")
        if timestamps is None:
            raise ChartDataError("Missing 'timestamp' array")

        indicators = bar_data.get("indicators")
        if indicators is None:
            raise ChartDataError("Missing 'indicators'")

        quote_list = indicators.get("quote")
        if quote_list is None or len(quote_list) == 0:
            raise ChartDataError("Missing 'indicators.quote'")

        quote = quote_list[0]
        required_fields = ["open", "high", "low", "close", "volume"]
        for field in required_fields:
            if field not in quote:
                raise ChartDataError(f"Missing field '{field}' in quote")

        # Check array lengths match
        n = len(timestamps)
        for field in required_fields:
            if len(quote[field]) != n:
                raise ChartDataError(f"Length mismatch: {field} has {len(quote[field])} rows, expected {n}")

    except ChartDataError:
        raise
    except (KeyError, TypeError) as e:
        raise ChartDataError(f"Malformed payload: {e}") from e

    # Build DataFrame with tz-aware UTC timestamps
    df_data = {
        "timestamp": pd.to_datetime([pd.Timestamp.fromtimestamp(ts, tz="UTC") for ts in timestamps]),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    }
    df = pd.DataFrame(df_data)

    # Drop rows with any null in the five fields
    n_before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    n_dropped = n_before - len(df)

    df = df.set_index("timestamp")
    df = df.astype(np.float64)
    df.attrs["n_dropped"] = int(n_dropped)

    return df


def realized_variance(bars: pd.DataFrame, overnight: str = "separate") -> tuple[pd.Series, pd.Series]:
    """Compute realized variance from intraday bars.

    Args:
        bars: DataFrame with tz-aware UTC index and columns [open, high, low, close, volume]
        overnight: "separate" (default), "drop", or "include" for overnight return handling

    Returns:
        (rv, overnight_ret) where both are Series indexed by trading date (naive UTC midnight).
        rv: sum of squared intraday returns (no annualisation).
        overnight_ret: overnight log returns (indexed by next day).

    Raises:
        ValueError: if overnight mode is invalid
    """
    if overnight not in ("separate", "drop", "include"):
        raise ValueError(f"overnight must be 'separate', 'drop', or 'include', got {overnight!r}")

    # Convert index to New York trading date
    bars_ny = bars.copy()
    bars_ny["trading_date"] = bars_ny.index.tz_convert("America/New_York").date

    rv_dict = {}
    overnight_dict = {}
    prev_close = None
    prev_date = None

    for trading_date, day_data in bars_ny.groupby("trading_date", sort=True):
        closes = day_data["close"].to_numpy()
        opens = day_data["open"].to_numpy()

        # Intraday returns: log(close_0/open_0), log(close_k/close_{k-1})
        first_return = np.log(closes[0] / opens[0])
        intraday_returns = [first_return]
        for k in range(1, len(closes)):
            intraday_returns.append(np.log(closes[k] / closes[k-1]))
        intraday_returns = np.array(intraday_returns)

        # Realized variance of the day (sum of squared intraday returns)
        rv_d = float(np.sum(intraday_returns ** 2))
        rv_dict[pd.Timestamp(trading_date)] = rv_d

        # Overnight return: log(open_0 / prev_close)
        if prev_close is not None:
            overnight_ret = np.log(opens[0] / prev_close)
            overnight_dict[pd.Timestamp(trading_date)] = overnight_ret

        prev_close = closes[-1]
        prev_date = trading_date

    rv_series = pd.Series(rv_dict, dtype=np.float64)
    rv_series.index.name = None

    if overnight == "separate":
        overnight_series = pd.Series(overnight_dict, dtype=np.float64)
        overnight_series.index.name = None
        return rv_series, overnight_series
    elif overnight == "drop":
        return rv_series, pd.Series([], dtype=np.float64)
    else:  # include
        # Include mode: rv_d = intraday sum of squares + overnight_d^2, drop first day
        rv_incl = {}
        for date, og_ret in overnight_dict.items():
            rv_incl[date] = rv_dict[date] + og_ret ** 2
        rv_series_incl = pd.Series(rv_incl, dtype=np.float64)
        rv_series_incl.index.name = None
        overnight_series = pd.Series(overnight_dict, dtype=np.float64)
        overnight_series.index.name = None
        return rv_series_incl, overnight_series


def rv_noise_comparison(rv_a: pd.Series, rv_b: pd.Series) -> dict:
    """Compare realized variance series over common trading days.

    Args:
        rv_a, rv_b: Series indexed by trading date (DatetimeIndex)

    Returns:
        dict with keys: n_common, mean_log_ratio, sd_log_ratio, corr

    Raises:
        ValueError: if fewer than 3 common days
    """
    common_idx = rv_a.index.intersection(rv_b.index)
    if len(common_idx) < 3:
        raise ValueError(f"Fewer than 3 common days: {len(common_idx)}")

    rv_a_common = rv_a.loc[common_idx]
    rv_b_common = rv_b.loc[common_idx]

    log_ratio = np.log(rv_a_common.to_numpy()) - np.log(rv_b_common.to_numpy())
    mean_lr = float(np.mean(log_ratio))
    sd_lr = float(np.std(log_ratio, ddof=1))
    corr = float(np.corrcoef(rv_a_common, rv_b_common)[0, 1])

    return {
        "n_common": len(common_idx),
        "mean_log_ratio": mean_lr,
        "sd_log_ratio": sd_lr,
        "corr": corr,
    }


def volterra_increment_constant(H: float) -> float:
    """Compute c(H) = 1 + 2H * int_0^inf ((u+1)^(H-1/2) - u^(H-1/2))^2 du.

    So that Var(W~_{t+D} - W~_t) = c(H) D^(2H) for the Volterra process.

    Args:
        H: Hurst exponent in (0, 1/2]

    Returns:
        c(H) computed by adaptive quadrature

    Raises:
        ValueError: if H not in (0, 1/2]
    """
    if not (0 < H <= 0.5):
        raise ValueError(f"H must be in (0, 1/2], got {H}")

    def integrand(u):
        return ((u + 1.0) ** (H - 0.5) - u ** (H - 0.5)) ** 2

    integral = quad(integrand, 0, np.inf, limit=200)[0]
    return 1.0 + 2.0 * H * integral


class ForwardVarianceCurve(namedtuple("ForwardVarianceCurve", ["tenors", "xi0", "flagged", "has_negative"])):
    """Forward variance curve: tenors (years), xi0 (annualised forward variance), flags for negative."""
    pass


def forward_variance_curve(levels_by_tenor: dict[float, float], enforce: bool = False) -> ForwardVarianceCurve:
    """Compute forward variance curve from tenor -> vol mapping.

    Args:
        levels_by_tenor: dict {tenor in years: decimal vol}
        enforce: if True, raise ValueError for negative forward variance

    Returns:
        ForwardVarianceCurve with tenors, xi0, flagged, has_negative

    Raises:
        ValueError: if dict empty, tenor <= 0, vol <= 0, or enforce=True and xi0 < 0
    """
    if not levels_by_tenor:
        raise ValueError("levels_by_tenor must not be empty")

    for tenor, vol in levels_by_tenor.items():
        if tenor <= 0:
            raise ValueError(f"tenor must be positive, got {tenor}")
        if vol <= 0:
            raise ValueError(f"vol must be positive, got {vol}")

    # Sort tenors
    sorted_tenors = sorted(levels_by_tenor.keys())
    tenors = np.array(sorted_tenors, dtype=np.float64)

    # Compute total variance: w_i = T_i * sigma_i^2
    total_vars = tenors * np.array([levels_by_tenor[t] ** 2 for t in sorted_tenors])

    # Forward variance on each interval
    xi0_values = []
    T_prev = 0.0
    w_prev = 0.0
    for i, (T, w) in enumerate(zip(tenors, total_vars)):
        xi0_i = (w - w_prev) / (T - T_prev)
        xi0_values.append(xi0_i)
        T_prev = T
        w_prev = w

    xi0 = np.array(xi0_values, dtype=np.float64)
    flagged = xi0 < 0
    has_negative = bool(np.any(flagged))

    if enforce and has_negative:
        raise ValueError(f"enforce=True but forward variance is negative at tenor(s): {tenors[flagged]}")

    return ForwardVarianceCurve(tenors=tenors, xi0=xi0, flagged=flagged, has_negative=has_negative)


def calibrate_rbergomi_real(
    log_rv: np.ndarray,
    returns: np.ndarray,
    dlog_vix: np.ndarray,
    vix_curve: dict[float, float],
    dt: float = 1 / 252,
    lags: range = range(1, 31),
) -> dict:
    """Calibrate rough Bergomi to realized variance and returns.

    Args:
        log_rv: daily log variance (length n)
        returns: daily log returns (length n-1)
        dlog_vix: daily change of log VIX (length n-1)
        vix_curve: dict {tenor in years: decimal vol}
        dt: time step (default 1/252 for daily data)
        lags: range of lag distances for m2 fit

    Returns:
        dict with keys: H, eta, rho, xi0_curve, diagnostics
    """
    log_rv = np.asarray(log_rv, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    dlog_vix = np.asarray(dlog_vix, dtype=np.float64)

    n = len(log_rv)
    lags_list = list(lags)

    # Compute m2(D) = mean over t of (log_rv[t+D] - log_rv[t])^2 for each lag
    m2_values = []
    for lag in lags_list:
        if lag >= n:
            break
        diff_sq = (log_rv[lag:] - log_rv[:-lag]) ** 2
        m2_values.append(np.mean(diff_sq))

    if len(m2_values) < 2:
        raise ValueError("Not enough lags to calibrate")

    lags_fit = lags_list[:len(m2_values)]

    # Fit log m2(D) = log(eta^2 c(H)) + 2H log(D dt)
    # X = log(D dt), Y = log(m2)
    X = np.log(np.array(lags_fit, dtype=np.float64) * dt)
    Y = np.log(np.array(m2_values, dtype=np.float64))

    # Least squares: Y = a + b X, where b = 2H
    n_fit = len(X)
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    b = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
    a = Y_mean - b * X_mean

    H = b / 2.0

    # eta = sqrt(exp(a) / c(H))
    c_H = volterra_increment_constant(H)
    eta = np.sqrt(np.exp(a) / c_H)

    # rho: correlation of normalized returns with dlog_vix, divided by attenuation factor
    sqrt_rv = np.exp(log_rv[:-1] / 2.0)
    normalized_returns = returns / sqrt_rv

    # Attenuation factor: sqrt(2H) / ((H + 1/2) * sqrt(c(H)))
    atten = np.sqrt(2 * H) / ((H + 0.5) * np.sqrt(c_H))

    # Pearson correlation
    corr = np.corrcoef(normalized_returns, dlog_vix)[0, 1]
    rho = corr / atten
    rho = np.clip(rho, -1.0, 1.0)

    # Forward variance curve from vix_curve
    xi0_curve = forward_variance_curve(vix_curve, enforce=False)

    return {
        "H": float(H),
        "eta": float(eta),
        "rho": float(rho),
        "xi0_curve": xi0_curve,
        "diagnostics": {"lags": lags_fit, "m2": m2_values, "c_H": c_H, "atten": atten},
    }


def moving_block_bootstrap(
    x: np.ndarray,
    block: int,
    n_boot: int,
    seed: int,
    stat: Callable[[np.ndarray], float],
    alpha: float = 0.95,
) -> tuple[float, float, float]:
    """Moving block bootstrap for dependent data.

    Args:
        x: 1-D array
        block: block length (must be in [1, n])
        n_boot: number of bootstrap resamples
        seed: random seed
        stat: statistic function(x) -> float
        alpha: confidence level (default 0.95)

    Returns:
        (estimate, lo, hi) where estimate = stat(x), and (lo, hi) are quantile intervals

    Raises:
        ValueError: if block < 1, block > n, or n_boot < 1
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if block < 1 or block > n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    estimate = float(stat(x))

    rng = np.random.default_rng(seed)

    # Number of blocks needed
    n_blocks = int(np.ceil(n / block))

    # Bootstrap resamples
    boot_stats = []
    for _ in range(n_boot):
        # Random start positions for blocks
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        indices = []
        for start in starts:
            indices.extend(range(start, start + block))
        # Truncate to length n
        indices = indices[:n]
        x_resample = x[indices]
        boot_stats.append(float(stat(x_resample)))

    boot_stats = np.array(boot_stats, dtype=np.float64)

    # Quantiles
    lo = float(np.quantile(boot_stats, (1 - alpha) / 2))
    hi = float(np.quantile(boot_stats, (1 + alpha) / 2))

    return estimate, lo, hi


def non_overlapping_windows(n_prices: int, length: int) -> list[tuple[int, int]]:
    """Generate non-overlapping return windows of specified length.

    Args:
        n_prices: number of prices (N+1 for N return days)
        length: window length in days

    Returns:
        list of (start, stop) tuples where S[start:stop+1] are the prices.

    Raises:
        ValueError: if length < 1
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    n_returns = n_prices - 1
    if n_returns < length:
        return []

    windows = []
    for k in range(n_returns // length):
        start = k * length
        stop = (k + 1) * length
        windows.append((start, stop))

    return windows


class ReplayResult(namedtuple("ReplayResult", ["pnl", "gains", "costs", "payoff", "premium", "positions", "turnover"])):
    """Result of replay_hedge: per-window arrays of P&L components."""
    pass


def replay_hedge(
    S_window: np.ndarray,
    iv_window: np.ndarray,
    strike: float,
    cost: float,
    policy: str | Callable,
    T: float,
) -> ReplayResult:
    """Replay hedging of a short European call with realized prices and vol.

    Args:
        S_window: price array (n, N+1)
        iv_window: implied vol array (n, N+1), decimal annualised
        strike: strike price
        cost: proportional cost rate
        policy: "bs_delta" or callable policy(i, S_hist, iv_hist, prev_pos) -> (n,)
        T: maturity in years

    Returns:
        ReplayResult with per-window arrays

    Raises:
        ValueError: if cost < 0, shape mismatch, or S/iv not positive
    """
    from rough_hedge.pricing import bs_call, bs_delta
    from rough_hedge.hedging import turnover as ref_turnover

    S_window = np.asarray(S_window, dtype=np.float64)
    iv_window = np.asarray(iv_window, dtype=np.float64)

    if S_window.ndim != 2 or iv_window.ndim != 2:
        raise ValueError(f"S_window and iv_window must be 2-D")

    n, N_plus_1 = S_window.shape
    N = N_plus_1 - 1

    if iv_window.shape != (n, N_plus_1):
        raise ValueError(f"iv_window shape {iv_window.shape} must match S_window {S_window.shape}")

    if cost < 0:
        raise ValueError(f"cost must be >= 0, got {cost}")

    if np.any(S_window <= 0) or np.any(iv_window <= 0):
        raise ValueError("All prices and implied vols must be positive")

    # Premium received
    premium = bs_call(S_window[:, 0], strike, iv_window[:, 0], T)

    # Execute hedging policy
    positions = np.zeros((n, N), dtype=np.float64)
    gains_arr = np.zeros(n, dtype=np.float64)
    costs_arr = np.zeros(n, dtype=np.float64)

    for i in range(N):
        if callable(policy):
            S_hist = S_window[:, :i+1]
            iv_hist = iv_window[:, :i+1]
            prev_pos = positions[:, i-1] if i > 0 else np.zeros(n)
            positions[:, i] = policy(i, S_hist, iv_hist, prev_pos)
        elif policy == "bs_delta":
            tau = T * (1 - i / N)
            positions[:, i] = bs_delta(S_window[:, i], strike, iv_window[:, i], tau, kind="call")
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Trading cost on change
        prev_pos = positions[:, i-1] if i > 0 else np.zeros(n)
        trade_size = np.abs(positions[:, i] - prev_pos)
        costs_arr += cost * trade_size * S_window[:, i]

        # Gains from the position
        gains_arr += positions[:, i] * (S_window[:, i+1] - S_window[:, i])

    # Unwind at maturity: cost on final position
    costs_arr += cost * np.abs(positions[:, N-1]) * S_window[:, N]

    # Payoff of short call
    payoff = np.maximum(S_window[:, N] - strike, 0.0)

    # P&L
    pnl = premium + gains_arr - costs_arr - payoff

    # Turnover
    turnover_arr = ref_turnover(positions, unwind=True)

    return ReplayResult(
        pnl=pnl,
        gains=gains_arr,
        costs=costs_arr,
        payoff=payoff,
        premium=premium,
        positions=positions,
        turnover=turnover_arr,
    )


def verify_manifest(data_dir: str | Path) -> list[str]:
    """Verify manifest integrity against raw files.

    Args:
        data_dir: root data directory

    Returns:
        Empty list if all OK, otherwise list of problem strings.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "MANIFEST.json"

    if not manifest_path.exists():
        return [f"{manifest_path}: missing"]

    try:
        with open(manifest_path) as f:
            entries = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return [f"{manifest_path}: {e}"]

    problems = []
    for entry in entries:
        symbol = entry.get("symbol")
        interval = entry.get("interval")
        if not symbol or not interval:
            continue

        filename = f"{symbol}_{interval}.json"
        file_path = data_dir / "raw" / filename

        if not file_path.exists():
            problems.append(f"{filename}: missing")
            continue

        file_bytes = file_path.read_bytes()
        expected_sha = entry.get("sha256")
        expected_size = entry.get("bytes")

        actual_sha = hashlib.sha256(file_bytes).hexdigest()
        actual_size = len(file_bytes)

        if actual_sha != expected_sha:
            problems.append(f"{filename}: sha256 mismatch")

        if actual_size != expected_size:
            problems.append(f"{filename}: bytes mismatch")

    return problems
