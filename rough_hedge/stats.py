"""Statistical methods for hedging comparison: paired bootstrap CVaR95 differences, Holm adjustment, effect size, seed aggregation.

Claim C16: paired-bootstrap coverage on synthetic data with known differences; supports H1 (main comparison decision rule).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats as sps

from rough_hedge.risk import cvar_tail_mean


class BootstrapResult(NamedTuple):
    """Paired bootstrap confidence interval and hypothesis test result."""
    estimate: float
    lo: float
    hi: float
    pvalue: float
    se: float
    n_boot: int
    level: float


class EffectSize(NamedTuple):
    """Cohen's d effect size with confidence interval."""
    d: float
    lo: float
    hi: float
    level: float


class SeedSummary(NamedTuple):
    """Summary of per-seed estimates with replication statistics."""
    mean: float
    sd: float
    se: float
    lo: float
    hi: float
    n_seeds: int
    n_positive: int
    level: float


def paired_bootstrap(loss_a, loss_b, n_boot=2000, seed=0, alpha=0.95, level=0.95):
    """Paired bootstrap confidence interval and p-value for CVaR alpha difference.
    
    Statistic: CVaR_alpha(loss_a) - CVaR_alpha(loss_b), computed on equal-length loss vectors.
    Recipe (bit-reproducible): resample with equal probability from both losses jointly, then
    compute tail-mean difference on each bootstrap replicate. Confidence interval from quantiles,
    p-value from the two-sided add-one rule.
    
    Args:
        loss_a, loss_b: 1-D arrays, same length n >= 20, finite values
        n_boot: int >= 10, number of bootstrap replicates
        seed: int, random seed
        alpha: float in [0, 1), quantile level for CVaR (default 0.95)
        level: float in (0, 1), confidence level for interval (default 0.95)
    
    Returns:
        BootstrapResult with estimate (point CVaR diff), lo/hi (interval), pvalue (two-sided),
        se (bootstrap sd), n_boot, level
    
    Raises:
        ValueError: for invalid inputs
    """
    # Validation
    loss_a = np.asarray(loss_a, dtype=np.float64)
    loss_b = np.asarray(loss_b, dtype=np.float64)
    
    if loss_a.ndim != 1 or loss_b.ndim != 1:
        raise ValueError("loss_a and loss_b must be 1-D")
    if len(loss_a) != len(loss_b):
        raise ValueError(f"loss_a and loss_b must have equal length, got {len(loss_a)} and {len(loss_b)}")
    if len(loss_a) < 20:
        raise ValueError(f"sample size must be >= 20, got {len(loss_a)}")
    if not np.all(np.isfinite(loss_a)) or not np.all(np.isfinite(loss_b)):
        raise ValueError("loss_a and loss_b must be finite")
    if n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {n_boot}")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")
    
    n = len(loss_a)
    
    # Compute point estimate
    estimate = cvar_tail_mean(loss_a, alpha) - cvar_tail_mean(loss_b, alpha)
    
    # Bootstrap resampling (bit-reproducible)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    
    # Bootstrap differences.  The statistic is the same interval-overlap tail
    # mean used by cvar_tail_mean, evaluated across all resamples at once.  This
    # avoids Python dispatch and validation for every replicate (the coverage
    # test alone evaluates 80,000 replicates) while preserving the generated
    # indices and hence the documented seeded recipe exactly.
    i = np.arange(n, dtype=np.float64)
    overlap = np.maximum(0.0, (i + 1.0) / n - np.maximum(i / n, alpha))
    weights = overlap / (1.0 - alpha)
    samples_a = np.sort(loss_a[idx], axis=1)
    samples_b = np.sort(loss_b[idx], axis=1)
    d = np.sum(samples_a * weights, axis=1) - np.sum(samples_b * weights, axis=1)
    
    # Confidence interval from quantiles
    q = (1.0 - level) / 2.0
    lo = float(np.quantile(d, q, method='linear'))
    hi = float(np.quantile(d, 1.0 - q, method='linear'))
    
    # Two-sided p-value with add-one rule
    le = (1 + int(np.sum(d <= 0.0))) / (n_boot + 1.0)
    ge = (1 + int(np.sum(d >= 0.0))) / (n_boot + 1.0)
    pvalue = float(min(1.0, 2.0 * min(le, ge)))
    
    # Bootstrap standard error
    se = float(d.std(ddof=1))
    
    return BootstrapResult(estimate, lo, hi, pvalue, se, n_boot, level)


def holm(pvalues, alpha=0.05):
    """Holm step-down adjustment of p-values.
    
    Rejects the k-th smallest p-value if p_(k) <= alpha / (m - k + 1) where m is the total
    number of tests. Stops at the first non-rejection.
    
    Args:
        pvalues: 1-D array of p-values in [0, 1]
        alpha: float in (0, 1), significance level (default 0.05)
    
    Returns:
        (adjusted, reject): tuple of 1-D float and 1-D bool arrays in input order
        - adjusted: adjusted p-values, min(1, monotone in sorted p)
        - reject: True for each p <= adjusted p
    
    Raises:
        ValueError: for empty, out-of-range, or NaN p-values or invalid alpha
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    
    if pvalues.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if len(pvalues) == 0:
        raise ValueError("pvalues cannot be empty")
    if not np.all((pvalues >= 0.0) & (pvalues <= 1.0)) or np.any(np.isnan(pvalues)):
        raise ValueError("pvalues must be in [0, 1] with no NaN")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    m = len(pvalues)
    
    # Sort indices stably
    order = np.argsort(pvalues, kind='stable')
    
    # Compute adjusted p-values
    adjusted = np.empty(m, dtype=np.float64)
    prev_adj = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvalues[idx]
        adj = max(prev_adj, min(1.0, adj))  # Monotone and bounded
        adjusted[idx] = adj
        prev_adj = adj
    
    # Reject where adjusted <= alpha
    reject = (adjusted <= alpha)
    
    return adjusted, reject


def cohens_d(diff, level=0.95):
    """Cohen's d effect size with confidence interval.
    
    Computes standardized mean difference d = mean(diff) / std(diff, ddof=1) with
    standard error accounting for sampling variability and effect size uncertainty.
    Interval: d ± z_{(1+level)/2} * sqrt(1/n + d^2/(2n))
    
    Args:
        diff: 1-D array of paired differences, at least 3 values
        level: float in (0, 1), confidence level (default 0.95)
    
    Returns:
        EffectSize with d, lo, hi (interval), level
    
    Raises:
        ValueError: for invalid input
    """
    diff = np.asarray(diff, dtype=np.float64)
    
    if diff.ndim != 1:
        raise ValueError("diff must be 1-D")
    if len(diff) < 3:
        raise ValueError(f"need at least 3 differences, got {len(diff)}")
    if not np.all(np.isfinite(diff)):
        raise ValueError("diff must be finite")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")
    
    n = len(diff)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    
    if sd_diff == 0.0:
        raise ValueError("effect size undefined when std(diff) == 0")
    
    d = mean_diff / sd_diff
    se = math.sqrt(1.0 / n + d**2 / (2.0 * n))
    z = sps.norm.ppf((1.0 + level) / 2.0)
    lo = d - z * se
    hi = d + z * se
    
    return EffectSize(d, lo, hi, level)


def seed_aggregate(per_seed_estimates, level=0.95):
    """Summary of per-seed estimates with replication-unit statistics.
    
    Treats each input value as one seed's estimate. Computes mean, standard deviation
    (ddof=1), and t-interval (seeds are the replication unit).
    
    Args:
        per_seed_estimates: 1-D array of per-seed point estimates, at least 2 values
        level: float in (0, 1), confidence level (default 0.95)
    
    Returns:
        SeedSummary with mean, sd, se, lo, hi, n_seeds, n_positive, level
    
    Raises:
        ValueError: for invalid input
    """
    per_seed_estimates = np.asarray(per_seed_estimates, dtype=np.float64)
    
    if per_seed_estimates.ndim != 1:
        raise ValueError("per_seed_estimates must be 1-D")
    if len(per_seed_estimates) < 2:
        raise ValueError(f"need at least 2 seeds, got {len(per_seed_estimates)}")
    if not np.all(np.isfinite(per_seed_estimates)):
        raise ValueError("estimates must be finite")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")
    
    n_seeds = len(per_seed_estimates)
    mean = float(per_seed_estimates.mean())
    sd = float(per_seed_estimates.std(ddof=1))
    se = sd / math.sqrt(n_seeds)
    
    # t-interval
    t_crit = sps.t.ppf((1.0 + level) / 2.0, n_seeds - 1)
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    
    n_positive = int(np.sum(per_seed_estimates > 0.0))
    
    return SeedSummary(mean, sd, se, lo, hi, n_seeds, n_positive, level)
