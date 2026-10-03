"""Path-signature mathematics for hedging with rough volatility (API: tests/api/R2_1.md).

Tensor convention: level k holds d**k entries indexed by word (i1,...,ik) in lexicographic order
(i1 slowest); entry = iterated integral of dX^{i1} ... dX^{ik} over 0 < s1 < ... < sk < 1.
Level 0 (constant 1) is NOT stored. Signatures computed via Chen's identity on piecewise-linear paths.
"""
from __future__ import annotations

import numbers
import numpy as np

__all__ = ["signature_dim", "truncated_signature", "prefix_signatures", "lead_lag"]


def signature_dim(d: int, depth: int) -> int:
    """Total number of signature coefficients at levels 1..depth for dimension d."""
    if isinstance(d, bool) or isinstance(depth, bool):
        raise ValueError("depth must be an integer")
    if not isinstance(d, numbers.Integral) or d < 1:
        raise ValueError("d must be a positive integer")
    if not isinstance(depth, numbers.Integral) or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    return sum(d ** k for k in range(1, depth + 1))


def _segment_exp(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """Closed form: level k of a straight segment = delta^{(x)k} / k!.
    
    Args:
        delta: (..., d) array of increments
        depth: truncation level
        
    Returns:
        List of (..., d**k) arrays for k=1..depth
    """
    delta = np.asarray(delta, dtype=np.float64)
    levels = []
    cur = np.ones(delta.shape[:-1] + (1,), dtype=np.float64)
    for k in range(1, depth + 1):
        cur = (cur[..., :, None] * delta[..., None, :]).reshape(delta.shape[:-1] + (-1,)) / k
        levels.append(cur)
    return levels


def _tensor_product(a_levels: list[np.ndarray], b_levels: list[np.ndarray], d: int, depth: int) -> list[np.ndarray]:
    """Chen product: concatenation of two signatures viewed as their tensor product.
    
    Args:
        a_levels: List of level arrays for first signature
        b_levels: List of level arrays for second signature
        d: Dimension of the path
        depth: Truncation level
        
    Returns:
        List of combined level arrays
    """
    a0 = [np.ones(a_levels[0].shape[:-1] + (1,), dtype=np.float64)] + a_levels
    b0 = [np.ones(b_levels[0].shape[:-1] + (1,), dtype=np.float64)] + b_levels
    out = []
    for k in range(1, depth + 1):
        acc = None
        for i in range(k + 1):
            x, y = a0[i], b0[k - i]
            term = (x[..., :, None] * y[..., None, :]).reshape(x.shape[:-1] + (-1,))
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def truncated_signature(path, depth: int):
    """Signature of a piecewise-linear path via Chen's identity.
    
    Args:
        path: (n, d) or (B, n, d) array or tensor; n >= 2, d >= 1
        depth: int >= 1; truncation level
        
    Returns:
        (feat,) or (B, feat) array/tensor; feat = signature_dim(d, depth)
    """
    # Validate depth
    if isinstance(depth, bool) or not isinstance(depth, numbers.Integral):
        raise ValueError("depth must be an integer")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    
    # Detect input type and convert to numpy for processing
    is_torch = False
    try:
        import torch
        if isinstance(path, torch.Tensor):
            is_torch = True
            device = path.device
            dtype = path.dtype
            path_np = path.detach().cpu().numpy().astype(np.float64)
        else:
            path_np = np.asarray(path, dtype=np.float64)
    except ImportError:
        path_np = np.asarray(path, dtype=np.float64)
    
    # Validate shape
    if path_np.ndim not in (2, 3):
        raise ValueError("path must be 2-D or 3-D")
    
    single_path = path_np.ndim == 2
    if single_path:
        path_np = path_np[None, ...]
    
    B, n, d = path_np.shape
    
    if n < 2:
        raise ValueError("path must have at least 2 points")
    if d < 1:
        raise ValueError("path must have at least 1 dimension")
    
    # Check for NaN/inf
    if not np.isfinite(path_np).all():
        raise ValueError("path contains NaN or inf")
    
    # Compute signature as Chen product of segment exponentials
    cur_levels = None
    for j in range(n - 1):
        delta = path_np[:, j + 1] - path_np[:, j]
        seg_levels = _segment_exp(delta, depth)
        if cur_levels is None:
            cur_levels = seg_levels
        else:
            cur_levels = _tensor_product(cur_levels, seg_levels, d, depth)
    
    # Flatten levels and return
    flat = np.concatenate(cur_levels, axis=-1)
    
    if single_path:
        flat = flat[0]
    
    # Convert back to torch if needed
    if is_torch:
        import torch
        flat = torch.tensor(flat, dtype=dtype, device=device)
    
    return flat


def prefix_signatures(path, depth: int):
    """Signatures of all prefixes of a path.
    
    Args:
        path: (n, d) or (B, n, d) array/tensor; n >= 1, d >= 1
        depth: int >= 1; truncation level
        
    Returns:
        (n, feat) or (B, n, feat) array/tensor
    """
    # Validate depth
    if isinstance(depth, bool) or not isinstance(depth, numbers.Integral):
        raise ValueError("depth must be an integer")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    
    # Detect input type and convert
    is_torch = False
    try:
        import torch
        if isinstance(path, torch.Tensor):
            is_torch = True
            device = path.device
            dtype = path.dtype
            path_np = path.detach().cpu().numpy().astype(np.float64)
        else:
            path_np = np.asarray(path, dtype=np.float64)
    except ImportError:
        path_np = np.asarray(path, dtype=np.float64)
    
    # Validate shape
    if path_np.ndim not in (2, 3):
        raise ValueError("path must be 2-D or 3-D")
    
    single_path = path_np.ndim == 2
    if single_path:
        path_np = path_np[None, ...]
    
    B, n, d = path_np.shape
    
    if d < 1:
        raise ValueError("path must have at least 1 dimension")
    
    # Check for NaN/inf
    if not np.isfinite(path_np).all():
        raise ValueError("path contains NaN or inf")
    
    # Initialize output with zeros for row 0
    feat_dim = signature_dim(d, depth)
    out = np.zeros((B, n, feat_dim), dtype=np.float64)
    
    # Compute prefixes
    cur_levels = None
    for j in range(n - 1):
        delta = path_np[:, j + 1] - path_np[:, j]
        seg_levels = _segment_exp(delta, depth)
        if cur_levels is None:
            cur_levels = seg_levels
        else:
            cur_levels = _tensor_product(cur_levels, seg_levels, d, depth)
        out[:, j + 1] = np.concatenate(cur_levels, axis=-1)
    
    if single_path:
        out = out[0]
    
    # Convert back to torch if needed
    if is_torch:
        import torch
        out = torch.tensor(out, dtype=dtype, device=device)
    
    return out


def lead_lag(path):
    """Lead-lag transformation of a path.
    
    Args:
        path: (n, d) or (B, n, d) array/tensor; n >= 1, d >= 1
        
    Returns:
        (2n-1, 2d) or (B, 2n-1, 2d) array/tensor
    """
    # Detect input type and convert
    is_torch = False
    try:
        import torch
        if isinstance(path, torch.Tensor):
            is_torch = True
            device = path.device
            dtype = path.dtype
            path_np = path.detach().cpu().numpy().astype(np.float64)
        else:
            path_np = np.asarray(path, dtype=np.float64)
    except ImportError:
        path_np = np.asarray(path, dtype=np.float64)
    
    # Validate shape and contents
    if path_np.ndim not in (2, 3):
        raise ValueError("path must be 2-D or 3-D")
    
    single_path = path_np.ndim == 2
    if single_path:
        path_np = path_np[None, ...]
    
    B, n, d = path_np.shape
    
    if d < 1:
        raise ValueError("path must have at least 1 dimension")
    
    # Check for NaN/inf
    if not np.isfinite(path_np).all():
        raise ValueError("path contains NaN or inf")
    
    # Build lead-lag: rows alternate between (X_i, X_i) and (X_{i+1}, X_i)
    out = np.zeros((B, 2 * n - 1, 2 * d), dtype=np.float64)
    for i in range(n):
        out[:, 2 * i] = np.concatenate([path_np[:, i], path_np[:, i]], axis=-1)
        if i < n - 1:
            out[:, 2 * i + 1] = np.concatenate([path_np[:, i + 1], path_np[:, i]], axis=-1)
    
    if single_path:
        out = out[0]
    
    # Convert back to torch if needed
    if is_torch:
        import torch
        out = torch.tensor(out, dtype=dtype, device=device)
    
    return out
