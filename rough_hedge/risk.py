"""Risk measures for hedging loss distributions: CVaR (dual form), entropic risk, value-at-risk.

Conventions: losses L = -pnl; all functions return a Python float for numpy input,
a 0-d float64 torch tensor with autograd for torch input.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def cvar_tail_mean(loss, alpha):
    """Conditional Value-at-Risk as the tail mean: (1/(1-alpha)) * integral_alpha^1 of empirical quantile.

    Uses interval-overlap weights on the sorted sample.

    Args:
        loss: 1-D array or tensor, shape (n,)
        alpha: float in [0, 1)

    Returns:
        float (numpy) or 0-d float64 tensor with autograd (torch)

    Raises:
        ValueError: for invalid alpha, empty sample, or NaN values
    """
    is_torch = torch is not None and isinstance(loss, torch.Tensor)

    if is_torch:
        loss_np = loss.detach().cpu().numpy() if loss.requires_grad else loss.numpy()
    else:
        loss_np = np.asarray(loss, dtype=np.float64)

    if loss_np.ndim != 1:
        raise ValueError(f"loss must be 1-D, got shape {loss_np.shape}")
    if loss_np.size == 0:
        raise ValueError("loss cannot be empty")
    if not np.all(np.isfinite(loss_np)):
        raise ValueError("loss contains non-finite values")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    n = loss_np.size
    sorted_loss = np.sort(loss_np)
    sorted_idx = np.argsort(loss_np)

    # Interval-overlap weights: fraction of each sorted element in the [alpha, 1] interval
    i = np.arange(n)
    overlap = np.maximum(0.0, (i + 1.0) / n - np.maximum(i / n, alpha))
    weights = overlap / (1.0 - alpha)

    tail_mean_val = np.dot(sorted_loss, weights)

    # For torch: create a custom autograd function
    if is_torch:
        class CVaRTailMeanFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, loss_t):
                ctx.sorted_idx = sorted_idx
                ctx.weights = weights
                return torch.tensor(tail_mean_val, dtype=torch.float64)

            @staticmethod
            def backward(ctx, grad_output):
                grad_values = np.zeros_like(loss_np)
                grad_values[ctx.sorted_idx] = ctx.weights
                return grad_output * torch.tensor(grad_values, dtype=torch.float64)

        result = CVaRTailMeanFunc.apply(loss)
        return result
    else:
        return float(tail_mean_val)


def cvar_ru(loss, alpha):
    """CVaR via the Rockafellar-Uryasev dual form: min_v v + (1/(1-alpha)) E[max(loss - v, 0)].

    Minimizes the objective exactly (the minimum is at the lower alpha-quantile).

    Args:
        loss: 1-D array or tensor, shape (n,)
        alpha: float in [0, 1)

    Returns:
        float or 0-d torch tensor, equals cvar_tail_mean to 1e-9

    Raises:
        ValueError: for invalid alpha, empty sample, or NaN values
    """
    is_torch = torch is not None and isinstance(loss, torch.Tensor)

    if is_torch:
        loss_np = loss.detach().cpu().numpy() if loss.requires_grad else loss.numpy()
    else:
        loss_np = np.asarray(loss, dtype=np.float64)

    if loss_np.ndim != 1:
        raise ValueError(f"loss must be 1-D, got shape {loss_np.shape}")
    if loss_np.size == 0:
        raise ValueError("loss cannot be empty")
    if not np.all(np.isfinite(loss_np)):
        raise ValueError("loss contains non-finite values")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    # The optimal v is the lower alpha-quantile
    n = loss_np.size
    idx = int(np.ceil(n * alpha)) - 1
    idx = max(0, min(idx, n - 1))
    v_star = np.sort(loss_np)[idx]

    # Evaluate the objective at v_star
    excess = np.maximum(loss_np - v_star, 0.0)
    cvar_val = v_star + np.mean(excess) / (1.0 - alpha)

    if is_torch:
        # For torch, we need to compute gradients
        class CVaRRUFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, loss_t):
                ctx.v_star = v_star
                ctx.n = n
                ctx.alpha = alpha
                return torch.tensor(cvar_val, dtype=torch.float64)

            @staticmethod
            def backward(ctx, grad_output):
                above = (loss_np > ctx.v_star).astype(np.float64)
                grad_v = grad_output * (1.0 - (above.sum() / ((1.0 - ctx.alpha) * ctx.n)))
                grad_loss = grad_output * above / ((1.0 - ctx.alpha) * ctx.n)
                return grad_loss * torch.ones(ctx.n, dtype=torch.float64)

        result = CVaRRUFunc.apply(loss)
        return result
    else:
        return float(cvar_val)


def ru_objective(v, loss, alpha):
    """Rockafellar-Uryasev objective: v + (1/(1-alpha)) E[max(loss - v, 0)].

    Args:
        v: float or 0-d tensor
        loss: 1-D array or tensor
        alpha: float in [0, 1)

    Returns:
        float or 0-d torch tensor

    Raises:
        ValueError: for invalid alpha or non-1-D loss
    """
    is_torch = torch is not None and isinstance(loss, torch.Tensor)
    is_torch_v = torch is not None and isinstance(v, torch.Tensor)
    is_torch = is_torch or is_torch_v

    if is_torch:
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float64)
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float64)

        loss_np = loss.detach().cpu().numpy() if loss.requires_grad else loss.numpy()
        v_np = float(v.detach().cpu().numpy()) if v.requires_grad or is_torch_v else float(v)
    else:
        loss_np = np.asarray(loss, dtype=np.float64)
        v_np = float(v)

    if loss_np.ndim != 1:
        raise ValueError(f"loss must be 1-D, got shape {loss_np.shape}")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    excess = np.maximum(loss_np - v_np, 0.0)
    obj_val = v_np + np.mean(excess) / (1.0 - alpha)

    if is_torch:
        class RUObjectiveFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, v_t, loss_t):
                ctx.v_np = v_np
                ctx.n = loss_np.size
                ctx.alpha = alpha
                return torch.tensor(obj_val, dtype=torch.float64)

            @staticmethod
            def backward(ctx, grad_output):
                above = (loss_np > ctx.v_np).astype(np.float64)
                grad_v = grad_output * (1.0 - (above.sum() / ((1.0 - ctx.alpha) * ctx.n)))
                grad_loss = grad_output * above / ((1.0 - ctx.alpha) * ctx.n)
                return grad_v, grad_loss * torch.ones(ctx.n, dtype=torch.float64)

        result = RUObjectiveFunc.apply(v, loss)
        return result
    else:
        return float(obj_val)


def value_at_risk(loss, alpha):
    """Value-at-Risk: lower empirical quantile at level alpha.

    Returns: sorted[max(ceil(n*alpha), 1) - 1]

    Args:
        loss: 1-D array
        alpha: float in [0, 1)

    Returns:
        float

    Raises:
        ValueError: for invalid alpha, empty sample, or NaN values
    """
    loss_np = np.asarray(loss, dtype=np.float64)

    if loss_np.ndim != 1:
        raise ValueError(f"loss must be 1-D, got shape {loss_np.shape}")
    if loss_np.size == 0:
        raise ValueError("loss cannot be empty")
    if not np.all(np.isfinite(loss_np)):
        raise ValueError("loss contains non-finite values")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    n = loss_np.size
    idx = max(int(np.ceil(n * alpha)) - 1, 0)
    idx = min(idx, n - 1)
    return float(np.sort(loss_np)[idx])


def entropic_risk(pnl, gamma):
    """Entropic risk measure: (1/gamma) log E[exp(-gamma pnl)].

    Computed with log-sum-exp for numerical stability.

    Args:
        pnl: 1-D array or tensor, shape (n,)
        gamma: float > 0

    Returns:
        float (numpy) or 0-d float64 tensor with autograd (torch)

    Raises:
        ValueError: for gamma <= 0, empty sample, or NaN values
    """
    is_torch = torch is not None and isinstance(pnl, torch.Tensor)

    if is_torch:
        pnl_np = pnl.detach().cpu().numpy() if pnl.requires_grad else pnl.numpy()
    else:
        pnl_np = np.asarray(pnl, dtype=np.float64)

    if pnl_np.ndim != 1:
        raise ValueError(f"pnl must be 1-D, got shape {pnl_np.shape}")
    if pnl_np.size == 0:
        raise ValueError("pnl cannot be empty")
    if not np.all(np.isfinite(pnl_np)):
        raise ValueError("pnl contains non-finite values")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be positive and finite, got {gamma}")

    # Use log-sum-exp for stability
    log_mean_exp = np.logaddexp.reduce(-gamma * pnl_np) - np.log(pnl_np.size)
    rho = log_mean_exp / gamma

    if is_torch:
        class EntropicRiskFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, pnl_t):
                ctx.gamma = gamma
                ctx.n = pnl_np.size
                return torch.tensor(rho, dtype=torch.float64)

            @staticmethod
            def backward(ctx, grad_output):
                e = np.exp(-ctx.gamma * pnl_np)
                softmax = e / e.sum()
                return grad_output * torch.tensor(-softmax, dtype=torch.float64)

        result = EntropicRiskFunc.apply(pnl)
        return result
    else:
        return float(rho)
