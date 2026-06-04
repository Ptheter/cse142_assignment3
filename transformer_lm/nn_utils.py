"""Numerically-stable softmax, activation functions, and cross-entropy loss."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of the same shape summing to 1 along ``dim``.
    """

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    x_exp = torch.exp(x_shifted)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)

    return x_exp / x_sum


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    
    sigmoid_x = 1 / (1 + torch.exp(-x))

    return x * sigmoid_x


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """
    # max(l_i)
    max_logits = torch.max(logits, dim=-1, keepdim=True).values

    # l_i,j - max(l_i)
    shifted_logits = logits - max_logits

    # log(sum_j exp(l_i,j - max(l_i)))
    log_sum_exp = torch.log(
        torch.sum(torch.exp(shifted_logits), dim=-1)
    )

    # max(l_i) + log(sum(...))
    normalization_term = max_logits.squeeze(-1) + log_sum_exp

    # l_i,t_i
    target_logits = torch.gather(
        logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # full loss expression
    loss = normalization_term - target_logits

    # mean over B*T positions
    return torch.mean(loss)
