"""Training losses (Eqs. 3, 7, 8 of the paper)."""

from __future__ import annotations

import torch
from torch import Tensor


def reconstruction_loss(x: Tensor, recons: Tensor, use_loss_var: bool = True) -> Tensor:
    """Mean squared error, divided by ``Var(x)`` when ``use_loss_var``."""
    mse = torch.mean((recons - x).pow(2))
    if use_loss_var:
        return mse / (torch.var(x) + 1e-8)
    return mse


def auxiliary_loss(x: Tensor, recons: Tensor, auxk_recons: Tensor | None, coef: float) -> Tensor | float:
    """AuxK loss (Gao et al., 2024): dead features reconstruct the residual ``x - recons``.

    The MSE is normalised by the residual's per-token variance, so its scale stays constant as the
    reconstruction improves during training.
    """
    if auxk_recons is None:
        return 0.0
    error = x - recons
    normalised_mse = (error - auxk_recons).pow(2).mean() / (error - error.mean(dim=-1, keepdim=True)).pow(2).mean()
    return coef * normalised_mse.nan_to_num(0)
