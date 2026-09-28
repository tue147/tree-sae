"""Reconstruction quality (Appendix D.2): variance explained and downstream cross-entropy loss when the
LLM activation is replaced by the SAE reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from ..models import BaseSAE


def _sum_over_dim_mean_over_tokens(x: Tensor) -> Tensor:
    return torch.mean(torch.sum(x, dim=-1))


def variance_explained(x: Tensor, recons: Tensor) -> float:
    """``1 - SS_res / SS_tot`` where ``SS_tot`` is taken around each token's own mean."""
    ss_res = _sum_over_dim_mean_over_tokens((x - recons) ** 2)
    ss_tot = _sum_over_dim_mean_over_tokens((x - torch.mean(x, dim=-1, keepdim=True)) ** 2) + 1e-10
    return (1 - ss_res / ss_tot).item()


@dataclass
class ReconstructionResult:
    mse: float
    variance_explained: float
    downstream_ce_loss: float


@torch.no_grad()
def evaluate_reconstruction(
    model: HookedTransformer, sae: BaseSAE, tokens: Tensor, batch_size: int, device: str | torch.device
) -> ReconstructionResult:
    """Splice the SAE into ``model`` at its hook and average the metrics over batches of ``tokens``."""
    sae.use_error_term = False
    sae = sae.to(device)
    mse, r2, ce = [], [], []

    def splice(acts: Tensor, hook) -> Tensor:
        _, cache = sae.run_with_cache(acts)
        recons = cache["hook_sae_recons"]
        mse.append(_sum_over_dim_mean_over_tokens((recons - acts) ** 2).item())
        r2.append(variance_explained(acts, recons))
        return recons

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, splice)], reset_hooks_end=True):
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start : start + batch_size]
            ce.append(model.loss_fn(model(batch), batch).cpu().float())
    return ReconstructionResult(float(np.mean(mse)), float(np.mean(r2)), float(np.mean(ce)))
