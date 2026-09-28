"""Standard ReLU SAE with an L1 sparsity penalty (Bricken et al., 2023)."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import BaseSAE, Stats, TopK, TrainingOutput


class ReLUSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_name: str,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
        l1_coef: float = 1 / 16,
    ) -> None:
        super().__init__(d_in, d_sae, hook_name, dead_steps_threshold, dead_threshold, auxk, standardize)
        self.l1_coef = l1_coef

    def encode(self, x: Tensor) -> tuple[Tensor, TopK, TopK | None, Tensor, Stats | None]:
        x, stats = self._preprocess(x)
        hidden_pre = self.hook_sae_acts_pre(self.encoder(x - self.b_dec))
        latents = self.hook_sae_acts_post(torch.relu(hidden_pre))

        # Express the active features in the same (values, indices) format as the TopK SAEs.
        max_active = torch.max(torch.sum(latents > self.cfg.dead_threshold, dim=-1)).item()
        values, indices = torch.topk(latents, k=max_active, sorted=False)
        topk = TopK(values, indices)

        self.update_steps_since_fired(topk)
        dead, auxk = self.dead_fraction_and_auxk(hidden_pre)
        return latents, topk, auxk, dead, stats

    def forward_training(self, x: Tensor) -> TrainingOutput:
        latents, topk, auxk, dead, stats = self.encode(x)
        recons = self.decode(latents, stats)
        l1_loss = torch.abs(topk.values).sum(dim=-1).mean() * self.l1_coef
        return TrainingOutput(topk, [recons], [self.decode_auxk(auxk, latents)], [dead], l1_loss)
