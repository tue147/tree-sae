"""Top-k SAE (Gao et al., 2024): keep the k largest pre-activations of every token."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import BaseSAE, Stats, TopK, TrainingOutput


class TopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_name: str,
        k: int,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
    ) -> None:
        super().__init__(d_in, d_sae, hook_name, dead_steps_threshold, dead_threshold, auxk, standardize)
        self.k = k

    def encode(self, x: Tensor) -> tuple[Tensor, TopK, TopK | None, Tensor, Stats | None]:
        x, stats = self._preprocess(x)
        hidden_pre = self.hook_sae_acts_pre(self.encoder(x - self.b_dec))

        values, indices = torch.topk(hidden_pre, k=self.k, sorted=False)
        topk = TopK(torch.relu(values), indices)
        latents = torch.zeros_like(hidden_pre)
        latents = self.hook_sae_acts_post(latents.scatter_(-1, topk.indices, topk.values))

        self.update_steps_since_fired(topk)
        dead, auxk = self.dead_fraction_and_auxk(hidden_pre)
        return latents, topk, auxk, dead, stats

    def forward_training(self, x: Tensor) -> TrainingOutput:
        latents, topk, auxk, dead, stats = self.encode(x)
        recons = self.decode(latents, stats)
        return TrainingOutput(topk, [recons], [self.decode_auxk(auxk, latents)], [dead], 0.0)
