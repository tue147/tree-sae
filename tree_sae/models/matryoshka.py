"""Matryoshka SAE (Bussmann et al., 2025): nested prefixes of the dictionary each reconstruct x."""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

from .base import Stats, TrainingOutput
from .topk import TopKSAE


class MatryoshkaSAE(TopKSAE):
    def __init__(
        self,
        d_in: int,
        hook_name: str,
        features_per_layer: list[int],
        k: int,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
        use_loss_var: bool = True,
    ) -> None:
        """
        Args:
            features_per_layer: Number of features in each nested layer, e.g. ``[6144, 18432]``;
                the dictionary size is their sum and layer ``l``'s prefix covers layers ``1..l``.
            k: Total number of active features per token.
            use_loss_var: Divide every prefix MSE by ``Var(x)``.
        """
        super().__init__(
            d_in, sum(features_per_layer), hook_name, k, dead_steps_threshold, dead_threshold, auxk, standardize
        )
        assert len(features_per_layer) >= 2 and all(n > 0 for n in features_per_layer)
        self.features_per_layer = list(features_per_layer)
        # End index of every prefix except the full dictionary.
        self.prefix_ends = list(itertools.accumulate(features_per_layer))[:-1]
        self.use_loss_var = use_loss_var

    def decode_prefix(self, latents: Tensor, end: int, stats: Stats | None = None) -> Tensor:
        recons = (latents[..., :end] @ self.decoder.weight[..., :end].T) + self.b_dec
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return self.hook_sae_recons(recons)

    def prefix_loss(self, x: Tensor, latents: Tensor, stats: Stats | None) -> Tensor:
        """Sum of the reconstruction losses of all strict prefixes (the full one is added by the trainer)."""
        loss = 0.0
        for end in self.prefix_ends:
            mse = torch.mean((x - self.decode_prefix(latents, end, stats)) ** 2)
            loss += mse / (torch.var(x) + 1e-8) if self.use_loss_var else mse
        return loss

    def forward_training(self, x: Tensor) -> TrainingOutput:
        latents, topk, auxk, dead, stats = self.encode(x)
        recons = self.decode(latents, stats)
        auxk_recons = self.decode_auxk(auxk, latents)
        return TrainingOutput(topk, [recons], [auxk_recons], [dead], self.prefix_loss(x, latents, stats))
