"""Matching Pursuit SAE (Costa et al., 2025): greedy residual pursuit over the decoder atoms."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import BaseSAE, Stats, TopK, TrainingOutput


class MPSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_name: str,
        k: int,
        dead_steps_threshold: int = 10_000_000,
        dead_threshold: float = 1e-3,
        standardize: bool = True,
        eps: float = 5e-3,
        use_pre_encoder_bias: bool = False,
    ) -> None:
        """
        Args:
            k: Maximum number of pursuit steps, i.e. the maximum number of active features per token;
                a token stops early once its support is stable or its residual norm is below ``eps``.
            use_pre_encoder_bias: Subtract/add ``b_dec`` around the pursuit (off in the paper).
        """
        super().__init__(d_in, d_sae, hook_name, dead_steps_threshold, dead_threshold, None, standardize)
        self.k = k
        self.eps = eps
        self.use_pre_encoder_bias = use_pre_encoder_bias

    def encode(self, x: Tensor) -> tuple[Tensor, TopK, TopK | None, Tensor, Stats | None]:
        x, stats = self._preprocess(x)
        if self.use_pre_encoder_bias:
            x = x - self.b_dec
        x = self.hook_sae_acts_pre(x)
        latents = self.hook_sae_acts_post(self._matching_pursuit(x))

        values, indices = torch.topk(latents, k=self.k, sorted=False)
        topk = TopK(values, indices)
        self.update_steps_since_fired(topk)
        dead, auxk = self.dead_fraction_and_auxk(latents)
        return latents, topk, auxk, dead, stats

    def _matching_pursuit(self, x: Tensor) -> Tensor:
        residual = x.clone()
        z = torch.zeros([*x.shape[:-1], self.cfg.d_sae], dtype=x.dtype, device=x.device)
        prev_support = torch.zeros_like(z).bool()
        done = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)

        for _ in range(self.k):
            if done.all():
                break
            values, indices = torch.max(torch.relu(residual @ self.decoder.weight), dim=-1, keepdim=True)
            step = torch.zeros_like(z)
            step.scatter_(-1, indices, values.to(step.dtype))
            z = torch.where(done.unsqueeze(-1), z, z + step)
            residual = torch.where(done.unsqueeze(-1), residual, residual - step @ self.decoder.weight.T)

            # A token has converged once its support stops changing or its residual is small.
            support = z != 0
            converged = (support == prev_support).all(dim=-1) | (residual.norm(dim=-1) < self.eps)
            done = done | converged
            prev_support = support
        return z

    def decode(self, latents: Tensor, stats: Stats | None = None) -> Tensor:
        recons = latents @ self.decoder.weight.T
        if self.use_pre_encoder_bias:
            recons = recons + self.b_dec
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return self.hook_sae_recons(recons)

    def forward_training(self, x: Tensor) -> TrainingOutput:
        latents, topk, _, dead, stats = self.encode(x)
        recons = self.decode(latents, stats)
        return TrainingOutput(topk, [recons], [None], [dead], 0.0)
