"""Shared building blocks for all sparse autoencoders in this package.

Every SAE maps a model activation ``x`` to sparse feature activations ``f(x)`` and back:

    f(x)  = sigma(W_enc (x - b))          (Eq. 1 of the paper)
    x_hat = W_dec f(x) + b                (Eq. 2)

The SAEs are ``transformer_lens`` ``HookedRootModule``s, so their intermediate activations
(``hook_sae_acts_post`` etc.) can be cached or patched like any TransformerLens model, and they
expose ``cfg``, ``W_enc``, ``W_dec`` and ``b_dec`` with the same conventions as SAELens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import einops
import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from transformer_lens.hook_points import HookedRootModule, HookPoint


class TopK(NamedTuple):
    """Values and indices of the active (or selected) features of each token."""

    values: Tensor
    indices: Tensor


class Stats(NamedTuple):
    """Per-token statistics used to undo input standardisation."""

    mean: Tensor
    std: Tensor


class TrainingOutput(NamedTuple):
    """Everything a training step needs from one forward pass.

    Single-level SAEs return one-element lists; the Tree SAE returns one entry per privilege
    layer (cumulative reconstructions, Eq. 7).
    """

    topk: TopK
    recons: list[Tensor]
    auxk_recons: list[Tensor | None]
    dead_fraction: list[Tensor]
    extra_loss: Tensor | float


@dataclass
class SAEConfig:
    d_in: int
    d_sae: int
    hook_name: str
    dead_steps_threshold: int
    dead_threshold: float
    auxk: int | None
    standardize: bool


def standardize(x: Tensor, eps: float = 1e-5) -> tuple[Tensor, Stats]:
    """Standardise each token to zero mean and unit variance across the model dimension."""
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, Stats(mu, std)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
    """Weiszfeld's algorithm over all leading dimensions; used to initialise ``b_dec``."""
    points = einops.rearrange(points, "... d -> (...) d")
    curr = points.mean(dim=0)
    for _ in range(max_iter):
        prev = curr
        weights = 1 / torch.norm(points - curr, dim=1)
        weights /= weights.sum()
        curr = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(curr - prev) < tol:
            break
    return curr


def unit_norm_decoder(decoder: Linear) -> None:
    """Rescale every decoder vector (column of ``decoder.weight``) to unit norm."""
    decoder.weight.data /= decoder.weight.data.norm(dim=0)


@torch.no_grad()
def remove_parallel_decoder_grad(decoder: Linear) -> None:
    """Project out the gradient component parallel to each (unit-norm) decoder vector."""
    if decoder.weight.grad is None:
        return
    parallel = einops.einsum(decoder.weight.grad, decoder.weight, "d_in d_sae, d_in d_sae -> d_sae")
    decoder.weight.grad -= einops.einsum(parallel, decoder.weight, "d_sae, d_in d_sae -> d_in d_sae")


_IDENTITY = torch.nn.Identity()


@contextmanager
def hooks_disabled(sae: HookedRootModule):
    """Temporarily replace every hook point of ``sae`` with an identity module."""
    try:
        for name in sae.hook_dict:
            setattr(sae, name, _IDENTITY)
        yield
    finally:
        for name, hook in sae.hook_dict.items():
            setattr(sae, name, hook)


class BaseSAE(ABC, HookedRootModule):
    """Linear encoder/decoder with a shared pre-encoder bias ``b`` and dead-feature tracking."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_name: str,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
    ) -> None:
        """
        Args:
            d_in: Dimension of the model activations.
            d_sae: Number of SAE features.
            hook_name: TransformerLens hook the SAE is trained on.
            dead_steps_threshold: A feature is dead if it has not fired for this many steps.
            dead_threshold: A feature fires on a token if its activation exceeds this value.
            auxk: Number of dead features used by the auxiliary loss (``None`` disables it).
            standardize: Standardise every input token before encoding.
        """
        super().__init__()
        self.cfg = SAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=dead_threshold,
            auxk=auxk,
            standardize=standardize,
        )
        self.use_error_term = False
        self.encoder = Linear(d_in, d_sae, bias=False)
        self.decoder = Linear(d_sae, d_in, bias=False)
        self.b_dec = Parameter(torch.zeros(d_in))
        # Steps since each feature last fired.
        self.register_buffer("steps_since_fired", torch.zeros(d_sae, dtype=torch.long))

        # Tied initialisation with unit-norm decoder vectors.
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        unit_norm_decoder(self.decoder)

        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        self.setup()

    @abstractmethod
    def encode(self, x: Tensor) -> tuple:
        """Return ``(latents, ...)``; the first element is the dense feature activation tensor."""

    @abstractmethod
    def forward_training(self, x: Tensor) -> TrainingOutput: ...

    def decode(self, latents: Tensor, stats: Stats | None = None) -> Tensor:
        recons = (latents @ self.decoder.weight.T) + self.b_dec
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return self.hook_sae_recons(recons)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x``; with ``use_error_term`` the output equals ``x`` exactly but gradients
        flow through the SAE features (as in SAELens)."""
        latents, *_, stats = self._encode_with_stats(x)
        sae_out = self.decode(latents, stats)
        if self.use_error_term:
            with torch.no_grad():
                with hooks_disabled(self):
                    clean_latents, *_, clean_stats = self._encode_with_stats(x)
                    clean_recons = self.decode(clean_latents, clean_stats)
                error = x - clean_recons
            error.requires_grad_()
            sae_out = sae_out + self.hook_sae_error(error)
        return self.hook_sae_output(sae_out)

    def _encode_with_stats(self, x: Tensor) -> tuple[Tensor, Stats | None]:
        out = self.encode(x)
        return out[0], out[-1]

    def _preprocess(self, x: Tensor) -> tuple[Tensor, Stats | None]:
        x = self.hook_sae_input(x)
        stats = None
        if self.cfg.standardize:
            x, stats = standardize(x)
        return x, stats

    def update_steps_since_fired(self, topk: TopK) -> None:
        fired = torch.zeros_like(self.steps_since_fired)
        fired.scatter_add_(
            dim=0,
            index=topk.indices.reshape(-1),
            src=(topk.values > self.cfg.dead_threshold).to(fired.dtype).reshape(-1),
        )
        self.steps_since_fired *= 1 - fired.clamp(max=1)
        self.steps_since_fired += 1

    def dead_fraction_and_auxk(self, hidden_pre: Tensor) -> tuple[Tensor, TopK | None]:
        """Fraction of dead features and the ``auxk`` largest dead pre-activations (Gao et al.)."""
        dead_mask = self.steps_since_fired >= self.cfg.dead_steps_threshold
        hidden_pre = hidden_pre * dead_mask
        dead = torch.sum(dead_mask, dtype=torch.float32).detach() / self.cfg.d_sae
        auxk = None
        if self.cfg.auxk is not None:
            values, indices = torch.topk(hidden_pre, k=self.cfg.auxk, sorted=False)
            auxk = TopK(values, indices)
        return dead, auxk

    def decode_auxk(self, auxk: TopK | None, like: Tensor) -> Tensor | None:
        """Reconstruct the residual from the selected dead features (no de-standardisation)."""
        if auxk is None:
            return None
        auxk_latents = torch.zeros_like(like)
        auxk_latents.scatter_(-1, auxk.indices, torch.relu(auxk.values))
        return self.decode(auxk_latents)

    @property
    def W_enc(self) -> Tensor:
        return self.encoder.weight.T

    @property
    def W_dec(self) -> Tensor:
        return self.decoder.weight.T

    @property
    def b_enc(self) -> Tensor:
        return -self.b_dec
