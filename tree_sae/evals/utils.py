"""Small helpers shared by the SAEBench-derived evaluations."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TypeVar

import torch
from torch import Tensor
from tqdm.autonotebook import tqdm

from ..analysis.activations import dense_from_sparse, sae_sparse_activations
from ..models import BaseSAE

T = TypeVar("T")


def batchify(data: Sequence[T] | Tensor, batch_size: int, show_progress: bool = False) -> Generator:
    """Yield consecutive slices of ``batch_size`` elements."""
    for i in tqdm(range(0, len(data), batch_size), disable=not show_progress):
        yield data[i : i + batch_size]


def get_sae_acts(
    acts: Tensor,
    sae: BaseSAE,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
    convert_to_cpu: bool = False,
    verbose: bool = True,
) -> Tensor:
    """Dense ``(..., d_sae)`` feature activations of ``acts``."""
    indices, values = sae_sparse_activations(sae, acts, batch_size, device, verbose=verbose)
    return dense_from_sparse(sae.cfg.d_sae, indices, values, device="cpu" if convert_to_cpu else device)
