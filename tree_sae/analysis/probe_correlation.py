"""Activation coverage vs. reconstruction condition for individual parents (Figs. 1, 10-13).

For a parent ``p`` and candidate children ``c``:

* activation coverage ``S_cov(p, c)``: fraction of ``c``'s tokens on which ``p`` also fires (Eq. 4);
* reconstruction score ``S_res(p, c) = min(<d*_c, d_c>, <d*_c, d_p>)`` (Eq. 5), where the child concept
  direction ``d*_c`` is the weight of a probe trained to detect ``c``'s activation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..evals.hierarchy import HierarchyPairs, probe_similarity, train_child_probes
from ..models import BaseSAE


def activation_coverage(indices: Tensor, parent: int, d_sae: int) -> Tensor:
    """``S_cov(parent, c)`` for every feature ``c`` (NaN for features that never fire).

    ``indices`` are ``(n_tokens, k)`` active feature ids, with inactive entries set to ``d_sae``.
    """
    counts = torch.bincount(indices.flatten(), minlength=d_sae + 1)[:d_sae].float()
    parent_rows = (indices == parent).any(dim=1)
    with_parent = torch.bincount(indices[parent_rows].flatten(), minlength=d_sae + 1)[:d_sae].float()
    return with_parent / counts


def children_by_coverage(
    indices: Tensor, parent: int, d_sae: int, n_children: int = 10, min_count: int = 100
) -> tuple[list[int], list[float]]:
    """Features with the highest activation coverage by ``parent`` that fire on ``>= min_count`` tokens."""
    coverage = activation_coverage(indices, parent, d_sae)
    counts = torch.bincount(indices.flatten(), minlength=d_sae + 1)[:d_sae]
    coverage[(counts < min_count) | torch.isnan(coverage)] = -1
    coverage[parent] = -1
    top = coverage.topk(n_children)
    keep = top.values >= 0
    return top.indices[keep].tolist(), top.values[keep].tolist()


@dataclass
class ChildReport:
    child: int
    coverage: float
    parent_similarity: float  # <d*_c, d_p>
    child_similarity: float  # <d*_c, d_c>
    parent_rank: int  # rank of the parent among all features by similarity to d*_c (1 = most similar)

    @property
    def reconstruction_score(self) -> float:
        return min(self.parent_similarity, self.child_similarity)


def probe_report(
    sae: BaseSAE,
    acts: Tensor,
    indices: Tensor,
    parent: int,
    children: list[int],
    device: str | torch.device,
    probe_epochs: int = 50,
    probe_batch_size: int = 4096,
) -> list[ChildReport]:
    """Train one probe per child and report coverage and similarities of the parent and child."""
    coverage = activation_coverage(indices, parent, sae.cfg.d_sae)
    probe = train_child_probes(children, indices, acts, probe_epochs, probe_batch_size, device).to(sae.W_dec.dtype)
    reports = []
    with torch.no_grad():
        for i, child in enumerate(children):
            sim = probe_similarity(probe.weights[i].unsqueeze(0), sae.W_dec, device).float().cpu()
            rank = int((torch.argsort(-sim) == parent).nonzero().item()) + 1
            reports.append(ChildReport(child, coverage[child].item(), sim[parent].item(), sim[child].item(), rank))
    return reports


def parent_similarity_by_child_rank(pairs: HierarchyPairs, n_children: int) -> np.ndarray:
    """Fig. 13: mean parent similarity of the 1st, 2nd, ... MCS child.

    ``pairs`` come from ``run_hierarchy_eval(..., max_children=n_children)``, where the children of
    each sampled parent are consecutive; only parents with exactly ``n_children`` children are used.
    """
    families: list[list[float]] = []
    current: list[float] = []
    current_parent = None
    for (parent, _), (parent_sim, _) in zip(pairs.pairs, pairs.similarities):
        if current and (parent != current_parent or len(current) == n_children):
            families.append(current)
            current = []
        current_parent = parent
        current.append(parent_sim)
    if current:
        families.append(current)
    return np.array([f for f in families if len(f) == n_children]).mean(axis=0)


@torch.no_grad()
def top_principal_direction(acts: Tensor) -> Tensor:
    """First principal component of ``acts`` (``n_samples, d``), computed in float64."""
    x = acts.double()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return v[0].float()


def features_aligned_with(direction: Tensor, W_dec: Tensor, k: int = 5) -> torch.return_types.topk:
    """Features whose decoder vectors have the largest ``|cos|`` with ``direction`` (e.g. the
    "PCA feature" of Sun et al., 2025, discussed in Section 2.2)."""
    sims = torch.nn.functional.cosine_similarity(W_dec.float().cpu(), direction.float().cpu().unsqueeze(0), dim=1)
    return sims.abs().topk(k)
