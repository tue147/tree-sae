"""Geometry of the child feature subspace (Section 6.1, Fig. 9).

The activations on which any descendant of a parent fires are projected on their top-2 principal
components, colored by which child fires, and the child feature vectors are drawn in the same plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from ..models import TreeSAE


def descendants(sae: TreeSAE, parent: int) -> list[int]:
    """All features below ``parent`` in the Tree SAE structure (breadth-first, unique, sorted)."""
    found: set[int] = set()
    queue = [parent]
    while queue:
        children = sae.children_of(queue.pop(0))
        queue.extend(children)
        found.update(children)
    return sorted(found)


def label_by_child(indices: Tensor, children: Sequence[int]) -> tuple[Tensor, Tensor]:
    """Rows on which any of ``children`` fires, and for each such row the first child in its top-k list."""
    is_child = torch.isin(indices, torch.as_tensor(list(children), device=indices.device))
    mask = is_child.any(dim=1)
    k = indices.shape[1]
    positions = torch.arange(k, device=indices.device).expand(int(mask.sum()), k)
    first = torch.where(is_child[mask], positions, torch.full_like(positions, k)).min(dim=1).values
    return mask, indices[mask][torch.arange(len(first)), first]


def fit_child_pca(acts: Tensor, indices: Tensor, children: Sequence[int], n_components: int = 2) -> PCA:
    mask, _ = label_by_child(indices, children)
    if mask.sum() < 2:
        raise ValueError("fewer than two tokens activate the children")
    return PCA(n_components=n_components).fit(acts[mask].float().cpu().numpy())


def plot_child_geometry(
    sae: TreeSAE,
    acts: Tensor,
    indices: Tensor,
    parent: int,
    vectors: Literal["encoder", "decoder"] = "encoder",
    vector_length: float = 10.0,
    min_points: int = 0,
    ax=None,
    title: str | None = None,
):
    """Scatter the child-activating tokens in their top-2 PCA plane with the child feature vectors.

    Args:
        indices: ``(n_tokens, k)`` active feature ids of ``acts`` (inactive entries padded to ``d_sae``).
        vectors: Which feature vectors to draw: encoder vectors (as in the paper) or decoder vectors.
        vector_length: Length of the drawn directions, in PCA units.
        min_points: Only draw children labelling more than this many points.
    """
    import matplotlib.pyplot as plt

    children = descendants(sae, parent)
    pca = fit_child_pca(acts, indices, children)
    mask, labels = label_by_child(indices, children)
    coords = pca.transform(acts[mask].float().cpu().numpy())
    labels = labels.cpu().numpy()

    ids, counts = np.unique(labels, return_counts=True)
    ids = list(ids[np.argsort(-counts)])
    cmap = plt.get_cmap("tab10", max(len(ids), 1))
    colors = {fid: cmap(i % cmap.N) for i, fid in enumerate(ids)}

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=[colors[f] for f in labels], s=100, alpha=0.4, edgecolor="none", zorder=2)

    weights = sae.W_dec if vectors == "decoder" else sae.W_enc.T
    handles, names = [], []
    for fid in ids:
        if (labels == fid).sum() <= min_points:
            continue
        direction = pca.components_ @ weights[fid].detach().float().cpu().numpy()
        norm = np.linalg.norm(direction)
        if np.isfinite(norm) and norm > 1e-12:
            end = direction / norm * vector_length
            ax.plot([0, end[0]], [0, end[1]], linestyle="--", linewidth=3, color=colors[fid], zorder=3)
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[fid], markersize=8))
        names.append(f"child {int(fid)}")
    if handles:
        ax.legend(handles, names, loc="best", fontsize=18)
    ax.set_xlabel("PC1", fontsize=25)
    ax.set_ylabel("PC2", fontsize=25)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(title or f"Children of feature {parent}", fontsize=28)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    return ax, pca
