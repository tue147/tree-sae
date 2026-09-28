"""Child feature diversity (Section 6.3, Table 3): how often siblings fire on the same token."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..models import TreeSAE


def children_by_parent(sae: TreeSAE, layer: int) -> dict[int, list[int]]:
    """``{parent: [global child ids]}`` for the features of ``layer`` (root children excluded)."""
    families: dict[int, list[int]] = {}
    start = sae.layer_start(layer)
    for i, parent in enumerate(sae.parent_index(layer).tolist()):
        if parent != sae.root_index(layer):
            families.setdefault(parent, []).append(start + i)
    return families


@torch.no_grad()
def sibling_cooccurrence(families: dict[int, list[int]], indices: Tensor) -> float:
    """Mean pairwise IoU ``|A n B| / |A u B|`` of the token sets of siblings (ordered pairs).

    Families whose IoU matrix contains NaN (a child that never fires together with an empty
    sibling) are skipped.
    """
    tokens = indices.reshape(-1, indices.shape[-1])
    total, n_pairs = 0.0, 0
    for children in families.values():
        if len(children) < 2:
            continue
        fires = torch.stack([(tokens == c).any(dim=1) for c in children], dim=1).float()
        both = fires.T @ fires
        freq = fires.sum(dim=0)
        iou = both / (freq.view(1, -1) + freq.view(-1, 1) - both)
        off_diagonal = iou.sum() - iou.diag().sum()
        if torch.isnan(off_diagonal):
            continue
        total += off_diagonal
        n_pairs += len(children) * (len(children) - 1)
    return float(total / n_pairs) if n_pairs > 0 else 0.0


def average_sibling_cooccurrence(sae: TreeSAE, indices: Tensor) -> float:
    """Table 3: sibling co-occurrence averaged over the privilege layers ``1..L-1``."""
    return float(np.mean([sibling_cooccurrence(children_by_parent(sae, l), indices) for l in range(1, sae.n_layers)]))
