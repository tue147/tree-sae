"""Hierarchy metric (Section 5.2): do parent/child pairs satisfy the reconstruction condition?

For a sample of parents, children are found either with Masked Cosine Similarity (MCS; any SAE)
or read from the Tree SAE structure. A linear probe trained to detect each child's activation
gives the child concept direction ``d*_c``; a pair is *hierarchical* when both the parent and the
child decoder vectors are among the top-5 features most similar to ``d*_c`` (Eq. 5).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor
from tqdm import tqdm

from ..models import BaseSAE, TreeSAE
from .probing import LinearProbe, train_multi_probe


@torch.no_grad()
def mcs_children(
    indices: Tensor,
    values: Tensor,
    d_sae: int,
    device: str | torch.device,
    n_children: int = 20,
    binary: bool = True,
    scaled: bool = False,
    value_threshold: float = 1e-3,
    child_batch_size: int = 128,
    parent_batch_size: int = 128,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """Top ``n_children`` children of every feature by Masked Cosine Similarity.

    ``MCS(p, c)`` is the cosine similarity between the activations of ``p`` and ``c`` over the
    tokens where ``c`` is active (Bricken et al., 2023). Variants (Appendix I):

    * ``binary``: activations are replaced by 1 when active, so ``MCS = sqrt(S_cov)`` (Eq. 4).
    * ``scaled``: the score is multiplied by ``min(max_c, max_p) / max_p``, where ``max_*`` are the
      maximum activations of both features (Bussmann et al., 2025).

    The paper uses ``binary=True, scaled=False``.

    Args:
        indices, values: ``(n_tokens, k)`` sparse feature activations; entries with
            ``value <= value_threshold`` (or index ``>= d_sae``) are ignored.

    Returns:
        ``(child_ids, scores)``, both ``(d_sae, n_children)`` on CPU: the best children of every
        parent, best first (a feature is never its own child).
    """
    device = torch.device(device)
    indices = indices.to(device, dtype=torch.long)
    orig_dtype = values.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    values = values.to(device, dtype=compute_dtype)
    n_tokens = indices.shape[0]
    n_children = min(n_children, d_sae)

    active = values > value_threshold
    rows = active.nonzero(as_tuple=False)[:, 0]
    feats = indices[active]
    vals = values[active]
    in_range = feats < d_sae
    rows, feats, vals = rows[in_range], feats[in_range], vals[in_range]

    if scaled:
        max_act = torch.zeros((d_sae,), device=device, dtype=compute_dtype)
        max_act.scatter_reduce_(dim=0, index=feats, src=vals, reduce="amax", include_self=True)
        max_act = max_act.clamp_min(1e-12)
    else:
        max_act = torch.ones((d_sae,), device=device, dtype=compute_dtype)

    # Parent-side entries sorted by feature, with slice offsets per feature.
    order = torch.argsort(feats)
    p_feat, p_row, p_val = feats[order], rows[order], vals[order]
    starts = torch.zeros((d_sae + 1,), dtype=torch.long, device=device)
    starts[1:] = torch.cumsum(torch.bincount(p_feat, minlength=d_sae), dim=0)
    # Child-side entries sorted by feature.
    order = torch.argsort(feats)
    c_feat, c_row, c_val = feats[order], rows[order], vals[order]
    if binary:
        c_val = (c_val > 0).to(compute_dtype)
        p_val = (p_val > 0).to(compute_dtype)

    best_scores = torch.full((n_children, d_sae), float("-inf"), dtype=compute_dtype)
    best_ids = torch.full((n_children, d_sae), -1, dtype=torch.long)

    for c_start in tqdm(range(0, d_sae, child_batch_size), disable=not verbose, desc="MCS children"):
        c_end = min(c_start + child_batch_size, d_sae)
        n_batch = c_end - c_start
        left = torch.searchsorted(c_feat, torch.tensor(c_start, dtype=c_feat.dtype, device=device))
        right = torch.searchsorted(c_feat, torch.tensor(c_end, dtype=c_feat.dtype, device=device))
        if right - left == 0:  # none of these features is ever active
            continue
        r_b = c_row[left:right]
        cols = c_feat[left:right] - c_start
        v_b = c_val[left:right]

        child_norm_sq = torch.zeros((n_batch,), device=device, dtype=compute_dtype)
        child_norm_sq.scatter_add_(0, cols, v_b * v_b)
        child_norm = torch.sqrt(child_norm_sq).clamp_min(1e-12)
        if scaled:
            child_max = torch.zeros((n_batch,), device=device, dtype=compute_dtype)
            child_max.scatter_reduce_(0, cols, v_b, reduce="amax", include_self=True)
        else:
            child_max = torch.ones((n_batch,), device=device, dtype=compute_dtype)

        coords = torch.stack([r_b, cols], dim=0)
        child_acts_t = torch.sparse_coo_tensor(coords, v_b, size=(n_tokens, n_batch), device=device).coalesce()
        child_mask_t = torch.sparse_coo_tensor(coords, torch.ones_like(v_b), size=(n_tokens, n_batch), device=device)
        child_acts_t, child_mask_t = child_acts_t.transpose(0, 1), child_mask_t.coalesce().transpose(0, 1)

        for p_start in range(0, d_sae, parent_batch_size):
            p_end = min(p_start + parent_batch_size, d_sae)
            n_par = p_end - p_start
            parent_acts = torch.zeros((n_tokens, n_par), device=device, dtype=compute_dtype).float()
            s, e = int(starts[p_start].item()), int(starts[p_end].item())
            if e > s:
                parent_acts.view(-1).index_add_(0, p_row[s:e] * n_par + (p_feat[s:e] - p_start), p_val[s:e])

            # Cosine similarity restricted to the tokens where the child is active.
            numerator = torch.sparse.mm(child_acts_t, parent_acts)
            parent_norm_sq = torch.sparse.mm(child_mask_t, parent_acts.pow(2))
            cosine = numerator / (child_norm.unsqueeze(1) * torch.sqrt(parent_norm_sq).clamp_min(1e-12) + 1e-10)
            max_p = max_act[p_start:p_end].unsqueeze(0)
            score = (torch.minimum(child_max.unsqueeze(1), max_p) * cosine) / max_p

            child_ids = torch.arange(c_start, c_end, device=device)
            parent_ids = torch.arange(p_start, p_end, device=device)
            score = score.masked_fill(child_ids.unsqueeze(1).eq(parent_ids.unsqueeze(0)), float("-inf"))

            # Merge this (children x parents) block into the running per-parent top-k.
            score = score.to("cpu", dtype=orig_dtype)
            all_scores = torch.cat([best_scores[:, p_start:p_end], score], dim=0)
            all_ids = torch.cat(
                [best_ids[:, p_start:p_end], torch.arange(c_start, c_end).unsqueeze(1).expand(score.shape)], dim=0
            )
            top_scores, top_rows = torch.topk(all_scores, k=n_children, dim=0)
            best_scores[:, p_start:p_end] = top_scores
            best_ids[:, p_start:p_end] = all_ids.gather(0, top_rows)
    return best_ids.T, best_scores.T


def tree_children(sae: TreeSAE, parent: int) -> list[int]:
    """Children of ``parent`` in the Tree SAE structure (all layers, in layer order)."""
    return sae.children_of(parent)


def train_child_probes(
    children: list[int],
    indices: Tensor,
    acts: Tensor,
    num_epochs: int = 50,
    batch_size: int = 4096,
    device: str | torch.device = "cuda",
) -> LinearProbe:
    """One logistic-regression probe per child, detecting the tokens on which the child is active."""
    labels = torch.stack([(indices == child).any(dim=1) for child in children], dim=-1).long()
    with torch.enable_grad():
        return train_multi_probe(
            acts.float(), labels, num_epochs=num_epochs, batch_size=batch_size, show_progress=False, device=device
        )


def probe_similarity(probe_weight: Tensor, W_dec: Tensor, device: str | torch.device) -> Tensor:
    """Similarity of every decoder vector with a probe direction.

    Both matrices are divided by their largest row norm; with unit-norm decoder vectors this is
    the cosine similarity.
    """

    def normalize(x: Tensor) -> Tensor:
        return x / x.norm(2, dim=1).max()

    return (normalize(W_dec.to(device)) @ normalize(probe_weight.to(device)).T).squeeze(1)


@dataclass
class HierarchyPairs:
    """Parent/child pairs found by one procedure and the probe similarities of each child."""

    pairs: list[tuple[int, int]] = field(default_factory=list)
    similarities: list[tuple[float, float]] = field(default_factory=list)  # (parent, child) similarity
    top_features: list[Tensor] = field(default_factory=list)  # 50 features most similar to d*_c

    def score(self, top: int = 5) -> float:
        """Fraction of pairs whose parent and child are both among the ``top`` most similar features."""
        hits = [p in tops[:top] and c in tops[:top] for (p, c), tops in zip(self.pairs, self.top_features)]
        return sum(hits) / len(hits) if hits else float("nan")


@torch.no_grad()
def run_hierarchy_eval(
    sae: BaseSAE,
    acts: Tensor,
    indices: Tensor,
    values: Tensor,
    device: str | torch.device,
    n_parents: int = 2000,
    dense_ratio: float = 0.5,
    max_children: int = 5,
    min_child_count: int = 100,
    binary: bool = True,
    scaled: bool = False,
    probe_epochs: int = 50,
    probe_batch_size: int = 4096,
    seed: int | None = 42,
) -> dict[str, HierarchyPairs]:
    """Evaluate MCS pairs (and, for a Tree SAE, tree-structure pairs) of randomly sampled parents.

    Parents are drawn with replacement from the ``dense_ratio`` most frequently active features.
    Each parent keeps at most ``max_children`` children that fire on at least ``min_child_count``
    tokens; for a Tree SAE both procedures keep the same number of children per parent.

    Args:
        acts: ``(n_tokens, d_model)`` LLM activations.
        indices, values: ``(n_tokens, k)`` sparse SAE activations of ``acts``.

    Returns:
        ``{"mcs": HierarchyPairs, "tree": HierarchyPairs}`` (``"tree"`` is empty for other SAEs).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    d_sae = sae.cfg.d_sae
    indices = indices.clone()
    indices[values < 1e-3] = d_sae  # pad inactive entries
    counts = torch.bincount(indices.flatten(), minlength=d_sae + 1)
    frequencies = counts.clone()
    frequencies[d_sae] = -1
    frequencies = frequencies / (indices.shape[0] * indices.shape[1])
    dense = frequencies.topk(k=int(d_sae * dense_ratio), largest=True, sorted=True).indices

    mcs_ids, _ = mcs_children(indices, values, d_sae, device, n_children=20, binary=binary, scaled=scaled, verbose=True)

    # Sampled parents may repeat; every draw is evaluated.
    selected: list[tuple[int, list[int], list[int]]] = []
    for _ in range(n_parents):
        parent = int(dense[random.randint(0, len(dense) - 1)])
        n_keep = max_children
        mcs = [c for c in mcs_ids[parent].tolist() if c >= 0 and int(counts[c]) >= min_child_count]
        n_keep = min(n_keep, len(mcs))
        tree = []
        if isinstance(sae, TreeSAE):
            tree = [c for c in tree_children(sae, parent) if counts[c] >= min_child_count]
            n_keep = min(n_keep, len(tree))
            tree = tree[:n_keep]
        selected.append((parent, mcs[:n_keep], tree))

    all_children = [c for _, mcs, tree in selected for c in (*mcs, *tree)]
    probe = train_child_probes(all_children, indices, acts, probe_epochs, probe_batch_size, device).to(sae.W_dec.dtype)

    results = {"mcs": HierarchyPairs(), "tree": HierarchyPairs()}
    i = 0
    for parent, mcs, tree in selected:
        for key, children in (("mcs", mcs), ("tree", tree)):
            for child in children:
                sim = probe_similarity(probe.weights[i].unsqueeze(0), sae.W_dec, device).cpu()
                results[key].pairs.append((parent, child))
                results[key].similarities.append((sim[parent].item(), sim[child].item()))
                results[key].top_features.append(sim.topk(k=50).indices)
                i += 1
    return results
