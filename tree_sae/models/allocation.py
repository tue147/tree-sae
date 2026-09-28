"""Greedy optimal child allocation (Algorithm 1 of the paper)."""

from __future__ import annotations

import heapq

import torch
from torch import Tensor


def greedy_allocation(capacities: Tensor, n_children: int, eligible: Tensor) -> Tensor:
    """Number of children ``k*_p`` per parent that maximises ``min_p C_p / k_p`` (Eq. 10).

    Repeatedly gives one child to the parent with the largest payoff ``C_p / (k_p + 1)`` using a
    max-heap, which is optimal by Theorem 4.1 and runs in ``O(n_children log n_parents)``.

    Args:
        capacities: ``C_p`` for every candidate parent, shape ``(n_parents,)``.
        n_children: Number of children ``s_l`` to distribute.
        eligible: Boolean mask of parents allowed to receive children. Parents with zero capacity
            are also skipped; if every eligible parent has zero capacity the children are spread
            uniformly over the eligible parents instead.

    Returns:
        Integer quotas of shape ``(n_parents,)`` that sum to ``n_children`` whenever any parent is
        eligible.
    """
    n_parents = capacities.numel()
    quotas = torch.zeros(n_parents, dtype=torch.long, device=capacities.device)
    if n_children <= 0 or n_parents == 0:
        return quotas

    caps = capacities.detach().to(torch.float).cpu().numpy()
    elig = eligible.detach().cpu().numpy()
    heap = [(-(caps[p] / 1.0), p) for p in range(n_parents) if elig[p] and caps[p] > 0]
    heapq.heapify(heap)

    if heap:
        for _ in range(n_children):
            _, p = heapq.heappop(heap)
            quotas[p] += 1
            heapq.heappush(heap, (-(caps[p] / (quotas[p].item() + 1.0)), p))
    else:
        eligible_ids = [p for p in range(n_parents) if elig[p]]
        if eligible_ids:
            base, remainder = divmod(n_children, len(eligible_ids))
            for p in eligible_ids:
                quotas[p] = base
            for j in range(remainder):
                quotas[eligible_ids[j]] += 1
    return quotas.to(capacities.device)
