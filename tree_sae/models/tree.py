"""Tree SAE (Section 4 of the paper).

The dictionary is split into ``L`` privilege layers. Every feature of layer ``l >= 1`` has one
parent, chosen among all features of layers ``< l`` or an imaginary always-active root. The
parent of the ``i``-th feature of layer ``l`` is stored in the allocation vector ``a_l``
(``parent_index[l][i]``; the root is encoded as the index ``layer_start(l)``).

* Activation coverage (Eq. 6): a feature may only fire on tokens where its parent fires. Each
  layer keeps its own top-``k_l`` among the features whose parent is active.
* Reconstruction condition (Eq. 7): layer ``l``'s cumulative reconstruction ``sum_{t<=l} x_hat_t``
  must reconstruct ``x`` on its own, exactly as the Matryoshka prefixes.
* Per-layer AuxK (Eq. 8): dead features of a layer, gated by their parents, reconstruct the
  residual of that layer's cumulative reconstruction.
* Dynamic allocation (Algorithms 1-2): every ``realloc_interval`` steps the dead children of each
  layer are moved to the parents with the largest capacity ``C_p``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .allocation import greedy_allocation
from .base import Stats, TopK, TrainingOutput
from .matryoshka import MatryoshkaSAE


class TreeSAE(MatryoshkaSAE):
    def __init__(
        self,
        d_in: int,
        hook_name: str,
        features_per_layer: list[int],
        k_per_layer: list[int],
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
        use_loss_var: bool = True,
        parent_eligibility_steps: int | None = None,
        realloc_interval: int = 3000,
        realloc_interval_growth: float = 2.0,
        max_realloc_interval: int = 10_000,
        root_reset_step: int | None = 50_000,
        root_init_frac: float = 0.0,
        aux_layers: tuple[int, ...] = (0,),
    ) -> None:
        """
        Args:
            features_per_layer: Number of features ``s_l`` of every privilege layer.
            k_per_layer: Number of active features of every privilege layer (sums to the L0).
            parent_eligibility_steps: A feature may receive children only if it fired within this
                many steps (defaults to ``dead_steps_threshold``).
            realloc_interval: Steps before the first reallocation; after every reallocation the
                interval is multiplied by ``realloc_interval_growth`` up to ``max_realloc_interval``.
            root_reset_step: At this step every dead child is attached to the root and dynamic
                allocation stops (``None`` disables the reset).
            root_init_frac: Minimum fraction of every layer's children attached to the root, both at
                initialisation and at every reallocation (Appendix G; 0 in the paper's GPT-2 runs).
            aux_layers: 0-based privilege layers that get the auxiliary loss.
        """
        assert len(k_per_layer) == len(features_per_layer)
        super().__init__(
            d_in,
            hook_name,
            features_per_layer,
            sum(k_per_layer),
            dead_steps_threshold,
            dead_threshold,
            auxk,
            standardize,
            use_loss_var,
        )
        self.k_per_layer = list(k_per_layer)
        self.n_layers = len(features_per_layer)
        # Layer l spans features [boundaries[l], boundaries[l + 1]).
        self.boundaries = [0, *self.prefix_ends, self.cfg.d_sae]

        # Per-layer training state. Layer 0 hangs from the root only, so it has no allocation.
        self.steps_since_fired_per_layer: dict[int, Tensor] = {}
        self.fired_in_batch: dict[int, Tensor] = {}
        self.capacity: dict[int, Tensor] = {}  # C_p of every candidate parent (+ root, last)
        for layer in range(1, self.n_layers):
            n_parents = self.layer_start(layer)
            self.register_buffer(
                f"parent_index_{layer}",
                torch.randint(0, n_parents + 1, (self.layer_size(layer),), dtype=torch.long),
            )
            self.steps_since_fired_per_layer[layer] = torch.zeros(self.layer_size(layer), dtype=torch.long)
            self.fired_in_batch[layer] = torch.zeros(self.layer_size(layer), dtype=torch.long)
            self.capacity[layer] = torch.zeros(n_parents + 1, dtype=torch.float)
        self.steps_since_fired_per_layer[0] = torch.zeros(self.layer_size(0), dtype=torch.long)
        self.fired_in_batch[0] = torch.zeros(self.layer_size(0), dtype=torch.long)

        self.parent_eligibility_steps = (
            dead_steps_threshold if parent_eligibility_steps is None else parent_eligibility_steps
        )
        self.initial_realloc_interval = realloc_interval
        self.realloc_interval = realloc_interval
        self.realloc_interval_growth = realloc_interval_growth
        self.max_realloc_interval = max_realloc_interval
        self.root_reset_step = root_reset_step
        self.root_init_frac = float(root_init_frac)
        self.aux_layers = set(aux_layers)
        self._steps = 0
        self._steps_since_realloc = 0
        self._root_reset_done = False
        self._enforce_root_minimum_at_init()

    # ------------------------------------------------------------------ structure helpers

    def layer_start(self, layer: int) -> int:
        return self.boundaries[layer]

    def layer_size(self, layer: int) -> int:
        return self.boundaries[layer + 1] - self.boundaries[layer]

    def parent_index(self, layer: int) -> Tensor:
        """Allocation vector ``a_l``: parent of every feature of ``layer`` (root = ``layer_start``)."""
        return getattr(self, f"parent_index_{layer}")

    def children_of(self, parent: int) -> list[int]:
        """Global ids of the children of ``parent`` across all layers (excluding root children)."""
        children = []
        for layer in range(1, self.n_layers):
            if parent >= self.layer_start(layer):  # parents always sit in an earlier layer
                continue
            index = self.parent_index(layer)
            children += ((index == parent).nonzero().squeeze(-1) + self.layer_start(layer)).tolist()
        return children

    def root_index(self, layer: int) -> int:
        return self.layer_start(layer)

    def _parent_pool_state(self, layer: int, state: dict[int, Tensor], device: torch.device) -> Tensor:
        """Concatenate a per-layer state over all parent layers ``< layer``."""
        return torch.cat([state[t].to(device) for t in range(layer)], dim=0)

    def _parent_active_mask(self, latents: Tensor, layer: int) -> Tensor:
        """Whether the parent of every feature of ``layer`` is active on each token."""
        if layer == 0:
            return torch.ones_like(latents[..., : self.layer_size(0)])
        n_parents = self.layer_start(layer)
        parents = torch.cat([latents[..., :n_parents], torch.ones_like(latents[..., :1])], dim=-1)
        return parents[..., self.parent_index(layer)] > self.cfg.dead_threshold

    # ------------------------------------------------------------------ forward

    def encode(self, x: Tensor) -> tuple[Tensor, TopK, list[TopK | None], list[Tensor], Stats | None]:
        x, stats = self._preprocess(x)
        hidden_pre = self.hook_sae_acts_pre(self.encoder(x - self.b_dec))

        # Activation coverage (Eq. 6): each layer picks its top-k among features whose parent fires.
        latents = torch.zeros_like(hidden_pre)
        for layer in range(self.n_layers):
            start, end = self.boundaries[layer], self.boundaries[layer + 1]
            mask = self._parent_active_mask(latents, layer)
            values, indices = torch.topk(hidden_pre[..., start:end] * mask, k=self.k_per_layer[layer])
            latents.scatter_(-1, indices + start, values)
        latents = self.hook_sae_acts_post(torch.relu(latents))
        topk = TopK(*torch.topk(latents, k=self.k, sorted=False))

        auxk_per_layer: list[TopK | None] = []
        dead_per_layer: list[Tensor] = []
        for layer in range(self.n_layers):
            start, end = self.boundaries[layer], self.boundaries[layer + 1]
            in_layer = (topk.indices >= start) & (topk.indices < end)
            parent_mask = self._parent_active_mask(latents, layer)
            self._update_steps_since_fired(layer, TopK(topk.values[in_layer], topk.indices[in_layer] - start))
            dead, auxk = self._dead_fraction_and_auxk(layer, hidden_pre[..., start:end], parent_mask)
            if auxk is not None:
                auxk = TopK(auxk.values, auxk.indices + start)
            auxk_per_layer.append(auxk)
            dead_per_layer.append(dead)
        return latents, topk, auxk_per_layer, dead_per_layer, stats

    def forward_training(self, x: Tensor) -> TrainingOutput:
        latents, topk, auxk_per_layer, dead_per_layer, stats = self.encode(x)
        recons = [self.decode_prefix(latents, end, stats) for end in self.prefix_ends]
        recons.append(self.decode(latents, stats))
        auxk_recons = [self.decode_auxk(auxk, latents) for auxk in auxk_per_layer]
        return TrainingOutput(topk, recons, auxk_recons, dead_per_layer, 0.0)

    def _update_steps_since_fired(self, layer: int, topk: TopK) -> None:
        steps = self.steps_since_fired_per_layer[layer].to(topk.indices.device)
        fired = torch.zeros_like(steps)
        fired.scatter_add_(
            dim=0,
            index=topk.indices.reshape(-1),
            src=(topk.values > self.cfg.dead_threshold).to(fired.dtype).reshape(-1),
        )
        self.fired_in_batch[layer] = fired.clamp(max=1)
        steps *= 1 - fired.clamp(max=1)
        steps += 1
        self.steps_since_fired_per_layer[layer] = steps

    def _dead_fraction_and_auxk(
        self, layer: int, hidden_pre: Tensor, parent_mask: Tensor
    ) -> tuple[Tensor, TopK | None]:
        """Dead fraction of ``layer`` and, for ``aux_layers``, its top-``auxk`` dead features
        restricted to tokens where their parent is active."""
        dead_mask = self.steps_since_fired_per_layer[layer] >= self.cfg.dead_steps_threshold
        dead = torch.sum(dead_mask, dtype=torch.float32).detach() / hidden_pre.shape[-1]
        if layer not in self.aux_layers or not self.cfg.auxk:
            return dead, None
        hidden_pre = hidden_pre * dead_mask * parent_mask.to(hidden_pre.dtype)
        values, indices = torch.topk(hidden_pre, k=min(self.cfg.auxk, hidden_pre.shape[-1]), sorted=False)
        return dead, TopK(values, indices)

    # ------------------------------------------------------------------ dynamic allocation

    @torch.no_grad()
    def update_allocation(self, loss: Tensor) -> None:
        """Accumulate capacities with this step's loss and reallocate when scheduled.

        Capacity rule: the step's total loss is split equally among all candidate parents that fired
        at least once in the batch, the root always counting as active.
        """
        device = self.encoder.weight.device
        loss = loss.detach().to(device=device, dtype=torch.float)
        self._steps += 1
        self._steps_since_realloc += 1
        for layer in range(1, self.n_layers):
            fired = self._parent_pool_state(layer, self.fired_in_batch, device).to(torch.float)
            fired = torch.cat([fired, torch.ones(1, device=device)], dim=0)
            self.capacity[layer] = self.capacity[layer].to(device)
            self.capacity[layer] += fired * (loss / fired.sum().clamp(min=1.0))
            # NOTE: the schedule is checked inside the layer loop (as in the code that produced the
            # released checkpoints), so deeper layers receive this step's loss after their reset.
            self._maybe_reset_or_reallocate()

    def _maybe_reset_or_reallocate(self) -> None:
        if self.root_reset_step is not None and not self._root_reset_done and self._steps >= self.root_reset_step:
            self.attach_dead_children_to_root()
            self._root_reset_done = True
        if self._steps_since_realloc >= self.realloc_interval and not self._root_reset_done:
            self.reallocate_dead_children()
            self._steps_since_realloc = 0
            self.realloc_interval = min(
                self.max_realloc_interval, int(self.realloc_interval * self.realloc_interval_growth)
            )

    def _dead_children(self, layer: int, device: torch.device) -> Tensor:
        return self.steps_since_fired_per_layer[layer].to(device) >= self.cfg.dead_steps_threshold

    @torch.no_grad()
    def reallocate_dead_children(self) -> None:
        """Algorithm 2: move dead children so every layer's allocation approaches ``k*_l``."""
        device = self.encoder.weight.device
        for layer in range(1, self.n_layers):
            n_children = self.layer_size(layer)
            n_parents = self.layer_start(layer)
            root = self.root_index(layer)
            capacity = self.capacity[layer].to(device)
            parent_steps = self._parent_pool_state(layer, self.steps_since_fired_per_layer, device)
            eligible = torch.cat(
                [parent_steps < self.parent_eligibility_steps, torch.tensor([True], device=device)], dim=0
            )
            index = self.parent_index(layer)
            counts = torch.bincount(index, minlength=n_parents + 1)

            # Optional root minimum: first move dead children of the largest families to the root,
            # then keep the root's children fixed during this round.
            fix_root = self.root_init_frac > 0.0
            if fix_root:
                need = max(0, math.ceil(self.root_init_frac * n_children) - counts[root].item())
                if need > 0:
                    dead = self._dead_children(layer, device)
                    donors = torch.arange(n_parents + 1, device=device)
                    donors = donors[donors != root]
                    for j in torch.argsort(counts[donors], descending=True).tolist():
                        if need <= 0:
                            break
                        candidates = torch.nonzero((index == donors[j].item()) & dead).flatten()
                        if candidates.numel() == 0:
                            continue
                        take = min(need, candidates.numel())
                        index[candidates[torch.randperm(candidates.numel(), device=device)[:take]]] = root
                        need -= take
                    counts = torch.bincount(index, minlength=n_parents + 1)

            # Algorithm 1 over the parent pool (the root takes part unless it is fixed).
            n_free = n_children - counts[root].item() if fix_root else n_children
            n_pool = n_parents if fix_root else n_parents + 1
            quotas = torch.zeros(n_parents + 1, dtype=torch.long, device=device)
            if max(0, n_free) > 0:
                quotas[:n_pool] = greedy_allocation(capacity[:n_pool], max(0, n_free), eligible[:n_pool])
            if fix_root:
                quotas[root] = counts[root].item()

            deficits = (quotas - counts).clamp(min=0)
            extras = (counts - quotas).clamp(min=0)
            if fix_root:
                extras[root] = 0
                deficits[root] = 0

            # Only dead children move: take them from over-full parents, give them to under-full ones.
            dead = self._dead_children(layer, device)
            movers = []
            for p in torch.nonzero(extras > 0).flatten().tolist():
                candidates = torch.nonzero((index == p) & dead).flatten()
                if candidates.numel() > 0:
                    movers.append(candidates[: min(extras[p].item(), candidates.numel())])
            receivers = [
                torch.full((deficits[p].item(),), p, device=device, dtype=torch.long)
                for p in torch.nonzero(deficits > 0).flatten().tolist()
            ]
            if movers and receivers:
                movers, receivers = torch.cat(movers), torch.cat(receivers)
                n_moves = min(movers.numel(), receivers.numel())
                index[movers[:n_moves]] = receivers[:n_moves]
            self.capacity[layer].zero_()

    @torch.no_grad()
    def attach_dead_children_to_root(self) -> None:
        """Root reset: every currently dead child becomes a child of the root."""
        for layer in range(1, self.n_layers):
            index = self.parent_index(layer)
            index[self._dead_children(layer, index.device)] = self.root_index(layer)

    @torch.no_grad()
    def _enforce_root_minimum_at_init(self) -> None:
        if self.root_init_frac <= 0.0:
            return
        for layer in range(1, self.n_layers):
            index = self.parent_index(layer)
            root = self.root_index(layer)
            # float32 ceil, as in the original implementation
            min_root = int(torch.ceil(self.root_init_frac * torch.tensor(self.layer_size(layer), dtype=torch.float)))
            deficit = min_root - (index == root).sum().item()
            non_root = torch.nonzero(index != root).flatten()
            if deficit <= 0 or non_root.numel() == 0:
                continue
            take = min(deficit, non_root.numel())
            index[non_root[torch.randperm(non_root.numel(), device=index.device)[:take]]] = root
