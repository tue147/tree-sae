"""Convert Lightning checkpoints of the original research code to this package's SAEs."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch

from .base import BaseSAE
from .matryoshka import MatryoshkaSAE
from .mp import MPSAE
from .relu import ReLUSAE
from .topk import TopKSAE
from .tree import TreeSAE

_RESEARCH_NAMES = {
    "sae": "relu",
    "topk": "topk",
    "mpsae": "mp",
    "topk_matryoshka": "matryoshka",
    "topk_tree_sae": "tree",
}


def _layer_sizes(boundaries: list[int], d_sae: int) -> list[int]:
    return [b - a for a, b in itertools.pairwise([0, *boundaries, d_sae])]


def convert_research_state(
    state_dict: dict[str, torch.Tensor],
    hparams: dict[str, Any],
    matryoshka_boundaries: list[int] | None = None,
    tree_kwargs: dict[str, Any] | None = None,
) -> BaseSAE:
    """Build an SAE from the ``state_dict``/``hyper_parameters`` of a research checkpoint.

    Args:
        matryoshka_boundaries: Cumulative layer boundaries (e.g. ``[1536, 4608, 13824]``) of a
            Matryoshka SAE; the research checkpoints do not store them.
        tree_kwargs: Tree SAE training options (reallocation schedule etc.) to record in the
            converted SAE; they are not stored in the research checkpoints and do not affect
            inference.
    """
    kind = _RESEARCH_NAMES[hparams["sae_name"]]
    state = {k.removeprefix("autoencoder."): v for k, v in state_dict.items() if k.startswith("autoencoder.")}
    d_sae, d_in = state["encoder.weight"].shape
    batch_tokens = hparams["batch_size"] * hparams["max_length"] * hparams.get("accumulate_grad_batches", 1)
    dead_steps = hparams.get("dead_steps_threshold") or hparams["dead_tokens_threshold"] // batch_tokens
    common = dict(
        d_in=d_in,
        hook_name=hparams["hook_names"][0],
        dead_steps_threshold=dead_steps,
        dead_threshold=hparams["dead_threshold"],
        standardize=hparams["standardize"],
    )
    if kind == "relu":
        sae = ReLUSAE(d_sae=d_sae, auxk=hparams["auxk"], l1_coef=hparams["sparsity_coef"], **common)
    elif kind == "topk":
        sae = TopKSAE(d_sae=d_sae, k=hparams["k"], auxk=hparams["auxk"], **common)
    elif kind == "mp":
        sae = MPSAE(d_sae=d_sae, k=hparams["k"], **common)
    elif kind == "matryoshka":
        if matryoshka_boundaries is None:
            raise ValueError("Matryoshka checkpoints do not store their layer sizes; pass matryoshka_boundaries.")
        sae = MatryoshkaSAE(
            features_per_layer=_layer_sizes(matryoshka_boundaries, d_sae),
            k=hparams["k"],
            auxk=hparams["auxk"],
            use_loss_var=hparams["use_loss_var"],
            **common,
        )
    else:
        sae = TreeSAE(
            features_per_layer=_layer_sizes(hparams["sizes"], d_sae),
            k_per_layer=hparams["list_k"],
            auxk=hparams["auxk"],
            use_loss_var=hparams["use_loss_var"],
            **{"legacy_allocation_order": True, **(tree_kwargs or {})},
            **common,
        )

    renamed = {"pre_encoder_bias": "b_dec", "last_nonzero": "steps_since_fired"}
    new_state = {}
    for key, value in state.items():
        if key.startswith("allocate_natrices."):
            key = f"parent_index_{[0, *hparams['sizes']].index(int(key.split('.')[1]))}"
        new_state[renamed.get(key, key)] = value
    sae.load_state_dict(new_state)
    return sae


def convert_research_checkpoint(path: str | Path, **kwargs: Any) -> BaseSAE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return convert_research_state(ckpt["state_dict"], ckpt["hyper_parameters"], **kwargs)
