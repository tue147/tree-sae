"""Save / load SAEs as ``sae.safetensors`` + ``config.json`` (locally or from the Hugging Face Hub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .base import BaseSAE
from .matryoshka import MatryoshkaSAE
from .mp import MPSAE
from .relu import ReLUSAE
from .topk import TopKSAE
from .tree import TreeSAE

HF_REPO_ID = "tueminh/tree-sae"

_TYPES: dict[type, str] = {TreeSAE: "tree", MatryoshkaSAE: "matryoshka", MPSAE: "mp", TopKSAE: "topk", ReLUSAE: "relu"}
_CLASSES = {name: cls for cls, name in _TYPES.items()}


def sae_config(sae: BaseSAE) -> dict[str, Any]:
    """Constructor arguments that rebuild ``sae``."""
    kind = next(name for cls, name in _TYPES.items() if type(sae) is cls)
    cfg = sae.cfg
    config: dict[str, Any] = dict(
        type=kind,
        d_in=cfg.d_in,
        hook_name=cfg.hook_name,
        dead_steps_threshold=cfg.dead_steps_threshold,
        dead_threshold=cfg.dead_threshold,
        standardize=cfg.standardize,
    )
    if kind == "relu":
        config.update(d_sae=cfg.d_sae, auxk=cfg.auxk, l1_coef=sae.l1_coef)
    elif kind == "topk":
        config.update(d_sae=cfg.d_sae, k=sae.k, auxk=cfg.auxk)
    elif kind == "mp":
        config.update(d_sae=cfg.d_sae, k=sae.k, eps=sae.eps, use_pre_encoder_bias=sae.use_pre_encoder_bias)
    elif kind == "matryoshka":
        config.update(features_per_layer=sae.features_per_layer, k=sae.k, auxk=cfg.auxk, use_loss_var=sae.use_loss_var)
    else:
        config.update(
            features_per_layer=sae.features_per_layer,
            k_per_layer=sae.k_per_layer,
            auxk=cfg.auxk,
            use_loss_var=sae.use_loss_var,
            parent_eligibility_steps=sae.parent_eligibility_steps,
            realloc_interval=sae.initial_realloc_interval,
            realloc_interval_growth=sae.realloc_interval_growth,
            max_realloc_interval=sae.max_realloc_interval,
            root_reset_step=sae.root_reset_step,
            root_init_frac=sae.root_init_frac,
            aux_layers=sorted(sae.aux_layers),
            legacy_allocation_order=sae.legacy_allocation_order,
        )
    return config


def save_sae(sae: BaseSAE, directory: str | Path, extra: dict[str, Any] | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in sae.state_dict().items()}
    save_file(state, directory / "sae.safetensors")
    config = sae_config(sae)
    if extra:
        config["metadata"] = extra
    (directory / "config.json").write_text(json.dumps(config, indent=2))


def load_sae(directory: str | Path, device: str | torch.device = "cpu") -> BaseSAE:
    """Load an SAE saved with :func:`save_sae`."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    kind = config["type"]
    kwargs = {k: v for k, v in config.items() if k not in ("type", "metadata")}
    if "aux_layers" in kwargs:
        kwargs["aux_layers"] = tuple(kwargs["aux_layers"])
    sae = _CLASSES[kind](**kwargs)
    sae.load_state_dict(load_file(directory / "sae.safetensors"))
    return sae.to(device)


def from_pretrained(name: str, device: str | torch.device = "cpu", repo_id: str = HF_REPO_ID) -> BaseSAE:
    """Download a released SAE, e.g. ``from_pretrained("gpt2-small/tree_sae_4layer_l0_32")``."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id, allow_patterns=[f"{name}/*"])
    return load_sae(Path(local) / name, device)
