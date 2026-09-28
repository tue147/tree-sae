"""Convert the research checkpoints of the paper into released SAEs (safetensors + config.json).

Downloads every checkpoint listed in ``manifest.yaml`` from the (private) research repository,
converts it with ``tree_sae.models.convert``, checks that the converted SAE gives the same outputs
as the stored weights, and writes ``<out_dir>/gpt2-small/<paper name>/``.

    HF_TOKEN=... python scripts/hf/convert_checkpoints.py --out_dir hf_release
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from tree_sae.models import TreeSAE
from tree_sae.models.convert import convert_research_checkpoint
from tree_sae.models.io import load_sae, save_sae
from tree_sae.training.config import SAESpec

MANIFEST = Path(__file__).with_name("manifest.yaml")
TOKENS_PER_STEP = 5 * 1024


def tree_kwargs() -> dict:
    """Paper training options of the Tree SAE (not stored in the research checkpoints)."""
    spec = SAESpec(type="tree")
    return dict(
        parent_eligibility_steps=spec.parent_eligibility_tokens // TOKENS_PER_STEP,
        realloc_interval=spec.realloc_interval,
        realloc_interval_growth=spec.realloc_interval_growth,
        max_realloc_interval=spec.max_realloc_interval,
        root_reset_step=spec.root_reset_step,
        root_init_frac=spec.root_init_frac,
        aux_layers=tuple(spec.aux_layers),
    )


@torch.no_grad()
def check_roundtrip(path: Path, reference, device: str) -> None:
    """The saved SAE must reproduce the converted SAE exactly."""
    loaded = load_sae(path, device)
    x = torch.randn(64, reference.cfg.d_in, generator=torch.Generator().manual_seed(0)).to(device)
    reference = reference.to(device).eval()
    assert torch.equal(loaded(x), reference(x)), path
    if isinstance(reference, TreeSAE):
        for layer in range(1, reference.n_layers):
            assert torch.equal(loaded.parent_index(layer), reference.parent_index(layer))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="hf_release")
    parser.add_argument("--repo_id", default="tueminh/tree_sae", help="Repository with the research checkpoints")
    parser.add_argument("--cache_dir", default="research_checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", nargs="*", default=None, help="Convert only these paper names")
    args = parser.parse_args()

    manifest = yaml.safe_load(MANIFEST.read_text())
    for name, entry in manifest["saes"].items():
        if args.only and name not in args.only:
            continue
        ckpt = hf_hub_download(args.repo_id, entry["file"], local_dir=args.cache_dir, token=os.environ.get("HF_TOKEN"))
        kwargs = {}
        if entry["type"] == "matryoshka":
            kwargs["matryoshka_boundaries"] = entry["boundaries"]
        if entry["type"] == "tree":
            kwargs["tree_kwargs"] = tree_kwargs()
        sae = convert_research_checkpoint(ckpt, **kwargs)
        expected = entry.get("features_per_layer")
        if expected is not None:
            assert sae.features_per_layer == expected, (name, sae.features_per_layer, expected)
        out = Path(args.out_dir) / "gpt2-small" / name
        save_sae(sae, out, extra={"paper_name": name, "source_checkpoint": entry["file"]})
        check_roundtrip(out, sae, args.device)
        print(f"{name}: {entry['file']} -> {out}")


if __name__ == "__main__":
    main()
