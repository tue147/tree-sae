"""Fast CPU unit tests (no model downloads)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import torch

from tree_sae.analysis.geometry import descendants, label_by_child
from tree_sae.analysis.probe_correlation import activation_coverage, children_by_coverage
from tree_sae.analysis.tree_vis import save_feature_tree, subtree
from tree_sae.models import MatryoshkaSAE, MPSAE, ReLUSAE, TopKSAE, TreeSAE
from tree_sae.models.allocation import greedy_allocation
from tree_sae.models.io import load_sae, save_sae
from tree_sae.training.config import TrainConfig
from tree_sae.training.trainer import build_sae

ROOT = Path(__file__).parents[2]


def _tree(**kwargs) -> TreeSAE:
    torch.manual_seed(0)
    return TreeSAE(16, "blocks.0.hook_resid_pre", [8, 16, 40], [3, 2, 1], dead_steps_threshold=5, auxk=8, **kwargs)


# ------------------------------------------------------------------ Algorithm 1


@pytest.mark.parametrize("seed", range(20))
def test_greedy_allocation_is_optimal(seed):
    """Theorem 4.1: the greedy quotas maximise min_p C_p / k_p (brute force on small cases)."""
    g = torch.Generator().manual_seed(seed)
    n_parents, n_children = 4, 6
    capacities = torch.rand(n_parents, generator=g) * 10
    eligible = torch.ones(n_parents, dtype=torch.bool)
    quotas = greedy_allocation(capacities, n_children, eligible)
    assert quotas.sum() == n_children

    def payoff(k):
        return min(capacities[p].item() / k[p] for p in range(n_parents) if k[p] > 0)

    best = max(
        payoff(k) for k in itertools.product(range(n_children + 1), repeat=n_parents) if sum(k) == n_children
    )
    assert payoff(quotas.tolist()) == pytest.approx(best)


def test_greedy_allocation_respects_eligibility():
    capacities = torch.tensor([5.0, 0.0, 3.0, 9.0])
    eligible = torch.tensor([True, True, True, False])
    quotas = greedy_allocation(capacities, 7, eligible)
    assert quotas.sum() == 7 and quotas[3] == 0 and quotas[1] == 0
    # all eligible capacities zero -> uniform over eligible parents
    quotas = greedy_allocation(torch.zeros(3), 5, torch.tensor([True, False, True]))
    assert quotas.tolist() == [3, 0, 2]


# ------------------------------------------------------------------ models


def test_tree_activation_coverage_holds():
    """Eq. 6: a feature can only be active on tokens where its parent is active."""
    sae = _tree()
    x = torch.randn(4, 7, 16)
    latents, *_ = sae.encode(x)
    for layer in range(1, sae.n_layers):
        start, end = sae.boundaries[layer], sae.boundaries[layer + 1]
        parents = torch.cat([latents[..., :start], torch.ones_like(latents[..., :1])], dim=-1)
        parent_active = parents[..., sae.parent_index(layer)] > sae.cfg.dead_threshold
        assert not ((latents[..., start:end] > 0) & ~parent_active).any()
    # per-layer sparsity
    for layer, k in enumerate(sae.k_per_layer):
        start, end = sae.boundaries[layer], sae.boundaries[layer + 1]
        assert ((latents[..., start:end] > 0).sum(-1) <= k).all()


def test_root_init_frac():
    sae = _tree(root_init_frac=0.5)
    for layer in range(1, sae.n_layers):
        n_root = (sae.parent_index(layer) == sae.root_index(layer)).sum().item()
        assert n_root >= 0.5 * sae.layer_size(layer)


def test_children_and_descendants():
    sae = _tree()
    for layer in range(1, sae.n_layers):
        sae.parent_index(layer).fill_(sae.root_index(layer))
    sae.parent_index(1)[:3] = 2  # features 8, 9, 10 are children of 2
    sae.parent_index(2)[:2] = torch.tensor([8, 2])  # 24 -> 8, 25 -> 2
    assert sae.children_of(2) == [8, 9, 10, 25]
    assert sae.children_of(8) == [24]
    assert sae.children_of(sae.root_index(2)) == []  # root slot of the last layer: no valid children
    assert descendants(sae, 2) == [8, 9, 10, 24, 25]
    assert subtree(sae, 2)[8] == [24]


@pytest.mark.parametrize(
    "sae",
    [
        ReLUSAE(12, 40, "h", dead_steps_threshold=3, auxk=4),
        TopKSAE(12, 40, "h", k=5, dead_steps_threshold=3, auxk=4),
        MatryoshkaSAE(12, "h", [10, 30], k=5, dead_steps_threshold=3, auxk=4),
        MPSAE(12, 40, "h", k=5),
        TreeSAE(12, "h", [10, 30], [4, 1], dead_steps_threshold=3, auxk=4, root_reset_step=None),
    ],
    ids=["relu", "topk", "matryoshka", "mp", "tree"],
)
def test_save_load_roundtrip(sae, tmp_path):
    save_sae(sae, tmp_path / "sae")
    loaded = load_sae(tmp_path / "sae")
    assert type(loaded) is type(sae)
    x = torch.randn(3, 12)
    sae.eval(), loaded.eval()
    assert torch.equal(loaded(x), sae(x))
    assert json.loads((tmp_path / "sae" / "config.json").read_text())["hook_name"] == "h"


def test_error_term_returns_input():
    sae = TopKSAE(12, 40, "h", k=5, dead_steps_threshold=3, auxk=4)
    sae.use_error_term = True
    x = torch.randn(3, 12)
    torch.testing.assert_close(sae(x), x)


# ------------------------------------------------------------------ configs


@pytest.mark.parametrize("path", sorted((ROOT / "configs").rglob("*.yaml")), ids=lambda p: p.stem)
def test_configs_build(path):
    cfg = TrainConfig.from_yaml(path)
    assert cfg.dead_steps_threshold == 1953 and cfg.max_steps == 97657
    spec = cfg.sae
    if spec.features_per_layer:
        spec.features_per_layer = [max(1, n // 256) for n in spec.features_per_layer]  # shrink for speed
        if spec.type == "tree":
            spec.k_per_layer = [min(k, n) for k, n in zip(spec.k_per_layer, spec.features_per_layer)]
        else:
            spec.k = min(spec.k, sum(spec.features_per_layer))
        spec.auxk = min(spec.auxk or 0, sum(spec.features_per_layer)) or None
    else:
        spec.d_sae = 96
        if spec.auxk:
            spec.auxk = 32
    sae = build_sae(spec, 768, cfg.hook_name, cfg.dead_steps_threshold, cfg.parent_eligibility_steps)
    assert sae.cfg.hook_name == "blocks.5.hook_resid_pre"
    if isinstance(sae, TreeSAE):
        assert sae.parent_eligibility_steps == 9 and sae.aux_layers == {0} and sae.root_reset_step == 50_000


def test_manifest_matches_configs():
    import yaml

    manifest = yaml.safe_load((ROOT / "scripts/hf/manifest.yaml").read_text())["saes"]
    config_names = {p.stem for p in (ROOT / "configs").rglob("*.yaml")}
    assert set(manifest) == config_names
    for name, entry in manifest.items():
        cfg = TrainConfig.from_yaml(next((ROOT / "configs").rglob(f"{name}.yaml")))
        if cfg.sae.features_per_layer:
            assert entry["features_per_layer"] == cfg.sae.features_per_layer
        if entry["type"] == "matryoshka":
            assert entry["boundaries"] == list(itertools.accumulate(cfg.sae.features_per_layer))[:-1]


# ------------------------------------------------------------------ analysis


def test_activation_coverage():
    d_sae = 5
    indices = torch.tensor([[0, 1], [0, 2], [1, 5], [0, 1]])  # 5 = inactive padding
    coverage = activation_coverage(indices, parent=0, d_sae=d_sae)
    assert coverage[1].item() == pytest.approx(2 / 3)
    assert coverage[2].item() == 1.0
    assert torch.isnan(coverage[3])
    children, _ = children_by_coverage(indices, 0, d_sae, n_children=3, min_count=1)
    assert children == [2, 1]


def test_label_by_child():
    indices = torch.tensor([[3, 7], [7, 3], [1, 2]])
    mask, labels = label_by_child(indices, [7, 3])
    assert mask.tolist() == [True, True, False] and labels.tolist() == [3, 7]


def test_feature_tree_html(tmp_path):
    tree = {1: [4, 5], 4: [9]}
    data = {f: {"sparsity": 0.1, "top_activations": [{"activation": 1.0, "sequence": "a <b>"}]} for f in (1, 4, 5, 9)}
    out = tmp_path / "tree.html"
    save_feature_tree(tree, data, str(out))
    html = out.read_text()
    assert "Feature 9" in html and "d3" in html


def test_paper_figures(tmp_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/figures/plot_metrics.py"), "--results",
         str(ROOT / "results/paper_results.json"), "--out", str(tmp_path), "--format", "png"],
        check=True, capture_output=True,
    )
    assert len(list(tmp_path.glob("fig*.png"))) == 7
