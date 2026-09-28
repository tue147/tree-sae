"""Bit-exact regression tests of the evaluation / analysis code against the original research code.

``fixtures/evals_cpu.pt`` holds the inputs and outputs of the original implementations (CPU) for:
sparse feature extraction of every SAE type, MCS in its four variants (fp32 and bf16), the full
hierarchy pipeline (seeded), composition, sibling co-occurrence and reconstruction metrics on a
tiny random transformer. The SAEs are rebuilt from their raw research state dicts, which also
tests the checkpoint converter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tree_sae.analysis.activations import sae_sparse_activations
from tree_sae.analysis.cooccurrence import average_sibling_cooccurrence
from tree_sae.evals.composition import max_cosine_similarities
from tree_sae.evals.hierarchy import mcs_children, run_hierarchy_eval
from tree_sae.evals.reconstruction import evaluate_reconstruction, variance_explained
from tree_sae.models.convert import convert_research_state

FIXTURE = Path(__file__).parent / "fixtures" / "evals_cpu.pt"
D_SAE = 256
SIZES = [16, 48, 144]


@pytest.fixture(scope="module")
def golden() -> dict:
    torch.set_num_threads(1)
    return torch.load(FIXTURE, weights_only=False)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return "unknown"


def _exact(golden: dict) -> bool:
    return golden["platform"] == _cpu_model() and golden["torch_version"] == torch.__version__


def _same(actual, expected, exact: bool, what: str = "") -> None:
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    if exact or not expected.is_floating_point():
        assert torch.equal(actual, expected), what
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=what)


def _sae(golden: dict, name: str):
    state, hparams = golden["states"][name]
    return convert_research_state(state, hparams, matryoshka_boundaries=SIZES)


@pytest.mark.parametrize("name", ["relu", "topk", "matryoshka", "mp", "tree"])
def test_sparse_activations(golden, name):
    indices, values = sae_sparse_activations(_sae(golden, name), golden["acts"][:1500], 512, "cpu")
    exp_indices, exp_values = golden["sparse"][name]
    _same(indices, exp_indices, True, "indices")
    _same(values, exp_values, _exact(golden), "values")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("scaled", [True, False])
def test_mcs_children(golden, dtype, binary, scaled):
    indices, values = golden["mcs_input"]
    ids, scores = mcs_children(indices, values.to(dtype), D_SAE, "cpu", n_children=20, binary=binary, scaled=scaled)
    exp_ids, exp_scores = golden["mcs"][(str(dtype), binary, scaled)]
    _same(scores, exp_scores, _exact(golden), "scores")
    if _exact(golden):
        _same(ids, exp_ids, True, "ids")


@pytest.mark.parametrize("name", ["tree", "topk"])
def test_hierarchy_pipeline(golden, name):
    args = golden["hierarchy_args"]
    sae = _sae(golden, name)
    indices, values = sae_sparse_activations(sae, golden["acts"], 512, "cpu")
    with torch.no_grad():
        results = run_hierarchy_eval(
            sae,
            golden["acts"],
            indices,
            values,
            "cpu",
            n_parents=args["n_parents"],
            dense_ratio=args["ratio"],
            min_child_count=args["min_sample"],
            probe_epochs=args["epochs"],
            probe_batch_size=args["bs"],
            seed=42,
        )
    for key in ("mcs", "tree"):
        expected = golden["hierarchy"][name][key]
        assert results[key].pairs == expected["pair"]
        exact = _exact(golden)
        for (p, c), (ep, ec) in zip(results[key].similarities, expected["sim"]):
            _same([p, c], [ep, ec], exact, "similarity")
        if exact:
            for top, etop in zip(results[key].top_features, expected["top"]):
                _same(top, etop, True, "top features")
    assert golden["hierarchy"]["tree"]["tree"]["pair"], "fixture must contain tree-structure pairs"


@pytest.mark.parametrize("name", ["relu", "topk", "matryoshka", "mp", "tree"])
def test_composition(golden, name):
    _same(max_cosine_similarities(_sae(golden, name)), golden["composition"][name], _exact(golden))


def test_sibling_cooccurrence(golden):
    indices, values = golden["tree_sparse_full"]
    indices = indices.clone()
    indices[values < 1e-3] = D_SAE
    rate = average_sibling_cooccurrence(_sae(golden, "tree"), indices)
    _same(rate, golden["cooccurrence"], _exact(golden))


@pytest.mark.parametrize("name", ["topk", "tree", "mp", "relu"])
def test_reconstruction(golden, name):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, n_ctx=32, d_head=8, n_heads=4, d_vocab=97, act_fn="gelu", device="cpu"
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(golden["tiny_state"])
    result = evaluate_reconstruction(model, _sae(golden, name), golden["tiny_tokens"], 8, "cpu")
    expected = golden["reconstruction"][name]
    actual = torch.tensor([result.mse, result.variance_explained, result.downstream_ce_loss], dtype=torch.float64)
    _same(actual, torch.tensor([float(v) for v in expected], dtype=torch.float64), _exact(golden))


def test_variance_explained(golden):
    x, y, expected = golden["variance_explained"]
    _same(variance_explained(x, y), expected, _exact(golden))
