"""GPU regression tests of the SAEBench-derived evals against the original research code.

The fixture holds the absorption/splitting metrics and the AutoInterp results (with a deterministic
fake judge instead of the OpenAI API) that the original code produced for a fixed GPT-2 SAE.
"""

from __future__ import annotations

import hashlib
import random
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tree_sae.models import TopKSAE

FIXTURE = Path(__file__).parent / "fixtures" / "gpu" / "saebench.pt"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing"),
]


class _FakeCompletions:
    async def create(self, model, messages, n, max_tokens, stream):
        text = messages[-1]["content"]
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        if "Here is the explanation" in text:
            content = ", ".join(str(1 + (h >> (8 * i)) % 14) for i in range(3))
        else:
            content = f"This neuron activates on concept {h % 1000}."
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class FakeAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, weights_only=False)


def _sae(golden=None):
    """The fixed SAE the fixture was produced with (seeded init, encoder scaled by 4)."""
    torch.manual_seed(0)
    sae = TopKSAE(768, 6144, "blocks.5.hook_resid_pre", k=32, dead_steps_threshold=1953, auxk=256)
    with torch.no_grad():
        sae.encoder.weight.mul_(4.0)
    return sae.cuda()


def test_absorption_matches_original(golden, tmp_path):
    from tree_sae.evals.absorption import AbsorptionEvalConfig, run_absorption_eval

    # the research code read the absorption activations at blocks.5.hook_resid_post
    config = AbsorptionEvalConfig(
        model_name="gpt2-small",
        random_seed=42,
        llm_batch_size=32,
        llm_dtype="float32",
        use_sae_hook_for_absorption=False,
    )
    result = run_absorption_eval(
        config, [("sae", _sae(golden))], 5, "cuda", str(tmp_path / "out"), True, artifacts_dir=tmp_path / "artifacts"
    )
    assert result["sae"]["eval_result_metrics"] == golden["absorption"]
    assert result["sae"]["eval_result_details"] == golden["absorption_details"]


def test_autointerp_matches_original(golden, tmp_path, monkeypatch):
    from tree_sae.evals.autointerp import AutoInterpEvalConfig
    from tree_sae.evals.autointerp import run as autointerp

    monkeypatch.setattr(autointerp, "AsyncOpenAI", FakeAsyncOpenAI)
    config = AutoInterpEvalConfig(model_name="gpt2-small", llm_batch_size=32, llm_dtype="float32", random_seed=42)
    config.n_latents = 20
    config.total_tokens = 200_000
    random.seed(0), np.random.seed(0), torch.manual_seed(0)  # the eval seeds itself; this must not matter
    result = autointerp.run_eval(
        config,
        [("sae", _sae(golden))],
        5,
        "cuda",
        "fake-key",
        str(tmp_path / "out"),
        True,
        None,
        str(tmp_path / "artifacts"),
    )
    # Per-latent results must match exactly; the mean may differ in the last digits because the
    # original collects results in asyncio completion order.
    metrics = result["sae"]["eval_result_metrics"]["autointerp"]
    for key, value in golden["autointerp"]["autointerp"].items():
        assert metrics[key] == pytest.approx(value, rel=1e-6)
    latents = {
        int(k): {kk: v[kk] for kk in ("explanation", "predictions", "correct seqs", "score")}
        for k, v in result["sae"]["eval_result_unstructured"].items()
    }
    assert latents == golden["autointerp_latents"]
