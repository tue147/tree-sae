"""GPU regression tests: 20 real training steps on GPT-2-small layer-5 activations.

Mirrors the paper setup (fp32 GPT-2, 16-mixed SAE training, batches of 5 x 1024 Pile tokens,
two sanity validation steps) and compares with fixtures produced by the original research code.
Losses must be bit-identical on the GPU model/torch version that produced the fixtures (NVIDIA
B200, torch 2.7.1); weights and allocation vectors are compared through SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import pytest
import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping

from tree_sae.models import TreeSAE
from tree_sae.training import SAETrainingModule
from tree_sae.training.config import SAESpec
from tree_sae.training.data import LLMActivations
from tree_sae.training.trainer import build_sae, seed_all

FIXTURES = Path(__file__).parent / "fixtures" / "gpu"
NAMES = sorted(p.stem.removeprefix("gpu_") for p in FIXTURES.glob("gpu_*.pt"))
HOOK = "blocks.5.hook_resid_pre"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _spec(golden: dict) -> tuple[SAESpec, int, int]:
    spec, common = golden["spec"], golden["common"]
    tokens_per_step = common["batch_size"] * common["max_length"]
    dead_steps = common["dead_tokens_threshold"] // tokens_per_step
    d_sae = int(common["expansion_factor"] * 768)
    kind = {"topk": "topk", "mpsae": "mp", "topk_matryoshka": "matryoshka", "topk_tree_sae": "tree"}[spec["sae_name"]]
    ends = [*spec.get("sizes", []), d_sae]
    features_per_layer = [b - a for a, b in itertools.pairwise([0, *ends])]
    sae_spec = SAESpec(type=kind, d_sae=d_sae, k=spec.get("k"), auxk=common["auxk"], auxk_coef=common["auxk_coef"])
    if kind in ("matryoshka", "tree"):
        sae_spec.features_per_layer = features_per_layer
    if kind == "tree":
        sae_spec.k_per_layer = spec["list_k"]
        sae_spec.realloc_interval = spec["nbatch_dynamic_alloc"]
        sae_spec.realloc_interval_growth = spec["increase_nbatch_per_alloc"]
        sae_spec.max_realloc_interval = spec["max_nbatch_dynamic_alloc"]
        sae_spec.root_reset_step = spec["force_root_attach_steps"]
        sae_spec.root_init_frac = spec["root_init_frac"]
        sae_spec.aux_layers = list(spec["aux_layers"])
    return sae_spec, dead_steps, spec.get("parent_eligibility_steps", 0)


@pytest.fixture(scope="module")
def gpt2():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2-small", device="cuda:0")


@pytest.mark.parametrize("name", NAMES)
def test_gpu_training_matches_original(name: str, gpt2) -> None:
    golden = torch.load(FIXTURES / f"gpu_{name}.pt", weights_only=False)
    tokens = torch.load(FIXTURES / "gpt2_pile_tokens.pt")
    exact = golden["gpu"] == torch.cuda.get_device_name(0) and golden["torch_version"] == torch.__version__

    spec, dead_steps, eligibility_steps = _spec(golden)
    seed_all(42)
    sae = build_sae(spec, 768, HOOK, dead_steps, eligibility_steps)
    module = SAETrainingModule(
        sae, lr=golden["common"]["lr"], auxk_coef=spec.auxk_coef, preprocess=LLMActivations(gpt2, HOOK)
    )

    class Recorder(Callback):
        loss, metrics, alloc, final = [], [], [], None

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.loss.append(outputs["loss"].detach().float().cpu())
            self.metrics.append({k: v.detach().float().cpu() for k, v in trainer.callback_metrics.items()})
            if isinstance(sae, TreeSAE):
                self.alloc.append({l: _sha(sae.parent_index(l)) for l in range(1, sae.n_layers)})
            if batch_idx == len(tokens["train"]) - 1:
                self.final = {k: _sha(v) for k, v in sae.state_dict().items()}

    rec = Recorder()
    loader = lambda x: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=None)  # noqa: E731
    Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_steps=len(tokens["train"]),
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[rec, EarlyStopping(monitor="loss/total", min_delta=0.005, patience=1, mode="min")],
    ).fit(module, train_dataloaders=loader(tokens["train"]), val_dataloaders=loader(tokens["val"]))

    expected_loss = golden["loss"].reshape(-1)
    actual_loss = torch.stack([l.reshape(()) for l in rec.loss])
    if exact:
        assert torch.equal(actual_loss, expected_loss), (actual_loss, expected_loss)
    else:
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-5)

    for step, old in enumerate(golden["metrics"]):
        for key, value in old.items():
            new = (
                key.replace("loss/auxk", "loss/aux")
                .replace("train/dead/abs_", "dead/")
                .replace("train/dead/abs", "dead/0")
            )
            new = "loss/mse/0" if new == "loss/mse" else "loss/aux/0" if new == "loss/aux" else new
            if new.startswith(("loss/mse/", "loss/aux/", "dead/")) or new == "loss/total":
                if exact:
                    assert torch.equal(rec.metrics[step][new].reshape(()), value.reshape(())), (step, key)
                else:
                    torch.testing.assert_close(
                        rec.metrics[step][new].reshape(()), value.reshape(()), rtol=1e-3, atol=1e-5
                    )

    if golden["alloc"] and exact:
        boundaries = [0, *golden["spec"]["sizes"]]
        for step, old in enumerate(golden["alloc"]):
            assert {boundaries.index(int(k)): v for k, v in old.items()} == rec.alloc[step], step
    if exact:
        rename = {"pre_encoder_bias": "b_dec"}
        for key, digest in golden["final"].items():
            if key.startswith("allocate_natrices."):
                new = f"parent_index_{[0, *golden['spec']['sizes']].index(int(key.split('.')[1]))}"
            else:
                new = rename.get(key, key)
            assert rec.final[new] == digest, key
