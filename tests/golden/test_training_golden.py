"""Bit-exact regression tests of SAE training against the original research implementation.

The fixtures in ``fixtures/train_*.pt`` were produced by running the original code (the code that
trained the released checkpoints) for 30 steps on tiny synthetic data with shrunken thresholds,
so that dead features, AuxK, dynamic reallocation and the root reset all trigger. Each test
re-runs the same configuration with this package under Lightning's Trainer and requires every
per-step loss, dead fraction and allocation vector, and the final weights, to be identical.

Exact equality is only expected on the hardware/software that produced the fixtures (see the
``platform`` / ``torch_version`` fields: CPU model + torch version, 1 thread); elsewhere the
comparison falls back to ``rtol=1e-5`` and discrete quantities must still match exactly.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch
from lightning.pytorch import Callback, Trainer

from tree_sae.models import MPSAE, MatryoshkaSAE, ReLUSAE, TopKSAE, TreeSAE
from tree_sae.training import SAETrainingModule
from tree_sae.training.trainer import seed_all

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = [
    "relu",
    "topk",
    "matryoshka_2layer",
    "matryoshka_4layer",
    "mp",
    "tree_2layer",
    "tree_4layer",
    "tree_4layer_options",
]
HOOK = "blocks.0.hook_resid_pre"


def _load(name: str) -> dict:
    return torch.load(FIXTURES / f"train_{name}.pt", weights_only=False)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return "unknown"


def _exact_expected(golden: dict) -> bool:
    return golden["torch_version"] == torch.__version__ and golden["platform"] == _cpu_model()


def _assert_same(actual: torch.Tensor, expected: torch.Tensor, exact: bool, what: str) -> None:
    actual, expected = torch.as_tensor(actual).reshape(expected.shape), expected
    if exact or not expected.is_floating_point():
        assert torch.equal(actual, expected), f"{what}: {actual} != {expected}"
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, msg=what)


def _build_sae(golden: dict):
    spec, common = golden["spec"], golden["common"]
    d_in = golden["inputs"].shape[-1]
    d_sae = int(common["expansion_factor"] * d_in)
    dead_steps = common["dead_tokens_threshold"] // (common["batch_size"] * common["max_length"])
    base = dict(hook_name=HOOK, dead_steps_threshold=dead_steps, dead_threshold=common["dead_threshold"])
    name = spec["sae_name"]
    if name == "sae":
        return ReLUSAE(d_in, d_sae, auxk=common["auxk"], l1_coef=common["sparsity_coef"], **base)
    if name == "topk":
        return TopKSAE(d_in, d_sae, k=spec["k"], auxk=common["auxk"], **base)
    if name == "mpsae":
        return MPSAE(d_in, d_sae, k=spec["k"], **base)
    ends = [*spec["sizes"], d_sae]
    features_per_layer = [b - a for a, b in itertools.pairwise([0, *ends])]
    if name == "topk_matryoshka":
        return MatryoshkaSAE(d_in, features_per_layer=features_per_layer, k=spec["k"], auxk=common["auxk"], **base)
    return TreeSAE(
        d_in,
        features_per_layer=features_per_layer,
        k_per_layer=spec["list_k"],
        auxk=common["auxk"],
        parent_eligibility_steps=spec["parent_eligibility_steps"],
        realloc_interval=spec["nbatch_dynamic_alloc"],
        realloc_interval_growth=spec["increase_nbatch_per_alloc"],
        max_realloc_interval=spec["max_nbatch_dynamic_alloc"],
        root_reset_step=spec["force_root_attach_steps"],
        root_init_frac=spec["root_init_frac"],
        aux_layers=tuple(spec["aux_layers"]),
        **base,
    )


class _Recorder(Callback):
    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self.loss, self.metrics, self.parent_index, self.steps_since_fired = [], [], [], []
        self.final_state, self.tree_state = None, None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        sae = pl_module.sae
        self.loss.append(outputs["loss"].detach().clone())
        self.metrics.append({k: v.detach().clone() for k, v in trainer.callback_metrics.items()})
        self.steps_since_fired.append(sae.steps_since_fired.clone())
        if isinstance(sae, TreeSAE):
            self.parent_index.append({l: sae.parent_index(l).clone() for l in range(1, sae.n_layers)})
        if batch_idx == self.n_steps - 1:
            self.final_state = {k: v.detach().clone() for k, v in sae.state_dict().items()}
            if isinstance(sae, TreeSAE):
                self.tree_state = dict(
                    steps_since_fired={l: v.clone() for l, v in sae.steps_since_fired_per_layer.items()},
                    capacity={l: v.clone() for l, v in sae.capacity.items()},
                    realloc_interval=sae.realloc_interval,
                    root_reset_done=sae._root_reset_done,
                )


def _run(golden: dict) -> tuple[torch.Tensor, _Recorder]:
    common, n_steps = golden["common"], golden["n_steps"]
    torch.set_num_threads(1)
    seed_all(golden["seed"])
    sae = _build_sae(golden)
    init_state = {k: v.detach().clone() for k, v in sae.state_dict().items()}
    module = SAETrainingModule(sae, lr=common["lr"], auxk_coef=common["auxk_coef"], use_loss_var=common["use_loss_var"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(golden["inputs"]), batch_size=common["batch_size"], shuffle=False
    )
    recorder = _Recorder(n_steps)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=n_steps,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[recorder],
    )
    trainer.fit(module, train_dataloaders=loader)
    return init_state, recorder


def _new_state_name(old: str, boundaries: list[int]) -> str:
    if old == "pre_encoder_bias":
        return "b_dec"
    if old == "last_nonzero":
        return "steps_since_fired"
    if old.startswith("allocate_natrices."):
        return f"parent_index_{boundaries.index(int(old.split('.')[1]))}"
    return old


def _metric_map(old: dict) -> dict[str, str]:
    """Old metric name -> new metric name, for the loss/dead metrics."""
    mapping = {"loss/total": "loss/total", "loss/additional_loss": "loss/extra"}
    for key in old:
        if key == "loss/mse":
            mapping[key] = "loss/mse/0"
        elif key == "loss/auxk":
            mapping[key] = "loss/aux/0"
        elif key == "train/dead/abs":
            mapping[key] = "dead/0"
        elif key.startswith("loss/mse/"):
            mapping[key] = key
        elif key.startswith("loss/auxk/"):
            mapping[key] = key.replace("auxk", "aux")
        elif key.startswith("train/dead/abs_"):
            mapping[key] = f"dead/{key.rsplit('_', 1)[1]}"
    return mapping


@pytest.mark.parametrize("name", NAMES)
def test_training_matches_original(name: str) -> None:
    golden = _load(name)
    exact = _exact_expected(golden)
    init_state, rec = _run(golden)
    boundaries = [0, *golden["spec"].get("sizes", [])]

    for old, expected in golden["init_state"].items():
        _assert_same(init_state[_new_state_name(old, boundaries)], expected, True, f"init {old}")

    for step in range(golden["n_steps"]):
        _assert_same(rec.loss[step], golden["loss"][step], exact, f"loss step {step}")
        old_metrics = golden["metrics"][step]
        for old, new in _metric_map(old_metrics).items():
            _assert_same(rec.metrics[step][new], old_metrics[old], exact, f"{old} step {step}")
        if golden["alloc"]:
            for key, expected in golden["alloc"][step].items():
                layer = boundaries.index(int(key))
                _assert_same(rec.parent_index[step][layer], expected, True, f"allocation {key} step {step}")
        else:
            _assert_same(rec.steps_since_fired[step], golden["last_nonzero"][step], True, f"last_nonzero step {step}")

    for old, expected in golden["final_state"].items():
        _assert_same(rec.final_state[_new_state_name(old, boundaries)], expected, exact, f"final {old}")

    if golden["tree_trackers"] is not None:
        trackers = golden["tree_trackers"]
        for key, expected in trackers["last_nonzero_sizes"].items():
            _assert_same(
                rec.tree_state["steps_since_fired"][boundaries.index(int(key))], expected, True, f"steps {key}"
            )
        for key, expected in trackers["loss_accumulate_sizes"].items():
            _assert_same(rec.tree_state["capacity"][boundaries.index(int(key))], expected, exact, f"capacity {key}")
        assert rec.tree_state["realloc_interval"] == trackers["nbatch_dynamic_alloc"]
        assert rec.tree_state["root_reset_done"] == trackers["forced_root_done"]
