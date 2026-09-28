"""Regenerate ``fixtures/train_tree_4layer_fixed_order.pt``: training goldens of the Tree SAE with the
fixed allocation order (``legacy_allocation_order=False``), produced by this package itself.

    python tests/golden/make_fixed_order_fixture.py
"""

import copy
from pathlib import Path

import test_training_golden as g
import torch

if __name__ == "__main__":
    golden = copy.deepcopy(g._load("tree_4layer"))
    golden["spec"]["legacy_allocation_order"] = False
    init_state, rec = g._run(golden)
    fixture = dict(
        golden,
        init_state=init_state,
        loss=torch.stack(rec.loss),
        final_state=rec.final_state,
        alloc=rec.parent_index,
        platform=g._cpu_model(),
        torch_version=torch.__version__,
        metrics=[{k: v for k, v in m.items() if k.startswith(("loss/", "dead/"))} for m in rec.metrics],
    )
    torch.save(fixture, Path(__file__).parent / "fixtures" / "train_tree_4layer_fixed_order.pt")
