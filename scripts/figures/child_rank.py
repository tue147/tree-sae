"""Parent similarity of the 1st ... 20th MCS child (Fig. 13; Appendix E).

For each SAE, 100 parents are sampled; their top-20 MCS children are probed and the similarity of
the parent decoder vector with each child's probe direction is averaged per child rank.

    python scripts/figures/child_rank.py --saes hf:gpt2-small/topk_l0_32 hf:gpt2-small/topk_l0_48 \
        --titles '$L_0 = 32$' '$L_0 = 48$'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _common import add_common_args, prepare  # noqa: E402

from tree_sae.analysis.probe_correlation import parent_similarity_by_child_rank  # noqa: E402
from tree_sae.evals.hierarchy import run_hierarchy_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, single_sae=False)
    parser.add_argument("--saes", nargs="+", required=True)
    parser.add_argument("--titles", nargs="+", default=None)
    parser.add_argument("--n_parents", type=int, default=100)
    parser.add_argument("--n_children", type=int, default=20)
    args = parser.parse_args()

    curves = []
    for location in args.saes:
        args.sae = location
        s = prepare(args)
        results = run_hierarchy_eval(s.sae, s.acts, s.indices, s.values, args.device, n_parents=args.n_parents,
                                     max_children=args.n_children, seed=args.seed)
        curves.append(parent_similarity_by_child_rank(results["mcs"], args.n_children))

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, len(curves), sharey=True, figsize=(5.5 * len(curves), 5), squeeze=False)
    for ax, curve, title in zip(axes[0], curves, args.titles or args.saes):
        ax.plot(np.arange(len(curve)), curve, marker="o")
        ax.set_title(title, fontsize=24)
        ax.set_xticks(np.arange(0, len(curve), 2))
        ax.tick_params(axis="both", labelsize=20)
    fig.text(0.06, 0.5, "Parent correlation", va="center", rotation="vertical", fontsize=28)
    fig.text(0.5, 0.02, "Top Child", va="center", fontsize=24)
    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "child_rank.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
