"""Activation coverage vs. reconstruction condition for one parent (Figs. 1, 10, 11, 12).

Finds the children of ``--parent`` with the highest activation coverage, trains one probe per child
and plots, per child, the reconstruction score and the activation coverage.

    python scripts/figures/probe_correlation.py --sae hf:gpt2-small/tree_sae_2layer_l0_32 --parent 1343
    python scripts/figures/probe_correlation.py --sae hf:gpt2-small/tree_sae_2layer_l0_32 --parent 1343 \
        --children 7487 12625 15275
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _common import add_common_args, prepare  # noqa: E402

from tree_sae.analysis.probe_correlation import children_by_coverage, probe_report  # noqa: E402


def plot(reports, path: Path) -> None:
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 22, "axes.titlesize": 22, "axes.linewidth": 1.5})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    x = np.arange(len(reports))
    labels = [f"Feature \n{r.child}" for r in reports]
    recon = [r.reconstruction_score for r in reports]
    coverage = [r.coverage for r in reports]
    bars1 = ax1.bar(x, recon, 0.6, color="#4c72b0", edgecolor="black", linewidth=1)
    bars2 = ax2.bar(x, coverage, 0.6, color="#4c72b0", edgecolor="black", linewidth=1)
    ax1.set_title("Reconstruction Condition")
    ax1.set_ylabel("Reconstruction Score")
    ax1.set_ylim(min(0, min(recon)), max(recon) * 1.1 + 1e-3)
    ax2.set_title("Activation Coverage")
    ax2.set_ylabel("Coverage Score")
    ax2.set_ylim(0, 1.1)
    for ax, bars in ((ax1, bars1), (ax2, bars2)):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=18,
            )
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--children", type=int, nargs="*", default=None, help="Default: top children by coverage")
    parser.add_argument("--n_children", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=100, help="Minimum number of firing tokens of a child")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    s = prepare(args)
    children = args.children
    if not children:
        children, _ = children_by_coverage(s.indices, args.parent, s.sae.cfg.d_sae, args.n_children, args.min_count)
    reports = probe_report(s.sae, s.acts, s.indices, args.parent, children, args.device)
    print(f"{'child':>8} {'coverage':>9} {'sim(parent)':>12} {'sim(child)':>11} {'parent rank':>12} {'S_res':>7}")
    for r in reports:
        print(
            f"{r.child:>8} {r.coverage:>9.3f} {r.parent_similarity:>12.3f} {r.child_similarity:>11.3f} "
            f"{r.parent_rank:>12} {r.reconstruction_score:>7.3f}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot(reports, out / f"probe_correlation_parent{args.parent}.pdf")


if __name__ == "__main__":
    main()
