"""Geometry of the child feature subspace of a Tree SAE parent (Fig. 9).

python scripts/figures/child_geometry.py --sae hf:gpt2-small/tree_sae_4layer_l0_32 --parent 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _common import add_common_args, prepare  # noqa: E402

from tree_sae.analysis.geometry import plot_child_geometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--vectors", choices=["encoder", "decoder"], default="encoder")
    parser.add_argument("--vector_length", type=float, default=10.0)
    args = parser.parse_args()

    s = prepare(args)
    ax, pca = plot_child_geometry(
        s.sae,
        s.acts,
        s.indices,
        args.parent,
        vectors=args.vectors,
        vector_length=args.vector_length,
        title="PCA of child features",
    )
    print("explained variance ratio:", pca.explained_variance_ratio_)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f"child_geometry_parent{args.parent}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
