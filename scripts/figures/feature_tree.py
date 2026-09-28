"""Interactive HTML tree of a Tree SAE parent and its descendants (Figs. 17-20).

python scripts/figures/feature_tree.py --sae hf:gpt2-small/tree_sae_2layer_l0_48 --root 3410
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import add_common_args, prepare  # noqa: E402

from tree_sae.analysis.tree_vis import build_feature_data, save_feature_tree, subtree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--k", type=int, default=10, help="Examples per feature")
    args = parser.parse_args()

    s = prepare(args)
    tree = subtree(s.sae, args.root)
    shape = (args.n_seqs, args.seq_len, -1)
    data = build_feature_data(tree, s.model, s.indices.view(shape), s.values.view(shape), s.tokens, k=args.k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_feature_tree(tree, data, str(out / f"feature_tree_{args.root}.html"))
    print(out / f"feature_tree_{args.root}.html")


if __name__ == "__main__":
    main()
