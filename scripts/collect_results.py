"""Collect ``evaluate.py`` outputs into the schema of ``results/paper_results.json``.

SAEs must have been evaluated under their paper names, e.g. ``--sae tree_sae_4layer_l0_32=...``.

    python scripts/collect_results.py --outputs outputs --out results/my_results.json
    python scripts/figures/plot_metrics.py --results results/my_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

MAIN = re.compile(r"^(topk|mp_sae|matryoshka_[24]layer|tree_sae_[24]layer)_(?:l0|k)_(\d+)$")
SCALING = re.compile(r"^(matryoshka_[24]layer|tree_sae_[24]layer)_(6k|49k)$")
FIELDS = {
    "absorption": ["absorption", "splitting"],
    "autointerp": ["autointerp"],
    "composition": ["composition"],
    "reconstruction": ["variance_explained", "downstream_ce_loss"],
    "cooccurrence": ["sibling_cooccurrence"],
}
L0 = [32, 48, 64, 80]
SIZES = ["6k", "24k", "49k"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="results/my_results.json")
    args = parser.parse_args()

    main_values: dict = defaultdict(lambda: defaultdict(dict))  # metric -> series -> L0 -> value
    scaling_values: dict = defaultdict(lambda: defaultdict(dict))  # metric -> series -> size -> value

    def add(name: str, metric: str, value, suffix: str = "") -> None:
        if value is None:
            return
        if m := MAIN.match(name):
            main_values[metric][m.group(1) + suffix][int(m.group(2))] = value
            if m.group(2) == "32" and m.group(1) not in ("topk", "mp_sae"):
                scaling_values[metric][m.group(1) + suffix]["24k"] = value  # 24k scaling point = the L0=32 SAE
        elif m := SCALING.match(name):
            scaling_values[metric][m.group(1) + suffix][m.group(2)] = value

    for folder, fields in FIELDS.items():
        for path in sorted(Path(args.outputs, folder).glob("*.json")):
            result = json.loads(path.read_text())
            for field in fields:
                add(path.stem, field, result.get(field))
    for path in sorted(Path(args.outputs, "hierarchy").glob("*.json")):
        result = json.loads(path.read_text())
        if path.stem.startswith("tree_sae"):
            add(path.stem, "hierarchy", result["hierarchy_mcs"], suffix="_mcs")
            add(path.stem, "hierarchy", result["hierarchy_tree_structure"], suffix="_tree_structure")
        else:
            add(path.stem, "hierarchy", result["hierarchy_mcs"])

    def series(values: dict, keys: list) -> dict:
        return {
            metric: {name: [points.get(k) for k in keys] for name, points in by_series.items()
                     if all(k in points for k in keys)}
            for metric, by_series in values.items()
        }

    out = {"main": {"x": "L0", "L0": L0, **series(main_values, L0)},
           "scaling": {"x": "dictionary size", "dictionary_size": SIZES, **series(scaling_values, SIZES)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
