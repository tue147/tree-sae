"""Draw the metric figures of the paper (Figs. 3-8 and 16).

By default the numbers of the paper (``results/paper_results.json``) are plotted; pass a file
produced by ``scripts/collect_results.py`` to plot your own evaluation results.

    python scripts/figures/plot_metrics.py --results results/paper_results.json --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tree_sae.analysis.plotting import plot_metrics  # noqa: E402

LOWER, HIGHER = "Lower is better", "Higher is better"
BENCHMARKS = {"splitting": ("Splitting", LOWER), "absorption": ("Absorption", LOWER),
              "autointerp": ("AutoInterp", HIGHER), "composition": ("Composition", LOWER)}
RECONSTRUCTION = {"variance_explained": ("$R^2$", HIGHER), "downstream_ce_loss": ("Cross Entropy", LOWER)}


def select(results: dict, metrics: dict) -> tuple[dict, dict]:
    data = {title: results[key] for key, (title, _) in metrics.items() if key in results}
    return data, {title: label for title, label in metrics.values()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/paper_results.json")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--format", default="pdf")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    main_res, scaling = results["main"], results.get("scaling")
    # MP-SAE is plotted at its measured average L0.
    x_main = {key: main_res["L0"] for key in ("topk", "matryoshka_2layer", "matryoshka_4layer",
                                                "tree_sae_2layer", "tree_sae_4layer")}
    x_main["mp_sae"] = main_res.get("mp_sae_average_L0", main_res["L0"])
    figures = {}

    data, labels = select(main_res, BENCHMARKS)
    figures["fig3_benchmarks"] = plot_metrics(data, x_main, labels, "$L_0$")
    data, labels = select(main_res, RECONSTRUCTION)
    figures["fig6_reconstruction"] = plot_metrics(data, x_main, labels, "$L_0$", figsize=(10, 4.5), legend_ncol=3)
    hierarchy_x = {k: x_main.get(k, main_res["L0"]) for k in main_res["hierarchy"]}
    figures["fig4_hierarchy"] = plot_metrics({"Hierarchy": main_res["hierarchy"]}, hierarchy_x,
                                             {"Hierarchy": HIGHER}, "$L_0$", figsize=(8, 4.5), legend="right")
    if "hierarchy_mcs_variants" in main_res:
        figures["fig16_mcs_variants"] = plot_metrics({"Hierarchy": main_res["hierarchy_mcs_variants"]},
                                                     main_res["L0"], {"Hierarchy": HIGHER}, "$L_0$",
                                                     figsize=(8, 4), legend="right", right_legend_content=0.72)
    if scaling:
        sizes = scaling["dictionary_size"]
        data, labels = select(scaling, BENCHMARKS)
        figures["fig8_benchmarks_scaling"] = plot_metrics(data, sizes, labels, "Size", legend_ncol=4)
        data, labels = select(scaling, RECONSTRUCTION)
        figures["fig7_reconstruction_scaling"] = plot_metrics(data, sizes, labels, "Size", figsize=(10, 4.5),
                                                              legend_ncol=2)
        figures["fig5_hierarchy_scaling"] = plot_metrics({"Hierarchy": scaling["hierarchy"]}, sizes,
                                                         {"Hierarchy": HIGHER}, "Size", figsize=(8, 4.5),
                                                         legend="right")
    for name, fig in figures.items():
        path = out / f"{name}.{args.format}"
        fig.savefig(path, bbox_inches="tight")
        print(path)


if __name__ == "__main__":
    main()
