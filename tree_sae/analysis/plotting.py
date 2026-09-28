"""Plot style of the paper's metric figures (Figs. 3-8, 16)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

RED, BLUE, GREEN, ORANGE = "#ec0c0c", "#3d29d3", "#2ca02c", "#dc8d0e"

# series key -> (label, color, marker, line style, alpha)
STYLE = {
    "topk": ("TopK", GREEN, "^", "-", 0.4),
    "mp_sae": ("Matching Pursuit", ORANGE, "D", "-", 0.4),
    "matryoshka_2layer": ("Matryoshka 2 layers", BLUE, "s", ":", 0.4),
    "matryoshka_4layer": ("Matryoshka 4 layers", BLUE, "s", "-", 0.4),
    "tree_sae_2layer": ("Tree 2 layers", RED, "o", ":", 1.0),
    "tree_sae_4layer": ("Tree 4 layers", RED, "o", "-", 1.0),
    "tree_sae_2layer_mcs": ("Tree 2 layers (MCS)", RED, "+", ":", 1.0),
    "tree_sae_4layer_mcs": ("Tree 4 layers (MCS)", RED, "+", "-", 1.0),
    "tree_sae_2layer_tree_structure": ("Tree 2 layers (Structure)", RED, "x", ":", 1.0),
    "tree_sae_4layer_tree_structure": ("Tree 4 layers (Structure)", RED, "x", "-", 1.0),
}
for variant, color in (
    ("binary", RED),
    ("scaled_binary", BLUE),
    ("correlation", GREEN),
    ("scaled_correlation", ORANGE),
):
    for layers, style in (("2layer", ":"), ("4layer", "-")):
        label = f"Tree {layers[0]} layers {variant.replace('_', ' ').title().replace('Scaled', 'Scale')}"
        STYLE[f"tree_sae_{layers}_{variant}"] = (label, color, "o", style, 1.0)

PARAMS = {
    "font.size": 14,
    "title_size": 18,
    "xlabel_size": 20,
    "ylabel_size": 18,
    "xtick_size": 16,
    "ytick_size": 14,
    "legend_size": 16,
    "line_width": 2,
    "marker_size": 8,
}


def plot_metrics(
    data: dict[str, dict[str, list[float]]],
    x: dict[str, list] | list,
    y_labels: dict[str, str],
    x_label: str,
    ncols: int | None = None,
    figsize: tuple[float, float] = (18, 4),
    legend: str = "bottom",
    legend_ncol: int = 6,
    right_legend_content: float = 0.78,
):
    """One panel per metric; one line per series, with its average marked at the right edge.

    Args:
        data: ``{metric title: {series key: values}}``; series keys index ``STYLE``.
        x: x values shared by all series, or ``{series key: x values}``.
    """
    plt.rcParams.update({"font.size": PARAMS["font.size"]})
    n = len(data)
    ncols = ncols or n
    fig, axes = plt.subplots((n + ncols - 1) // ncols, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    handles, labels = [], []
    for ax, (metric, series) in zip(axes, data.items()):
        xs_all = []
        for key, values in series.items():
            label, color, marker, style, alpha = STYLE[key]
            xs = x[key] if isinstance(x, dict) else x
            xs_all.append(xs)
            ax.plot(
                xs,
                values,
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=PARAMS["line_width"],
                markersize=PARAMS["marker_size"],
                alpha=alpha,
            )
            if label not in labels:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=color,
                        marker=marker,
                        linestyle=style,
                        linewidth=PARAMS["line_width"],
                        markersize=PARAMS["marker_size"],
                    )
                )
                labels.append(label)
        ax.set_title(metric, fontsize=PARAMS["title_size"])
        ax.set_xlabel(x_label, fontsize=PARAMS["xlabel_size"])
        ax.set_ylabel(y_labels[metric], fontsize=PARAMS["ylabel_size"])
        ax.set_xticks(xs_all[0])
        ax.tick_params(axis="x", labelsize=PARAMS["xtick_size"])
        ax.tick_params(axis="y", labelsize=PARAMS["ytick_size"])
        if metric == "$R^2$":
            values = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
            pad = max(0.0005, (values.max() - values.min()) * 0.12)
            ax.set_ylim(values.min() - pad, 1.0)
            ax.margins(y=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(linestyle="--", alpha=0.3)

    if legend == "right":
        plt.tight_layout(rect=(0, 0, right_legend_content, 1))
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(right_legend_content + 0.01, 0.5),
            ncol=1,
            fontsize=PARAMS["legend_size"],
            frameon=True,
        )
    else:
        plt.tight_layout()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=legend_ncol,
            fontsize=PARAMS["legend_size"],
            frameon=True,
        )
        plt.subplots_adjust(bottom=0.15)

    # Averages over the x axis, drawn just right of each panel.
    for ax, series in zip(axes, data.values()):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim((x_min, x_max + x_max * 0.05))
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks([])
        for key, values in series.items():
            _, color, _, style, _ = STYLE[key]
            right.plot(
                [0.95, 1.04],
                [np.mean(values)] * 2,
                transform=ax.get_yaxis_transform(),
                color=color,
                linestyle=style,
                linewidth=PARAMS["line_width"],
            )
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig
