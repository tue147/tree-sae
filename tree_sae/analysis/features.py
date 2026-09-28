"""Inspect individual features: firing rate, max-activating examples, activation histograms."""

from __future__ import annotations

import einops
import torch
from torch import Tensor
from transformer_lens import HookedTransformer


def firing_rate(feature: int, indices: Tensor, values: Tensor, threshold: float = 1e-3) -> float:
    """Fraction of tokens on which ``feature`` fires; ``indices``/``values`` are ``(batch, seq, k)``."""
    n_active = ((indices == feature) & (values > threshold)).sum().item()
    batch, seq = indices.shape[:-1]
    return n_active / (batch * seq)


def top_positions(x: Tensor, k: int, buffer: int = 0, no_overlap: bool = True) -> Tensor:
    """``(k, 2)`` ``(sequence, position)`` indices of the ``k`` largest entries of ``x`` (``batch, seq``).

    Positions within ``buffer`` of either end of a sequence are skipped; with ``no_overlap`` no two
    selected positions of the same sequence are within ``buffer`` of each other.
    """
    assert buffer * 2 < x.size(1), "buffer is too large for the sequence length"
    if buffer > 0:
        x = x[:, buffer:-buffer]
    order = x.flatten().argsort(-1, descending=True)
    rows, cols = order // x.size(1), order % x.size(1) + buffer
    if not no_overlap:
        return torch.stack((rows, cols), dim=1)[:k]
    selected = torch.empty((0, 2), device=x.device).long()
    while len(selected) < k and len(rows) > 0:
        selected = torch.cat((selected, torch.tensor([[rows[0], cols[0]]], device=x.device)))
        overlapping = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
        rows, cols = rows[~overlapping], cols[~overlapping]
    return selected[:k]


def index_with_buffer(x: Tensor, positions: Tensor, buffer: int | None = None) -> Tensor:
    """Entries of ``x`` at ``positions`` with ``buffer`` tokens of context on each side."""
    rows, cols = positions.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k window", window=buffer * 2 + 1)
        cols = cols.clone()
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k window", window=buffer * 2 + 1) + torch.arange(
            -buffer, buffer + 1, device=cols.device
        )
    return x[rows, cols]


def max_activating_examples(
    model: HookedTransformer,
    feature: int,
    indices: Tensor,
    values: Tensor,
    tokens: Tensor,
    k: int = 10,
    buffer: int = 10,
) -> list[tuple[float, list[str], int]]:
    """Top ``k`` ``(activation, context tokens, position of the token in the context)``.

    ``indices``/``values`` are ``(n_seqs, seq_len, k)`` sparse feature activations of ``tokens``.
    """
    per_token = torch.max(torch.where(indices == feature, values, torch.zeros_like(values)), dim=-1).values
    positions = top_positions(per_token, k=k, buffer=buffer)
    contexts = index_with_buffer(tokens, positions, buffer=buffer)
    activations = index_with_buffer(per_token, positions).tolist()
    examples = [(act, model.to_str_tokens(ctx), buffer) for act, ctx in zip(activations, contexts)]
    return sorted(examples, key=lambda x: x[0], reverse=True)[:k]


def print_examples(examples: list[tuple[float, list[str], int]]) -> None:
    """Print examples with the activating token highlighted (rich table)."""
    from rich import print as rprint
    from rich.table import Table

    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, pos in examples:
        text = "".join(f"[b u green]{t}[/]" if i == pos else t for i, t in enumerate(str_toks))
        table.add_row(f"{act:.3f}", repr(text.replace("�", "").replace("\n", "↵")))
    rprint(table)


def activation_histogram(feature: int, indices: Tensor, values: Tensor, threshold: float = 1e-3, **layout):
    """Plotly histogram of the non-zero activations of ``feature``."""
    import plotly.express as px

    active = values[(indices == feature) & (values > threshold)]
    fig = px.histogram(
        active.float().cpu().numpy(),
        nbins=50,
        title="Activation density",
        labels={"value": "Activation"},
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
    )
    fig.update_layout(bargap=0.02, showlegend=False, **layout)
    return fig
