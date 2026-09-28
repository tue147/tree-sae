"""Collect LLM activations and sparse SAE feature activations for evaluation and analysis."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ..models import BaseSAE, MatryoshkaSAE, TopKSAE, TreeSAE
from ..training.data import token_dataloader

PILE = "monology/pile-uncopyrighted"


def load_eval_tokens(n_seqs: int, seq_len: int, tokenizer_name: str, dataset: str = PILE) -> Tensor:
    """First ``n_seqs`` sequences of ``seq_len`` tokens of the Pile validation split."""
    loader = token_dataloader(dataset, tokenizer_name, seq_len, batch_size=1, split="val")
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_seqs:
            break
        batches.append(batch[0])
    return torch.cat(batches, dim=0)


@torch.no_grad()
def llm_activations(
    model: HookedTransformer, tokens: Tensor, hook_name: str, batch_size: int = 32, verbose: bool = True
) -> Tensor:
    """Activations at ``hook_name`` for every token, flattened to ``(n_seqs * seq_len, d_model)`` on CPU."""
    cache = []

    def hook_fn(acts: Tensor, hook) -> None:
        cache.append(acts.reshape(acts.shape[0] * acts.shape[1], -1).cpu().detach())

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        for start in tqdm(range(0, len(tokens), batch_size), disable=not verbose, desc="LLM activations"):
            model(tokens[start : start + batch_size])
    return torch.cat(cache)


@torch.no_grad()
def sae_sparse_activations(
    sae: BaseSAE, acts: Tensor, batch_size: int, device: str | torch.device, verbose: bool = False
) -> tuple[Tensor, Tensor]:
    """``(indices, values)`` of the active features of every row of ``acts`` (on CPU).

    Top-k style SAEs return their top-``k`` selection (values may be zero). For the other SAEs the
    active features are read from ``hook_sae_acts_post`` and padded to the largest L0 in ``acts``.
    """
    sae.eval()
    sae.to(device)
    loader = DataLoader(TensorDataset(acts), batch_size=batch_size, shuffle=False)
    indices, values = [], []
    if isinstance(sae, (TreeSAE, TopKSAE, MatryoshkaSAE)):
        for (batch,) in tqdm(loader, disable=not verbose):
            topk = sae.forward_training(batch.to(device)).topk
            indices.append(topk.indices.cpu())
            values.append(topk.values.cpu())
        return torch.cat(indices), torch.cat(values)

    latents_list, max_active = [], 0

    def hook_fn(latents: Tensor, hook) -> None:
        nonlocal max_active
        mask = latents > 0
        max_active = max(max_active, mask.sum(-1).max().item())
        coords = mask.nonzero(as_tuple=False).t()
        latents_list.append(torch.sparse_coo_tensor(coords, latents[mask], latents.shape).coalesce().detach().cpu())

    for (batch,) in tqdm(loader, disable=not verbose):
        sae.run_with_hooks(batch.to(device), fwd_hooks=[(lambda name: "acts_post" in name, hook_fn)])
    for sparse in latents_list:
        v, i = torch.topk(sparse.to_dense(), k=max_active, sorted=False)
        indices.append(i)
        values.append(v)
    return torch.cat(indices), torch.cat(values)


def dense_from_sparse(d_sae: int, indices: Tensor, values: Tensor, device: str | torch.device = "cpu") -> Tensor:
    """Scatter ``(indices, values)`` back into dense ``(..., d_sae)`` feature activations."""
    dense = torch.zeros((*indices.shape[:-1], d_sae), device=device, dtype=values.dtype)
    return dense.scatter_(-1, indices.to(device), values.to(device))


def feature_frequencies(indices: Tensor, values: Tensor, d_sae: int, threshold: float = 1e-3) -> Tensor:
    """Number of rows on which every feature fires (activation > ``threshold``)."""
    active = indices[values > threshold]
    return torch.bincount(active.flatten(), minlength=d_sae)
