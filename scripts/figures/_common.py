"""Shared setup of the analysis figure scripts: SAE, eval tokens, LLM and SAE activations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from tree_sae.analysis.activations import llm_activations, load_eval_tokens, sae_sparse_activations
from tree_sae.models import BaseSAE
from tree_sae.models.io import from_pretrained, load_sae


def add_common_args(parser: argparse.ArgumentParser, single_sae: bool = True) -> None:
    if single_sae:
        parser.add_argument(
            "--sae",
            required=True,
            help="Directory of a saved SAE or hf:<name>, e.g. hf:gpt2-small/tree_sae_2layer_l0_32",
        )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="gpt2-small")
    parser.add_argument("--tokenizer_name", default="gpt2")
    parser.add_argument("--n_seqs", type=int, default=10_000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--seed", type=int, default=42)


@dataclass
class Setup:
    model: HookedTransformer
    sae: BaseSAE
    tokens: Tensor  # (n_seqs, seq_len)
    acts: Tensor  # (n_seqs * seq_len, d_model)
    indices: Tensor  # (n_seqs * seq_len, k), inactive entries set to d_sae
    values: Tensor


def prepare(args, dtype: torch.dtype = torch.bfloat16) -> Setup:
    """Load everything in ``dtype`` (bfloat16, as in the paper analyses)."""
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device, dtype=dtype)
    location = args.sae
    sae = from_pretrained(location[3:], args.device) if location.startswith("hf:") else load_sae(location, args.device)
    sae = sae.to(dtype=dtype)
    tokens = load_eval_tokens(args.n_seqs, args.seq_len, args.tokenizer_name)
    acts = llm_activations(model, tokens, sae.cfg.hook_name)
    indices, values = sae_sparse_activations(sae, acts, 128, args.device, verbose=True)
    indices[values < 1e-3] = sae.cfg.d_sae
    return Setup(model, sae, tokens, acts, indices, values)
