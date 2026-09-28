"""Data and activation helpers for AutoInterp (adapted from SAEBench, MIT license)."""


from typing import Any

import einops
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from ...models import BaseSAE
from ..utils import get_sae_acts


def get_dataset_list_of_strs(
    dataset_name: str, column_name: str, min_row_chars: int, total_chars: int
) -> list[str]:
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    total_chars_so_far = 0
    result = []

    for row in dataset:
        if len(row[column_name]) > min_row_chars:  # type: ignore
            result.append(row[column_name])  # type: ignore
            total_chars_so_far += len(row[column_name])  # type: ignore
            if total_chars_so_far > total_chars:
                break

    return result


def load_and_tokenize_dataset(
    dataset_name: str,
    ctx_len: int,
    num_tokens: int,
    tokenizer: Any,
    column_name: str = "text",
    add_bos: bool = True,
) -> torch.Tensor:
    dataset = get_dataset_list_of_strs(dataset_name, column_name, 100, num_tokens * 5)

    tokens = tokenize_and_concat_dataset(
        tokenizer, dataset, ctx_len, add_bos=add_bos, max_tokens=num_tokens
    )

    assert (tokens.shape[0] * tokens.shape[1]) > num_tokens

    return tokens


def tokenize_and_concat_dataset(
    tokenizer: Any,
    dataset: list[str],
    seq_len: int,
    add_bos: bool = True,
    max_tokens: int | None = None,
) -> torch.Tensor:
    full_text = tokenizer.eos_token.join(dataset)  # type: ignore

    # divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [
        full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)
    ]
    tokens = tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()  # type: ignore

    # remove pad token
    tokens = tokens[tokens != tokenizer.pad_token_id]  # type: ignore

    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len

    # drop last batch if not full
    tokens = tokens[: num_batches * seq_len]
    tokens = einops.rearrange(
        tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
    )

    if add_bos:
        tokens[:, 0] = tokenizer.bos_token_id  # type: ignore
    return tokens


# NOTE: the jaxtyping/beartype decorators are kept on purpose. beartype checks ``list[int]``
# arguments by sampling an element with Python's global ``random``, which shifts the random state
# used later to shuffle the AutoInterp scoring examples; keeping them reproduces the paper runs.
@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_bos_pad_eos_mask(
    tokens: Int[torch.Tensor, "dataset_size seq_len"], tokenizer: AutoTokenizer | Any
) -> Bool[torch.Tensor, "dataset_size seq_len"]:
    mask = (
        (tokens == tokenizer.pad_token_id)  # type: ignore
        | (tokens == tokenizer.eos_token_id)  # type: ignore
        | (tokens == tokenizer.bos_token_id)  # type: ignore
    ).to(dtype=torch.bool)
    return ~mask


@jaxtyped(typechecker=beartype)
@torch.no_grad
def collect_sae_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    sae: BaseSAE | Any,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: list[int] | None = None,
    activation_dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "dataset_size seq_len indexed_d_sae"]:
    """Collects SAE activations for a given set of tokens.
    Note: If evaluating many SAEs, it is more efficient to use save_activations() and encode_precomputed_activations()."""
    sae_acts = []

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD: Float[torch.Tensor, "batch seq_len d_model"] = cache[hook_name]

        sae_act_BLF: Float[torch.Tensor, "batch seq_len d_sae"] = get_sae_acts(
            resid_BLD, sae, batch_size, sae.W_dec.device, verbose=False
        )

        if selected_latents is not None:
            sae_act_BLF = sae_act_BLF[:, :, selected_latents]

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

        attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)

        sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]

        if activation_dtype is not None:
            sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)

        sae_acts.append(sae_act_BLF)

    all_sae_acts_BLF = torch.cat(sae_acts, dim=0)
    return all_sae_acts_BLF


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_feature_activation_sparsity(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    sae: BaseSAE | Any,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> Float[torch.Tensor, "d_sae"]:
    """Get the activation sparsity for each SAE feature.
    Note: If evaluating many SAEs, it is more efficient to use save_activations() and get the sparsity from the saved activations."""
    device = sae.W_dec.device
    running_sum_F = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
    total_tokens = 0

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD: Float[torch.Tensor, "batch seq_len d_model"] = cache[hook_name]

        sae_act_BLF: Float[torch.Tensor, "batch seq_len d_sae"] = get_sae_acts(
            resid_BLD, sae, batch_size, sae.W_dec.device, verbose=False
        )
        # make act to zero or one
        sae_act_BLF = (sae_act_BLF > 0).to(dtype=torch.float32)

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

        attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)

        sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]
        total_tokens += attn_mask_BL.sum().item()

        running_sum_F += einops.reduce(sae_act_BLF, "B L F -> F", "sum")

    return running_sum_F / total_tokens


def get_k_largest_indices(
    x: Tensor,
    k: int,
    buffer: int = 0,
    no_overlap: bool = False,
) -> Tensor:
    """
    Args:
        x:          The 2D tensor to get the top k largest elements from.
        k:          The number of top elements to get.
        buffer:     We won't choose any elements within `buffer` from the start or end of their seq (this helps if we
                    want more context around the chosen tokens).
        no_overlap: If True, this ensures that no 2 top-activating tokens are in the same seq and within `buffer` of
                    each other.

    Returns:
        indices: The index positions of the top k largest elements.
    """
    x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = []
        seen_positions = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if (row, col) not in seen_positions:
                unique_indices.append((row, col))
                for offset in range(-buffer, buffer + 1):
                    seen_positions.add((row, col + offset))
            if len(unique_indices) == k:
                break
        rows, cols = torch.tensor(
            unique_indices, dtype=torch.int64, device=x.device
        ).unbind(dim=-1)

    return torch.stack((rows, cols), dim=1)[:k]


def get_iw_sample_indices(
    x: Tensor,
    k: int,
    buffer: int = 0,
    use_squared_values: bool = True,
) -> Tensor:
    """
    This function returns k indices from x, importance-sampled (i.e. chosen with probabilities in proportion to their
    values). This is mean to be an alternative to quantile sampling, which accomplishes a similar thing.

    Also includes an optional threshold above which we won't sample.
    """
    x = x[:, buffer:-buffer]
    if use_squared_values:
        x = x.pow(2)

    probabilities = x.flatten() / x.sum()
    indices = torch.multinomial(probabilities, k, replacement=False)

    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Tensor,
    indices: Tensor,
    buffer: int = 0,
) -> Tensor:
    """
    This function returns the tensor you get when indexing into `x` with indices, and taking a +-buffer range around
    each index. For example, if `indices` is a list of the top activating tokens (returned by `get_k_largest_indices`),
    then this function can get you the sequence context.
    """
    assert indices.ndim == 2, "indices must have 2 dimensions"
    assert indices.shape[1] == 2, "indices must have 2 columns"
    rows, cols = indices.unbind(dim=-1)
    rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
    cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + torch.arange(
        -buffer, buffer + 1, device=cols.device
    )
    return x[rows, cols]
