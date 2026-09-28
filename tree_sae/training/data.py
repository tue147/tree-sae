"""Streaming token data and on-the-fly LLM activations."""

from __future__ import annotations

import math

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _collate(batch) -> tuple[Tensor]:
    return (torch.stack([item["input_ids"] for item in batch]),)


def _concat_and_tokenize(batch, tokenizer: PreTrainedTokenizerBase, max_length: int) -> dict:
    """Join documents with EOS and cut the token stream into ``max_length`` chunks.

    Based on https://github.com/EleutherAI/sae/blob/19d95a4/sae/data.py.
    """
    output = tokenizer(
        tokenizer.eos_token.join([""] + batch["text"]),
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_overflowing_tokens=True,
    )
    overflowing = output.pop("overflowing_tokens", None)
    output.pop("overflow_to_sample_mapping", None)
    if overflowing is not None:
        output["input_ids"] += [
            overflowing[i * max_length : (i + 1) * max_length] for i in range(math.ceil(len(overflowing) / max_length))
        ]
    # Drop the last (probably incomplete) chunk.
    return {k: v[:-1] for k, v in output.items()}


def token_dataloader(
    dataset: str, tokenizer_name: str, max_length: int, batch_size: int, split: str = "train", num_workers: int = 1
) -> DataLoader:
    """Stream ``dataset`` (train split, or ``val``/``test`` files of the Pile) as token batches."""
    if split == "train":
        data = load_dataset(dataset, split="train", streaming=True)
    else:
        name = {"val": "validation", "test": "test"}[split]
        data = load_dataset(dataset, data_files={name: f"{split}.jsonl.zst"}, split=name, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_length = min(tokenizer.model_max_length, max_length)
    data = data.map(
        _concat_and_tokenize,
        batched=True,
        batch_size=1024,  # large batches drop fewer tokens
        remove_columns=data.column_names or ["text", "meta"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    ).with_format("torch")
    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)


class LLMActivations:
    """Callable mapping a token batch to the activations of ``model`` at ``hook_name``."""

    def __init__(self, model: HookedTransformer, hook_name: str) -> None:
        self.model = model
        self.hook_name = hook_name
        # Later blocks cannot influence ``blocks.{i}.*`` activations, so skip them.
        parts = hook_name.split(".")
        self.stop_at_layer = int(parts[1]) + 1 if parts[0] == "blocks" else None

    @torch.no_grad()
    def __call__(self, batch: tuple[Tensor]) -> Tensor:
        cache = {}

        def hook_fn(acts: Tensor, hook) -> None:
            cache["acts"] = acts.detach()

        self.model.run_with_hooks(batch[0], fwd_hooks=[(self.hook_name, hook_fn)], stop_at_layer=self.stop_at_layer)
        return cache["acts"].to(self.model.cfg.device)
