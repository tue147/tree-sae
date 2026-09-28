"""Training configuration (one YAML file per SAE in ``configs/``)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SAESpec:
    """Architecture of the SAE; ``type`` is one of ``relu``, ``topk``, ``matryoshka``, ``mp``, ``tree``."""

    type: str
    d_sae: int | None = None  # relu / topk / mp (matryoshka and tree use sum(features_per_layer))
    k: int | None = None  # topk / matryoshka / mp
    features_per_layer: list[int] | None = None  # matryoshka / tree
    k_per_layer: list[int] | None = None  # tree
    auxk: int | None = 256
    auxk_coef: float = 1 / 32
    l1_coef: float = 1 / 16  # relu
    standardize: bool = True
    dead_threshold: float = 1e-3
    # Tree SAE (Appendix C / G)
    parent_eligibility_tokens: int = 50_000
    realloc_interval: int = 3000
    realloc_interval_growth: float = 2.0
    max_realloc_interval: int = 10_000
    root_reset_step: int | None = 50_000
    root_init_frac: float = 0.0
    aux_layers: list[int] = field(default_factory=lambda: [0])


@dataclass
class TrainConfig:
    name: str
    sae: SAESpec
    model_name: str = "gpt2-small"
    tokenizer_name: str = "gpt2"
    model_dtype: str = "float32"
    hook_name: str = "blocks.5.hook_resid_pre"
    dataset: str = "monology/pile-uncopyrighted"
    batch_size: int = 5  # sequences per step
    seq_len: int = 1024
    max_tokens: int = 500_000_000
    lr: float = 1e-4
    use_loss_var: bool = True
    dead_tokens_threshold: int = 10_000_000
    precision: str = "16-mixed"
    seed: int = 42
    num_workers: int = 1
    # Checkpointing / logging
    output_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 50
    early_stopping_min_delta: float = 0.005
    early_stopping_patience: int = 1
    wandb_project: str | None = "tree-sae"

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def max_steps(self) -> int:
        return -(-self.max_tokens // self.tokens_per_step)

    @property
    def dead_steps_threshold(self) -> int:
        return self.dead_tokens_threshold // self.tokens_per_step

    @property
    def parent_eligibility_steps(self) -> int:
        return self.sae.parent_eligibility_tokens // self.tokens_per_step

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = dict(data)
        data["sae"] = SAESpec(**data["sae"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text())
        data.update(overrides)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
