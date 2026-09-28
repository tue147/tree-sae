"""Build an SAE from a config and train it on streamed LLM activations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from transformer_lens import HookedTransformer

from ..models import SAE_CLASSES, BaseSAE
from ..models.io import save_sae
from .config import SAESpec, TrainConfig
from .data import LLMActivations, token_dataloader
from .module import SAETrainingModule


def seed_all(seed: int) -> None:
    """Seed every RNG and set the numerics used for all paper runs."""
    seed_everything(seed=seed, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("high")


def build_sae(
    spec: SAESpec, d_in: int, hook_name: str, dead_steps_threshold: int, parent_eligibility_steps: int
) -> BaseSAE:
    common = dict(
        d_in=d_in,
        hook_name=hook_name,
        dead_steps_threshold=dead_steps_threshold,
        dead_threshold=spec.dead_threshold,
        standardize=spec.standardize,
    )
    cls = SAE_CLASSES[spec.type]
    if spec.type == "relu":
        return cls(d_sae=spec.d_sae, auxk=spec.auxk, l1_coef=spec.l1_coef, **common)
    if spec.type == "topk":
        return cls(d_sae=spec.d_sae, k=spec.k, auxk=spec.auxk, **common)
    if spec.type == "mp":
        return cls(d_sae=spec.d_sae, k=spec.k, **common)
    if spec.type == "matryoshka":
        return cls(features_per_layer=spec.features_per_layer, k=spec.k, auxk=spec.auxk, **common)
    if spec.type == "tree":
        return cls(
            features_per_layer=spec.features_per_layer,
            k_per_layer=spec.k_per_layer,
            auxk=spec.auxk,
            parent_eligibility_steps=parent_eligibility_steps,
            realloc_interval=spec.realloc_interval,
            realloc_interval_growth=spec.realloc_interval_growth,
            max_realloc_interval=spec.max_realloc_interval,
            root_reset_step=spec.root_reset_step,
            root_init_frac=spec.root_init_frac,
            aux_layers=tuple(spec.aux_layers),
            **common,
        )
    raise ValueError(f"Unknown SAE type {spec.type!r}")


def train(cfg: TrainConfig, device: str = "cuda", max_steps: int | None = None, use_wandb: bool = True) -> BaseSAE:
    """Train the SAE described by ``cfg`` and save it to ``{output_dir}/{name}/``.

    Following the paper runs, the Lightning checkpoint with the lowest logged training loss among
    the periodic checkpoints (every ``save_every_n_steps``) is kept, and training stops early if the
    loss stops improving by ``early_stopping_min_delta``.
    """
    dtype = getattr(torch, cfg.model_dtype)
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device, dtype=dtype)
    train_loader = token_dataloader(
        cfg.dataset, cfg.tokenizer_name, cfg.seq_len, cfg.batch_size, "train", cfg.num_workers
    )
    val_loader = token_dataloader(cfg.dataset, cfg.tokenizer_name, cfg.seq_len, cfg.batch_size, "val", cfg.num_workers)

    seed_all(cfg.seed)
    sae = build_sae(cfg.sae, model.cfg.d_model, cfg.hook_name, cfg.dead_steps_threshold, cfg.parent_eligibility_steps)
    module = SAETrainingModule(
        sae,
        lr=cfg.lr,
        auxk_coef=cfg.sae.auxk_coef,
        use_loss_var=cfg.use_loss_var,
        preprocess=LLMActivations(model, cfg.hook_name),
    )

    out_dir = Path(cfg.output_dir) / cfg.name
    logger = False
    if use_wandb and cfg.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(name=cfg.name, project=cfg.wandb_project, save_dir="wandb_logs", log_model=False)
    trainer = Trainer(
        max_epochs=1,
        max_steps=max_steps or cfg.max_steps,
        precision=cfg.precision,
        accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=[int(device.split(":")[1])] if ":" in device else 1,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[
            EarlyStopping(
                monitor="loss/total",
                min_delta=cfg.early_stopping_min_delta,
                patience=cfg.early_stopping_patience,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                monitor="loss/total",
                dirpath=out_dir,
                filename="lightning",
                save_top_k=1,
                mode="min",
                save_weights_only=True,
                enable_version_counter=False,
                every_n_train_steps=cfg.save_every_n_steps,
            ),
        ],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = trainer.checkpoint_callback.best_model_path
    if best:
        state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(state)
    save_sae(sae, out_dir, extra={"train_config": cfg.to_dict()})
    (out_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    return sae
