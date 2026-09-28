"""Feature absorption and splitting on the first-letter task (Chanin et al., 2024; SAEBench).

Adapted from SAEBench (https://github.com/adamkarvonen/SAEBench, MIT license). The paper reports
``mean_full_absorption_score`` as *Absorption* and ``mean_num_split_features`` as *Splitting*.
"""

from __future__ import annotations

import gc
import os
import random
import statistics
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from ...models import BaseSAE
from .eval_config import AbsorptionEvalConfig
from .eval_output import (
    EVAL_TYPE_ID_ABSORPTION,
    AbsorptionEvalOutput,
    AbsorptionMeanMetrics,
    AbsorptionMetricCategories,
    AbsorptionResultDetail,
)
from .common import RESULTS_DIR
from .feature_absorption import FEATURE_ABSORPTION_EXPERIMENT_NAME, run_feature_absortion_experiment
from .k_sparse_probing import SPARSE_PROBING_EXPERIMENT_NAME, run_k_sparse_probing_experiment


def _aggregate_results_df(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        df[["letter", "absorption_fraction", "is_full_absorption"]]
        .groupby(["letter"])
        .sum()
        .reset_index()
        .merge(
            df[["letter", "num_probe_true_positives", "split_feats"]]
            .groupby(["letter"])
            .agg({"num_probe_true_positives": "mean", "split_feats": lambda x: x.iloc[0]})
            .reset_index()
        )
    )
    agg_df["num_split_feats"] = agg_df["split_feats"].apply(len)
    agg_df["mean_absorption_fraction"] = agg_df["absorption_fraction"] / agg_df["num_probe_true_positives"]
    agg_df["num_full_absorption"] = agg_df["is_full_absorption"]
    agg_df["full_absorption_rate"] = agg_df["num_full_absorption"] / agg_df["num_probe_true_positives"]
    return agg_df


def run_absorption_eval(
    config: AbsorptionEvalConfig,
    saes: list[tuple[str, BaseSAE]],
    layer: int,
    device: str,
    output_dir: str,
    force_rerun: bool = False,
    artifacts_dir: str | Path = RESULTS_DIR,
) -> dict[str, dict]:
    """Run the k-sparse probing (splitting) and absorption experiments for every ``(name, sae)``.

    Intermediate probes and data frames are cached under ``artifacts_dir`` (default
    ``$TREE_SAE_ARTIFACTS/absorption`` or ``./artifacts/absorption``); the first-letter probe is
    shared by all SAEs of a model/layer.
    """
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.set_grad_enabled(True)

    artifacts_dir = Path(artifacts_dir)
    probes_dir = artifacts_dir / "probes"
    sparse_probing_dir = artifacts_dir / SPARSE_PROBING_EXPERIMENT_NAME
    eval_id = str(uuid.uuid4())
    llm_dtype = getattr(torch, config.llm_dtype)
    model = HookedTransformer.from_pretrained_no_processing(config.model_name, device=device, dtype=llm_dtype)
    results = {}
    for sae_name, sae in saes:
        sae = sae.to(device=device, dtype=llm_dtype)
        result_path = os.path.join(output_dir, sae_name)
        if os.path.exists(result_path) and not force_rerun:
            print(f"Skipping {sae_name}: {result_path} exists")
            continue
        os.makedirs(output_dir, exist_ok=True)

        k_sparse_results = run_k_sparse_probing_experiment(
            model=model,
            sae=sae,
            layer=layer,
            sae_name=sae_name,
            batch_size=config.llm_batch_size,
            force=force_rerun,
            max_k_value=config.max_k_value,
            f1_jump_threshold=config.f1_jump_threshold,
            prompt_template=config.prompt_template,
            prompt_token_pos=config.prompt_token_pos,
            device=device,
            k_sparse_probe_l1_decay=config.k_sparse_probe_l1_decay,
            k_sparse_probe_batch_size=config.k_sparse_probe_batch_size,
            k_sparse_probe_num_epochs=config.k_sparse_probe_num_epochs,
            experiment_dir=sparse_probing_dir,
            probes_dir=probes_dir,
        )
        if (k_sparse_results["f1_probe"] > config.min_GT_probe_f1).sum() < config.min_feats_for_eval:
            print(f"Cannot evaluate absorption: too few first-letter features in {config.model_name}")
            break

        raw_df = run_feature_absortion_experiment(
            model=model,
            sae=sae,
            layer=layer,
            sae_name=sae_name,
            force=force_rerun,
            max_k_value=config.max_k_value,
            feature_split_f1_jump_threshold=config.f1_jump_threshold,
            prompt_template=config.prompt_template,
            prompt_token_pos=config.prompt_token_pos,
            batch_size=config.llm_batch_size,
            device=device,
            experiment_dir=artifacts_dir / FEATURE_ABSORPTION_EXPERIMENT_NAME,
            sparse_probing_experiment_dir=sparse_probing_dir,
            probes_dir=probes_dir,
        )

        absorption_fractions, full_absorption_rates, n_split_features, details = [], [], [], []
        for _, row in _aggregate_results_df(raw_df).iterrows():
            probe_f1 = k_sparse_results[k_sparse_results["letter"] == row["letter"]]["f1_probe"].item()
            if probe_f1 <= config.min_GT_probe_f1:
                continue
            absorption_fractions.append(row["mean_absorption_fraction"])
            full_absorption_rates.append(row["full_absorption_rate"])
            n_split_features.append(row["num_split_feats"])
            details.append(
                AbsorptionResultDetail(
                    first_letter=row["letter"],
                    mean_absorption_fraction=row["mean_absorption_fraction"],
                    full_absorption_rate=row["full_absorption_rate"],
                    num_full_absorption=row["num_full_absorption"],
                    num_probe_true_positives=row["num_probe_true_positives"],
                    num_split_features=row["num_split_feats"],
                )
            )

        output = AbsorptionEvalOutput(
            eval_type_id=EVAL_TYPE_ID_ABSORPTION,
            eval_config=config,
            eval_id=eval_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=AbsorptionMetricCategories(
                mean=AbsorptionMeanMetrics(
                    mean_absorption_fraction_score=statistics.mean(absorption_fractions),
                    mean_full_absorption_score=statistics.mean(full_absorption_rates),
                    mean_num_split_features=statistics.mean(n_split_features),
                    std_dev_absorption_fraction_score=statistics.stdev(absorption_fractions),
                    std_dev_full_absorption_score=statistics.stdev(full_absorption_rates),
                    std_dev_num_split_features=statistics.stdev(n_split_features),
                )
            ),
            eval_result_details=details,
            sae_bench_commit_hash=sae_name,
            sae_lens_id=sae_name,
            sae_lens_version="_",
            sae_lens_release_id="local",
            sae_cfg_dict=asdict(sae.cfg),
        )
        results[sae_name] = asdict(output)
        output.to_json_file(result_path, indent=2)
        gc.collect()
        torch.cuda.empty_cache()
    return results
