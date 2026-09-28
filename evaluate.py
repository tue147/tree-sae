"""Evaluate SAEs with the metrics of the paper.

SAEs are given as ``name=location`` pairs, where ``location`` is a directory written by
``tree_sae.models.io.save_sae`` or a released SAE on the Hugging Face Hub (``hf:<name>``).

Examples:
    python evaluate.py hierarchy --sae tree_4l_32=hf:gpt2-small/tree_sae_4layer_l0_32
    python evaluate.py absorption --sae topk_32=hf:gpt2-small/topk_l0_32 mat_32=checkpoints/matryoshka_2layer_l0_32
    OPENAI_API_KEY=... python evaluate.py autointerp --sae tree_2l_32=hf:gpt2-small/tree_sae_2layer_l0_32

Metrics: hierarchy (Sec. 5.2), absorption + splitting (Sec. 5.1), composition (Sec. 5.1),
reconstruction (variance explained + downstream CE, Sec. 5.3), autointerp (Sec. 5.3),
cooccurrence (sibling co-occurrence of Tree SAEs, Sec. 6.3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch

from tree_sae.models import BaseSAE, TreeSAE
from tree_sae.models.io import from_pretrained, load_sae

METRICS = ["hierarchy", "absorption", "composition", "reconstruction", "autointerp", "cooccurrence"]


def load(location: str, device: str) -> BaseSAE:
    if location.startswith("hf:"):
        return from_pretrained(location.removeprefix("hf:"), device=device)
    return load_sae(location, device=device)


def layer_of(sae: BaseSAE) -> int:
    return int(sae.cfg.hook_name.split(".")[1])


def save(result: dict, output_dir: Path, metric: str, name: str) -> None:
    path = output_dir / metric / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float))
    print(f"{metric} / {name}: saved {path}")


def cached_activations(args, sae: BaseSAE, model_dtype: torch.dtype, cache: dict):
    """Pile-validation tokens and LLM activations (shared by all SAEs of one run)."""
    from transformer_lens import HookedTransformer

    from tree_sae.analysis.activations import llm_activations, load_eval_tokens

    key = (sae.cfg.hook_name, model_dtype)
    if key not in cache:
        model = HookedTransformer.from_pretrained(args.model_name, device=args.device, dtype=model_dtype)
        tokens = load_eval_tokens(args.n_seqs, args.seq_len, args.tokenizer_name)
        cache[key] = (model, tokens, llm_activations(model, tokens, sae.cfg.hook_name, args.batch_size))
    return cache[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metric", choices=METRICS)
    parser.add_argument("--sae", nargs="+", required=True, help="name=location pairs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_name", default="gpt2-small")
    parser.add_argument("--tokenizer_name", default="gpt2")
    parser.add_argument("--seq_len", type=int, default=128, help="Tokens per sequence of the eval data")
    parser.add_argument("--n_seqs", type=int, default=10_000, help="Number of eval sequences (Pile validation)")
    parser.add_argument("--batch_size", type=int, default=32, help="LLM batch size")
    parser.add_argument("--sae_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    # hierarchy
    parser.add_argument("--n_parents", type=int, default=2000)
    parser.add_argument("--dense_ratio", type=float, default=0.5)
    parser.add_argument("--max_children", type=int, default=5)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--mcs_binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mcs_scaled", action=argparse.BooleanOptionalAction, default=False)
    # absorption
    parser.add_argument(
        "--absorption_sae_hook",
        action="store_true",
        help="Run the absorption stage at the SAE's hook instead of blocks.{layer}.hook_resid_post (paper setting)",
    )
    # autointerp
    parser.add_argument("--n_latents", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    saes = [(pair.split("=", 1)[0], pair.split("=", 1)[1]) for pair in args.sae]
    cache: dict = {}

    if args.metric == "absorption":
        from tree_sae.evals.absorption import AbsorptionEvalConfig, run_absorption_eval

        loaded = [(name, load(loc, args.device)) for name, loc in saes]
        config = AbsorptionEvalConfig(
            model_name=args.model_name,
            random_seed=args.seed,
            llm_batch_size=args.batch_size,
            llm_dtype="float32",
            use_sae_hook_for_absorption=args.absorption_sae_hook,
        )
        results = run_absorption_eval(
            config, loaded, layer_of(loaded[0][1]), args.device, str(output_dir / "absorption_raw"), True
        )
        for name, result in results.items():
            metrics = result["eval_result_metrics"]["mean"]
            save(
                {
                    "absorption": metrics["mean_full_absorption_score"],
                    "splitting": metrics["mean_num_split_features"],
                    "saebench": result,
                },
                output_dir,
                "absorption",
                name,
            )
        return

    if args.metric == "autointerp":
        from tree_sae.evals.autointerp import AutoInterpEvalConfig, run_autointerp_eval

        api_key = os.environ["OPENAI_API_KEY"]
        loaded = [(name, load(loc, args.device)) for name, loc in saes]
        config = AutoInterpEvalConfig(
            model_name=args.model_name, llm_batch_size=args.batch_size, llm_dtype="float32", random_seed=args.seed
        )
        config.n_latents = args.n_latents
        results = run_autointerp_eval(
            config, loaded, layer_of(loaded[0][1]), args.device, api_key, str(output_dir / "autointerp_raw"), True
        )
        for name, result in results.items():
            save(
                {"autointerp": result["eval_result_metrics"]["autointerp"]["autointerp_score"]},
                output_dir,
                "autointerp",
                name,
            )
        return

    for name, location in saes:
        sae = load(location, args.device)
        if args.metric == "composition":
            from tree_sae.evals.composition import composition_score

            save({"composition": composition_score(sae, verbose=True)}, output_dir, "composition", name)

        elif args.metric == "reconstruction":
            from tree_sae.evals.reconstruction import evaluate_reconstruction

            model, tokens, _ = cached_activations(args, sae, torch.float32, cache)
            result = evaluate_reconstruction(model, sae, tokens, args.batch_size, args.device)
            save(dataclasses.asdict(result), output_dir, "reconstruction", name)

        elif args.metric in ("hierarchy", "cooccurrence"):
            from tree_sae.analysis.activations import sae_sparse_activations

            # As in the paper runs, the LLM and the SAE run in bfloat16 for these analyses.
            _, _, acts = cached_activations(args, sae, torch.bfloat16, cache)
            sae = sae.to(dtype=torch.bfloat16)
            indices, values = sae_sparse_activations(sae, acts, args.sae_batch_size, args.device, verbose=True)
            if args.metric == "cooccurrence":
                from tree_sae.analysis.cooccurrence import average_sibling_cooccurrence

                if not isinstance(sae, TreeSAE):
                    raise ValueError("co-occurrence is defined for Tree SAEs only")
                padded = indices.clone()
                padded[values < 1e-3] = sae.cfg.d_sae
                save(
                    {"sibling_cooccurrence": average_sibling_cooccurrence(sae, padded)},
                    output_dir,
                    "cooccurrence",
                    name,
                )
                continue

            from tree_sae.evals.hierarchy import run_hierarchy_eval

            with torch.no_grad():
                results = run_hierarchy_eval(
                    sae,
                    acts,
                    indices,
                    values,
                    args.device,
                    n_parents=args.n_parents,
                    dense_ratio=args.dense_ratio,
                    max_children=args.max_children,
                    binary=args.mcs_binary,
                    scaled=args.mcs_scaled,
                    probe_batch_size=args.probe_batch_size,
                    seed=args.seed,
                )
            save(
                {
                    "hierarchy_mcs": results["mcs"].score(),
                    "hierarchy_tree_structure": results["tree"].score() if isinstance(sae, TreeSAE) else None,
                    "pairs": {
                        k: dataclasses.asdict(v) | {"top_features": [t[:5].tolist() for t in v.top_features]}
                        for k, v in results.items()
                    },
                },
                output_dir,
                "hierarchy",
                name,
            )


if __name__ == "__main__":
    main()
