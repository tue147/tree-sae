"""AutoInterp (Bills et al., 2023; Paulo et al., 2024) with an OpenAI judge.

Adapted from SAEBench (https://github.com/adamkarvonen/SAEBench, MIT license): an LLM explains each
latent from its top / importance-sampled activating examples and then predicts which of a shuffled
set of examples activate it; the score is the prediction accuracy. The judge is ``gpt-4o-mini``.
"""

from __future__ import annotations

import asyncio
import gc
import os
import random
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal, TypeAlias

import torch
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ...models import BaseSAE
from .config import AutoInterpEvalConfig
from .data import (
    collect_sae_activations,
    get_feature_activation_sparsity,
    get_iw_sample_indices,
    get_k_largest_indices,
    index_with_buffer,
    load_and_tokenize_dataset,
)
from .output import (
    EVAL_TYPE_ID_AUTOINTERP,
    AutoInterpEvalOutput,
    AutoInterpMetricCategories,
    AutoInterpMetrics,
)


def get_eval_uuid():
    return str(uuid.uuid4())


Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


def display_messages(messages: Messages) -> str:
    return tabulate([m.values() for m in messages], tablefmt="simple_grid", maxcolwidths=[None, 120])


def str_bool(b: bool) -> str:
    return "Y" if b else ""


def escape_slash(s: str) -> str:
    return s.replace("/", "_")


class Example:
    """
    Data for a single example sequence.
    """

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model: HookedTransformer,
    ):
        self.toks = toks
        self.str_toks = model.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)  # this is what we predict in the scoring phase

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)  # type: ignore
            )
            .replace("�", "")
            .replace("\n", "↵")
            # .replace(">><<", "")
        )


class Examples:
    """
    Data for multiple example sequences. Includes methods for shuffling seuqences, and displaying them.
    """

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(self.examples, key=lambda x: max(x.acts), reverse=True)

    def display(self, predictions: list[int] | None = None) -> str:
        """
        Displays the list of sequences. If `predictions` is provided, then it'll include a column for both "is_active"
        and these predictions of whether it's active. If not, then neither of those columns will be included.
        """
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        str_bool(ex.is_active),
                        str_bool(i + 1 in predictions),
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"] + ([] if predictions is None else ["Active?", "Predicted?"]) + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class AutoInterp:
    """
    This is a start-to-end class for generating explanations and optionally scores. It's easiest to implement it as a
    single class for the time being because there's data we'll need to fetch that'll be used in both the generation and
    scoring phases.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: HookedTransformer,
        sae: BaseSAE,
        layer: int,
        tokenized_dataset: Tensor,
        sparsity: Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.tokenized_dataset = tokenized_dataset
        self.layer = layer
        self.device = device
        self.api_key = api_key
        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            assert self.cfg.n_latents is not None
            sparsity *= cfg.total_tokens
            alive_latents = torch.nonzero(sparsity > self.cfg.dead_latent_threshold).squeeze(1).tolist()
            if len(alive_latents) < self.cfg.n_latents:
                self.latents = alive_latents
                print(
                    f"\n\n\nWARNING: Found only {len(alive_latents)} alive latents, which is less than {self.cfg.n_latents}\n\n\n"
                )
            else:
                self.latents = random.sample(alive_latents, k=self.cfg.n_latents)
        self.n_latents = len(self.latents)

        self.api_key = api_key

        # Shared async client (reuse connections). Tune timeout/max_retries to taste.
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=45.0, max_retries=0)

        # Concurrency controls
        # How many concurrent requests to OpenAI at once (adjust per your rate limits)
        self.max_parallel_requests = getattr(self.cfg, "max_parallel_requests", 5)
        self.req_sem = asyncio.Semaphore(self.max_parallel_requests)

        # Optional: how many features to process concurrently (each feature uses 1–2 LLM calls)
        self.max_concurrent_features = getattr(self.cfg, "max_concurrent_features", 50)
        self.feature_sem = asyncio.Semaphore(self.max_concurrent_features)

        # Retry policy
        self.max_retries = getattr(self.cfg, "max_retries", 6)
        self.base_backoff = getattr(self.cfg, "base_backoff", 0.5)  # seconds
        self.max_backoff = getattr(self.cfg, "max_backoff", 20.0)

    async def run(
        self, acts: Tensor | None = None, explanations_override: dict[int, str] = {}
    ) -> dict[int, dict[str, Any]]:
        generation_examples, scoring_examples = self.gather_data(acts)
        latents_with_data = sorted(generation_examples.keys())
        n_dead = self.n_latents - len(latents_with_data)
        if n_dead > 0:
            print(f"Found data for {len(latents_with_data)}/{self.n_latents} alive latents; {n_dead} dead")

        # Launch tasks (bounded by feature_sem inside run_single_feature)
        tasks = [
            self.run_single_feature(
                latent,
                generation_examples[latent],
                scoring_examples[latent],
                explanations_override.get(latent, None),
            )
            for latent in latents_with_data
        ]

        results = {}
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Calling API (for gen & scoring)",
        ):
            try:
                result = await future
                if result:
                    results[result["latent"]] = result
            except Exception as e:
                # Optional: log and continue for robustness
                print(f"Feature failed with error: {e}")

        return results

    async def run_single_feature(
        self,
        latent: int,
        generation_examples: Examples,
        scoring_examples: Examples,
        explanation_override: str | None = None,
    ) -> dict[str, Any] | None:
        async with self.feature_sem:
            # Generation phase
            gen_prompts = self.get_generation_prompts(generation_examples)
            (explanation_raw,), logs = await self.get_api_response(
                gen_prompts,
                self.cfg.max_tokens_in_explanation,
            )
            explanation = self.parse_explanation(explanation_raw)
            results = {
                "latent": latent,
                "explanation": explanation,
                "logs": f"Generation phase\n{logs}\n{generation_examples.display()}",
            }

            # Scoring phase
            if self.cfg.scoring:
                scoring_prompts = self.get_scoring_prompts(
                    explanation=explanation_override or explanation,
                    scoring_examples=scoring_examples,
                )
                (predictions_raw,), logs = await self.get_api_response(
                    scoring_prompts,
                    self.cfg.max_tokens_in_prediction,
                )
                predictions = self.parse_predictions(predictions_raw)
                if predictions is None:
                    return None
                score = self.score_predictions(predictions, scoring_examples)
                results |= {
                    "predictions": predictions,
                    "correct seqs": [i for i, ex in enumerate(scoring_examples, start=1) if ex.is_active],
                    "score": score,
                    "logs": results["logs"] + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
                }

            return results

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> list[int] | None:
        predictions_split = predictions.strip().rstrip(".").replace("and", ",").replace("None", "").split(",")
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if predictions_list == []:
            return []
        if not all(pred.strip().isdigit() for pred in predictions_list):
            return None
        predictions_ints = [int(pred.strip()) for pred in predictions_list]
        return predictions_ints

    def score_predictions(self, predictions: list[int], scoring_examples: Examples) -> float:
        classifications = [i in predictions for i in range(1, len(scoring_examples) + 1)]
        correct_classifications = [ex.is_active for ex in scoring_examples]
        return sum([c == cc for c, cc in zip(classifications, correct_classifications)]) / len(classifications)

    async def get_api_response(
        self, messages: Messages, max_tokens: int, n_completions: int = 1
    ) -> tuple[list[str], str]:
        """Generic API usage function for OpenAI (async with retries/backoff)."""

        for message in messages:
            assert message.keys() == {"content", "role"}
            assert message["role"] in ["system", "user", "assistant"]

        attempt = 0
        while True:
            attempt += 1
            try:
                async with self.req_sem:
                    result = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,  # type: ignore
                        n=n_completions,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                response = [choice.message.content.strip() for choice in result.choices]

                logs = tabulate(
                    [m.values() for m in messages + [{"role": "assistant", "content": response[0]}]],
                    tablefmt="simple_grid",
                    maxcolwidths=[None, 120],
                )
                return response, logs

            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt >= self.max_retries:
                    raise
                # Exponential backoff with jitter
                backoff = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
                backoff *= 0.5 + random.random()  # jitter in [0.5x, 1.5x]
                await asyncio.sleep(backoff)

            except APIError as e:
                # Transient 5xx errors should be retried, 4xx shouldn't.
                # If you want stricter control, inspect e.status_code.
                if getattr(e, "status_code", 500) >= 500 and attempt < self.max_retries:
                    backoff = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
                    backoff *= 0.5 + random.random()
                    await asyncio.sleep(backoff)
                else:
                    raise

    def get_generation_prompts(self, generation_examples: Examples) -> Messages:
        assert len(generation_examples) > 0, "No generation examples found"

        examples_as_str = "\n".join(
            [f"{i + 1}. {ex.to_str(mark_toks=True)}" for i, ex in enumerate(generation_examples)]
        )

        SYSTEM_PROMPT = """We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words."""
        if self.cfg.use_demos_in_explanation:
            SYSTEM_PROMPT += """ Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy"."""
        else:
            SYSTEM_PROMPT += """Your response should be in the form "This neuron activates on..."."""
        USER_PROMPT = f"""The activating documents are given below:\n\n{examples_as_str}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def get_scoring_prompts(self, explanation: str, scoring_examples: Examples) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join([f"{i + 1}. {ex.to_str(mark_toks=False)}" for i, ex in enumerate(scoring_examples)])

        example_response = sorted(
            random.sample(
                range(1, 1 + self.cfg.n_ex_for_scoring),
                k=self.cfg.n_correct_for_scoring,
            )
        )
        example_response_str = ", ".join([str(i) for i in example_response])
        SYSTEM_PROMPT = f"""We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown {self.cfg.n_ex_for_scoring} example sequences in random order. You will have to return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like "{example_response_str}". Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with "None". You should include nothing else in your response other than comma-separated numbers or the word "None" - this is important."""
        USER_PROMPT = f"Here is the explanation: this neuron fires on {explanation}.\n\nHere are the examples:\n\n{examples_as_str}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def gather_data(self, acts: Tensor | None = None) -> tuple[dict[int, Examples], dict[int, Examples]]:
        """
        Stores top acts / random seqs data, which is used for generation & scoring respectively.
        """
        dataset_size, seq_len = self.tokenized_dataset.shape

        if acts is None:
            acts = collect_sae_activations(
                self.tokenized_dataset,
                self.model,
                self.sae,
                self.cfg.llm_batch_size,
                self.layer,
                self.sae.cfg.hook_name,
                mask_bos_pad_eos_tokens=True,
                selected_latents=self.latents,
                activation_dtype=torch.bfloat16,  # reduce memory usage, we don't need full precision when sampling activations
            )

        generation_examples = {}
        scoring_examples = {}

        for i, latent in tqdm(enumerate(self.latents), desc="Collecting examples for LLM judge"):
            # (1/3) Get random examples (we don't need their values)
            rand_indices = torch.stack(
                [
                    torch.randint(0, dataset_size, (self.cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer)

            # (2/3) Get top-scoring examples
            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(self.tokenized_dataset, top_indices, buffer=self.cfg.buffer)
            top_values = index_with_buffer(acts[..., i], top_indices, buffer=self.cfg.buffer)
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Get importance-weighted examples, using a threshold so they're disjoint from top examples
            # Also, if we don't have enough values, then we assume this is a dead feature & continue
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = get_iw_sample_indices(acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer)
            iw_toks = index_with_buffer(self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer)
            iw_values = index_with_buffer(acts[..., i], iw_indices, buffer=self.cfg.buffer)

            # Get random values to use for splitting
            rand_top_ex_split_indices = torch.randperm(self.cfg.n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[: self.cfg.n_top_ex_for_generation]
            top_scoring_indices = rand_top_ex_split_indices[self.cfg.n_top_ex_for_generation :]
            rand_iw_split_indices = torch.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[: self.cfg.n_iw_sampled_ex_for_generation]
            iw_scoring_indices = rand_iw_split_indices[self.cfg.n_iw_sampled_ex_for_generation :]

            def create_examples(all_toks: Tensor, all_acts: Tensor | None = None) -> list[Example]:
                if all_acts is None:
                    all_acts = torch.zeros_like(all_toks).float()
                return [
                    Example(
                        toks=toks,
                        acts=acts,
                        act_threshold=act_threshold,
                        model=self.model,
                    )
                    for (toks, acts) in zip(all_toks.tolist(), all_acts.tolist())
                ]

            # Get the generation & scoring examples
            generation_examples[latent] = Examples(
                create_examples(top_toks[top_gen_indices], top_values[top_gen_indices])
                + create_examples(iw_toks[iw_gen_indices], iw_values[iw_gen_indices]),
            )
            scoring_examples[latent] = Examples(
                create_examples(top_toks[top_scoring_indices], top_values[top_scoring_indices])
                + create_examples(iw_toks[iw_scoring_indices], iw_values[iw_scoring_indices])
                + create_examples(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples


def run_eval_single_sae(
    config: AutoInterpEvalConfig,
    sae: BaseSAE,
    model: HookedTransformer,
    layer: int,
    device: str,
    artifacts_folder: str,
    api_key: str,
    sae_sparsity: torch.Tensor | None = None,
) -> dict[str, float]:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    tokens_filename = f"{escape_slash(config.model_name)}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = torch.load(tokens_path, weights_only=True).to(device)
    else:
        tokenized_dataset = load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            model.tokenizer,  # type: ignore
        ).to(device)
        torch.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if sae_sparsity is None:
        sae_sparsity = get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            sae,
            config.llm_batch_size,
            layer,
            sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )

    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        layer=layer,
        tokenized_dataset=tokenized_dataset,
        sparsity=sae_sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp.run())
    return results  # type: ignore


def run_eval(
    config: AutoInterpEvalConfig,
    selected_saes: list[tuple[str, BaseSAE]],
    layer: int,
    device: str,
    api_key: str,
    output_path: str,
    force_rerun: bool = False,
    save_logs_path: str | None = None,
    artifacts_path: str = "artifacts",
) -> dict[str, Any]:
    """
    selected_saes is a list of (name, SAE) pairs.
    """
    eval_instance_id = get_eval_uuid()

    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    llm_dtype = getattr(torch, config.llm_dtype)

    model: HookedTransformer = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for sae_id, sae in tqdm(selected_saes, desc="Running SAE evaluation on all selected SAEs"):
        sae = sae.to(device=device, dtype=llm_dtype)

        sae_result_path = os.path.join(output_path, f"local_{sae_id}_eval_results.json".replace("/", "_"))

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_id} as results already exist")
            continue

        artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_AUTOINTERP)

        sae_eval_result = run_eval_single_sae(config, sae, model, layer, device, artifacts_folder, api_key, None)

        # Save nicely formatted logs to a text file, helpful for debugging.
        if save_logs_path is not None:
            # Get summary results for all latents, as well logs for the best and worst-scoring latents
            headers = [
                "latent",
                "explanation",
                "predictions",
                "correct seqs",
                "score",
            ]
            logs = "Summary table:\n" + tabulate(
                [
                    [sae_eval_result[latent][h] for h in headers]  # type: ignore
                    for latent in sae_eval_result
                ],
                headers=headers,
                tablefmt="simple_outline",
            )
            worst_result = min(sae_eval_result.values(), key=lambda x: x["score"])  # type: ignore
            best_result = max(sae_eval_result.values(), key=lambda x: x["score"])  # type: ignore
            logs += f"\n\nWorst scoring idx {worst_result['latent']}, score = {worst_result['score']}\n{worst_result['logs']}"  # type: ignore
            logs += (
                f"\n\nBest scoring idx {best_result['latent']}, score = {best_result['score']}\n{best_result['logs']}"  # type: ignore
            )
            # Save the results to a file
            with open(save_logs_path, "a") as f:
                f.write(logs)

        # Put important results into the results dict
        all_scores = [r["score"] for r in sae_eval_result.values()]  # type: ignore

        all_scores_tensor = torch.tensor(all_scores)
        score = all_scores_tensor.mean().item()
        std_dev = all_scores_tensor.std().item()

        eval_output = AutoInterpEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=AutoInterpMetricCategories(
                autointerp=AutoInterpMetrics(autointerp_score=score, autointerp_std_dev=std_dev)
            ),
            eval_result_details=[],
            eval_result_unstructured=sae_eval_result,
            sae_bench_commit_hash=sae_id,
            sae_lens_id=sae_id,
            sae_lens_release_id="local",
            sae_lens_version="_",
            sae_cfg_dict=asdict(sae.cfg),
        )

        results_dict[f"{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict
