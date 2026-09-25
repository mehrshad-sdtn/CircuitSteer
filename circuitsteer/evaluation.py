from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .config import GLOBAL_SEED
from .data import DatasetItem, text_of
from .scoring import make_score_function

# Nothing here thresholds a result. Every coefficient's raw delta, ppl and
# norm_ppl is reported and selection is a plain argmax on delta; fluency and
# degeneracy are measured (norm_ppl, degenerate_frac, valid_frac) so that any
# cutoff is applied post hoc to the emitted numbers.
#
# One data-handling choice does affect the aggregates and is not a threshold:
# an empty continuation has no toxicity score, so it is NaN rather than 0.0
# and drops out of the nan-aware means. Scoring it 0.0 would assert that a
# collapsed generation is perfectly clean. degenerate_frac and valid_frac
# record exactly how many were affected, and the per-prompt CSV keeps the
# raw values, so either convention can be reconstructed downstream.


def _nanmean(values) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmean(array))


def _nanmax(values) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmax(array))


def evaluate_coefficient(
    steerer,
    dataset_label: str,
    test_items: list[DatasetItem],
    coefficient: float,
    seed: int = GLOBAL_SEED,
    samples_per_prompt: int = 1,
    batch_size: int = 32,
) -> dict[str, float | int]:
    """Score one steering coefficient.

    `samples_per_prompt` > 1 follows the RealToxicityPrompts protocol, which
    draws several continuations per prompt at temperature 1.0 and reports
    both the mean and the expected *maximum* score. At the default of 1 the
    generation seeds, and therefore the numbers, match the original run.
    """
    score = make_score_function(
        dataset_label,
        steerer.model,
        steerer.device,
    )
    steered_hooks = (
        steerer.hooks(coefficient) if coefficient != 0.0 else None
    )
    # The sycophancy score is teacher-forced and deterministic, so extra
    # samples would only duplicate it.
    is_sycophancy = dataset_label == "Sycophancy"
    if is_sycophancy:
        samples_per_prompt = 1

    # One flat list of (item_index, sample_index) so every prompt in the
    # split is generated in a few large batches instead of one at a time.
    plan = [
        (index, sample)
        for index in range(len(test_items))
        for sample in range(samples_per_prompt)
    ]
    prompts = [text_of(test_items[index]) for index, _ in plan]
    gen_seeds = [
        seed + index * samples_per_prompt + sample for index, sample in plan
    ]
    steerer.model.reset_hooks()

    if batch_size > 1 and hasattr(steerer, "generate_batch"):
        base_outputs = steerer.generate_batch(
            prompts, 0.0, seed, batch_size=batch_size
        )
        steerer.model.reset_hooks()
        steered_outputs = steerer.generate_batch(
            prompts, coefficient, seed, batch_size=batch_size
        )
    else:
        base_outputs = [
            steerer.generate(prompt, 0.0, gen_seed)
            for prompt, gen_seed in zip(prompts, gen_seeds)
        ]
        steered_outputs = [
            steerer.generate(prompt, coefficient, gen_seed)
            for prompt, gen_seed in zip(prompts, gen_seeds)
        ]
    steerer.model.reset_hooks()

    base_perplexities = [steerer.perplexity(text) for text in base_outputs]
    steered_perplexities = [
        steerer.perplexity(text) for text in steered_outputs
    ]
    n_generations = len(steered_outputs)
    n_degenerate = sum(
        1 for text in steered_outputs if not text.strip()
    )

    per_item_base: dict[int, list[float]] = {
        index: [] for index in range(len(test_items))
    }
    per_item_steered: dict[int, list[float]] = {
        index: [] for index in range(len(test_items))
    }
    # A method may steer by rewriting the input rather than by adding a
    # vector (the prompt baseline). Sycophancy is scored by teacher
    # forcing, never from a generation, so such a method must be able to
    # reach the scorer too or it would score a flat zero for the wrong
    # reason.
    transform = getattr(steerer, "transform_item", None)
    for (index, _), base_output, steered_output in zip(
        plan, base_outputs, steered_outputs
    ):
        item = test_items[index]
        steered_item = (
            transform(item) if (transform and coefficient != 0.0) else item
        )
        if is_sycophancy:
            per_item_base[index].append(score(item, hooks=None))
            per_item_steered[index].append(
                score(steered_item, hooks=steered_hooks))
        else:
            per_item_base[index].append(score(item, output=base_output))
            per_item_steered[index].append(
                score(item, output=steered_output)
            )

    base_means = [_nanmean(per_item_base[i]) for i in range(len(test_items))]
    steered_means = [
        _nanmean(per_item_steered[i]) for i in range(len(test_items))
    ]
    base_maxima = [_nanmax(per_item_base[i]) for i in range(len(test_items))]
    steered_maxima = [
        _nanmax(per_item_steered[i]) for i in range(len(test_items))
    ]

    differences = np.asarray(base_means) - np.asarray(steered_means)
    max_differences = np.asarray(base_maxima) - np.asarray(steered_maxima)
    valid = differences[~np.isnan(differences)]
    mean_base_ppl = _nanmean(base_perplexities)

    return {
        "delta": _nanmean(differences),
        "delta_std": (
            float(valid.std(ddof=1)) if valid.size > 1 else 0.0
        ),
        # The standard error is what belongs on an error bar; delta_std is
        # the spread across prompts, which does not shrink with n.
        "delta_sem": (
            float(valid.std(ddof=1) / math.sqrt(valid.size))
            if valid.size > 1
            else 0.0
        ),
        "delta_max": _nanmean(max_differences),
        # Raw perplexities are reported alongside the ratio: norm_ppl
        # divides two means of a heavy-tailed quantity, so the components
        # are needed to tell a real fluency change from an outlier.
        "base_ppl": mean_base_ppl,
        "steered_ppl": _nanmean(steered_perplexities),
        "base_ppl_median": float(
            np.nanmedian(base_perplexities)
        ) if base_perplexities else math.nan,
        "steered_ppl_median": float(
            np.nanmedian(steered_perplexities)
        ) if steered_perplexities else math.nan,
        "norm_ppl": (
            _nanmean(steered_perplexities) / mean_base_ppl
            if mean_base_ppl and not math.isnan(mean_base_ppl)
            else math.nan
        ),
        "norm_ppl_median": (
            float(np.nanmedian(steered_perplexities))
            / float(np.nanmedian(base_perplexities))
            if base_perplexities
            and float(np.nanmedian(base_perplexities)) > 0
            else math.nan
        ),
        "valid_frac": (
            float(valid.size / len(differences)) if len(differences) else 0.0
        ),
        "degenerate_frac": (
            float(n_degenerate / n_generations) if n_generations else 0.0
        ),
        "n": len(test_items),
        "samples_per_prompt": samples_per_prompt,
        # Kept for the paired significance test: the prompt-level pairs
        # are the unit with real statistical power, not the 3 seeds.
        "_base_scores": base_means,
        "_steered_scores": steered_means,
    }


def select_best_coefficient(
    results: pd.DataFrame,
    max_norm_ppl: float | None = None,
    max_degenerate_frac: float | None = None,
) -> float:
    """Coefficient with the largest delta.

    No fluency or degeneracy filtering is applied by default: the run
    reports raw delta, ppl and norm_ppl for every coefficient, and any
    threshold is a post hoc decision made on the emitted numbers rather
    than something baked into the run. Pass `max_norm_ppl` and/or
    `max_degenerate_frac` to reinstate the guards.

    Callers should pass results measured on the validation split: choosing
    on the same prompts used to report biases the reported delta upward.
    """
    usable = results[results["delta"].notna()]
    if max_norm_ppl is not None:
        usable = usable[
            usable["norm_ppl"].notna() & (usable["norm_ppl"] < max_norm_ppl)
        ]
    if (
        max_degenerate_frac is not None
        and "degenerate_frac" in usable.columns
    ):
        usable = usable[usable["degenerate_frac"] <= max_degenerate_frac]
    if usable.empty:
        return 0.0
    return float(usable.loc[usable["delta"].idxmax()]["coeff"])


def qualitative_rows(
    results: pd.DataFrame,
    test_items: list[DatasetItem],
    steerer,
    dataset_label: str,
    model_key: str,
    method_name: str = "CircuitSteer",
    n_samples: int = 50,
    seed: int = GLOBAL_SEED,
    best_coefficient: float | None = None,
) -> list[dict[str, object]]:
    if best_coefficient is None:
        best_coefficient = select_best_coefficient(results)
    score = make_score_function(
        dataset_label,
        steerer.model,
        steerer.device,
    )
    items = random.Random(seed).sample(
        test_items,
        min(n_samples, len(test_items)),
    )
    hooks = (
        steerer.hooks(best_coefficient)
        if best_coefficient != 0.0
        else None
    )
    prompts = [text_of(item) for item in items]
    steerer.model.reset_hooks()
    if hasattr(steerer, "generate_batch"):
        base_outputs = steerer.generate_batch(prompts, 0.0, seed)
        steerer.model.reset_hooks()
        steered_outputs = steerer.generate_batch(
            prompts, best_coefficient, seed
        )
    else:
        base_outputs = [
            steerer.generate(prompt, 0.0, seed + i)
            for i, prompt in enumerate(prompts)
        ]
        steered_outputs = [
            steerer.generate(prompt, best_coefficient, seed + i)
            for i, prompt in enumerate(prompts)
        ]
    steerer.model.reset_hooks()

    rows = []
    for item, prompt, base_output, steered_output in zip(
        items, prompts, base_outputs, steered_outputs
    ):
        base_score = score(item, output=base_output, hooks=None)
        steered_score = score(item, output=steered_output, hooks=hooks)
        base_ppl = steerer.perplexity(base_output)
        steered_ppl = steerer.perplexity(steered_output)
        rows.append(
            {
                "model": model_key,
                "dataset": dataset_label,
                "method": method_name,
                "best_coeff": best_coefficient,
                "prompt": prompt,
                "base_output": base_output,
                "steered_output": steered_output,
                "base_score": round(base_score, 4),
                "steered_score": round(steered_score, 4),
                "delta": round(base_score - steered_score, 4),
                "base_ppl": round(base_ppl, 4),
                "steered_ppl": round(steered_ppl, 4),
                "norm_ppl": round(steered_ppl / base_ppl, 4)
                if base_ppl and not math.isnan(base_ppl)
                else math.nan,
                "degenerate": not steered_output.strip(),
            }
        )
    return rows
