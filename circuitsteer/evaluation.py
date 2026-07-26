from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .config import GLOBAL_SEED
from .data import DatasetItem, text_of
from .scoring import make_score_function


def evaluate_coefficient(
    steerer,
    dataset_label: str,
    test_items: list[DatasetItem],
    coefficient: float,
    seed: int = GLOBAL_SEED,
) -> dict[str, float | int]:
    score = make_score_function(
        dataset_label,
        steerer.model,
        steerer.device,
    )
    steered_hooks = (
        steerer.hooks(coefficient) if coefficient != 0.0 else None
    )
    base_scores: list[float] = []
    steered_scores: list[float] = []
    base_perplexities: list[float] = []
    steered_perplexities: list[float] = []

    for index, item in enumerate(test_items):
        prompt = text_of(item)
        steerer.model.reset_hooks()
        if dataset_label == "Sycophancy":
            base_score = score(item, hooks=None)
            steered_score = score(item, hooks=steered_hooks)
            base_output = steerer.generate(prompt, 0.0, seed + index)
            steered_output = steerer.generate(
                prompt,
                coefficient,
                seed + index,
            )
        else:
            base_output = steerer.generate(prompt, 0.0, seed + index)
            steered_output = steerer.generate(
                prompt,
                coefficient,
                seed + index,
            )
            base_score = score(item, output=base_output)
            steered_score = score(item, output=steered_output)

        base_scores.append(base_score)
        steered_scores.append(steered_score)
        base_perplexities.append(steerer.perplexity(base_output))
        steered_perplexities.append(steerer.perplexity(steered_output))

    differences = np.asarray(base_scores) - np.asarray(steered_scores)
    return {
        "delta": float(differences.mean()),
        "delta_std": (
            float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
        ),
        "norm_ppl": float(
            np.mean(steered_perplexities)
            / max(np.mean(base_perplexities), 1e-6)
        ),
        "n": len(test_items),
    }


def qualitative_rows(
    results: pd.DataFrame,
    test_items: list[DatasetItem],
    steerer,
    dataset_label: str,
    model_key: str,
    method_name: str = "CircuitSteer",
    n_samples: int = 50,
    seed: int = GLOBAL_SEED,
) -> list[dict[str, object]]:
    valid = results[results["norm_ppl"] < 1.5].copy()
    if valid.empty:
        valid = results.copy()
    best = valid.loc[valid["delta"].idxmax()]
    best_coefficient = float(best["coeff"])
    score = make_score_function(
        dataset_label,
        steerer.model,
        steerer.device,
    )
    items = random.Random(seed).sample(
        test_items,
        min(n_samples, len(test_items)),
    )
    rows = []
    for index, item in enumerate(items):
        prompt = text_of(item)
        base_output = steerer.generate(prompt, 0.0, seed + index)
        steered_output = steerer.generate(
            prompt,
            best_coefficient,
            seed + index,
        )
        hooks = (
            steerer.hooks(best_coefficient)
            if best_coefficient != 0.0
            else None
        )
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
                "norm_ppl": round(
                    steered_ppl / max(base_ppl, 1e-6),
                    4,
                ),
            }
        )
    return rows
