from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset

from circuitsteer import CircuitSteer
from circuitsteer.data import LOADERS, make_datasets, text_of
from circuitsteer.io import append_row, read_rows
from experiments.common import (
    add_circuit_arguments,
    config_from_args,
    release_model,
)

FIELDS = [
    "model",
    "benchmark",
    "behavior",
    "best_lambda",
    "n",
    "base_acc",
    "steered_acc",
    "delta_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base and steered MMLU accuracy."
    )
    add_circuit_arguments(parser)
    parser.add_argument(
        "--behavior",
        choices=LOADERS,
        default="RTP",
        help="Dataset used to discover the steering circuit.",
    )
    parser.add_argument("--best-lambda", type=float, default=-3.0)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--mmlu-samples", type=int, default=1000)
    return parser.parse_args()


def last_log_probabilities(
    steerer: CircuitSteer,
    prompt: str,
    coefficient: float,
    token_ids: list[int],
) -> list[float]:
    hooks = (
        steerer.hooks(coefficient) if coefficient != 0.0 else None
    )
    tokens = steerer.model.to_tokens(prompt)
    with torch.no_grad():
        if coefficient != 0.0 and hooks:
            with steerer.model.hooks(fwd_hooks=hooks):
                logits = steerer.model(tokens)
        else:
            logits = steerer.model(tokens)
    log_probabilities = torch.log_softmax(
        logits[0, -1, :].float(),
        dim=-1,
    )
    return [
        float(log_probabilities[token_id].item())
        for token_id in token_ids
    ]


def format_mmlu(item) -> str:
    choices = item["choices"]
    return (
        "The following is a multiple choice question. "
        "Answer with the letter of the correct option.\n\n"
        f"{item['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def evaluate_mmlu(
    steerer: CircuitSteer,
    items,
    coefficient: float,
) -> tuple[float, float, int]:
    letter_ids = [
        int(
            steerer.model.to_tokens(
                f" {letter}",
                prepend_bos=False,
            )[0, 0].item()
        )
        for letter in "ABCD"
    ]
    if len(set(letter_ids)) != 4:
        raise RuntimeError(f"Answer tokens are not distinct: {letter_ids}")

    base_correct = 0
    steered_correct = 0
    for index, item in enumerate(items, start=1):
        prompt = format_mmlu(item)
        gold = int(item["answer"])
        base_scores = last_log_probabilities(
            steerer,
            prompt,
            0.0,
            letter_ids,
        )
        steered_scores = last_log_probabilities(
            steerer,
            prompt,
            coefficient,
            letter_ids,
        )
        base_correct += int(int(np.argmax(base_scores)) == gold)
        steered_correct += int(int(np.argmax(steered_scores)) == gold)
        if index % 200 == 0:
            print(
                f"MMLU [{index}/{len(items)}] "
                f"base={base_correct / index:.3f} "
                f"steered={steered_correct / index:.3f}"
            )
    n_items = len(items)
    return (
        base_correct / max(n_items, 1),
        steered_correct / max(n_items, 1),
        n_items,
    )


def load_mmlu(n: int, seed: int):
    dataset = load_dataset("cais/mmlu", "all", split="test")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(indices[: min(n, len(indices))])


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / "circuitsteer_mmlu.csv"
    done = {row["model"] for row in read_rows(output_path)}
    mmlu_items = None

    for model_key in args.models:
        if model_key in done:
            print(f"{model_key}: already complete")
            continue
        source = make_datasets(
            labels=[args.behavior],
            n=args.dataset_size,
            seed=args.seed,
        )[args.behavior]
        toxic = [text_of(item) for item in source["train_toxic"]]
        benign = [text_of(item) for item in source["train_benign"]]
        steerer = CircuitSteer(model_key, config=config_from_args(args))
        steerer.fit(toxic, benign)
        try:
            if mmlu_items is None:
                mmlu_items = load_mmlu(args.mmlu_samples, args.seed)
            base_accuracy, steered_accuracy, n_items = evaluate_mmlu(
                steerer,
                mmlu_items,
                args.best_lambda,
            )
            row = {
                "model": model_key,
                "benchmark": "MMLU",
                "behavior": args.behavior,
                "best_lambda": args.best_lambda,
                "n": n_items,
                "base_acc": round(base_accuracy, 4),
                "steered_acc": round(steered_accuracy, 4),
                "delta_acc": round(
                    base_accuracy - steered_accuracy,
                    4,
                ),
            }
            append_row(output_path, row, FIELDS)
            done.add(model_key)
            print(
                f"{model_key}: base={base_accuracy:.3f} "
                f"steered={steered_accuracy:.3f}"
            )
        finally:
            release_model(steerer)
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
