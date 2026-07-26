from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from circuitsteer import CIRCUIT_DEFAULTS, CircuitSteer
from circuitsteer.data import LOADERS, make_datasets, text_of
from circuitsteer.evaluation import evaluate_coefficient
from circuitsteer.io import append_row, read_rows
from experiments.common import (
    add_circuit_arguments,
    config_from_args,
    release_model,
)

FIELDS = [
    "model",
    "task",
    "param",
    "value",
    "is_default",
    "n_edges",
    "n_features",
    "n_layers",
    "jaccard_vs_default",
    "vec_cos_vs_default",
    "eval_lambda",
    "delta",
    "norm_ppl",
]

GRIDS = {
    "act_thresh": (1.0, 1.5, 2.0, 3.0),
    "sim_thresh": (0.05, 0.10, 0.20, 0.30),
    "diff_thresh": (0.02, 0.05, 0.10, 0.15),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time threshold sensitivity experiments."
    )
    add_circuit_arguments(parser)
    parser.set_defaults(
        sim_thresh=CIRCUIT_DEFAULTS.sim_thresh,
        act_thresh=CIRCUIT_DEFAULTS.act_thresh,
        diff_thresh=CIRCUIT_DEFAULTS.diff_thresh,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=LOADERS,
        default=["RTP", "Sycophancy"],
    )
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--eval-lambda", type=float, default=-3.0)
    return parser.parse_args()


def edge_set(steerer: CircuitSteer) -> set[tuple[str, str]]:
    return {edge for edge, _ in steerer.circuit}


def vector_signature(steerer: CircuitSteer) -> tuple:
    if not steerer.steer_vecs:
        return ("empty",)
    parts = [
        steerer.steer_vecs[layer]
        .detach()
        .float()
        .cpu()
        .numpy()
        .round(4)
        .tobytes()
        for layer in sorted(steerer.steer_vecs)
    ]
    return tuple(sorted(steerer.steer_vecs)), b"".join(parts)


def vector_cosine(
    vectors: dict[int, torch.Tensor],
    reference: dict[int, torch.Tensor],
) -> float:
    shared = [layer for layer in vectors if layer in reference]
    if not shared:
        return float("nan")
    similarities = [
        float(
            torch.nn.functional.cosine_similarity(
                vectors[layer].detach().float().unsqueeze(0),
                reference[layer].detach().float().unsqueeze(0),
            ).item()
        )
        for layer in shared
    ]
    return float(np.mean(similarities))


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def feature_count(steerer: CircuitSteer) -> int:
    count = 0
    for layer in steerer.steer_vecs:
        features = {
            int(origin.split("_")[1])
            for (origin, _), _score in steerer.circuit
            if int(origin.split("_")[0][1:]) == layer
        }
        count += len(features)
    return count


def install_edge_cache(steerer: CircuitSteer):
    original = steerer._build_edges
    cache = {}

    def cached(texts):
        key = (
            steerer.config.act_thresh,
            steerer.config.sim_thresh,
            id(texts),
        )
        if key not in cache:
            cache[key] = original(texts)
        return cache[key]

    steerer._build_edges = cached
    return original


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / "circuitsteer_sensitivity.csv"
    previous_rows = read_rows(output_path)
    done = {
        (
            row["model"],
            row["task"],
            row["param"],
            float(row["value"]),
        )
        for row in previous_rows
    }
    defaults = config_from_args(args)

    for model_key in args.models:
        pending = [
            (task, parameter, value)
            for task in args.tasks
            for parameter, grid in GRIDS.items()
            for value in grid
            if (model_key, task, parameter, value) not in done
        ]
        if not pending:
            print(f"{model_key}: all settings already complete")
            continue

        datasets = make_datasets(
            labels=args.tasks,
            n=args.dataset_size,
            seed=args.seed,
        )
        steerer = CircuitSteer(model_key, config=defaults)
        original_build_edges = install_edge_cache(steerer)
        try:
            for task in args.tasks:
                dataset = datasets[task]
                toxic = [
                    text_of(item) for item in dataset["train_toxic"]
                ]
                benign = [
                    text_of(item) for item in dataset["train_benign"]
                ]
                test_items = dataset["test_prompts"]
                evaluation_cache = {}

                def run_setting(parameter: str, value: float):
                    steerer.config = replace(
                        defaults,
                        **{parameter: value},
                    )
                    steerer.fit(toxic, benign)
                    edges = edge_set(steerer)
                    vectors = dict(steerer.steer_vecs)
                    return edges, vectors, feature_count(steerer)

                reference_edges, reference_vectors, _ = run_setting(
                    "sim_thresh",
                    defaults.sim_thresh,
                )

                for parameter, grid in GRIDS.items():
                    default_value = getattr(defaults, parameter)
                    for value in grid:
                        key = (model_key, task, parameter, value)
                        if key in done:
                            print(
                                f"{model_key}/{task} "
                                f"{parameter}={value}: done"
                            )
                            continue
                        edges, vectors, n_features = run_setting(
                            parameter,
                            value,
                        )
                        steerer.steer_vecs = vectors
                        signature = vector_signature(steerer)
                        if signature not in evaluation_cache:
                            metrics = evaluate_coefficient(
                                steerer,
                                task,
                                test_items,
                                args.eval_lambda,
                                seed=args.seed,
                            )
                            evaluation_cache[signature] = metrics
                        metrics = evaluation_cache[signature]
                        row = {
                            "model": model_key,
                            "task": task,
                            "param": parameter,
                            "value": value,
                            "is_default": value == default_value,
                            "n_edges": len(edges),
                            "n_features": n_features,
                            "n_layers": len(vectors),
                            "jaccard_vs_default": jaccard(
                                edges,
                                reference_edges,
                            ),
                            "vec_cos_vs_default": vector_cosine(
                                vectors,
                                reference_vectors,
                            ),
                            "eval_lambda": args.eval_lambda,
                            "delta": metrics["delta"],
                            "norm_ppl": metrics["norm_ppl"],
                        }
                        append_row(output_path, row, FIELDS)
                        done.add(key)
                        print(
                            f"{model_key}/{task} {parameter}={value}: "
                            f"edges={len(edges)} "
                            f"features={n_features} "
                            f"delta={metrics['delta']:+.3f} "
                            f"ppl={metrics['norm_ppl']:.2f}"
                        )
        finally:
            steerer._build_edges = original_build_edges
            release_model(steerer)
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
