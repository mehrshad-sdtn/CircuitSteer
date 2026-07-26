from __future__ import annotations

import argparse

import pandas as pd

from circuitsteer import CircuitSteer
from circuitsteer.config import DEFAULT_COEFFICIENTS
from circuitsteer.data import LOADERS, make_datasets, text_of
from circuitsteer.evaluation import (
    evaluate_coefficient,
    qualitative_rows,
)
from experiments.common import (
    add_circuit_arguments,
    config_from_args,
    release_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main CircuitSteer experiments."
    )
    add_circuit_arguments(parser)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=LOADERS,
        default=list(LOADERS),
    )
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        default=list(DEFAULT_COEFFICIENTS),
    )
    parser.add_argument("--qualitative-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    numeric_rows: list[dict[str, object]] = []
    qualitative_output: list[dict[str, object]] = []

    for model_key in args.models:
        datasets = make_datasets(
            labels=args.tasks,
            n=args.dataset_size,
            seed=args.seed,
        )
        steerer = CircuitSteer(model_key, config=config_from_args(args))

        for dataset_label, dataset in datasets.items():
            test_items = dataset["test_prompts"]
            toxic = [text_of(item) for item in dataset["train_toxic"]]
            benign = [text_of(item) for item in dataset["train_benign"]]
            steerer.fit(toxic, benign)

            coefficient_rows = []
            for coefficient in args.coefficients:
                metrics = evaluate_coefficient(
                    steerer,
                    dataset_label,
                    test_items,
                    coefficient,
                    seed=args.seed,
                )
                row = {
                    "method": "CircuitSteer",
                    "model": model_key,
                    "dataset": dataset_label,
                    "coeff": coefficient,
                    **metrics,
                }
                coefficient_rows.append(row)
                print(
                    f"{model_key}/{dataset_label} "
                    f"lambda={coefficient:+.1f} "
                    f"delta={metrics['delta']:+.3f} "
                    f"ppl={metrics['norm_ppl']:.2f}"
                )

            numeric_rows.extend(coefficient_rows)
            qualitative_output.extend(
                qualitative_rows(
                    pd.DataFrame(coefficient_rows),
                    test_items,
                    steerer,
                    dataset_label,
                    model_key,
                    n_samples=args.qualitative_samples,
                    seed=args.seed,
                )
            )

        release_model(steerer)

    results_path = args.output_dir / "circuitsteer_results.csv"
    qualitative_path = args.output_dir / "circuitsteer_qual.csv"
    pd.DataFrame(numeric_rows).to_csv(results_path, index=False)
    pd.DataFrame(qualitative_output).to_csv(
        qualitative_path,
        index=False,
    )
    print(f"Results: {results_path}")
    print(f"Qualitative outputs: {qualitative_path}")


if __name__ == "__main__":
    main()
