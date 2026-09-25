from __future__ import annotations

import argparse

import pandas as pd

from circuitsteer import CircuitSteer
from circuitsteer.config import DEFAULT_COEFFICIENTS
from circuitsteer.data import LOADERS, make_datasets, text_of
from circuitsteer.evaluation import (
    evaluate_coefficient,
    qualitative_rows,
    select_best_coefficient,
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
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=0,
        help=(
            "Steer and score this many instances instead of the usual "
            "test split, drawn test-first then val then unused train. "
            "0 keeps the normal split behaviour."
        ),
    )
    parser.add_argument(
        "--select-mode",
        default=None,
        choices=("aligned", "none", "anti", "random"),
        help=(
            "Edge-selection rule for the ablation. All modes draw from "
            "the same candidate pool and return the same number of "
            "edges, so differences cannot be explained by circuit size. "
            "aligned = CircuitSteer; none = NoGeo; anti = the k most "
            "anti-aligned edges; random = k edges drawn uniformly. "
            "Overrides --align-mode when given."
        ),
    )
    parser.add_argument(
        "--n-steer-edges",
        type=int,
        default=None,
        help="Edge budget for the steering vectors (default: top_k * 2).",
    )
    parser.add_argument(
        "--select-seed",
        type=int,
        default=0,
        help="Seed for --select-mode random.",
    )
    parser.add_argument(
        "--align-mode",
        default="positive",
        choices=("positive", "none", "negative"),
        help=(
            "Circuit-discovery ablation. positive = CircuitSteer "
            "(co-activation + geometric alignment); none = NoGeo "
            "(co-activation only); negative = NegativeAlign "
            "(co-activation + ANTI-aligned decoder directions)."
        ),
    )
    parser.add_argument(
        "--shuffle-sae",
        action="store_true",
        help="ShuffledSAE ablation: keep real encoder and real decoder "
             "directions but permute their correspondence.",
    )
    parser.add_argument(
        "--random-sae",
        action="store_true",
        help="RandomSAE ablation: replace trained SAE weights with random "
             "ones of matched scale, to test whether the learned features "
             "are load-bearing.",
    )
    parser.add_argument(
        "--method",
        default="circuitsteer",
        help=(
            "circuitsteer, or a baseline name from baselines.registry. "
            "Baselines run through this identical protocol."
        ),
    )
    parser.add_argument(
        "--fixed-lambda",
        nargs="+",
        default=None,
        metavar="DATASET=VALUE",
        help=(
            "Use one fixed coefficient per dataset for every seed, e.g. "
            "RTP=-3 Emotion=-4. Skips per-seed selection, so the seed "
            "spread reflects the effect rather than selection noise."
        ),
    )
    parser.add_argument(
        "--release-saes",
        action="store_true",
        help=(
            "Free SAEs after circuit discovery. Needed to fit "
            "Llama-3.1-8B on a 24GB card; reloaded per dataset."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Prompts generated per forward pass. TransformerLens decodes "
            "one token per Python step, so batching is ~30x faster."
        ),
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help=(
            "Continuations per prompt. The RealToxicityPrompts protocol "
            "uses 25; 1 reproduces the original single-sample run."
        ),
    )
    return parser.parse_args()


def build_steerer(method: str, model_key: str, config):
    """Instantiate the method or a baseline.

    Baselines subclass CircuitSteer, so everything after this call -
    hooks, batched generation, perplexity, scoring, splits, statistics -
    is the same code for every method. Only the derivation of
    `steer_vecs` differs.
    """
    if method == "circuitsteer":
        return CircuitSteer(model_key, config=config)
    from baselines.registry import get_baseline

    return get_baseline(method)(model_key, config=config)


def parse_fixed_lambda(pairs) -> dict[str, float]:
    if not pairs:
        return {}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--fixed-lambda expects DATASET=VALUE, got {pair!r}")
        name, value = pair.split("=", 1)
        out[name] = float(value)
    return out


def main() -> None:
    args = parse_args()
    fixed_lambdas = parse_fixed_lambda(args.fixed_lambda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    numeric_rows: list[dict[str, object]] = []
    qualitative_output: list[dict[str, object]] = []
    # Per-prompt paired scores: the unit the significance test runs on.
    prompt_rows: list[dict[str, object]] = []
    circuit_rows: list[dict[str, object]] = []

    for model_key in args.models:
        datasets = make_datasets(
            labels=args.tasks,
            n=args.dataset_size,
            seed=args.seed,
        )
        steerer = build_steerer(
            args.method, model_key, config_from_args(args)
        )
        random_sae = getattr(args, "random_sae", False)
        shuffle_sae = getattr(args, "shuffle_sae", False)
        method_name = getattr(steerer, "name", "CircuitSteer")
        if args.select_mode:
            variant = {"aligned": "Full", "none": "NoGeo",
                       "anti": "AntiAlign",
                       "random": "RandomEdges"}[args.select_mode]
        else:
            variant = {"positive": "Full", "none": "NoGeo",
                       "negative": "NegativeAlign"}[args.align_mode]
        if random_sae:
            variant = "RandomSAE"
        if shuffle_sae:
            variant = "ShuffledSAE"

        for dataset_label, dataset in datasets.items():
            test_items = dataset["test_prompts"]
            val_items = dataset.get("val_prompts") or []
            if args.eval_subset:
                # Test-first, then val, then unused train, so the steered
                # sample sits as far outside the discovery set as possible.
                pool = list(test_items) + list(val_items)

                # Only top up with items of the same SHAPE. Sycophancy's
                # held-out items are (question, sycophantic, honest)
                # tuples while its train entries are concatenated
                # strings, so an unchecked top-up injects unscoreable
                # items into the eval set.
                def compatible(item) -> bool:
                    if not pool:
                        return True
                    reference = pool[0]
                    if isinstance(reference, tuple):
                        return isinstance(item, tuple) and \
                            len(item) == len(reference)
                    return isinstance(item, str)

                pool += [i for i in dataset["train_toxic"]
                         if compatible(i) and i not in pool]
                test_items = pool[: args.eval_subset]
                val_items = []
                if len(test_items) < args.eval_subset:
                    print(
                        f"{dataset_label}: only {len(test_items)} compatible "
                        f"instances available (asked {args.eval_subset})"
                    )
                print(
                    f"{dataset_label}: steering {len(test_items)} instances "
                    f"(test-first subset)"
                )
            benign_items = dataset.get("test_benign") or []
            toxic = [text_of(item) for item in dataset["train_toxic"]]
            benign = [text_of(item) for item in dataset["train_benign"]]
            if random_sae:
                from experiments.alignment_geometry import randomise_saes

                randomise_saes(steerer, calibration_texts=toxic[:24])
            if shuffle_sae:
                from experiments.alignment_geometry import shuffle_saes

                shuffle_saes(steerer)
            # Methods whose intervention is dataset-specific (the prompt
            # baseline) need to know which behaviour they are steering.
            steerer.dataset = dataset_label
            steerer.fit(toxic, benign)
            # Persist the circuit itself: edge list, per-edge cosine and the
            # pooled per-layer vectors. This is what the selection-criterion
            # and interpretability figures are drawn from, and nothing else
            # in the pipeline saves it.
            circuit_rows.append({
                "model": model_key, "dataset": dataset_label,
                "variant": variant, "n_edges": len(steerer.circuit),
                # Pool size and mean cosine are what show the variants
                # were matched on budget and differ on geometry.
                "pool_size": int(getattr(steerer, "pool_size", -1)),
                "mean_cos": float(
                    getattr(steerer, "selected_cosine", float("nan"))
                ),
                "n_features": {
                    int(k): int(v) for k, v in
                    (steerer.feature_vectors()[1].items()
                     if hasattr(steerer, "feature_vectors") else [])
                },
                "layers": sorted(steerer.steer_vecs),
                "edges": [
                    {"src": e[0], "dst": e[1], "score": float(sc),
                     "cos": float(getattr(steerer, "edge_cosines", {}).get(e, float("nan")))}
                    for e, sc in steerer.circuit[: args.top_k * 2]
                ],
            })
            if args.release_saes:
                steerer.release_saes()

            def evaluate(items, coefficient, split):
                metrics = evaluate_coefficient(
                    steerer,
                    dataset_label,
                    items,
                    coefficient,
                    seed=args.seed,
                    samples_per_prompt=args.samples_per_prompt,
                    batch_size=args.batch_size,
                )
                base = metrics.pop("_base_scores", [])
                steered = metrics.pop("_steered_scores", [])
                for position, (b, s_) in enumerate(zip(base, steered)):
                    prompt_rows.append(
                        {
                            "model": model_key,
                            "dataset": dataset_label,
                            "split": split,
                            "seed": args.seed,
                            "coeff": coefficient,
                            "prompt_index": position,
                            "base_score": b,
                            "steered_score": s_,
                        }
                    )
                return metrics

            def row_for(split, coefficient, metrics, selected=False):
                return {
                    "method": method_name,
                    "variant": variant,
                    "model": model_key,
                    "dataset": dataset_label,
                    "split": split,
                    "seed": args.seed,
                    "coeff": coefficient,
                    "selected": selected,
                    **metrics,
                }

            fixed = fixed_lambdas.get(dataset_label)
            sweep_rows = []

            if fixed is not None:
                # One coefficient for this (model, dataset), identical for
                # every seed. Selection already happened on validation, so
                # no sweep runs here and the test prompts stay untouched
                # by any selection decision.
                best_coefficient = fixed
                print(
                    f"{model_key}/{dataset_label} "
                    f"fixed lambda={best_coefficient:+.1f} "
                    "(same for all seeds)"
                )
            else:
                selection_items = val_items or test_items
                selection_split = "val" if val_items else "test"
                if not val_items:
                    print(
                        f"{dataset_label}: no validation split "
                        "(increase --dataset-size); selecting on test, "
                        "which biases the reported delta upward."
                    )
                for coefficient in args.coefficients:
                    metrics = evaluate(
                        selection_items, coefficient, selection_split
                    )
                    row = row_for(selection_split, coefficient, metrics)
                    sweep_rows.append(row)
                    print(
                        f"{model_key}/{dataset_label} [{selection_split}] "
                        f"lambda={coefficient:+.1f} "
                        f"delta={metrics['delta']:+.3f} "
                        f"ppl={metrics['norm_ppl']:.2f} "
                        f"degen={metrics['degenerate_frac']:.2f}"
                    )
                numeric_rows.extend(sweep_rows)

            if fixed is None:
                # No fluency filtering: the largest delta wins, and every
                # coefficient's raw ppl / norm_ppl is emitted below so any
                # threshold can be applied post hoc.
                best_coefficient = select_best_coefficient(
                    pd.DataFrame(sweep_rows)
                )
                print(
                    f"{model_key}/{dataset_label} "
                    f"best lambda={best_coefficient:+.1f} "
                    f"(argmax delta on {selection_split}, unfiltered)"
                )

            # Full sweep on the held-out test split too, so the reported
            # table is not limited to one coefficient.
            for coefficient in args.coefficients:
                test_metrics = evaluate(test_items, coefficient, "test")
                numeric_rows.append(
                    row_for(
                        "test",
                        coefficient,
                        test_metrics,
                        selected=(coefficient == best_coefficient),
                    )
                )
                marker = (
                    " <- reported"
                    if coefficient == best_coefficient
                    else ""
                )
                print(
                    f"{model_key}/{dataset_label} [test] "
                    f"lambda={coefficient:+.1f} "
                    f"delta={test_metrics['delta']:+.4f} "
                    f"ppl={test_metrics['norm_ppl']:.4f} "
                    f"degen={test_metrics['degenerate_frac']:.2f}"
                    f"{marker}"
                )

            # Control: the same steering applied to benign prompts. A
            # targeted circuit should barely move this; a vector that just
            # suppresses fluency moves it as much as the toxic split.
            if benign_items:
                control_metrics = evaluate(
                    benign_items, best_coefficient, "test_benign"
                )
                numeric_rows.append(
                    row_for(
                        "test_benign",
                        best_coefficient,
                        control_metrics,
                        selected=True,
                    )
                )
                print(
                    f"{model_key}/{dataset_label} [test_benign] "
                    f"delta={control_metrics['delta']:+.4f} "
                    f"ppl={control_metrics['norm_ppl']:.4f}"
                )

            qualitative_output.extend(
                qualitative_rows(
                    pd.DataFrame(sweep_rows),
                    test_items,
                    steerer,
                    dataset_label,
                    model_key,
                    method_name=method_name,
                    n_samples=args.qualitative_samples,
                    seed=args.seed,
                    best_coefficient=best_coefficient,
                )
            )

        release_model(steerer)

    results_path = args.output_dir / "circuitsteer_results.csv"
    qualitative_path = args.output_dir / "circuitsteer_qual.csv"
    prompt_path = args.output_dir / "circuitsteer_prompt_scores.csv"
    pd.DataFrame(numeric_rows).to_csv(results_path, index=False)
    pd.DataFrame(qualitative_output).to_csv(
        qualitative_path,
        index=False,
    )
    pd.DataFrame(prompt_rows).to_csv(prompt_path, index=False)
    import json as _json
    (args.output_dir / "circuits.json").write_text(_json.dumps(circuit_rows))
    print(f"Results: {results_path}")
    print(f"Per-prompt scores: {prompt_path}")
    print(f"Qualitative outputs: {qualitative_path}")


if __name__ == "__main__":
    main()
