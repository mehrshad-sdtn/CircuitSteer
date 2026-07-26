from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from circuitsteer import CircuitSteer
from circuitsteer.data import load_rtp, text_of
from circuitsteer.io import append_row, read_rows
from circuitsteer.scoring import toxicity_score
from experiments.common import (
    add_circuit_arguments,
    config_from_args,
    release_model,
)

FIELDS = [
    "model",
    "method",
    "dataset",
    "k",
    "layers",
    "coeff",
    "delta",
    "delta_std",
    "norm_ppl",
    "n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CircuitSteer and CAA across layer counts."
    )
    add_circuit_arguments(parser)
    parser.add_argument("--best-lambda", type=float, default=-3.0)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        metavar="K",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize steering vectors before evaluation.",
    )
    return parser.parse_args()


@torch.no_grad()
def build_caa_vectors(
    steerer: CircuitSteer,
    layers: list[int],
    toxic: list[str],
    benign: list[str],
) -> dict[int, torch.Tensor]:
    def mean_residual(texts: list[str]) -> dict[int, torch.Tensor]:
        accumulated = {layer: None for layer in layers}
        for text in tqdm(texts, desc="CAA activations", leave=False):
            _, cache = steerer.model.run_with_cache(
                text,
                stop_at_layer=layers[-1] + 1,
            )
            for layer in layers:
                residual = cache[
                    f"blocks.{layer}.hook_resid_post"
                ][:, -1, :][0].float()
                previous = accumulated[layer]
                accumulated[layer] = (
                    residual if previous is None else previous + residual
                )
        return {
            layer: value / max(len(texts), 1)
            for layer, value in accumulated.items()
        }

    toxic_means = mean_residual(toxic)
    benign_means = mean_residual(benign)
    return {
        layer: (toxic_means[layer] - benign_means[layer]).to(
            steerer.dtype
        )
        for layer in layers
    }


def maybe_normalize(
    vectors: dict[int, torch.Tensor],
    normalize: bool,
) -> dict[int, torch.Tensor]:
    if not normalize:
        return vectors
    normalized = {}
    for layer, vector in vectors.items():
        norm = float(vector.norm())
        normalized[layer] = (
            (vector / norm).to(vector.dtype) if norm > 1e-8 else vector
        )
    return normalized


@torch.no_grad()
def generate_restricted(
    steerer: CircuitSteer,
    prompt: str,
    vectors: dict[int, torch.Tensor],
    layers: list[int],
    coefficient: float,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    hooks = (
        steerer.hooks(coefficient, vectors=vectors, layers=layers)
        if coefficient != 0.0
        else []
    )
    kwargs = {
        "max_new_tokens": 40,
        "temperature": 1.0,
        "verbose": False,
    }
    if hooks:
        with steerer.model.hooks(fwd_hooks=hooks):
            output = steerer.model.generate(prompt, **kwargs)
    else:
        output = steerer.model.generate(prompt, **kwargs)
    return output[len(prompt):].strip()


def save_plot(results_path, plot_path) -> None:
    frame = pd.read_csv(results_path)
    models = [
        model
        for model in ("gemma", "llama")
        if model in frame["model"].unique()
    ]
    if not models:
        return
    figure, axes = plt.subplots(
        len(models),
        2,
        figsize=(11, 4 * len(models)),
        squeeze=False,
    )
    metrics = [
        ("delta", "Behavioral reduction delta"),
        ("norm_ppl", "Normalized PPL"),
    ]
    for row_index, model_key in enumerate(models):
        subset = frame[frame["model"] == model_key]
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index][column_index]
            for method, style in (
                ("CircuitSteer", "o-"),
                ("CAA", "s--"),
            ):
                method_rows = subset[
                    subset["method"] == method
                ].sort_values("k")
                axis.plot(
                    method_rows["k"],
                    method_rows[metric],
                    style,
                    label=method,
                )
            if metric == "norm_ppl":
                axis.axhline(1.0, color="gray", linestyle=":", linewidth=1)
            axis.set_xlabel("Number of intervention layers")
            axis.set_ylabel(label)
            axis.set_title(f"{model_key} / RTP")
            axis.set_xticks(sorted(subset["k"].unique()))
            axis.legend()
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_path = args.output_dir / "layercount_rtp.csv"
    plot_path = args.output_dir / "layercount_rtp.png"
    done = {
        (row["model"], row["method"], int(row["k"]))
        for row in read_rows(results_path)
    }
    data = load_rtp(n=args.dataset_size, seed=args.seed)
    toxic = [text_of(item) for item in data["train_toxic"]]
    benign = [text_of(item) for item in data["train_benign"]]
    test_prompts = [
        text_of(item)
        for item in data["test_prompts"][: args.eval_samples]
    ]

    for model_key in args.models:
        if all(
            (model_key, method, k) in done
            for method in ("CircuitSteer", "CAA")
            for k in args.layers
        ):
            print(f"{model_key}: already complete")
            continue

        steerer = CircuitSteer(model_key, config=config_from_args(args))
        steerer.fit(toxic, benign)
        try:
            circuit_vectors, feature_counts = steerer.feature_vectors(
                include_destinations=True
            )
            populated_layers = [
                layer
                for layer in steerer.sae_layers
                if layer in circuit_vectors
            ]
            caa_all = build_caa_vectors(
                steerer,
                list(steerer.sae_layers),
                toxic,
                benign,
            )
            caa_vectors = {
                layer: caa_all[layer] for layer in populated_layers
            }
            circuit_vectors = maybe_normalize(
                circuit_vectors,
                args.normalize,
            )
            caa_vectors = maybe_normalize(caa_vectors, args.normalize)
            order = sorted(
                populated_layers,
                key=lambda layer: -feature_counts.get(layer, 0),
            )
            layer_counts = [k for k in args.layers if k <= len(order)]

            base = []
            for index, prompt in enumerate(
                tqdm(test_prompts, desc="Base generation", leave=False)
            ):
                steerer.model.reset_hooks()
                output = generate_restricted(
                    steerer,
                    prompt,
                    {},
                    [],
                    0.0,
                    args.seed + index,
                )
                base.append(
                    (
                        toxicity_score(output, steerer.device),
                        steerer.perplexity(output),
                    )
                )
            base_scores = np.asarray([item[0] for item in base])
            mean_base_ppl = float(np.mean([item[1] for item in base]))

            for method, vectors in (
                ("CircuitSteer", circuit_vectors),
                ("CAA", caa_vectors),
            ):
                for k in layer_counts:
                    key = (model_key, method, k)
                    if key in done:
                        continue
                    selected_layers = order[:k]
                    steered_scores = []
                    steered_perplexities = []
                    for index, prompt in enumerate(test_prompts):
                        steerer.model.reset_hooks()
                        output = generate_restricted(
                            steerer,
                            prompt,
                            vectors,
                            selected_layers,
                            args.best_lambda,
                            args.seed + index,
                        )
                        steered_scores.append(
                            toxicity_score(output, steerer.device)
                        )
                        steered_perplexities.append(
                            steerer.perplexity(output)
                        )
                    differences = base_scores - np.asarray(steered_scores)
                    normalized_ppl = float(
                        np.mean(steered_perplexities)
                        / max(mean_base_ppl, 1e-6)
                    )
                    row = {
                        "model": model_key,
                        "method": method,
                        "dataset": "RTP",
                        "k": k,
                        "layers": "|".join(map(str, selected_layers)),
                        "coeff": args.best_lambda,
                        "delta": round(float(differences.mean()), 4),
                        "delta_std": (
                            round(float(differences.std(ddof=1)), 4)
                            if len(differences) > 1
                            else 0.0
                        ),
                        "norm_ppl": round(normalized_ppl, 4),
                        "n": len(test_prompts),
                    }
                    append_row(results_path, row, FIELDS)
                    done.add(key)
                    print(
                        f"{model_key}/{method} k={k}: "
                        f"delta={row['delta']:+.4f} "
                        f"ppl={row['norm_ppl']:.3f}"
                    )
        finally:
            release_model(steerer)

    if results_path.exists():
        save_plot(results_path, plot_path)
    print(f"Results: {results_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
