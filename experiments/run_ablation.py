"""Size-matched component ablation, one process per model.

`main.py --select-mode X` reloads the model, reruns circuit discovery and
regenerates the unsteered baseline for every variant. None of that
depends on the variant:

* Full / NoGeo / NegativeAlign / RandomEdges differ only in which edges
  are taken from the pool, and the pool (with its per-edge cosines) is
  built once per dataset. Selection is re-run on it.
* ShuffledSAE is the one variant that changes the pool, since it
  permutes decoder directions; it gets its own discovery, after which
  the original decoders are restored.
* DenseNoSAE runs the same discovery with no SAE at all: each residual
  dimension is a "feature" whose activation is the coordinate value and
  whose direction is the unit basis vector. Alignment between two basis
  vectors is 1 for the same dimension and 0 otherwise, so the only
  aligned edges link a dimension to itself at the next layer. It keeps
  the top `--dense-top-n` coordinates per layer, because an absolute
  threshold tuned for SAE activations marks hundreds of residual
  coordinates as active.
* MLPNeurons replaces each SAE with the MLP of the same layer: neuron k's
  activation is its post-nonlinearity value and its direction is its
  output weight W_out[k], normalised to unit length like an SAE decoder
  column. Neurons outnumber residual dimensions, so their directions are
  not orthogonal and the alignment test has real choices to make; what
  they lack relative to SAE features is monosemanticity. Activations are
  dense and signed, so it keeps the top `--dense-top-n` neurons per layer.
* The unsteered generations and their perplexities are identical for
  every variant (same prompts, same seed, no hooks), so they are cached.

Qualitative samples are skipped. Metrics come from the same
`evaluate_coefficient` as the main runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import pandas as pd
import torch

from circuitsteer import CircuitSteer
from circuitsteer.data import make_datasets, text_of
from circuitsteer.evaluation import evaluate_coefficient
from experiments.alignment_geometry import shuffle_saes
from experiments.common import add_circuit_arguments, config_from_args

VARIANTS = {
    "Full": "aligned",
    "NoGeo": "none",
    "NegativeAlign": "anti",
    "RandomEdges": "random",
}


def eval_items(dataset, n: int) -> list:
    """Test-first subset, topped up with same-shape items (as run_main)."""
    pool = list(dataset["test_prompts"]) + list(dataset.get("val_prompts") or [])

    def compatible(item) -> bool:
        if not pool:
            return True
        reference = pool[0]
        if isinstance(reference, tuple):
            return isinstance(item, tuple) and len(item) == len(reference)
        return isinstance(item, str)

    pool += [i for i in dataset["train_toxic"] if compatible(i) and i not in pool]
    return pool[:n]


def cache_baseline(steerer) -> None:
    """Memoise the lambda=0 generations and all perplexities."""
    generate_batch = steerer.generate_batch
    perplexity = steerer.perplexity
    generations: dict = {}
    perplexities: dict = {}

    def cached_generate(prompts, coefficient, seed, **kwargs):
        if coefficient != 0.0:
            return generate_batch(prompts, coefficient, seed, **kwargs)
        key = (tuple(prompts), seed)
        if key not in generations:
            generations[key] = generate_batch(prompts, 0.0, seed, **kwargs)
        return list(generations[key])

    def cached_perplexity(text):
        if text not in perplexities:
            perplexities[text] = perplexity(text)
        return perplexities[text]

    steerer.generate_batch = cached_generate
    steerer.perplexity = cached_perplexity


class DenseBasis(torch.nn.Module):
    """Stand-in for an SAE: identity encoder, standard-basis decoder."""

    def __init__(self, d_model: int, device, dtype) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(
            torch.eye(d_model, device=device, dtype=dtype),
            requires_grad=False)

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.to(self.W_dec.dtype)


class NeuronDictionary(torch.nn.Module):
    """Stand-in for an SAE: MLP neurons with unit output directions."""

    hook_name = "blocks.{}.mlp.hook_post"

    def __init__(self, w_out: torch.Tensor, dtype) -> None:
        super().__init__()
        directions = torch.nn.functional.normalize(w_out.float(), dim=1)
        self.W_dec = torch.nn.Parameter(directions.to(dtype),
                                        requires_grad=False)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.to(self.W_dec.dtype)


def select(steerer, mode: str) -> None:
    steerer.config = dataclasses.replace(steerer.config, select_mode=mode)
    steerer.circuit = steerer._select_edges(steerer.pool)
    cosines = [steerer.edge_cosines.get(e, 0.0) for e, _ in steerer.circuit]
    steerer.selected_cosine = (
        sum(cosines) / len(cosines) if cosines else float("nan")
    )
    steerer.build_steering_vectors()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_circuit_arguments(parser)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--eval-subset", type=int, default=50)
    parser.add_argument("--coefficient", type=float, default=-4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-steer-edges", type=int, default=60)
    parser.add_argument("--select-seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+",
                        default=[*VARIANTS, "ShuffledSAE"])
    parser.add_argument("--dense-top-n", type=int, default=32)
    parser.add_argument("--dense-diff-thresh", type=float, default=None,
                        help="Specificity gate for DenseNoSAE/MLPNeurons only.")
    parser.add_argument("--discovery-only", action="store_true",
                        help="Build circuits and report their size; no steering.")
    parser.add_argument("--coefficients", type=float, nargs="+", default=None,
                        help="Evaluate every variant at each of these "
                             "(overrides --coefficient).")
    args = parser.parse_args()
    args.select_mode = "aligned"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (model_key,) = args.models
    steerer = CircuitSteer(model_key, config=config_from_args(args))
    cache_baseline(steerer)
    datasets = make_datasets(labels=args.tasks, n=args.dataset_size,
                             seed=args.seed)
    rows, prompt_rows = [], []
    out_csv = args.output_dir / f"ablation_{model_key}.csv"

    coefficients = args.coefficients or [args.coefficient]

    def run(variant: str, dataset_label: str, items) -> None:
        if args.discovery_only:
            print(f"SIZE {variant} {model_key} {dataset_label}: "
                  f"edges={len(steerer.circuit)} pool={steerer.pool_size}",
                  flush=True)
            return
        for coefficient in coefficients:
            run_one(variant, dataset_label, items, coefficient)

    def run_one(variant: str, dataset_label: str, items, coefficient) -> None:
        t0 = time.time()
        metrics = evaluate_coefficient(
            steerer, dataset_label, items, coefficient,
            seed=args.seed, batch_size=args.batch_size,
        )
        base = metrics.pop("_base_scores", [])
        steered = metrics.pop("_steered_scores", [])
        for i, (b, s) in enumerate(zip(base, steered)):
            prompt_rows.append({"model": model_key, "dataset": dataset_label,
                                "variant": variant, "coeff": coefficient,
                                "prompt_index": i,
                                "base_score": b, "steered_score": s})
        rows.append({
            "model": model_key, "dataset": dataset_label, "variant": variant,
            "seed": args.seed, "coeff": coefficient,
            "n_edges": len(steerer.circuit), "pool_size": steerer.pool_size,
            "mean_cos": steerer.selected_cosine,
            "layers": json.dumps(sorted(steerer.steer_vecs)),
            **metrics,
        })
        print(f"RESULT {variant} {model_key} {dataset_label} "
              f"lambda={coefficient:g}: "
              f"delta={metrics['delta']:.4f} norm_ppl={metrics['norm_ppl']:.3f} "
              f"edges={len(steerer.circuit)} cos={steerer.selected_cosine:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # Written after every unit so a dead box still leaves results.
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        pd.DataFrame(prompt_rows).to_csv(
            args.output_dir / f"ablation_{model_key}_prompts.csv", index=False)

    for dataset_label, dataset in datasets.items():
        t0 = time.time()
        items = eval_items(dataset, args.eval_subset)
        toxic = [text_of(i) for i in dataset["train_toxic"]]
        benign = [text_of(i) for i in dataset["train_benign"]]
        steerer.dataset = dataset_label

        steerer.fit(toxic, benign)
        print(f"TIMING discovery {dataset_label}: {time.time() - t0:.0f}s",
              flush=True)
        for variant, mode in VARIANTS.items():
            if variant in args.variants:
                select(steerer, mode)
                run(variant, dataset_label, items)

        if "ShuffledSAE" in args.variants:
            originals = {l: s.W_dec.detach().clone()
                         for l, s in steerer.saes.items()}
            shuffle_saes(steerer)
            steerer.config = dataclasses.replace(
                steerer.config, select_mode="aligned")
            steerer.discover_circuit(toxic, benign)
            steerer.build_steering_vectors()
            run("ShuffledSAE", dataset_label, items)
            for layer, w in originals.items():
                steerer.saes[layer].W_dec = torch.nn.Parameter(
                    w, requires_grad=False)
        if "DenseNoSAE" in args.variants:
            saes, config = steerer.saes, steerer.config
            d_model = steerer.model.cfg.d_model
            steerer.saes = {l: DenseBasis(d_model, steerer.device, steerer.dtype)
                            for l in saes}
            steerer.config = dataclasses.replace(
                config, select_mode="aligned", act_top_n=args.dense_top_n,
                **({} if args.dense_diff_thresh is None
                   else {"diff_thresh": args.dense_diff_thresh}))
            steerer.discover_circuit(toxic, benign)
            steerer.build_steering_vectors()
            run("DenseNoSAE", dataset_label, items)
            steerer.saes, steerer.config = saes, config
        if "MLPNeurons" in args.variants:
            saes, config = steerer.saes, steerer.config
            def write_directions(block) -> torch.Tensor:
                # Gemma-2 normalises the MLP output again before adding it
                # to the residual stream; its learned scale is part of the
                # direction a neuron actually writes.
                w_out = block.mlp.W_out.detach()
                post = getattr(block, "ln2_post", None)
                if post is not None and hasattr(post, "w"):
                    w_out = w_out * post.w.detach()
                return w_out

            steerer.saes = {
                l: NeuronDictionary(write_directions(steerer.model.blocks[l]),
                                    steerer.dtype)
                for l in saes}
            steerer.config = dataclasses.replace(
                config, select_mode="aligned", act_top_n=args.dense_top_n,
                **({} if args.dense_diff_thresh is None
                   else {"diff_thresh": args.dense_diff_thresh}))
            steerer.discover_circuit(toxic, benign)
            steerer.build_steering_vectors()
            run("MLPNeurons", dataset_label, items)
            steerer.saes, steerer.config = saes, config
        print(f"TIMING dataset {dataset_label}: {time.time() - t0:.0f}s",
              flush=True)

    print("ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
