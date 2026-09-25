"""Dump the features each selection rule picks, for the semantic map.

The semantic map asks where in the SAE's space of meanings a selection
rule lands. That needs the actual feature ids each rule chose, which only
circuit discovery produces - so this runs discovery once per (model,
dataset) and applies all four rules to the SAME pool, then writes the
selections and a 3-D PCA of the chosen decoder directions.

No generation and no scoring, so it is far cheaper than a steering run:
the cost is one pass over the training text per dataset.

Previously this lived only inside a packed Colab job; it is a repo script
now so the Emotion maps can be produced the same way as the RTP ones.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from circuitsteer.config import CircuitConfig
from circuitsteer.core import CircuitSteer
from circuitsteer.data import make_datasets
from circuitsteer.selection import select_edges

MODES = ["aligned", "none", "anti", "random"]


def export(steerer, config, model_key: str, task: str, k: int,
           n: int, seed: int, out_dir: Path) -> None:
    data = make_datasets(labels=[task], n=n, seed=seed)[task]
    steerer.discover_circuit(data["train_toxic"], data["train_benign"])
    pool, cosines = steerer.pool, steerer.edge_cosines
    print(f"{model_key}/{task}: pool={len(pool)} edges", flush=True)

    selections, vectors = {}, {}
    for mode in MODES:
        chosen = select_edges(pool, cosines, mode=mode, k=k,
                              sim_thresh=config.sim_thresh, seed=0)
        steerer.circuit = chosen
        vecs, counts = steerer.feature_vectors()
        per_layer: dict[int, set[int]] = {}
        for (source, _destination), _ in chosen:
            layer, feature = source.split("_")
            per_layer.setdefault(int(layer[1:]), set()).add(int(feature))
        selections[mode] = {
            "edges": [{"src": e[0], "dst": e[1],
                       "cos": float(cosines.get(e, 0.0))} for e, _ in chosen],
            "features": {str(a): sorted(b) for a, b in per_layer.items()},
            "counts": {str(a): int(b) for a, b in counts.items()},
            "mean_cos": (
                float(np.mean([cosines.get(e, 0.0) for e, _ in chosen]))
                if chosen else float("nan")
            ),
        }
        vectors[mode] = {str(l): v.detach().float().cpu().numpy()
                         for l, v in vecs.items()}

    rows, meta = [], []
    for mode in MODES:
        for layer, features in selections[mode]["features"].items():
            weights = steerer.saes[int(layer)].W_dec[
                torch.tensor(features, device=steerer.device)
            ].detach().float().cpu().numpy()
            weights = weights / (
                np.linalg.norm(weights, axis=1, keepdims=True) + 1e-9
            )
            rows.append(weights)
            meta += [{"mode": mode, "layer": int(layer), "feature": f}
                     for f in features]
    matrix = np.concatenate(rows, 0)
    centred = matrix - matrix.mean(0, keepdims=True)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    basis = right[:3]
    coords = centred @ basis.T

    efficiency = {}
    for mode in MODES:
        parts = list(vectors[mode].values())
        total = sum(float(np.linalg.norm(v)) for v in parts)
        efficiency[mode] = (
            float(np.linalg.norm(sum(parts)) / total) if total
            else float("nan")
        )

    payload = {
        "model": model_key, "task": task, "k": k, "pool_size": len(pool),
        "selections": selections, "efficiency": efficiency,
        "explained_var": (singular[:3] ** 2 / (singular ** 2).sum()).tolist(),
        "points": [{**m, "xyz": [round(float(c), 5) for c in coords[i]]}
                   for i, m in enumerate(meta)],
        "layer_norms": {
            m: {l: float(np.linalg.norm(v)) for l, v in vectors[m].items()}
            for m in MODES
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"demo_{model_key}_{task}.json"
    path.write_text(json.dumps(payload))
    print(f"wrote {path}  efficiency={efficiency}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gemma", "llama"])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args()

    for model_key in args.models:
        config = CircuitConfig(select_mode="aligned", n_steer_edges=args.k)
        print(f"\n===== loading {model_key} =====", flush=True)
        steerer = CircuitSteer(model_key, config=config)
        for task in args.tasks:
            try:
                export(steerer, config, model_key, task, args.k,
                       args.dataset_size, args.seed, args.out_dir)
            except Exception as exc:                      # noqa: BLE001
                import traceback
                print(f"FAILED {model_key}/{task}: {exc}", flush=True)
                traceback.print_exc()
        del steerer
        gc.collect()
        torch.cuda.empty_cache()
    print("GEOMETRY DONE", flush=True)


if __name__ == "__main__":
    main()
