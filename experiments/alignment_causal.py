"""Do the layers cooperate, and is the angle what makes them cooperate?

Two tests of the destructive-interference claim that the static geometry
of the steering vectors cannot settle.

ADDITIVITY. Steer at each circuit layer alone, measure the residual-stream
displacement each one produces, then steer at all of them together. If
the layers cooperate, the joint displacement is about the sum of the
individual ones; if they fight, it is much less. The ratio

    ||d_all|| / sum_l ||d_l||

*is* destructive interference, measured rather than inferred. Reported for
Full / NoGeo / NegativeAlign.

ANGLE SWEEP. The ablations above confound angle with circuit size
(NegativeAlign keeps ~10 edges, Full ~140), so they cannot isolate the
angle. Here we take the Full circuit and synthetically rotate every later
layer's vector to sit at a chosen cosine c with the first layer's, holding
norms fixed, and sweep c from -1 to +1. Only the angle varies, so any
response in behaviour or fluency is caused by alignment.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from circuitsteer import CircuitSteer
from circuitsteer.config import CircuitConfig
from circuitsteer.data import LOADERS, text_of
from circuitsteer.evaluation import evaluate_coefficient

FIXED_LAMBDA = {
    ("gemma", "RTP"): -3.0, ("gemma", "Sycophancy"): -4.0,
    ("gemma", "Jigsaw"): -4.0, ("gemma", "Emotion"): -4.0,
    ("llama", "RTP"): -4.0, ("llama", "Sycophancy"): -4.0,
    ("llama", "Jigsaw"): -4.0, ("llama", "Emotion"): -2.0,
}


def rotate_to_cosine(anchor: torch.Tensor, vector: torch.Tensor,
                     cosine: float) -> torch.Tensor:
    """Rebuild `vector` at a chosen cosine to `anchor`, norm preserved.

    Decompose the vector into the anchor direction plus the orthogonal
    remainder, then recombine with weights (c, sqrt(1-c^2)). The result
    has exactly the requested cosine and the original magnitude, so the
    sweep changes the angle and nothing else.
    """
    a = anchor.float()
    v = vector.float()
    a_hat = a / (a.norm() + 1e-12)
    parallel = (v @ a_hat) * a_hat
    perpendicular = v - parallel
    p_norm = perpendicular.norm()
    # Relative tolerance: the leftover after projecting out the anchor
    # scales with the vector's own magnitude, so an absolute threshold
    # never fires and p_hat becomes amplified floating-point noise.
    if float(p_norm) <= 1e-6 * float(v.norm() + 1e-12):
        perpendicular = torch.randn_like(v)
        perpendicular = perpendicular - (perpendicular @ a_hat) * a_hat
        p_norm = perpendicular.norm()
    p_hat = perpendicular / (p_norm + 1e-12)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    sine = float(np.sqrt(max(0.0, 1.0 - cosine ** 2)))
    return (v.norm() * (cosine * a_hat + sine * p_hat)).to(vector.dtype)


@torch.no_grad()
def displacement(steerer, texts, layers, coefficient, probe_layer):
    """Mean change in the residual stream at `probe_layer` under steering."""
    base, steered = [], []
    for text in texts:
        steerer.model.reset_hooks()
        _, cache = steerer.model.run_with_cache(
            text, stop_at_layer=probe_layer + 1)
        base.append(cache[f"blocks.{probe_layer}.hook_resid_post"][0, -1, :]
                    .float().cpu().numpy())
        hooks = steerer.hooks(coefficient, layers=layers)
        steerer.model.reset_hooks()
        with steerer.model.hooks(fwd_hooks=hooks):
            _, cache = steerer.model.run_with_cache(
                text, stop_at_layer=probe_layer + 1)
        steered.append(cache[f"blocks.{probe_layer}.hook_resid_post"][0, -1, :]
                       .float().cpu().numpy())
    return (np.stack(steered) - np.stack(base)).mean(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["gemma", "llama"])
    p.add_argument("--datasets", nargs="+", default=["RTP", "Sycophancy"])
    p.add_argument("--dataset-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-probe", type=int, default=60)
    p.add_argument("--skip-additivity", action="store_true",
                   help="Run only the angle sweep (the additivity arm is "
                        "confounded: NegativeAlign changes circuit size as "
                        "well as angle).")
    p.add_argument("--cosines", nargs="+", type=float,
                   default=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    args = p.parse_args()

    steerer = CircuitSteer(args.model, config=CircuitConfig())
    for label in args.datasets:
        split = LOADERS[label](n=args.dataset_size, seed=args.seed)
        toxic = [text_of(i) for i in split["train_toxic"]]
        benign = [text_of(i) for i in split["train_benign"]]
        probe_texts = [text_of(i) for i in split["test_prompts"]][: args.n_probe]
        coefficient = FIXED_LAMBDA[(args.model, label)]

        # ---------- additivity, per circuit variant ----------
        for variant, mode in (() if args.skip_additivity else
                              (("Full", "positive"), ("NoGeo", "none"),
                               ("NegativeAlign", "negative"))):
            steerer.config = CircuitConfig(align_mode=mode)
            steerer.fit(toxic, benign)
            layers = sorted(steerer.steer_vecs)
            if len(layers) < 2:
                print("###ADD###" + json.dumps({
                    "model": args.model, "dataset": label, "variant": variant,
                    "skipped": f"only {len(layers)} layer(s)"}), flush=True)
                continue
            probe = max(layers)
            singles = [
                displacement(steerer, probe_texts, [l], coefficient, probe)
                for l in layers
            ]
            joint = displacement(
                steerer, probe_texts, layers, coefficient, probe)
            sum_of_norms = float(sum(np.linalg.norm(d) for d in singles))
            ratio = (float(np.linalg.norm(joint)) / sum_of_norms
                     if sum_of_norms else 0.0)
            print("###ADD###" + json.dumps({
                "model": args.model, "dataset": label, "variant": variant,
                "layers": layers, "probe_layer": probe,
                "n_edges": len(steerer.circuit),
                "single_norms": [float(np.linalg.norm(d)) for d in singles],
                "joint_norm": float(np.linalg.norm(joint)),
                "sum_of_norms": sum_of_norms,
                "additivity": ratio,
                # how much of each single displacement survives jointly
                "cos_joint_vs_sum": float(
                    joint @ sum(singles) /
                    (np.linalg.norm(joint) * np.linalg.norm(sum(singles)) + 1e-12)
                )}), flush=True)

        # ---------- angle sweep on the Full circuit ----------
        steerer.config = CircuitConfig(align_mode="positive")
        steerer.fit(toxic, benign)
        original = {l: v.clone() for l, v in steerer.steer_vecs.items()}
        layers = sorted(original)
        if len(layers) < 2:
            print("###SWEEP###" + json.dumps({
                "model": args.model, "dataset": label,
                "skipped": "circuit has <2 layers"}), flush=True)
            continue
        anchor = original[layers[0]]
        test_items = split["test_prompts"]
        for cosine in args.cosines:
            steerer.steer_vecs = {layers[0]: anchor.clone()}
            for l in layers[1:]:
                steerer.steer_vecs[l] = rotate_to_cosine(
                    anchor, original[l], cosine)
            metrics = evaluate_coefficient(
                steerer, label, test_items, coefficient,
                seed=args.seed, batch_size=32)
            metrics.pop("_base_scores", None)
            metrics.pop("_steered_scores", None)
            probe = max(layers)
            disp = displacement(
                steerer, probe_texts, layers, coefficient, probe)
            print("###SWEEP###" + json.dumps({
                "model": args.model, "dataset": label, "cosine": cosine,
                "coeff": coefficient, "layers": layers,
                "displacement_norm": float(np.linalg.norm(disp)),
                **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                   for k, v in metrics.items()}}), flush=True)
        steerer.steer_vecs = original
    print("ALIGNMENT CAUSAL DONE", flush=True)


if __name__ == "__main__":
    main()
