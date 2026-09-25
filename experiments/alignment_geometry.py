"""Experiment 1: is geometric alignment a good circuit criterion?

The paper's mechanism claim is destructive interference - features that
encode the same concept at different depths are often geometrically
misaligned, so intervening at several layers cancels instead of
reinforcing. This dumps the per-layer steering vectors for four circuit
variants so the claim can be read off the geometry directly:

    Full          cos(d_l, d_l+1) >  sim_thresh   (the method)
    NoGeo         co-activation only, no cosine condition
    NegativeAlign cos(d_l, d_l+1) < -sim_thresh   (anti-aligned only)
    RandomSAE     trained SAE replaced by a random projection

Laid head-to-tail, an aligned circuit's vectors should march outward into
a long resultant while an anti-aligned one curls back on itself. The
scalar version of that picture is the alignment efficiency

    ||sum_l v_l|| / sum_l ||v_l||

which is 1.0 when every layer pushes the same way and ~0 when they
cancel. It is scale-free, so it does not depend on the coefficient.

Only circuit discovery runs here - no generation, no scoring - and
vectors are emitted as JSON per variant the moment a fit returns, so a
reclaimed session costs at most one fit.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from circuitsteer import CircuitSteer
from circuitsteer.config import CircuitConfig
from circuitsteer.data import LOADERS, text_of

VARIANTS = ["Full", "NoGeo", "NegativeAlign", "RandomSAE"]


@torch.no_grad()
def randomise_saes(steerer, calibration_texts=None, seed: int = 0) -> None:
    """Replace the learned dictionary with a random one of the same shape.

    A naive randomisation makes the ablation vacuous: random encoder rows
    almost never push a feature past tau_act, so no features co-activate,
    no edges form and the circuit comes back empty. That confounds "the
    directions are random" with "the activation statistics are different",
    and only the first is the thing we want to test.

    So after randomising we calibrate the encoder bias per layer against
    real activations, choosing it so the random dictionary fires roughly
    as many features per input as the trained one did. Sparsity is then
    matched and the only remaining difference is that the directions carry
    no learned structure.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets = {}
    if calibration_texts:
        for layer, sae in steerer.saes.items():
            counts = []
            for text in calibration_texts:
                _, cache = steerer.model.run_with_cache(
                    text, stop_at_layer=layer + 1)
                residual = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
                acts = sae.encode(residual)[0]
                counts.append(int((acts > steerer.config.act_thresh).sum()))
            targets[layer] = max(1, int(np.median(counts)))
        print(f"RandomSAE: trained-SAE L0 targets {targets}")

    # Decoder directions need matched COSINE structure as well as matched
    # sparsity. Pure i.i.d. directions in d_model dimensions are almost
    # exactly orthogonal (cos ~ 0 +/- 1/sqrt(d)), so the cos > tau_sim
    # criterion is a ~5-sigma event, no edges ever form, and the ablation
    # becomes vacuous rather than informative. We therefore mix each random
    # direction with a shared component, d_i = a*u + sqrt(1-a^2)*n_i, which
    # gives cos(d_i, d_j) ~ a^2; a is chosen per layer so the typical
    # cosine matches the trained dictionary's.
    for layer, sae in steerer.saes.items():
        w_dec = getattr(sae, "W_dec", None)
        if w_dec is not None:
            real = torch.nn.functional.normalize(
                w_dec.float()[:4096].cpu(), dim=1)
            idx = torch.randperm(real.shape[0], generator=generator)[:768]
            sample = real[idx]
            gram = (sample @ sample.T).fill_diagonal_(0)
            tau = steerer.config.sim_thresh
            # The criterion fires on the TAIL, not the typical pair: the
            # trained dictionary is itself near-orthogonal (median |cos|
            # ~0.014), so matching the median leaves every random pair far
            # below tau and no edge can ever form. Match the exceedance
            # RATE instead - the quantity that actually governs whether
            # edges appear - by binary-searching the mixing weight.
            target_rate = float((gram > tau).float().mean())
            shared = torch.randn(w_dec.shape[1], generator=generator)
            shared = shared / shared.norm()
            noise = torch.nn.functional.normalize(
                torch.randn((768, w_dec.shape[1]), generator=generator), dim=1)

            def rate_for(a_val: float) -> float:
                mixed = torch.nn.functional.normalize(
                    a_val * shared.unsqueeze(0)
                    + np.sqrt(1 - a_val ** 2) * noise, dim=1)
                g = (mixed @ mixed.T).fill_diagonal_(0)
                return float((g > tau).float().mean())

            lo, hi = 0.0, 0.95
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if rate_for(mid) < target_rate:
                    lo = mid
                else:
                    hi = mid
            a = 0.5 * (lo + hi)
            full_noise = torch.nn.functional.normalize(
                torch.randn(w_dec.shape, generator=generator), dim=1)
            mixed = torch.nn.functional.normalize(
                a * shared.unsqueeze(0)
                + np.sqrt(1 - a ** 2) * full_noise, dim=1)
            scale = float(w_dec.float().norm(dim=1).median())
            setattr(sae, "W_dec", torch.nn.Parameter(
                (mixed * scale).to(w_dec.device, w_dec.dtype),
                requires_grad=False))
            print(f"RandomSAE L{layer}: trained P(cos>{tau})={target_rate:.5f}"
                  f" -> mixing a={a:.3f} (random rate {rate_for(a):.5f})")
        w_enc = getattr(sae, "W_enc", None)
        if w_enc is not None:
            scale = float(w_enc.float().std())
            noise = torch.randn(w_enc.shape, generator=generator,
                                dtype=torch.float32)
            setattr(sae, "W_enc", torch.nn.Parameter(
                (noise * scale).to(w_enc.device, w_enc.dtype),
                requires_grad=False))
        if hasattr(sae, "b_enc") and sae.b_enc is not None:
            sae.b_enc = torch.nn.Parameter(
                torch.zeros_like(sae.b_enc), requires_grad=False)

    if not calibration_texts:
        return

    # Calibrate: pick the bias so that the target number of features clear
    # tau_act on real inputs.
    for layer, sae in steerer.saes.items():
        pre = []
        for text in calibration_texts:
            _, cache = steerer.model.run_with_cache(
                text, stop_at_layer=layer + 1)
            residual = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
            pre.append(sae.encode(residual)[0].float().cpu())
        stacked = torch.stack(pre)
        k = targets[layer]
        # value that the k-th largest pre-activation takes, per input
        kth = torch.topk(stacked, k=min(k, stacked.shape[1]), dim=1).values[:, -1]
        shift = float(steerer.config.act_thresh - kth.median())
        if hasattr(sae, "b_enc") and sae.b_enc is not None:
            sae.b_enc = torch.nn.Parameter(
                torch.full_like(sae.b_enc, shift), requires_grad=False)
        print(f"RandomSAE L{layer}: bias shift {shift:+.3f} to match L0={k}")


@torch.no_grad()
def shuffle_saes(steerer, seed: int = 0) -> None:
    """Permute which feature index owns which decoder direction.

    RandomSAE is structurally empty: an edge needs a pair that is BOTH
    co-activating AND aligned, and in a trained dictionary those two are
    correlated because features that fire together were learned to point
    together. Randomising the directions makes them independent, so the
    joint rate collapses to ~0 no matter how carefully each marginal is
    matched. Shuffling keeps the real encoder and the real set of decoder
    directions - identical shapes, identical geometry, identical sparsity -
    and destroys only that learned correspondence, which is the thing the
    ablation is meant to isolate.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for layer, sae in steerer.saes.items():
        w_dec = getattr(sae, "W_dec", None)
        if w_dec is None:
            continue
        perm = torch.randperm(w_dec.shape[0], generator=generator)
        setattr(sae, "W_dec", torch.nn.Parameter(
            w_dec[perm.to(w_dec.device)].clone(), requires_grad=False))
        print(f"ShuffledSAE L{layer}: permuted {w_dec.shape[0]} directions")


def geometry(vectors: dict[int, torch.Tensor]) -> dict:
    """Chain geometry of the per-layer steering vectors."""
    layers = sorted(vectors)
    # SAE decoder weights carry requires_grad, so the pooled
    # steering vectors inherit it; detach before leaving torch.
    mat = np.stack(
        [vectors[l].detach().float().cpu().numpy() for l in layers]
    )
    norms = np.linalg.norm(mat, axis=1)
    resultant = mat.sum(0)
    efficiency = (
        float(np.linalg.norm(resultant) / norms.sum()) if norms.sum() else 0.0
    )
    # Pairwise cosines between consecutive layers.
    cos = []
    for a, b in zip(mat[:-1], mat[1:]):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos.append(float(a @ b / denom) if denom else 0.0)
    return {
        "layers": layers,
        "norms": [float(v) for v in norms],
        "resultant_norm": float(np.linalg.norm(resultant)),
        "sum_of_norms": float(norms.sum()),
        "alignment_efficiency": efficiency,
        "consecutive_cosines": cos,
        "vectors": mat.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["gemma", "llama"])
    parser.add_argument("--datasets", nargs="+", default=list(LOADERS))
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load the model and SAEs ONCE. The variants differ only in how
    # edges are filtered, so re-fitting is cheap next to a reload.
    steerer = CircuitSteer(args.model, config=CircuitConfig())
    splits = {}
    for label in args.datasets:
        split = LOADERS[label](n=args.dataset_size, seed=args.seed)
        splits[label] = ([text_of(i) for i in split["train_toxic"]],
                         [text_of(i) for i in split["train_benign"]])

    def run(label, variant, mode):
        toxic, benign = splits[label]
        steerer.config = CircuitConfig(align_mode=mode)
        try:
            steerer.fit(toxic, benign)
        except Exception as error:
            print("###GEO###" + json.dumps({
                "model": args.model, "dataset": label, "variant": variant,
                "error": str(error)[:200]}), flush=True)
            return
        if not steerer.steer_vecs:
            payload = {"model": args.model, "dataset": label,
                       "variant": variant, "empty_circuit": True,
                       "n_edges": len(steerer.circuit)}
        else:
            payload = {"model": args.model, "dataset": label,
                       "variant": variant, "n_edges": len(steerer.circuit),
                       **geometry(steerer.steer_vecs)}
        print("###GEO###" + json.dumps(payload), flush=True)

    # Trained-SAE variants share the loaded SAEs.
    for label in args.datasets:
        for variant, mode in (("Full", "positive"), ("NoGeo", "none"),
                              ("NegativeAlign", "negative")):
            run(label, variant, mode)

    # RandomSAE is destructive to the weights, so it goes last.
    randomise_saes(steerer)
    for label in args.datasets:
        run(label, "RandomSAE", "positive")

    print("ALIGNMENT GEOMETRY DONE", flush=True)


if __name__ == "__main__":
    main()
