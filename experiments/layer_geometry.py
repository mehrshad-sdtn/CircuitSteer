"""Motivation study: does a trait's SAE direction stay aligned across depth?

For one behaviour, every layer gets a "trait direction": the top-k SAE
features ranked by contrastive activation (mean on trait prompts minus
mean on benign prompts), their decoder vectors summed with those
contrastive weights and normalised. This is what naive per-layer
steering would add. The question is how well layer l's direction agrees
with layer l+1, l+2, ...

A low cosine is only evidence of misalignment relative to two anchors:

* null  - the same construction on k random live features, which is
          what high-dimensional near-orthogonality alone produces;
* ceiling - split-half reliability at the SAME layer: two disjoint halves
          of the prompts, each giving its own direction. No cross-layer
          cosine can be expected to beat the noise floor this measures.

The dense difference-of-means residual direction is computed alongside,
with no SAE involved, to separate "the trait rotates with depth" from
"each layer's SAE splits a shared trait into different features".

Only forward passes to cache residuals - no generation, no steering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from circuitsteer.config import MODEL_CONFIGS
from circuitsteer.data import make_datasets, text_of


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


def trait_direction(acts_t, acts_b, w_dec, k):
    """Top-k contrastive features, decoder vectors weighted by the gap."""
    diff = acts_t.mean(0) - acts_b.mean(0)
    top = torch.topk(diff, k).indices
    return unit((diff[top, None] * w_dec[top]).sum(0)), top, diff


@torch.no_grad()
def cache_residuals(model, texts, layers, device):
    names = {f"blocks.{l}.hook_resid_post" for l in layers}
    out = torch.empty(len(texts), len(layers), model.cfg.d_model)
    for i, text in enumerate(texts):
        _, cache = model.run_with_cache(
            text, names_filter=lambda n: n in names)
        for j, l in enumerate(layers):
            out[i, j] = cache[f"blocks.{l}.hook_resid_post"][0, -1].float().cpu()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemma", choices=list(MODEL_CONFIGS))
    p.add_argument("--task", default="RTP")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="default: every layer of the model")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--dataset-size", type=int, default=500)
    p.add_argument("--n-null", type=int, default=20)
    p.add_argument("--n-split", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/layer_geometry"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    profile = MODEL_CONFIGS[args.model]
    model = HookedTransformer.from_pretrained(
        profile.model_name, device=device, dtype=torch.bfloat16).eval()
    layers = args.layers or list(range(model.cfg.n_layers))

    data = make_datasets([args.task], n=args.dataset_size, seed=args.seed)[args.task]
    toxic = [text_of(i) for i in data["train_toxic"]]
    benign = [text_of(i) for i in data["train_benign"]]
    print(f"{len(toxic)} trait / {len(benign)} benign prompts, "
          f"layers {layers[0]}..{layers[-1]}", flush=True)
    res_t = cache_residuals(model, toxic, layers, device)
    res_b = cache_residuals(model, benign, layers, device)
    del model
    torch.cuda.empty_cache()

    # Split-half partitions, fixed across layers so the ceiling at every
    # layer is measured on the same prompt halves.
    splits = []
    for _ in range(args.n_split):
        pt, pb = rng.permutation(len(toxic)), rng.permutation(len(benign))
        splits.append(tuple(torch.from_numpy(x) for x in (
            pt[: len(pt) // 2], pt[len(pt) // 2:],
            pb[: len(pb) // 2], pb[len(pb) // 2:])))

    sae_dir, dense_dir, null_dirs, top_vecs = [], [], [], []
    ceiling, dense_ceiling, per_layer = [], [], []
    for j, layer in enumerate(layers):
        loaded = SAE.from_pretrained(
            release=profile.sae_release,
            sae_id=profile.sae_id_format.format(layer), device=device)
        sae = (loaded[0] if isinstance(loaded, tuple) else loaded).eval()
        with torch.no_grad():
            a_t = sae.encode(res_t[:, j].to(device, sae.W_dec.dtype)).float()
            a_b = sae.encode(res_b[:, j].to(device, sae.W_dec.dtype)).float()
        w_dec = sae.W_dec.detach().float()

        v, top, diff = trait_direction(a_t, a_b, w_dec, args.k)
        sae_dir.append(v.cpu())
        top_vecs.append(torch.nn.functional.normalize(w_dec[top], dim=1).cpu())
        d = unit(res_t[:, j].mean(0) - res_b[:, j].mean(0))
        dense_dir.append(d)

        # Null: k random live features with the same contrastive weighting
        # rule, so only the choice of features differs.
        live = torch.nonzero((a_t > 0).any(0) | (a_b > 0).any(0)).squeeze(-1)
        nulls = []
        for _ in range(args.n_null):
            pick = live[torch.from_numpy(
                rng.choice(len(live), args.k, replace=False)).to(live.device)]
            nulls.append(unit((diff[pick, None] * w_dec[pick]).sum(0)).cpu())
        null_dirs.append(torch.stack(nulls))

        halves, dense_halves = [], []
        for t1, t2, b1, b2 in splits:
            h1, _, _ = trait_direction(a_t[t1], a_b[b1], w_dec, args.k)
            h2, _, _ = trait_direction(a_t[t2], a_b[b2], w_dec, args.k)
            halves.append(float(h1 @ h2))
            g1 = unit(res_t[t1, j].mean(0) - res_b[b1, j].mean(0))
            g2 = unit(res_t[t2, j].mean(0) - res_b[b2, j].mean(0))
            dense_halves.append(float(g1 @ g2))
        ceiling.append(float(np.mean(halves)))
        dense_ceiling.append(float(np.mean(dense_halves)))
        per_layer.append({
            "layer": layer,
            "cos_sae_dense": float(v.cpu() @ d),
            "top_diff_mean": float(diff[top].mean()),
            "ceiling": ceiling[-1], "dense_ceiling": dense_ceiling[-1],
        })
        print(f"L{layer}: split-half {ceiling[-1]:.3f} "
              f"dense {dense_ceiling[-1]:.3f} "
              f"cos(sae,dense) {per_layer[-1]['cos_sae_dense']:.3f}", flush=True)
        del sae, a_t, a_b, w_dec
        torch.cuda.empty_cache()

    S, D = torch.stack(sae_dir), torch.stack(dense_dir)
    sae_cos = (S @ S.T).numpy()
    dense_cos = (D @ D.T).numpy()
    # Null cross-layer cosine: random directions at layer l vs layer m.
    N = torch.stack(null_dirs)                       # [L, n_null, d]
    null_cos = torch.einsum("lnd,mnd->lmn", N, N).mean(-1).numpy()

    # Feature-level view: every top-k pair between consecutive layers,
    # which is exactly the population the method's cosine filter sees.
    pairs = []
    for j in range(len(layers) - 1):
        c = (top_vecs[j] @ top_vecs[j + 1].T).flatten().numpy()
        pairs.append({"layer": layers[j], "next": layers[j + 1],
                      "mean": float(c.mean()),
                      "frac_above_0.10": float((c > 0.10).mean()),
                      "frac_below_-0.10": float((c < -0.10).mean()),
                      "cosines": c.round(4).tolist()})

    L = len(layers)
    by_gap = []
    for g in range(1, L):
        idx = [(i, i + g) for i in range(L - g)]
        by_gap.append({
            "gap": g,
            "sae": float(np.mean([sae_cos[i, m] for i, m in idx])),
            "dense": float(np.mean([dense_cos[i, m] for i, m in idx])),
            "null": float(np.mean([null_cos[i, m] for i, m in idx])),
        })

    # Interference of the naive all-layer vector (unit per layer, so this
    # is scale-free): 1 when every layer pushes the same way.
    eff = float(S.sum(0).norm() / len(layers))
    result = {
        "model": args.model, "task": args.task, "k": args.k,
        "layers": layers, "per_layer": per_layer, "by_gap": by_gap,
        "sae_cos": sae_cos.round(4).tolist(),
        "dense_cos": dense_cos.round(4).tolist(),
        "null_cos": null_cos.round(4).tolist(),
        "adjacent_pairs": pairs,
        "naive_efficiency": eff,
        "dense_efficiency": float(D.sum(0).norm() / len(layers)),
        # The unit directions themselves, for plotting them directly.
        "sae_vectors": S.numpy().round(5).tolist(),
        "dense_vectors": D.numpy().round(5).tolist(),
    }
    path = args.out / f"layer_geometry_{args.model}_{args.task}.json"
    path.write_text(json.dumps(result))
    print(f"efficiency naive-SAE {eff:.3f} dense "
          f"{result['dense_efficiency']:.3f}", flush=True)
    for row in by_gap[:6]:
        print(f"gap {row['gap']}: sae {row['sae']:.3f} "
              f"dense {row['dense']:.3f} null {row['null']:.3f}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
