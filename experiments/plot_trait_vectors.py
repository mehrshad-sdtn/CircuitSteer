"""The per-layer trait direction itself, one row per layer.

Each row is a layer's unit SAE trait direction (experiments/layer_geometry.py)
restricted to the residual dimensions that matter most. Columns are
chosen and ordered by a rule that favours no layer: the `n_dims` largest
by maximum |component| across all layers, sorted by the layer-averaged
direction. Sorting by one layer's own values would make that row look
structured and every other row look like noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from experiments.palette import (
    ALIGNED, DPI, INK, NOGEO, PAGE_WIDTH, SIZE_ANNOT, use_paper_style,
)

METHOD_LAYERS = (6, 12, 18, 24)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path,
                   default=Path("results/layer_geometry/layer_geometry_gemma_RTP.json"))
    p.add_argument("--n-dims", type=int, default=64)
    p.add_argument("--kind", choices=("sae", "dense"), default="sae")
    p.add_argument("--width", type=float, default=PAGE_WIDTH)
    p.add_argument("--height", type=float, default=2.9)
    p.add_argument("--no-cos", action="store_true",
                   help="omit the per-row cosine column (narrow layouts)")
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/fig_trait_vectors"))
    args = p.parse_args()
    r = json.loads(args.geometry.read_text())
    V = np.asarray(r[f"{args.kind}_vectors"])          # [layers, d_model]
    layers = r["layers"]
    use_paper_style()

    dims = np.argsort(-np.abs(V).max(0))[: args.n_dims]
    dims = dims[np.argsort(-V[:, dims].mean(0))]
    M = V[:, dims]
    # A few large components would otherwise wash every other cell out;
    # values beyond the cap saturate rather than setting the scale.
    lim = float(np.quantile(np.abs(M), 0.98))
    cmap = LinearSegmentedColormap.from_list(
        "div", [NOGEO, "#ffffff", ALIGNED])

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) if l % 6 == 0 else ""
                        for l in layers])
    for tick, l in zip(ax.get_yticklabels(), layers):
        if l in METHOD_LAYERS:
            tick.set_color(NOGEO)
            tick.set_fontweight("bold")
    ax.set_xticks([])
    ax.set_xlabel(f"Residual dimension (top {args.n_dims} of {V.shape[1]})")
    ax.set_ylabel("Layer")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.07, extend="both")
    cb.set_label("component", fontsize=SIZE_ANNOT)
    cb.outline.set_visible(False)

    # Cosine to the first method layer, alongside each row, so the visual
    # impression can be checked against the number.
    if not args.no_cos:
        ref = layers.index(METHOD_LAYERS[0])
        cos = V @ V[ref]
        for i, c in enumerate(cos):
            ax.text(M.shape[1] + 0.8, i, f"{c:.2f}", va="center", ha="left",
                    fontsize=SIZE_ANNOT - 1.5, color=INK, clip_on=False)
        ax.text(M.shape[1] + 0.8, -0.9, f"cos(L, L{METHOD_LAYERS[0]})",
                ha="left", va="bottom", fontsize=SIZE_ANNOT - 1, color=INK,
                clip_on=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=DPI,
                    bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
