"""The per-layer trait direction as a path through depth.

The unit SAE trait directions of every layer (experiments/layer_geometry.py)
are projected onto their first two principal components and joined in
layer order. PCA is centred, so the axes show how the direction moves
between layers rather than the component all layers share; the variance
the two components keep is printed on the axes and must be reported
with the figure, since the plane drops the rest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from experiments.palette import (
    DPI, INK, MUTED, NOGEO, PAGE_WIDTH, SIZE_ANNOT, use_paper_style,
)

METHOD_LAYERS = (6, 12, 18, 24)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path,
                   default=Path("results/layer_geometry/layer_geometry_gemma_RTP.json"))
    p.add_argument("--kind", choices=("sae", "dense"), default="sae")
    p.add_argument("--width", type=float, default=PAGE_WIDTH / 2)
    p.add_argument("--height", type=float, default=None,
                   help="fix the height; the plane is then not drawn to scale")
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/fig_trait_trajectory"))
    args = p.parse_args()
    r = json.loads(args.geometry.read_text())
    V = np.asarray(r[f"{args.kind}_vectors"])
    layers = np.asarray(r["layers"])
    use_paper_style()

    centred = V - V.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    xy = centred @ vt[:2].T
    # Orient so the path starts on the left, for a left-to-right read.
    if xy[0, 0] > xy[-1, 0]:
        xy[:, 0] *= -1

    fig, ax = plt.subplots(
        figsize=(args.width, args.height or args.width * 0.82))
    cmap = plt.get_cmap("Blues")
    shade = 0.3 + 0.7 * (layers - layers.min()) / np.ptp(layers)
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    ax.add_collection(LineCollection(
        segs, colors=cmap(shade[1:]), linewidths=1.6, zorder=1))
    ax.scatter(xy[:, 0], xy[:, 1], c=cmap(shade), s=14, zorder=2,
               edgecolors="white", linewidths=0.4)

    for i, l in enumerate(layers):
        if l in METHOD_LAYERS:
            ax.scatter(*xy[i], s=62, facecolors="none", edgecolors=NOGEO,
                       linewidths=1.2, zorder=3)
            ax.annotate(f"L{l}", xy[i], xytext=(5, 5),
                        textcoords="offset points", color=NOGEO,
                        fontsize=SIZE_ANNOT, fontweight="bold")
        elif l in (layers.min(), layers.max()):
            # The last layer sits next to the last method layer; label it
            # on the far side so the two do not collide.
            offset = (-16, 3) if l == layers.max() else (5, -9)
            ax.annotate(f"L{l}", xy[i], xytext=offset,
                        textcoords="offset points", color=MUTED,
                        fontsize=SIZE_ANNOT)

    i6, i24 = (list(layers).index(l) for l in (6, 24))
    cos = float(V[i6] @ V[i24])
    ax.text(0.98, 0.97, f"cos(L6, L24) = {cos:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=SIZE_ANNOT, color=INK)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.08)
    if args.height is None:
        ax.set_aspect("equal", adjustable="datalim")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=DPI,
                    bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png  PC1 {var[0]:.3f} PC2 {var[1]:.3f} "
          f"sum {var[:2].sum():.3f}")


if __name__ == "__main__":
    main()
