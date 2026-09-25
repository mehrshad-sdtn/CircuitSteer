"""Motivation figure: cross-layer geometry of a trait's SAE features.

(a) cosine between every pair of layers' SAE trait directions;
(b) cosine against layer gap, with the dense difference-of-means
    direction, the random-feature null and the split-half ceiling;
(c) cosines of every top-k trait-feature pair between adjacent layers,
    against random decoder pairs, with the method's sim_thresh marked.

Reads the JSON written by experiments/layer_geometry.py and the random
decoder-pair null.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.palette import (
    ALIGNED, DPI, GRID, INK, MUTED, NOGEO, PAGE_WIDTH, RANDOM, SIZE_ANNOT,
    use_paper_style,
)

SIM_THRESH = 0.10
METHOD_LAYERS = (6, 12, 18, 24)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path,
                   default=Path("results/layer_geometry/layer_geometry_gemma_RTP.json"))
    p.add_argument("--null", type=Path,
                   default=Path("results/layer_geometry/pair_null_gemma.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/fig_layer_geometry"))
    args = p.parse_args()
    r = json.loads(args.geometry.read_text())
    null = np.asarray(json.loads(args.null.read_text())["cosines"])
    use_paper_style()

    fig, axes = plt.subplots(
        1, 3, figsize=(PAGE_WIDTH, 2.35),
        gridspec_kw={"width_ratios": [1.05, 1, 1], "wspace": 0.55})

    # (a) layer x layer heatmap
    ax = axes[0]
    S = np.asarray(r["sae_cos"])
    im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1, origin="upper",
                   aspect="auto")
    ticks = [0, 6, 12, 18, 24]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    for l in METHOD_LAYERS:
        for m in METHOD_LAYERS:
            if l < m:
                ax.plot(m, l, "o", ms=3.2, mfc="none", mec=NOGEO, mew=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.set_title("cos", fontsize=SIZE_ANNOT, pad=3)
    cb.outline.set_visible(False)
    ax.set_title("(a) SAE trait direction", loc="left")

    # (b) cosine vs layer gap
    ax = axes[1]
    g = np.array([row["gap"] for row in r["by_gap"]])
    ceiling = np.mean([row["ceiling"] for row in r["per_layer"]])
    ax.axhspan(ceiling - 0.005, ceiling + 0.005, color=GRID, lw=0)
    ax.axhline(ceiling, color=MUTED, lw=0.8, ls=(0, (3, 2)),
               label="split-half ceiling")
    ax.plot(g, [row["dense"] for row in r["by_gap"]], color=INK, lw=1.3,
            label="dense (no SAE)")
    ax.plot(g, [row["sae"] for row in r["by_gap"]], color=ALIGNED, lw=1.8,
            label="SAE top-$k$")
    ax.plot(g, [row["null"] for row in r["by_gap"]], color=RANDOM, lw=1.2,
            label="random features")
    ax.axvline(6, color=NOGEO, lw=0.8, ls=":")
    ax.text(6.5, 0.83, "method\nspacing", color=NOGEO,
            fontsize=SIZE_ANNOT, va="bottom")
    ax.set_xlim(1, g[-1])
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("Layer gap")
    ax.set_ylabel("Cosine")
    # Direct labels rather than a legend, which would sit on the lines.
    for x, y, text, colour, va in (
        (g[-1], ceiling + 0.02, "split-half ceiling", MUTED, "bottom"),
        (12, 0.63, "dense (no SAE)", INK, "bottom"),
        (7.5, 0.30, "SAE top-$k$", ALIGNED, "top"),
        (g[-1], 0.035, "random features", RANDOM, "bottom"),
    ):
        ax.text(x, y, text, color=colour, fontsize=SIZE_ANNOT, va=va,
                ha="right" if x == g[-1] else "left")
    ax.set_title("(b) Drift with depth", loc="left")

    # (c) adjacent-layer feature-pair cosines
    ax = axes[2]
    trait = np.concatenate([p_["cosines"] for p_ in r["adjacent_pairs"]])
    bins = np.linspace(-0.2, 0.6, 81)
    ax.hist(null, bins=bins, density=True, color=RANDOM, alpha=0.55,
            label="random pairs", lw=0)
    ax.hist(trait, bins=bins, density=True, histtype="step", color=ALIGNED,
            lw=1.4, label="trait pairs")
    ax.set_yscale("log")
    ax.axvline(SIM_THRESH, color=NOGEO, lw=1.0, ls="--")
    f_t, f_n = (trait > SIM_THRESH).mean(), (null > SIM_THRESH).mean()
    ax.text(SIM_THRESH + 0.02, 0.97,
            f"> {SIM_THRESH:.2f}:\ntrait {100 * f_t:.1f}%\nrandom {100 * f_n:.1f}%",
            transform=ax.get_xaxis_transform(), va="top",
            fontsize=SIZE_ANNOT, color=INK)
    ax.set_xlabel("Cosine, layer $l$ vs $l{+}1$")
    ax.set_ylabel("Density (log)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(c) Feature pairs", loc="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=DPI,
                    bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png  trait>{SIM_THRESH}={f_t:.3f} "
          f"null>{SIM_THRESH}={f_n:.4f}")


if __name__ == "__main__":
    main()
