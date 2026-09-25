"""Angle sweep as polar geometry.

The swept variable is an angle, so the figure is drawn as one: angular
position IS the true angle between consecutive layers' steering vectors
(0 = parallel, 180 = opposed), and distance from the centre is the
behavioural reduction it produces. The dashed ring is no effect; inside
it steering makes the behaviour worse.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.palette import (  # noqa: E402
    DPI, GEMMA, GRID, HARM, INK, INK2, LLAMA, MUTED, PAGE_WIDTH,
    SIZE_LABEL, SIZE_LEGEND, SIZE_TITLE, SURFACE, display_name,
    use_paper_style,
)


def panel(ax, d, dataset):
    sub = d[d.dataset == dataset]
    lo = float(min(sub.delta.min(), 0.0))
    hi = float(sub.delta.max())
    span = hi - lo
    rmin, rmax = lo - 0.22 * span, hi + 0.16 * span

    ax.set_facecolor(SURFACE)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(rmin, rmax)
    ax.set_rorigin(rmin)

    # "worse than no steering" lives inside the zero ring
    theta_fill = np.linspace(0, np.pi, 200)
    ax.fill_between(theta_fill, rmin, 0.0, color=HARM, zorder=0, linewidth=0)
    ax.plot(theta_fill, np.zeros_like(theta_fill), color=MUTED, lw=1.1,
            ls=(0, (4, 3)), zorder=2)

    for model, colour in (("gemma", GEMMA), ("llama", LLAMA)):
        s = sub[sub.model == model].sort_values("cosine")
        if s.empty:
            continue
        theta = np.arccos(np.clip(s.cosine.to_numpy(), -1, 1))
        r = s.delta.to_numpy()
        ax.plot(theta, r, "-", color=colour, lw=2.6, zorder=4,
                solid_capstyle="round")
        ax.scatter(theta, r, s=30, color=colour, edgecolor=SURFACE,
                   linewidth=1.4, zorder=5)
        best = int(np.argmax(r))
        ax.scatter([theta[best]], [r[best]], s=190, facecolor="none",
                   edgecolor=colour, linewidth=2.0, zorder=6)

    ax.set_thetagrids(
        [0, 45, 90, 135, 180],
        labels=["parallel", "", "orthogonal", "", "opposed"],
        fontsize=SIZE_LABEL, color=INK2)
    ax.tick_params(axis="x", pad=13)
    ax.set_rgrids([0.0], labels=[""], color=MUTED)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_title(dataset, fontsize=SIZE_TITLE, color=INK,
                 pad=4, y=0.98)


def main() -> None:
    use_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/fig_angle_polar.png"))
    ap.add_argument("--datasets", nargs="+",
                    default=["RTP", "Sycophancy"],
                    help="Which datasets to draw, one panel each.")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)

    # Half-disc axes are square, so the unused lower half is cropped by
    # a tight bbox; generous wspace keeps "parallel" on the left panel from
    # colliding with "opposed" on the right.
    panels = list(a.datasets)
    # Half-disc axes are wide and short, so more than two across a
    # text-width figure leaves no room for the angular tick labels and
    # "opposed"/"parallel" run into the neighbouring panel.
    ncols = min(2, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(PAGE_WIDTH, 2.6 * nrows),
                             subplot_kw=dict(projection="polar"),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat = [ax for row in axes for ax in row]
    for extra in flat[len(panels):]:
        extra.set_visible(False)
    for ax, ds in zip(flat, panels):
        panel(ax, d, ds)

    handles = [plt.Line2D([], [], color=GEMMA, lw=2.4, label="Gemma-2-2B"),
               plt.Line2D([], [], color=LLAMA, lw=2.4, label="Llama-3.1-8B"),
               plt.Line2D([], [], color=MUTED, lw=1.3, ls=(0, (4, 3)),
                          label="No effect")]
    fig.legend(handles=handles, fontsize=SIZE_LEGEND, frameon=False,
               ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               labelcolor=INK2, handlelength=2.4, columnspacing=2.6)

    fig.suptitle("Behavioural steering is strongest between the extremes",
                 fontsize=SIZE_TITLE + 1, color=INK, y=1.20)
    fig.text(0.5, 1.10,
             "Angle between consecutive layers’ steering vectors  "
             "·  radius = behaviour reduction",
             fontsize=SIZE_LABEL, color=MUTED, ha="center")
    fig.subplots_adjust(top=0.86, bottom=0.06, wspace=0.62)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=DPI, facecolor=SURFACE,
                bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".pdf"), facecolor=SURFACE,
                bbox_inches="tight")
    print(f"wrote {a.out} ({DPI} dpi) and {a.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
