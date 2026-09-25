"""Figure 1: steering strength against fluency cost, one point per method.

Each point is the mean over the eight Table 1 cells (2 models x 4 tasks)
of the per-cell Delta and median normalised perplexity, read from
results/paper_table.csv so the figure cannot drift from the table. Bars
are the standard error across the eight cells. Vacuous cells (no
measurable effect at any coefficient, dashes in the table) count as
Delta = 0 and are left out of the perplexity mean.

Colours come from the shared palette: the method's blue, orange for its
closest competitor (ITI), and three greys for the remaining baselines
(dark: SAE-based, mid: other steering baselines, light: prompting).
Groups also differ in marker shape; identity is carried by direct
labels. The light red band is outside the perplexity window.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from palette import (ALIGNED, DARK_GREY, DPI, GRID, INK, INK2,
                     LIGHT_GREY, MID_GREY, MUTED, ORANGE, WINDOW_FILL,
                     SIZE_ANNOT, SIZE_LABEL, SIZE_LEGEND,
                     SIZE_TITLE, use_paper_style)

SOURCE = Path("results/paper_table.csv")
OUT = Path("results/figures/fig1_pareto")

NAMES = {
    "Prompt": "Prompt", "CAA-1L": "CAA", "CAA-top3": "CAA (ML)",
    "RepE": "RepE", "ITI": "ITI", "LoReFT": "LoReFT", "SpARE": "SpARE",
    "SAE-SSV": "SAE-SSV", "FeatureFlow": "FeatureFlow",
    "CircuitSteer": "CircuitSteer",
}
#: method -> (colour, marker, size)
GROUP = {
    "CircuitSteer": (ALIGNED, "*", 150), "ITI": (ORANGE, "D", 30),
    "SpARE": (DARK_GREY, "s", 32), "SAE-SSV": (DARK_GREY, "s", 32),
    "FeatureFlow": (DARK_GREY, "s", 32),
    "CAA-1L": (MID_GREY, "^", 40), "CAA-top3": (MID_GREY, "^", 40),
    "RepE": (MID_GREY, "^", 40), "LoReFT": (MID_GREY, "^", 40),
    "Prompt": (LIGHT_GREY, "o", 36),
}

#: Upper bound of the perplexity window; right of it is out of window.
WINDOW_HI = 2.0
X_MIN, X_MAX = 0.75, 2.3
Y_MIN, Y_MAX = -0.012, 0.125

#: Direct-label offsets in points, set against the rendered figure.
OFFSETS = {
    "Prompt": (0, -11), "CAA-1L": (-6, 5), "CAA-top3": (-5, 6),
    "RepE": (6, -9), "ITI": (6, 4), "LoReFT": (6, 3), "SpARE": (6, -9),
    "SAE-SSV": (-4, 9), "FeatureFlow": (-6, 5), "CircuitSteer": (-7, 7),
}


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    return (t.groupby("method", sort=False)
            .agg(delta=("delta_mean", "mean"), delta_se=("delta_mean", "sem"),
                 ppl=("ppl_median", "mean"), ppl_se=("ppl_median", "sem")))


def frontier(s: pd.DataFrame) -> pd.DataFrame:
    """Methods no other method beats on both axes (lower ppl, higher Delta)."""
    keep = [m for m, r in s.iterrows()
            if not ((s.ppl <= r.ppl) & (s.delta >= r.delta)
                    & ((s.ppl < r.ppl) | (s.delta > r.delta))).any()]
    return s.loc[keep].sort_values("ppl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--width", type=float, default=3.5)
    #: Same aspect as the draft's fig_overview.pdf (564.35 x 446.34 pt), so
    #: the wrapfigure block renders at the same height.
    ap.add_argument("--aspect", type=float, default=564.35 / 446.34)
    args = ap.parse_args()

    use_paper_style()
    t = pd.read_csv(args.source)
    t.loc[t.vacuous, "delta_mean"] = 0.0
    t.loc[t.vacuous, "ppl_median"] = np.nan
    s = summarise(t)
    print(s.round(3).to_string())
    front = frontier(s)

    fig, ax = plt.subplots(figsize=(args.width, args.width / args.aspect))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    ax.axvspan(WINDOW_HI, X_MAX, color=WINDOW_FILL, zorder=0, linewidth=0)
    ax.text(WINDOW_HI + 0.03, Y_MAX - 0.003, "outside\nPPL window",
            fontsize=SIZE_ANNOT - 1, color=MUTED, va="top")
    ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)),
               zorder=1)
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)

    ax.plot(front.ppl, front.delta, color=INK2, linewidth=1.0,
            linestyle=(0, (1, 1.5)), zorder=2)

    for method, r in s.iterrows():
        colour, marker, size = GROUP[method]
        ours = method == "CircuitSteer"
        clipped = r.ppl > X_MAX
        x = X_MAX - 0.02 if clipped else r.ppl
        ax.errorbar(x, r.delta, yerr=r.delta_se,
                    xerr=None if clipped else r.ppl_se, fmt="none",
                    ecolor=colour, elinewidth=0.8, alpha=0.5, capsize=0,
                    zorder=3)
        ax.scatter(x, r.delta, s=size, marker=">" if clipped else marker,
                   color=colour, edgecolor="white", linewidth=0.8,
                   zorder=5 if ours else 4)
        label = NAMES[method]
        if clipped:
            label += f"\n(PPL {r.ppl:.1f})"
        dx, dy = OFFSETS[method]
        ax.annotate(label, (x, r.delta), textcoords="offset points",
                    xytext=(dx, dy),
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
                    fontsize=SIZE_ANNOT + (0.5 if ours else -0.5),
                    fontweight="bold" if ours else "normal",
                    color=INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.05",
                              fc="none" if clipped else "white",
                              ec="none", alpha=0.9))

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
    ax.set_xticklabels(["", "1.0", "", "1.5", "", "2.0", ""])
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"Normalized perplexity ($\downarrow$ better)")
    ax.set_ylabel(r"Behavior reduction $\Delta$ ($\uparrow$ better)")

    # CircuitSteer and ITI are labelled on the plot; the legend only
    # explains the grey groups, in one row under the axes so it never
    # overlaps data and the axes keep the figure's full width.
    handles = [
        Line2D([], [], marker="s", color=DARK_GREY, linestyle="none",
               markersize=4, label="SAE-based"),
        Line2D([], [], marker="^", color=MID_GREY, linestyle="none",
               markersize=4.5, label="Other steering"),
        Line2D([], [], marker="o", color=LIGHT_GREY, linestyle="none",
               markersize=4, label="Prompting"),
        Line2D([], [], color=INK2, linestyle=(0, (1, 1.5)),
               linewidth=1.0, label="Pareto frontier"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), fontsize=SIZE_LEGEND - 1.5,
               frameon=False, handletextpad=0.25, columnspacing=0.7,
               handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), pad=0.2)
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.025,
             "Steering strength vs. fluency", ha="center",
             fontsize=SIZE_TITLE, color=INK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=DPI)
    print(f"wrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
