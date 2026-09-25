"""What each method does as the steering coefficient is pushed.

The main table reports every method at one tuned coefficient, which hides
what a practitioner most wants to know: how much room there is either
side of that point. This draws each method's own curve over a coefficient
range two orders of magnitude wide.

Perplexity and behaviour change are separate measures, so they get
separate rows - never two y-scales on one panel - and the models get
separate columns, because a coefficient does not mean the same thing to
a 2B and an 8B model.

The magnitude of lambda is NOT comparable across methods either: a unit
of CAA is not a unit of ITI, since each carries its own vector norm. The
readable quantity is the SHAPE of a curve, and how wide a range of
coefficients a method keeps inside the usable perplexity band.

All points come from one machine and one software stack. The same seed
and coefficient measured on a different GPU read +0.108 where this one
reads +0.235, so a curve spliced from two environments would show a
hardware artifact as if it were a property of the method.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from experiments.palette import (
    DPI, GRID, INK, MUTED, PAGE_WIDTH, SIZE_LABEL, SIZE_LEGEND, SIZE_TITLE,
    SURFACE, use_paper_style,
)

SOURCE = Path("results/lambda_sweep/sweep.csv")
OUT = Path("results/figures")

#: Only the method under test carries a hue. Five saturated lines
#: crossing each other on a log axis is a colour puzzle, not a figure:
#: the reader has to hold five hue-name bindings while tracing curves
#: that overlap. The baselines instead separate by dash pattern and by
#: lightness, which leaves exactly one thing on the page that is
#: coloured, and it is the comparison the figure exists to make.
METHODS = {
    "circuitsteer": ("CircuitSteer", "#2B6CB0", 2.6, "-"),
    "iti": ("ITI", "#3d3d3d", 1.5, (0, (5, 2))),
    "caa_top3": ("CAA (multi-layer)", "#6b6b6b", 1.5, (0, (1.5, 1.5))),
    "repe": ("RepE", "#8f8f8f", 1.5, (0, (6, 1.5, 1.5, 1.5))),
    "spare": ("SpARE", "#b3b3b3", 1.5, (0, (3, 1, 1, 1, 1, 1))),
}

#: The sweep also measured -32, which is dropped here: every method is
#: far outside the window by -16, so the last point only stretched the
#: axis and compressed the region where the curves differ.
COEFFS = [-16.0, -8.0, -4.0, -2.0, -1.0]
MODELS = [("gemma", "Gemma-2-2B"), ("llama", "Llama-3.1-8B")]
BAND = "#e8efe9"
PPL_LO, PPL_HI = 0.25, 2.0


def load() -> pd.DataFrame:
    table = pd.read_csv(SOURCE)
    table = table[table["split"] == "test"]
    # Averaged over the two datasets: the question is the method's
    # overall behaviour under pressure, not one dataset's.
    table = table[table["coeff"].isin(COEFFS)]
    return (
        table.groupby(["method", "model", "coeff"])[
            ["delta", "norm_ppl_median"]
        ].mean().reset_index()
    )


def _axis(ax, log, model_label, top_row):
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.4, 5e4)
        ax.axhspan(PPL_LO, PPL_HI, color=BAND, zorder=0, lw=0)
        if top_row:
            ax.set_title(model_label, fontsize=SIZE_TITLE, color=INK,
                         pad=6)
    else:
        ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gemma", "llama"])
    parser.add_argument("--layout", choices=("vertical", "horizontal"),
                        default="vertical",
                        help="With one model: vertical stacks perplexity "
                             "over behaviour, horizontal puts them side "
                             "by side.")
    parser.add_argument("--out", type=Path,
                        default=OUT / "fig_lambda_robustness.png")
    args = parser.parse_args()

    use_paper_style()
    data = load()
    models = [m for m in MODELS if m[0] in args.models]
    # One model: the two measures are the only grid dimension, so the
    # layout flag chooses which way they are stacked. Two models: they
    # are always columns and the measures are always rows.
    single = len(models) == 1
    if single and args.layout == "horizontal":
        shape, size = (1, 2), (PAGE_WIDTH, 2.6)
    elif single:
        shape, size = (2, 1), (PAGE_WIDTH * 0.56, 4.6)
    else:
        shape, size = (2, 2), (PAGE_WIDTH, 5.0)
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False,
                             sharex=not (single and args.layout ==
                                         "horizontal"))
    fig.patch.set_facecolor(SURFACE)

    for col, (key, label) in enumerate(models):
        for index, (column, log) in enumerate(
            (("norm_ppl_median", True), ("delta", False))
        ):
            if single and args.layout == "horizontal":
                ax = axes[0][index]
            elif single:
                ax = axes[index][0]
            else:
                ax = axes[index][col]
            # With one model side by side, the name goes above
            # both panels as a suptitle instead of over one.
            _axis(ax, log, label,
                  index == 0 and not (single and
                  args.layout == "horizontal"))
            for method, (_, colour, width, dash) in METHODS.items():
                sub = data[
                    (data["method"] == method) & (data["model"] == key)
                ].sort_values("coeff", ascending=False)
                if sub.empty:
                    continue
                ours = method == "circuitsteer"
                ax.plot([abs(c) for c in sub["coeff"]], sub[column],
                        color=colour, linewidth=width, linestyle=dash,
                        marker="o" if ours else None,
                        markersize=4.5, markeredgecolor="white",
                        markeredgewidth=0.8,
                        zorder=6 if ours else 3)
            if col == 0 or single:
                ax.set_ylabel(
                    "perplexity ratio" if log else "behaviour reduction",
                    fontsize=SIZE_LABEL, color=MUTED)

    handles = [Line2D([], [], color=c, linewidth=w, linestyle=d,
                      marker="o" if n == "CircuitSteer" else None,
                      markersize=4.5, markeredgecolor="white", label=n)
               for n, c, w, d in METHODS.values()]
    # The x label goes on the panels in the bottom row - both of them
    # when one model is laid out side by side.
    for row in ([axes[0]] if (single and args.layout == "horizontal")
                else [axes[-1]]):
        for ax in row:
            if ax.get_visible():
                ax.set_xlabel("steering strength  |\u03bb|",
                              fontsize=SIZE_LABEL, color=MUTED)

    wide = single and args.layout == "horizontal"
    if wide:
        fig.suptitle(models[0][1], fontsize=SIZE_TITLE, color=INK, y=1.06)
    fig.legend(handles=handles, loc="lower center",
               ncol=3 if single and not wide else 5, frameon=False,
               fontsize=SIZE_LEGEND, labelcolor=MUTED,
               bbox_to_anchor=(0.5, -0.20 if wide else -0.08))
    fig.subplots_adjust(top=0.93, bottom=0.18, left=0.135, right=0.985,
                        hspace=0.22, wspace=0.28)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=DPI,
                    facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
