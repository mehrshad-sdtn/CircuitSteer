"""The coefficient sweep as a table, for the appendix.

Everything measured, not a selection: all five methods, both models,
both datasets, every coefficient. The main-text figure shows one model
because that is where the effect is clean; this is the full record
behind it, so a reader can see what was left out and why.

Cells read "delta (ppl)". A cell whose perplexity ratio falls outside
the [0.25, 2.0] window is marked, because a large delta there is bought
with fluency and is not a usable operating point.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

SOURCE = Path("results/lambda_sweep/sweep.csv")
OUT = Path("results")

METHODS = [("circuitsteer", r"\textsc{CircuitSteer}"),
           ("iti", r"\textsc{ITI}"),
           ("caa_top3", r"\textsc{CAA} (multi-layer)"),
           ("repe", r"\textsc{RepE}"),
           ("spare", r"\textsc{SpARE}")]
MODELS = [("gemma", "Gemma-2-2B"), ("llama", "Llama-3.1-8B-Instruct")]
DATASETS = ["RTP", "Emotion"]
COEFFS = [-1.0, -2.0, -4.0, -8.0, -16.0]
PPL_LO, PPL_HI = 0.25, 2.0

CAPTION = r"""\caption{
    Coefficient sweep, all methods and cells. Each entry is behavioural
    reduction $\Delta$ with the median perplexity ratio $\overline{PPL}$
    beneath it, at a single seed. \colorbox{red!15}{Shaded} entries fall
    outside the $[0.25, 2.0]$ fluency window: the reduction there is
    bought with fluency and is not a usable operating point.
}"""


def load() -> pd.DataFrame:
    table = pd.read_csv(SOURCE)
    table = table[(table["split"] == "test") & table["coeff"].isin(COEFFS)]
    return table.groupby(["method", "model", "dataset", "coeff"])[
        ["delta", "norm_ppl_median"]
    ].mean().reset_index()


def _ppl(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def latex(data: pd.DataFrame) -> str:
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\renewcommand{\arraystretch}{1.25}",
           r"\setlength{\tabcolsep}{5pt}", CAPTION,
           r"\label{tab:lambda_sweep}", r"\vspace{1mm}",
           r"\begin{tabular}{ll" + "r" * len(COEFFS) + "}", r"\toprule",
           "Method & Data & " + " & ".join(
               rf"$\lambda{{=}}{c:g}$" for c in COEFFS) + r" \\"]
    for model_key, model_label in MODELS:
        out += [r"\midrule",
                rf"\multicolumn{{{2 + len(COEFFS)}}}{{l}}"
                rf"{{\emph{{{model_label}}}}} \\"]
        for method_key, method_label in METHODS:
            for index, dataset in enumerate(DATASETS):
                cells = []
                for coeff in COEFFS:
                    row = data[
                        (data.method == method_key)
                        & (data.model == model_key)
                        & (data.dataset == dataset)
                        & (data.coeff == coeff)
                    ]
                    if row.empty:
                        cells.append(r"\text{---}")
                        continue
                    delta = float(row.iloc[0]["delta"])
                    ppl = float(row.iloc[0]["norm_ppl_median"])
                    shade = ("" if PPL_LO <= ppl <= PPL_HI
                             else r"\cellcolor{red!15}")
                    cells.append(
                        rf"{shade}$\substack{{{delta:+.3f}\\"
                        rf"{{\scriptsize {_ppl(ppl)}}}}}$")
                name = method_label if index == 0 else ""
                out.append(f"{name} & \\textit{{{dataset}}} & "
                           + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def tidy(data: pd.DataFrame) -> Path:
    path = OUT / "lambda_sweep_table.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "model", "dataset", "lambda",
                         "delta", "ppl_median", "in_window"])
        for method_key, label in METHODS:
            for model_key, _ in MODELS:
                for dataset in DATASETS:
                    for coeff in COEFFS:
                        row = data[
                            (data.method == method_key)
                            & (data.model == model_key)
                            & (data.dataset == dataset)
                            & (data.coeff == coeff)]
                        if row.empty:
                            continue
                        delta = float(row.iloc[0]["delta"])
                        ppl = float(row.iloc[0]["norm_ppl_median"])
                        writer.writerow([
                            label.replace(r"\textsc{", "").replace("}", ""),
                            model_key, dataset, f"{coeff:g}",
                            f"{delta:.4f}", f"{ppl:.4f}",
                            PPL_LO <= ppl <= PPL_HI])
    return path


if __name__ == "__main__":
    data = load()
    (OUT / "lambda_sweep_table.tex").write_text(latex(data) + "\n")
    path = tidy(data)
    print(f"wrote {OUT / 'lambda_sweep_table.tex'} and {path}")
