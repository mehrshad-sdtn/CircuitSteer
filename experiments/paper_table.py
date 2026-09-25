"""Main baseline table for the paper.

One row per method, one cell per (model, dataset): the behaviour-change
delta averaged over the three seeds with its standard deviation, and the
median perplexity ratio for the same entry. Every entry uses a single
coefficient shared by all three seeds, chosen on validation only.

Reads the aggregate written by make_main_table.py and emits the table as
markdown (terminal), LaTeX (paper) and tidy CSV.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

SOURCE = Path("results/main_table_v2.csv")
OUT = Path("results")

#: Baselines in rough order of lineage, ours last.
ORDER = ["Prompt", "CAA-1L", "CAA-top3", "RepE", "ITI", "SpARE",
         "SAE-SSV", "FeatureFlow", "LoReFT", "CircuitSteer"]
MODELS = [("gemma", "Gemma-2-2B"), ("llama", "Llama-3.1-8B")]
DATASETS = ["RTP", "Jigsaw", "Emotion", "Sycophancy"]

CAPTION = r"""\caption{
    Behavioral reduction ($\Delta\uparrow$) and normalized perplexity
    ($\overline{PPL}$, median steered-to-base ratio), mean over three
    seeds. One coefficient per entry, selected on validation $\Delta$
    subject to $\overline{PPL} \in [0.25, 2.0]$.
    \colorbox{green!18}{\textbf{Best}} and
    \colorbox{cyan!14}{\uline{second best}} per column;
    \colorbox{red!15}{shaded} entries fall outside the window and are
    unranked. Dashes~(---): not applicable, no measurable effect at
    any coefficient.
}"""




ROWS = [
    ("Prompt", r"\textsc{Prompt}"),
    ("CAA-1L", r"\textsc{CAA} (single-layer)"),
    ("CAA-top3", r"\textsc{CAA} (multi-layer)"),
    ("RepE", r"\textsc{RepE}"),
    ("ITI", r"\textsc{ITI}"),
    ("LoReFT", r"\textsc{LoReFT}"),
    None,                                   # rule: SAE-based methods below
    ("SpARE", r"\textsc{SpARE}"),
    ("SAE-SSV", r"\textsc{SAE-SSV}"),
    ("FeatureFlow", r"\textsc{FeatureFlow}"),
    None,
    ("CircuitSteer", r"\textbf{\textsc{CircuitSteer}}"),
]

DASH = r"\text{---}"
#: Two orthogonal things are encoded, so they get separate
#: colours: rank among usable entries, and whether the entry
#: is usable at all. A cell can never be both, since ranking
#: only considers in-window entries.
SHADE = r"\cellcolor{red!15}"          # outside the window
BEST = r"\cellcolor{green!18}"         # best in-window delta
SECOND = r"\cellcolor{cyan!14}"        # second best


def load():
    rows = {}
    for row in csv.DictReader(SOURCE.open()):
        rows[(row["method"], row["model"], row["dataset"])] = row
    return rows


def cells(row):
    """Delta and the median perplexity ratio.

    A vacuous entry is one the intervention cannot reach at all - a
    last-token edit against a teacher-forced score - so it is a dash
    rather than a real zero.
    """
    if row is None or str(row["vacuous"]).lower() == "true":
        return None, None
    return float(row["delta"]), float(row["norm_ppl_median"])


def markdown(rows):
    lines = []
    for key, label in MODELS:
        lines.append(f"\n### {label}\n")
        lines.append("| Method | " + " | ".join(
            f"{d} d | {d} ppl" for d in DATASETS) + " |")
        lines.append("|" + "---|" * (1 + 2 * len(DATASETS)))
        for entry in ROWS:
            if entry is None:
                continue
            method, _ = entry
            parts = []
            for dataset in DATASETS:
                delta, ppl = cells(rows.get((method, key, dataset)))
                parts += (["n/a", "n/a"] if delta is None
                          else [f"{delta:+.3f}", f"{ppl:.2f}"])
            lines.append(f"| {method} | " + " | ".join(parts) + " |")
    return "\n".join(lines)


PPL_LO, PPL_HI = 0.25, 2.0


def _ppl(value: float) -> str:
    """Perplexity ratio at a width proportional to how precise it is.

    The table is scaled to \\textwidth, so the widest cell decides the
    font size of every other cell. A ratio in the thousands conveys
    nothing in its hundredths, and printing them costs three characters
    in the column that happens to be the widest in the table.
    """
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _ranked(rows):
    """Best and second-best delta per (model, dataset) column.

    Only entries inside the fluency window are eligible. A larger delta
    bought by destroying fluency is not a better result, and a ratio far
    BELOW 1 is not either: it means the steered text collapsed into
    something trivially predictable, which is why the window has a floor
    as well as a ceiling. Ineligible entries are still printed in full,
    shaded rather than dropped.
    """
    marks = {}
    for key, _ in MODELS:
        for dataset in DATASETS:
            scored = []
            for entry in ROWS:
                if entry is None:
                    continue
                delta, ppl = cells(rows.get((entry[0], key, dataset)))
                if delta is not None and PPL_LO <= ppl <= PPL_HI:
                    scored.append((delta, entry[0]))
            scored.sort(reverse=True)
            for rank, (_, method) in enumerate(scored[:2]):
                marks[(method, key, dataset)] = rank
    return marks


def latex(rows):
    marks = _ranked(rows)
    out = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.75}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\small",
        CAPTION,
        r"\label{tab:main_results}",
        r"\vspace{1.5mm}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l | rr rr rr rr | rr rr rr rr}",
        r"\toprule",
        r"\multirow{3}{*}{\textbf{Method}} & "
        r"\multicolumn{8}{c|}{\textbf{Gemma-2-2B}} & "
        r"\multicolumn{8}{c}{\textbf{Llama-3.1-8B-Instruct}} \\",
        " & " + "\\multicolumn{2}{c}{\\textit{RTP}} & \\multicolumn{2}{c}{\\textit{Jigsaw}} & \\multicolumn{2}{c}{\\textit{Emotion}} & \\multicolumn{2}{c|}{\\textit{Sycophancy}} & \\multicolumn{2}{c}{\\textit{RTP}} & \\multicolumn{2}{c}{\\textit{Jigsaw}} & \\multicolumn{2}{c}{\\textit{Emotion}} & \\multicolumn{2}{c}{\\textit{Sycophancy}}" + r" \\",
        r"\cmidrule(lr){2-9}\cmidrule(lr){10-17}",
        " & " + " & ".join(
            r"$\Delta$ & $\overline{PPL}$" for _ in MODELS for _ in DATASETS
        ) + r" \\",
        r"\midrule",
    ]

    for entry in ROWS:
        if entry is None:
            out.append(r"\midrule")
            continue
        method, label = entry
        parts = []
        for key, _ in MODELS:
            for dataset in DATASETS:
                row = rows.get((method, key, dataset))
                delta, ppl = cells(row)
                if delta is None:
                    parts += [DASH, DASH]
                    continue
                text = f"{delta:.3f}"
                rank = marks.get((method, key, dataset))
                # Bold/underline are kept alongside the colour so the
                # ranking still reads on a greyscale printout, where the
                # green and cyan tints collapse to near-identical greys.
                if rank == 0:
                    text = rf"\mathbf{{{text}}}"
                    shade = BEST
                elif rank == 1:
                    text = rf"\uline{{{text}}}"
                    shade = SECOND
                elif PPL_LO <= ppl <= PPL_HI:
                    shade = ""
                else:
                    shade = SHADE
                parts += [shade + f"${text}$", shade + f"${_ppl(ppl)}$"]
        out.append(f"{label}\n  & " + "\n  & ".join(parts) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(out)


def tidy(rows):
    path = OUT / "paper_table.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "model", "dataset", "lambda", "n_seeds",
                         "delta_mean", "delta_sd", "ppl_median", "p_wilcoxon",
                         "vacuous"])
        for method in ORDER:
            for key, _ in MODELS:
                for dataset in DATASETS:
                    row = rows.get((method, key, dataset))
                    if row is None:
                        continue
                    writer.writerow([
                        method, key, dataset,
                        row["lambda"].split("|")[0], row["n_seeds"],
                        f"{float(row['delta']):.4f}",
                        f"{float(row['delta_std']):.4f}",
                        f"{float(row['norm_ppl_median']):.4f}",
                        row["p_wilcoxon"], row["vacuous"],
                    ])
    return path


if __name__ == "__main__":
    rows = load()
    print(markdown(rows))
    (OUT / "paper_table.tex").write_text(latex(rows) + "\n")
    path = tidy(rows)
    print(f"\nwrote {path} and {OUT / 'paper_table.tex'}")
