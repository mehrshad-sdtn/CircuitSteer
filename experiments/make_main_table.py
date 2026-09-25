"""Merge per-method summaries into the paper's main comparison table.

Reports, per (model, dataset, method): behaviour reduction (mean over
seeds, with the seed-level SD), the MEDIAN perplexity ratio, and the
prompt-level Wilcoxon p and Cohen's d.

The perplexity column is the median ratio, not norm_ppl. norm_ppl is a
ratio of means over a heavy-tailed quantity, so one degenerate base
continuation moves it by orders of magnitude - gemma/Jigsaw reads 398
under the mean and 1.27 under the median. The median is also the
statistic the coherence judge was calibrated against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: label -> summary.csv, in the order the paper lists them.
METHODS = {
    "CAA-1L": RESULTS / "baselines/caa_1l/summary.csv",
    "CAA-top3": RESULTS / "baselines/caa_top3/summary.csv",
    "RepE": RESULTS / "baselines/repe/summary.csv",
    "ITI": RESULTS / "baselines/iti/summary.csv",
    "LoReFT": RESULTS / "baselines/loreft/summary.csv",
    "SpARE": RESULTS / "baselines/spare/summary.csv",
    "SAE-SSV": RESULTS / "baselines/saessv/summary.csv",
    "Prompt": RESULTS / "baselines/prompt/summary.csv",
    "CircuitSteer": RESULTS / "circuitsteer/summary.csv",
}

KEEP = [
    "model", "dataset", "n_seeds", "lambda", "delta", "delta_std",
    "ci95_lo", "ci95_hi",
    # Both the ratio-of-means and the median ratio, plus the absolute
    # perplexities behind them, so a reader can see what the ratio is a
    # ratio of.
    "norm_ppl", "norm_ppl_median",
    "base_ppl", "steered_ppl", "base_ppl_median", "steered_ppl_median",
    "p_wilcoxon", "cohen_d", "benign_delta", "degen_frac",
]


def load() -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for label, path in METHODS.items():
        if not path.exists():
            missing.append(label)
            continue
        frame = pd.read_csv(path)
        frame["method"] = label
        for column in KEEP:
            if column not in frame.columns:
                frame[column] = pd.NA
        frames.append(frame[KEEP + ["method"]])
    if not frames:
        raise SystemExit("no method summaries found")
    return pd.concat(frames, ignore_index=True), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "main_table.csv")
    args = parser.parse_args()

    table, missing = load()
    order = [m for m in METHODS if m not in missing]
    table["method"] = pd.Categorical(table["method"], order, ordered=True)
    table = table.sort_values(["model", "dataset", "method"])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    # A cell where every seed returned exactly the same delta AND no
    # test could be computed did not measure anything: the intervention
    # never reached the metric. Last-token methods hit this on
    # Sycophancy, which is teacher-forced and reads logits[:-1], so the
    # one position they modify is the one position it ignores. Marking
    # it beats printing 0.0000 as though it were an effect.
    vacuous = (
        table["delta"].abs().lt(1e-12)
        & table["delta_std"].abs().lt(1e-12)
        & table["p_wilcoxon"].isna()
    )
    table["vacuous"] = vacuous

    show = table.copy()
    show["delta"] = [
        "n/a (no effect on metric)" if v
        else f"{d:+.4f} ±{sd:.4f}"
        for d, sd, v in zip(show["delta"], show["delta_std"], show["vacuous"])
    ]
    show["ppl_med"] = show["norm_ppl_median"].map(lambda v: f"{v:.2f}")
    show["p"] = show["p_wilcoxon"].map(lambda v: f"{v:.4f}")
    show["d"] = show["cohen_d"].map(lambda v: f"{v:+.3f}")
    columns = ["model", "dataset", "method", "n_seeds", "lambda",
               "delta", "ppl_med", "p", "d"]
    print(show[columns].to_string(index=False))

    if missing:
        print(f"\nMISSING (no results yet): {', '.join(missing)}")
    if vacuous.any():
        n = int(vacuous.sum())
        print(f"\n{n} cell(s) marked n/a: the steering never reached the "
              f"metric, so the zero is an artefact, not a measurement.")

    table.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
