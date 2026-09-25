"""Re-select every method's coefficient under one operating-point rule.

The reported cell is whatever coefficient validation picks. With a narrow
grid that choice is nearly forced; with a wide one a pure delta-argmax
runs to whichever end steers hardest, and reports a cell whose perplexity
ratio is in the hundreds. Neither is a rule -- the first is an accident of
the grid, the second is an accident of noise.

This re-runs the aggregation for every method from its per-coefficient
raw results, choosing lambda as the argmax of pooled VALIDATION delta
among coefficients whose pooled validation perplexity stays within the
window. Same rule for every method, selection still on validation only,
and no GPU: the full lambda curve was already recorded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.analyze import analyse, load

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: label -> directory holding that method's per-unit raw results.
SOURCES = {
    "CAA-1L": RESULTS / "baselines/caa_1l",
    "CAA-top3": RESULTS / "baselines/caa_top3",
    "RepE": RESULTS / "baselines/repe",
    "ITI": RESULTS / "baselines/iti",
    "LoReFT": RESULTS / "baselines/loreft",
    "SpARE": RESULTS / "baselines/spare",
    "SAE-SSV": RESULTS / "baselines/saessv",
    "FeatureFlow": RESULTS / "baselines/featureflow",
    "Prompt": RESULTS / "baselines/prompt",
    "CircuitSteer": RESULTS / "circuitsteer",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-val-ppl", type=float, default=2.0)
    parser.add_argument("--out-name", default="summary_windowed.csv")
    parser.add_argument(
        "--circuitsteer-dir", type=Path, default=None,
        help="Override CircuitSteer's raw directory, for a fresh run.",
    )
    args = parser.parse_args()

    sources = dict(SOURCES)
    if args.circuitsteer_dir is not None:
        sources["CircuitSteer"] = args.circuitsteer_dir

    for label, directory in sources.items():
        if not directory.exists():
            print(f"{label:14s} no directory at {directory}")
            continue
        try:
            summary, prompts = load(directory)
        except SystemExit as exc:
            print(f"{label:14s} {exc}")
            continue
        if "val" not in set(summary.get("split", [])):
            print(f"{label:14s} no validation rows; left as-is")
            continue
        table = analyse(summary, prompts, max_val_ppl=args.max_val_ppl)
        destination = directory / args.out_name
        table.to_csv(destination, index=False)
        cells = len(table)
        inside = int((table["norm_ppl_median"] <= args.max_val_ppl).sum())
        print(f"{label:14s} {cells} cells, {inside} within "
              f"ppl<={args.max_val_ppl}  -> {destination.name}")


if __name__ == "__main__":
    main()
