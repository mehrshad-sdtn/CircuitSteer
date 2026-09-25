"""Assemble the new CircuitSteer results into one comparable set.

Three runs feed the new column, all measured on the same machine:

  newrun/   the full re-run (both models, all four datasets)
  llamarun/ Llama re-measured on the wider coefficient grid
  jigx/     Gemma Jigsaw re-measured on the extended grid

Later sources win, so a cell present in more than one place takes the
most recent measurement of it. Results from a DIFFERENT machine are not
merged here: the same seed and coefficient measured on an A100 under
torch 2.11 read +0.235 where the A6000 under torch 2.7 reads +0.108, so
mixing them would put two different measurement conditions in one table.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def cell_of(name: str) -> tuple[str, str, str] | None:
    """(model, dataset, seed) for a unit directory, or None if unparsable."""
    parts = name.split("_")
    if len(parts) < 4:
        return None
    return parts[-3], parts[-2], parts[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path,
                        help="run directories, earliest first")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    chosen: dict[tuple[str, str, str], Path] = {}
    for source in args.sources:
        if not source.exists():
            print(f"  ! missing {source}")
            continue
        for unit in sorted(source.iterdir()):
            if not (unit / "circuitsteer_results.csv").exists():
                continue
            key = cell_of(unit.name)
            if key is None:
                continue
            if key in chosen:
                print(f"  {key[0]}/{key[1]}/{key[2]}: "
                      f"{chosen[key].parent.name} -> {source.name}")
            chosen[key] = unit

    args.out.mkdir(parents=True, exist_ok=True)
    for key, unit in sorted(chosen.items()):
        destination = args.out / f"CS_{key[0]}_{key[1]}_{key[2]}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(unit, destination)
    print(f"\n{len(chosen)} units -> {args.out}")
    models = sorted({k[0] for k in chosen})
    for model in models:
        datasets = sorted({k[1] for k in chosen if k[0] == model})
        for dataset in datasets:
            seeds = sorted(k[2] for k in chosen if k[:2] == (model, dataset))
            print(f"  {model:6s} {dataset:11s} {len(seeds)} seeds")


if __name__ == "__main__":
    main()
