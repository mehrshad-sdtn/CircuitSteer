"""Turn pairwise coherence verdicts into a perplexity-ratio range.

Reports the interval of steered/base perplexity ratio over which the
steered text is, with confidence, at least as understandable as the
unsteered text for the same prompt.

Method notes:

* Verdicts are de-biased by order first. Each pair is judged twice with
  the sides swapped; only pairs whose two verdicts agree after
  un-swapping are used. Order-inconsistent pairs are counted and
  reported rather than broken by a coin flip.
* Rates are binned on log2(ratio) with Wilson confidence intervals. No
  parametric shape is assumed: the low tail (steered cheaper than base)
  and high tail (steered blown up) are different phenomena in this data -
  low ratios are driven largely by pathological *base* samples - so a
  symmetric model around 1.0 would be wrong.
* The accepted range is the contiguous run of bins whose Wilson LOWER
  bound clears the target. Using the lower bound means thin bins cannot
  widen the range by luck.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def resolve(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two orders of each real pair into one verdict."""
    real = frame[frame.kind == "real"].copy()
    # Normalise to a statement about the steered text.
    def about_steered(row):
        if row.verdict == "equivalent":
            return "comparable"
        winner_is_steered = (row.verdict.upper() == row.steered_side)
        return "steered_better" if winner_is_steered else "steered_worse"

    real["call"] = real.apply(about_steered, axis=1)
    grouped = real.groupby("pair_id")
    rows = []
    for pair_id, g in grouped:
        calls = set(g.call)
        meta = g.iloc[0]
        rows.append({
            "pair_id": pair_id,
            "model": meta.model,
            "dataset": meta.dataset,
            "coeff": meta.coeff,
            "base_ppl": meta.base_ppl,
            "steered_ppl": meta.steered_ppl,
            "n_orders": len(g),
            "consistent": len(calls) == 1 and len(g) == 2,
            "call": calls.pop() if len(calls) == 1 else "inconsistent",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verdicts_csv", type=Path)
    parser.add_argument("--target", type=float, default=0.90,
                        help="Required lower-bound rate of 'not worse'.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.verdicts_csv)

    # --- controls first: they bound what any calibration can claim ---
    print("=== controls ===")
    for kind, expectation in (
        ("control_identical", "should be ~100% 'equivalent'"),
        ("control_mismatched", "judge should call the foreign text worse"),
    ):
        sub = frame[frame.kind == kind]
        if sub.empty:
            print(f"{kind}: none"); continue
        counts = sub.verdict.value_counts().to_dict()
        rate = (sub.verdict == "equivalent").mean()
        lo, hi = wilson(int((sub.verdict == "equivalent").sum()), len(sub))
        print(f"{kind}: n={len(sub)} equivalent={rate:.1%} "
              f"[{lo:.1%},{hi:.1%}] {counts}  <- {expectation}")

    pairs = resolve(frame)
    n_total = len(pairs)
    consistent = pairs[pairs.consistent & (pairs.call != "inconsistent")]
    print(f"\n=== order consistency ===")
    print(f"pairs judged: {n_total} | order-consistent: {len(consistent)} "
          f"({len(consistent)/max(n_total,1):.1%}) | "
          f"position-bias dropped: {n_total-len(consistent)}")

    usable = consistent[
        np.isfinite(consistent.base_ppl) & np.isfinite(consistent.steered_ppl)
        & (consistent.base_ppl > 0) & (consistent.steered_ppl > 0)
    ].copy()
    usable["ratio"] = usable.steered_ppl / usable.base_ppl
    usable["lr"] = np.log2(usable.ratio)
    usable["ok"] = usable.call.isin(["comparable", "steered_better"])

    print(f"\n=== overall ===")
    print(usable.call.value_counts().to_string())
    print(f"'at least as understandable': {usable.ok.mean():.1%}")

    edges = [-np.inf, -3, -2, -1, -0.5, 0.5, 1, 1.5, 2, 3, np.inf]
    names = ["<0.125", "0.125-0.25", "0.25-0.5", "0.5-0.71", "0.71-1.41",
             "1.41-2", "2-2.83", "2.83-4", "4-8", ">8"]
    usable["band"] = pd.cut(usable.lr, bins=edges, labels=names)

    print(f"\n=== rate of 'at least as understandable' by ratio band ===")
    rows = []
    for name in names:
        sub = usable[usable.band == name]
        n = len(sub)
        k = int(sub.ok.sum())
        lo, hi = wilson(k, n)
        rows.append({"band": name, "n": n, "ok": k,
                     "rate": (k / n if n else np.nan),
                     "wilson_lo": lo, "wilson_hi": hi,
                     "passes": bool(n > 0 and lo >= args.target)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    passing = table[table.passes]
    print(f"\n=== accepted ratio range (Wilson lower bound >= {args.target}) ===")
    if passing.empty:
        print("NO band clears the target.")
        print("Either the target is too strict for this sample size, or the")
        print("ratio does not track coherence. Check the controls above and")
        print("the overall rate before reading anything into a threshold.")
    else:
        idx = [names.index(b) for b in passing.band]
        contiguous, run = [], [idx[0]]
        for prev, cur in zip(idx, idx[1:]):
            (run.append(cur) if cur == prev + 1
             else (contiguous.append(run), run := [cur]))
        contiguous.append(run)
        longest = max(contiguous, key=len)
        lo_band, hi_band = names[longest[0]], names[longest[-1]]
        covered = usable[usable.band.isin([names[i] for i in longest])]
        print(f"bands {lo_band} .. {hi_band}")
        print(f"ratio approx [{2**edges[longest[0]+0]:.3g}, "
              f"{2**edges[longest[-1]+1]:.3g}]"
              if np.isfinite(edges[longest[0]]) and np.isfinite(edges[longest[-1]+1])
              else f"bands {lo_band} .. {hi_band} (open-ended)")
        print(f"covers {len(covered)}/{len(usable)} "
              f"({len(covered)/len(usable):.1%}) of judged pairs, "
              f"rate {covered.ok.mean():.1%}")
        if len(contiguous) > 1:
            print(f"note: {len(contiguous)} disjoint passing runs - "
                  "the relationship is not a simple band")

    if args.out:
        table.to_csv(args.out, index=False)
        usable.to_csv(args.out.with_name(args.out.stem + "_pairs.csv"),
                      index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
