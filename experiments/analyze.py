"""Aggregate multi-seed CircuitSteer runs into a reportable table.

Two levels of statistics, because they answer different questions:

* seed level (n = number of seeds) asks "is the effect stable across
  re-draws of the split and the sampling noise?". With 3 seeds the
  interval uses t(0.975, df=2) = 4.303, so it is honest but wide.
* prompt level (n = prompts x seeds) asks "is the per-prompt score drop
  distinguishable from zero?". This is where the power is. Prompts are
  re-drawn per seed from a shared pool, so treating the pooled pairs as
  fully independent is mildly optimistic; the Wilcoxon column is reported
  alongside because toxicity scores are heavily skewed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, prompts = [], []
    for path in sorted(results_dir.rglob("circuitsteer_results.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = _seed_of(path)
        summary.append(frame)
    for path in sorted(results_dir.rglob("circuitsteer_prompt_scores.csv")):
        prompts.append(pd.read_csv(path))
    if not summary:
        raise SystemExit(f"no results under {results_dir}")
    return pd.concat(summary, ignore_index=True), (
        pd.concat(prompts, ignore_index=True) if prompts else pd.DataFrame()
    )


def _seed_of(path: Path) -> int:
    for part in path.parts[::-1]:
        if "seed" in part:
            digits = "".join(c for c in part.split("seed")[-1] if c.isdigit())
            if digits:
                return int(digits)
    return -1


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Mean, and the half-width of a t-based 95% interval."""
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, np.nan
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return mean, sd, half


def choose_lambda(
    summary: pd.DataFrame,
    max_val_ppl: float | None = None,
) -> dict[tuple[str, str], float]:
    """One coefficient per (model, dataset), shared by every seed.

    The protocol is to tune lambda once per entry, not per seed. A run
    that argmaxes validation within each seed picks a different lambda
    per seed, which quietly turns the reported number into a max over a
    grid x seeds and inflates it. We pool instead: average the validation
    delta over seeds at each coefficient and take the argmax of that.

    `max_val_ppl` restricts the argmax to coefficients whose pooled
    VALIDATION perplexity ratio stays within the window. Without it, a
    wide grid makes the argmax run to whichever end steers hardest, which
    on a fluency-destroying setting is not an operating point anyone
    would ship. If no coefficient qualifies, the entry falls back to the
    gentlest one on the grid rather than being dropped, so the cell is
    still reported and can be seen to be out of range.

    Selection uses validation only; test is never consulted.
    """
    val = summary[summary["split"] == "val"]
    chosen: dict[tuple[str, str], float] = {}
    if val.empty:
        return chosen
    grouped = val.groupby(["model", "dataset", "coeff"])
    pooled = grouped["delta"].mean()
    ppl_column = (
        "norm_ppl_median" if "norm_ppl_median" in val.columns else "norm_ppl"
    )
    pooled_ppl = grouped[ppl_column].mean()
    for (model, dataset), group in pooled.groupby(level=[0, 1]):
        if max_val_ppl is not None:
            ppl = pooled_ppl.loc[(model, dataset)]
            eligible = group[
                group.index.get_level_values("coeff").map(
                    lambda c: bool(ppl.get(c, np.inf) <= max_val_ppl)
                )
            ]
            if not eligible.empty:
                chosen[(model, dataset)] = float(eligible.idxmax()[2])
                continue
            # Nothing in range: report the gentlest setting instead of
            # silently taking the most destructive one.
            coeffs = group.index.get_level_values("coeff")
            chosen[(model, dataset)] = float(max(coeffs, key=lambda c: c))
            continue
        chosen[(model, dataset)] = float(group.idxmax()[2])
    return chosen


def analyse(summary: pd.DataFrame, prompts: pd.DataFrame,
            per_seed_lambda: bool = False,
            max_val_ppl: float | None = None) -> pd.DataFrame:
    rows = []
    test = summary[summary["split"] == "test"]
    fixed = ({} if per_seed_lambda
             else choose_lambda(summary, max_val_ppl=max_val_ppl))
    if fixed:
        # One lambda per entry, applied to all seeds.
        keep = [
            np.isclose(row["coeff"], fixed[(row["model"], row["dataset"])])
            if (row["model"], row["dataset"]) in fixed else False
            for _, row in test.iterrows()
        ]
        test = test[pd.Series(keep, index=test.index)]
    elif "selected" in test.columns:
        # No validation rows in these results (the run already fixed
        # lambda upstream), so fall back to the flag it recorded.
        test = test[test["selected"].astype(str).str.lower() == "true"]
    benign = summary[summary["split"] == "test_benign"]

    for (model, dataset), group in test.groupby(["model", "dataset"]):
        delta_mean, delta_sd, delta_half = ci95(group["delta"].to_numpy(float))
        ppl_mean, ppl_sd, _ = ci95(group["norm_ppl"].to_numpy(float))
        base_ppl = float(np.nanmean(group.get("base_ppl", np.nan)))
        steered_ppl = float(np.nanmean(group.get("steered_ppl", np.nan)))
        # Perplexity is heavy-tailed: a single degenerate base continuation
        # can move the ratio-of-means by orders of magnitude (we have seen
        # base_ppl = 2.8e7 in one seed). The median columns are what the
        # paper table reports, and what the coherence judge was calibrated
        # against; each seed contributes its own within-run median and we
        # average those across seeds.
        def _seed_mean(column: str) -> float:
            if column not in group.columns:
                return np.nan
            return float(np.nanmean(group[column].to_numpy(float)))

        base_ppl_median = _seed_mean("base_ppl_median")
        steered_ppl_median = _seed_mean("steered_ppl_median")
        norm_ppl_median = _seed_mean("norm_ppl_median")

        control = benign[
            (benign["model"] == model) & (benign["dataset"] == dataset)
        ]
        control_mean = (
            float(np.nanmean(control["delta"].to_numpy(float)))
            if not control.empty
            else np.nan
        )

        # Pair only the rows for the coefficient actually selected in
        # that seed. Pooling every coefficient mixes in lambda=0 (delta
        # identically zero) and the positive lambdas (which push the
        # behaviour the other way), which cancels the effect and drives
        # the p-value to 1 regardless of the result.
        paired = pd.DataFrame()
        if not prompts.empty:
            chosen = group[["seed", "coeff"]].drop_duplicates()
            slices = []
            for _, row in chosen.iterrows():
                slices.append(
                    prompts[
                        (prompts["model"] == model)
                        & (prompts["dataset"] == dataset)
                        & (prompts["split"] == "test")
                        & (prompts["seed"] == row["seed"])
                        & (np.isclose(prompts["coeff"], row["coeff"]))
                    ]
                )
            if slices:
                paired = pd.concat(slices, ignore_index=True)

        t_p = w_p = cohen_d = np.nan
        n_pairs = 0
        boot_lo = boot_hi = np.nan
        if not paired.empty:
            diff = (
                paired["base_score"].to_numpy(float)
                - paired["steered_score"].to_numpy(float)
            )
            diff = diff[~np.isnan(diff)]
            n_pairs = len(diff)
            if n_pairs > 1 and np.std(diff) > 0:
                t_p = float(stats.ttest_1samp(diff, 0.0).pvalue)
                try:
                    w_p = float(stats.wilcoxon(diff).pvalue)
                except ValueError:
                    w_p = np.nan
                cohen_d = float(diff.mean() / diff.std(ddof=1))
                rng = np.random.default_rng(0)
                boots = [
                    rng.choice(diff, n_pairs, replace=True).mean()
                    for _ in range(10000)
                ]
                boot_lo, boot_hi = np.percentile(boots, [2.5, 97.5])

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n_seeds": int(group["delta"].notna().sum()),
                "lambda": "|".join(
                    f"{c:g}" for c in group.sort_values("seed")["coeff"]
                ),
                "delta": delta_mean,
                "delta_std": delta_sd,
                "ci95_lo": delta_mean - delta_half,
                "ci95_hi": delta_mean + delta_half,
                "base_ppl": base_ppl,
                "steered_ppl": steered_ppl,
                "norm_ppl": ppl_mean,
                "ppl_std": ppl_sd,
                "base_ppl_median": base_ppl_median,
                "steered_ppl_median": steered_ppl_median,
                "norm_ppl_median": norm_ppl_median,
                "benign_delta": control_mean,
                "n_pairs": n_pairs,
                "p_paired_t": t_p,
                "p_wilcoxon": w_p,
                "cohen_d": cohen_d,
                "boot_lo": boot_lo,
                "boot_hi": boot_hi,
                "degen_frac": float(
                    np.nanmean(group["degenerate_frac"].to_numpy(float))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "dataset"])


def _distinct_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are unique. Low means repetitive."""
    words = str(text).split()
    if len(words) < n:
        return np.nan
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def diagnose(results_dir: Path) -> pd.DataFrame:
    """Why norm_ppl can look 'good' while the text is broken.

    norm_ppl divides two *means*, and perplexity is heavy-tailed, so a few
    high-perplexity base samples can drag the ratio far below 1 even when
    nothing improved. Repetitive steered text is the other trap: it has
    genuinely low perplexity and low toxicity, yet it is not fluent output.
    The median per-sample ratio and the distinct-4-gram rate separate the
    two cases.
    """
    frames = []
    for path in sorted(results_dir.rglob("circuitsteer_qual.csv")):
        # A run with --qualitative-samples 0 writes the file with no
        # header at all, which read_csv raises on rather than returning
        # empty. The diagnostics are optional, so skip instead of dying.
        if path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        frame["seed"] = _seed_of(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    qual = pd.concat(frames, ignore_index=True)

    rows = []
    for (model, dataset), g in qual.groupby(["model", "dataset"]):
        base_ppl = g["base_ppl"].to_numpy(float)
        steered_ppl = g["steered_ppl"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = steered_ppl / base_ppl
        ratio = ratio[np.isfinite(ratio)]
        rows.append({
            "model": model,
            "dataset": dataset,
            "ppl_ratio_mean_of_means": (
                np.nanmean(steered_ppl) / np.nanmean(base_ppl)
            ),
            "ppl_ratio_median": float(np.median(ratio)) if ratio.size else np.nan,
            "base_ppl_median": float(np.nanmedian(base_ppl)),
            "base_ppl_max": float(np.nanmax(base_ppl)),
            "steered_ppl_median": float(np.nanmedian(steered_ppl)),
            "distinct4_base": float(
                np.nanmean([_distinct_ratio(t) for t in g["base_output"]])
            ),
            "distinct4_steered": float(
                np.nanmean([_distinct_ratio(t) for t in g["steered_output"]])
            ),
            "empty_steered": float(
                np.mean([not str(t).strip() for t in g["steered_output"]])
            ),
        })
    return pd.DataFrame(rows).sort_values(["model", "dataset"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--max-val-ppl",
        type=float,
        default=None,
        help="Restrict lambda selection to coefficients whose pooled "
             "VALIDATION perplexity ratio is at most this. Without it a "
             "wide grid makes the argmax run to whichever end steers "
             "hardest, regardless of what it does to fluency.",
    )
    parser.add_argument(
        "--per-seed-lambda", action="store_true",
        help="argmax validation within each seed (inflates the result; "
             "kept only for comparison against the old tables)")
    args = parser.parse_args()

    summary, prompts = load(args.results_dir)
    table = analyse(summary, prompts, per_seed_lambda=args.per_seed_lambda,
                    max_val_ppl=args.max_val_ppl)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    diag = diagnose(args.results_dir)
    if not diag.empty:
        print("\n--- fluency diagnostics ---")
        print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        if not diag.empty:
            diag.to_csv(args.out.with_name(args.out.stem + "_diag.csv"),
                        index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
