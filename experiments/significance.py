"""Paired significance tests for Table 1.

Every method is evaluated on the same held-out test prompts per seed, with
identical unsteered generations (base scores match exactly across methods),
so methods can be compared prompt by prompt. For each cell this reports

* each method's reduction against zero (Wilcoxon signed-rank on the
  per-prompt reductions base - steered, pooled over the three seeds), at the
  coefficient Table 1 reports for that method;
* CircuitSteer against every baseline (Wilcoxon signed-rank on the paired
  difference of per-prompt reductions), with a bootstrap 95% interval on the
  mean difference and a Holm correction over all comparisons.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results")
SOURCES = {
    "Prompt": "baselines/prompt", "CAA-1L": "baselines/caa_1l",
    "CAA-top3": "baselines/caa_top3", "RepE": "baselines/repe",
    "ITI": "baselines/iti", "LoReFT": "baselines/loreft",
    "SpARE": "baselines/spare", "SAE-SSV": "baselines/saessv",
    "FeatureFlow": "baselines/featureflow", "CircuitSteer": "circuitsteer_v2",
}
KEY = ["model", "dataset", "seed", "prompt_index"]


def load_scores(directory: str) -> pd.DataFrame:
    files = glob.glob(str(RESULTS / directory / "**/circuitsteer_prompt_scores.csv"),
                      recursive=True)
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame[frame.split == "test"].drop_duplicates(KEY + ["coeff"])


def selected(table: pd.DataFrame, method: str, model: str, dataset: str):
    row = table[(table.method == method) & (table.model == model)
                & (table.dataset == dataset)].iloc[0]
    lambdas = [float(x) for x in str(row["lambda"]).split("|")]
    return lambdas, bool(row.vacuous), row.norm_ppl_median


def reductions(scores, model, dataset, lambdas):
    """Per-prompt base - steered at the per-seed selected coefficient."""
    parts = []
    for seed, lam in zip((42, 43, 44), lambdas if len(lambdas) == 3 else lambdas * 3):
        g = scores[(scores.model == model) & (scores.dataset == dataset)
                   & (scores.seed == seed) & np.isclose(scores.coeff, lam)]
        parts.append(g.assign(d=g.base_score - g.steered_score)[KEY + ["d"]])
    return pd.concat(parts, ignore_index=True)


def wilcoxon(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return np.nan
    return float(stats.wilcoxon(x).pvalue)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = order.notna().sum()
    adj, running = {}, 0.0
    for rank, (idx, value) in enumerate(order.dropna().items()):
        running = max(running, min(1.0, (m - rank) * value))
        adj[idx] = running
    return pd.Series(adj).reindex(p.index)


def main() -> None:
    table = pd.read_csv(RESULTS / "main_table_v2.csv")
    scores = {m: load_scores(d) for m, d in SOURCES.items()}
    cells = sorted({(r.model, r.dataset) for r in table.itertuples()})

    own, versus = [], []
    for model, dataset in cells:
        cs_l, _, _ = selected(table, "CircuitSteer", model, dataset)
        cs = reductions(scores["CircuitSteer"], model, dataset, cs_l)
        for method in SOURCES:
            lam, vacuous, ppl = selected(table, method, model, dataset)
            if vacuous:
                own.append(dict(model=model, dataset=dataset, method=method,
                                n=0, delta=np.nan, p=np.nan, in_window=False))
                continue
            d = reductions(scores[method], model, dataset, lam)
            own.append(dict(model=model, dataset=dataset, method=method,
                            n=len(d), delta=d.d.mean(), p=wilcoxon(d.d),
                            in_window=0.25 <= ppl <= 2.0))
            if method == "CircuitSteer":
                continue
            m = cs.merge(d, on=KEY, suffixes=("_cs", "_b"))
            diff = m.d_cs - m.d_b
            lo, hi = boot_ci(diff)
            versus.append(dict(model=model, dataset=dataset, baseline=method,
                               n=len(m), cs=m.d_cs.mean(), base=m.d_b.mean(),
                               diff=diff.mean(), lo=lo, hi=hi,
                               p=wilcoxon(diff), base_in_window=0.25 <= ppl <= 2.0))
    own = pd.DataFrame(own)
    versus = pd.DataFrame(versus)
    versus["p_holm"] = holm(versus.p)
    out = RESULTS / "significance"
    out.mkdir(exist_ok=True)
    own.to_csv(out / "vs_zero.csv", index=False)
    versus.to_csv(out / "cs_vs_baselines.csv", index=False)
    pd.set_option("display.width", 200)
    print(own.round(4).to_string(index=False))
    print(versus.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
