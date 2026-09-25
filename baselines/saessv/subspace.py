"""SAE-SSV's subspace selection, as plain arrays.

Stage 1 of the method: find the small set of SAE dimensions that carry
the behaviour, before stage 2 learns a steering vector inside it. Kept
free of torch and sae_lens so it can be read and tested without a model.

Follows the reference implementation at
github.com/Ineedanamehere/SAE-SSV (SAESTEER/extractor.py):

* rank every SAE dimension by a binary-class ANOVA F-statistic and keep
  the top 128 that are actually active;
* train M = 50 linear probes on independently sampled halves of the
  data, restricted to those dimensions, and average their weights into
  one concept vector;
* sweep the top-d dimensions of that concept vector by magnitude and
  keep the smallest d whose class separation is within tolerance of the
  best, so the final subspace is as small as the separation allows.
"""
from __future__ import annotations

import numpy as np


def f_statistics(latents: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Binary-class ANOVA F per SAE dimension."""
    latents = np.asarray(latents, dtype=np.float32)
    labels = np.asarray(labels)
    if latents.ndim != 2:
        raise ValueError("latents must be [n_samples, n_features]")
    if latents.shape[0] != labels.shape[0]:
        raise ValueError("latents and labels must have the same length")

    class0, class1 = latents[labels == 0], latents[labels == 1]
    if class0.size == 0 or class1.size == 0:
        raise ValueError("both classes must be present")

    overall = latents.mean(axis=0)
    n0, n1 = class0.shape[0], class1.shape[0]
    between = (n0 * (class0.mean(axis=0) - overall) ** 2
               + n1 * (class1.mean(axis=0) - overall) ** 2)
    var0 = class0.var(axis=0, ddof=1) if n0 > 1 else np.zeros(latents.shape[1])
    var1 = class1.var(axis=0, ddof=1) if n1 > 1 else np.zeros(latents.shape[1])
    within = ((max(n0 - 1, 0) * var0 + max(n1 - 1, 0) * var1)
              / max(latents.shape[0] - 2, 1))
    scores = between / (within + 1e-8)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def coarse_subspace(
    latents: np.ndarray, labels: np.ndarray, top_k: int = 128
) -> np.ndarray:
    """Top-k active dimensions by F-statistic.

    Dimensions that are zero for every sample are skipped: a dead SAE
    feature has no variance to separate classes with, and including it
    would spend part of the budget on a dimension the steering vector
    could never move.
    """
    latents = np.asarray(latents, dtype=np.float32)
    scores = f_statistics(latents, labels)
    active = ~np.all(latents == 0, axis=0)
    ranked = np.argsort(-scores)
    ranked = ranked[active[ranked]]
    if ranked.size == 0:
        raise ValueError("no active SAE dimensions")
    return ranked[: max(1, min(int(top_k), ranked.size))].astype(np.int64)


def concept_vector(
    latents: np.ndarray,
    labels: np.ndarray,
    n_probes: int = 50,
    subset_fraction: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Mean weight vector over M probes fitted on random halves.

    Averaging many probes on resampled subsets is what makes the
    direction stable: a single probe on high-dimensional sparse latents
    picks up whichever dimensions happen to separate that particular
    sample.
    """
    from sklearn.linear_model import LogisticRegression

    latents = np.asarray(latents, dtype=np.float32)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    if len(class0) == 0 or len(class1) == 0:
        raise ValueError("both classes must be present")

    take0 = max(1, int(np.ceil(len(class0) * subset_fraction)))
    take1 = max(1, int(np.ceil(len(class1) * subset_fraction)))

    weights = []
    for _ in range(n_probes):
        idx = np.concatenate([
            rng.choice(class0, take0, replace=False),
            rng.choice(class1, take1, replace=False),
        ])
        if len(np.unique(labels[idx])) < 2:
            continue
        probe = LogisticRegression(max_iter=500)
        probe.fit(latents[idx], labels[idx])
        weights.append(probe.coef_[0])
    if not weights:
        raise ValueError("no probe could be fitted")
    return np.mean(weights, axis=0).astype(np.float32)


def fine_subspace(
    latents: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    separation_fraction: float = 0.95,
    candidates: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
) -> tuple[np.ndarray, float]:
    """Smallest top-d subspace reaching `separation_fraction` of the best.

    Separation is the gap between class means when the data is projected
    on the concept vector restricted to those d dimensions. Returns
    (indices into `latents`' columns, achieved separation).
    """
    latents = np.asarray(latents, dtype=np.float32)
    labels = np.asarray(labels)
    order = np.argsort(-np.abs(weights))

    sweep = []
    for d in candidates:
        if d > len(order):
            continue
        dims = order[:d]
        projection = latents[:, dims] @ weights[dims]
        gap = float(projection[labels == 1].mean()
                    - projection[labels == 0].mean())
        sweep.append((d, dims, gap))
    if not sweep:
        raise ValueError("no candidate subspace sizes fit the data")

    best = max(sweep, key=lambda item: item[2])
    threshold = best[2] * separation_fraction if best[2] > 0 else best[2]
    for d, dims, gap in sweep:
        if gap >= threshold:
            return dims.astype(np.int64), gap
    return best[1].astype(np.int64), best[2]


def initial_vector(
    source_centroid: np.ndarray,
    target_centroid: np.ndarray,
    dims: np.ndarray,
    d_sae: int,
) -> np.ndarray:
    """Stage-2 starting point: the unit source->target direction on `dims`."""
    vector = np.zeros(d_sae, dtype=np.float32)
    direction = target_centroid - source_centroid
    norm = float(np.linalg.norm(direction[dims]))
    if norm > 0:
        vector[dims] = direction[dims] / norm
    return vector


def distance_loss(
    steered: np.ndarray,
    target_centroid: np.ndarray,
    source_centroid: np.ndarray,
) -> float:
    """Pull towards the target centroid, push off the source one.

    The 0.5 on the source term is the reference implementation's, not
    the paper's symmetric form; keeping it means the two do the same
    thing.
    """
    return float(np.sum((steered - target_centroid) ** 2)
                 - 0.5 * np.sum((steered - source_centroid) ** 2))
