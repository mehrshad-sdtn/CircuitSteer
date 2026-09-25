"""SRPS feature selection, as plain arrays.

The paper's equations 1-4, kept free of torch and sae_lens so they can
be read and tested without a model:

    mu_i    = mean_j ( a+_ij - a-_ij )                              (1)
    delta_i = mean_j 1(a+_ij > theta) - mean_j 1(a-_ij > theta)     (2)
    I_i     = mu_i + beta * delta_i                                 (3)
    s       = sum_{i in top-k} alpha_i * W_dec[i]                   (4)

where a+ and a- are SAE feature activations at the last token with and
without the contrastive prompt, alpha_i is feature i's mean activation
over the positive samples, and top-k is by sensitivity score I.

Reference: Wang, Shu, Wang, Ma and Du, "Improving LLM Reasoning through
Interpretable Role-Playing Steering" (SRPS), arXiv:2506.07335, 2025.
"""
from __future__ import annotations

import numpy as np


def sensitivity(
    positive: np.ndarray,
    negative: np.ndarray,
    beta: float = 1.0,
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (score, mu, delta) per SAE feature.

    `positive` and `negative` are [n_samples, n_features] activations.

    beta and theta are described in the paper as tunable and their values
    are not reported, so the defaults here are ours: theta = 0 treats any
    non-zero activation of a ReLU-sparse SAE as "activated", and beta = 1
    weights the two terms equally. Both are exposed so the choice can be
    varied rather than buried.
    """
    if positive.shape != negative.shape:
        raise ValueError("positive and negative must have the same shape")
    if positive.size == 0:
        raise ValueError("no activations supplied")

    mu = (positive - negative).mean(axis=0)
    delta = ((positive > theta).mean(axis=0)
             - (negative > theta).mean(axis=0))
    return mu + beta * delta, mu, delta


def select_features(
    positive: np.ndarray,
    negative: np.ndarray,
    k: int = 15,
    beta: float = 1.0,
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k features by sensitivity, with their positive-sample weights.

    Returns (indices, alpha, score). alpha_i is the mean activation of
    feature i over the positive samples, which equation 4 uses to weight
    that feature's decoder vector.
    """
    score, _, _ = sensitivity(positive, negative, beta=beta, theta=theta)
    k = min(k, score.shape[0])
    indices = np.argsort(score)[::-1][:k].copy()
    alpha = positive[:, indices].mean(axis=0)
    return indices, alpha, score[indices]


def steering_vector(
    decoder: np.ndarray,
    indices: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Equation 4: the activation-weighted sum of selected decoder rows.

    `decoder` is W_dec with shape [n_features, d_model].
    """
    if len(indices) != len(alpha):
        raise ValueError("indices and alpha must have the same length")
    if len(indices) == 0:
        return np.zeros(decoder.shape[1])
    return (alpha[:, None] * decoder[indices]).sum(axis=0)


def apply_shift(
    residual: np.ndarray,
    shift: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    """Equations 5 and 6: add the shift, then restore the original norm.

    The renormalisation is not decoration. Without it the added vector
    grows the residual stream without bound as the coefficient rises,
    and the method's own stability argument rests on it, so a version
    that omitted it would not be this baseline.
    """
    updated = residual + coefficient * shift
    before = np.linalg.norm(residual, axis=-1, keepdims=True)
    after = np.linalg.norm(updated, axis=-1, keepdims=True)
    return updated * (before / np.maximum(after, 1e-9))


def shift_last_token(residual, shift, coefficient):
    """Equations 5 and 6 on a torch residual-stream tensor.

    `residual` is [batch, pos, d_model]. Only the last position moves,
    and it keeps the norm it had. Torch is imported here rather than at
    module scope so the array maths above stays importable without it.
    """
    if coefficient == 0.0:
        return residual
    residual = residual.clone()
    last = residual[:, -1, :]
    before = last.norm(dim=-1, keepdim=True)
    updated = last + coefficient * shift.to(last.dtype)
    after = updated.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    residual[:, -1, :] = updated * (before / after)
    return residual
