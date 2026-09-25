"""Feature-flow tracing and seed selection, as plain arrays.

The flow graph is the paper's data-free matching: a feature at one layer
is linked to the feature at the next layer whose decoder direction has
the highest cosine similarity with it,

    j = argmax_k cos(d_i^(A), d_k^(B)),                             (1)

and a chain continues only while that similarity stays above the
residual-stream cutoff t = 0.5 the paper uses to delimit a feature's
similarity span.

Reference: Laptev, Balagansky, Aksenov and Gavrilov, "Analyze Feature
Flow to Enhance Interpretation and Steering in Language Models",
arXiv:2502.03032, 2025.

Seed selection is ours. The paper picks the starting feature by hand
from its Neuronpedia description, which cannot be repeated fairly over
eight model-dataset cells and three seeds, so the seed is the single
feature, over every SAE layer, that best separates target from benign
training prompts (`standardized_contrast`).
"""
from __future__ import annotations

import numpy as np

#: The paper's residual similarity-span cutoff, t^(R).
SPAN_CUTOFF = 0.5


def standardized_contrast(
    positive: np.ndarray,
    negative: np.ndarray,
    min_freq: float = 0.1,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-feature (mean_pos - mean_neg) / pooled std.

    Standardising makes scores comparable across layers, whose SAEs
    activate on different scales. A feature must fire on at least
    `min_freq` of the target prompts: otherwise a feature seen once, on
    one prompt, with near-zero variance elsewhere, wins on a tiny
    denominator. Such features score -inf.
    """
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative must share the feature axis")
    if positive.size == 0 or negative.size == 0:
        raise ValueError("no activations supplied")
    diff = positive.mean(axis=0) - negative.mean(axis=0)
    pooled = np.sqrt((positive.var(axis=0) + negative.var(axis=0)) / 2)
    score = diff / (pooled + eps)
    score[(positive > 0).mean(axis=0) < min_freq] = -np.inf
    return score


def select_seed(
    positive: dict[int, np.ndarray],
    negative: dict[int, np.ndarray],
    min_freq: float = 0.1,
) -> tuple[int, int, float]:
    """The (layer, feature, score) with the largest contrast anywhere."""
    best = (None, None, -np.inf)
    for layer in sorted(positive):
        score = standardized_contrast(
            positive[layer], negative[layer], min_freq=min_freq,
        )
        index = int(np.argmax(score))
        if score[index] > best[2]:
            best = (layer, index, float(score[index]))
    if best[0] is None:
        raise ValueError(
            f"no feature fires on {min_freq:.0%} of target prompts"
        )
    return best


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def match(direction: np.ndarray, decoder: np.ndarray) -> tuple[int, float]:
    """Equation 1: the decoder row most cosine-similar to `direction`."""
    cosines = _unit(decoder) @ _unit(direction)
    index = int(np.argmax(cosines))
    return index, float(cosines[index])


def trace_flow(
    seed_layer: int,
    seed_feature: int,
    decoders: dict[int, np.ndarray],
    cutoff: float = SPAN_CUTOFF,
) -> list[tuple[int, int, float]]:
    """The seed's chain as [(layer, feature, cosine to its neighbour)].

    Traced outward from the seed in both directions, one layer at a time,
    each step matching from the previous chain feature rather than the
    seed, and stopping at the first link below `cutoff`. The seed's own
    cosine is recorded as 1.0. Returned in layer order.
    """
    layers = sorted(decoders)
    if seed_layer not in decoders:
        raise ValueError(f"seed layer {seed_layer} has no decoder")
    position = layers.index(seed_layer)
    chain = {seed_layer: (seed_feature, 1.0)}
    for step in (1, -1):
        feature = seed_feature
        current = position
        while 0 <= current + step < len(layers):
            source = decoders[layers[current]][feature]
            target_layer = layers[current + step]
            index, cosine = match(source, decoders[target_layer])
            if cosine < cutoff:
                break
            chain[target_layer] = (index, cosine)
            feature, current = index, current + step
    return [(layer, *chain[layer]) for layer in sorted(chain)]


def steering_vectors(
    chain: list[tuple[int, int, float]],
    decoders: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """One unit decoder direction per chain layer.

    The paper adds s * V with V the feature embeddings. Normalising makes
    s mean the same thing whatever an SAE family's decoder norms are.
    """
    return {
        layer: _unit(decoders[layer][feature])
        for layer, feature, _ in chain
    }
