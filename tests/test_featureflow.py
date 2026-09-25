"""Feature Flow: seed selection and data-free flow tracing, on arrays.

Reference: Laptev et al., arXiv:2502.03032.

The properties worth guarding are the ones that would still produce a
plausible chain if they were wrong: matching from the seed at every hop
instead of from the previous link, ignoring the span cutoff, a seed
chosen by raw rather than standardised contrast, and vectors whose norm
leaks the SAE family's decoder scale into the coefficient.
"""
from __future__ import annotations

import numpy as np

from baselines.featureflow.flow import (
    match, select_seed, standardized_contrast, steering_vectors, trace_flow,
)
from baselines.registry import _BASELINES, available

D, F = 16, 10


def unit(v):
    return v / np.linalg.norm(v)


def test_standardized_contrast_prefers_separation_over_raw_gap():
    """A large but noisy gap loses to a smaller, cleaner one."""
    rng = np.random.default_rng(0)
    neg = np.abs(rng.normal(0.1, 0.02, size=(200, 2)))
    pos = neg.copy()
    pos[:, 0] += rng.normal(5.0, 10.0, size=200).clip(0)   # big, noisy
    pos[:, 1] += 1.0                                       # small, clean
    score = standardized_contrast(pos, neg)
    assert score[1] > score[0]


def test_rarely_firing_features_are_excluded():
    pos = np.zeros((100, 2))
    neg = np.zeros((100, 2))
    pos[0, 0] = 50.0                    # fires on 1% of target prompts
    pos[:, 1] = 0.5
    score = standardized_contrast(pos, neg, min_freq=0.1)
    assert score[0] == -np.inf
    assert np.isfinite(score[1])


def test_seed_is_searched_over_every_layer():
    rng = np.random.default_rng(1)
    neg = {l: np.abs(rng.normal(0, 0.05, size=(50, F))) for l in (6, 12)}
    pos = {l: a.copy() for l, a in neg.items()}
    pos[12][:, 4] += 3.0
    assert select_seed(pos, neg)[:2] == (12, 4)


def test_match_is_the_most_cosine_similar_row_regardless_of_norm():
    rng = np.random.default_rng(2)
    direction = rng.normal(size=D)
    decoder = rng.normal(size=(F, D))
    decoder[7] = 0.01 * direction + 0.001 * rng.normal(size=D)   # tiny norm
    index, cosine = match(direction, decoder)
    assert index == 7 and cosine > 0.99


def _decoders_with_chain(cosines, seed=3):
    """Layers 0..n with a planted chain feature 2 whose successive links
    have the given cosines; every other row is random and nearly
    orthogonal in high dimension."""
    rng = np.random.default_rng(seed)
    dim = 256
    current = unit(rng.normal(size=dim))
    decoders = {0: rng.normal(size=(F, dim)) * 0.01}
    decoders[0][2] = current
    for layer, c in enumerate(cosines, start=1):
        noise = unit(rng.normal(size=dim))
        noise = unit(noise - noise @ current * current)
        nxt = c * current + np.sqrt(1 - c ** 2) * noise
        decoders[layer] = rng.normal(size=(F, dim)) * 0.01
        decoders[layer][2] = nxt
        current = nxt
    return decoders


def test_chain_follows_each_link_from_the_previous_feature():
    """Links are 0.9 apart, so layer 0 and layer 3 are only 0.9**3 alike;
    matching from the seed at every hop would still pass, but matching
    from the previous link is what the recorded cosines must show."""
    decoders = _decoders_with_chain([0.9, 0.9, 0.9])
    chain = trace_flow(0, 2, decoders, cutoff=0.5)
    assert [l for l, _, _ in chain] == [0, 1, 2, 3]
    assert all(f == 2 for _, f, _ in chain)
    assert [round(c, 2) for _, _, c in chain] == [1.0, 0.9, 0.9, 0.9]


def test_chain_stops_at_the_first_link_below_the_cutoff():
    decoders = _decoders_with_chain([0.9, 0.3, 0.9])
    chain = trace_flow(0, 2, decoders, cutoff=0.5)
    assert [l for l, _, _ in chain] == [0, 1]


def test_chain_is_traced_backwards_from_a_middle_seed():
    decoders = _decoders_with_chain([0.9, 0.9, 0.9])
    chain = trace_flow(2, 2, decoders, cutoff=0.5)
    assert [l for l, _, _ in chain] == [0, 1, 2, 3]


def test_steering_vectors_are_unit_norm():
    decoders = _decoders_with_chain([0.9])
    decoders = {l: d * 7.5 for l, d in decoders.items()}
    vectors = steering_vectors([(0, 2, 1.0), (1, 2, 0.9)], decoders)
    for v in vectors.values():
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_registered():
    # Checked without importing the class, which pulls in sae_lens.
    assert "featureflow" in available()
    assert _BASELINES["featureflow"] == (
        "baselines.featureflow.featureflow", "FeatureFlowSteerer",
    )
