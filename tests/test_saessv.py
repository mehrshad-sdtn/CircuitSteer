"""SAE-SSV stage-1 subspace selection, checked piece by piece.

Reference: He et al., arXiv:2505.16188, and SAESTEER/extractor.py in
github.com/Ineedanamehere/SAE-SSV.

These guard the failures that would still yield a plausible-looking
subspace: an F-statistic that ignores within-class variance, a coarse
stage that keeps dead SAE dimensions, a concept vector built from one
probe rather than fifty, and a fine stage that always returns the
largest candidate instead of the smallest adequate one.
"""
from __future__ import annotations

import numpy as np
import pytest

from baselines.saessv.subspace import (
    coarse_subspace, concept_vector, distance_loss, f_statistics,
    fine_subspace, initial_vector,
)

N, D = 60, 40
SIGNAL = [5, 11, 23]
DEAD = [0, 1, 2]


def make_latents(seed=0, gap=3.0):
    """Class 1 activates SIGNAL; DEAD dimensions are zero throughout."""
    rng = np.random.default_rng(seed)
    latents = np.abs(rng.normal(scale=0.2, size=(2 * N, D))).astype(np.float32)
    labels = np.concatenate([np.zeros(N), np.ones(N)])
    latents[N:, SIGNAL] += gap
    latents[:, DEAD] = 0.0
    return latents, labels


def test_f_statistic_ranks_the_separating_dimensions_first():
    latents, labels = make_latents()
    scores = f_statistics(latents, labels)
    assert set(np.argsort(-scores)[:len(SIGNAL)].tolist()) == set(SIGNAL)


def test_f_statistic_penalises_within_class_variance():
    """Two dimensions with the same mean gap rank by their spread."""
    rng = np.random.default_rng(1)
    latents = np.zeros((2 * N, 2), dtype=np.float32)
    labels = np.concatenate([np.zeros(N), np.ones(N)])
    latents[:, 0] = rng.normal(scale=0.1, size=2 * N)
    latents[:, 1] = rng.normal(scale=3.0, size=2 * N)
    latents[N:, :] += 1.0                       # identical mean shift
    scores = f_statistics(latents, labels)
    assert scores[0] > scores[1]


def test_coarse_subspace_skips_dead_dimensions():
    latents, labels = make_latents()
    dims = coarse_subspace(latents, labels, top_k=10)
    assert len(dims) == 10
    assert not (set(dims.tolist()) & set(DEAD)), "kept an all-zero dimension"
    assert set(SIGNAL).issubset(set(dims.tolist()))


def test_coarse_subspace_is_capped_by_the_available_dimensions():
    latents, labels = make_latents()
    dims = coarse_subspace(latents, labels, top_k=10_000)
    assert len(dims) == D - len(DEAD)


def test_concept_vector_weights_the_signal_dimensions_highest():
    latents, labels = make_latents()
    dims = coarse_subspace(latents, labels, top_k=12)
    weights = concept_vector(latents[:, dims], labels, n_probes=10)
    ranked = [int(dims[i]) for i in np.argsort(-np.abs(weights))]
    assert set(ranked[:len(SIGNAL)]) == set(SIGNAL)


def test_concept_vector_averages_over_probes():
    """Fifty probes on resampled halves must not equal a single probe."""
    latents, labels = make_latents()
    dims = coarse_subspace(latents, labels, top_k=12)
    one = concept_vector(latents[:, dims], labels, n_probes=1, seed=7)
    many = concept_vector(latents[:, dims], labels, n_probes=50, seed=7)
    assert not np.allclose(one, many)


def test_fine_subspace_returns_the_smallest_adequate_subspace():
    latents, labels = make_latents()
    dims = coarse_subspace(latents, labels, top_k=32)
    weights = concept_vector(latents[:, dims], labels, n_probes=10)
    local, separation = fine_subspace(
        latents[:, dims], labels, weights, candidates=(4, 8, 16, 32)
    )
    assert separation > 0
    assert len(local) <= 32
    # the three real dimensions are enough, so it must not take all 32
    assert len(local) < 32


def test_initial_vector_is_a_unit_direction_on_the_subspace():
    rng = np.random.default_rng(2)
    source = rng.normal(size=D).astype(np.float32)
    target = rng.normal(size=D).astype(np.float32)
    dims = np.array(SIGNAL)
    vector = initial_vector(source, target, dims, D)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)
    off = np.setdiff1d(np.arange(D), dims)
    assert np.allclose(vector[off], 0.0), "leaked outside the subspace"
    # points from source towards target
    assert float(vector[dims] @ (target - source)[dims]) > 0


def test_distance_loss_keeps_the_reference_half_weight():
    """The reference halves the source term; the paper writes it symmetric."""
    steered = np.array([1.0, 0.0])
    target = np.array([0.0, 0.0])
    source = np.array([2.0, 0.0])
    got = distance_loss(steered, target, source)
    assert np.isclose(got, 1.0 - 0.5 * 1.0)
    assert not np.isclose(got, 1.0 - 1.0), "symmetric form is not the reference"


def test_rejects_degenerate_inputs():
    latents, labels = make_latents()
    with pytest.raises(ValueError):
        f_statistics(latents, labels[:5])
    with pytest.raises(ValueError):
        f_statistics(latents, np.zeros(2 * N))      # one class only
    with pytest.raises(ValueError):
        coarse_subspace(np.zeros((10, 5)), np.r_[np.zeros(5), np.ones(5)])


def test_hook_round_trips_through_the_sae_at_the_last_token():
    """Read from source: importing the steerer needs sae_lens."""
    from pathlib import Path

    body = Path("baselines/saessv/saessv.py").read_text()
    section = body[body.index("def hooks("):]
    assert "sae.encode" in section and "sae.decode" in section
    assert "act[:, -1, :]" in section
    assert "hook_resid_post" in section
