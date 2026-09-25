"""SpARE / SRPS: each of the paper's equations, checked on its own.

Reference: Wang et al., arXiv:2506.07335, equations 1-6.

The properties worth guarding are the ones that would still produce
plausible numbers if they were wrong: a sensitivity score that ignored
the frequency term, a steering vector that dropped the activation
weights, a shift applied to every token instead of the last, and a
missing renormalisation - which would only show up as the coefficient
grew.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.spare.selection import (
    apply_shift, select_features, sensitivity, steering_vector,
)

N, F = 40, 12
SIGNAL = [3, 7]


def make_activations(seed=0, gap=2.0):
    """Positive samples activate SIGNAL features more, and more often."""
    rng = np.random.default_rng(seed)
    negative = np.abs(rng.normal(scale=0.05, size=(N, F)))
    positive = np.abs(rng.normal(scale=0.05, size=(N, F)))
    positive[:, SIGNAL] += gap
    negative[:, SIGNAL] = 0.0          # never active without the prompt
    return positive, negative


def test_mu_is_the_mean_activation_difference():
    """Equation 1."""
    positive, negative = make_activations()
    _, mu, _ = sensitivity(positive, negative)
    assert np.allclose(mu, (positive - negative).mean(axis=0))


def test_delta_is_the_activation_frequency_difference():
    """Equation 2, at threshold theta."""
    positive, negative = make_activations()
    theta = 0.5
    _, _, delta = sensitivity(positive, negative, theta=theta)
    expected = ((positive > theta).mean(axis=0)
                - (negative > theta).mean(axis=0))
    assert np.allclose(delta, expected)
    # the signal features fire every time with the prompt and never without
    for i in SIGNAL:
        assert np.isclose(delta[i], 1.0)


def test_score_combines_strength_and_frequency_with_beta():
    """Equation 3: beta must actually weight the frequency term."""
    positive, negative = make_activations()
    score_a, mu, delta = sensitivity(positive, negative, beta=0.0)
    assert np.allclose(score_a, mu), "beta=0 must reduce to mu alone"
    score_b, _, _ = sensitivity(positive, negative, beta=3.0)
    assert np.allclose(score_b, mu + 3.0 * delta)
    assert not np.allclose(score_a, score_b)


def test_selects_the_contrastive_features():
    positive, negative = make_activations()
    indices, alpha, score = select_features(positive, negative, k=2)
    assert sorted(indices.tolist()) == SIGNAL
    # alpha is the mean activation over positive samples, per equation 4
    assert np.allclose(alpha, positive[:, indices].mean(axis=0))


def test_steering_vector_is_the_activation_weighted_decoder_sum():
    """Equation 4."""
    rng = np.random.default_rng(1)
    decoder = rng.normal(size=(F, 6))
    indices = np.array(SIGNAL)
    alpha = np.array([2.0, 3.0])
    got = steering_vector(decoder, indices, alpha)
    expected = 2.0 * decoder[3] + 3.0 * decoder[7]
    assert np.allclose(got, expected)
    # dropping the weights would give a different vector
    assert not np.allclose(got, decoder[indices].sum(axis=0))


def test_shift_preserves_the_residual_norm():
    """Equation 6 - the property that keeps large coefficients stable."""
    rng = np.random.default_rng(2)
    residual = rng.normal(size=(4, 6))
    shift = rng.normal(size=6)
    for coefficient in (-1.0, -8.0, -40.0):
        out = apply_shift(residual, shift, coefficient)
        assert np.allclose(np.linalg.norm(out, axis=-1),
                           np.linalg.norm(residual, axis=-1), atol=1e-8)


def test_shift_still_changes_direction():
    """Norm preservation must not make the intervention a no-op."""
    rng = np.random.default_rng(3)
    residual = rng.normal(size=(1, 6))
    shift = rng.normal(size=6)
    out = apply_shift(residual, shift, -4.0)
    cosine = float(np.squeeze(out @ residual.T) /
                   (np.linalg.norm(out) * np.linalg.norm(residual)))
    assert cosine < 0.999


def test_zero_coefficient_is_a_no_op():
    rng = np.random.default_rng(4)
    residual = rng.normal(size=(3, 6))
    shift = rng.normal(size=6)
    assert np.allclose(apply_shift(residual, shift, 0.0), residual)


def test_rejects_mismatched_inputs():
    positive, negative = make_activations()
    with pytest.raises(ValueError):
        sensitivity(positive, negative[:5])
    with pytest.raises(ValueError):
        steering_vector(np.zeros((F, 6)), np.array([1, 2]), np.array([1.0]))


def test_hook_shifts_only_the_last_token_and_keeps_its_norm():
    """Equation 5 applies at the last token, not across the sequence."""
    from baselines.spare.selection import shift_last_token

    shift = torch.ones(6, dtype=torch.float64) * 0.4
    residual = torch.randn(2, 5, 6, dtype=torch.float64)
    out = shift_last_token(residual, shift, -3.0)

    # earlier positions untouched
    assert torch.allclose(out[:, :-1, :], residual[:, :-1, :])
    # last position moved, but with its norm preserved
    assert not torch.allclose(out[:, -1, :], residual[:, -1, :])
    assert torch.allclose(out[:, -1, :].norm(dim=-1),
                          residual[:, -1, :].norm(dim=-1), atol=1e-8)


def test_hook_is_a_no_op_at_zero_coefficient():
    from baselines.spare.selection import shift_last_token

    residual = torch.randn(2, 5, 6, dtype=torch.float64)
    out = shift_last_token(residual, torch.ones(6, dtype=torch.float64), 0.0)
    assert torch.allclose(out, residual)


def test_hook_registers_on_the_residual_stream_at_the_paper_layer():
    """Read from source: importing the steerer needs sae_lens."""
    from pathlib import Path as _P

    body = _P("baselines/spare/spare.py").read_text()
    section = body[body.index("def hooks("):]
    assert "hook_resid_post" in section
    assert "shift_last_token" in section
