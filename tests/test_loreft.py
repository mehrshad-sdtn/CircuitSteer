"""LoReFT's intervention, checked against the equation it claims.

Reference: Wu et al., arXiv:2404.03592; pyreft/interventions.py.

    LoReFT(h) = h + R^T ( W h + b - R h )

The properties worth guarding are the ones that would still train and
still produce plausible generations if they were broken: a rotation that
drifts off the orthonormal manifold during optimisation, an edit that
leaks outside the rank-r subspace, and a scale that does not actually
scale the edit.
"""
from __future__ import annotations

import pytest
import torch

from baselines.loreft.intervention import LoReFT

D, R = 32, 4


def test_rotation_is_orthonormal_at_initialisation():
    module = LoReFT(D, rank=R)
    basis = module.basis                       # [d, r]
    gram = basis.T @ basis
    assert torch.allclose(gram, torch.eye(R), atol=1e-5), gram


def test_rotation_stays_orthonormal_after_optimisation():
    """The parametrisation must hold, not merely start, orthonormal.

    A penalty-based implementation passes the initialisation test and
    fails this one, and the "edit confined to a subspace" claim goes
    with it.
    """
    module = LoReFT(D, rank=R)
    optimiser = torch.optim.AdamW(module.parameters(), lr=0.1)
    hidden = torch.randn(8, D)
    for _ in range(20):
        loss = module(hidden).pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    gram = module.basis.T @ module.basis
    assert torch.allclose(gram, torch.eye(R), atol=1e-4), gram


def test_edit_stays_inside_the_rank_r_subspace():
    """The change must have no component orthogonal to R."""
    module = LoReFT(D, rank=R)
    hidden = torch.randn(6, D)
    delta = module.delta(hidden)
    basis = module.basis                       # [d, r], orthonormal columns
    # Remove the component inside the subspace; nothing should remain.
    inside = (delta @ basis) @ basis.T
    assert torch.allclose(delta, inside, atol=1e-4)
    assert int(torch.linalg.matrix_rank(delta, tol=1e-4)) <= R


def test_matches_the_published_equation():
    module = LoReFT(D, rank=R)
    hidden = torch.randn(5, D)
    basis = module.basis
    expected = hidden + (
        (module.source(hidden) - hidden @ basis) @ basis.T
    )
    assert torch.allclose(module(hidden), expected, atol=1e-6)


def test_scale_multiplies_the_edit():
    module = LoReFT(D, rank=R)
    hidden = torch.randn(4, D)
    delta = module.delta(hidden)
    assert torch.allclose(module(hidden, scale=0.0), hidden, atol=1e-6)
    assert torch.allclose(module(hidden, scale=2.0), hidden + 2 * delta,
                          atol=1e-5)
    assert torch.allclose(module(hidden, scale=-1.0), hidden - delta,
                          atol=1e-5)


def test_parameter_count_is_low_rank():
    """The selling point is parameter efficiency; hold it to that."""
    module = LoReFT(D, rank=R)
    # R is d*r, W is d*r, b is r
    assert module.parameter_count() == D * R + D * R + R
    assert module.parameter_count() < D * D


def test_rejects_invalid_rank():
    with pytest.raises(ValueError):
        LoReFT(D, rank=0)
    with pytest.raises(ValueError):
        LoReFT(D, rank=D + 1)


def test_preserves_input_dtype():
    module = LoReFT(D, rank=R, dtype=torch.float32)
    hidden = torch.randn(3, D, dtype=torch.float16)
    assert module(hidden).dtype == torch.float16


def test_hook_uses_the_coefficient_magnitude():
    """A trained edit has no direction to flip, so |lambda| is used.

    Read from source: importing the steerer needs transformer_lens.
    """
    from pathlib import Path

    body = Path("baselines/loreft/loreft.py").read_text()
    section = body[body.index("def hooks("):]
    assert "abs(float(coefficient))" in section
    assert "hook_resid_post" in body


def test_applies_at_every_position_not_just_the_last():
    """Guards the failure that made SpARE's Sycophancy cells vacuous."""
    from pathlib import Path

    body = Path("baselines/loreft/loreft.py").read_text()
    section = body[body.index("def _hooks_for("):body.index("def fit(")]
    assert "[:, -1, :]" not in section, "edit restricted to the last token"
