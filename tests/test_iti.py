"""ITI's probing, head selection and head-level hook.

The failure modes worth guarding are the silent ones: a shift landing on
the wrong heads, a direction pointing the wrong way (which would make a
negative coefficient amplify the behaviour it is meant to suppress), and
a hook attached to the residual stream instead of head outputs. All
three would still run, and still produce plausible-looking numbers.

The probe maths is tested through `probes.py`, which is pure numpy, so
these run without a model or a GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.iti.probes import fit_head_probes

N_LAYERS, N_HEADS, D_HEAD = 3, 4, 8
SIGNAL = {(0, 0), (1, 2)}


def make_activations(n=40, seed=0, gap=3.0):
    """Two classes separable at exactly the heads in SIGNAL."""
    rng = np.random.default_rng(seed)
    shape = (n, N_LAYERS, N_HEADS, D_HEAD)
    control = rng.normal(size=shape) * 0.1
    target = rng.normal(size=shape) * 0.1
    for layer, head in SIGNAL:
        target[:, layer, head, 0] += gap
    return target, control


def test_selects_the_separable_heads():
    target, control = make_activations()
    selected, _, scores = fit_head_probes(target, control, n_select=2)
    assert set(selected) == SIGNAL
    for layer, head in SIGNAL:
        assert scores[layer, head] > 0.9


def test_shift_is_zero_for_unselected_heads():
    target, control = make_activations()
    selected, shifts, _ = fit_head_probes(target, control, n_select=2)
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            norm = np.linalg.norm(shifts[layer, head])
            if (layer, head) in SIGNAL:
                assert norm > 0, f"selected head {(layer, head)} not shifted"
            else:
                assert norm == 0, f"unselected head {(layer, head)} shifted"


def test_direction_points_towards_the_target_class():
    """A negative coefficient must move activations away from the target.

    The separable heads were built with the target class displaced along
    +dim0, so the fitted direction has to be +dim0. A flipped sign would
    make the shared negative-lambda sweep amplify the behaviour.
    """
    target, control = make_activations()
    _, shifts, _ = fit_head_probes(target, control, n_select=2)
    for layer, head in SIGNAL:
        assert shifts[layer, head, 0] > 0


def test_shift_magnitude_is_sigma_along_the_direction():
    target, control = make_activations()
    _, shifts, _ = fit_head_probes(target, control, n_select=2)
    features = np.concatenate([target, control], axis=0)
    for layer, head in SIGNAL:
        theta = shifts[layer, head]
        unit = theta / np.linalg.norm(theta)
        expected = np.std(features[:, layer, head, :] @ unit)
        assert np.isclose(np.linalg.norm(theta), expected, rtol=1e-6)


def test_probe_direction_variant_also_finds_the_heads():
    target, control = make_activations()
    selected, shifts, _ = fit_head_probes(
        target, control, n_select=2, direction="probe"
    )
    assert set(selected) == SIGNAL
    for layer, head in SIGNAL:
        assert np.linalg.norm(shifts[layer, head]) > 0


def test_n_select_is_capped_at_the_number_of_heads():
    target, control = make_activations()
    selected, _, _ = fit_head_probes(target, control, n_select=10_000)
    assert len(selected) == N_LAYERS * N_HEADS


def test_rejects_bad_direction_and_mismatched_shapes():
    target, control = make_activations()
    with pytest.raises(ValueError):
        fit_head_probes(target, control, n_select=2, direction="bogus")
    with pytest.raises(ValueError):
        fit_head_probes(target, control[:5], n_select=2)


def _hook_fn(shifts, coefficient, layer):
    """Mirror of ITISteerer.hooks' inner function, torch only."""
    scaled = torch.tensor(shifts) * coefficient

    def hook(z, hook=None, **kwargs):
        del kwargs
        return z + scaled[layer].to(z.dtype)

    return hook


def test_hook_adds_scaled_shift_to_the_right_heads_only():
    target, control = make_activations()
    _, shifts, _ = fit_head_probes(target, control, n_select=2)
    layer, head = sorted(SIGNAL)[0]

    z = torch.zeros(2, 5, N_HEADS, D_HEAD, dtype=torch.float64)
    out = _hook_fn(shifts, -2.0, layer)(z)

    expected = torch.tensor(shifts[layer, head]) * -2.0
    assert torch.allclose(out[0, 0, head], expected, atol=1e-8)
    untouched = next(h for h in range(N_HEADS) if (layer, h) not in SIGNAL)
    assert torch.allclose(
        out[0, 0, untouched], torch.zeros(D_HEAD, dtype=torch.float64)
    )
    # every position in the sequence is shifted: ITI applies at each token
    assert torch.allclose(out[1, 4, head], expected, atol=1e-8)


def test_hook_names_target_head_outputs_not_the_residual_stream():
    """Guards the one line that decides where the intervention lands.

    Read from source rather than imported: importing the steerer pulls in
    sae_lens and transformer_lens, which this test does not need and
    which are absent on a machine without a GPU environment.
    """
    source = Path(__file__).resolve().parent.parent / "baselines/iti/iti.py"
    body = source.read_text()
    hook_section = body[body.index("def hooks("):]
    assert "attn.hook_z" in hook_section
    # The prose mentions the residual stream to explain the override;
    # what matters is that no residual hook is actually registered.
    assert "hook_resid" not in hook_section
