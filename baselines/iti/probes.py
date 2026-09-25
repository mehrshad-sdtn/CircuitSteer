"""ITI's probing and head selection, as plain arrays.

Kept free of torch, transformer_lens and sae_lens so the algorithm that
defines the baseline can be read and tested without a GPU or a model.
`iti.py` collects the activations and applies the result; everything the
paper specifies about *which* heads to move and *how far* lives here.

From Li et al. (2023), section 3:

* one logistic probe per attention head, fitted on the head's last-token
  output x_l^h (after Att, before the output projection);
* heads ranked by probe accuracy on a held-out split, top K kept;
* each kept head shifted by sigma_l^h * theta_l^h, where theta is a unit
  direction and sigma is the standard deviation of activations projected
  onto it, estimated over training and validation together.
"""
from __future__ import annotations

import numpy as np

DIRECTIONS = ("mass_mean", "probe")


def fit_head_probes(
    target: np.ndarray,
    control: np.ndarray,
    n_select: int,
    direction: str = "mass_mean",
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Return (selected heads, per-head shifts, per-head probe accuracy).

    `target` and `control` are [n, n_layers, n_heads, d_head] arrays of
    last-token head outputs. The returned shift array has the same
    layer/head/d_head shape and is exactly zero for every head outside
    the selected set, which is what makes the intervention sparse.

    Head ranking uses a split the probe was not fitted on. Ranking on
    training accuracy instead would push every head towards 1.0 on
    high-dimensional activations and make the ordering meaningless.
    """
    from sklearn.linear_model import LogisticRegression

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if target.shape != control.shape:
        raise ValueError("target and control must have the same shape")
    if target.size == 0:
        raise ValueError("no activations supplied")

    n, n_layers, n_heads, d_head = target.shape
    features = np.concatenate([target, control], axis=0)
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    cut = max(1, int(len(order) * (1 - val_fraction)))
    train_idx, val_idx = order[:cut], order[cut:]
    if len(val_idx) == 0:                      # tiny inputs: score on train
        val_idx = train_idx

    scores = np.zeros((n_layers, n_heads))
    weights = np.zeros((n_layers, n_heads, d_head))
    for layer in range(n_layers):
        for head in range(n_heads):
            x = features[:, layer, head, :]
            probe = LogisticRegression(max_iter=1000)
            probe.fit(x[train_idx], labels[train_idx])
            scores[layer, head] = probe.score(x[val_idx], labels[val_idx])
            weights[layer, head] = probe.coef_[0]

    n_select = min(n_select, n_layers * n_heads)
    flat = np.argsort(scores.ravel())[::-1][:n_select]
    selected = [(int(i // n_heads), int(i % n_heads)) for i in flat]

    shifts = np.zeros((n_layers, n_heads, d_head))
    for layer, head in selected:
        if direction == "probe":
            theta = weights[layer, head]
        else:
            theta = (target[:, layer, head, :].mean(0)
                     - control[:, layer, head, :].mean(0))
        norm = float(np.linalg.norm(theta))
        if norm < 1e-8:
            continue
        theta = theta / norm
        sigma = float(np.std(features[:, layer, head, :] @ theta))
        shifts[layer, head] = theta * sigma

    return selected, shifts, scores
