"""How steering edges are chosen from the candidate pool.

This is the ablation's independent variable, kept free of torch and
sae_lens so the rule can be read - and tested - on its own.

Every rule draws from the same pool (co-activating feature pairs that
survived the contrastive filter) and returns the same number of edges.
That is what lets a difference between variants be attributed to the
selection rule rather than to one variant steering with more features
than another.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

Edge = tuple[str, str]
ScoredEdge = tuple[Edge, float]

MODES = ("aligned", "none", "anti", "random", "negative")


def select_edges(
    pool: Sequence[ScoredEdge],
    cosines: dict[Edge, float],
    mode: str,
    k: int,
    sim_thresh: float,
    seed: int = 0,
) -> list[ScoredEdge]:
    """Return at most `k` edges from `pool` under `mode`.

    `pool` is ordered by contrastive difference, strongest first.

    "anti" is a bottom-k rule rather than a threshold on purpose. A
    threshold at cos < -sim_thresh can match nothing, and a circuit with
    no edges applies no steering at all, so the resulting zero effect is
    an artefact rather than evidence about anti-aligned features.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if mode == "aligned":
        chosen = [e for e in pool if cosines.get(e[0], 0.0) > sim_thresh]
    elif mode == "negative":
        # Legacy threshold, kept only to reproduce earlier runs.
        chosen = [e for e in pool if cosines.get(e[0], 0.0) < -sim_thresh]
    elif mode == "none":
        chosen = list(pool)
    elif mode == "anti":
        chosen = sorted(pool, key=lambda e: cosines.get(e[0], 0.0))
    else:  # random
        chosen = list(pool)
        random.Random(seed).shuffle(chosen)

    return chosen[:k]
