"""Selection rules for the ablation variants.

The ablation's whole argument is that variants differ ONLY in which
edges they select, so these properties are what make the comparison
mean anything: same budget, never empty, and the anti rule actually
picking anti-aligned edges.
"""
from __future__ import annotations

import pytest

from circuitsteer.selection import select_edges

SIM = 0.10


def pick(pool, cosines, mode, k, seed=0):
    return select_edges(pool, cosines, mode=mode, k=k, sim_thresh=SIM,
                        seed=seed)


@pytest.fixture
def pool_and_cosines():
    # 100 edges, cosine spread over [-1, 1), contrastive diff descending.
    pool, cosines = [], {}
    for i in range(100):
        edge = (f"L0_{i}", f"L1_{i}")
        pool.append((edge, 1.0 - i / 100))
        cosines[edge] = -1.0 + 2.0 * i / 100
    return pool, cosines


def test_all_modes_return_the_same_budget(pool_and_cosines):
    pool, cosines = pool_and_cosines
    sizes = {
        mode: len(pick(pool, cosines, mode, 20))
        for mode in ("aligned", "none", "anti", "random")
    }
    assert set(sizes.values()) == {20}, sizes


def test_anti_selects_the_most_anti_aligned_and_is_never_empty(
    pool_and_cosines,
):
    pool, cosines = pool_and_cosines
    chosen = pick(pool, cosines, "anti", 10)
    assert len(chosen) == 10
    picked = [cosines[edge] for edge, _ in chosen]
    assert max(picked) < 0, picked
    # and they are the ten lowest cosines in the pool
    assert sorted(picked) == sorted(cosines.values())[:10]


def test_aligned_selects_only_aligned_edges(pool_and_cosines):
    pool, cosines = pool_and_cosines
    chosen = pick(pool, cosines, "aligned", 20)
    assert all(cosines[edge] > SIM for edge, _ in chosen)


def test_anti_is_non_empty_where_the_legacy_threshold_is_not():
    """The failure that made one ablation cell vacuous.

    With every cosine above -sim_thresh the legacy rule selects nothing
    and the run applies no steering at all, so its zero effect says
    nothing about anti-alignment. The bottom-k rule still returns a
    circuit.
    """
    pool = [((f"L0_{i}", f"L1_{i}"), 1.0) for i in range(50)]
    cosines = {edge: 0.5 for edge, _ in pool}

    assert pick(pool, cosines, "negative", 10) == []
    assert len(pick(pool, cosines, "anti", 10)) == 10


def test_random_is_seeded_and_reproducible(pool_and_cosines):
    pool, cosines = pool_and_cosines
    a = pick(pool, cosines, "random", 15, seed=1)
    b = pick(pool, cosines, "random", 15, seed=1)
    c = pick(pool, cosines, "random", 15, seed=2)
    assert a == b
    assert a != c


def test_selection_never_exceeds_a_small_pool():
    pool = [((f"L0_{i}", f"L1_{i}"), 1.0) for i in range(3)]
    cosines = {edge: 0.9 for edge, _ in pool}
    assert len(pick(pool, cosines, "aligned", 60)) == 3


def test_config_from_args_survives_an_unset_select_mode():
    """`--select-mode` is optional, so its value is None when unused.

    getattr returns that None rather than its fallback, which made
    CircuitConfig reject every run that did not pass the flag - including
    all the baselines, which never do.
    """
    import argparse

    from experiments.common import config_from_args

    args = argparse.Namespace(
        sim_thresh=0.10, act_thresh=1.5, diff_thresh=0.05, top_k=30,
        steer_pool="sum", align_mode=None, select_mode=None,
        n_steer_edges=None, select_seed=None,
    )
    config = config_from_args(args)
    assert config.select_mode == "aligned"
    assert config.align_mode == "positive"
    assert config.select_seed == 0
