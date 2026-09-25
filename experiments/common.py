from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from circuitsteer.config import MODEL_CONFIGS, CircuitConfig


def add_circuit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CONFIGS,
        default=list(MODEL_CONFIGS),
    )
    parser.add_argument("--sim-thresh", type=float, default=0.10)
    parser.add_argument("--act-thresh", type=float, default=1.5)
    parser.add_argument("--diff-thresh", type=float, default=0.05)
    parser.add_argument(
        "--act-top-n",
        type=int,
        default=None,
        help="Keep the n most active features per layer instead of "
             "thresholding at --act-thresh, so pool density does not "
             "depend on the SAE family's activation scale.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--steer-pool",
        choices=("sum", "mean", "unit"),
        default="sum",
    )
    parser.add_argument(
        "--steer-nodes",
        choices=("source", "both"),
        default="source",
        help="'both' also steers each edge's destination feature, which "
             "is what puts the deepest SAE layer into the steering vector",
    )
    parser.add_argument(
        "--tie-break",
        choices=("name", "cosine"),
        default="name",
        help="Order of edges tied on contrastive specificity. 'cosine' "
             "prefers the more aligned edge, following the method's own "
             "criterion rather than the edge id.",
    )
    parser.add_argument(
        "--sae-layers",
        nargs="+",
        type=int,
        default=None,
        help="Restrict the circuit to a subset of the model's SAE layers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )


def config_from_args(args: argparse.Namespace) -> CircuitConfig:
    return CircuitConfig(
        sim_thresh=args.sim_thresh,
        act_thresh=args.act_thresh,
        act_top_n=getattr(args, "act_top_n", None),
        diff_thresh=args.diff_thresh,
        top_k=args.top_k,
        steer_pool=args.steer_pool,
        # Same defensive read as select_mode below: absent on parsers
        # that predate the flag, and None must not reach the config.
        steer_nodes=getattr(args, "steer_nodes", None) or "source",
        tie_break=getattr(args, "tie_break", None) or "name",
        sae_layers=(
            tuple(args.sae_layers)
            if getattr(args, "sae_layers", None) else None
        ),
        align_mode=getattr(args, "align_mode", None) or "positive",
        # `--select-mode` defaults to None so that leaving it off keeps
        # the legacy --align-mode path. getattr returns that None rather
        # than the fallback, since the attribute exists, so the `or` is
        # what stops every run without the flag from failing validation.
        select_mode=getattr(args, "select_mode", None) or "aligned",
        n_steer_edges=getattr(args, "n_steer_edges", None),
        select_seed=getattr(args, "select_seed", None) or 0,
    )


def release_model(steerer) -> None:
    steerer.model.reset_hooks()
    steerer.saes.clear()
    del steerer.model
    del steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
