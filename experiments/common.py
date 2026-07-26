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
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--steer-pool",
        choices=("sum", "mean"),
        default="sum",
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
        diff_thresh=args.diff_thresh,
        top_k=args.top_k,
        steer_pool=args.steer_pool,
    )


def release_model(steerer) -> None:
    steerer.model.reset_hooks()
    steerer.saes.clear()
    del steerer.model
    del steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
