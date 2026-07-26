from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    sae_release: str
    sae_layers: tuple[int, ...]
    sae_id_format: str


MODEL_CONFIGS = {
    "gemma": ModelConfig(
        model_name="google/gemma-2-2b",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_layers=(6, 12, 18, 24),
        sae_id_format="layer_{}/width_16k/canonical",
    ),
    "llama": ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        sae_release="llama_scope_lxr_8x",
        sae_layers=(3, 8, 16, 24, 29),
        sae_id_format="l{}r_8x",
    ),
}


@dataclass(frozen=True)
class CircuitConfig:
    sim_thresh: float = 0.10
    act_thresh: float = 1.5
    diff_thresh: float = 0.05
    top_k: int = 30
    steer_pool: str = "sum"

    def __post_init__(self) -> None:
        if self.steer_pool not in {"mean", "sum"}:
            raise ValueError("steer_pool must be 'mean' or 'sum'")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


CIRCUIT_DEFAULTS = CircuitConfig()
DEFAULT_COEFFICIENTS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
GLOBAL_SEED = 42
