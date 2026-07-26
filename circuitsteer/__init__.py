"""CircuitSteer: SAE-based multi-layer circuit steering."""

from .config import CIRCUIT_DEFAULTS, MODEL_CONFIGS, CircuitConfig

__all__ = [
    "CIRCUIT_DEFAULTS",
    "MODEL_CONFIGS",
    "CircuitConfig",
    "CircuitSteer",
]


def __getattr__(name: str):
    """Load the GPU-heavy core only when it is requested."""
    if name == "CircuitSteer":
        from .core import CircuitSteer

        return CircuitSteer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
