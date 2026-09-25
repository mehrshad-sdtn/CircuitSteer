"""Name -> baseline class. Imports are lazy so that running the method
never pulls in baseline dependencies, and vice versa.
"""
from __future__ import annotations

# name -> (module path, class name)
_BASELINES: dict[str, tuple[str, str]] = {
    "caa": ("baselines.caa.caa", "CAASteerer"),
    "caa_1l": ("baselines.caa.caa", "CAASingleLayerSteerer"),
    "caa_top3": ("baselines.caa.caa", "CAATop3LayersSteerer"),
    "repe": ("baselines.repe.repe", "RepESteerer"),
    "prompt": ("baselines.prompting.prompting", "PromptSteerer"),
    "iti": ("baselines.iti.iti", "ITISteerer"),
    "spare": ("baselines.spare.spare", "SpARESteerer"),
    "saessv": ("baselines.saessv.saessv", "SAESSVSteerer"),
    "loreft": ("baselines.loreft.loreft", "LoReFTSteerer"),
    "featureflow": ("baselines.featureflow.featureflow", "FeatureFlowSteerer"),
}


def available() -> list[str]:
    return sorted(_BASELINES)


def get_baseline(name: str):
    if name not in _BASELINES:
        raise ValueError(
            f"Unknown baseline {name!r}; available: {available()}"
        )
    module_path, class_name = _BASELINES[name]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def register(name: str, module_path: str, class_name: str) -> None:
    """Add a baseline. Each new baseline appends one line here."""
    _BASELINES[name] = (module_path, class_name)
