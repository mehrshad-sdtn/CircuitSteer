"""Shared harness so every baseline is measured exactly like CircuitSteer.

A baseline subclasses `BaselineSteerer` and implements one method:
`fit(toxic, benign)`, which must populate `self.steer_vecs` as
{layer: tensor}. Everything downstream — hook placement, batched
generation, perplexity, scoring, the val/test split, the benign control
and the statistics — is inherited unchanged from `CircuitSteer` and
`evaluation.evaluate_coefficient`.

That inheritance is the point: the comparison is only meaningful if the
sole difference between the method and a baseline is how the steering
vector is derived, not how it is applied or scored. Do not re-implement
generation or scoring inside a baseline.
"""
from __future__ import annotations

from circuitsteer.core import CircuitSteer


class BaselineSteerer(CircuitSteer):
    """CircuitSteer's evaluation path with a different vector derivation."""

    #: Name recorded in the `method` column of every emitted row.
    name: str = "Baseline"

    #: Most baselines need no sparse autoencoders. Skipping them saves
    #: several GB, which is what lets Llama-3.1-8B run on a 24GB card.
    requires_saes: bool = False

    def _load_saes(self):
        if self.requires_saes:
            return super()._load_saes()
        print(f"{self.name}: SAEs not required, skipping load")
        return {}

    @property
    def steer_layers(self) -> tuple[int, ...]:
        """Layers a baseline steers at.

        Defaults to the same layers CircuitSteer's SAEs sit on, so a
        baseline is not handed an unfair advantage or disadvantage purely
        from steering at different depths.
        """
        return self.sae_layers

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement fit() and set steer_vecs"
        )
