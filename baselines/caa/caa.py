"""CAA - Contrastive Activation Addition.

Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition"
(arXiv:2312.06681). The steering vector at a layer is the difference of
mean residual-stream activations between the positive and negative
prompt sets, read at the final token position.

This is the natural baseline for CircuitSteer: identical hook sites and
identical application, but the direction comes from a raw activation
difference rather than from SAE feature-circuit discovery. An
implementation already existed inline in experiments/layer_count.py; it
is lifted here so both live behind one interface.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer


class CAASteerer(BaselineSteerer):
    name = "CAA"
    requires_saes = False

    @torch.no_grad()
    def _mean_residual(self, texts: list[str]) -> dict[int, torch.Tensor]:
        layers = list(self.steer_layers)
        totals: dict[int, torch.Tensor | None] = {l: None for l in layers}
        for text in tqdm(texts, desc=f"{self.name} activations", leave=False):
            _, cache = self.model.run_with_cache(
                text,
                stop_at_layer=layers[-1] + 1,
            )
            for layer in layers:
                residual = cache[
                    f"blocks.{layer}.hook_resid_post"
                ][:, -1, :][0].float()
                previous = totals[layer]
                totals[layer] = (
                    residual if previous is None else previous + residual
                )
        return {
            layer: value / max(len(texts), 1)
            for layer, value in totals.items()
        }

    #: How many layers to steer at. None means every candidate layer.
    n_layers: int | None = None

    @staticmethod
    def _layer_strength(
        difference: torch.Tensor,
        toxic_mean: torch.Tensor,
        benign_mean: torch.Tensor,
    ) -> float:
        """How strongly a layer separates the two prompt sets.

        The raw difference norm is not comparable across depth, because
        residual-stream norms grow substantially with layer index. Scaling
        by the average activation norm at that layer turns it into a
        relative effect size, so "top-k by activation" does not collapse
        into "the k deepest layers".
        """
        scale = 0.5 * (toxic_mean.norm() + benign_mean.norm())
        if float(scale) == 0.0:
            return 0.0
        return float(difference.norm() / scale)

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        candidates = list(self.steer_layers)
        print(
            f"{self.name}: contrasting {len(toxic)} vs {len(benign)} texts "
            f"over candidate layers {candidates}"
        )
        toxic_means = self._mean_residual(toxic)
        benign_means = self._mean_residual(benign)

        differences = {
            layer: toxic_means[layer] - benign_means[layer]
            for layer in candidates
        }
        strengths = {
            layer: self._layer_strength(
                differences[layer], toxic_means[layer], benign_means[layer]
            )
            for layer in candidates
        }
        ranked = sorted(candidates, key=lambda l: -strengths[l])
        print(
            f"{self.name}: layer strengths "
            + ", ".join(f"L{l}={strengths[l]:.4f}" for l in ranked)
        )

        chosen = ranked if self.n_layers is None else ranked[: self.n_layers]
        self.steer_vecs = {
            layer: differences[layer].to(self.dtype) for layer in chosen
        }
        print(
            f"{self.name}: steering at {sorted(self.steer_vecs)} "
            f"(of {len(candidates)} candidates)"
        )


class CAASingleLayerSteerer(CAASteerer):
    """CAA applied at the single strongest layer."""

    name = "CAA-1L"
    n_layers = 1


class CAATop3LayersSteerer(CAASteerer):
    """CAA applied at the three strongest layers."""

    name = "CAA-top3"
    n_layers = 3
