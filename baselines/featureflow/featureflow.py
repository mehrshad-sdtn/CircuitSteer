"""Feature Flow: multi-layer steering along a data-free SAE flow graph.

Reference: Laptev, Balagansky, Aksenov and Gavrilov, "Analyze Feature
Flow to Enhance Interpretation and Steering in Language Models",
arXiv:2502.03032, 2025. No code is released; this follows the paper.

The method, from the paper:

* link each feature to the next layer's feature with the most similar
  decoder direction, without data (equation 1 in `flow.py`), which
  yields a chain following one concept through depth;
* steer with the paper's cumulative multi-layer "activation" rule,
  h <- h + s * V, adding the chain feature's direction at every chain
  layer and at every token.

Choices the paper leaves open, recorded because they change the number:

1. Seed feature. The paper chooses it by hand from Neuronpedia labels.
   Here it is the single most contrastive feature over the SAE layers
   on the training split (`flow.select_seed`), so the choice repeats
   across cells and seeds without a human in the loop.
2. Layers. The paper matches consecutive layers. Here the chain runs
   over CircuitSteer's SAE layers, the harness default, so both methods
   draw on the same SAEs and depths. Links therefore span several
   layers, and a chain may stop at the paper's 0.5 cutoff sooner than
   it would layer by layer; the traced chain is printed and kept on
   `self.chain` so its length can be reported.
3. Sign. The paper steers towards a theme with positive s. As with CAA,
   RepE, ITI and SpARE here, the seed points towards the TARGET
   behaviour, so a negative coefficient suppresses it and the shared
   lambda sweep keeps its meaning across methods.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer
from baselines.featureflow.flow import (
    SPAN_CUTOFF, select_seed, steering_vectors, trace_flow,
)


class FeatureFlowSteerer(BaselineSteerer):
    name = "FeatureFlow"
    requires_saes = True            # the harness's SAEs, at its layers

    #: The paper's residual similarity-span cutoff.
    cutoff: float = SPAN_CUTOFF

    #: A seed must fire on at least this share of target prompts.
    min_freq: float = 0.1

    @property
    def steer_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.steer_vecs)) or self.sae_layers

    @torch.no_grad()
    def _activations(self, texts: list[str]) -> dict[int, np.ndarray]:
        """Last-token SAE activations per layer: {layer: [n, features]}.

        Last token, as CircuitSteer's own circuit discovery and SpARE
        read them, so the seed is chosen from the same signal.
        """
        names = {f"blocks.{l}.hook_resid_post" for l in self.sae_layers}
        out: dict[int, list[np.ndarray]] = {l: [] for l in self.sae_layers}
        for text in tqdm(texts, desc=f"{self.name} encoding", leave=False):
            _, cache = self.model.run_with_cache(
                text, names_filter=lambda n: n in names,
                stop_at_layer=self.sae_layers[-1] + 1,
            )
            for layer in self.sae_layers:
                residual = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
                encoded = self.saes[layer].encode(residual)[0]
                out[layer].append(encoded.float().cpu().numpy())
        return {layer: np.stack(rows) for layer, rows in out.items()}

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        if not toxic or not benign:
            raise ValueError("FeatureFlow needs target and benign prompts")
        positive = self._activations(toxic)
        negative = self._activations(benign)
        layer, feature, score = select_seed(
            positive, negative, min_freq=self.min_freq,
        )
        decoders = {
            l: self.saes[l].W_dec.detach().float().cpu().numpy()
            for l in self.sae_layers
        }
        self.chain = trace_flow(layer, feature, decoders, cutoff=self.cutoff)
        vectors = steering_vectors(self.chain, decoders)
        self.steer_vecs = {
            l: torch.tensor(v, device=self.device, dtype=self.dtype)
            for l, v in vectors.items()
        }
        links = ", ".join(f"L{l}/{f} ({c:.2f})" for l, f, c in self.chain)
        print(
            f"{self.name}: seed L{layer}/{feature} (contrast {score:.2f}); "
            f"chain of {len(self.chain)}/{len(self.sae_layers)} layers: "
            f"{links}"
        )
