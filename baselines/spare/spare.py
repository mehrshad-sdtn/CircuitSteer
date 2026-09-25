"""SpARE / SRPS: sparse-autoencoder steering from contrastive activations.

Reference: Wang, Shu, Wang, Ma and Du, "Improving LLM Reasoning through
Interpretable Role-Playing Steering", arXiv:2506.07335 (EMNLP 2025). The
paper reports no public code release, so this follows its equations.

The method, from the paper's section 3:

* run contrastive pairs through one SAE and take the last token's
  feature activations, a+ with the behaviour-bearing prompt and a-
  without;
* score every feature by how much its activation shifts, in both
  strength and frequency: I = mu + beta * delta (equations 1-3);
* keep the top k features and build one steering vector as their
  activation-weighted decoder sum, s = sum_i alpha_i * W_dec[i] (4);
* at inference add lambda * s to the residual stream at the LAST TOKEN
  of layer l (5), then rescale so the residual keeps its original norm
  (6).

Paper parameters used here: k = 15; layer 25 for both models; the
Gemma-Scope 65k SAE for Gemma-2-2B and the Llama-Scope 131k SAE for
Llama-3.1-8B, which are what the paper specifies.

Two departures, both recorded because they change how the number reads:

1. beta and theta are called tunable and their values are not reported.
   Defaults here are beta = 1 and theta = 0; see `selection.py`.
2. The paper steers towards a desired behaviour with a positive lambda.
   Here the contrast points towards the TARGET behaviour, matching CAA,
   RepE and ITI in this repository, so a negative coefficient suppresses
   and the shared lambda sweep keeps its meaning across methods.

This is the only baseline that needs an SAE, and a different one from
the method's: `requires_saes` stays False so the harness does not load
CircuitSteer's, and `fit` loads the paper's SAE itself and frees it
afterwards.
"""
from __future__ import annotations

import gc

import numpy as np
import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer
from baselines.spare.selection import (
    select_features, shift_last_token, steering_vector,
)

#: The paper's SAE per model: layer 25, largest available width.
SAE_CHOICE = {
    "gemma": ("gemma-scope-2b-pt-res-canonical",
              "layer_25/width_65k/canonical", 25),
    "llama": ("llama_scope_lxr_32x", "l25r_32x", 25),
}


class SpARESteerer(BaselineSteerer):
    name = "SpARE"
    requires_saes = False           # it loads its own, not CircuitSteer's

    #: Paper: "we choose the top 15 SAE latent features".
    top_k: int = 15

    #: Unreported in the paper; see selection.sensitivity.
    beta: float = 1.0
    theta: float = 0.0

    @property
    def steer_layers(self) -> tuple[int, ...]:
        return (SAE_CHOICE[self.model_key][2],)

    def _load_sae(self):
        from sae_lens import SAE

        release, sae_id, layer = SAE_CHOICE[self.model_key]
        print(f"{self.name}: loading {release} / {sae_id} (layer {layer})",
              flush=True)
        sae = SAE.from_pretrained(release, sae_id, device=self.device)
        sae = sae[0] if isinstance(sae, tuple) else sae
        return sae.to(self.device), layer

    @torch.no_grad()
    def _activations(self, sae, layer: int, texts: list[str]) -> np.ndarray:
        """Last-token SAE feature activations: [n_texts, n_features]."""
        name = f"blocks.{layer}.hook_resid_post"
        out = []
        for text in tqdm(texts, desc=f"{self.name} encoding", leave=False):
            _, cache = self.model.run_with_cache(
                text, names_filter=lambda n: n == name,
                stop_at_layer=layer + 1,
            )
            residual = cache[name][:, -1, :]
            out.append(sae.encode(residual)[0].float().cpu().numpy())
        return np.stack(out)

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        n = min(len(toxic), len(benign))
        if n == 0:
            raise ValueError("SpARE needs paired target/benign prompts")
        toxic, benign = toxic[:n], benign[:n]

        sae, layer = self._load_sae()
        positive = self._activations(sae, layer, toxic)
        negative = self._activations(sae, layer, benign)
        print(f"{self.name}: {n} pairs, {positive.shape[1]} SAE features "
              f"at layer {layer}")

        indices, alpha, score = select_features(
            positive, negative, k=self.top_k,
            beta=self.beta, theta=self.theta,
        )
        decoder = sae.W_dec.detach().float().cpu().numpy()
        shift = steering_vector(decoder, indices, alpha)

        self.selected_features = indices.tolist()
        self.feature_scores = score.tolist()
        self.steer_vecs = {
            layer: torch.tensor(shift, device=self.device, dtype=self.dtype)
        }
        print(
            f"{self.name}: top-{len(indices)} features {indices[:8].tolist()}"
            f"{'...' if len(indices) > 8 else ''}, "
            f"sensitivity {score.min():.3f}-{score.max():.3f}, "
            f"||s||={np.linalg.norm(shift):.3f}"
        )

        del sae
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def hooks(self, coefficient, vectors=None, layers=None):
        """Equations 5 and 6: shift the last token, then restore its norm.

        Overrides the shared hook for two reasons the paper is explicit
        about. The shift lands on the last token only, not every
        position, and the residual is rescaled to its original norm
        afterwards - without which the stream grows without bound as the
        coefficient rises.
        """
        del vectors, layers
        source = getattr(self, "steer_vecs", None)
        if not source or coefficient == 0.0:
            return []
        scaled = {l: v * coefficient for l, v in source.items()}

        def hook(residual, hook=None, **kwargs):
            del kwargs
            layer = hook.layer()
            if layer not in scaled:
                return residual
            return shift_last_token(residual, scaled[layer], 1.0)

        return [(f"blocks.{layer}.hook_resid_post", hook) for layer in scaled]
