"""SAE-SSV: supervised steering inside a sparse-representation subspace.

Reference: He, Jin, Shen, Payani, Zhang and Du, "SAE-SSV: Supervised
Steering in Sparse Representation Spaces for Reliable Control of
Language Models", arXiv:2505.16188 (EMNLP 2025); code at
github.com/Ineedanamehere/SAE-SSV.

The method has two stages:

Stage 1, subspace selection (`subspace.py`): encode last-token
activations with an SAE, rank dimensions by a binary ANOVA F-statistic,
keep the top 128, average 50 linear probes fitted on random halves into
one concept vector, and shrink to the smallest top-d subspace that keeps
most of the class separation.

Stage 2, vector learning (here): starting from the unit source->target
centroid direction, minimise

    L = lambda_dist * ( ||z' - mu+||^2 - 0.5 ||z' - mu-||^2 )
      + lambda_lm   * NLL(target continuation | steered state)
      + lambda_reg  * ||v||_1

over the subspace, where z' = encode(h) + v.

At inference the intervention is an SAE round trip, which is what the
reference does and what the paper's "h' = h + lambda v" understates:

    h_last <- decode( encode(h_last) + lambda * v )

so the last token's activation is REPLACED by a reconstruction. The SAE
reconstruction error therefore rides along with the steering, and a
coefficient of zero would not be a no-op - the harness treats zero as
"no hooks", so the unsteered baseline stays clean.

Deviation from the released code, approved and deliberate: it estimates
the language-model term's gradient by finite differences, one forward
pass per selected dimension, which costs about 130 forward passes per
sample and roughly 416k per unit. That term is exactly differentiable
through the SAE decoder and the model, so this uses autograd: the same
gradient, exact rather than an epsilon approximation, at one forward and
one backward. Everything else follows the reference.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer
from baselines.saessv.subspace import (
    coarse_subspace, concept_vector, fine_subspace, initial_vector,
)

#: Paper: LLaMA3.1-8B layer 16 with the 32k SAE; Gemma2-2B layers 13-16
#: with the 16k SAE. Layer 16 sits in both, so both use it.
SAE_CHOICE = {
    "gemma": ("gemma-scope-2b-pt-res-canonical",
              "layer_16/width_16k/canonical", 16),
    "llama": ("llama_scope_lxr_8x", "l16r_8x", 16),
}


class SAESSVSteerer(BaselineSteerer):
    name = "SAE-SSV"
    requires_saes = False           # loads its own, not CircuitSteer's

    # Stage 1, from the reference.
    coarse_k: int = 128
    n_probes: int = 50
    subset_fraction: float = 0.5
    separation_fraction: float = 0.95

    # Stage 2, from the reference's train() defaults.
    lambda_dist: float = 1.0
    lambda_lm: float = 0.5
    lambda_reg: float = 0.01
    learning_rate: float = 0.01
    max_iter: int = 100
    batch_size: int = 32
    #: Tokens of the target continuation scored by the language-model term.
    lm_horizon: int = 20

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
        return sae.to(self.device).float(), layer

    @torch.no_grad()
    def _last_activation(self, layer: int, text: str):
        name = f"blocks.{layer}.hook_resid_post"
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(
            tokens, names_filter=lambda n: n == name, stop_at_layer=layer + 1
        )
        return cache[name][0, -1, :].detach(), tokens

    @torch.no_grad()
    def _encode_all(self, sae, layer: int, texts: list[str]) -> np.ndarray:
        out = []
        for text in tqdm(texts, desc=f"{self.name} encoding", leave=False):
            activation, _ = self._last_activation(layer, text)
            out.append(
                sae.encode(activation.float().unsqueeze(0))[0]
                .detach().cpu().numpy()
            )
        return np.stack(out)

    def _lm_nll(self, sae, layer, vector, source_tokens, target_tokens,
                source_latent):
        """NLL of the target continuation under the steered state.

        Differentiable in `vector`: the decoded activation is spliced
        into the forward pass, so autograd carries the gradient back
        through the decoder.
        """
        steered = sae.decode((source_latent + vector).unsqueeze(0))[0]

        def hook(act, hook=None, **kwargs):
            del kwargs
            act = act.clone()
            act[0, -1, :] = steered.to(act.dtype)
            return act

        logits = self.model.run_with_hooks(
            source_tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)],
        )
        horizon = min(target_tokens.shape[1], self.lm_horizon)
        total, count = logits.new_zeros(()), 0
        for t in range(1, horizon):
            if t - 1 >= logits.shape[1]:
                break
            token = int(target_tokens[0, t])
            if token >= logits.shape[-1]:
                continue
            total = total - torch.log_softmax(
                logits[0, t - 1, :].float(), dim=-1
            )[token]
            count += 1
        return total / max(count, 1) if count else None

    def _stage2_budget(self) -> tuple[int, int]:
        """Iterations and batch size, overridable for smoke tests.

        Stage 2 is the expensive half, so a short run needs a way to cut
        it without editing the defaults the real run depends on.
        """
        iters = int(os.environ.get("SAESSV_MAX_ITER", self.max_iter))
        batch = int(os.environ.get("SAESSV_BATCH", self.batch_size))
        return max(1, iters), max(1, batch)

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        n = min(len(toxic), len(benign))
        if n == 0:
            raise ValueError("SAE-SSV needs paired target/benign prompts")
        toxic, benign = toxic[:n], benign[:n]

        sae, layer = self._load_sae()
        positive = self._encode_all(sae, layer, toxic)     # label 1
        negative = self._encode_all(sae, layer, benign)    # label 0
        latents = np.concatenate([negative, positive], axis=0)
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        d_sae = latents.shape[1]

        # Stage 1.
        coarse = coarse_subspace(latents, labels, top_k=self.coarse_k)
        weights = concept_vector(
            latents[:, coarse], labels, n_probes=self.n_probes,
            subset_fraction=self.subset_fraction,
        )
        local, separation = fine_subspace(
            latents[:, coarse], labels, weights,
            separation_fraction=self.separation_fraction,
        )
        dims = coarse[local]
        print(f"{self.name}: subspace {len(coarse)} -> {len(dims)} dims "
              f"(separation {separation:.3f}) of {d_sae}")

        # Stage 2.
        source_centroid = negative.mean(axis=0)
        target_centroid = positive.mean(axis=0)
        vector = torch.tensor(
            initial_vector(source_centroid, target_centroid, dims, d_sae),
            device=self.device, dtype=torch.float32, requires_grad=True,
        )
        mask = torch.zeros(d_sae, device=self.device, dtype=torch.bool)
        mask[torch.tensor(dims, device=self.device)] = True
        mu_pos = torch.tensor(target_centroid, device=self.device)
        mu_neg = torch.tensor(source_centroid, device=self.device)

        max_iter, batch_size = self._stage2_budget()
        if (max_iter, batch_size) != (self.max_iter, self.batch_size):
            print(f"{self.name}: stage 2 overridden to {max_iter} iters "
                  f"x {batch_size} batch", flush=True)
        rng = np.random.default_rng(0)
        optimiser = torch.optim.SGD([vector], lr=self.learning_rate)
        for iteration in range(max_iter):
            src = rng.integers(0, n, batch_size)
            tgt = rng.integers(0, n, batch_size)
            optimiser.zero_grad()

            dist_total, lm_total, used = 0.0, 0.0, 0
            for a, b in zip(src, tgt):
                activation, source_tokens = self._last_activation(
                    layer, benign[a]
                )
                with torch.no_grad():
                    latent = sae.encode(
                        activation.float().unsqueeze(0)
                    )[0].detach()
                steered = latent + vector
                distance = ((steered - mu_pos) ** 2).sum() \
                    - 0.5 * ((steered - mu_neg) ** 2).sum()
                loss = self.lambda_dist * distance / batch_size
                dist_total += float(distance) / batch_size

                nll = self._lm_nll(
                    sae, layer, vector,
                    self.model.to_tokens(benign[a]),
                    self.model.to_tokens(toxic[b]), latent,
                )
                if nll is not None:
                    loss = loss + self.lambda_lm * nll / batch_size
                    lm_total += float(nll) / batch_size
                    used += 1
                loss.backward()

            reg = self.lambda_reg * vector[mask].abs().sum()
            reg.backward()
            with torch.no_grad():
                vector.grad[~mask] = 0.0
            optimiser.step()
            with torch.no_grad():
                vector[~mask] = 0.0

            if iteration == 0 or (iteration + 1) % 20 == 0:
                print(f"{self.name}: iter {iteration + 1}/{max_iter} "
                      f"dist={dist_total:.3f} lm={lm_total:.3f} "
                      f"reg={float(reg):.4f} ||v||={float(vector.norm()):.3f}",
                      flush=True)

        self.ssv = vector.detach()
        self.subspace_dims = dims.tolist()
        self.sae = sae
        self.sae_layer = layer
        # A residual-space summary for anything that inspects a steerer;
        # `hooks()` does not read it, so the round trip is not applied twice.
        with torch.no_grad():
            self.steer_vecs = {
                layer: sae.decode(self.ssv.unsqueeze(0))[0].to(self.dtype)
            }
        print(f"{self.name}: ||ssv||={float(self.ssv.norm()):.4f} "
              f"over {len(dims)} dims")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def hooks(self, coefficient, vectors=None, layers=None):
        """Encode the last token, add lambda * ssv, decode, replace.

        The reference steers inside the SAE's latent space and writes the
        decoded result back, rather than adding a vector to the residual
        stream, so the reconstruction is part of the intervention.
        """
        del vectors, layers
        ssv = getattr(self, "ssv", None)
        if ssv is None or coefficient == 0.0:
            return []
        sae, layer = self.sae, self.sae_layer
        shift = ssv * coefficient

        @torch.no_grad()
        def hook(act, hook=None, **kwargs):
            del kwargs
            act = act.clone()
            last = act[:, -1, :].float()
            steered = sae.encode(last) + shift
            act[:, -1, :] = sae.decode(steered).to(act.dtype)
            return act

        return [(f"blocks.{layer}.hook_resid_post", hook)]
