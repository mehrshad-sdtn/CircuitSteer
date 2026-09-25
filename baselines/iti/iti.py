"""Inference-Time Intervention (Li et al., 2023).

Reference: "Inference-Time Intervention: Eliciting Truthful Answers from
a Language Model", NeurIPS 2023; code at github.com/likenneth/honest_llama.

The method, as specified in the paper's section 3:

* probe the output of every attention head, x_l^h, taken after Att and
  BEFORE the output projection Q_l^h. In TransformerLens that tensor is
  `blocks.{l}.attn.hook_z`, so the hook lands exactly where the paper's
  equation does;
* fit one logistic probe per head, p_t(x) = sigmoid(<t, x>), on the
  last-token activation of each concatenated example;
* rank heads by probe accuracy on a held-out split and keep the top K;
* shift each selected head by alpha * sigma_l^h * theta_l^h, where sigma
  is the standard deviation of activations projected on theta. Equation
  2 of the paper:

      x_{l+1} = x_l + sum_h Q_l^h [ Att_l^h(P_l^h x_l) + a s_l^h t_l^h ]

  with theta the zero vector for heads outside the selected set;
* apply it at every generated token, autoregressively.

Two deliberate departures, both recorded because they affect how the
number should be read:

1. The direction is the mass-mean shift (benign mean -> target mean),
   which the paper reports as its best variant and uses everywhere
   outside its direction ablation. Probe weights remain available via
   `direction="probe"`.
2. The paper's direction points towards the desired behaviour and uses
   a positive alpha. Here it points towards the TARGET behaviour, matching
   CAA and RepE in this repository, so a negative coefficient suppresses
   and the shared lambda sweep keeps its meaning across methods.

Unlike every other baseline here, ITI intervenes per attention head
rather than on the residual stream, so it overrides `hooks()` instead of
populating `steer_vecs` with one vector per layer. That is the method,
not a convenience: restricting it to residual-stream additions at the
SAE layers would not be ITI.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer
from baselines.iti.probes import fit_head_probes


class ITISteerer(BaselineSteerer):
    name = "ITI"
    requires_saes = False

    #: Number of attention heads to intervene on. 48 is the paper's
    #: value for LLaMA-7B (1024 heads, ~4.7%). Gemma-2-2B has far fewer
    #: heads in total, so the same K is a much larger fraction; `fit`
    #: reports the fraction so this is visible rather than buried.
    n_heads: int = 48

    #: "mass_mean" is the paper's best and default; "probe" is its
    #: probe-weight ablation.
    direction: str = "mass_mean"

    #: Fraction of examples held out to rank heads by probe accuracy.
    val_fraction: float = 0.2

    @torch.no_grad()
    def _head_activations(self, texts: list[str]) -> np.ndarray:
        """Last-token output of every head: [n_texts, n_layers, n_heads, d_head]."""
        collected = []
        n_layers = self.model.cfg.n_layers
        names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
        for text in tqdm(texts, desc=f"{self.name} reading", leave=False):
            _, cache = self.model.run_with_cache(
                text, names_filter=lambda n: n.endswith("attn.hook_z")
            )
            per_layer = [cache[name][0, -1] for name in names]
            collected.append(
                torch.stack(per_layer).float().cpu().numpy()
            )
        return np.stack(collected)

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        n = min(len(toxic), len(benign))
        if n == 0:
            raise ValueError("ITI needs paired target/benign examples")
        toxic, benign = toxic[:n], benign[:n]

        target = self._head_activations(toxic)
        control = self._head_activations(benign)
        n_layers, n_heads, d_head = target.shape[1:]
        total = n_layers * n_heads
        print(
            f"{self.name}: probing {total} heads ({n_layers} layers x "
            f"{n_heads}), d_head={d_head}, {n} pairs"
        )

        selected, shifts, scores = fit_head_probes(
            target, control, n_select=self.n_heads,
            direction=self.direction, val_fraction=self.val_fraction,
        )
        picked = np.array([scores[l, h] for l, h in selected])
        print(
            f"{self.name}: top {len(selected)} of {total} heads "
            f"({len(selected) / total:.1%}), probe accuracy "
            f"{picked.min():.3f}-{picked.max():.3f} "
            f"(median over all heads {np.median(scores):.3f})"
        )

        self.head_shifts = torch.tensor(
            shifts, device=self.device, dtype=self.dtype
        )
        self.selected_heads = selected
        self.head_scores = scores
        # A per-layer summary so anything inspecting a steerer sees
        # something sensible; `hooks()` does not read it, so there is no
        # risk of the shift being applied twice.
        self.steer_vecs = {
            layer: self.head_shifts[layer].flatten()
            for layer in sorted({l for l, _ in selected})
        }
        print(
            f"{self.name}: intervening on layers "
            f"{sorted({l for l, _ in selected})}"
        )

    def hooks(self, coefficient, vectors=None, layers=None):
        """Add coefficient * sigma * theta to each selected head's output.

        Overrides the residual-stream hook because ITI is defined on head
        outputs before the output projection.
        """
        del vectors, layers
        shifts = getattr(self, "head_shifts", None)
        if shifts is None or coefficient == 0.0:
            return []
        scaled = shifts * coefficient
        touched = sorted({layer for layer, _ in self.selected_heads})

        def hook(z, hook=None, **kwargs):
            del kwargs
            # z: [batch, pos, n_heads, d_head]
            return z + scaled[hook.layer()].to(z.dtype)

        return [(f"blocks.{layer}.attn.hook_z", hook) for layer in touched]
