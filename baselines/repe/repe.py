"""RepE - Representation Engineering (Zou et al., 2023).

Reimplemented from the MIT-licensed reference implementation at
https://github.com/andyzoujm/representation-engineering (repe/rep_readers.py,
repe/rep_reading_pipeline.py), arXiv:2310.01405.

Only the *reading* half is ported. The reference package builds on
HuggingFace `transformers` pipelines, while this project runs
TransformerLens `HookedTransformer`; running their pipeline end to end
would give RepE different generation, tokenisation, perplexity and
scoring code from every other method in the table, which would make the
comparison meaningless. The *control* half needs no porting at all -
RepE's intervention is h <- h + lambda*v at selected layers, exactly what
`CircuitSteer.hooks()` already does - so the single thing that differs
between methods stays "how the vector is derived".

The reading method (Linear Artificial Tomography) is PCA on *paired
differences*, not on raw activations, following the reference:

    relative[layer] = relative[layer][::2] - relative[layer][1::2]
    ... PCA(n_components=1).fit(recentered differences)

which is a genuinely different estimator from CAA's difference-in-means:
CAA averages the paired differences, RepE takes their leading principal
component.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer


class RepESteerer(BaselineSteerer):
    name = "RepE"
    requires_saes = False

    #: The reference returns a unit-norm PCA component. Left at unit norm
    #: a shared lambda grid is not comparable across methods: CAA's
    #: difference-in-means inherits the residual-stream scale (norm in the
    #: tens) while a unit vector barely moves the model at lambda=-4.
    #:
    #: We therefore rescale by the MEAN projection of the paired
    #: differences onto the direction - i.e. |mean(differences) . d|,
    #: which is CAA's own difference-in-means magnitude measured along
    #: RepE's direction. Scaling by the projection *spread* instead would
    #: be perverse: perfectly consistent differences have zero spread, so
    #: the cleanest possible signal would produce a vanishing vector.
    #: Set False to reproduce the reference's raw unit-norm direction.
    scale_to_data: bool = True

    @torch.no_grad()
    def _hidden_states(self, texts: list[str]) -> dict[int, np.ndarray]:
        layers = list(self.steer_layers)
        out: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        for text in tqdm(texts, desc=f"{self.name} reading", leave=False):
            _, cache = self.model.run_with_cache(
                text, stop_at_layer=max(layers) + 1
            )
            for layer in layers:
                vec = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                out[layer].append(vec.float().cpu().numpy())
        return {layer: np.stack(v) for layer, v in out.items()}

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        from sklearn.decomposition import PCA

        n = min(len(toxic), len(benign))
        if n == 0:
            raise ValueError("RepE needs paired target/benign prompts")
        toxic, benign = toxic[:n], benign[:n]
        print(
            f"{self.name}: LAT over {n} pairs at layers "
            f"{list(self.steer_layers)}"
        )

        target = self._hidden_states(toxic)
        control = self._hidden_states(benign)

        self.steer_vecs = {}
        for layer in self.steer_layers:
            # Paired differences, as in the reference's [::2] - [1::2] on
            # an interleaved [target, benign, target, benign, ...] stack.
            differences = target[layer] - control[layer]
            centred = differences - differences.mean(0, keepdims=True)

            pca = PCA(n_components=1, random_state=0).fit(centred)
            direction = pca.components_[0]

            # Sign: the reference resolves it by checking whether the
            # labelled member of each pair projects high or low. Here the
            # positive member is the target prompt, so the direction must
            # point from benign towards target.
            projection = differences @ direction
            if float(projection.mean()) < 0:
                direction = -direction

            if self.scale_to_data:
                magnitude = abs(float(differences.mean(0) @ direction))
                direction = direction * magnitude

            self.steer_vecs[layer] = torch.tensor(
                direction, device=self.device, dtype=self.dtype
            )
        print(f"{self.name}: reading vectors for {sorted(self.steer_vecs)}")
