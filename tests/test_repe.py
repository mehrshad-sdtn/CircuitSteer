import sys
import types
import unittest

import numpy as np
import torch

sae_lens_stub = types.ModuleType("sae_lens")
sae_lens_stub.SAE = object
transformer_lens_stub = types.ModuleType("transformer_lens")
transformer_lens_stub.HookedTransformer = object
sys.modules.setdefault("sae_lens", sae_lens_stub)
sys.modules.setdefault("transformer_lens", transformer_lens_stub)

from baselines.registry import available, get_baseline  # noqa: E402
from baselines.repe.repe import RepESteerer  # noqa: E402
from circuitsteer.core import CircuitSteer  # noqa: E402


class FakeCache(dict):
    pass


class PlantedModel:
    """Target and benign differ along a planted direction, plus noise on
    an orthogonal axis with LARGER variance.

    This is the case that separates RepE from CAA: PCA on the raw
    activations would lock onto the noise axis, while PCA on the paired
    differences recovers the planted direction because the noise is
    shared within a pair and cancels.
    """

    def __init__(self, layers, planted, noise_scale=50.0, seed=0):
        self.layers = layers
        self.planted = np.asarray(planted, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale
        self._shared = {}

    def run_with_cache(self, text, stop_at_layer=None):
        del stop_at_layer
        kind, index = text.split("-")
        if index not in self._shared:
            self._shared[index] = self.rng.normal(0, self.noise_scale, 2)
        shared = self._shared[index]
        vec = shared + (self.planted if kind == "t" else np.zeros(2))
        cache = FakeCache()
        for layer in self.layers:
            cache[f"blocks.{layer}.hook_resid_post"] = torch.tensor(
                [[vec]], dtype=torch.float32
            )
        return None, cache

    def reset_hooks(self):
        return None


class RepEHarnessTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("repe", available())
        self.assertIs(get_baseline("repe"), RepESteerer)

    def test_reuses_the_shared_evaluation_path(self):
        for attr in ("hooks", "generate", "generate_batch", "perplexity"):
            self.assertIs(getattr(RepESteerer, attr),
                          getattr(CircuitSteer, attr))

    def test_skips_sae_loading(self):
        steerer = RepESteerer.__new__(RepESteerer)
        steerer.requires_saes = False
        self.assertEqual(steerer._load_saes(), {})


class LATTests(unittest.TestCase):
    def make(self, planted=(3.0, 0.0), scale=True):
        s = RepESteerer.__new__(RepESteerer)
        s.device, s.dtype = "cpu", torch.float32
        s.sae_layers = (1, 2)
        s.scale_to_data = scale
        s.model = PlantedModel(s.sae_layers, planted)
        return s

    def test_recovers_the_planted_direction_not_the_noise_axis(self):
        s = self.make(planted=(3.0, 0.0))
        n = 60
        s.fit([f"t-{i}" for i in range(n)], [f"b-{i}" for i in range(n)])
        v = s.steer_vecs[1].numpy()
        unit = v / np.linalg.norm(v)
        # Planted direction is +x; noise dominates variance on both axes
        # but cancels within pairs.
        self.assertGreater(abs(float(unit[0])), 0.99)

    def test_sign_points_from_benign_towards_target(self):
        s = self.make(planted=(3.0, 0.0))
        n = 60
        s.fit([f"t-{i}" for i in range(n)], [f"b-{i}" for i in range(n)])
        self.assertGreater(float(s.steer_vecs[1][0]), 0.0)

    def test_negative_planted_direction_flips_sign(self):
        s = self.make(planted=(-3.0, 0.0))
        n = 60
        s.fit([f"t-{i}" for i in range(n)], [f"b-{i}" for i in range(n)])
        self.assertLess(float(s.steer_vecs[1][0]), 0.0)

    def test_unit_norm_mode_matches_the_reference(self):
        s = self.make(planted=(3.0, 0.0), scale=False)
        n = 40
        s.fit([f"t-{i}" for i in range(n)], [f"b-{i}" for i in range(n)])
        self.assertAlmostEqual(
            float(torch.linalg.norm(s.steer_vecs[1])), 1.0, places=4
        )

    def test_scaled_mode_matches_the_caa_magnitude(self):
        """Scale must track the size of the paired difference.

        Regression: scaling by the projection *spread* made a perfectly
        consistent signal (zero spread) collapse to a zero vector.
        """
        s = self.make(planted=(3.0, 0.0), scale=True)
        n = 40
        s.fit([f"t-{i}" for i in range(n)], [f"b-{i}" for i in range(n)])
        norm = float(torch.linalg.norm(s.steer_vecs[1]))
        self.assertAlmostEqual(norm, 3.0, places=3)

    def test_consistent_differences_do_not_collapse_the_vector(self):
        s = self.make(planted=(5.0, 0.0), scale=True)
        n = 30
        s.fit([f"t-{i}" for i in range(n)], [f"b-{i}" for i in range(n)])
        self.assertGreater(float(torch.linalg.norm(s.steer_vecs[1])), 4.0)

    def test_steers_every_candidate_layer(self):
        s = self.make()
        s.fit([f"t-{i}" for i in range(20)], [f"b-{i}" for i in range(20)])
        self.assertEqual(sorted(s.steer_vecs), [1, 2])

    def test_empty_input_is_rejected(self):
        s = self.make()
        with self.assertRaises(ValueError):
            s.fit([], [])


if __name__ == "__main__":
    unittest.main()
