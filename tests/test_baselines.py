import sys
import types
import unittest

import torch

sae_lens_stub = types.ModuleType("sae_lens")
sae_lens_stub.SAE = object
transformer_lens_stub = types.ModuleType("transformer_lens")
transformer_lens_stub.HookedTransformer = object
sys.modules.setdefault("sae_lens", sae_lens_stub)
sys.modules.setdefault("transformer_lens", transformer_lens_stub)

from baselines.caa.caa import (  # noqa: E402
    CAASingleLayerSteerer,
    CAASteerer,
    CAATop3LayersSteerer,
)
from baselines.common import BaselineSteerer  # noqa: E402
from baselines.registry import available, get_baseline  # noqa: E402
from circuitsteer.core import CircuitSteer  # noqa: E402


class FakeCache(dict):
    pass


class FakeModel:
    """Residual at the last position is a fixed vector per text."""

    def __init__(self, values, layers=(1, 2)):
        self.values = values
        self.layers = layers

    def run_with_cache(self, text, stop_at_layer=None):
        del stop_at_layer
        value = self.values[text]
        cache = FakeCache()
        for layer in self.layers:
            cache[f"blocks.{layer}.hook_resid_post"] = torch.tensor(
                [[[float(value), float(value) * 2]]]
            )
        return None, cache


class SeparationModel:
    """Layer L separates the two classes by a margin of L."""

    def __init__(self, layers):
        self.layers = layers

    def run_with_cache(self, text, stop_at_layer=None):
        del stop_at_layer
        toxic = text.startswith("t")
        cache = FakeCache()
        for layer in self.layers:
            base = 100.0                      # same norm at every layer
            offset = float(layer) if toxic else -float(layer)
            cache[f"blocks.{layer}.hook_resid_post"] = torch.tensor(
                [[[base + offset, base]]]
            )
        return None, cache


class HarnessTests(unittest.TestCase):
    def test_baselines_inherit_the_evaluation_path(self):
        """Hooks/generation/perplexity must be CircuitSteer's own code."""
        for attr in ("hooks", "generate", "generate_batch", "perplexity"):
            self.assertIs(
                getattr(BaselineSteerer, attr),
                getattr(CircuitSteer, attr),
                f"{attr} must not be overridden by the baseline harness",
            )

    def test_fit_is_abstract(self):
        steerer = BaselineSteerer.__new__(BaselineSteerer)
        with self.assertRaises(NotImplementedError):
            steerer.fit(["a"], ["b"])

    def test_saes_are_skipped_unless_required(self):
        steerer = BaselineSteerer.__new__(BaselineSteerer)
        steerer.requires_saes = False
        self.assertEqual(steerer._load_saes(), {})

    def test_registry_resolves_caa(self):
        self.assertIn("caa", available())
        self.assertIs(get_baseline("caa"), CAASteerer)

    def test_unknown_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            get_baseline("nope")


class CAATests(unittest.TestCase):
    def make(self):
        steerer = CAASteerer.__new__(CAASteerer)
        steerer.device = "cpu"
        steerer.dtype = torch.float32
        steerer.sae_layers = (1, 2)
        steerer.model = FakeModel({"t1": 4.0, "t2": 6.0, "b1": 1.0, "b2": 3.0})
        return steerer

    def test_vector_is_the_contrastive_mean_difference(self):
        steerer = self.make()
        steerer.fit(["t1", "t2"], ["b1", "b2"])
        # mean toxic = 5, mean benign = 2 -> difference 3 (and 6 in dim 2)
        torch.testing.assert_close(
            steerer.steer_vecs[1], torch.tensor([3.0, 6.0])
        )
        torch.testing.assert_close(
            steerer.steer_vecs[2], torch.tensor([3.0, 6.0])
        )

    def test_steers_the_same_layers_as_the_method(self):
        steerer = self.make()
        steerer.fit(["t1"], ["b1"])
        self.assertEqual(sorted(steerer.steer_vecs), [1, 2])


class LayerSelectionTests(unittest.TestCase):
    """Deeper layers separate more strongly in this fixture, so ranking
    must pick the deepest ones."""

    def make(self, cls):
        steerer = cls.__new__(cls)
        steerer.device = "cpu"
        steerer.dtype = torch.float32
        steerer.sae_layers = (2, 4, 6, 8)
        steerer.model = SeparationModel(steerer.sae_layers)
        return steerer

    def test_single_layer_picks_the_strongest(self):
        steerer = self.make(CAASingleLayerSteerer)
        steerer.fit(["t1", "t2"], ["b1", "b2"])
        self.assertEqual(sorted(steerer.steer_vecs), [8])

    def test_top3_picks_the_three_strongest(self):
        steerer = self.make(CAATop3LayersSteerer)
        steerer.fit(["t1", "t2"], ["b1", "b2"])
        self.assertEqual(sorted(steerer.steer_vecs), [4, 6, 8])

    def test_plain_caa_keeps_every_layer(self):
        steerer = self.make(CAASteerer)
        steerer.fit(["t1"], ["b1"])
        self.assertEqual(sorted(steerer.steer_vecs), [2, 4, 6, 8])

    def test_strength_is_scaled_by_activation_norm(self):
        """Equal absolute differences at different residual scales must
        not rank by depth alone."""
        small = torch.tensor([1.0, 0.0])
        big_mean = torch.tensor([1000.0, 0.0])
        small_mean = torch.tensor([10.0, 0.0])
        deep = CAASteerer._layer_strength(small, big_mean, big_mean)
        shallow = CAASteerer._layer_strength(small, small_mean, small_mean)
        self.assertGreater(shallow, deep)


if __name__ == "__main__":
    unittest.main()
