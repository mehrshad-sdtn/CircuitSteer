import sys
import types
import unittest

import torch

# The unit tests exercise tensor logic without installing or downloading
# TransformerLens, SAELens, model weights, or SAE weights.
sae_lens_stub = types.ModuleType("sae_lens")
sae_lens_stub.SAE = object
transformer_lens_stub = types.ModuleType("transformer_lens")
transformer_lens_stub.HookedTransformer = object
sys.modules["sae_lens"] = sae_lens_stub
sys.modules["transformer_lens"] = transformer_lens_stub

from circuitsteer.config import CircuitConfig  # noqa: E402
from circuitsteer.core import CircuitSteer  # noqa: E402


class FakeSae:
    def __init__(self, decoder_vectors):
        self.W_dec = torch.tensor(decoder_vectors, dtype=torch.float32)


class FakeHook:
    def __init__(self, layer):
        self._layer = layer

    def layer(self):
        return self._layer


class CircuitTensorTests(unittest.TestCase):
    def make_steerer(self):
        steerer = CircuitSteer.__new__(CircuitSteer)
        steerer.config = CircuitConfig(top_k=1, steer_pool="sum")
        steerer.device = "cpu"
        steerer.dtype = torch.float32
        steerer.circuit = [
            (("L1_0", "L2_1"), 0.8),
            (("L1_1", "L2_0"), 0.6),
            (("L1_2", "L2_2"), 0.4),
        ]
        steerer.saes = {
            1: FakeSae([[1, 2], [3, 4], [5, 6]]),
            2: FakeSae([[7, 8], [9, 10], [11, 12]]),
        }
        return steerer

    def test_feature_vectors_match_notebook_top_k_behavior(self):
        steerer = self.make_steerer()
        vectors, counts = steerer.feature_vectors()

        # The notebook selects top_k * 2 edges and origin features.
        self.assertEqual(counts, {1: 2})
        torch.testing.assert_close(vectors[1], torch.tensor([4.0, 6.0]))

    def test_destination_features_can_be_included_for_layer_sweep(self):
        steerer = self.make_steerer()
        vectors, counts = steerer.feature_vectors(
            include_destinations=True
        )

        self.assertEqual(counts, {1: 2, 2: 2})
        torch.testing.assert_close(vectors[2], torch.tensor([16.0, 18.0]))

    def test_hooks_add_scaled_vector_only_to_selected_layer(self):
        steerer = self.make_steerer()
        steerer.steer_vecs = {1: torch.tensor([1.0, 2.0])}
        hooks = steerer.hooks(-2.0)

        self.assertEqual(hooks[0][0], "blocks.1.hook_resid_post")
        residual = torch.zeros((1, 1, 2))
        changed = hooks[0][1](residual, hook=FakeHook(1))
        torch.testing.assert_close(
            changed,
            torch.tensor([[[-2.0, -4.0]]]),
        )


if __name__ == "__main__":
    unittest.main()
