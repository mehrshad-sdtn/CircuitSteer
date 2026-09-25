import sys
import types
import unittest

import numpy as np
import torch

for _name, _attr in (("sae_lens", "SAE"), ("transformer_lens", "HookedTransformer")):
    _m = types.ModuleType(_name)
    setattr(_m, _attr, object)
    sys.modules.setdefault(_name, _m)

from experiments.alignment_causal import rotate_to_cosine  # noqa: E402


def cosine(a, b):
    a, b = a.float(), b.float()
    return float(a @ b / (a.norm() * b.norm()))


class RotationTests(unittest.TestCase):
    """The sweep must change ONLY the angle."""

    def setUp(self):
        torch.manual_seed(0)
        self.anchor = torch.randn(64)
        self.vector = torch.randn(64)

    def test_hits_the_requested_cosine(self):
        for target in (-1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0):
            out = rotate_to_cosine(self.anchor, self.vector, target)
            self.assertAlmostEqual(cosine(self.anchor, out), target, places=4)

    def test_preserves_magnitude(self):
        """If the norm changed, the sweep would confound angle with
        intervention strength."""
        for target in (-1.0, 0.0, 0.5, 1.0):
            out = rotate_to_cosine(self.anchor, self.vector, target)
            self.assertAlmostEqual(
                float(out.norm()), float(self.vector.norm()), places=3)

    def test_cosine_is_clamped(self):
        out = rotate_to_cosine(self.anchor, self.vector, 4.0)
        self.assertAlmostEqual(cosine(self.anchor, out), 1.0, places=4)

    def test_handles_a_collinear_vector(self):
        collinear = self.anchor * 2.5
        out = rotate_to_cosine(self.anchor, collinear, 0.0)
        self.assertAlmostEqual(cosine(self.anchor, out), 0.0, places=4)
        self.assertAlmostEqual(
            float(out.norm()), float(collinear.norm()), places=3)

    def test_cosine_one_reproduces_the_anchor_direction(self):
        out = rotate_to_cosine(self.anchor, self.vector, 1.0)
        unit_a = self.anchor / self.anchor.norm()
        unit_o = out.float() / out.float().norm()
        torch.testing.assert_close(unit_a, unit_o, atol=1e-4, rtol=1e-4)


class AdditivityMathTests(unittest.TestCase):
    """The ratio ||d_all|| / sum||d_l|| is the interference measure."""

    def test_perfectly_cooperating_layers_give_one(self):
        singles = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        joint = sum(singles)
        ratio = np.linalg.norm(joint) / sum(np.linalg.norm(s) for s in singles)
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_opposed_layers_cancel_to_zero(self):
        singles = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
        joint = sum(singles)
        ratio = np.linalg.norm(joint) / sum(np.linalg.norm(s) for s in singles)
        self.assertAlmostEqual(ratio, 0.0, places=6)

    def test_orthogonal_layers_sit_between(self):
        singles = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        joint = sum(singles)
        ratio = np.linalg.norm(joint) / sum(np.linalg.norm(s) for s in singles)
        self.assertAlmostEqual(ratio, 2 ** 0.5 / 2, places=6)


if __name__ == "__main__":
    unittest.main()
