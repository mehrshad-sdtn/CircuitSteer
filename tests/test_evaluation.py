import unittest
from unittest.mock import patch

from circuitsteer.evaluation import evaluate_coefficient


class FakeModel:
    def reset_hooks(self):
        return None


class FakeSteerer:
    def __init__(self):
        self.model = FakeModel()
        self.device = "cpu"

    def hooks(self, coefficient):
        return [("fake", coefficient)]

    def generate(self, prompt, coefficient, seed):
        del prompt, seed
        return "base" if coefficient == 0.0 else "steered"

    def perplexity(self, output):
        return 2.0 if output == "base" else 4.0


def fake_score_function(dataset_label, model, device):
    del dataset_label, model, device

    def score(item, output=None, hooks=None):
        del item, hooks
        return 0.8 if output == "base" else 0.3

    return score


class EvaluationTests(unittest.TestCase):
    @patch(
        "circuitsteer.evaluation.make_score_function",
        side_effect=fake_score_function,
    )
    def test_delta_and_normalized_perplexity(self, _mock_score):
        metrics = evaluate_coefficient(
            FakeSteerer(),
            "RTP",
            ["one", "two"],
            coefficient=-3.0,
            seed=42,
        )

        self.assertAlmostEqual(metrics["delta"], 0.5)
        self.assertAlmostEqual(metrics["delta_std"], 0.0)
        self.assertAlmostEqual(metrics["norm_ppl"], 2.0)
        self.assertEqual(metrics["n"], 2)


if __name__ == "__main__":
    unittest.main()
