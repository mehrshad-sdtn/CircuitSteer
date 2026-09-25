import math
import unittest
from unittest.mock import patch

import pandas as pd

from circuitsteer.evaluation import (
    evaluate_coefficient,
    select_best_coefficient,
)


class FakeModel:
    def reset_hooks(self):
        return None


class FakeSteerer:
    """Steering halves the score and doubles perplexity."""

    def __init__(self, steered_output="steered"):
        self.model = FakeModel()
        self.device = "cpu"
        self.steered_output = steered_output
        self.seeds = []

    def hooks(self, coefficient):
        return [("fake", coefficient)]

    def generate(self, prompt, coefficient, seed):
        del prompt
        self.seeds.append(seed)
        return "base" if coefficient == 0.0 else self.steered_output

    def perplexity(self, output):
        if not output.strip():
            return math.nan
        return 2.0 if output == "base" else 4.0


def fake_score_function(dataset_label, model, device):
    del dataset_label, model, device

    def score(item, output=None, hooks=None):
        del item, hooks
        # Mirrors the real scorers: nothing to score means NaN, not 0.0.
        if not (output or "").strip():
            return math.nan
        return 0.8 if output == "base" else 0.3

    return score


@patch(
    "circuitsteer.evaluation.make_score_function",
    side_effect=fake_score_function,
)
class EvaluationTests(unittest.TestCase):
    def test_delta_and_normalized_perplexity(self, _mock):
        metrics = evaluate_coefficient(
            FakeSteerer(),
            "RTP",
            ["one", "two"],
            coefficient=-3.0,
            seed=42,
        )

        self.assertAlmostEqual(metrics["delta"], 0.5)
        self.assertAlmostEqual(metrics["delta_std"], 0.0)
        self.assertAlmostEqual(metrics["delta_sem"], 0.0)
        self.assertAlmostEqual(metrics["norm_ppl"], 2.0)
        self.assertAlmostEqual(metrics["valid_frac"], 1.0)
        self.assertAlmostEqual(metrics["degenerate_frac"], 0.0)
        self.assertEqual(metrics["n"], 2)

    def test_base_and_steered_share_a_seed_per_prompt(self, _mock):
        """The comparison is paired: prompt i must use the same seed for
        its base and steered generation, so the delta is not sampling
        noise. Batching runs all base generations before all steered
        ones, so the two halves are compared rather than interleaved."""
        steerer = FakeSteerer()
        evaluate_coefficient(
            steerer,
            "RTP",
            ["one", "two", "three"],
            coefficient=-3.0,
            seed=42,
        )
        base_seeds = steerer.seeds[:3]
        steered_seeds = steerer.seeds[3:]
        self.assertEqual(base_seeds, [42, 43, 44])
        self.assertEqual(base_seeds, steered_seeds)

    def test_multiple_samples_use_distinct_seeds(self, _mock):
        steerer = FakeSteerer()
        metrics = evaluate_coefficient(
            steerer,
            "RTP",
            ["one", "two"],
            coefficient=-3.0,
            seed=42,
            samples_per_prompt=3,
        )
        expected = [42, 43, 44, 45, 46, 47]
        self.assertEqual(steerer.seeds[:6], expected)
        self.assertEqual(steerer.seeds[6:], expected)
        self.assertEqual(metrics["samples_per_prompt"], 3)
        self.assertAlmostEqual(metrics["delta_max"], 0.5)


@patch(
    "circuitsteer.evaluation.make_score_function",
    side_effect=fake_score_function,
)
class DegenerateSteeringTests(unittest.TestCase):
    """Steering that collapses generation must not look like a win.

    Before the fix an empty continuation scored 0.0 and had perplexity 0.0,
    so it produced the largest possible delta *and* a norm_ppl of 0.0 that
    sailed through the fluency guard.
    """

    def test_empty_outputs_do_not_earn_a_free_delta(self, _mock):
        metrics = evaluate_coefficient(
            FakeSteerer(steered_output=""),
            "RTP",
            ["one", "two"],
            coefficient=-4.0,
            seed=42,
        )

        # The old behaviour was delta == 0.8 (the largest achievable) with
        # norm_ppl == 0.0, i.e. a perfect score for a broken model.
        self.assertTrue(math.isnan(metrics["delta"]))
        self.assertTrue(math.isnan(metrics["norm_ppl"]))
        self.assertAlmostEqual(metrics["degenerate_frac"], 1.0)
        self.assertAlmostEqual(metrics["valid_frac"], 0.0)

    def test_partial_collapse_is_reported(self, _mock):
        class Sometimes(FakeSteerer):
            def generate(self, prompt, coefficient, seed):
                self.seeds.append(seed)
                if coefficient == 0.0:
                    return "base"
                return "" if seed % 2 == 0 else "steered"

        metrics = evaluate_coefficient(
            Sometimes(),
            "RTP",
            ["one", "two", "three", "four"],
            coefficient=-4.0,
            seed=42,
        )
        self.assertAlmostEqual(metrics["degenerate_frac"], 0.5)
        self.assertAlmostEqual(metrics["valid_frac"], 0.5)
        self.assertAlmostEqual(metrics["delta"], 0.5)


class SelectBestCoefficientTests(unittest.TestCase):
    """Selection is unfiltered by default: raw argmax on delta.

    Fluency and degeneracy thresholds are reported, not enforced, so they
    can be applied post hoc to the emitted numbers.
    """

    def results(self):
        return pd.DataFrame(
            [
                # Largest delta, but it broke the model.
                {"coeff": -4.0, "delta": 0.8, "norm_ppl": 0.02,
                 "degenerate_frac": 1.0},
                {"coeff": -3.0, "delta": 0.5, "norm_ppl": 1.2,
                 "degenerate_frac": 0.0},
                {"coeff": 0.0, "delta": 0.0, "norm_ppl": 1.0,
                 "degenerate_frac": 0.0},
            ]
        )

    def test_unfiltered_selection_takes_the_largest_delta(self):
        self.assertEqual(select_best_coefficient(self.results()), -4.0)

    def test_low_norm_ppl_is_not_treated_as_a_failure(self):
        """norm_ppl below 1 is reported, never used to reject."""
        results = pd.DataFrame(
            [
                {"coeff": -2.0, "delta": 0.4, "norm_ppl": 0.08,
                 "degenerate_frac": 0.0},
                {"coeff": 0.0, "delta": 0.0, "norm_ppl": 1.0,
                 "degenerate_frac": 0.0},
            ]
        )
        self.assertEqual(select_best_coefficient(results), -2.0)

    def test_upper_ppl_guard_does_not_catch_collapsed_text(self):
        """A `norm_ppl < 1.5` threshold is one-sided.

        Collapsed or repetitive steering has perplexity far *below* the
        baseline, so it passes an upper bound and still wins on delta.
        Anyone filtering post hoc needs a lower bound too, or the
        degeneracy fraction.
        """
        self.assertEqual(
            select_best_coefficient(self.results(), max_norm_ppl=1.5),
            -4.0,
        )

    def test_two_sided_ppl_guard_catches_collapsed_text(self):
        usable = self.results()
        usable = usable[usable["norm_ppl"] > 0.5]
        self.assertEqual(
            select_best_coefficient(usable, max_norm_ppl=1.5),
            -3.0,
        )

    def test_degeneracy_guard_can_be_reinstated_post_hoc(self):
        self.assertEqual(
            select_best_coefficient(self.results(), max_degenerate_frac=0.1),
            -3.0,
        )

    def test_nan_delta_is_never_selected(self):
        results = pd.DataFrame(
            [
                {"coeff": -4.0, "delta": math.nan, "norm_ppl": math.nan,
                 "degenerate_frac": 1.0},
                {"coeff": -1.0, "delta": 0.2, "norm_ppl": 1.05,
                 "degenerate_frac": 0.0},
                {"coeff": 0.0, "delta": 0.0, "norm_ppl": 1.0,
                 "degenerate_frac": 0.0},
            ]
        )
        self.assertEqual(select_best_coefficient(results), -1.0)

    def test_all_nan_falls_back_to_baseline(self):
        results = pd.DataFrame(
            [
                {"coeff": -4.0, "delta": math.nan, "norm_ppl": math.nan,
                 "degenerate_frac": 1.0},
            ]
        )
        self.assertEqual(select_best_coefficient(results), 0.0)


if __name__ == "__main__":
    unittest.main()
