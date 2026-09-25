import math
import unittest
from contextlib import contextmanager

import torch

from circuitsteer.scoring import (
    anger_score,
    make_score_function,
    sycophancy_score,
    toxicity_score,
)

# ' (A)' and ' (B)' share their leading ' (' token under every tokenizer this
# project targets (GPT-2, Llama-3.1, Gemma-2), so this toy vocabulary
# reproduces the property the real scorer has to cope with.
VOCAB = {"<bos>": 0, "Q": 1, " (": 2, "A": 3, "B": 4, ")": 5}
PIECES = sorted(
    (piece for piece in VOCAB if piece != "<bos>"),
    key=len,
    reverse=True,
)


class FakeModel:
    """Scores 'A' above 'B' by exactly 1.0 logit, unless hooked."""

    def __init__(self):
        self.prefer = "A"

    def to_tokens(self, text, prepend_bos=True):
        ids = [VOCAB["<bos>"]] if prepend_bos else []
        position = 0
        while position < len(text):
            for piece in PIECES:
                if text.startswith(piece, position):
                    ids.append(VOCAB[piece])
                    position += len(piece)
                    break
            else:
                raise ValueError(f"untokenizable: {text[position:]!r}")
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, tokens, **kwargs):
        del kwargs
        logits = torch.zeros(1, tokens.shape[1], len(VOCAB))
        favoured = VOCAB[self.prefer]
        other = VOCAB["B" if self.prefer == "A" else "A"]
        logits[..., favoured] = 2.0
        logits[..., other] = 1.0
        return logits

    @contextmanager
    def hooks(self, fwd_hooks=None):
        del fwd_hooks
        self.prefer = "B"
        try:
            yield self
        finally:
            self.prefer = "A"


class SycophancyScoreTests(unittest.TestCase):
    def test_shared_first_token_does_not_collapse_the_score(self):
        """Regression: scoring only the first token measured ' (', which is
        identical for both answers, so the metric was constant."""
        model = FakeModel()
        syco_is_a = sycophancy_score("Q", " (A)", " (B)", model)
        syco_is_b = sycophancy_score("Q", " (B)", " (A)", model)

        self.assertNotAlmostEqual(syco_is_a, syco_is_b, places=6)
        # softmax over summed answer log-probs; the answers differ by
        # exactly 1.0 logit, so the score is sigmoid(+/-1).
        self.assertAlmostEqual(syco_is_a, 1 / (1 + math.exp(-1.0)), places=5)
        self.assertAlmostEqual(syco_is_b, 1 / (1 + math.exp(1.0)), places=5)
        self.assertGreater(syco_is_a, 0.5)
        self.assertLess(syco_is_b, 0.5)

    def test_answer_order_does_not_decide_the_score(self):
        """156 of the first 400 items have the sycophantic answer as (B),
        so the metric must not track the letter."""
        model = FakeModel()
        self.assertAlmostEqual(
            sycophancy_score("Q", " (A)", " (B)", model),
            1 - sycophancy_score("Q", " (B)", " (A)", model),
            places=5,
        )

    def test_hooks_change_the_score(self):
        model = FakeModel()
        unsteered = sycophancy_score("Q", " (A)", " (B)", model)
        steered = sycophancy_score(
            "Q",
            " (A)",
            " (B)",
            model,
            hooks=[("fake", None)],
        )
        self.assertGreater(unsteered, 0.5)
        self.assertLess(steered, 0.5)

    def test_hooks_are_released_after_scoring(self):
        model = FakeModel()
        sycophancy_score("Q", " (A)", " (B)", model, hooks=[("fake", None)])
        self.assertEqual(model.prefer, "A")

    def test_blank_question_is_not_scored(self):
        self.assertTrue(
            math.isnan(sycophancy_score("  ", " (A)", " (B)", FakeModel()))
        )


class ScoreFunctionTests(unittest.TestCase):
    def test_sycophancy_items_must_carry_both_answers(self):
        score = make_score_function("Sycophancy", FakeModel(), "cpu")
        with self.assertRaises(TypeError):
            score(("Q", " (A)"))
        with self.assertRaises(TypeError):
            score("Q")

    def test_sycophancy_three_tuple_is_accepted(self):
        score = make_score_function("Sycophancy", FakeModel(), "cpu")
        self.assertAlmostEqual(
            score(("Q", " (A)", " (B)")),
            1 / (1 + math.exp(-1.0)),
            places=5,
        )

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            make_score_function("Nope", FakeModel(), "cpu")


class EmptyOutputTests(unittest.TestCase):
    """Empty continuations must not score as clean text.

    These run without downloading Detoxify or the emotion pipeline: the
    blank check short-circuits before the classifier is constructed.
    """

    def test_toxicity_of_empty_output_is_nan(self):
        self.assertTrue(math.isnan(toxicity_score("", "cpu")))
        self.assertTrue(math.isnan(toxicity_score("   \n ", "cpu")))

    def test_anger_of_empty_output_is_nan(self):
        self.assertTrue(math.isnan(anger_score("", "cpu")))
        self.assertTrue(math.isnan(anger_score("  ", "cpu")))

    def test_empty_output_scores_nan_through_score_function(self):
        score = make_score_function("RTP", FakeModel(), "cpu")
        self.assertTrue(math.isnan(score("prompt", output="")))
        self.assertTrue(math.isnan(score("prompt", output=None)))


if __name__ == "__main__":
    unittest.main()
