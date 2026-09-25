import sys
import types
import unittest
import unittest.mock

for _n, _a in (("sae_lens", "SAE"), ("transformer_lens", "HookedTransformer")):
    _m = types.ModuleType(_n); setattr(_m, _a, object)
    sys.modules.setdefault(_n, _m)

import torch  # noqa: E402

from baselines.prompting.prompting import (  # noqa: E402
    EMOTION, PROMPTS, SYCOPHANCY, TOXICITY, PromptSteerer,
)
from baselines.registry import available, get_baseline  # noqa: E402
from circuitsteer.core import CircuitSteer  # noqa: E402


class PromptBaselineTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("prompt", available())
        self.assertIs(get_baseline("prompt"), PromptSteerer)

    def test_fit_produces_no_steering_vectors(self):
        s = PromptSteerer.__new__(PromptSteerer)
        s.dataset = "RTP"
        s.fit(["a"], ["b"])
        self.assertEqual(s.steer_vecs, {})

    def test_prefix_is_behaviour_specific(self):
        s = PromptSteerer.__new__(PromptSteerer)
        for ds, expected in (("RTP", TOXICITY), ("Jigsaw", TOXICITY),
                             ("Emotion", EMOTION),
                             ("Sycophancy", SYCOPHANCY)):
            s.dataset = ds
            self.assertIs(s.prefix(), expected)

    def test_every_dataset_has_a_prompt(self):
        self.assertEqual(set(PROMPTS),
                         {"RTP", "Jigsaw", "Emotion", "Sycophancy"})

    def test_toxicity_prompt_pins_the_topic(self):
        """Without this the model can dodge the behaviour by changing
        subject, which inflates the score while destroying the task."""
        for text in (TOXICITY, EMOTION):
            self.assertIn("same topic", text)

    def test_transform_item_rewrites_a_sycophancy_question(self):
        s = PromptSteerer.__new__(PromptSteerer)
        s.dataset = "Sycophancy"
        q, syco, honest = s.transform_item(("Q?", " (A)", " (B)"))
        self.assertTrue(q.startswith(SYCOPHANCY))
        self.assertTrue(q.endswith("Q?"))
        self.assertEqual((syco, honest), (" (A)", " (B)"))

    def test_transform_item_rewrites_a_plain_prompt(self):
        s = PromptSteerer.__new__(PromptSteerer)
        s.dataset = "RTP"
        self.assertEqual(s.transform_item("hello"), TOXICITY + "hello")

    def test_coefficient_switches_the_instruction_on_and_off(self):
        """lambda is a binary switch for this method: 0 is the unsteered
        control, non-zero prepends the instruction."""
        s = PromptSteerer.__new__(PromptSteerer)
        s.dataset = "RTP"
        seen = []

        def recorder(self, prompts, coefficient, seed, *a, **k):
            seen.append(list(prompts))
            return ["CONT"] * len(prompts)

        with unittest.mock.patch.object(CircuitSteer, "generate_batch",
                                        recorder):
            s.generate_batch(["alpha", "beta"], 0.0, seed=42)
            s.generate_batch(["alpha", "beta"], -5.0, seed=42)

        self.assertEqual(seen[0], ["alpha", "beta"])          # control
        self.assertTrue(all(p.startswith(TOXICITY) for p in seen[1]))
        self.assertTrue(seen[1][0].endswith("alpha"))

    def test_single_generate_also_switches(self):
        s = PromptSteerer.__new__(PromptSteerer)
        s.dataset = "Emotion"
        seen = []

        def recorder(self, prompt, coefficient, seed, *a, **k):
            seen.append(prompt)
            return "CONT"

        with unittest.mock.patch.object(CircuitSteer, "generate", recorder):
            s.generate("alpha", 0.0, seed=42)
            s.generate("alpha", -5.0, seed=42)

        self.assertEqual(seen[0], "alpha")
        self.assertEqual(seen[1], EMOTION + "alpha")


if __name__ == "__main__":
    unittest.main()
