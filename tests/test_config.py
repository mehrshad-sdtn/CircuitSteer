import unittest

from circuitsteer.config import CircuitConfig, MODEL_CONFIGS


class CircuitConfigTests(unittest.TestCase):
    def test_paper_defaults(self):
        config = CircuitConfig()
        self.assertEqual(config.sim_thresh, 0.10)
        self.assertEqual(config.act_thresh, 1.5)
        self.assertEqual(config.diff_thresh, 0.05)
        self.assertEqual(config.top_k, 30)
        self.assertEqual(config.steer_pool, "sum")

    def test_models_have_matching_layer_configuration(self):
        self.assertEqual(MODEL_CONFIGS["gemma"].sae_layers, (6, 12, 18, 24))
        self.assertEqual(
            MODEL_CONFIGS["llama"].sae_layers,
            (3, 8, 16, 24, 29),
        )

    def test_invalid_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            CircuitConfig(steer_pool="median")

    def test_nonpositive_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            CircuitConfig(top_k=0)


if __name__ == "__main__":
    unittest.main()
