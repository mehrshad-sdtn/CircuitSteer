import unittest
from unittest.mock import patch

from circuitsteer.data import _split, make_datasets, text_of


class DatasetUtilityTests(unittest.TestCase):
    def test_text_of_supports_strings_and_sycophancy_tuples(self):
        self.assertEqual(text_of("plain prompt"), "plain prompt")
        self.assertEqual(text_of(("question", " (A)")), "question")

    def test_split_is_seeded_and_keeps_pairs_aligned(self):
        toxic = [f"toxic-{index}" for index in range(10)]
        benign = [f"benign-{index}" for index in range(10)]
        first = _split(toxic, benign, fraction=0.8, seed=42)
        second = _split(toxic, benign, fraction=0.8, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train_toxic"]), 8)
        self.assertEqual(len(first["test_prompts"]), 2)
        for toxic_item, benign_item in zip(
            first["train_toxic"],
            first["train_benign"],
        ):
            self.assertEqual(
                toxic_item.rsplit("-", 1)[1],
                benign_item.rsplit("-", 1)[1],
            )

    def test_make_datasets_loads_only_requested_tasks(self):
        fake_split = {
            "train_toxic": ["t"],
            "train_benign": ["b"],
            "test_prompts": ["x"],
        }
        fake_loader = unittest.mock.Mock(return_value=fake_split)
        with patch.dict(
            "circuitsteer.data.LOADERS",
            {"Fake": fake_loader},
            clear=True,
        ):
            result = make_datasets(["Fake"], n=12, seed=7)

        self.assertEqual(result, {"Fake": fake_split})
        fake_loader.assert_called_once_with(n=12, seed=7)

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            make_datasets(["not-a-dataset"])


if __name__ == "__main__":
    unittest.main()
