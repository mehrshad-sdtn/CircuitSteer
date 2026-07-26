import tempfile
import unittest
from pathlib import Path

from circuitsteer.io import append_row, read_rows, write_rows


class CsvUtilityTests(unittest.TestCase):
    def test_append_read_and_replace_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.csv"
            fields = ["model", "score"]

            append_row(path, {"model": "gemma", "score": 0.5}, fields)
            append_row(path, {"model": "llama", "score": 0.7}, fields)
            rows = read_rows(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["model"], "gemma")

            write_rows(
                path,
                [{"model": "gemma", "score": 0.9}],
                fields,
            )
            replaced = read_rows(path)
            self.assertEqual(replaced, [{"model": "gemma", "score": "0.9"}])


if __name__ == "__main__":
    unittest.main()
