import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    ROOT / "main.py",
    ROOT / ".env.example",
    *sorted((ROOT / "circuitsteer").glob("*.py")),
    *sorted((ROOT / "experiments").glob("*.py")),
]

SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{10,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(
        r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)"
    ),
    "private key": re.compile(
        r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"
    ),
}


class RepositorySafetyTests(unittest.TestCase):
    def test_expected_project_files_exist(self):
        expected = [
            "README.md",
            "requirements.txt",
            ".gitignore",
            "circuitsteer/core.py",
            "experiments/run_main.py",
            "experiments/sensitivity.py",
            "experiments/mmlu.py",
            "experiments/gsm8k.py",
            "experiments/layer_count.py",
        ]
        missing = [path for path in expected if not (ROOT / path).exists()]
        self.assertEqual(missing, [])

    def test_no_credential_patterns_in_publishable_source(self):
        findings = []
        for path in SOURCE_PATHS:
            text = path.read_text(encoding="utf-8")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{path.relative_to(ROOT)}: {label}")
        self.assertEqual(findings, [])

    def test_no_colab_only_paths_or_inline_login(self):
        forbidden = (
            "/content/drive",
            "google.colab import drive",
            "login(token=",
        )
        findings = []
        for path in SOURCE_PATHS:
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                if value in text:
                    findings.append(
                        f"{path.relative_to(ROOT)} contains {value!r}"
                    )
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
