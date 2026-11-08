import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREREG = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")


class TestPreregistrationFrozen(unittest.TestCase):
    def test_hypothesis_lines_present(self):
        for needle in ("1.30", "0.85", "1.18", "1e-5"):
            self.assertIn(needle, PREREG)

    def test_ladder_and_verdict_size_present(self):
        self.assertIn("1024, 2048, 4096, 8192, 16384", PREREG)
        self.assertIn("Verdict size: 16384", PREREG)

    def test_device_named(self):
        self.assertIn("RTX 5050 Laptop", PREREG)

    def test_three_arms_named(self):
        for arm in ("TRITON", "EAGER", "COMPILED"):
            self.assertIn(arm, PREREG)


class TestNoEmDash(unittest.TestCase):
    def test_no_em_dash_in_text_files(self):
        offenders = []
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or not path.is_file():
                continue
            if path.suffix in (".md", ".py", ".toml", ".txt", ".yml"):
                if chr(0x2014) in path.read_text(encoding="utf-8",
                                                 errors="ignore"):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
