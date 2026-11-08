import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw_runs.json"
SUMMARY = ROOT / "results" / "summary.json"


def load_make_results():
    spec = importlib.util.spec_from_file_location(
        "make_results", ROOT / "scripts" / "make_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(RAW.exists() and SUMMARY.exists(),
                     "results not yet committed")
class TestResultsRecompute(unittest.TestCase):
    def test_summary_recomputes_from_raw(self):
        mod = load_make_results()
        raw = json.loads(RAW.read_text())
        committed = json.loads(SUMMARY.read_text())
        self.assertEqual(mod.compute(raw), committed)

    def test_raw_record_names_device_and_machine(self):
        raw = json.loads(RAW.read_text())
        self.assertEqual(raw["machine"], "BlackBox")
        self.assertIn("RTX 5050 Laptop", raw["device"])

    def test_every_timed_cell_has_25_calls(self):
        raw = json.loads(RAW.read_text())
        for cols, arms in raw["results_by_size"].items():
            for arm, record in arms.items():
                self.assertEqual(len(record["times_ms"]), 25,
                                 f"{cols} {arm}")

    def test_results_md_carries_verdicts_and_device(self):
        text = (ROOT / "RESULTS.md").read_text()
        raw = json.loads(RAW.read_text())
        self.assertIn(raw["device"], text)
        summary = json.loads(SUMMARY.read_text())
        for verdict in summary["verdicts"].values():
            self.assertIn(verdict, text)


if __name__ == "__main__":
    unittest.main()
