import importlib.util
import pathlib
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_quality_retention.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_quality_retention", SCRIPT_PATH)
retention = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(retention)


class QualityRetentionScriptTest(unittest.TestCase):
    def test_retention_ratio(self):
        self.assertAlmostEqual(retention.compute_retention(0.78, 0.8), 0.975)

    def test_missing_or_zero_baseline_returns_none(self):
        self.assertIsNone(retention.compute_retention(0.78, 0.0))
        self.assertIsNone(retention.compute_retention(None, 0.8))


if __name__ == "__main__":
    unittest.main()
