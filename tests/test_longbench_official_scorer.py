import importlib.util
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "score_longbench_official.py"
SPEC = importlib.util.spec_from_file_location("score_longbench_official", SCRIPT_PATH)
longbench = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["score_longbench_official"] = longbench
SPEC.loader.exec_module(longbench)


class LongBenchOfficialScorerTest(unittest.TestCase):
    def test_qa_f1_score(self):
        score = longbench.qa_f1_score("The answer is South West", "South West", {})
        self.assertGreater(score, 0.6)

    def test_count_score_penalizes_extra_numbers(self):
        self.assertEqual(longbench.count_score("There are 3 passages.", "3", {}), 1.0)
        self.assertEqual(longbench.count_score("3 or 4", "3", {}), 0.5)

    def test_score_rows_uses_task_metric(self):
        summary = longbench.score_rows(
            [
                {
                    "benchmark": "longbench",
                    "task": "passage_count",
                    "item_id": "a",
                    "response": "3",
                    "expected": ["3"],
                    "ok": True,
                },
                {
                    "benchmark": "longbench",
                    "task": "hotpotqa",
                    "item_id": "b",
                    "response": "Paris",
                    "expected": ["Paris"],
                    "ok": True,
                },
            ]
        )
        self.assertEqual(summary["scored_items"], 2)
        self.assertAlmostEqual(summary["by_task"]["passage_count"]["score_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
