import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "score_mt_bench.py"
SPEC = importlib.util.spec_from_file_location("score_mt_bench", SCRIPT_PATH)
mt = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(mt)


class MtBenchJudgeScriptTest(unittest.TestCase):
    def test_parse_judge_json_inside_text(self):
        parsed = mt.parse_judge_json('Result:\n{"score": 8, "rationale": "good"}')
        self.assertEqual(parsed["score"], 8)

    def test_normalize_swapped_winner(self):
        self.assertEqual(mt.normalize_winner("A", swapped=True), "candidate")
        self.assertEqual(mt.normalize_winner("B", swapped=True), "baseline")
        self.assertEqual(mt.normalize_winner("tie", swapped=False), "tie")

    def test_pairwise_summary_counts_candidate_wins(self):
        summary = mt.summarize_judgments(
            [
                {"ok": True, "winner": "candidate", "judge_consistent": True},
                {"ok": True, "winner": "tie", "judge_consistent": True},
                {"ok": True, "winner": "baseline", "judge_consistent": False},
            ],
            "pairwise",
        )
        self.assertEqual(summary["candidate_wins"], 1)
        self.assertEqual(summary["ties"], 1)
        self.assertAlmostEqual(summary["candidate_win_tie_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
