import importlib.util
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_quality_eval.py"
SPEC = importlib.util.spec_from_file_location("run_quality_eval", SCRIPT_PATH)
quality_eval = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["run_quality_eval"] = quality_eval
SPEC.loader.exec_module(quality_eval)


class QualityEvalScriptTest(unittest.TestCase):
    def test_extract_choice(self):
        self.assertEqual(quality_eval.extract_choice("Answer: C"), "C")
        self.assertEqual(quality_eval.extract_choice("答案是 B。"), "B")

    def test_extract_gsm8k_answer(self):
        answer = "Work here.\n#### 1,234"
        self.assertEqual(quality_eval.extract_gsm8k_answer(answer), "1234")

    def test_token_f1(self):
        self.assertAlmostEqual(quality_eval.token_f1("South West Ultras", "South West Ultras fan club"), 0.75)

    def test_cap_per_key(self):
        items = [
            quality_eval.EvalItem("bench", "a", "1", "p", "A", "choice_accuracy", 4),
            quality_eval.EvalItem("bench", "a", "2", "p", "A", "choice_accuracy", 4),
            quality_eval.EvalItem("bench", "b", "3", "p", "A", "choice_accuracy", 4),
        ]
        capped = quality_eval.cap_per_key(items, "task", 1)
        self.assertEqual([item.item_id for item in capped], ["1", "3"])


if __name__ == "__main__":
    unittest.main()
