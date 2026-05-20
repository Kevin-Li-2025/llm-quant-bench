import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_ablation_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_ablation_matrix", SCRIPT_PATH)
ablation = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ablation)


class AblationMatrixScriptTest(unittest.TestCase):
    def test_safe_name_replaces_spaces(self):
        self.assertEqual(ablation.safe_name("vLLM AWQ/Marlin"), "vLLM_AWQ_Marlin")

    def test_render_report_includes_metrics(self):
        report = ablation.render_report(
            {
                "experiments": [
                    {
                        "name": "run-a",
                        "runtime": "vLLM",
                        "quant": "AWQ",
                        "status": "ok",
                        "load_summary": {
                            "tokens": {"output_token_throughput": 10.0},
                            "requests": {"success_rate": 1.0},
                        },
                        "quality_summary": {
                            "by_benchmark": {
                                "mmlu": {"score_mean": 0.8},
                                "gsm8k": {"score_mean": 0.6},
                            }
                        },
                    }
                ]
            }
        )
        self.assertIn("run-a", report)
        self.assertIn("0.7000", report)


if __name__ == "__main__":
    unittest.main()
