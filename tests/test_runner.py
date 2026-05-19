import unittest

from llm_quant_bench.runner import BenchmarkTargets, summarize_records


class RunnerSummaryTest(unittest.TestCase):
    def test_summary_includes_serving_and_category_metrics(self):
        records = [
            {
                "sample_id": "a",
                "category": "reasoning",
                "context_tokens": 8192,
                "baseline": {"ok": True, "quality_score": 1.0},
                "candidate": {
                    "ok": True,
                    "quality_score": 0.98,
                    "severe_error": False,
                    "ttft_s": 1.0,
                    "inter_token_latency_s": 0.1,
                    "time_per_output_token_s": 0.1,
                    "tokens_per_second": 10.0,
                    "latency_s": 3.0,
                    "output_tokens": 20,
                    "objective_score": True,
                },
                "comparison": {"result": "tie"},
            },
            {
                "sample_id": "b",
                "category": "format",
                "context_tokens": 200,
                "baseline": {"ok": True, "quality_score": 1.0},
                "candidate": {
                    "ok": True,
                    "quality_score": 1.0,
                    "severe_error": False,
                    "ttft_s": 0.5,
                    "inter_token_latency_s": 0.05,
                    "time_per_output_token_s": 0.05,
                    "tokens_per_second": 20.0,
                    "latency_s": 1.5,
                    "output_tokens": 20,
                    "objective_score": True,
                },
                "comparison": {"result": "win"},
            },
        ]
        targets = BenchmarkTargets(
            max_p95_ttft_s=2.0,
            max_p95_inter_token_latency_s=0.2,
            max_p95_time_per_output_token_s=0.2,
            min_p05_tokens_per_second=5.0,
        )
        summary = summarize_records(records, targets, benchmark_duration_s=10.0)

        self.assertAlmostEqual(summary["comparison"]["quality_retention_capped"], 0.99)
        self.assertAlmostEqual(summary["candidate"]["output_token_throughput"], 4.0)
        self.assertAlmostEqual(summary["candidate"]["request_throughput"], 0.2)
        self.assertTrue(summary["passed"]["p95_inter_token_latency_s"])
        self.assertTrue(summary["passed"]["weakly_scored_rate"])
        self.assertEqual(summary["by_category"]["reasoning"]["pairs"], 1)

    def test_weakly_scored_records_fail_by_default(self):
        records = [
            {
                "sample_id": "a",
                "category": "open",
                "context_tokens": 100,
                "baseline": {"ok": True, "quality_score": 1.0},
                "candidate": {
                    "ok": True,
                    "quality_score": 1.0,
                    "severe_error": False,
                    "output_tokens": 20,
                    "objective_score": False,
                },
                "comparison": {"result": "tie", "judge_used": False},
            }
        ]
        summary = summarize_records(records, BenchmarkTargets())
        self.assertEqual(summary["scoring"]["weakly_scored_rate"], 1.0)
        self.assertFalse(summary["passed"]["weakly_scored_rate"])


if __name__ == "__main__":
    unittest.main()
