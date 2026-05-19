import math
import unittest

from llm_quant_bench.metrics import (
    geometric_usability_score,
    perplexity_delta_percent,
    perplexity_from_token_logprobs,
    quantile,
    score_with_rules,
)


class MetricsTest(unittest.TestCase):
    def test_rule_score_json_and_contains(self):
        sample = {"require_json": True, "contains_all": ["answer"]}
        score = score_with_rules(sample, '{"answer": 42}', ok=True)
        self.assertEqual(score.score, 1.0)
        self.assertFalse(score.severe_error)

    def test_number_match(self):
        sample = {"expected_number": 180, "number_tolerance": 0}
        score = score_with_rules(sample, "The result is 180 parts.", ok=True)
        self.assertEqual(score.score, 1.0)

    def test_quantile(self):
        self.assertEqual(quantile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(quantile([1, 2, 3, 4, 5], 0.95), 4.8)

    def test_ppl(self):
        ppl = perplexity_from_token_logprobs([math.log(0.5), math.log(0.25)])
        self.assertAlmostEqual(ppl, math.sqrt(8))
        self.assertAlmostEqual(perplexity_delta_percent(10, 11), 10.0)

    def test_usability(self):
        self.assertAlmostEqual(
            geometric_usability_score(
                quality_retention=0.98,
                stability_rate=0.997,
                performance_attainment_rate=0.96,
                context_success_rate=0.99,
            ),
            0.98 * 0.997 * 0.96 * 0.99,
        )


if __name__ == "__main__":
    unittest.main()

