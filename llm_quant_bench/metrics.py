from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any


@dataclass
class RuleScore:
    score: float
    checks: dict[str, bool]
    severe_error: bool
    objective: bool


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def score_with_rules(sample: dict[str, Any], text: str, *, ok: bool) -> RuleScore:
    if not ok or not text.strip():
        return RuleScore(
            score=0.0,
            checks={"non_empty": False},
            severe_error=True,
            objective=False,
        )

    checks: dict[str, bool] = {}

    expected = sample.get("expected")
    if expected is not None:
        checks["exact_match"] = normalize_text(text) == normalize_text(str(expected))

    expected_regex = sample.get("expected_regex")
    if expected_regex:
        checks["regex_match"] = re.search(str(expected_regex), text, re.DOTALL) is not None

    contains_all = sample.get("contains_all") or []
    if contains_all:
        lower_text = text.lower()
        checks["contains_all"] = all(str(part).lower() in lower_text for part in contains_all)

    if sample.get("require_json"):
        checks["json_valid"] = _is_json(text)

    if "expected_number" in sample:
        checks["number_match"] = _number_match(
            text,
            float(sample["expected_number"]),
            float(sample.get("number_tolerance", 0.0)),
        )

    min_chars = sample.get("min_chars")
    if min_chars is not None:
        checks["min_chars"] = len(text) >= int(min_chars)

    max_chars = sample.get("max_chars")
    if max_chars is not None:
        checks["max_chars"] = len(text) <= int(max_chars)

    reference = sample.get("reference") or sample.get("expected")
    if reference is not None and "exact_match" not in checks:
        checks["reference_similarity"] = similarity(text, str(reference)) >= float(
            sample.get("min_similarity", 0.72)
        )

    objective = bool(checks)
    if not checks:
        checks["non_empty"] = bool(text.strip())

    score = sum(1.0 for passed in checks.values() if passed) / len(checks)
    severe_error = _is_severe(sample, checks, score)
    return RuleScore(
        score=score,
        checks=checks,
        severe_error=severe_error,
        objective=objective,
    )


def compare_candidate_to_baseline(
    baseline_score: float,
    candidate_score: float,
    baseline_text: str,
    candidate_text: str,
    *,
    epsilon: float = 0.03,
) -> str:
    """Return win, tie, or lose without a judge model."""
    if candidate_score > baseline_score + epsilon:
        return "win"
    if candidate_score + epsilon < baseline_score:
        return "lose"
    if similarity(baseline_text, candidate_text) >= 0.5:
        return "tie"
    return "tie"


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def perplexity_from_token_logprobs(token_logprobs: list[float]) -> float:
    """PPL = exp(-mean(log p(token_i))). Logprobs must be natural-log values."""
    if not token_logprobs:
        return float("nan")
    return math.exp(-mean(token_logprobs))


def perplexity_delta_percent(baseline_ppl: float, candidate_ppl: float) -> float:
    """Relative PPL increase: (candidate - baseline) / baseline * 100."""
    if baseline_ppl <= 0 or math.isnan(baseline_ppl) or math.isnan(candidate_ppl):
        return float("nan")
    return (candidate_ppl - baseline_ppl) / baseline_ppl * 100.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def safe_ratio(numerator: float, denominator: float, *, cap: float | None = None) -> float:
    if denominator <= 0:
        value = 1.0 if numerator >= denominator else 0.0
    else:
        value = numerator / denominator
    if cap is not None:
        value = min(value, cap)
    return value


def geometric_usability_score(
    *,
    quality_retention: float,
    stability_rate: float,
    performance_attainment_rate: float,
    context_success_rate: float,
) -> float:
    return (
        quality_retention
        * stability_rate
        * performance_attainment_rate
        * context_success_rate
    )


def _is_json(text: str) -> bool:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        json.loads(cleaned)
        return True
    except json.JSONDecodeError:
        return False


def _number_match(text: str, expected: float, tolerance: float) -> bool:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    for number in numbers:
        if abs(float(number) - expected) <= tolerance:
            return True
    return False


def _is_severe(sample: dict[str, Any], checks: dict[str, bool], score: float) -> bool:
    severity = str(sample.get("severity", "normal")).lower()
    if severity in {"critical", "high"} and score < 1.0:
        return True
    if sample.get("require_json") and checks.get("json_valid") is False:
        return True
    return False
