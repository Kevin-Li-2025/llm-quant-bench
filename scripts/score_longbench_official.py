#!/usr/bin/env python3
"""Score LongBench v1 samples with task-specific metrics.

This scorer consumes the `samples.jsonl` written by `scripts/run_quality_eval.py`
and applies the LongBench v1 dataset-to-metric mapping: QA F1, ROUGE-L,
classification, retrieval/count, and code similarity depending on the task.

The implementation is dependency-light so it can run on benchmark machines
without installing the original LongBench package. It follows the official task
metric mapping, but ROUGE-L and code similarity are local Python equivalents, so
publishable leaderboard claims should still include the exact scorer revision.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import string
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


MetricFn = Callable[[str, str, dict[str, Any]], float]


DATASET_TO_METRIC: dict[str, str] = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_zh",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_similarity",
    "repobench-p": "code_similarity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a LongBench task is not in the official v1 metric map.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_samples(args.samples)
    summary = score_rows(rows, strict=args.strict)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "longbench_official_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "longbench_official_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def read_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("benchmark") in {None, "longbench"}:
                    rows.append(row)
    return rows


def score_rows(rows: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    scored_rows: list[dict[str, Any]] = []
    unsupported: dict[str, int] = {}

    for row in rows:
        task = str(row.get("task") or "")
        metric_name = DATASET_TO_METRIC.get(task)
        if metric_name is None:
            unsupported[task] = unsupported.get(task, 0) + 1
            if strict:
                raise SystemExit(f"Unsupported LongBench task for official scoring: {task}")
            continue

        prediction = str(row.get("response") or row.get("prediction") or "")
        answers = row.get("expected")
        if not isinstance(answers, list):
            answers = [answers]
        if task in {"trec", "triviaqa", "samsum", "lsht"}:
            prediction = prediction.lstrip("\n").split("\n")[0]

        context = {"all_classes": row.get("all_classes") or []}
        answer_scores = [
            metric(metric_name, prediction, str(answer), context)
            for answer in answers
            if answer is not None
        ]
        score = max(answer_scores) if answer_scores else 0.0
        scored_rows.append(
            {
                "item_id": row.get("item_id"),
                "task": task,
                "metric": metric_name,
                "score": score,
                "score_percent": 100.0 * score,
                "ok": bool(row.get("ok", True)),
            }
        )

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({row["task"] for row in scored_rows}):
        group = [row for row in scored_rows if row["task"] == task]
        scores = [float(row["score"]) for row in group]
        by_task[task] = {
            "items": len(group),
            "metric": group[0]["metric"] if group else None,
            "score_mean": statistics.fmean(scores) if scores else None,
            "score_percent": 100.0 * statistics.fmean(scores) if scores else None,
        }

    task_scores = [
        row["score_percent"]
        for row in by_task.values()
        if row.get("score_percent") is not None
    ]
    micro_scores = [float(row["score"]) for row in scored_rows]
    return {
        "profile": "longbench_v1_task_metrics_dependency_light",
        "items": len(rows),
        "scored_items": len(scored_rows),
        "unsupported_tasks": unsupported,
        "macro_score_percent": statistics.fmean(task_scores) if task_scores else None,
        "micro_score_percent": 100.0 * statistics.fmean(micro_scores) if micro_scores else None,
        "by_task": by_task,
        "notes": [
            "Task-to-metric mapping follows LongBench v1.",
            "ROUGE-L is computed with a local LCS F-measure implementation.",
            "Code similarity uses difflib SequenceMatcher rather than fuzzywuzzy.",
            "Use the exact official LongBench repository revision for leaderboard claims.",
        ],
    }


def metric(name: str, prediction: str, ground_truth: str, context: dict[str, Any]) -> float:
    metrics: dict[str, MetricFn] = {
        "qa_f1": qa_f1_score,
        "qa_f1_zh": qa_f1_zh_score,
        "rouge_l": rouge_l_score,
        "rouge_zh": rouge_zh_score,
        "classification": classification_score,
        "retrieval": retrieval_score,
        "retrieval_zh": retrieval_zh_score,
        "count": count_score,
        "code_similarity": code_similarity_score,
    }
    return metrics[name](prediction, ground_truth, context)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(char for char in value if char not in set(string.punctuation))

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def normalize_zh_answer(text: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punctuation = set(string.punctuation + cn_punctuation)
    return "".join(char for char in text.lower() if char not in punctuation and not char.isspace())


def qa_f1_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    return f1_score(list(normalize_zh_answer(prediction)), list(normalize_zh_answer(ground_truth)))


def f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_zh_score(prediction: str, ground_truth: str, context: dict[str, Any]) -> float:
    return rouge_l_score(" ".join(normalize_zh_answer(prediction)), " ".join(normalize_zh_answer(ground_truth)), context)


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = [0] * (len(short) + 1)
    for long_token in long:
        current = [0]
        for idx, short_token in enumerate(short, start=1):
            if long_token == short_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def classification_score(prediction: str, ground_truth: str, context: dict[str, Any]) -> float:
    classes = [str(item) for item in context.get("all_classes") or []]
    if not classes:
        return 1.0 if ground_truth in prediction else 0.0
    matches = [class_name for class_name in classes if class_name in prediction]
    matches = [
        match
        for match in matches
        if not (match in ground_truth and match != ground_truth)
    ]
    return (1.0 / len(matches)) if ground_truth in matches and matches else 0.0


def retrieval_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    match = re.search(r"Paragraph (\d+)", ground_truth)
    target = match.group(1) if match else ground_truth
    return number_hit_rate(prediction, target)


def retrieval_zh_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    match = re.search(r"段落(\d+)", ground_truth)
    target = match.group(1) if match else ground_truth
    return number_hit_rate(prediction, target)


def count_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    return number_hit_rate(prediction, ground_truth)


def number_hit_rate(prediction: str, target: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if str(number) == str(target)) / len(numbers)


def code_similarity_score(prediction: str, ground_truth: str, _: dict[str, Any]) -> float:
    first_code_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            first_code_line = line
            break
    return SequenceMatcher(None, first_code_line, ground_truth).ratio()


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# LongBench Official-Metric Report",
        "",
        f"- Profile: {summary['profile']}",
        f"- Items: {summary['items']}",
        f"- Scored items: {summary['scored_items']}",
        f"- Macro score: {fmt(summary.get('macro_score_percent'))}%",
        f"- Micro score: {fmt(summary.get('micro_score_percent'))}%",
        "",
        "| Task | Items | Metric | Score |",
        "|---|---:|---|---:|",
    ]
    for task, row in sorted(summary["by_task"].items()):
        lines.append(
            f"| {task} | {row['items']} | {row['metric']} | {fmt(row.get('score_percent'))}% |"
        )
    if summary.get("unsupported_tasks"):
        lines.extend(["", "## Unsupported Tasks", ""])
        for task, count in sorted(summary["unsupported_tasks"].items()):
            lines.append(f"- {task}: {count}")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
