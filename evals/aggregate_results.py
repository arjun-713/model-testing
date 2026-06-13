#!/usr/bin/env python3
"""Aggregate response and DeepEval artifacts from independently run shards."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.constants import METRIC_NAMES


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def aggregate(
    artifact_dir: Path,
    dataset_file: Path,
    question_count: int,
    expected_shards: int,
    metric_threshold: float = 0.5,
    minimum_coverage: float = 0.9,
    minimum_pass_rate: float = 0.9,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_files = sorted(artifact_dir.glob("shard-*/evaluation-summary.json"))
    response_files = sorted(artifact_dir.glob("shard-*/responses.json"))
    generation_files = sorted(artifact_dir.glob("shard-*/summary.json"))
    if len(summary_files) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} evaluation summaries, found {len(summary_files)}"
        )
    if len(response_files) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} response files, found {len(response_files)}"
        )

    shard_summaries = [load_json(path) for path in summary_files]
    generation_summaries = [load_json(path) for path in generation_files]
    responses = [entry for path in response_files for entry in load_json(path)]
    source = load_json(dataset_file)[:question_count]
    source_order = {entry["id"]: index for index, entry in enumerate(source)}
    responses.sort(key=lambda entry: source_order.get(entry.get("id"), question_count))

    if len(responses) != question_count:
        raise ValueError(f"Expected {question_count} responses, found {len(responses)}")
    response_ids = [entry.get("id") for entry in responses]
    if len(set(response_ids)) != question_count:
        raise ValueError("Aggregated responses contain missing or duplicate IDs")

    cases = [case for summary in shard_summaries for case in summary.get("cases", [])]
    cases.sort(key=lambda case: source_order.get(case.get("id"), question_count))
    metrics: dict[str, Any] = {}
    for metric_name in METRIC_NAMES:
        scores: list[float] = []
        pass_count = 0
        for case in cases:
            metric = case.get("metrics", {}).get(metric_name, {})
            score = metric.get("score")
            if isinstance(score, (int, float)):
                numeric_score = float(score)
                scores.append(numeric_score)
                pass_count += int(numeric_score >= metric_threshold)
        metrics[metric_name] = {
            "average_score": round(statistics.fmean(scores), 4) if scores else None,
            "minimum_score": round(min(scores), 4) if scores else None,
            "maximum_score": round(max(scores), 4) if scores else None,
            "pass_count": pass_count,
            "evaluated_count": len(scores),
            "pass_rate": round(pass_count / len(scores), 4) if scores else None,
            "coverage": round(len(scores) / question_count, 4),
        }

    errors = [error for summary in shard_summaries for error in summary.get("errors", [])]
    missing_metrics: list[dict[str, Any]] = []
    for case in cases:
        for metric_name in METRIC_NAMES:
            metric = case.get("metrics", {}).get(metric_name, {})
            if not isinstance(metric.get("score"), (int, float)):
                missing_metrics.append(
                    {
                        "id": case.get("id"),
                        "metric": metric_name,
                        "retrieval_context_count": case.get(
                            "retrieval_context_count"
                        ),
                        "retrieval_context_characters": case.get(
                            "retrieval_context_characters"
                        ),
                        "error": metric.get("error"),
                    }
                )
    gate_failures: list[str] = []
    for name in METRIC_NAMES:
        metric = metrics[name]
        if metric["coverage"] < minimum_coverage:
            gate_failures.append(
                f"{name} coverage {metric['coverage']:.4f} is below {minimum_coverage:.4f}"
            )
        if metric["average_score"] is None or metric["average_score"] < metric_threshold:
            gate_failures.append(
                f"{name} average {metric['average_score']} is below {metric_threshold:.4f}"
            )
        if metric["pass_rate"] is None or metric["pass_rate"] < minimum_pass_rate:
            gate_failures.append(
                f"{name} pass rate {metric['pass_rate']} is below {minimum_pass_rate:.4f}"
            )

    summary = {
        "response_model": shard_summaries[0].get("response_model"),
        "judge_model": shard_summaries[0].get("judge_model"),
        "question_count": question_count,
        "shard_count": expected_shards,
        "include_reason": shard_summaries[0].get("include_reason"),
        "gates": {
            "metric_threshold": metric_threshold,
            "minimum_coverage": minimum_coverage,
            "minimum_pass_rate": minimum_pass_rate,
            "passed": not gate_failures,
            "failures": gate_failures,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "generation_total_seconds": round(
            sum(float(item.get("total_duration_seconds", 0)) for item in generation_summaries),
            3,
        ),
        "evaluation_total_seconds": round(
            sum(float(item.get("duration_seconds", 0)) for item in shard_summaries), 3
        ),
        "metrics": metrics,
        "cases": cases,
        "errors": errors,
        "missing_metrics": missing_metrics,
        "shards": shard_summaries,
    }
    return summary, responses


def write_report(summary: dict[str, Any], output_file: Path) -> None:
    lines = [
        "# Qwen response and Gemma judge report",
        "",
        f"- Response model: `{summary['response_model']}`",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Questions: {summary['question_count']}",
        f"- Shards: {summary['shard_count']}",
        f"- Judge reasons: {summary.get('include_reason')}",
        f"- Quality gate: {'PASS' if summary['gates']['passed'] else 'FAIL'}",
        f"- Sum of generation time: {summary['generation_total_seconds']} seconds",
        f"- Sum of evaluation time: {summary['evaluation_total_seconds']} seconds",
        "",
        "| Metric | Average | Min | Max | Coverage | Pass rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in METRIC_NAMES:
        metric = summary["metrics"][name]
        lines.append(
            f"| {name} | {metric['average_score']} | {metric['minimum_score']} | "
            f"{metric['maximum_score']} | {metric['coverage']} | "
            f"{metric['pass_rate']} |"
        )
    if summary["gates"]["failures"]:
        lines.extend(["", "## Quality gate failures", ""])
        lines.extend(f"- {failure}" for failure in summary["gates"]["failures"])
    lines.extend(["", "## Per-question scores", ""])
    for case in summary["cases"]:
        values = ", ".join(
            f"{name}: {case.get('metrics', {}).get(name, {}).get('score')}"
            for name in METRIC_NAMES
        )
        lines.append(f"- `{case.get('id')}`: {values}")
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    if summary["missing_metrics"]:
        lines.extend(["", "## Missing metric rows", ""])
        for item in summary["missing_metrics"]:
            lines.append(
                f"- `{item['id']}` / {item['metric']}: "
                f"contexts={item['retrieval_context_count']}, "
                f"characters={item['retrieval_context_characters']}, "
                f"error={item['error']}"
            )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--question-count", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-coverage", type=float, default=0.9)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary, responses = aggregate(
            args.artifact_dir,
            args.dataset,
            args.question_count,
            args.expected_shards,
            args.metric_threshold,
            args.minimum_coverage,
            args.minimum_pass_rate,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Aggregation failed: {exc}")
        return 1

    save_json(args.output_dir / "responses.json", responses)
    save_json(args.output_dir / "evaluation-summary.json", summary)
    write_report(summary, args.output_dir / "evaluation-report.md")
    if not summary["gates"]["passed"]:
        print("Aggregation failed the metric quality gates.")
        return 1
    if summary["errors"]:
        print("Aggregation passed with partial judge warnings.")
    else:
        print(f"Aggregated {summary['question_count']} evaluated responses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
