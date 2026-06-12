#!/usr/bin/env python3
"""Retry only missing DeepEval metric scores and merge recovered results."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.constants import METRIC_NAMES
from evals.metrics import build_metrics
from evals.run_evaluation import build_test_cases, load_json_array, save_json


def metric_payload(metric: Any, error: str | None = None) -> dict[str, Any]:
    return {
        "score": getattr(metric, "score", None),
        "success": getattr(metric, "success", None),
        "threshold": getattr(metric, "threshold", None),
        "reason": getattr(metric, "reason", None),
        "error": error or getattr(metric, "error", None),
        "evaluation_model": getattr(metric, "evaluation_model", None),
    }


def missing_pairs(summary: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for case in summary.get("cases", []):
        case_id = case.get("id")
        if not isinstance(case_id, str):
            continue
        metrics = case.get("metrics", {})
        for name in METRIC_NAMES:
            score = metrics.get(name, {}).get("score")
            if not isinstance(score, (int, float)):
                pairs.append((case_id, name))
    return pairs


def recompute_summary(summary: dict[str, Any]) -> None:
    threshold = float(summary["threshold"])
    errors: list[str] = []
    for name in METRIC_NAMES:
        scores: list[float] = []
        pass_count = 0
        for case in summary.get("cases", []):
            case_id = str(case.get("id", "unknown"))
            metric = case.get("metrics", {}).get(name)
            if not isinstance(metric, dict):
                errors.append(f"{case_id}: {name} result is missing.")
                continue
            score = metric.get("score")
            if isinstance(score, (int, float)):
                numeric_score = float(score)
                scores.append(numeric_score)
                pass_count += int(numeric_score >= threshold)
            else:
                error = metric.get("error") or "no numeric score"
                errors.append(f"{case_id}: {name} failed: {error}")
        summary["metrics"][name] = {
            "average_score": round(sum(scores) / len(scores), 4) if scores else None,
            "minimum_score": round(min(scores), 4) if scores else None,
            "maximum_score": round(max(scores), 4) if scores else None,
            "pass_count": pass_count,
            "evaluated_count": len(scores),
            "pass_rate": round(pass_count / len(scores), 4) if scores else None,
        }
    summary["errors"] = errors


def retry(args: argparse.Namespace) -> int:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    responses = load_json_array(args.responses)
    goldens = load_json_array(args.goldens)
    test_cases = build_test_cases(
        responses,
        goldens,
        args.response_model,
        expected_count=len(responses),
    )
    cases_by_id = {str(case.name): case for case in test_cases}
    summary_cases = {
        str(case.get("id")): case for case in summary.get("cases", [])
    }
    for test_case in test_cases:
        case_id = str(test_case.name)
        if case_id in summary_cases:
            continue
        case = {
            "id": case_id,
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "metrics": {},
        }
        summary.setdefault("cases", []).append(case)
        summary_cases[case_id] = case
    pairs = missing_pairs(summary)
    retry_log: list[dict[str, Any]] = []
    started = time.perf_counter()

    def measure_pair(pair: tuple[str, str]) -> tuple[str, str, dict[str, Any]]:
        case_id, metric_name = pair
        metric = build_metrics(
            args.judge_model,
            args.base_url,
            args.threshold,
            include_reason=args.include_reason,
            metric_names=(metric_name,),
        )[0]
        error: str | None = None
        try:
            metric.measure(cases_by_id[case_id])
        except Exception as exc:  # DeepEval provider errors vary by metric.
            error = f"{type(exc).__name__}: {exc}"
        payload = metric_payload(metric, error)
        return case_id, metric_name, payload

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        measured = list(executor.map(measure_pair, pairs))

    for case_id, metric_name, payload in measured:
        summary_cases[case_id].setdefault("metrics", {})[metric_name] = payload
        retry_log.append(
            {
                "id": case_id,
                "metric": metric_name,
                "score": payload["score"],
                "error": payload["error"],
            }
        )

    recompute_summary(summary)
    summary["retry"] = {
        "attempted_count": len(pairs),
        "recovered_count": sum(
            isinstance(item["score"], (int, float)) for item in retry_log
        ),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": retry_log,
    }
    save_json(args.summary, summary)
    save_json(args.output, summary["retry"])
    print(
        f"Retried {len(pairs)} missing metric scores; "
        f"recovered {summary['retry']['recovered_count']}."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument(
        "--include-reason",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
    )
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    return args


if __name__ == "__main__":
    raise SystemExit(retry(parse_args()))
