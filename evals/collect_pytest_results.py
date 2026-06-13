#!/usr/bin/env python3
"""Collect per-case DeepEval pytest results into the shard artifact contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evals.run_evaluation import (
    build_test_cases,
    load_json_array,
    save_json,
    summarize_result,
    utc_now,
    write_report,
)


def collect(args: argparse.Namespace) -> int:
    responses = load_json_array(args.responses)
    goldens = load_json_array(args.goldens)
    test_cases = build_test_cases(
        responses,
        goldens,
        args.response_model,
        expected_count=args.expected_count,
    )
    result_files = sorted(args.result_dir.glob("q-*.json"))
    test_results = [
        json.loads(path.read_text(encoding="utf-8")) for path in result_files
    ]
    raw_result = {"test_results": test_results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "deepeval-result.json", raw_result)
    summary, errors = summarize_result(
        raw_result=raw_result,
        response_model=args.response_model,
        judge_model=args.judge_model,
        threshold=args.threshold,
        expected_count=args.expected_count,
        started_at=args.started_at,
        duration_seconds=args.duration_seconds,
        confident_enabled=bool(os.environ.get("CONFIDENT_API_KEY")),
        include_reason=args.include_reason,
        max_concurrent=args.workers,
        expected_cases=test_cases,
    )
    summary["execution_mode"] = "deepeval test run"
    summary["completed_at"] = utc_now()
    save_json(args.output_dir / "evaluation-summary.json", summary)
    write_report(summary, args.output_dir / "evaluation-report.md")
    print(
        f"Collected {len(test_results)}/{args.expected_count} pytest eval results "
        f"with {len(errors)} missing or invalid metric entries."
    )
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument(
        "--include-reason",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(collect(parse_args()))

