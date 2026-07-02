#!/usr/bin/env python3
"""Evaluate generated responses with DeepEval and a local Ollama judge."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.test_case import LLMTestCase

from evals.constants import METRIC_NAMES
from evals.metrics import build_metrics


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array.")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain only JSON objects.")
    return value


def build_test_cases(
    responses: list[dict[str, Any]],
    goldens: list[dict[str, Any]],
    response_model: str,
    expected_count: int,
) -> list[LLMTestCase]:
    if len(responses) != expected_count:
        raise ValueError(
            f"Expected {expected_count} generated responses, found {len(responses)}."
        )

    goldens_by_id: dict[str, dict[str, Any]] = {}
    for golden in goldens:
        metadata = golden.get("additional_metadata")
        golden_id = metadata.get("id") if isinstance(metadata, dict) else None
        if not isinstance(golden_id, str) or not golden_id:
            raise ValueError("Every golden must contain additional_metadata.id.")
        if golden_id in goldens_by_id:
            raise ValueError(f"Duplicate golden id: {golden_id}.")
        goldens_by_id[golden_id] = golden

    test_cases: list[LLMTestCase] = []
    seen_ids: set[str] = set()
    for response in responses:
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("Every response must contain a valid id.")
        if response_id in seen_ids:
            raise ValueError(f"Duplicate response id: {response_id}.")
        seen_ids.add(response_id)

        golden = goldens_by_id.get(response_id)
        if golden is None:
            raise ValueError(f"No golden found for response {response_id}.")
        if response.get("input") != golden.get("input"):
            raise ValueError(f"Input mismatch for {response_id}.")

        actual_output = response.get("actual_output")
        retrieval_context = response.get("retrieval_context")
        expected_output = golden.get("expected_output")
        if not isinstance(actual_output, str) or not actual_output.strip():
            raise ValueError(f"{response_id}: actual_output is empty.")
        if not isinstance(expected_output, str) or not expected_output.strip():
            raise ValueError(f"{response_id}: expected_output is empty.")
        valid_retrieval_context = (
            isinstance(retrieval_context, list)
            and all(isinstance(context, str) for context in retrieval_context)
            and any(context.strip() for context in retrieval_context)
        )
        if not valid_retrieval_context:
            golden_context = golden.get("context")
            if (
                isinstance(golden_context, list)
                and all(isinstance(context, str) for context in golden_context)
                and any(context.strip() for context in golden_context)
            ):
                retrieval_context = golden_context
            else:
                raise ValueError(
                    f"{response_id}: retrieval_context is missing or empty and "
                    "the golden has no valid context fallback."
                )
        retrieval_context = [
            context.strip() for context in retrieval_context if context.strip()
        ]

        golden_metadata = golden.get("additional_metadata")
        category = (
            golden_metadata.get("category")
            if isinstance(golden_metadata, dict)
            else None
        )
        test_cases.append(
            LLMTestCase(
                name=response_id,
                input=response["input"],
                actual_output=actual_output.strip(),
                expected_output=expected_output.strip(),
                retrieval_context=retrieval_context,
                metadata={
                    "id": response_id,
                    "category": category,
                    "response_model": response_model,
                    "retrieval_context_count": len(retrieval_context),
                    "retrieval_context_characters": sum(
                        len(context) for context in retrieval_context
                    ),
                },
                tags=[response_model, str(category or "uncategorized")],
            )
        )
    return test_cases


def summarize_result(
    raw_result: dict[str, Any],
    response_model: str,
    judge_model: str,
    threshold: float,
    expected_count: int,
    started_at: str,
    duration_seconds: float,
    confident_enabled: bool,
    include_reason: bool,
    max_concurrent: int = 1,
    metric_names: tuple[str, ...] = METRIC_NAMES,
    expected_cases: list[LLMTestCase] | None = None,
    judge_backend: str = "ollama",
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    test_results = raw_result.get("test_results")
    if not isinstance(test_results, list):
        return {}, ["DeepEval result does not contain test_results."]
    if len(test_results) != expected_count:
        errors.append(
            f"Expected {expected_count} DeepEval test results, found {len(test_results)}."
        )

    metric_scores: dict[str, list[float]] = {name: [] for name in metric_names}
    metric_passes: dict[str, int] = {name: 0 for name in metric_names}
    cases: list[dict[str, Any]] = []

    results_by_id: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(test_results):
        metadata = result.get("metadata")
        case_id = metadata.get("id") if isinstance(metadata, dict) else None
        case_id = str(case_id or result.get("name") or f"index-{index}")
        results_by_id[case_id] = result

    ordered_results: list[tuple[str, dict[str, Any]]] = []
    if expected_cases is None:
        ordered_results = list(results_by_id.items())
    else:
        for case in expected_cases:
            case_id = str(case.name)
            result = results_by_id.get(case_id)
            if result is None:
                errors.append(f"{case_id}: DeepEval test result is missing.")
                result = {
                    "name": case_id,
                    "input": case.input,
                    "actual_output": case.actual_output,
                    "expected_output": case.expected_output,
                    "metadata": case.metadata,
                    "metrics_data": [],
                }
            ordered_results.append((case_id, result))

    for case_id, result in ordered_results:
        metadata = result.get("metadata")
        metric_data = result.get("metrics_data")
        if not isinstance(metric_data, list):
            errors.append(f"{case_id}: metrics_data is missing.")
            metric_data = []

        metrics_by_name = {
            metric.get("name"): metric
            for metric in metric_data
            if isinstance(metric, dict) and isinstance(metric.get("name"), str)
        }
        case_metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            metric = metrics_by_name.get(metric_name)
            if metric is None:
                errors.append(f"{case_id}: {metric_name} result is missing.")
                continue

            score = metric.get("score")
            metric_error = metric.get("error")
            if metric_error:
                errors.append(f"{case_id}: {metric_name} failed: {metric_error}")
            if not isinstance(score, (int, float)):
                errors.append(f"{case_id}: {metric_name} has no numeric score.")
            else:
                numeric_score = float(score)
                metric_scores[metric_name].append(numeric_score)
                if bool(metric.get("success")):
                    metric_passes[metric_name] += 1

            case_metrics[metric_name] = {
                "score": score,
                "success": metric.get("success"),
                "threshold": metric.get("threshold"),
                "reason": metric.get("reason"),
                "error": metric_error,
                "evaluation_model": metric.get("evaluation_model"),
            }

        cases.append(
            {
                "id": case_id,
                "input": result.get("input"),
                "actual_output": result.get("actual_output"),
                "expected_output": result.get("expected_output"),
                "retrieval_context_count": (
                    metadata.get("retrieval_context_count")
                    if isinstance(metadata, dict)
                    else None
                ),
                "retrieval_context_characters": (
                    metadata.get("retrieval_context_characters")
                    if isinstance(metadata, dict)
                    else None
                ),
                "metrics": case_metrics,
            }
        )

    metric_summary: dict[str, Any] = {}
    for metric_name in metric_names:
        scores = metric_scores[metric_name]
        metric_summary[metric_name] = {
            "average_score": round(statistics.fmean(scores), 4) if scores else None,
            "minimum_score": round(min(scores), 4) if scores else None,
            "maximum_score": round(max(scores), 4) if scores else None,
            "pass_count": metric_passes[metric_name],
            "evaluated_count": len(scores),
            "pass_rate": round(metric_passes[metric_name] / len(scores), 4)
            if scores
            else None,
        }

    generator_scores = [
        metric_summary[name]["average_score"]
        for name in ("Faithfulness", "Answer Relevancy")
        if metric_summary[name]["average_score"] is not None
    ]
    summary = {
        "response_model": response_model,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "threshold": threshold,
        "question_count": expected_count,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": round(duration_seconds, 3),
        "confident_ai_enabled": confident_enabled,
        "include_reason": include_reason,
        "max_concurrent": max_concurrent,
        "confident_link": raw_result.get("confident_link"),
        "test_run_id": raw_result.get("test_run_id"),
        "generator_average_score": round(statistics.fmean(generator_scores), 4)
        if generator_scores
        else None,
        "metrics": metric_summary,
        "cases": cases,
        "errors": errors,
    }
    return summary, errors


def write_report(summary: dict[str, Any], output_file: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        f"# DeepEval report: {summary['response_model']}",
        "",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Judge backend: `{summary.get('judge_backend', 'ollama')}`",
        f"- Questions: {summary['question_count']}",
        f"- Evaluation time: {summary['duration_seconds']} seconds",
        f"- Concurrent workers: {summary['max_concurrent']}",
        f"- Generator average: {summary['generator_average_score']}",
        f"- Confident AI report: {summary.get('confident_link') or 'not uploaded'}",
        "",
        "| Metric | Average | Min | Max | Pass rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric_name in METRIC_NAMES:
        metric = metrics[metric_name]
        lines.append(
            f"| {metric_name} | {metric['average_score']} | "
            f"{metric['minimum_score']} | {metric['maximum_score']} | "
            f"{metric['pass_rate']} |"
        )

    lines.extend(["", "## Per-question scores", ""])
    for case in summary["cases"]:
        score_text = ", ".join(
            f"{metric_name}: {metric.get('score')}"
            for metric_name, metric in case["metrics"].items()
        )
        lines.append(f"- `{case['id']}`: {score_text}")

    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        responses = load_json_array(args.responses)
        goldens = load_json_array(args.goldens)
        test_cases = build_test_cases(
            responses, goldens, args.response_model, args.expected_count
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        error_summary = {
            "response_model": args.response_model,
            "judge_model": args.judge_model,
            "judge_backend": args.judge_backend,
            "errors": [str(exc)],
        }
        save_json(args.output_dir / "evaluation-summary.json", error_summary)
        print(f"Evaluation setup failed: {exc}")
        return 1

    metrics = build_metrics(
        args.judge_model,
        args.base_url,
        args.threshold,
        include_reason=args.include_reason,
        judge_backend=args.judge_backend,
        # Two test cases run concurrently. Metrics inside each case stay
        # sequential so Ollama never receives six judge requests at once.
        async_mode=False,
    )
    started_at = utc_now()
    started = time.perf_counter()
    print(
        f"Evaluating {len(test_cases)} responses from {args.response_model} "
        f"with judge {args.judge_model} using "
        f"{args.max_concurrent} concurrent workers on backend {args.judge_backend}"
    )
    try:
        result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            identifier=args.identifier,
            hyperparameters={
                "response_model": args.response_model,
                "judge_model": args.judge_model,
                "judge_backend": args.judge_backend,
                "question_count": args.expected_count,
                "threshold": args.threshold,
                "include_reason": args.include_reason,
                "max_concurrent": args.max_concurrent,
            },
            async_config=AsyncConfig(
                run_async=args.max_concurrent > 1,
                max_concurrent=args.max_concurrent,
            ),
            display_config=DisplayConfig(
                show_indicator=True,
                print_results=True,
                verbose_mode=True,
                truncate_passing_cases=False,
                inspect_after_run=False,
                file_type="md",
                file_output_dir=str(args.output_dir),
            ),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            error_config=ErrorConfig(
                ignore_errors=True,
                skip_on_missing_params=False,
            ),
        )
    except Exception as exc:
        duration = time.perf_counter() - started
        error_summary = {
            "response_model": args.response_model,
            "judge_model": args.judge_model,
            "judge_backend": args.judge_backend,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": round(duration, 3),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        save_json(args.output_dir / "evaluation-summary.json", error_summary)
        print(f"DeepEval execution failed: {type(exc).__name__}: {exc}")
        return 1

    duration = time.perf_counter() - started
    raw_result = result.model_dump(mode="json")
    save_json(args.output_dir / "deepeval-result.json", raw_result)
    summary, errors = summarize_result(
        raw_result=raw_result,
        response_model=args.response_model,
        judge_model=args.judge_model,
        judge_backend=args.judge_backend,
        threshold=args.threshold,
        expected_count=args.expected_count,
        started_at=started_at,
        duration_seconds=duration,
        confident_enabled=bool(os.environ.get("CONFIDENT_API_KEY")),
        include_reason=args.include_reason,
        max_concurrent=args.max_concurrent,
        expected_cases=test_cases,
    )
    save_json(args.output_dir / "evaluation-summary.json", summary)
    write_report(summary, args.output_dir / "evaluation-report.md")

    if errors:
        print("Evaluation completed with missing or invalid metric results:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Evaluation passed: "
        f"generator_average_score={summary['generator_average_score']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--goldens", type=Path, default=Path("dataset/golden_dataset.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-model", required=True)
    parser.add_argument("--judge-model", default="gemma3:4b-it-qat")
    parser.add_argument(
        "--judge-backend",
        choices=("ollama", "airllm"),
        default="ollama",
    )
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-reason",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--identifier", required=True)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=int(os.environ.get("DEEPEVAL_MAX_CONCURRENT", "4")),
        help="Maximum DeepEval test cases evaluated concurrently.",
    )
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
