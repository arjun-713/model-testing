from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from deepeval import assert_test

from evals.metrics import build_metrics
from evals.run_evaluation import build_test_cases, load_json_array


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for the shard eval suite.")
    return value


RESPONSES_FILE = Path(required_env("EVAL_RESPONSES_FILE"))
GOLDENS_FILE = Path(required_env("EVAL_GOLDENS_FILE"))
RESULT_DIR = Path(required_env("EVAL_RESULT_DIR"))
RESPONSE_MODEL = required_env("EVAL_RESPONSE_MODEL")
JUDGE_MODEL = required_env("EVAL_JUDGE_MODEL")
THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", "0.5"))
INCLUDE_REASON = os.environ.get("EVAL_INCLUDE_REASON", "false").lower() == "true"

TEST_CASES = build_test_cases(
    load_json_array(RESPONSES_FILE),
    load_json_array(GOLDENS_FILE),
    RESPONSE_MODEL,
    expected_count=len(load_json_array(RESPONSES_FILE)),
)


def metric_payload(metric: Any, caught_error: str | None) -> dict[str, Any]:
    score = getattr(metric, "score", None)
    return {
        "name": metric.__name__,
        "score": score,
        "success": getattr(metric, "success", None),
        "threshold": getattr(metric, "threshold", None),
        "reason": getattr(metric, "reason", None),
        "error": getattr(metric, "error", None)
        or (caught_error if not isinstance(score, (int, float)) else None),
        "evaluation_model": getattr(metric, "evaluation_model", None),
    }


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda case: str(case.name))
def test_shard_response(test_case) -> None:
    metrics = build_metrics(
        JUDGE_MODEL,
        os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        THRESHOLD,
        include_reason=INCLUDE_REASON,
        async_mode=False,
    )
    caught_error: str | None = None
    try:
        # xdist supplies the two-way case concurrency. Metrics remain sequential
        # inside each case so Ollama receives at most two judge requests.
        assert_test(test_case=test_case, metrics=metrics, run_async=False)
    except Exception as exc:  # Scores and provider errors are exported below.
        caught_error = f"{type(exc).__name__}: {exc}"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "name": str(test_case.name),
        "input": test_case.input,
        "actual_output": test_case.actual_output,
        "expected_output": test_case.expected_output,
        "metadata": test_case.metadata,
        "metrics_data": [
            metric_payload(metric, caught_error) for metric in metrics
        ],
    }
    output = RESULT_DIR / f"{test_case.name}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
