#!/usr/bin/env python3
"""Compare benchmark summaries for multiple inference backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.constants import METRIC_NAMES


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_compare_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Backend benchmark comparison",
        "",
        "| Backend | Response model | Judge model | Prep seconds | Generation seconds | Evaluation seconds | Gate |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.get('backend')} | {summary.get('response_model')} | "
            f"{summary.get('judge_model')} | {summary.get('model_prep_total_seconds')} | "
            f"{summary.get('generation_total_seconds')} | "
            f"{summary.get('evaluation_total_seconds')} | "
            f"{'PASS' if summary.get('gates', {}).get('passed') else 'FAIL'} |"
        )

    for metric_name in METRIC_NAMES:
        lines.extend(
            [
                "",
                f"## {metric_name}",
                "",
                "| Backend | Average | Coverage | Pass rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for summary in summaries:
            metric = summary.get("metrics", {}).get(metric_name, {})
            lines.append(
                f"| {summary.get('backend')} | {metric.get('average_score')} | "
                f"{metric.get('coverage')} | {metric.get('pass_rate')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = [load_json(path) for path in args.summary]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "backend-compare.json", summaries)
    report = build_compare_report(summaries)
    (args.output_dir / "backend-compare.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
