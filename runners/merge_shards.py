#!/usr/bin/env python3
"""Merge response shards and their timing data into one validated result."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from runners.validate_responses import validate_entries
except ModuleNotFoundError:
    from validate_responses import validate_entries


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def merge_shards(
    source_file: Path,
    shards_dir: Path,
    output_dir: Path,
    expected_shards: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    source_entries = load_json(source_file)
    if not isinstance(source_entries, list):
        return ["Source dataset must contain a JSON array."]

    expected_ids = [entry.get("id") for entry in source_entries if isinstance(entry, dict)]
    if len(expected_ids) != len(source_entries) or any(
        not isinstance(entry_id, str) or not entry_id for entry_id in expected_ids
    ):
        return ["Every source entry must have a valid id."]

    response_files = sorted(shards_dir.glob("*/responses.json"))
    if len(response_files) != expected_shards:
        errors.append(
            f"Expected {expected_shards} shard response files, found {len(response_files)}."
        )

    outputs_by_id: dict[str, str] = {}
    shard_summaries: list[dict[str, Any]] = []
    job_timings: list[dict[str, Any]] = []
    pull_timings: list[dict[str, Any]] = []
    combined_events: list[dict[str, Any]] = []

    copied_shards_dir = output_dir / "shards"
    copied_shards_dir.mkdir(exist_ok=True)
    for artifact_dir in sorted(path for path in shards_dir.iterdir() if path.is_dir()):
        shutil.copytree(
            artifact_dir,
            copied_shards_dir / artifact_dir.name,
            dirs_exist_ok=True,
        )

    expected_id_set = set(expected_ids)
    for response_file in response_files:
        artifact_dir = response_file.parent
        try:
            entries = load_json(response_file)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Could not read {response_file}: {exc}")
            continue

        for validation_error in validate_entries(entries):
            errors.append(f"{artifact_dir.name}: {validation_error}")
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            output = entry.get("actual_output")
            if entry_id not in expected_id_set:
                errors.append(f"{artifact_dir.name}: unexpected response id {entry_id}.")
            elif entry_id in outputs_by_id:
                errors.append(f"Duplicate response across shards: {entry_id}.")
            elif isinstance(output, str) and output.strip():
                outputs_by_id[entry_id] = output.strip()

        summary_file = artifact_dir / "summary.json"
        job_timing_file = artifact_dir / "job-timing.json"
        pull_timing_file = artifact_dir / "model-pull-timing.json"
        if summary_file.is_file():
            shard_summaries.append(load_json(summary_file))
        else:
            errors.append(f"{artifact_dir.name}: summary.json is missing.")
        if job_timing_file.is_file():
            job_timings.append(load_json(job_timing_file))
        else:
            errors.append(f"{artifact_dir.name}: job-timing.json is missing.")
        if pull_timing_file.is_file():
            pull_timings.append(load_json(pull_timing_file))

        for jsonl_file in artifact_dir.glob("*.jsonl"):
            for line_number, line in enumerate(
                jsonl_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"{jsonl_file.name}:{line_number}: invalid JSONL: {exc}"
                    )
                    continue
                event["shard_artifact"] = artifact_dir.name
                combined_events.append(event)

    missing_ids = [entry_id for entry_id in expected_ids if entry_id not in outputs_by_id]
    if missing_ids:
        errors.append(f"Missing generated outputs for: {', '.join(missing_ids)}.")

    merged_entries = deepcopy(source_entries)
    for entry in merged_entries:
        if isinstance(entry, dict):
            entry["actual_output"] = outputs_by_id.get(entry.get("id"), "")

    errors.extend(validate_entries(merged_entries, len(source_entries)))
    save_json(output_dir / "responses.json", merged_entries)

    combined_events.sort(key=lambda event: str(event.get("timestamp", "")))
    with (output_dir / "combined.jsonl").open("w", encoding="utf-8") as handle:
        for event in combined_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    pipeline_window_seconds = None
    if job_timings:
        starts = [parse_timestamp(timing["started_at"]) for timing in job_timings]
        completions = [
            parse_timestamp(timing["completed_at"]) for timing in job_timings
        ]
        pipeline_window_seconds = round(
            (max(completions) - min(starts)).total_seconds(), 3
        )

    sequential_generation_seconds = round(
        sum(float(summary.get("total_duration_seconds", 0)) for summary in shard_summaries),
        3,
    )
    parallel_generation_seconds = (
        round(
            (
                max(parse_timestamp(summary["completed_at"]) for summary in shard_summaries)
                - min(parse_timestamp(summary["started_at"]) for summary in shard_summaries)
            ).total_seconds(),
            3,
        )
        if shard_summaries
        else None
    )
    summary = {
        "model_names": sorted(
            {str(item.get("model_name")) for item in shard_summaries}
        ),
        "models": sorted({str(item.get("model")) for item in shard_summaries}),
        "question_count": len(source_entries),
        "filled_count": len(outputs_by_id),
        "expected_shards": expected_shards,
        "received_shards": len(response_files),
        "max_tokens": sorted(
            {
                item.get("max_tokens")
                for item in shard_summaries
                if item.get("max_tokens") is not None
            }
        ),
        "temperatures": sorted(
            {
                item.get("temperature")
                for item in shard_summaries
                if item.get("temperature") is not None
            }
        ),
        "parallel_pipeline_window_seconds": pipeline_window_seconds,
        "sum_shard_pipeline_seconds": round(
            sum(float(timing.get("duration_seconds", 0)) for timing in job_timings),
            3,
        ),
        "max_shard_pipeline_seconds": round(
            max(
                (float(timing.get("duration_seconds", 0)) for timing in job_timings),
                default=0,
            ),
            3,
        ),
        "parallel_generation_window_seconds": parallel_generation_seconds,
        "sum_shard_generation_seconds": sequential_generation_seconds,
        "max_shard_generation_seconds": round(
            max(
                (
                    float(summary.get("total_duration_seconds", 0))
                    for summary in shard_summaries
                ),
                default=0,
            ),
            3,
        ),
        "sum_model_pull_seconds": round(
            sum(float(timing.get("duration_seconds", 0)) for timing in pull_timings),
            3,
        ),
        "max_model_pull_seconds": round(
            max(
                (float(timing.get("duration_seconds", 0)) for timing in pull_timings),
                default=0,
            ),
            3,
        ),
        "generation_speedup_vs_sequential": round(
            sequential_generation_seconds / parallel_generation_seconds, 3
        )
        if parallel_generation_seconds and parallel_generation_seconds > 0
        else None,
        "errors": errors,
    }
    save_json(output_dir / "summary.json", summary)

    report_lines = [
        "# Parallel response run",
        "",
        f"- Questions filled: {summary['filled_count']}/{summary['question_count']}",
        f"- Shards received: {summary['received_shards']}/{summary['expected_shards']}",
        f"- Parallel pipeline window: {summary['parallel_pipeline_window_seconds']} seconds",
        f"- Parallel generation window: {summary['parallel_generation_window_seconds']} seconds",
        f"- Sum of shard generation time: {summary['sum_shard_generation_seconds']} seconds",
        f"- Generation speedup vs sequential: {summary['generation_speedup_vs_sequential']}x",
        f"- Maximum model pull time: {summary['max_model_pull_seconds']} seconds",
        "",
        "The pipeline window covers shard setup, model download, generation, and validation.",
        "The generation window covers only response generation across the five shards.",
    ]
    if errors:
        report_lines.extend(["", "## Errors", ""])
        report_lines.extend(f"- {error}" for error in errors)
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dataset/responses.json"))
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=5)
    args = parser.parse_args()

    try:
        errors = merge_shards(
            args.source, args.shards_dir, args.output_dir, args.expected_shards
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Shard merge failed: {exc}")
        return 1

    if errors:
        print("Shard merge failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Merged and validated responses in {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
