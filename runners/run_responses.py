#!/usr/bin/env python3
"""Fill response outputs with a model served by Ollama."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from runners.validate_responses import validate_entries
except ModuleNotFoundError:
    from validate_responses import validate_entries


SYSTEM_PROMPT = """You answer Jenkins questions using only the supplied retrieval context.
Choose the evidence that most directly answers the question, even when the context contains
irrelevant or conflicting search results. Give a concise, technically actionable answer.
Do not mention the retrieval context. Return only the final answer."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prompt(entry: dict[str, Any]) -> str:
    contexts = entry.get("retrieval_context", [])
    if not isinstance(contexts, list) or not all(
        isinstance(context, str) for context in contexts
    ):
        raise ValueError(f"{entry.get('id', '<unknown>')}: invalid retrieval_context")

    return (
        f"Question:\n{entry['input']}\n\n"
        f"Retrieval context:\n{'\n\n'.join(contexts)}"
    )


def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("response_runner")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def post_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_ctx: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": 42,
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama: {exc.reason}") from exc


def save_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "responses.json"
    summary_file = args.output_dir / "summary.json"
    log_file = args.output_dir / f"{args.model_name}.log"
    jsonl_file = args.output_dir / f"{args.model_name}.jsonl"
    jsonl_file.unlink(missing_ok=True)
    logger = configure_logging(log_file)

    try:
        source_entries = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not load %s: %s", args.input, exc)
        return 1

    if not isinstance(source_entries, list) or not source_entries:
        logger.error("The input dataset must be a non-empty JSON array.")
        return 1
    if args.offset < 0 or args.offset >= len(source_entries):
        logger.error("--offset must be between 0 and %d.", len(source_entries) - 1)
        return 1
    if args.limit < 1 or args.offset + args.limit > len(source_entries):
        logger.error(
            "--limit must select entries within the %d-entry dataset.",
            len(source_entries),
        )
        return 1

    entries = deepcopy(source_entries[args.offset : args.offset + args.limit])
    for entry in entries:
        if isinstance(entry, dict):
            entry["actual_output"] = ""

    run_started_at = utc_now()
    run_start = time.perf_counter()
    failures: list[str] = []
    durations: list[float] = []
    append_jsonl(
        jsonl_file,
        {
            "event": "run_started",
            "timestamp": run_started_at,
            "model_name": args.model_name,
            "model": args.model,
            "input_file": str(args.input),
            "output_file": str(output_file),
            "question_count": len(entries),
            "offset": args.offset,
            "max_tokens": args.max_tokens,
            "num_ctx": args.num_ctx,
            "temperature": args.temperature,
            "ollama_base_url": args.base_url,
        },
    )
    logger.info(
        "Starting model=%s ollama_model=%s offset=%d questions=%d max_tokens=%d temperature=%.2f",
        args.model_name,
        args.model,
        args.offset,
        len(entries),
        args.max_tokens,
        args.temperature,
    )

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("input"), str):
            entry_id = f"index-{index - 1}"
            failures.append(entry_id)
            logger.error("%s has an invalid input entry.", entry_id)
            continue

        entry_id = str(entry.get("id", f"index-{index - 1}"))
        try:
            prompt = build_prompt(entry)
        except (KeyError, ValueError) as exc:
            failures.append(entry_id)
            logger.error("%s", exc)
            continue

        logger.info(
            "[%d/%d] Generating %s prompt_chars=%d",
            index,
            len(entries),
            entry_id,
            len(prompt),
        )
        started_at = utc_now()
        started = time.perf_counter()
        response: dict[str, Any] | None = None
        error: str | None = None

        for attempt in range(1, args.retries + 2):
            try:
                response = post_chat(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                )
                error = None
                break
            except (RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                error = str(exc)
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    entry_id,
                    attempt,
                    args.retries + 1,
                    error,
                )
                if attempt <= args.retries:
                    time.sleep(min(2**attempt, 10))

        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        if response is None:
            failures.append(entry_id)
            append_jsonl(
                jsonl_file,
                {
                    "event": "response_failed",
                    "timestamp": utc_now(),
                    "started_at": started_at,
                    "id": entry_id,
                    "model_name": args.model_name,
                    "model": args.model,
                    "duration_seconds": round(elapsed, 3),
                    "error": error,
                },
            )
            logger.error("%s failed after %.3f seconds.", entry_id, elapsed)
            save_json(output_file, entries)
            continue

        message = response.get("message")
        output = message.get("content", "") if isinstance(message, dict) else ""
        output = output.strip() if isinstance(output, str) else ""
        entry["actual_output"] = output
        if not output:
            failures.append(entry_id)

        append_jsonl(
            jsonl_file,
            {
                "event": "response_completed",
                "timestamp": utc_now(),
                "started_at": started_at,
                "id": entry_id,
                "question": entry["input"],
                "model_name": args.model_name,
                "model": args.model,
                "duration_seconds": round(elapsed, 3),
                "prompt_characters": len(prompt),
                "actual_output": output,
                "ollama_metrics": {
                    key: response.get(key)
                    for key in (
                        "done",
                        "done_reason",
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    )
                },
            },
        )
        logger.info(
            "%s completed in %.3f seconds prompt_tokens=%s output_tokens=%s",
            entry_id,
            elapsed,
            response.get("prompt_eval_count"),
            response.get("eval_count"),
        )
        logger.info("%s actual_output:\n%s", entry_id, output)
        save_json(output_file, entries)

    total_duration = time.perf_counter() - run_start
    validation_errors = validate_entries(entries, len(entries))
    summary = {
        "model_name": args.model_name,
        "model": args.model,
        "started_at": run_started_at,
        "completed_at": utc_now(),
        "question_count": len(entries),
        "offset": args.offset,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "filled_count": sum(
            isinstance(entry, dict)
            and isinstance(entry.get("actual_output"), str)
            and bool(entry["actual_output"].strip())
            for entry in entries
        ),
        "failed_ids": sorted(set(failures)),
        "total_duration_seconds": round(total_duration, 3),
        "average_response_seconds": round(sum(durations) / len(durations), 3)
        if durations
        else None,
        "validation_errors": validation_errors,
    }
    save_json(output_file, entries)
    save_json(summary_file, summary)
    append_jsonl(jsonl_file, {"event": "run_completed", **summary})

    if failures or validation_errors:
        logger.error(
            "Run failed: filled=%d/%d failed_ids=%s",
            summary["filled_count"],
            len(entries),
            summary["failed_ids"],
        )
        return 1

    logger.info(
        "Run passed: filled=%d/%d total_duration=%.3f seconds",
        summary["filled_count"],
        len(entries),
        total_duration,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model tag")
    parser.add_argument("--model-name", required=True, help="Artifact-safe model name")
    parser.add_argument(
        "--input", type=Path, default=Path("dataset/responses.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
