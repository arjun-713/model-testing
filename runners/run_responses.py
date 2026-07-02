#!/usr/bin/env python3
"""Fill response outputs with a model served by Ollama or AirLLM."""

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

from runners.airllm_backend import AirLLMSession

try:
    from runners.validate_responses import validate_entries
except ModuleNotFoundError:
    from validate_responses import validate_entries


CONCISE_SYSTEM_PROMPT = """
You are JenkinsBot, an expert AI assistant specialized in Jenkins and its ecosystem.

Answer the user's Jenkins question using only facts supported by the supplied retrieval context.

Response requirements:
- Use only the provided Jenkins retrieval context. Do not use external knowledge.
- If the context does not contain enough information, say that the provided context does not mention it.
- Give the direct answer or likely cause in the first sentence.
- Normally use 1 to 3 complete sentences and no more than 80 words.
- Include only the most relevant supported fix, configuration, or troubleshooting step.
- Use a short code or command snippet only when it is necessary to make the answer actionable.
- Paraphrase the evidence. Do not copy long passages, repeat the question, mention the retrieval context, add an introduction, or restate the same point.
- Prioritize a complete core answer over extra detail. Do not begin optional detail that may be cut off.
- Do not guess or add facts, assumptions, commands, or configuration values that are not explicitly supported by the context.

If the answer is not found in the provided context or prior conversation, respond with:
"I'm not able to answer based on the available information."

Return only the final answer.
""".strip()

ORIGINAL_SYSTEM_PROMPT = """
You are JenkinsBot, an expert AI assistant specialized in Jenkins and its ecosystem.

You help users with Jenkins-related topics such as CI/CD pipelines, plugin usage, configuration, administration, and troubleshooting.

You are provided with:
- Relevant retrieved context from Jenkins documentation, plugin metadata, or community sources.
- The prior conversation history, which may contain useful clarification or follow-up details.

Your job is to generate a clear, accurate, and helpful answer to the user's current query by:
- Carefully reading the retrieved context and identifying the parts that directly address the question.
- Synthesizing and rephrasing the relevant information in your own words.
- Providing a concise explanation that is easy to understand, rather than copy-pasting large sections of context verbatim.

You should not:
- Invent or assume facts that are not supported by the retrieved context or conversation history.
- Quote large blocks of text directly from the context unless absolutely necessary.
- Answer questions when no relevant information is available.

If the answer is not found in the provided context or prior conversation, respond with:
"I'm not able to answer based on the available information."

Be accurate, helpful, and concise.
""".strip()

PROMPT_PROFILES = {
    "concise": ("jenkins-concise-v1", CONCISE_SYSTEM_PROMPT),
    "original": ("jenkins-original-v1", ORIGINAL_SYSTEM_PROMPT),
}


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
    system_prompt: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system_prompt},
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


def build_backend_session(
    *,
    backend: str,
    model: str,
    system_prompt: str,
    request_timeout: float,
    airllm_compression: str | None,
    airllm_layer_shards_path: Path | None,
    airllm_profiling_mode: bool,
    airllm_device: str | None,
) -> AirLLMSession | None:
    if backend != "airllm":
        return None
    return AirLLMSession(
        model_id=model,
        system_prompt=system_prompt,
        request_timeout=request_timeout,
        compression=airllm_compression,
        layer_shards_saving_path=airllm_layer_shards_path,
        profiling_mode=airllm_profiling_mode,
        hf_token=os.environ.get("HF_TOKEN"),
        device=airllm_device,
    )


def generate_response(
    *,
    backend: str,
    backend_session: AirLLMSession | None,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_ctx: int,
    temperature: float,
    timeout: float,
    system_prompt: str,
) -> dict[str, Any]:
    if backend == "airllm":
        if backend_session is None:
            raise RuntimeError("AirLLM backend session was not initialized.")
        return backend_session.generate(
            user_prompt=prompt,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            temperature=temperature,
        )
    return post_chat(
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        temperature=temperature,
        timeout=timeout,
        system_prompt=system_prompt,
    )


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
    prompt_profile = getattr(args, "prompt_profile", "concise")
    prompt_version, system_prompt = PROMPT_PROFILES[prompt_profile]
    backend = getattr(args, "backend", "ollama")

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
            "--limit must be positive and offset + limit must not exceed %d.",
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
    prompt_eval_durations: list[int] = []
    backend_session: AirLLMSession | None = None
    backend_load_seconds: float | None = None
    if backend == "airllm":
        try:
            backend_session = build_backend_session(
                backend=backend,
                model=args.model,
                system_prompt=system_prompt,
                request_timeout=args.request_timeout,
                airllm_compression=args.airllm_compression,
                airllm_layer_shards_path=args.airllm_layer_shards_path,
                airllm_profiling_mode=args.airllm_profiling_mode,
                airllm_device=args.airllm_device,
            )
            backend_load_seconds = backend_session.load_duration_seconds
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
    append_jsonl(
        jsonl_file,
        {
            "event": "run_started",
            "timestamp": run_started_at,
            "model_name": args.model_name,
            "model": args.model,
            "prompt_version": prompt_version,
            "prompt_profile": prompt_profile,
            "input_file": str(args.input),
            "output_file": str(output_file),
            "question_count": len(entries),
            "offset": args.offset,
            "max_tokens": args.max_tokens,
            "num_ctx": args.num_ctx,
            "temperature": args.temperature,
            "ollama_base_url": args.base_url,
            "backend": backend,
        },
    )
    logger.info(
        "Starting backend=%s model=%s offset=%d questions=%d max_tokens=%d temperature=%.2f",
        backend,
        args.model_name,
        args.offset,
        len(entries),
        args.max_tokens,
        args.temperature,
    )

    warmup_metrics: dict[str, Any] = {}
    warm_prompt_cache = getattr(args, "warm_prompt_cache", False)
    if warm_prompt_cache:
        warmup_started = time.perf_counter()
        try:
            warmup = generate_response(
                backend=backend,
                backend_session=backend_session,
                base_url=args.base_url,
                model=args.model,
                prompt="Question:\n\nRetrieval context:\n",
                max_tokens=1,
                num_ctx=args.num_ctx,
                temperature=0.0,
                timeout=args.request_timeout,
                system_prompt=system_prompt,
            )
            warmup_metrics = {
                key: warmup.get(key)
                for key in (
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
            }
            append_jsonl(
                jsonl_file,
                {
                    "event": "prompt_cache_warmed",
                    "timestamp": utc_now(),
                    "duration_seconds": round(
                        time.perf_counter() - warmup_started, 3
                    ),
                    "ollama_metrics": warmup_metrics,
                },
            )
        except (RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Prompt cache warmup failed; continuing: %s", exc)

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
                response = generate_response(
                    backend=backend,
                    backend_session=backend_session,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                    system_prompt=system_prompt,
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
                "backend": backend,
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
        prompt_eval_duration = response.get("prompt_eval_duration")
        if isinstance(prompt_eval_duration, int):
            prompt_eval_durations.append(prompt_eval_duration)

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
                "backend": backend,
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
        "backend": backend,
        "prompt_version": prompt_version,
        "prompt_profile": prompt_profile,
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
        "backend_load_seconds": backend_load_seconds,
        "prompt_cache": {
            "enabled": warm_prompt_cache,
            "warmup_metrics": warmup_metrics,
            "first_prompt_eval_duration_ns": prompt_eval_durations[0]
            if prompt_eval_durations
            else None,
            "average_later_prompt_eval_duration_ns": round(
                sum(prompt_eval_durations[1:]) / len(prompt_eval_durations[1:])
            )
            if len(prompt_eval_durations) > 1
            else None,
        },
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
    parser.add_argument(
        "--backend",
        choices=("ollama", "airllm"),
        default="ollama",
        help="Inference backend used for response generation.",
    )
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
        "--prompt-profile",
        choices=sorted(PROMPT_PROFILES),
        default="concise",
        help="System prompt profile used for response generation.",
    )
    parser.add_argument(
        "--warm-prompt-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm Ollama's reusable system-prompt prefix before generation.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--airllm-compression", choices=("4bit", "8bit"))
    parser.add_argument("--airllm-layer-shards-path", type=Path)
    parser.add_argument(
        "--airllm-profiling-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--airllm-device")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
