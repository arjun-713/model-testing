#!/usr/bin/env python3
"""Fill response outputs with either an Ollama or Hugging Face model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from runners.validate_responses import validate_entries
except ModuleNotFoundError:
    from validate_responses import validate_entries


SYSTEM_PROMPT = """You are JenkinsBot, an expert AI assistant specialized in Jenkins and its ecosystem.

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
"""


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


def post_ollama_chat(
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


def load_huggingface_backend(model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face runtime is not installed. Install transformers, torch, "
            "accelerate, sentencepiece, and pillow before using provider=huggingface."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        token=token,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model, torch


def post_huggingface_chat(
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with (
        torch_module.inference_mode()
        if hasattr(torch_module, "inference_mode")
        else nullcontext()
    ):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
    output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return {
        "message": {"content": output},
        "done": True,
        "prompt_eval_count": int(input_len),
        "eval_count": int(outputs.shape[-1] - input_len),
        "provider": "huggingface",
    }


def generate_response(
    *,
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_ctx: int,
    temperature: float,
    timeout: float,
    base_url: str,
    huggingface_backend: tuple[Any, Any, Any] | None,
) -> dict[str, Any]:
    if provider == "ollama":
        return post_ollama_chat(
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            temperature=temperature,
            timeout=timeout,
        )
    if provider == "huggingface":
        if huggingface_backend is None:
            raise RuntimeError("Hugging Face backend was not initialized.")
        tokenizer, hf_model, torch_module = huggingface_backend
        return post_huggingface_chat(
            tokenizer=tokenizer,
            model=hf_model,
            torch_module=torch_module,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


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
    if args.limit < 1 or args.limit > len(source_entries):
        logger.error("--limit must be between 1 and %d.", len(source_entries))
        return 1

    provider = getattr(args, "provider", "ollama")
    entries = deepcopy(source_entries[: args.limit])
    for entry in entries:
        if isinstance(entry, dict):
            entry["actual_output"] = ""

    run_started_at = utc_now()
    run_start = time.perf_counter()
    failures: list[str] = []
    durations: list[float] = []
    huggingface_backend: tuple[Any, Any, Any] | None = None
    model_load_seconds: float | None = None

    append_jsonl(
        jsonl_file,
        {
            "event": "run_started",
            "timestamp": run_started_at,
            "provider": provider,
            "model_name": args.model_name,
            "model": args.model,
            "input_file": str(args.input),
            "output_file": str(output_file),
            "question_count": len(entries),
            "max_tokens": args.max_tokens,
            "num_ctx": args.num_ctx,
            "temperature": args.temperature,
            "ollama_base_url": args.base_url,
        },
    )
    logger.info(
        "Starting provider=%s model=%s questions=%d max_tokens=%d temperature=%.2f",
        provider,
        args.model,
        len(entries),
        args.max_tokens,
        args.temperature,
    )

    if provider == "huggingface":
        logger.info("Loading Hugging Face model %s", args.model)
        load_started = time.perf_counter()
        try:
            huggingface_backend = load_huggingface_backend(args.model)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        model_load_seconds = time.perf_counter() - load_started
        logger.info(
            "Loaded Hugging Face model %s in %.3f seconds",
            args.model,
            model_load_seconds,
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
                response = generate_response(
                    provider=provider,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                    base_url=args.base_url,
                    huggingface_backend=huggingface_backend,
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
                    "provider": provider,
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
                "provider": provider,
                "duration_seconds": round(elapsed, 3),
                "prompt_characters": len(prompt),
                "actual_output": output,
                "provider_metrics": {
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
                        "provider",
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
        "provider": provider,
        "started_at": run_started_at,
        "completed_at": utc_now(),
        "question_count": len(entries),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "model_load_seconds": round(model_load_seconds, 3)
        if model_load_seconds is not None
        else None,
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
    parser.add_argument("--model", required=True, help="Provider-specific model identifier")
    parser.add_argument("--model-name", required=True, help="Artifact-safe model name")
    parser.add_argument(
        "--provider",
        choices=("ollama", "huggingface"),
        default="ollama",
        help="Model provider backend",
    )
    parser.add_argument(
        "--input", type=Path, default=Path("dataset/responses.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
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
