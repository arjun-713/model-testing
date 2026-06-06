#!/usr/bin/env python3
"""Validate that every generated response contains a non-empty output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_entries(entries: Any, expected_count: int | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(entries, list):
        return ["Response file must contain a JSON array."]

    if expected_count is not None and len(entries) != expected_count:
        errors.append(
            f"Expected {expected_count} responses, but found {len(entries)}."
        )

    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"Entry {index} is not a JSON object.")
            continue

        entry_id = entry.get("id")
        label = entry_id if isinstance(entry_id, str) and entry_id else f"index {index}"
        if not isinstance(entry_id, str) or not entry_id.strip():
            errors.append(f"Entry {index} has no valid id.")
        elif entry_id in seen_ids:
            errors.append(f"Duplicate response id: {entry_id}.")
        else:
            seen_ids.add(entry_id)

        output = entry.get("actual_output")
        if not isinstance(output, str) or not output.strip():
            errors.append(f"{label}: actual_output is empty.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("response_file", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    if not args.response_file.is_file():
        print(f"Validation failed: {args.response_file} does not exist.")
        return 1

    try:
        entries = json.loads(args.response_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Validation failed: could not read response file: {exc}")
        return 1

    errors = validate_entries(entries, args.expected_count)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validation passed: all {len(entries)} actual_output fields are filled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
