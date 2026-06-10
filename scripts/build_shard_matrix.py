#!/usr/bin/env python3
"""Build a GitHub Actions matrix for a requested dataset prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_matrix(question_count: int, shard_size: int) -> list[dict[str, int]]:
    if question_count < 1:
        raise ValueError("question_count must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")

    return [
        {
            "shard": index + 1,
            "offset": offset,
            "limit": min(shard_size, question_count - offset),
        }
        for index, offset in enumerate(range(0, question_count, shard_size))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--question-count", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=10)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("dataset must contain a non-empty JSON array")
    if args.question_count > len(dataset):
        raise ValueError(
            f"question_count {args.question_count} exceeds dataset size {len(dataset)}"
        )

    matrix = build_matrix(args.question_count, args.shard_size)
    compact_matrix = json.dumps(matrix, separators=(",", ":"))
    output = {
        "question_count": args.question_count,
        "shard_count": len(matrix),
        "matrix": matrix,
    }
    print(json.dumps(output, indent=2))

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"question_count={args.question_count}\n")
            handle.write(f"shard_count={len(matrix)}\n")
            handle.write(f"matrix={compact_matrix}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
