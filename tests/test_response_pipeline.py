from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runners.merge_shards import merge_shards
from runners.run_responses import run
from runners.validate_responses import validate_entries


class ResponsePipelineTests(unittest.TestCase):
    def test_validator_rejects_blank_outputs(self) -> None:
        errors = validate_entries(
            [{"id": "q-001", "actual_output": ""}], expected_count=1
        )
        self.assertEqual(errors, ["q-001: actual_output is empty."])

    def test_runner_writes_outputs_logs_and_summary(self) -> None:
        source = [
            {
                "id": "q-001",
                "input": "Question one?",
                "actual_output": "old value",
                "retrieval_context": ["Context one."],
            },
            {
                "id": "q-002",
                "input": "Question two?",
                "actual_output": "",
                "retrieval_context": ["Context two."],
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_file = root / "responses.json"
            output_dir = root / "results"
            input_file.write_text(json.dumps(source), encoding="utf-8")
            args = argparse.Namespace(
                model="test-model:latest",
                model_name="test-model",
                input=input_file,
                output_dir=output_dir,
                offset=0,
                limit=2,
                max_tokens=512,
                num_ctx=4096,
                temperature=0.1,
                base_url="http://127.0.0.1:11434",
                request_timeout=5.0,
                retries=0,
            )

            responses = [
                {
                    "message": {"content": "Answer one."},
                    "done": True,
                    "eval_count": 3,
                    "prompt_eval_count": 10,
                },
                {
                    "message": {"content": "Answer two."},
                    "done": True,
                    "eval_count": 3,
                    "prompt_eval_count": 10,
                },
            ]
            with patch("runners.run_responses.post_chat", side_effect=responses):
                result = run(args)

            self.assertEqual(result, 0)
            generated = json.loads(
                (output_dir / "responses.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["actual_output"] for entry in generated],
                ["Answer one.", "Answer two."],
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["filled_count"], 2)
            self.assertEqual(summary["failed_ids"], [])
            self.assertTrue((output_dir / "test-model.log").is_file())
            self.assertTrue((output_dir / "test-model.jsonl").is_file())

    def test_runner_fails_when_model_returns_empty_output(self) -> None:
        source = [
            {
                "id": "q-001",
                "input": "Question?",
                "actual_output": "",
                "retrieval_context": ["Context."],
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_file = root / "responses.json"
            output_dir = root / "results"
            input_file.write_text(json.dumps(source), encoding="utf-8")
            args = argparse.Namespace(
                model="test-model:latest",
                model_name="test-model",
                input=input_file,
                output_dir=output_dir,
                offset=0,
                limit=1,
                max_tokens=512,
                num_ctx=4096,
                temperature=0.1,
                base_url="http://127.0.0.1:11434",
                request_timeout=5.0,
                retries=0,
            )

            response = {
                "message": {"content": "   "},
                "done": True,
                "eval_count": 0,
                "prompt_eval_count": 10,
            }
            with patch("runners.run_responses.post_chat", return_value=response):
                result = run(args)

            self.assertEqual(result, 1)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["filled_count"], 0)
            self.assertEqual(summary["failed_ids"], ["q-001"])

    def test_runner_selects_entries_by_offset(self) -> None:
        source = [
            {
                "id": f"q-{index:03d}",
                "input": f"Question {index}?",
                "actual_output": "",
                "retrieval_context": [f"Context {index}."],
            }
            for index in range(1, 5)
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_file = root / "responses.json"
            output_dir = root / "results"
            input_file.write_text(json.dumps(source), encoding="utf-8")
            args = argparse.Namespace(
                model="test-model:latest",
                model_name="test-model-shard-02",
                input=input_file,
                output_dir=output_dir,
                offset=2,
                limit=2,
                max_tokens=256,
                num_ctx=4096,
                temperature=0.1,
                base_url="http://127.0.0.1:11434",
                request_timeout=5.0,
                retries=0,
            )
            response = {
                "message": {"content": "Answer."},
                "done": True,
                "eval_count": 2,
                "prompt_eval_count": 8,
            }
            with patch("runners.run_responses.post_chat", return_value=response):
                result = run(args)

            self.assertEqual(result, 0)
            generated = json.loads(
                (output_dir / "responses.json").read_text(encoding="utf-8")
            )
            self.assertEqual([entry["id"] for entry in generated], ["q-003", "q-004"])

    def test_merger_reconstructs_source_order_and_timings(self) -> None:
        source = [
            {
                "id": f"q-{index:03d}",
                "input": f"Question {index}?",
                "actual_output": "",
                "retrieval_context": [f"Context {index}."],
            }
            for index in range(1, 5)
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_file = root / "source.json"
            shards_dir = root / "shards"
            output_dir = root / "combined"
            source_file.write_text(json.dumps(source), encoding="utf-8")

            for shard_number, indexes in enumerate(((2, 3), (0, 1)), start=1):
                shard_dir = shards_dir / f"artifact-{shard_number}"
                shard_dir.mkdir(parents=True)
                entries = []
                for index in indexes:
                    entry = dict(source[index])
                    entry["actual_output"] = f"Answer {index + 1}."
                    entries.append(entry)
                (shard_dir / "responses.json").write_text(
                    json.dumps(entries), encoding="utf-8"
                )
                (shard_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "model_name": "test-model",
                            "model": "test-model:latest",
                            "started_at": f"2026-01-01T00:00:0{shard_number}+00:00",
                            "completed_at": f"2026-01-01T00:00:1{shard_number}+00:00",
                            "total_duration_seconds": 10,
                            "max_tokens": 256,
                            "temperature": 0.1,
                        }
                    ),
                    encoding="utf-8",
                )
                (shard_dir / "job-timing.json").write_text(
                    json.dumps(
                        {
                            "started_at": f"2026-01-01T00:00:0{shard_number}Z",
                            "completed_at": f"2026-01-01T00:00:2{shard_number}Z",
                            "duration_seconds": 20,
                        }
                    ),
                    encoding="utf-8",
                )
                (shard_dir / "model-pull-timing.json").write_text(
                    json.dumps({"duration_seconds": 3}), encoding="utf-8"
                )
                (shard_dir / f"test-{shard_number}.jsonl").write_text(
                    json.dumps(
                        {
                            "event": "response_completed",
                            "timestamp": f"2026-01-01T00:00:1{shard_number}Z",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            errors = merge_shards(source_file, shards_dir, output_dir, 2)

            self.assertEqual(errors, [])
            merged = json.loads(
                (output_dir / "responses.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["actual_output"] for entry in merged],
                ["Answer 1.", "Answer 2.", "Answer 3.", "Answer 4."],
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["filled_count"], 4)
            self.assertEqual(summary["received_shards"], 2)
            self.assertEqual(summary["sum_shard_generation_seconds"], 20)


if __name__ == "__main__":
    unittest.main()
