from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                limit=2,
                max_tokens=512,
                num_ctx=4096,
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
                limit=1,
                max_tokens=512,
                num_ctx=4096,
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


if __name__ == "__main__":
    unittest.main()
