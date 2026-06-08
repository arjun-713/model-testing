from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from evals.metrics import build_metrics
from evals.run_evaluation import build_test_cases, summarize_result, write_report


class DeepEvalPipelineTests(unittest.TestCase):
    def test_requirements_install_ollama_client_for_deepeval(self) -> None:
        requirements = Path("requirements-eval.txt").read_text(encoding="utf-8")

        self.assertIn("deepeval==4.0.5", requirements)
        self.assertIn("ollama==0.6.2", requirements)

    def test_build_metrics_uses_qwen_ollama_judge(self) -> None:
        if find_spec("ollama") is None:
            self.skipTest("Ollama Python client is installed by requirements-eval.txt")

        metrics = build_metrics(
            "qwen3:4b-instruct",
            "http://127.0.0.1:11434",
            0.5,
        )

        self.assertEqual(
            [type(metric).__name__ for metric in metrics],
            [
                "FaithfulnessMetric",
                "AnswerRelevancyMetric",
                "ContextualRecallMetric",
            ],
        )
        self.assertTrue(all(metric.threshold == 0.5 for metric in metrics))
        self.assertTrue(
            all(
                metric.evaluation_model == "qwen3:4b-instruct (Ollama)"
                for metric in metrics
            )
        )

    def test_build_test_cases_joins_responses_and_goldens_by_id(self) -> None:
        responses = [
            {
                "id": "q-001",
                "input": "Question?",
                "actual_output": "Generated answer.",
                "retrieval_context": ["Retrieved evidence."],
            }
        ]
        goldens = [
            {
                "input": "Question?",
                "expected_output": "Expected answer.",
                "additional_metadata": {
                    "id": "q-001",
                    "category": "jenkins_core",
                },
            }
        ]

        test_cases = build_test_cases(responses, goldens, "qwen", 1)

        self.assertEqual(len(test_cases), 1)
        self.assertEqual(test_cases[0].expected_output, "Expected answer.")
        self.assertEqual(test_cases[0].retrieval_context, ["Retrieved evidence."])
        self.assertEqual(test_cases[0].metadata["response_model"], "qwen")

    def test_summary_exports_scores_reasons_and_generator_average(self) -> None:
        metrics = [
            {
                "name": "Faithfulness",
                "threshold": 0.5,
                "success": True,
                "score": 0.8,
                "reason": "Grounded.",
                "error": None,
                "evaluation_model": "qwen3:4b-instruct",
            },
            {
                "name": "Answer Relevancy",
                "threshold": 0.5,
                "success": True,
                "score": 0.6,
                "reason": "Relevant.",
                "error": None,
                "evaluation_model": "qwen3:4b-instruct",
            },
            {
                "name": "Contextual Recall",
                "threshold": 0.5,
                "success": False,
                "score": 0.4,
                "reason": "Some expected facts are missing.",
                "error": None,
                "evaluation_model": "qwen3:4b-instruct",
            },
        ]
        raw_result = {
            "test_results": [
                {
                    "name": "q-001",
                    "input": "Question?",
                    "actual_output": "Generated answer.",
                    "expected_output": "Expected answer.",
                    "metadata": {"id": "q-001"},
                    "metrics_data": metrics,
                }
            ],
            "confident_link": "https://app.confident-ai.com/test-runs/example",
            "test_run_id": "run-id",
        }

        summary, errors = summarize_result(
            raw_result=raw_result,
            response_model="qwen",
            judge_model="qwen3:4b-instruct",
            threshold=0.5,
            expected_count=1,
            started_at="2026-01-01T00:00:00+00:00",
            duration_seconds=12.5,
            confident_enabled=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(summary["generator_average_score"], 0.7)
        self.assertEqual(
            summary["metrics"]["Contextual Recall"]["average_score"], 0.4
        )
        self.assertEqual(
            summary["cases"][0]["metrics"]["Faithfulness"]["reason"], "Grounded."
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_file = Path(temporary_directory) / "report.md"
            write_report(summary, report_file)
            report = report_file.read_text(encoding="utf-8")
            self.assertIn("Generator average: 0.7", report)
            self.assertIn("Contextual Recall", report)

    def test_summary_fails_when_a_metric_is_missing(self) -> None:
        raw_result = {
            "test_results": [
                {
                    "name": "q-001",
                    "metadata": {"id": "q-001"},
                    "metrics_data": [],
                }
            ],
            "confident_link": None,
            "test_run_id": None,
        }

        _, errors = summarize_result(
            raw_result=raw_result,
            response_model="phi",
            judge_model="qwen3:4b-instruct",
            threshold=0.5,
            expected_count=1,
            started_at="2026-01-01T00:00:00+00:00",
            duration_seconds=1,
            confident_enabled=False,
        )

        self.assertEqual(len(errors), 3)
