from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.aggregate_results import aggregate
from scripts.build_shard_matrix import build_matrix


class ShardingTests(unittest.TestCase):
    def test_build_matrix_handles_full_and_partial_shards(self) -> None:
        self.assertEqual(
            build_matrix(25, 10),
            [
                {"shard": 1, "offset": 0, "limit": 10},
                {"shard": 2, "offset": 10, "limit": 10},
                {"shard": 3, "offset": 20, "limit": 5},
            ],
        )

    def test_aggregate_weights_scores_by_question(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = [
                {"id": "q-001"},
                {"id": "q-002"},
                {"id": "q-003"},
            ]
            dataset_file = root / "responses.json"
            dataset_file.write_text(json.dumps(dataset), encoding="utf-8")

            for shard, ids in ((1, ["q-001", "q-002"]), (2, ["q-003"])):
                shard_dir = root / "artifacts" / f"shard-{shard}"
                shard_dir.mkdir(parents=True)
                responses = [{"id": case_id, "actual_output": "answer"} for case_id in ids]
                (shard_dir / "responses.json").write_text(
                    json.dumps(responses), encoding="utf-8"
                )
                (shard_dir / "summary.json").write_text(
                    json.dumps({"total_duration_seconds": shard}), encoding="utf-8"
                )
                cases = []
                for case_id in ids:
                    score = 1.0 if case_id != "q-003" else 0.0
                    cases.append(
                        {
                            "id": case_id,
                            "metrics": {
                                name: {"score": score, "success": score >= 0.5}
                                for name in (
                                    "Faithfulness",
                                    "Answer Relevancy",
                                    "Contextual Recall",
                                )
                            },
                        }
                    )
                (shard_dir / "evaluation-summary.json").write_text(
                    json.dumps(
                        {
                            "response_model": "qwen3:4b-instruct",
                            "judge_model": "gemma3:4b-it-qat",
                            "duration_seconds": shard * 2,
                            "cases": cases,
                            "errors": [],
                        }
                    ),
                    encoding="utf-8",
                )

            summary, responses = aggregate(
                root / "artifacts", dataset_file, question_count=3, expected_shards=2
            )

            self.assertEqual(
                [entry["id"] for entry in responses],
                [entry["id"] for entry in dataset],
            )
            self.assertEqual(summary["metrics"]["Faithfulness"]["average_score"], 0.6667)
            self.assertEqual(summary["metrics"]["Faithfulness"]["evaluated_count"], 3)


if __name__ == "__main__":
    unittest.main()
