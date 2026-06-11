from __future__ import annotations

import unittest

from evals.retry_missing import missing_pairs, recompute_summary


class RetryMissingTests(unittest.TestCase):
    def test_only_missing_metric_pairs_are_selected(self) -> None:
        summary = {
            "cases": [
                {
                    "id": "q-001",
                    "metrics": {
                        "Faithfulness": {"score": 0.8},
                        "Answer Relevancy": {"score": 0.9},
                        "Contextual Recall": {"score": None},
                    },
                }
            ]
        }

        self.assertEqual(missing_pairs(summary), [("q-001", "Contextual Recall")])

    def test_recompute_removes_recovered_errors(self) -> None:
        summary = {
            "threshold": 0.5,
            "metrics": {},
            "errors": ["old failure"],
            "cases": [
                {
                    "id": "q-001",
                    "metrics": {
                        name: {"score": 0.8, "success": True, "error": None}
                        for name in (
                            "Faithfulness",
                            "Answer Relevancy",
                            "Contextual Recall",
                        )
                    },
                }
            ],
        }

        recompute_summary(summary)

        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["metrics"]["Contextual Recall"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
