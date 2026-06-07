"""DeepEval metrics used to judge generated Jenkins answers."""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel


METRIC_NAMES = ("Faithfulness", "Answer Relevancy", "Contextual Recall")


def build_metrics(
    judge_model_name: str,
    base_url: str,
    threshold: float,
):
    judge_model = OllamaModel(
        model=judge_model_name,
        base_url=base_url,
        temperature=0.0,
        generation_kwargs={
            "num_ctx": 16384,
            "num_predict": 1024,
            "seed": 42,
        },
    )
    common_options = {
        "model": judge_model,
        "threshold": threshold,
        "include_reason": True,
        "async_mode": False,
        "verbose_mode": True,
    }
    return [
        FaithfulnessMetric(**common_options),
        AnswerRelevancyMetric(**common_options),
        ContextualRecallMetric(**common_options),
    ]
