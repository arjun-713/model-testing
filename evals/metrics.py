"""DeepEval metrics used to judge generated Jenkins answers."""

from __future__ import annotations

import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel

from evals.constants import METRIC_NAMES

def build_metrics(
    judge_model_name: str,
    base_url: str,
    threshold: float,
):
    judge_num_ctx = int(os.environ.get("DEEPEVAL_JUDGE_NUM_CTX", "16384"))
    judge_num_predict = int(os.environ.get("DEEPEVAL_JUDGE_NUM_PREDICT", "1024"))
    judge_model = OllamaModel(
        model=judge_model_name,
        base_url=base_url,
        temperature=0.0,
        generation_kwargs={
            "num_ctx": judge_num_ctx,
            "num_predict": judge_num_predict,
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
