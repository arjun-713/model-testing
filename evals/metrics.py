"""DeepEval metrics used to judge generated Jenkins answers."""

from __future__ import annotations

import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel

from evals.airllm_model import AirLLMDeepEvalModel
from evals.constants import METRIC_NAMES

def build_metrics(
    judge_model_name: str,
    base_url: str,
    threshold: float,
    include_reason: bool = True,
    async_mode: bool = False,
    metric_names: tuple[str, ...] = METRIC_NAMES,
    judge_backend: str = "ollama",
):
    judge_num_ctx = int(os.environ.get("DEEPEVAL_JUDGE_NUM_CTX", "16384"))
    judge_num_predict = int(os.environ.get("DEEPEVAL_JUDGE_NUM_PREDICT", "1024"))
    generation_kwargs = {
        "num_ctx": judge_num_ctx,
        "num_predict": judge_num_predict,
        "seed": 42,
    }
    if judge_backend == "airllm":
        airllm_compression = os.environ.get("AIRLLM_JUDGE_COMPRESSION")
        if airllm_compression:
            generation_kwargs["compression"] = airllm_compression
        judge_model = AirLLMDeepEvalModel(
            model=judge_model_name,
            temperature=0.0,
            generation_kwargs=generation_kwargs,
            layer_shards_saving_path=os.environ.get("AIRLLM_JUDGE_LAYER_SHARDS_PATH"),
            hf_token=os.environ.get("HF_TOKEN"),
        )
    else:
        judge_model = OllamaModel(
            model=judge_model_name,
            base_url=base_url,
            temperature=0.0,
            generation_kwargs=generation_kwargs,
        )
    common_options = {
        "model": judge_model,
        "threshold": threshold,
        "include_reason": include_reason,
        "async_mode": async_mode,
        "verbose_mode": include_reason,
    }
    metric_types = {
        "Faithfulness": FaithfulnessMetric,
        "Answer Relevancy": AnswerRelevancyMetric,
        "Contextual Recall": ContextualRecallMetric,
    }
    unknown = set(metric_names) - set(metric_types)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    return [metric_types[name](**common_options) for name in metric_names]
