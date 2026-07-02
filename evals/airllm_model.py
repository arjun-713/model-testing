from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel

from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json

from runners.airllm_backend import AirLLMSession


class AirLLMDeepEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        generation_kwargs: Optional[dict] = None,
        layer_shards_saving_path: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.temperature = temperature
        self.generation_kwargs = dict(generation_kwargs or {})
        self.layer_shards_saving_path = layer_shards_saving_path
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        super().__init__(model=model)

    def load_model(self) -> AirLLMSession:
        max_new_tokens = int(self.generation_kwargs.get("num_predict", 1024))
        num_ctx = int(self.generation_kwargs.get("num_ctx", 16384))
        compression = self.generation_kwargs.get("compression")
        profiling_mode = bool(self.generation_kwargs.get("profiling_mode", False))
        return AirLLMSession(
            model_id=self.name,
            system_prompt=(
                "You are an evaluation model. Follow the output format strictly."
            ),
            request_timeout=3600.0,
            compression=compression,
            layer_shards_saving_path=(
                Path(layer_shards_saving_path)
                if (layer_shards_saving_path := self.layer_shards_saving_path)
                else None
            ),
            profiling_mode=profiling_mode,
            hf_token=self.hf_token,
            device=os.environ.get("AIRLLM_DEVICE"),
        )

    def _generate_text(self, prompt: str) -> str:
        max_new_tokens = int(self.generation_kwargs.get("num_predict", 1024))
        num_ctx = int(self.generation_kwargs.get("num_ctx", 16384))
        response = self.model.generate(
            user_prompt=prompt,
            max_tokens=max_new_tokens,
            num_ctx=num_ctx,
            temperature=self.temperature,
        )
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise DeepEvalError("AirLLM judge returned a non-string response.")
        return content

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> tuple[Union[str, BaseModel], float]:
        content = self._generate_text(prompt)
        if schema:
            parsed = trim_and_load_json(content)
            return schema.model_validate(parsed), 0.0
        return content, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> tuple[Union[str, BaseModel], float]:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self, *args, **kwargs) -> str:
        return f"{self.name} (AirLLM)"
