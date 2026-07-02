from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_airllm():
    try:
        from airllm import AutoModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AirLLM backend requires `airllm` to be installed."
        ) from exc
    return AutoModel


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AirLLM backend requires `torch` to be installed."
        ) from exc
    return torch


def _build_chat_prompt(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return (
        f"System instruction:\n{system_prompt}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Answer:\n"
    )


@dataclass
class AirLLMSession:
    model_id: str
    system_prompt: str
    request_timeout: float
    compression: str | None = None
    layer_shards_saving_path: Path | None = None
    profiling_mode: bool = False
    hf_token: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        AutoModel = _import_airllm()
        torch = _import_torch()

        kwargs: dict[str, Any] = {
            "profiling_mode": self.profiling_mode,
        }
        if self.compression:
            kwargs["compression"] = self.compression
        if self.layer_shards_saving_path is not None:
            kwargs["layer_shards_saving_path"] = str(self.layer_shards_saving_path)
        if self.hf_token:
            kwargs["hf_token"] = self.hf_token

        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(self.model_id, **kwargs)
        self.load_duration_seconds = round(time.perf_counter() - started, 3)
        self.tokenizer = self.model.tokenizer
        chosen_device = self.device
        if chosen_device is None:
            chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = chosen_device
        self._torch = torch

    def generate(
        self,
        *,
        user_prompt: str,
        max_tokens: int,
        num_ctx: int,
        temperature: float,
    ) -> dict[str, Any]:
        prompt_text = _build_chat_prompt(
            self.tokenizer,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        max_input_tokens = num_ctx if num_ctx > 0 else None
        encode_kwargs = {
            "return_tensors": "pt",
            "return_attention_mask": True,
            "truncation": True,
            "padding": False,
        }
        if max_input_tokens is not None:
            encode_kwargs["max_length"] = max_input_tokens
        encoded = self.tokenizer(prompt_text, **encode_kwargs)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        started = time.perf_counter()
        with self._torch.no_grad():
            generated = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        elapsed = time.perf_counter() - started

        sequences = generated.sequences[0]
        prompt_tokens = int(input_ids.shape[-1])
        generated_tokens = int(sequences.shape[-1] - prompt_tokens)
        completion_tokens = sequences[prompt_tokens:]
        output = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=True,
        ).strip()

        return {
            "message": {"content": output},
            "done": True,
            "done_reason": "stop",
            "total_duration": int(elapsed * 1_000_000_000),
            "load_duration": int(self.load_duration_seconds * 1_000_000_000),
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": None,
            "eval_count": generated_tokens,
            "eval_duration": int(elapsed * 1_000_000_000),
            "backend": "airllm",
            "device": self.device,
        }
