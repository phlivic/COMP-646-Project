"""Local Hugging Face backend for multimodal models such as Qwen3-VL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.contracts import LLMRequest, LLMResponse
from src.llm.base import MultiModalLLMClient


def resolve_dtype(value: str, torch_module: Any) -> str | Any:
    """Resolve dtype strings into torch dtypes."""

    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported HF_DTYPE: {value}")
    return mapping[lowered]


def build_qwen_messages(system_prompt: str, user_prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    """Build Qwen3-VL style chat messages for local inference."""

    text_parts = [part for part in (system_prompt.strip(), user_prompt.strip()) if part]
    combined_prompt = "\n\n".join(text_parts)
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": image_path.resolve().as_uri()})
    content.append({"type": "text", "text": combined_prompt})
    return [{"role": "user", "content": content}]


class HuggingFaceLocalClient(MultiModalLLMClient):
    """Local Transformers backend using AutoModelForImageTextToText."""

    backend_name = "hf_local"

    def __init__(self, config: AppConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The transformers and torch packages are required for the hf_local backend."
            ) from exc

        dtype = resolve_dtype(config.hf_dtype, torch)
        model_kwargs: dict[str, Any] = {
            "device_map": config.hf_device_map,
            "trust_remote_code": config.hf_trust_remote_code,
            "dtype": dtype if dtype != "auto" else "auto",
        }
        if config.hf_attn_implementation:
            model_kwargs["attn_implementation"] = config.hf_attn_implementation

        self._torch = torch
        self._config = config
        self._model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=config.hf_trust_remote_code,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one multimodal request through a local Transformers model."""

        messages = build_qwen_messages(request.system_prompt, request.user_prompt, request.image_paths)
        started_at = time.perf_counter()
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self._model.device)

        generation_kwargs: dict[str, Any] = {"max_new_tokens": request.max_tokens}
        if request.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = request.temperature
        else:
            generation_kwargs["do_sample"] = False

        with self._torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        raw_text = output_text[0] if output_text else ""
        generated_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0

        return LLMResponse(
            answer_text=raw_text,
            raw_text=raw_text,
            raw_payload={"output_text": output_text},
            model_name=self._config.model_name,
            latency_ms=latency_ms,
            usage={"completion_tokens": generated_tokens},
            finish_reason="stop",
        )

    def close(self) -> None:
        del self._model
        del self._processor
