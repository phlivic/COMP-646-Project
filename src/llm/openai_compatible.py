"""OpenAI-compatible multimodal backend."""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.contracts import LLMRequest, LLMResponse
from src.llm.base import MultiModalLLMClient


def image_path_to_data_url(image_path: Path) -> str:
    """Encode a local image as a base64 data URL."""

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def flatten_message_content(content: Any) -> str:
    """Normalize chat-completions message content into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") in {"text", "output_text"} and item.get("text"):
                parts.append(str(item["text"]))
                continue
            if item.get("type") == "output_text" and item.get("value"):
                parts.append(str(item["value"]))
                continue
        parts.append(str(item))
    return "\n".join(part for part in parts if part).strip()


class OpenAICompatibleClient(MultiModalLLMClient):
    """Chat-completions backend for OpenAI or OpenAI-compatible servers."""

    backend_name = "openai_compatible"

    def __init__(self, config: AppConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for the openai_compatible backend."
            ) from exc

        api_key = config.api_key or "EMPTY"
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.api_base_url or None,
            timeout=config.timeout_seconds,
        )
        self._config = config

    def _preferred_token_param(self) -> str:
        """Choose the initial token-limit parameter for the configured endpoint."""

        base_url = (self._config.api_base_url or "").lower()
        model_name = self._config.model_name.lower()
        if "api.openai.com" in base_url and model_name.startswith("gpt-5"):
            return "max_completion_tokens"
        return "max_tokens"

    def _create_completion_with_token_fallback(
        self,
        messages: list[dict[str, Any]],
        request: LLMRequest,
    ) -> Any:
        """Create one chat completion and retry if the token parameter is incompatible."""

        preferred_param = self._preferred_token_param()
        params: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": request.temperature,
            preferred_param: request.max_tokens,
        }

        try:
            return self._client.chat.completions.create(**params)
        except Exception as exc:
            alternate_param = "max_completion_tokens" if preferred_param == "max_tokens" else "max_tokens"
            message = str(exc)
            incompatible_param = (
                preferred_param in message and "not supported" in message
            ) or (
                preferred_param in message and alternate_param in message
            )
            if not incompatible_param:
                raise

            params.pop(preferred_param, None)
            params[alternate_param] = request.max_tokens
            return self._client.chat.completions.create(**params)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one multimodal request through chat completions."""

        user_content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        for image_path in request.image_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(image_path)},
                }
            )

        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": user_content})

        started_at = time.perf_counter()
        completion = self._create_completion_with_token_fallback(messages, request)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        payload = completion.model_dump(mode="json") if hasattr(completion, "model_dump") else dict(completion)
        choice = completion.choices[0]
        raw_text = flatten_message_content(choice.message.content)
        usage = payload.get("usage") if isinstance(payload, dict) else None

        return LLMResponse(
            answer_text=raw_text,
            raw_text=raw_text,
            raw_payload=payload if isinstance(payload, dict) else {"payload": str(payload)},
            model_name=str(getattr(completion, "model", "") or self._config.model_name),
            latency_ms=latency_ms,
            usage=usage if isinstance(usage, dict) else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )
