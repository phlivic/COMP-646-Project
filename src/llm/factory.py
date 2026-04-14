"""Factory for creating MM-LLM backends."""

from __future__ import annotations

from src.config import AppConfig
from src.llm.base import MultiModalLLMClient


def create_llm_client(config: AppConfig) -> MultiModalLLMClient:
    """Create the configured LLM backend."""

    if config.backend == "openai_compatible":
        from src.llm.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(config)

    if config.backend == "hf_local":
        from src.llm.hf_local import HuggingFaceLocalClient

        return HuggingFaceLocalClient(config)

    raise ValueError(f"Unsupported backend: {config.backend}")
