"""Backend interface for multimodal generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.contracts import LLMRequest, LLMResponse


class MultiModalLLMClient(ABC):
    """Abstract backend for multimodal direct-QA generation."""

    backend_name: str

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response from a multimodal request."""

    def close(self) -> None:
        """Release backend resources when needed."""

    def __enter__(self) -> "MultiModalLLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
