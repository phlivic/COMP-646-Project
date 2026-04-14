"""Shared data contracts for the MM-LLM benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QAExample:
    """One chart-QA evaluation record paired with a rendered chart image."""

    sample_id: str
    base_id: str
    style_id: str
    qa_id: str
    task_type: str
    answer_type: str
    chart_type: str
    split: str
    image_path: str
    abs_image_path: Path
    question: str
    gt_answer: str


@dataclass(frozen=True)
class LLMRequest:
    """A normalized multimodal generation request."""

    system_prompt: str
    user_prompt: str
    image_paths: list[Path]
    temperature: float
    max_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """A normalized multimodal generation response."""

    answer_text: str
    raw_text: str
    raw_payload: dict[str, Any]
    model_name: str
    latency_ms: float
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
