"""Core data models, constants, and shared parsing helpers for OCR benchmarking.

This module keeps the pieces that are used across IO, OCR inference, chart parsing,
and evaluation. The goal is to centralize common semantics so every layer refers
to the same chart/task definitions and text normalization rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


TITLE_BY_CHART = {
    "bar": "Bar Chart",
    "line": "Line Chart",
    "pie": "Pie Chart",
}

AXIS_LABELS_BY_CHART = {
    "bar": ("Category", "Value"),
    "line": ("Node", "Value"),
    "pie": (),
}

CATEGORY_PATTERN = re.compile(r"(?i)^cat\s*[- ]?\s*(\d+)$")
NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
PERCENT_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?%$")


@dataclass(frozen=True)
class QARecord:
    """One evaluation record from metadata.jsonl."""

    sample_id: str
    base_id: str
    style_id: str
    variant_id: str
    is_base_variant: bool
    variant_kind: str
    variation_types: list[str]
    variation_group: str
    source_image_path: str | None
    render_style: dict[str, Any] | None
    blur: dict[str, Any] | None
    noise: dict[str, Any] | None
    compression: dict[str, Any] | None
    resize: dict[str, Any] | None
    brightness: dict[str, Any] | None
    contrast: dict[str, Any] | None
    qa_id: str
    task_type: str
    answer_type: str
    chart_type: str
    split: str
    image_path: str
    question: str
    gt_answer: str
    num_points: int


@dataclass(frozen=True)
class ImageCase:
    """All metadata associated with a single rendered chart image."""

    image_path: str
    abs_image_path: Path
    chart_type: str
    split: str
    base_id: str
    style_id: str
    variant_id: str
    is_base_variant: bool
    variant_kind: str
    variation_types: list[str]
    variation_group: str
    source_image_path: str | None
    render_style: dict[str, Any] | None
    blur: dict[str, Any] | None
    noise: dict[str, Any] | None
    compression: dict[str, Any] | None
    resize: dict[str, Any] | None
    brightness: dict[str, Any] | None
    contrast: dict[str, Any] | None
    num_points: int
    image_width: int
    image_height: int
    categories_gt: list[str]
    values_gt: list[float]
    line_trend_gt: str | None
    qa_records: list[QARecord]


@dataclass(frozen=True)
class OCRToken:
    """A normalized OCR token with geometry attached."""

    text: str
    norm_text: str
    score: float | None
    poly: list[Any]
    bbox: tuple[float, float, float, float]
    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class ParsedChart:
    """Structured chart content recovered from OCR tokens."""

    image_path: str
    chart_type: str
    parse_status: str
    categories: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    percentages: list[float] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    trend: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QAPrediction:
    """Final answer record for one QA item."""

    sample_id: str
    base_id: str
    qa_id: str
    image_path: str
    chart_type: str
    split: str
    style_id: str
    variant_id: str
    is_base_variant: bool
    variant_kind: str
    variation_types: list[str]
    variation_group: str
    source_image_path: str | None
    render_style: dict[str, Any] | None
    blur: dict[str, Any] | None
    noise: dict[str, Any] | None
    compression: dict[str, Any] | None
    resize: dict[str, Any] | None
    brightness: dict[str, Any] | None
    contrast: dict[str, Any] | None
    task_type: str
    answer_type: str
    gt_answer: str
    pred_answer: str | None
    correct: bool
    ocr_status: str
    parse_status: str
    evidence_type: str
    evidence_complete: bool
    evidence_status: str
    recovered_count: int
    required_count: int
    missing_count: int
    error_type: str | None


def normalize_text(text: str) -> str:
    """Normalize OCR text into a stable comparison form."""

    text = str(text).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*%\s*$", "%", text)
    return text


def format_number(x: float) -> str:
    """Format numbers to match dataset answer strings."""

    return f"{x:.2f}".rstrip("0").rstrip(".")


def format_percent(x: float) -> str:
    """Format percentages to match pie-chart labels."""

    return f"{x:.1f}%"


def normalize_key(text: str) -> str:
    """Lowercase comparison key used for stable exact matching."""

    return normalize_text(text).lower()


def parse_category(text: str) -> str | None:
    """Parse category labels such as 'Cat-3' from OCR text."""

    match = CATEGORY_PATTERN.fullmatch(normalize_text(text))
    if not match:
        return None
    return f"Cat-{int(match.group(1))}"


def parse_number(text: str) -> float | None:
    """Parse plain numeric OCR tokens while excluding categories and percents."""

    cleaned = normalize_text(text).replace(",", "")
    if parse_category(cleaned) is not None or cleaned.endswith("%"):
        return None
    if not NUMBER_PATTERN.fullmatch(cleaned):
        return None
    return float(cleaned)


def parse_percent(text: str) -> float | None:
    """Parse percentage OCR tokens such as '45.6%'."""

    cleaned = normalize_text(text).replace(",", "")
    if not PERCENT_PATTERN.fullmatch(cleaned):
        return None
    return float(cleaned[:-1])


def classify_line_trend(values: Sequence[float]) -> str:
    """Classify a line series into increasing, decreasing, or fluctuating."""

    if len(values) < 2:
        return "fluctuating"
    increasing = all(values[i] < values[i + 1] for i in range(len(values) - 1))
    decreasing = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    if increasing:
        return "increasing"
    if decreasing:
        return "decreasing"
    return "fluctuating"
