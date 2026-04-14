"""Answer normalization tuned for generative chart-QA outputs."""

from __future__ import annotations

import re


CATEGORY_PATTERN = re.compile(r"(?i)\bcat\s*[- ]?\s*(\d+)\b")
NUMBER_PATTERN = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?(?!\s*%)")
TREND_LABELS = ("increasing", "decreasing", "fluctuating")


def normalize_text(text: str) -> str:
    """Normalize free-form text for stable comparison."""

    text = str(text).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_number(x: float) -> str:
    """Format numbers to match the dataset answer strings."""

    return f"{x:.2f}".rstrip("0").rstrip(".")


def extract_number(text: str | None) -> float | None:
    """Extract the first plain number from a model answer."""

    if text is None:
        return None
    normalized = normalize_text(text).replace(",", "")
    exact = NUMBER_PATTERN.fullmatch(normalized)
    if exact:
        return float(exact.group(0))
    match = NUMBER_PATTERN.search(normalized)
    if match is None:
        return None
    return float(match.group(0))


def extract_category(text: str | None) -> str | None:
    """Extract a Cat-N label from a model answer."""

    if text is None:
        return None
    match = CATEGORY_PATTERN.search(normalize_text(text))
    if match is None:
        return None
    return f"Cat-{int(match.group(1))}"


def extract_trend_label(text: str | None) -> str | None:
    """Extract one of the supported line-trend labels from a model answer."""

    if text is None:
        return None
    lowered = normalize_text(text).lower()
    for label in TREND_LABELS:
        if re.search(rf"\b{label}\b", lowered):
            return label
    return None


def normalize_answer(answer: str | None, answer_type: str, task_type: str) -> str | None:
    """Normalize a predicted or ground-truth answer into comparison form."""

    if answer is None:
        return None

    if answer_type == "number":
        value = extract_number(answer)
        return format_number(value) if value is not None else None

    if answer_type == "category":
        return extract_category(answer)

    if task_type == "line_trend":
        return extract_trend_label(answer)

    return normalize_text(answer).lower()
