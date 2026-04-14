"""Prompt builders for chart direct-QA."""

from __future__ import annotations

from src.contracts import QAExample


def build_system_prompt(version: str) -> str:
    """Build the system prompt used for direct chart QA."""

    if version != "chartqa_direct_v1":
        raise ValueError(f"Unsupported prompt version: {version}")

    return (
        "You are a precise chart question answering model. "
        "Answer the question using only the provided chart image. "
        "Return valid JSON with exactly one key named answer. "
        "Do not include markdown, code fences, or extra keys. "
        "For numeric answers, return only the number string without units. "
        "For category answers, return labels like Cat-3. "
        "For trend answers, return exactly one of increasing, decreasing, or fluctuating."
    )


def build_user_prompt(example: QAExample) -> str:
    """Build the user prompt for one QA example."""

    format_hint = "Return JSON only."
    if example.answer_type == "number":
        format_hint = 'Return JSON only. Example: {"answer": "47.5"}'
    elif example.answer_type == "category":
        format_hint = 'Return JSON only. Example: {"answer": "Cat-2"}'
    elif example.task_type == "line_trend":
        format_hint = 'Return JSON only. Example: {"answer": "increasing"}'

    return (
        f"Question: {example.question}\n"
        f"Expected answer type: {example.answer_type}\n"
        f"{format_hint}"
    )
