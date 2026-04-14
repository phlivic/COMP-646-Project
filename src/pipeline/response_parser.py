"""Robust parsing for model outputs that should contain JSON answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedModelAnswer:
    """Parsed answer extracted from a model response."""

    answer_text: str | None
    parse_status: str
    parse_error: str | None = None


def extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from free-form text."""

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_answer(text: str) -> ParsedModelAnswer | None:
    """Try to parse a JSON object with an answer field."""

    candidates = [text]
    fenced_match = FENCED_JSON_PATTERN.search(text)
    if fenced_match is not None:
        candidates.append(fenced_match.group(1).strip())
    extracted = extract_first_json_object(text)
    if extracted is not None:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "answer" in payload:
            answer_value = payload["answer"]
            if answer_value is None:
                return ParsedModelAnswer(answer_text=None, parse_status="json_missing_answer")
            return ParsedModelAnswer(answer_text=str(answer_value), parse_status="json_ok")
    return None


def parse_model_response(raw_text: str) -> ParsedModelAnswer:
    """Extract a usable answer from model output."""

    stripped = raw_text.strip()
    if not stripped:
        return ParsedModelAnswer(answer_text=None, parse_status="empty_response")

    parsed = parse_json_answer(stripped)
    if parsed is not None:
        return parsed

    return ParsedModelAnswer(
        answer_text=stripped,
        parse_status="plain_text_fallback",
        parse_error="Response was not valid JSON.",
    )
