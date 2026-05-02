"""Shared helpers for dataset variant metadata and variant-level metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, Mapping, Sequence


POSTPROCESS_FIELDS: tuple[str, ...] = (
    "blur",
    "noise",
    "compression",
    "resize",
    "brightness",
    "contrast",
)

VARIANT_METADATA_FIELDS: tuple[str, ...] = (
    "variant_id",
    "is_base_variant",
    "variant_kind",
    "variation_types",
    "variation_group",
    "source_image_path",
    "render_style",
    *POSTPROCESS_FIELDS,
)


def _is_non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if not _is_non_empty(value):
        return None
    if isinstance(value, dict):
        return value
    return {"value": value}


def _normalize_variation_types(value: Any) -> list[str]:
    if not _is_non_empty(value):
        return []
    if isinstance(value, str):
        raw_parts = value.replace("+", ",").split(",")
        return [part.strip() for part in raw_parts if part.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _default_variation_group(
    variant_id: str,
    is_base_variant: bool,
    variation_types: Sequence[str],
) -> str:
    if is_base_variant or not variation_types:
        return "base"
    return str(variant_id)


def extract_variant_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract normalized variant metadata from a dataset row.

    Older metadata files only have ``style_id`` and ``style_factors``. This
    helper maps them into the newer variant schema so existing datasets remain
    readable by the runners.
    """

    variant_id = str(row.get("variant_id") or row.get("style_id") or "base")
    transform_payloads = {
        field: _optional_dict(row.get(field))
        for field in POSTPROCESS_FIELDS
    }
    raw_render_style = row.get("render_style") if "render_style" in row else row.get("style_factors")
    render_style = _optional_dict(raw_render_style)

    variation_types = _normalize_variation_types(row.get("variation_types"))
    if not variation_types:
        variation_types = [
            field
            for field in POSTPROCESS_FIELDS
            if transform_payloads[field] is not None
        ]
    if not variation_types and render_style is not None and variant_id not in {"base", "reference"}:
        variation_types = ["render_style"]

    inferred_base = variant_id in {"base", "reference"} and not variation_types
    is_base_variant = _parse_bool(row.get("is_base_variant"), inferred_base)

    variant_kind = row.get("variant_kind")
    if not variant_kind:
        if is_base_variant:
            variant_kind = "base"
        elif any(transform_payloads[field] is not None for field in POSTPROCESS_FIELDS):
            variant_kind = "postprocess"
        else:
            variant_kind = "render_style"

    variation_group = str(
        row.get("variation_group")
        or _default_variation_group(variant_id, is_base_variant, variation_types)
    )

    return {
        "variant_id": variant_id,
        "is_base_variant": is_base_variant,
        "variant_kind": str(variant_kind),
        "variation_types": list(variation_types),
        "variation_group": variation_group,
        "source_image_path": (
            str(row["source_image_path"])
            if _is_non_empty(row.get("source_image_path"))
            else None
        ),
        "render_style": render_style,
        **transform_payloads,
    }


def variant_metadata_from_obj(obj: Any) -> dict[str, Any]:
    """Read variant metadata fields from a dataclass-like object."""

    return {
        field: getattr(obj, field)
        for field in VARIANT_METADATA_FIELDS
    }


def active_variation_types(record: Mapping[str, Any]) -> list[str]:
    """Return the active variation types for one prediction or metric record."""

    variation_types = _normalize_variation_types(record.get("variation_types"))
    for field in POSTPROCESS_FIELDS:
        if _is_non_empty(record.get(field)) and field not in variation_types:
            variation_types.append(field)
    if _is_non_empty(record.get("render_style")) and "render_style" not in variation_types:
        variation_types.append("render_style")
    if not variation_types and _parse_bool(record.get("is_base_variant"), False):
        return []
    return variation_types


def variation_combination_key(record: Mapping[str, Any]) -> str:
    """Return a stable key for the set of active variation types."""

    variation_types = active_variation_types(record)
    if not variation_types:
        return "base"
    return "+".join(sorted(set(variation_types)))


def _accuracy_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    correct = sum(1 for record in records if bool(record.get("correct")))
    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
    }


def aggregate_accuracy_by_variation_type(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate accuracy with multi-label membership for variations."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        variation_types = active_variation_types(record)
        if not variation_types:
            grouped["base"].append(record)
            continue
        for variation_type in variation_types:
            grouped[str(variation_type)].append(record)
    return {
        key: _accuracy_payload(group_records)
        for key, group_records in sorted(grouped.items())
    }


def aggregate_accuracy_by_variation_combination(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate accuracy by the exact set of active variation fields."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[variation_combination_key(record)].append(record)
    return {
        key: _accuracy_payload(group_records)
        for key, group_records in sorted(grouped.items())
    }


def aggregate_variation_field_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count QA records with each post-processing field present."""

    return {
        field: sum(1 for record in records if _is_non_empty(record.get(field)))
        for field in POSTPROCESS_FIELDS
    }


def aggregate_variation_combination_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count QA records by active variation field combination."""

    return dict(sorted(Counter(variation_combination_key(record) for record in records).items()))


def build_base_variant_comparison(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare each non-base prediction against the matching base prediction.

    Matching is performed at ``(base_id, qa_id)`` granularity, which makes the
    comparison robust to random image variants while preserving the same chart
    semantics and question.
    """

    base_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if _parse_bool(record.get("is_base_variant"), False):
            base_by_key[(str(record.get("base_id")), str(record.get("qa_id")))] = record

    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for record in records:
        if _parse_bool(record.get("is_base_variant"), False):
            continue
        base_record = base_by_key.get((str(record.get("base_id")), str(record.get("qa_id"))))
        if base_record is None:
            continue
        grouped[str(record.get("variation_group", "unknown"))].append((base_record, record))

    summary: dict[str, Any] = {}
    for group_key, pairs in sorted(grouped.items()):
        count = len(pairs)
        base_correct = sum(1 for base_record, _ in pairs if bool(base_record.get("correct")))
        variant_correct = sum(1 for _, variant_record in pairs if bool(variant_record.get("correct")))
        base_correct_variant_wrong = sum(
            1
            for base_record, variant_record in pairs
            if bool(base_record.get("correct")) and not bool(variant_record.get("correct"))
        )
        base_wrong_variant_correct = sum(
            1
            for base_record, variant_record in pairs
            if not bool(base_record.get("correct")) and bool(variant_record.get("correct"))
        )
        both_correct = sum(
            1
            for base_record, variant_record in pairs
            if bool(base_record.get("correct")) and bool(variant_record.get("correct"))
        )
        both_wrong = sum(
            1
            for base_record, variant_record in pairs
            if not bool(base_record.get("correct")) and not bool(variant_record.get("correct"))
        )
        base_accuracy = base_correct / count if count else 0.0
        variant_accuracy = variant_correct / count if count else 0.0
        summary[group_key] = {
            "count": count,
            "base_correct": base_correct,
            "variant_correct": variant_correct,
            "base_accuracy": base_accuracy,
            "variant_accuracy": variant_accuracy,
            "delta_accuracy": variant_accuracy - base_accuracy,
            "base_correct_variant_wrong": base_correct_variant_wrong,
            "base_wrong_variant_correct": base_wrong_variant_correct,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        }
    return summary
