"""Prediction evaluation and metrics aggregation for direct QA."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from src.common.answer_norm import normalize_answer
from src.common.variant_metadata import (
    aggregate_accuracy_by_variation_combination,
    aggregate_accuracy_by_variation_type,
    aggregate_variation_combination_counts,
    aggregate_variation_field_counts,
    build_base_variant_comparison,
    variant_metadata_from_obj,
)
from src.contracts import QAExample
from src.pipeline.response_parser import ParsedModelAnswer


def infer_error_type(
    request_error: str | None,
    parsed_answer: ParsedModelAnswer,
    normalized_pred: str | None,
    correct: bool,
) -> str | None:
    """Assign a coarse error bucket to a prediction."""

    if correct:
        return None
    if request_error:
        return "request_failed"
    if parsed_answer.parse_status == "empty_response":
        return "empty_response"
    if normalized_pred is None:
        return "unparseable_answer"
    return "wrong_answer"


def build_prediction_record(
    example: QAExample,
    raw_row: dict[str, Any],
    parsed_answer: ParsedModelAnswer,
) -> dict[str, Any]:
    """Build the final per-QA prediction record."""

    normalized_gt = normalize_answer(example.gt_answer, example.answer_type, example.task_type)
    normalized_pred = normalize_answer(parsed_answer.answer_text, example.answer_type, example.task_type)
    correct = normalized_pred is not None and normalized_pred == normalized_gt
    request_error = raw_row.get("request_error")

    return {
        "sample_id": example.sample_id,
        "base_id": example.base_id,
        "qa_id": example.qa_id,
        "image_path": example.image_path,
        "chart_type": example.chart_type,
        "split": example.split,
        "style_id": example.style_id,
        **variant_metadata_from_obj(example),
        "task_type": example.task_type,
        "answer_type": example.answer_type,
        "gt_answer": example.gt_answer,
        "pred_answer": parsed_answer.answer_text,
        "normalized_gt_answer": normalized_gt,
        "normalized_pred_answer": normalized_pred,
        "correct": correct,
        "response_parse_status": parsed_answer.parse_status,
        "response_parse_error": parsed_answer.parse_error,
        "error_type": infer_error_type(request_error, parsed_answer, normalized_pred, correct),
        "request_error": request_error,
        "backend": raw_row.get("backend"),
        "model_name": raw_row.get("model_name"),
        "latency_ms": raw_row.get("latency_ms"),
        "finish_reason": raw_row.get("finish_reason"),
    }


def aggregate_accuracy(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Aggregate accuracy by a categorical key."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)

    summary: dict[str, Any] = {}
    for group_key, group_records in sorted(grouped.items()):
        count = len(group_records)
        correct = sum(1 for record in group_records if record["correct"])
        summary[group_key] = {
            "count": count,
            "correct": correct,
            "accuracy": correct / count if count else 0.0,
        }
    return summary


def aggregate_distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Aggregate value counts for a categorical key."""

    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def build_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the run-level metrics summary."""

    correct = sum(1 for record in predictions if record["correct"])
    latencies = [float(record["latency_ms"]) for record in predictions if record["latency_ms"] is not None]

    return {
        "num_qa": len(predictions),
        "qa_accuracy": {
            "correct": correct,
            "count": len(predictions),
            "accuracy": correct / len(predictions) if predictions else 0.0,
        },
        "avg_latency_ms": mean(latencies) if latencies else None,
        "by_chart_type": aggregate_accuracy(predictions, "chart_type"),
        "by_task_type": aggregate_accuracy(predictions, "task_type"),
        "by_split": aggregate_accuracy(predictions, "split"),
        "by_style_id": aggregate_accuracy(predictions, "style_id"),
        "by_variant_id": aggregate_accuracy(predictions, "variant_id"),
        "by_variant_kind": aggregate_accuracy(predictions, "variant_kind"),
        "by_variation_group": aggregate_accuracy(predictions, "variation_group"),
        "by_variation_type": aggregate_accuracy_by_variation_type(predictions),
        "by_variation_combination": aggregate_accuracy_by_variation_combination(predictions),
        "variation_field_counts": aggregate_variation_field_counts(predictions),
        "variation_combination_counts": aggregate_variation_combination_counts(predictions),
        "base_variant_comparison": build_base_variant_comparison(predictions),
        "response_parse_status_distribution": aggregate_distribution(predictions, "response_parse_status"),
        "error_type_distribution": aggregate_distribution(
            [record for record in predictions if record["error_type"] is not None],
            "error_type",
        ),
    }
