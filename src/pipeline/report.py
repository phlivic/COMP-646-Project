"""Markdown reporting for direct-QA benchmark runs."""

from __future__ import annotations

from typing import Any


def build_report(summary: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    """Build a compact markdown report for one MM-LLM run."""

    lines: list[str] = []
    lines.append("# MM-LLM Direct-QA Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- QA records: {summary['num_qa']}")
    lines.append(f"- QA accuracy: {summary['qa_accuracy']['accuracy']:.4f}")
    if summary["avg_latency_ms"] is not None:
        lines.append(f"- Average latency (ms): {summary['avg_latency_ms']:.2f}")
    lines.append("")
    lines.append("## QA Accuracy by Task")
    lines.append("")
    lines.append("| task_type | correct | count | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for task_type, stats in summary["by_task_type"].items():
        lines.append(
            f"| {task_type} | {stats['correct']} | {stats['count']} | {stats['accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## QA Accuracy by Chart")
    lines.append("")
    lines.append("| chart_type | correct | count | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for chart_type, stats in summary["by_chart_type"].items():
        lines.append(
            f"| {chart_type} | {stats['correct']} | {stats['count']} | {stats['accuracy']:.4f} |"
        )

    failures = [record for record in predictions if not record["correct"]][:10]
    if failures:
        lines.append("")
        lines.append("## Sample Failures")
        lines.append("")
        lines.append("| sample_id | task_type | gt | pred | error_type |")
        lines.append("|---|---|---|---|---|")
        for record in failures:
            pred_answer = record["pred_answer"] if record["pred_answer"] is not None else ""
            error_type = record["error_type"] if record["error_type"] is not None else ""
            lines.append(
                f"| {record['sample_id']} | {record['task_type']} | {record['gt_answer']} | {pred_answer} | {error_type} |"
            )

    return "\n".join(lines) + "\n"
