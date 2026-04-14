"""Entry point for the PaddleOCR chart-QA baseline.

The main script stays intentionally small: it parses CLI arguments, coordinates
the end-to-end pipeline, and writes the final report. Chart parsing, OCR token
normalization, and evaluation logic live in neighboring modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    # Allow `python PaddleOCR/run_ocr.py ...` from the repository root while
    # keeping the project on standard package imports.
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from PaddleOCR.ocr_engine import build_ocr_engine, filter_tokens, normalize_ocr_result, run_ocr_on_image
from PaddleOCR.ocr_io import build_image_cases, load_metadata, load_raw_cache, write_jsonl
from PaddleOCR.ocr_logic import (
    aggregate_accuracy,
    aggregate_distribution,
    aggregate_line_point_metrics,
    aggregate_visible_text_metrics,
    answer_task,
    compute_line_point_metrics,
    compute_task_evidence_metrics,
    compute_visible_text_metrics,
    infer_error_type,
    infer_ocr_status,
    normalize_answer_for_compare,
    parse_chart,
)
from PaddleOCR.ocr_types import QAPrediction


REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def load_dotenv_defaults(env_path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from the repository .env file."""

    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values


def build_cli_defaults() -> dict[str, Any]:
    """Build CLI defaults from the repository .env file."""

    env_values = load_dotenv_defaults(ENV_PATH)
    return {
        "dataset_dir": env_values.get("MAIN_DATASET_DIR", "datasets/out"),
        "metadata_path": env_values.get("MAIN_METADATA_PATH") or None,
        "output_dir": env_values.get("PADDLEOCR_OUTPUT_DIR", "PaddleOCR/runs/latest"),
    }


def resolve_repo_path(path_value: str | None) -> Path | None:
    """Resolve relative paths against the repository root."""

    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PaddleOCR baseline run."""

    defaults = build_cli_defaults()
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR baseline over a generated chart QA dataset."
    )
    parser.add_argument("--dataset-dir", type=str, default=defaults["dataset_dir"])
    parser.add_argument("--metadata-path", type=str, default=defaults["metadata_path"])
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--min-ocr-score", type=float, default=0.30)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--reuse-ocr-cache", action="store_true")
    parser.add_argument("--text-det-limit-side-len", type=int, default=960)
    parser.add_argument(
        "--text-det-limit-type",
        type=str,
        default="max",
        choices=["max", "min"],
    )
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-textline-orientation", action="store_true")
    return parser.parse_args()


def build_report(
    summary: dict[str, Any],
    image_metrics: Sequence[dict[str, Any]],
    qa_predictions: Sequence[dict[str, Any]],
) -> str:
    """Build a compact markdown report for one OCR run."""

    lines: list[str] = []
    lines.append("# PaddleOCR Baseline Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Images: {summary['num_images']}")
    lines.append(f"- QA records: {summary['num_qa']}")
    lines.append(f"- QA accuracy: {summary['qa_accuracy']['accuracy']:.4f}")
    lines.append(f"- Evidence complete rate: {summary['task_evidence']['complete_rate']:.4f}")
    lines.append(
        f"- Accuracy given complete evidence: {summary['task_evidence']['accuracy_given_complete_evidence']:.4f}"
    )
    lines.append("")
    lines.append("## Raw OCR Text")
    lines.append("")
    lines.append("| component | matched | expected | recall | precision |")
    lines.append("|---|---:|---:|---:|---:|")
    for component, stats in summary["raw_visible_text_ocr"].items():
        lines.append(
            f"| {component} | {stats['matched_count']} | {stats['expected_count']} | {stats['recall']:.4f} | {stats['precision']:.4f} |"
        )
    lines.append("")
    lines.append("## Line Point Recovery")
    lines.append("")
    lines.append("| images | matched | recovered | expected | recall | precision | full_match_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    line_metrics = summary["line_point_recovery"]
    lines.append(
        f"| {line_metrics['image_count']} | {line_metrics['matched_count']} | {line_metrics['recovered_count']} | {line_metrics['expected_count']} | {line_metrics['recall']:.4f} | {line_metrics['precision']:.4f} | {line_metrics['full_match_rate']:.4f} |"
    )
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

    failed_predictions = [record for record in qa_predictions if not record["correct"]][:10]
    if failed_predictions:
        lines.append("")
        lines.append("## Sample Failures")
        lines.append("")
        lines.append("| sample_id | task_type | gt | pred | error_type |")
        lines.append("|---|---|---|---|---|")
        for record in failed_predictions:
            lines.append(
                f"| {record['sample_id']} | {record['task_type']} | {record['gt_answer']} | {record['pred_answer']} | {record['error_type']} |"
            )

    partial_images = [record for record in image_metrics if record["parse_status"] != "ok"][:10]
    if partial_images:
        lines.append("")
        lines.append("## Partial Or Failed Parses")
        lines.append("")
        lines.append("| image_path | chart_type | parse_status | token_count | warnings |")
        lines.append("|---|---|---|---:|---|")
        for record in partial_images:
            warnings = "; ".join(record["warnings"][:3])
            lines.append(
                f"| {record['image_path']} | {record['chart_type']} | {record['parse_status']} | {record['token_count']} | {warnings} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the full OCR baseline pipeline over a chart dataset."""

    args = parse_args()
    dataset_dir = resolve_repo_path(args.dataset_dir)
    if dataset_dir is None:
        raise ValueError("dataset_dir must not be empty.")
    metadata_path = resolve_repo_path(args.metadata_path) if args.metadata_path else dataset_dir / "metadata.jsonl"
    output_dir = resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must not be empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "ocr_vis" if args.save_vis else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    metadata_rows = load_metadata(metadata_path)
    cases = build_image_cases(metadata_rows, dataset_dir, args.split, args.limit_images)

    raw_cache_path = output_dir / "ocr_raw.jsonl"
    raw_cache = load_raw_cache(raw_cache_path) if args.reuse_ocr_cache else {}

    engine = build_ocr_engine(args)

    raw_records: list[dict[str, Any]] = []
    token_records: list[dict[str, Any]] = []
    parsed_chart_records: list[dict[str, Any]] = []
    image_metrics: list[dict[str, Any]] = []
    qa_predictions: list[dict[str, Any]] = []

    for case in cases:
        raw_record = raw_cache.get(case.image_path)
        if raw_record is None:
            raw_record = run_ocr_on_image(engine, case, args, vis_dir)
        raw_records.append(raw_record)

        tokens = filter_tokens(
            normalize_ocr_result(raw_record.get("res_json", {})),
            args.min_ocr_score,
        )
        ocr_status = infer_ocr_status(tokens)
        raw_visible_text_metrics = compute_visible_text_metrics(case, tokens)
        parsed_chart = parse_chart(tokens, case)
        line_point_metrics = compute_line_point_metrics(parsed_chart, case)

        token_records.append(
            {
                "image_path": case.image_path,
                "chart_type": case.chart_type,
                "split": case.split,
                "style_id": case.style_id,
                "token_count": len(tokens),
                "tokens": [asdict(token) for token in tokens],
            }
        )
        parsed_chart_records.append(
            {
                "image_path": case.image_path,
                "chart_type": case.chart_type,
                "split": case.split,
                "style_id": case.style_id,
                **asdict(parsed_chart),
            }
        )
        image_metrics.append(
            {
                "image_path": case.image_path,
                "chart_type": case.chart_type,
                "split": case.split,
                "style_id": case.style_id,
                "token_count": len(tokens),
                "ocr_status": ocr_status,
                "parse_status": parsed_chart.parse_status,
                "warnings": parsed_chart.warnings,
                "raw_visible_text_metrics": raw_visible_text_metrics,
                "line_point_metrics": line_point_metrics,
            }
        )

        for qa in case.qa_records:
            evidence = compute_task_evidence_metrics(parsed_chart, qa, case)
            pred_answer = answer_task(parsed_chart, qa)
            gt_norm = normalize_answer_for_compare(qa.gt_answer, qa.answer_type)
            pred_norm = normalize_answer_for_compare(pred_answer, qa.answer_type)
            correct = pred_norm is not None and pred_norm == gt_norm
            error_type = infer_error_type(pred_answer, correct, ocr_status, parsed_chart, evidence)
            prediction = QAPrediction(
                sample_id=qa.sample_id,
                image_path=qa.image_path,
                chart_type=qa.chart_type,
                split=qa.split,
                style_id=qa.style_id,
                task_type=qa.task_type,
                answer_type=qa.answer_type,
                gt_answer=qa.gt_answer,
                pred_answer=pred_answer,
                correct=correct,
                ocr_status=ocr_status,
                parse_status=parsed_chart.parse_status,
                evidence_type=evidence["evidence_type"],
                evidence_complete=evidence["evidence_complete"],
                evidence_status=evidence["evidence_status"],
                recovered_count=evidence["recovered_count"],
                required_count=evidence["required_count"],
                missing_count=evidence["missing_count"],
                error_type=error_type,
            )
            qa_predictions.append(asdict(prediction))

    engine.close()

    correct_count = sum(1 for record in qa_predictions if record["correct"])
    complete_predictions = [record for record in qa_predictions if record["evidence_complete"]]
    correct_with_incomplete = [
        record for record in qa_predictions if record["correct"] and not record["evidence_complete"]
    ]

    summary = {
        "num_images": len(cases),
        "num_qa": len(qa_predictions),
        "ocr_status_distribution": aggregate_distribution(image_metrics, "ocr_status"),
        "parse_status_distribution": aggregate_distribution(image_metrics, "parse_status"),
        "error_type_distribution": aggregate_distribution(
            [record for record in qa_predictions if record["error_type"] is not None],
            "error_type",
        ),
        "qa_accuracy": {
            "correct": correct_count,
            "count": len(qa_predictions),
            "accuracy": correct_count / len(qa_predictions) if qa_predictions else 0.0,
        },
        "raw_visible_text_ocr": aggregate_visible_text_metrics(image_metrics),
        "line_point_recovery": aggregate_line_point_metrics(image_metrics),
        "task_evidence": {
            "complete_count": len(complete_predictions),
            "count": len(qa_predictions),
            "complete_rate": len(complete_predictions) / len(qa_predictions) if qa_predictions else 0.0,
            "accuracy_given_complete_evidence": (
                sum(1 for record in complete_predictions if record["correct"]) / len(complete_predictions)
                if complete_predictions
                else 0.0
            ),
            "correct_with_incomplete_evidence": len(correct_with_incomplete),
        },
        "by_chart_type": aggregate_accuracy(qa_predictions, "chart_type"),
        "by_task_type": aggregate_accuracy(qa_predictions, "task_type"),
        "by_split": aggregate_accuracy(qa_predictions, "split"),
        "by_style_id": aggregate_accuracy(qa_predictions, "style_id"),
    }

    write_jsonl(raw_cache_path, raw_records)
    write_jsonl(output_dir / "ocr_tokens.jsonl", token_records)
    write_jsonl(output_dir / "parsed_charts.jsonl", parsed_chart_records)
    write_jsonl(output_dir / "image_metrics.jsonl", image_metrics)
    write_jsonl(output_dir / "qa_predictions.jsonl", qa_predictions)

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_report(summary, image_metrics, qa_predictions))

    print(f"[Done] Processed images: {len(cases)}")
    print(f"[Done] QA records: {len(qa_predictions)}")
    print(f"[Done] Output dir: {output_dir}")
    print(f"[Done] Summary: {output_dir / 'metrics_summary.json'}")
    print(f"[Done] Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
