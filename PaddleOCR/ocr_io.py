"""Dataset IO and lightweight persistence helpers for OCR benchmarking.

This module owns metadata loading, image-case construction, and JSONL helpers.
It deliberately stays simple so the entry script can focus on orchestration
instead of file-format details.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.common.variant_metadata import extract_variant_metadata

from .ocr_types import ImageCase, QARecord


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load metadata.jsonl into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image size without loading the full image into memory."""

    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def build_image_cases(
    metadata_rows: Sequence[dict[str, Any]],
    dataset_dir: Path,
    split: str,
    limit_images: int | None,
) -> list[ImageCase]:
    """Group QA rows by image so each chart is OCRed exactly once."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        if split != "all" and row["split"] != split:
            continue
        grouped[row["image_path"]].append(row)

    image_paths = sorted(grouped.keys())
    if limit_images is not None:
        image_paths = image_paths[:limit_images]

    cases: list[ImageCase] = []
    for image_path in image_paths:
        rows = sorted(grouped[image_path], key=lambda item: item["sample_id"])
        first = rows[0]
        case_variant_metadata = extract_variant_metadata(first)
        abs_image_path = dataset_dir / image_path
        width, height = read_image_size(abs_image_path)
        qa_records = []
        for row in rows:
            row_variant_metadata = extract_variant_metadata(row)
            qa_records.append(
                QARecord(
                    sample_id=row["sample_id"],
                    base_id=row["base_id"],
                    style_id=str(row.get("style_id") or row_variant_metadata["variant_id"]),
                    **row_variant_metadata,
                    qa_id=row["qa_id"],
                    task_type=row["task_type"],
                    answer_type=row["answer_type"],
                    chart_type=row["chart_type"],
                    split=row["split"],
                    image_path=row["image_path"],
                    question=row["question"],
                    gt_answer=str(row["answer"]),
                    num_points=int(row["num_points"]),
                )
            )
        cases.append(
            ImageCase(
                image_path=image_path,
                abs_image_path=abs_image_path,
                chart_type=first["chart_type"],
                split=first["split"],
                base_id=first["base_id"],
                style_id=str(first.get("style_id") or case_variant_metadata["variant_id"]),
                **case_variant_metadata,
                num_points=int(first["num_points"]),
                image_width=width,
                image_height=height,
                categories_gt=[str(x) for x in first.get("categories", [])],
                values_gt=[float(x) for x in first.get("values", [])],
                line_trend_gt=first.get("line_trend"),
                qa_records=qa_records,
            )
        )
    return cases


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_raw_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load cached OCR raw outputs keyed by image path."""

    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["image_path"]] = row
    return cache
