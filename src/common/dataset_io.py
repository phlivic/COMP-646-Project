"""Dataset IO helpers for the MM-LLM benchmark pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.common.variant_metadata import extract_variant_metadata
from src.contracts import QAExample


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


def build_qa_examples(
    metadata_rows: list[dict[str, Any]],
    dataset_dir: Path,
    split: str,
    limit_qa: int | None,
) -> list[QAExample]:
    """Build QAExample objects from dataset metadata."""

    filtered_rows = [
        row
        for row in metadata_rows
        if split == "all" or str(row.get("split")) == split
    ]
    filtered_rows = sorted(filtered_rows, key=lambda row: str(row["sample_id"]))
    if limit_qa is not None:
        filtered_rows = filtered_rows[:limit_qa]

    examples: list[QAExample] = []
    for row in filtered_rows:
        image_path = str(row["image_path"])
        abs_image_path = dataset_dir / image_path
        variant_metadata = extract_variant_metadata(row)
        examples.append(
            QAExample(
                sample_id=str(row["sample_id"]),
                base_id=str(row["base_id"]),
                style_id=str(row.get("style_id") or variant_metadata["variant_id"]),
                **variant_metadata,
                qa_id=str(row["qa_id"]),
                task_type=str(row["task_type"]),
                answer_type=str(row["answer_type"]),
                chart_type=str(row["chart_type"]),
                split=str(row["split"]),
                image_path=image_path,
                abs_image_path=abs_image_path,
                question=str(row["question"]),
                gt_answer=str(row["answer"]),
            )
        )
    return examples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_cache(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    """Load a JSONL cache file keyed by one field."""

    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[str(row[key_field])] = row
    return cache
