"""End-to-end direct-QA runner for the MM-LLM benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.dataset_io import build_qa_examples, load_jsonl_cache, load_metadata, write_jsonl
from src.config import AppConfig
from src.contracts import LLMRequest, QAExample
from src.llm.factory import create_llm_client
from src.pipeline.evaluator import build_prediction_record, build_summary
from src.pipeline.prompts import build_system_prompt, build_user_prompt
from src.pipeline.report import build_report
from src.pipeline.response_parser import parse_model_response


def ensure_inputs_exist(examples: list[QAExample], metadata_path: Path) -> None:
    """Fail early when the dataset inputs are incomplete."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    missing_images = [example.abs_image_path for example in examples if not example.abs_image_path.exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing dataset image: {missing_images[0]}")


def build_request(example: QAExample, config: AppConfig) -> LLMRequest:
    """Build one multimodal request from a QA example."""

    return LLMRequest(
        system_prompt=build_system_prompt(config.prompt_version),
        user_prompt=build_user_prompt(example),
        image_paths=[example.abs_image_path],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        metadata={
            "sample_id": example.sample_id,
            "task_type": example.task_type,
            "chart_type": example.chart_type,
        },
    )


def build_raw_row(
    example: QAExample,
    request: LLMRequest,
    response: dict[str, Any],
    backend: str,
    model_name: str,
) -> dict[str, Any]:
    """Build a JSON-serializable raw response record."""

    return {
        "sample_id": example.sample_id,
        "image_path": example.image_path,
        "chart_type": example.chart_type,
        "split": example.split,
        "style_id": example.style_id,
        "task_type": example.task_type,
        "answer_type": example.answer_type,
        "question": example.question,
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "backend": backend,
        "model_name": model_name,
        **response,
    }


def run_pipeline(config: AppConfig) -> None:
    """Run the full MM-LLM direct-QA pipeline."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_json_dict(), handle, ensure_ascii=False, indent=2)

    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.metadata_path}")

    metadata_rows = load_metadata(config.metadata_path)
    examples = build_qa_examples(metadata_rows, config.dataset_dir, config.split, config.limit_qa)
    ensure_inputs_exist(examples, config.metadata_path)

    raw_cache_path = config.output_dir / "raw_responses.jsonl"
    raw_cache = load_jsonl_cache(raw_cache_path, "sample_id") if config.reuse_cache else {}
    cache_rows = dict(raw_cache)

    predictions: list[dict[str, Any]] = []

    if not examples:
        summary = build_summary(predictions)
        write_jsonl(raw_cache_path, sorted(cache_rows.values(), key=lambda row: str(row["sample_id"])))
        write_jsonl(config.output_dir / "qa_predictions.jsonl", predictions)
        with (config.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        with (config.output_dir / "report.md").open("w", encoding="utf-8") as handle:
            handle.write(build_report(summary, predictions))
        print("[Done] QA records: 0")
        print(f"[Done] Output dir: {config.output_dir}")
        print(f"[Done] Summary: {config.output_dir / 'metrics_summary.json'}")
        print(f"[Done] Report: {config.output_dir / 'report.md'}")
        return

    with create_llm_client(config) as client:
        for example in examples:
            raw_row = raw_cache.get(example.sample_id)
            if raw_row is None:
                request = build_request(example, config)
                try:
                    response = client.generate(request)
                    raw_row = build_raw_row(
                        example,
                        request,
                        {
                            "raw_response_text": response.raw_text,
                            "raw_payload": response.raw_payload,
                            "latency_ms": response.latency_ms,
                            "usage": response.usage,
                            "finish_reason": response.finish_reason,
                            "request_error": None,
                        },
                        backend=client.backend_name,
                        model_name=response.model_name,
                    )
                except Exception as exc:
                    raw_row = build_raw_row(
                        example,
                        request,
                        {
                            "raw_response_text": "",
                            "raw_payload": {},
                            "latency_ms": None,
                            "usage": None,
                            "finish_reason": None,
                            "request_error": str(exc),
                        },
                        backend=client.backend_name,
                        model_name=config.model_name,
                    )
            cache_rows[example.sample_id] = raw_row
            parsed_answer = parse_model_response(str(raw_row.get("raw_response_text", "")))
            predictions.append(build_prediction_record(example, raw_row, parsed_answer))

    summary = build_summary(predictions)
    write_jsonl(raw_cache_path, sorted(cache_rows.values(), key=lambda row: str(row["sample_id"])))
    write_jsonl(config.output_dir / "qa_predictions.jsonl", predictions)

    with (config.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with (config.output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_report(summary, predictions))

    print(f"[Done] QA records: {len(predictions)}")
    print(f"[Done] Output dir: {config.output_dir}")
    print(f"[Done] Summary: {config.output_dir / 'metrics_summary.json'}")
    print(f"[Done] Report: {config.output_dir / 'report.md'}")
