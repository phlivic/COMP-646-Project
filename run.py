"""OCR-augmented GPT-5.4 nano chart-QA benchmark entry point.

This root-level runner combines the existing PaddleOCR pass with the existing
OpenAI-compatible multimodal backend. OCR is run once per image, then each QA
request sends both the chart image and a compact OCR context to GPT-5.4 nano.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PaddleOCR.ocr_engine import filter_tokens, normalize_ocr_result, run_ocr_on_image
from PaddleOCR.ocr_engine import build_ocr_engine as build_paddle_ocr_engine
from PaddleOCR.ocr_io import build_image_cases, load_metadata, load_raw_cache, write_jsonl
from PaddleOCR.ocr_logic import infer_ocr_status
from PaddleOCR.ocr_types import ImageCase, OCRToken
from src.common.dataset_io import build_qa_examples, load_jsonl_cache
from src.common.variant_metadata import build_base_variant_comparison, variant_metadata_from_obj
from src.config import AppConfig, load_dotenv_defaults, parse_bool, parse_optional_int, resolve_repo_path
from src.contracts import LLMRequest, QAExample
from src.llm.factory import create_llm_client
from src.pipeline.evaluator import build_prediction_record, build_summary
from src.pipeline.report import build_report
from src.pipeline.response_parser import parse_model_response


def build_cli_defaults() -> dict[str, Any]:
    """Build CLI defaults from .env, with GPT-5.4 nano as the default model."""

    env_values = load_dotenv_defaults(ENV_PATH)
    return {
        "dataset_dir": env_values.get("MAIN_DATASET_DIR", "datasets/out"),
        "metadata_path": env_values.get("MAIN_METADATA_PATH") or None,
        "output_dir": env_values.get("OCR_GPT_OUTPUT_DIR", "runs/ocr_gpt54nano/latest"),
        "split": env_values.get("RUN_SPLIT", "test"),
        "limit_qa": parse_optional_int(env_values.get("RUN_LIMIT_QA")),
        "reuse_ocr_cache": parse_bool(env_values.get("RUN_REUSE_OCR_CACHE"), default=True),
        "reuse_response_cache": parse_bool(env_values.get("RUN_REUSE_CACHE"), default=True),
        "model_name": env_values.get("OPENAI_MODEL_USE") or env_values.get("LLM_MODEL_NAME") or "gpt-5.4-nano",
        "api_key": env_values.get("OPENAI_API_KEY") or env_values.get("LLM_API_KEY") or None,
        "api_base_url": env_values.get("LLM_API_BASE_URL") or None,
        "temperature": float(env_values.get("LLM_TEMPERATURE", "0")),
        "max_tokens": int(env_values.get("LLM_MAX_TOKENS", "128")),
        "timeout_seconds": int(env_values.get("LLM_TIMEOUT_SECONDS", "60")),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the OCR-augmented GPT runner."""

    defaults = build_cli_defaults()
    parser = argparse.ArgumentParser(
        description="Run OCR first, then send chart image + OCR context to GPT-5.4 nano."
    )
    parser.add_argument("--dataset-dir", type=str, default=defaults["dataset_dir"])
    parser.add_argument("--metadata-path", type=str, default=defaults["metadata_path"])
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    parser.add_argument(
        "--split",
        type=str,
        default=defaults["split"],
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--limit-qa", type=int, default=defaults["limit_qa"])
    parser.add_argument("--model-name", type=str, default=defaults["model_name"])
    parser.add_argument("--api-key", type=str, default=defaults["api_key"])
    parser.add_argument("--api-base-url", type=str, default=defaults["api_base_url"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--timeout-seconds", type=int, default=defaults["timeout_seconds"])
    parser.add_argument("--min-ocr-score", type=float, default=0.30)
    parser.add_argument("--max-ocr-tokens", type=int, default=80)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument(
        "--reuse-ocr-cache",
        dest="reuse_ocr_cache",
        action="store_true",
        default=defaults["reuse_ocr_cache"],
    )
    parser.add_argument(
        "--no-reuse-ocr-cache",
        dest="reuse_ocr_cache",
        action="store_false",
    )
    parser.add_argument(
        "--reuse-response-cache",
        dest="reuse_response_cache",
        action="store_true",
        default=defaults["reuse_response_cache"],
    )
    parser.add_argument(
        "--no-reuse-response-cache",
        dest="reuse_response_cache",
        action="store_false",
    )
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


def build_app_config(args: argparse.Namespace, dataset_dir: Path, metadata_path: Path, output_dir: Path) -> AppConfig:
    """Build the AppConfig consumed by the existing OpenAI-compatible client."""

    model_name = str(args.model_name).strip()
    if not model_name:
        raise ValueError("model_name must not be empty. Pass --model-name or set OPENAI_MODEL_USE.")

    return AppConfig(
        dataset_dir=dataset_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        split=args.split,
        limit_qa=args.limit_qa,
        reuse_cache=bool(args.reuse_response_cache),
        backend="openai_compatible",
        model_name=model_name,
        api_key=args.api_key.strip() if isinstance(args.api_key, str) and args.api_key.strip() else None,
        api_base_url=(
            args.api_base_url.strip() if isinstance(args.api_base_url, str) and args.api_base_url.strip() else None
        ),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout_seconds=int(args.timeout_seconds),
        prompt_version="chartqa_ocr_augmented_v1",
        hf_device_map="auto",
        hf_dtype="auto",
        hf_attn_implementation=None,
        hf_trust_remote_code=False,
    )


def ensure_inputs_exist(examples: list[QAExample], metadata_path: Path) -> None:
    """Fail early if metadata or any selected chart image is missing."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    for example in examples:
        if not example.abs_image_path.exists():
            raise FileNotFoundError(f"Missing dataset image: {example.abs_image_path}")


def safe_config_payload(config: AppConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Create a config JSON payload without persisting API secrets."""

    payload = config.to_json_dict()
    payload["api_key"] = "<set>" if config.api_key else None
    payload.update(
        {
            "min_ocr_score": args.min_ocr_score,
            "max_ocr_tokens": args.max_ocr_tokens,
            "reuse_ocr_cache": bool(args.reuse_ocr_cache),
            "reuse_response_cache": bool(args.reuse_response_cache),
            "save_vis": bool(args.save_vis),
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_limit_type": args.text_det_limit_type,
            "use_doc_orientation_classify": bool(args.use_doc_orientation_classify),
            "use_doc_unwarping": bool(args.use_doc_unwarping),
            "use_textline_orientation": bool(args.use_textline_orientation),
        }
    )
    return payload


def normalized_bbox(token: OCRToken, case: ImageCase) -> tuple[float, float, float, float]:
    """Convert token bbox to normalized image coordinates."""

    width = max(float(case.image_width), 1.0)
    height = max(float(case.image_height), 1.0)
    x0, y0, x1, y1 = token.bbox
    return (
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    )


def token_sort_key(token: OCRToken) -> tuple[float, float]:
    """Sort OCR tokens approximately top-to-bottom and left-to-right."""

    line_height = max(token.h, 1.0)
    return (round(token.cy / line_height), token.cx)


def build_ocr_context(case: ImageCase, tokens: list[OCRToken], max_tokens: int) -> str:
    """Format OCR tokens into a compact prompt context."""

    sorted_tokens = sorted(tokens, key=token_sort_key)
    limited_tokens = sorted_tokens[: max(0, max_tokens)]

    lines = [
        f"Chart type metadata: {case.chart_type}",
        f"OCR token count after filtering: {len(tokens)}",
    ]
    if not limited_tokens:
        lines.append("No OCR text was detected.")
        return "\n".join(lines)

    lines.append("OCR tokens are sorted approximately top-to-bottom, left-to-right.")
    lines.append("Each token includes text, confidence score, and normalized bbox=(x0,y0,x1,y1).")
    for index, token in enumerate(limited_tokens, start=1):
        x0, y0, x1, y1 = normalized_bbox(token, case)
        score = "NA" if token.score is None else f"{token.score:.2f}"
        lines.append(
            f'{index}. text="{token.norm_text}" score={score} '
            f"bbox=({x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f})"
        )
    omitted_count = len(sorted_tokens) - len(limited_tokens)
    if omitted_count > 0:
        lines.append(f"{omitted_count} additional OCR tokens were omitted from the prompt.")
    return "\n".join(lines)


def build_system_prompt() -> str:
    """Build the system prompt for OCR-augmented chart QA."""

    return (
        "You are a precise chart question answering model. "
        "You receive the chart image plus OCR text extracted from the same image. "
        "Use the OCR text as auxiliary evidence because it may be incomplete or wrong; "
        "when OCR conflicts with the image, rely on the image. "
        "Return valid JSON with exactly one key named answer. "
        "Do not include markdown, code fences, or extra keys. "
        "For numeric answers, return only the number string without units. "
        "For category answers, return labels like Cat-3. "
        "For trend answers, return exactly one of increasing, decreasing, or fluctuating."
    )


def build_user_prompt(example: QAExample, ocr_context: str) -> str:
    """Build one user prompt containing the question and OCR evidence."""

    format_hint = "Return JSON only."
    if example.answer_type == "number":
        format_hint = 'Return JSON only. Example: {"answer": "47.5"}'
    elif example.answer_type == "category":
        format_hint = 'Return JSON only. Example: {"answer": "Cat-2"}'
    elif example.task_type == "line_trend":
        format_hint = 'Return JSON only. Example: {"answer": "increasing"}'

    return (
        f"Question: {example.question}\n"
        f"Expected answer type: {example.answer_type}\n\n"
        "OCR context:\n"
        f"{ocr_context}\n\n"
        f"{format_hint}"
    )


def build_request(example: QAExample, config: AppConfig, ocr_context: str) -> LLMRequest:
    """Build one multimodal request with image and OCR context."""

    return LLMRequest(
        system_prompt=build_system_prompt(),
        user_prompt=build_user_prompt(example, ocr_context),
        image_paths=[example.abs_image_path],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        metadata={
            "sample_id": example.sample_id,
            "task_type": example.task_type,
            "chart_type": example.chart_type,
            "ocr_context": ocr_context,
            **variant_metadata_from_obj(example),
        },
    )


def build_raw_response_row(
    example: QAExample,
    request: LLMRequest,
    response: dict[str, Any],
    backend: str,
    model_name: str,
    ocr_record: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable model response record."""

    return {
        "sample_id": example.sample_id,
        "image_path": example.image_path,
        "chart_type": example.chart_type,
        "split": example.split,
        "style_id": example.style_id,
        "base_id": example.base_id,
        "qa_id": example.qa_id,
        **variant_metadata_from_obj(example),
        "task_type": example.task_type,
        "answer_type": example.answer_type,
        "question": example.question,
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "ocr_status": ocr_record["ocr_status"],
        "ocr_token_count": ocr_record["token_count"],
        "ocr_context": request.metadata.get("ocr_context"),
        "backend": backend,
        "model_name": model_name,
        **response,
    }


def run_ocr_phase(
    args: argparse.Namespace,
    output_dir: Path,
    cases: list[ImageCase],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Run or load OCR for selected image cases and return prompt contexts."""

    raw_cache_path = output_dir / "ocr_raw.jsonl"
    raw_cache = load_raw_cache(raw_cache_path) if args.reuse_ocr_cache else {}
    vis_dir = output_dir / "ocr_vis" if args.save_vis else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict[str, Any]] = []
    token_records: list[dict[str, Any]] = []
    ocr_contexts: dict[str, str] = {}
    ocr_records_by_image: dict[str, dict[str, Any]] = {}

    if not cases:
        write_jsonl(raw_cache_path, raw_records)
        write_jsonl(output_dir / "ocr_tokens.jsonl", token_records)
        return ocr_contexts, ocr_records_by_image

    engine = build_paddle_ocr_engine(args)
    try:
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
            token_record = {
                "image_path": case.image_path,
                "chart_type": case.chart_type,
                "split": case.split,
                "style_id": case.style_id,
                **variant_metadata_from_obj(case),
                "ocr_status": ocr_status,
                "token_count": len(tokens),
                "tokens": [asdict(token) for token in tokens],
            }
            token_records.append(token_record)
            ocr_contexts[case.image_path] = build_ocr_context(case, tokens, args.max_ocr_tokens)
            ocr_records_by_image[case.image_path] = token_record
    finally:
        engine.close()

    write_jsonl(raw_cache_path, raw_records)
    write_jsonl(output_dir / "ocr_tokens.jsonl", token_records)
    return ocr_contexts, ocr_records_by_image


def write_empty_outputs(output_dir: Path, raw_cache_path: Path, cache_rows: dict[str, dict[str, Any]]) -> None:
    """Write empty evaluation outputs when no QA rows are selected."""

    predictions: list[dict[str, Any]] = []
    summary = build_summary(predictions)
    write_jsonl(raw_cache_path, sorted(cache_rows.values(), key=lambda row: str(row["sample_id"])))
    write_jsonl(output_dir / "qa_predictions.jsonl", predictions)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "base_variant_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(build_base_variant_comparison(predictions), handle, ensure_ascii=False, indent=2)
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_report(summary, predictions))


def checkpoint_predictions(
    output_dir: Path,
    raw_response_path: Path,
    cache_rows: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    """Persist partial raw responses and predictions for long API runs."""

    write_jsonl(raw_response_path, sorted(cache_rows.values(), key=lambda row: str(row["sample_id"])))
    write_jsonl(output_dir / "qa_predictions.jsonl", predictions)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full OCR-augmented GPT benchmark pipeline."""

    dataset_dir = resolve_repo_path(args.dataset_dir)
    if dataset_dir is None:
        raise ValueError("dataset_dir must not be empty.")
    metadata_path = resolve_repo_path(args.metadata_path) if args.metadata_path else dataset_dir / "metadata.jsonl"
    output_dir = resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must not be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_app_config(args, dataset_dir, metadata_path, output_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(safe_config_payload(config, args), handle, ensure_ascii=False, indent=2)

    metadata_rows = load_metadata(metadata_path)
    examples = build_qa_examples(metadata_rows, dataset_dir, args.split, args.limit_qa)
    ensure_inputs_exist(examples, metadata_path)

    selected_images = {example.image_path for example in examples}
    case_rows = [row for row in metadata_rows if str(row.get("image_path")) in selected_images]
    cases = build_image_cases(case_rows, dataset_dir, args.split, None)
    ocr_contexts, ocr_records_by_image = run_ocr_phase(args, output_dir, cases)

    raw_response_path = output_dir / "raw_responses.jsonl"
    raw_cache = load_jsonl_cache(raw_response_path, "sample_id") if args.reuse_response_cache else {}
    cache_rows = dict(raw_cache)

    if not examples:
        write_empty_outputs(output_dir, raw_response_path, cache_rows)
        print("[Done] QA records: 0")
        print(f"[Done] Output dir: {output_dir}")
        print(f"[Done] Summary: {output_dir / 'metrics_summary.json'}")
        print(f"[Done] Report: {output_dir / 'report.md'}")
        return

    predictions: list[dict[str, Any]] = []
    with create_llm_client(config) as client:
        for example in examples:
            raw_row = raw_cache.get(example.sample_id)
            if raw_row is None:
                ocr_context = ocr_contexts.get(example.image_path, "No OCR context is available.")
                ocr_record = ocr_records_by_image.get(
                    example.image_path,
                    {"ocr_status": "missing", "token_count": 0},
                )
                request = build_request(example, config, ocr_context)
                try:
                    response = client.generate(request)
                    raw_row = build_raw_response_row(
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
                        ocr_record=ocr_record,
                    )
                except Exception as exc:
                    raw_row = build_raw_response_row(
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
                        ocr_record=ocr_record,
                    )
            cache_rows[example.sample_id] = raw_row
            parsed_answer = parse_model_response(str(raw_row.get("raw_response_text", "")))
            predictions.append(build_prediction_record(example, raw_row, parsed_answer))
            if len(predictions) % 25 == 0 or len(predictions) == len(examples):
                checkpoint_predictions(output_dir, raw_response_path, cache_rows, predictions)
                print(f"[Progress] QA records: {len(predictions)}/{len(examples)}", flush=True)

    summary = build_summary(predictions)
    checkpoint_predictions(output_dir, raw_response_path, cache_rows, predictions)

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "base_variant_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(build_base_variant_comparison(predictions), handle, ensure_ascii=False, indent=2)
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_report(summary, predictions))

    print(f"[Done] OCR images: {len(cases)}")
    print(f"[Done] QA records: {len(predictions)}")
    print(f"[Done] Output dir: {output_dir}")
    print(f"[Done] Summary: {output_dir / 'metrics_summary.json'}")
    print(f"[Done] Report: {output_dir / 'report.md'}")


def main() -> None:
    """CLI entry point."""

    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
