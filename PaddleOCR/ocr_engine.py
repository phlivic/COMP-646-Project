"""PaddleOCR inference and token normalization helpers.

This module isolates the dependency on PaddleOCR itself and converts raw OCR
payloads into a stable token structure that the rest of the pipeline can use.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .ocr_types import ImageCase, OCRToken, normalize_text


def to_jsonable(obj: Any) -> Any:
    """Recursively convert array-like OCR outputs into plain Python objects."""

    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


def polygon_to_bbox(poly: Sequence[Any]) -> tuple[float, float, float, float]:
    """Convert an OCR polygon into an axis-aligned bounding box."""

    points: list[tuple[float, float]] = []
    for point in poly or []:
        if isinstance(point, Sequence) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def normalize_ocr_result(res_json: dict[str, Any]) -> list[OCRToken]:
    """Convert a PaddleOCR JSON payload into normalized OCRToken objects."""

    payload = res_json.get("res", res_json)
    texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores", [])
    polys = payload.get("rec_polys", payload.get("dt_polys", []))

    tokens: list[OCRToken] = []
    for index, text in enumerate(texts):
        norm_text = normalize_text(str(text))
        if not norm_text:
            continue
        score = float(scores[index]) if index < len(scores) else None
        poly = to_jsonable(polys[index]) if index < len(polys) else []
        x0, y0, x1, y1 = polygon_to_bbox(poly)
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        tokens.append(
            OCRToken(
                text=str(text),
                norm_text=norm_text,
                score=score,
                poly=poly,
                bbox=(x0, y0, x1, y1),
                cx=x0 + width / 2.0,
                cy=y0 + height / 2.0,
                w=width,
                h=height,
            )
        )
    return tokens


def filter_tokens(tokens: Sequence[OCRToken], min_score: float) -> list[OCRToken]:
    """Drop empty or low-confidence OCR tokens before chart parsing."""

    filtered: list[OCRToken] = []
    for token in tokens:
        if token.score is not None and token.score < min_score:
            continue
        if not token.norm_text:
            continue
        filtered.append(token)
    return filtered


def build_ocr_engine(args: argparse.Namespace):
    """Create the PaddleOCR engine with the CLI-selected options."""

    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
    )


def run_ocr_on_image(
    engine,
    case: ImageCase,
    args: argparse.Namespace,
    vis_dir: Path | None,
) -> dict[str, Any]:
    """Run OCR on one image and optionally save the PaddleOCR visualization."""

    result_list = engine.predict(
        str(case.abs_image_path),
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
        text_det_limit_side_len=args.text_det_limit_side_len,
        text_det_limit_type=args.text_det_limit_type,
    )

    if not result_list:
        return {
            "image_path": case.image_path,
            "ocr_status": "empty",
            "ocr_vis_path": None,
            "res_json": {},
        }

    result = result_list[0]
    res_json = to_jsonable(result.json)
    ocr_vis_path: str | None = None
    if vis_dir is not None:
        result.save_to_img(str(vis_dir))
        ocr_vis_path = f"ocr_vis/{case.abs_image_path.stem}_ocr_res_img{case.abs_image_path.suffix}"

    return {
        "image_path": case.image_path,
        "ocr_status": "text_found",
        "ocr_vis_path": ocr_vis_path,
        "res_json": res_json,
    }
