from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CHART_TYPES = ("bar", "line", "pie")


@dataclass
class PrototypeSample:
    sample_id: str
    chart_type: str
    categories: List[str]
    values: List[float]
    title: str
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate small chart prototypes and run PaddleOCR for quick inspection."
    )
    parser.add_argument("--output-dir", type=str, default="PaddleOCR/prototype_out")
    parser.add_argument("--num-per-type", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--text-det-limit-side-len", type=int, default=960)
    parser.add_argument("--text-det-limit-type", type=str, default="max", choices=["max", "min"])
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-textline-orientation", action="store_true")
    return parser.parse_args()


def build_samples(num_per_type: int, seed: int) -> List[PrototypeSample]:
    rng = random.Random(seed)
    samples: List[PrototypeSample] = []
    for chart_type in CHART_TYPES:
        for i in range(num_per_type):
            categories = [f"C{j + 1}" for j in range(5)]
            values = [float(rng.randint(12, 96)) for _ in categories]
            sample_id = f"{chart_type}_{i:02d}"
            title = f"{chart_type.upper()} SAMPLE {i}"
            image_path = f"images/{sample_id}.png"
            samples.append(
                PrototypeSample(
                    sample_id=sample_id,
                    chart_type=chart_type,
                    categories=categories,
                    values=values,
                    title=title,
                    image_path=image_path,
                )
            )
    return samples


def _pie_percent_texts(values: List[float]) -> List[str]:
    total = sum(values) if values else 1.0
    return [f"{(v / total) * 100:.1f}%" for v in values]


def expected_texts(sample: PrototypeSample) -> List[str]:
    texts = [sample.title]
    if sample.chart_type in ("bar", "line"):
        texts.extend(["Category", "Value"])
        texts.extend(sample.categories)
        texts.extend([f"{int(v)}" for v in sample.values])
    elif sample.chart_type == "pie":
        texts.extend(sample.categories)
        texts.extend(_pie_percent_texts(sample.values))
    return texts


def render_sample(sample: PrototypeSample, abs_image_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    abs_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    if sample.chart_type == "bar":
        ax.bar(sample.categories, sample.values, color="#4E79A7")
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(sample.values) * 1.18)
        for i, v in enumerate(sample.values):
            ax.text(i, v + 1.5, f"{int(v)}", ha="center", va="bottom", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    elif sample.chart_type == "line":
        ax.plot(sample.categories, sample.values, marker="o", color="#E15759", linewidth=2.2)
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(sample.values) * 1.18)
        for i, v in enumerate(sample.values):
            ax.text(i, v + 1.5, f"{int(v)}", fontsize=11, ha="center", va="bottom")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    elif sample.chart_type == "pie":
        ax.pie(
            sample.values,
            labels=sample.categories,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart type: {sample.chart_type}")

    ax.set_title(sample.title)
    fig.tight_layout()
    fig.savefig(abs_image_path, dpi=dpi)
    plt.close(fig)


def _to_jsonable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def parse_ocr_result_from_res_json(res_json: Dict) -> Tuple[List[Dict], List[str]]:
    payload = res_json.get("res", {})
    texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores", [])
    polys = payload.get("rec_polys", payload.get("dt_polys", []))

    items: List[Dict] = []
    for i, text in enumerate(texts):
        score = float(scores[i]) if i < len(scores) else None
        box = _to_jsonable(polys[i]) if i < len(polys) else None
        items.append({"text": str(text), "score": score, "box": box})
    return items, [str(t) for t in texts]


def write_report(records: List[Dict], output_dir: Path) -> None:
    jsonl_path = output_dir / "ocr_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_path = output_dir / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Prototype OCR Report\n\n")
        f.write("| sample_id | chart_type | image_path | ocr_vis_path | expected_texts | ocr_texts |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in records:
            expected_preview = ", ".join(row["expected_texts"][:12])
            ocr_preview = ", ".join(row["ocr_texts"][:10])
            f.write(
                f"| {row['sample_id']} | {row['chart_type']} | {row['image_path']} | {row['ocr_vis_path']} | {expected_preview} | {ocr_preview} |\n"
            )


def run_pipeline(args: argparse.Namespace) -> None:
    from paddleocr import PaddleOCR

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "ocr_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args.num_per_type, args.seed)
    for sample in samples:
        render_sample(sample, output_dir / sample.image_path, args.dpi)

    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
    )

    records: List[Dict] = []
    for sample in samples:
        abs_image_path = output_dir / sample.image_path
        result_list = ocr.predict(
            str(abs_image_path),
            use_doc_orientation_classify=args.use_doc_orientation_classify,
            use_doc_unwarping=args.use_doc_unwarping,
            use_textline_orientation=args.use_textline_orientation,
            text_det_limit_side_len=args.text_det_limit_side_len,
            text_det_limit_type=args.text_det_limit_type,
        )
        if not result_list:
            raise RuntimeError(f"No OCR result returned for: {abs_image_path}")

        res = result_list[0]
        res_json = _to_jsonable(res.json)
        items, texts = parse_ocr_result_from_res_json(res_json)

        # Official visualization output path style: <stem>_ocr_res_img.<suffix>
        res.save_to_img(str(vis_dir))
        ocr_vis_path = f"ocr_vis/{abs_image_path.stem}_ocr_res_img{abs_image_path.suffix}"

        record = {
            "sample_id": sample.sample_id,
            "chart_type": sample.chart_type,
            "image_path": sample.image_path,
            "ocr_vis_path": ocr_vis_path,
            "sample": asdict(sample),
            "expected_texts": expected_texts(sample),
            "ocr_settings": {
                "use_doc_orientation_classify": args.use_doc_orientation_classify,
                "use_doc_unwarping": args.use_doc_unwarping,
                "use_textline_orientation": args.use_textline_orientation,
                "text_det_limit_side_len": args.text_det_limit_side_len,
                "text_det_limit_type": args.text_det_limit_type,
            },
            "ocr_res_json": res_json,
            "ocr_items": items,
            "ocr_texts": texts,
        }
        records.append(record)

    ocr.close()
    write_report(records, output_dir)

    print(f"[Done] Generated {len(samples)} images in: {output_dir / 'images'}")
    print(f"[Done] OCR visualization images: {output_dir / 'ocr_vis'}")
    print(f"[Done] OCR JSONL report: {output_dir / 'ocr_results.jsonl'}")
    print(f"[Done] OCR Markdown report: {output_dir / 'report.md'}")
    print("")
    print("==== OCR Preview ====")
    for row in records:
        print(f"- {row['sample_id']} ({row['chart_type']})")
        print(f"  image: {row['image_path']}")
        print(f"  expected: {row['expected_texts'][:10]}")
        print(f"  ocr_texts: {row['ocr_texts'][:12]}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
