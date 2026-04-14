"""Chart dataset generator for OCR/rule/MM-LLM benchmarking.

This module defines a structured dataset with explicit task rules:

1) Bar charts:
   - Number of bars: 2, 3, 4, or 5.
   - Questions: max value, min value, average value.

2) Line charts:
   - Number of nodes: 3, 4, 5, or 6.
   - Questions: max value, min value, overall trend.
   - Trend labels: increasing, decreasing, fluctuating.

3) Pie charts:
   - Number of slices: 2, 3, or 10.
   - Questions: largest share, second largest share,
                smallest share, second smallest share.

The generator also supports visual style perturbations and exports JSONL
metadata for downstream evaluation scripts.

Default numeric value generation:
- Values are randomly generated from the integer range [12, 96].
- Sampling is done per chart instance (with seeded randomness for reproducibility).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------- Task Definitions -----------------------------

CHART_TYPES: Tuple[str, ...] = ("bar", "line", "pie")
BAR_POINT_COUNTS: Tuple[int, ...] = (2, 3, 4, 5)
LINE_POINT_COUNTS: Tuple[int, ...] = (3, 4, 5, 6)
PIE_POINT_COUNTS: Tuple[int, ...] = (2, 3, 10)
LINE_TRENDS: Tuple[str, ...] = ("increasing", "decreasing", "fluctuating")


# ----------------------------- Visual Definitions -----------------------------

FONT_CANDIDATES: Tuple[str, ...] = (
    "DejaVu Sans",
    "Liberation Sans",
    "Times New Roman",
    "Arial",
)

COLOR_THEMES: Dict[str, List[str]] = {
    "classic": ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"],
    "warm": ["#8C564B", "#E07A5F", "#F2CC8F", "#DDA15E", "#BC6C25", "#C97F60"],
    "cool": ["#277DA1", "#4D908E", "#43AA8B", "#577590", "#90BE6D", "#A8DADC"],
    "high_contrast": ["#000000", "#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3"],
}

QUALITY_PRESETS: Dict[str, Dict[str, int]] = {
    "high": {"dpi": 220, "jpeg_quality": 95},
    "medium": {"dpi": 140, "jpeg_quality": 80},
    "low": {"dpi": 96, "jpeg_quality": 55},
}

REFERENCE_STYLE = {
    "font_family": "DejaVu Sans",
    "color_theme": "classic",
    "label_rotation": 0,
    "quality": "high",
}


# ----------------------------- Data Structures -----------------------------

@dataclass(frozen=True)
class DatasetConfig:
    output_dir: str = "datasets/out"
    num_per_type: int = 100
    style_variants_per_base: int = 8
    seed: int = 42
    image_format: str = "jpg"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    metadata_filename: str = "metadata.jsonl"
    summary_filename: str = "summary.json"
    skip_reference_style: bool = True


@dataclass(frozen=True)
class VisualStyle:
    font_family: str
    color_theme: str
    label_rotation: int
    quality: str

    @property
    def render_quality(self) -> Dict[str, int]:
        return QUALITY_PRESETS[self.quality]


@dataclass(frozen=True)
class QAItem:
    """One question-answer item for a chart."""

    qa_id: str
    task_type: str
    question: str
    answer: str
    answer_type: str  # "number", "category", or "label"


@dataclass(frozen=True)
class ChartSpec:
    """Semantic chart content shared across style variants."""

    base_id: str
    chart_type: str
    categories: List[str]
    values: List[float]
    line_trend: Optional[str] = None


@dataclass(frozen=True)
class BaseChartSample:
    """A chart spec plus all QA tasks defined for that chart."""

    chart: ChartSpec
    qa_items: List[QAItem]


# ----------------------------- Numeric Helpers -----------------------------

def _format_number(x: float) -> str:
    """Format numbers compactly while keeping stable textual answers."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _make_categories(n: int) -> List[str]:
    return [f"Cat-{i + 1}" for i in range(n)]


def _unique_int_values(rng: random.Random, n: int, low: int = 10, high: int = 99) -> List[float]:
    """Generate unique random integer values to avoid ranking ties.

    Notes:
    - For this dataset, chart builders use the default range [12, 96].
    - Values are randomly sampled (without replacement) and cast to float.
    """
    if (high - low + 1) < n:
        raise ValueError("Value range too small for unique sampling.")
    return [float(v) for v in rng.sample(range(low, high + 1), n)]


def classify_line_trend(values: Sequence[float]) -> str:
    """Classify line trend into increasing, decreasing, or fluctuating."""
    inc = all(values[i] < values[i + 1] for i in range(len(values) - 1))
    dec = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    if inc:
        return "increasing"
    if dec:
        return "decreasing"
    return "fluctuating"


def _generate_line_values(rng: random.Random, n: int, trend: str) -> List[float]:
    base = _unique_int_values(rng, n, low=12, high=96)
    if trend == "increasing":
        return sorted(base)
    if trend == "decreasing":
        return sorted(base, reverse=True)

    # Fluctuating: not monotonic.
    for _ in range(80):
        candidate = _unique_int_values(rng, n, low=12, high=96)
        if classify_line_trend(candidate) == "fluctuating":
            return candidate

    # Deterministic fallback if random attempts fail.
    ordered = sorted(base)
    if n >= 3:
        return [ordered[0], ordered[-1], *ordered[1:-1]]
    return ordered


# ----------------------------- Task Construction -----------------------------

def _build_bar_sample(rng: random.Random, idx: int) -> BaseChartSample:
    n = rng.choice(BAR_POINT_COUNTS)
    categories = _make_categories(n)
    values = _unique_int_values(rng, n, low=12, high=96)

    chart = ChartSpec(
        base_id=f"bar_{idx:05d}",
        chart_type="bar",
        categories=categories,
        values=values,
        line_trend=None,
    )

    qa_items = [
        QAItem(
            qa_id="bar_max_value",
            task_type="bar_max_value",
            question="What is the maximum value in this bar chart?",
            answer=_format_number(max(values)),
            answer_type="number",
        ),
        QAItem(
            qa_id="bar_min_value",
            task_type="bar_min_value",
            question="What is the minimum value in this bar chart?",
            answer=_format_number(min(values)),
            answer_type="number",
        ),
        QAItem(
            qa_id="bar_avg_value",
            task_type="bar_avg_value",
            question="What is the average value of all bars in this chart?",
            answer=_format_number(sum(values) / len(values)),
            answer_type="number",
        ),
    ]
    return BaseChartSample(chart=chart, qa_items=qa_items)


def _build_line_sample(rng: random.Random, idx: int) -> BaseChartSample:
    n = rng.choice(LINE_POINT_COUNTS)
    categories = _make_categories(n)
    # Cycle trend labels to guarantee coverage of all three trend classes.
    target_trend = LINE_TRENDS[idx % len(LINE_TRENDS)]
    values = _generate_line_values(rng, n, target_trend)
    trend = classify_line_trend(values)

    chart = ChartSpec(
        base_id=f"line_{idx:05d}",
        chart_type="line",
        categories=categories,
        values=values,
        line_trend=trend,
    )

    qa_items = [
        QAItem(
            qa_id="line_max_value",
            task_type="line_max_value",
            question="What is the maximum value in this line chart?",
            answer=_format_number(max(values)),
            answer_type="number",
        ),
        QAItem(
            qa_id="line_min_value",
            task_type="line_min_value",
            question="What is the minimum value in this line chart?",
            answer=_format_number(min(values)),
            answer_type="number",
        ),
        QAItem(
            qa_id="line_trend",
            task_type="line_trend",
            question="What is the overall trend of the line chart? (increasing, decreasing, or fluctuating)",
            answer=trend,
            answer_type="label",
        ),
    ]
    return BaseChartSample(chart=chart, qa_items=qa_items)


def _build_pie_sample(rng: random.Random, idx: int) -> BaseChartSample:
    n = rng.choice(PIE_POINT_COUNTS)
    categories = _make_categories(n)
    values = _unique_int_values(rng, n, low=12, high=96)

    chart = ChartSpec(
        base_id=f"pie_{idx:05d}",
        chart_type="pie",
        categories=categories,
        values=values,
        line_trend=None,
    )

    desc = sorted(range(n), key=lambda i: values[i], reverse=True)
    asc = sorted(range(n), key=lambda i: values[i])

    largest = categories[desc[0]]
    second_largest = categories[desc[1]]
    smallest = categories[asc[0]]
    second_smallest = categories[asc[1]]

    qa_items = [
        QAItem(
            qa_id="pie_largest_share",
            task_type="pie_largest_share",
            question="Which category has the largest share in this pie chart?",
            answer=largest,
            answer_type="category",
        ),
        QAItem(
            qa_id="pie_second_largest_share",
            task_type="pie_second_largest_share",
            question="Which category has the second largest share in this pie chart?",
            answer=second_largest,
            answer_type="category",
        ),
        QAItem(
            qa_id="pie_smallest_share",
            task_type="pie_smallest_share",
            question="Which category has the smallest share in this pie chart?",
            answer=smallest,
            answer_type="category",
        ),
        QAItem(
            qa_id="pie_second_smallest_share",
            task_type="pie_second_smallest_share",
            question="Which category has the second smallest share in this pie chart?",
            answer=second_smallest,
            answer_type="category",
        ),
    ]
    return BaseChartSample(chart=chart, qa_items=qa_items)


def generate_base_samples(config: DatasetConfig) -> List[BaseChartSample]:
    """Generate semantic chart samples and chart-specific QA sets."""
    rng = random.Random(config.seed)
    samples: List[BaseChartSample] = []
    for chart_type in CHART_TYPES:
        for idx in range(config.num_per_type):
            if chart_type == "bar":
                samples.append(_build_bar_sample(rng, idx))
            elif chart_type == "line":
                samples.append(_build_line_sample(rng, idx))
            elif chart_type == "pie":
                samples.append(_build_pie_sample(rng, idx))
            else:
                raise ValueError(f"Unsupported chart_type: {chart_type}")
    return samples


# ----------------------------- Style Sampling -----------------------------

def _all_style_candidates() -> List[VisualStyle]:
    candidates: List[VisualStyle] = []
    for font, theme, rot, quality in product(
        FONT_CANDIDATES, COLOR_THEMES.keys(), (0, 30, 60, 90), QUALITY_PRESETS.keys()
    ):
        candidates.append(
            VisualStyle(
                font_family=font,
                color_theme=theme,
                label_rotation=rot,
                quality=quality,
            )
        )
    return candidates


def choose_styles_for_base(
    base_id: str,
    k: int,
    global_seed: int,
    include_reference: bool = True,
) -> List[Tuple[str, VisualStyle]]:
    """Deterministic style sampling per base sample."""
    token = f"{base_id}|{global_seed}".encode("utf-8")
    stable_seed = int(hashlib.sha256(token).hexdigest()[:8], 16)
    local_rng = random.Random(stable_seed)
    candidates = _all_style_candidates()
    local_rng.shuffle(candidates)

    variants: List[Tuple[str, VisualStyle]] = []
    if include_reference:
        reference = VisualStyle(**REFERENCE_STYLE)
        variants.append(("reference", reference))

    num_random = max(0, k - len(variants))
    for i, style in enumerate(candidates[:num_random]):
        variants.append((f"variant_{i:02d}", style))
    return variants


# ----------------------------- Rendering -----------------------------

def _safe_font(font_name: str, plt_mod) -> str:
    available = set(plt_mod.rcParams.get("font.sans-serif", []))
    if font_name in available or font_name == "DejaVu Sans":
        return font_name
    return "DejaVu Sans"


def render_chart(chart: ChartSpec, style: VisualStyle, output_path: Path, image_format: str) -> None:
    """Render one chart image from semantic spec + visual style."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = _safe_font(style.font_family, plt)
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    palette = COLOR_THEMES[style.color_theme]
    colors = [palette[i % len(palette)] for i in range(len(chart.values))]

    if chart.chart_type == "bar":
        ax.bar(chart.categories, chart.values, color=colors)
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        ax.set_title("Bar Chart")
        ax.set_ylim(0, max(chart.values) * 1.12)
        ax.tick_params(axis="x", rotation=style.label_rotation)
        for i, v in enumerate(chart.values):
            ax.text(i, v + 0.8, _format_number(v), fontsize=8, ha="center", va="bottom")
    elif chart.chart_type == "line":
        ax.plot(chart.categories, chart.values, marker="o", color=colors[0], linewidth=2.0)
        ax.set_xlabel("Node")
        ax.set_ylabel("Value")
        ax.set_title("Line Chart")
        ax.tick_params(axis="x", rotation=style.label_rotation)
        for i, v in enumerate(chart.values):
            ax.text(i, v, _format_number(v), fontsize=8, ha="center", va="bottom")
    elif chart.chart_type == "pie":
        ax.pie(
            chart.values,
            labels=chart.categories,
            colors=colors,
            autopct="%1.1f%%",
            startangle=110,
            textprops={"rotation": style.label_rotation},
        )
        ax.set_title("Pie Chart")
    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart_type: {chart.chart_type}")

    fig.tight_layout()
    quality = style.render_quality
    save_kwargs = {"dpi": quality["dpi"]}
    if image_format.lower() in {"jpg", "jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": quality["jpeg_quality"]}
    fig.savefig(output_path, format=image_format, **save_kwargs)
    plt.close(fig)


# ----------------------------- Metadata Export -----------------------------

def _allocate_counts(n: int, cfg: DatasetConfig) -> Dict[str, int]:
    """Allocate train/val/test counts using largest-remainder rounding."""
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}

    exact = {
        "train": n * cfg.train_ratio,
        "val": n * cfg.val_ratio,
        "test": n * cfg.test_ratio,
    }
    counts = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(counts.values())

    if remainder > 0:
        order = sorted(exact.keys(), key=lambda k: (exact[k] - counts[k], k), reverse=True)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

    return counts


def _build_split_plan(
    base_samples: Sequence[BaseChartSample],
    style_map: Dict[str, List[Tuple[str, VisualStyle]]],
    cfg: DatasetConfig,
) -> Dict[Tuple[str, str], str]:
    """Assign split at base granularity with stratification by chart type.

    All style variants for the same base_id are forced into the same split.
    """
    strata: Dict[str, List[str]] = {}
    for base_sample in base_samples:
        base_id = base_sample.chart.base_id
        chart_type = base_sample.chart.chart_type
        strata.setdefault(chart_type, []).append(base_id)

    split_plan: Dict[Tuple[str, str], str] = {}
    for chart_type, base_ids in sorted(strata.items()):
        token = f"{chart_type}|{cfg.seed}".encode("utf-8")
        stable_seed = int(hashlib.sha256(token).hexdigest()[:8], 16)
        local_rng = random.Random(stable_seed)
        ordered = list(base_ids)
        local_rng.shuffle(ordered)

        counts = _allocate_counts(len(ordered), cfg)
        train_end = counts["train"]
        val_end = train_end + counts["val"]

        base_split: Dict[str, str] = {}
        for base_id in ordered[:train_end]:
            base_split[base_id] = "train"
        for base_id in ordered[train_end:val_end]:
            base_split[base_id] = "val"
        for base_id in ordered[val_end:]:
            base_split[base_id] = "test"

        for base_id in ordered:
            split = base_split[base_id]
            for style_id, _ in style_map[base_id]:
                split_plan[(base_id, style_id)] = split

    return split_plan


def image_generation(config: DatasetConfig) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """Generate chart images + QA records.

    Returns:
        images: image paths (one per QA record)
        questions: question text list
        gts: answer list
        params: style factors list
    """
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_samples = generate_base_samples(config)
    base_samples = sorted(base_samples, key=lambda s: s.chart.base_id)

    include_reference = not config.skip_reference_style
    style_map: Dict[str, List[Tuple[str, VisualStyle]]] = {
        sample.chart.base_id: choose_styles_for_base(
            sample.chart.base_id,
            config.style_variants_per_base,
            config.seed,
            include_reference=include_reference,
        )
        for sample in base_samples
    }
    split_plan = _build_split_plan(base_samples, style_map, config)

    metadata: List[Dict] = []
    images: List[str] = []
    questions: List[str] = []
    gts: List[str] = []
    params: List[Dict] = []

    for base_sample in base_samples:
        style_variants = style_map[base_sample.chart.base_id]

        # Render image once for each style variant, then attach all QA rows.
        for style_id, style in style_variants:
            split = split_plan[(base_sample.chart.base_id, style_id)]
            filename = f"{base_sample.chart.base_id}__{style_id}.{config.image_format}"
            rel_path = f"{split}/{filename}"
            abs_path = out_dir / rel_path
            render_chart(base_sample.chart, style, abs_path, config.image_format)

            for qa in base_sample.qa_items:
                sample_id = f"{base_sample.chart.base_id}__{style_id}__{qa.qa_id}"
                record = {
                    "sample_id": sample_id,
                    "base_id": base_sample.chart.base_id,
                    "style_id": style_id,
                    "qa_id": qa.qa_id,
                    "task_type": qa.task_type,
                    "answer_type": qa.answer_type,
                    "split": split,
                    "chart_type": base_sample.chart.chart_type,
                    "num_points": len(base_sample.chart.categories),
                    "image_path": rel_path,
                    "question": qa.question,
                    "answer": qa.answer,
                    "categories": base_sample.chart.categories,
                    "values": base_sample.chart.values,
                    "line_trend": base_sample.chart.line_trend,
                    "style_factors": asdict(style),
                    "render_factors": style.render_quality,
                }
                metadata.append(record)
                images.append(rel_path)
                questions.append(qa.question)
                gts.append(qa.answer)
                params.append(record["style_factors"])

    metadata_path = out_dir / config.metadata_filename
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(metadata)
    summary_path = out_dir / config.summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return images, questions, gts, params


def style(config: DatasetConfig) -> Tuple[List[str], List[str], List[str]]:
    """Compatibility wrapper for style-only previews."""
    images, _, _, params = image_generation(config)
    names = [Path(p).name for p in images]
    descriptions = [
        f"font={p['font_family']}, theme={p['color_theme']}, rot={p['label_rotation']}, quality={p['quality']}"
        for p in params
    ]
    return images, names, descriptions


def build_summary(metadata: Sequence[Dict]) -> Dict:
    by_chart: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    by_style_id: Dict[str, int] = {}
    by_task: Dict[str, int] = {}
    by_points: Dict[str, int] = {}

    for row in metadata:
        by_chart[row["chart_type"]] = by_chart.get(row["chart_type"], 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        by_style_id[row["style_id"]] = by_style_id.get(row["style_id"], 0) + 1
        by_task[row["task_type"]] = by_task.get(row["task_type"], 0) + 1
        key = f"{row['chart_type']}_{row['num_points']}"
        by_points[key] = by_points.get(key, 0) + 1

    return {
        "num_records": len(metadata),
        "chart_type_distribution": by_chart,
        "split_distribution": by_split,
        "style_id_distribution": by_style_id,
        "task_distribution": by_task,
        "point_count_distribution": by_points,
    }


# -------------------------------- CLI --------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chart QA dataset with fixed chart/task rules and style perturbations."
    )
    parser.add_argument("--output-dir", type=str, default="datasets/out")
    parser.add_argument("--num-per-type", type=int, default=100)
    parser.add_argument("--style-variants-per-base", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-format", type=str, default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--metadata-filename", type=str, default="metadata.jsonl")
    parser.add_argument("--summary-filename", type=str, default="summary.json")
    parser.add_argument(
        "--skip-reference-style",
        dest="skip_reference_style",
        action="store_true",
        default=True,
        help="Skip generating the unprocessed reference style image (default: true).",
    )
    parser.add_argument(
        "--include-reference-style",
        dest="skip_reference_style",
        action="store_false",
        help="Also generate the unprocessed reference style image.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(
        output_dir=args.output_dir,
        num_per_type=args.num_per_type,
        style_variants_per_base=args.style_variants_per_base,
        seed=args.seed,
        image_format=args.image_format,
        metadata_filename=args.metadata_filename,
        summary_filename=args.summary_filename,
        skip_reference_style=args.skip_reference_style,
    )
    images, _, _, _ = image_generation(cfg)
    print(f"Generated {len(images)} QA records under: {cfg.output_dir}")
    print(f"Metadata: {Path(cfg.output_dir) / cfg.metadata_filename}")
    print(f"Summary: {Path(cfg.output_dir) / cfg.summary_filename}")


if __name__ == "__main__":
    main()
