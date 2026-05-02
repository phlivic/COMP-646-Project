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
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

POSTPROCESS_FIELDS: Tuple[str, ...] = (
    "blur",
    "noise",
    "compression",
    "resize",
    "brightness",
    "contrast",
)


# ----------------------------- Data Structures -----------------------------

@dataclass(frozen=True)
class DatasetConfig:
    output_dir: str = "datasets/out"
    num_per_type: int = 100
    style_variants_per_base: int = 8
    variants_config: Optional[str] = None
    include_base_variant: bool = True
    seed: int = 42
    image_format: str = "jpg"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    metadata_filename: str = "metadata.jsonl"
    summary_filename: str = "summary.json"
    skip_reference_style: bool = False


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
class VariantPlan:
    """One image variant to render or derive from the base image."""

    variant_id: str
    variant_kind: str
    style_id: str
    render_style: VisualStyle
    variation_types: List[str]
    variation_group: str
    transforms: Dict[str, Dict[str, Any]]
    source_variant_id: Optional[str] = None


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


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.-")
    return cleaned or "variant"


def _stable_int_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def _compact_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _clean_transform_params(value: Any) -> Dict[str, Any]:
    if value in (None, "", [], {}):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Transform config must be an object, got {type(value).__name__}.")
    return dict(value)


def _variation_group_for_transform(name: str, params: Dict[str, Any]) -> str:
    if name == "blur":
        method = params.get("method", "gaussian")
        radius = params.get("radius", params.get("value", ""))
        return f"blur:{method}:r={_compact_number(radius)}"
    if name == "noise":
        method = params.get("method", "gaussian")
        std = params.get("std", params.get("sigma", params.get("value", "")))
        return f"noise:{method}:std={_compact_number(std)}"
    if name == "compression":
        method = params.get("method", "jpeg")
        quality = params.get("quality", params.get("value", ""))
        return f"compression:{method}:q={_compact_number(quality)}"
    if name == "resize":
        scale = params.get("scale", params.get("factor", params.get("value", "")))
        return f"resize:scale={_compact_number(scale)}"
    if name == "brightness":
        factor = params.get("factor", params.get("value", ""))
        return f"brightness:factor={_compact_number(factor)}"
    if name == "contrast":
        factor = params.get("factor", params.get("value", ""))
        return f"contrast:factor={_compact_number(factor)}"
    return f"{name}:{json.dumps(params, sort_keys=True)}"


def _variation_group_from_transforms(transforms: Dict[str, Dict[str, Any]]) -> str:
    parts = [
        _variation_group_for_transform(name, transforms[name])
        for name in POSTPROCESS_FIELDS
        if transforms.get(name)
    ]
    return "+".join(parts) if parts else "base"


def _variant_id_from_transforms(transforms: Dict[str, Dict[str, Any]]) -> str:
    group = _variation_group_from_transforms(transforms)
    return _slugify(group.replace(":", "_").replace("+", "__").replace("=", ""))


def _load_postprocess_variant_specs(config_path: Optional[str]) -> List[Dict[str, Any]]:
    if not config_path:
        return []

    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        raw_specs = payload
    elif isinstance(payload, dict):
        raw_specs = payload.get("postprocess_variants") or payload.get("variants") or []
    else:
        raise ValueError("Variant config must be a JSON object or list.")

    specs: List[Dict[str, Any]] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("Each variant config entry must be a JSON object.")
        transforms: Dict[str, Dict[str, Any]] = {}
        for field in POSTPROCESS_FIELDS:
            params = _clean_transform_params(raw_spec.get(field))
            if params:
                transforms[field] = params
        if not transforms:
            continue
        variant_id = str(raw_spec.get("id") or raw_spec.get("variant_id") or _variant_id_from_transforms(transforms))
        specs.append(
            {
                "variant_id": _slugify(variant_id),
                "variation_group": str(raw_spec.get("variation_group") or _variation_group_from_transforms(transforms)),
                "transforms": transforms,
                "source_variant_id": str(raw_spec.get("source_variant_id") or raw_spec.get("source") or "base"),
            }
        )
    return specs


def build_variant_plans_for_base(
    base_id: str,
    config: DatasetConfig,
    postprocess_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[VariantPlan]:
    """Build the image variant plan for one semantic chart."""

    plans: List[VariantPlan] = []
    reference_style = VisualStyle(**REFERENCE_STYLE)
    include_base = config.include_base_variant and not config.skip_reference_style
    if include_base:
        plans.append(
            VariantPlan(
                variant_id="base",
                variant_kind="base",
                style_id="base",
                render_style=reference_style,
                variation_types=[],
                variation_group="base",
                transforms={},
                source_variant_id=None,
            )
        )

    style_variants = choose_styles_for_base(
        base_id,
        config.style_variants_per_base,
        config.seed,
        include_reference=False,
    )
    for style_id, style in style_variants:
        style_group = (
            f"render_style:font={style.font_family}"
            f"+theme={style.color_theme}"
            f"+rot={style.label_rotation}"
            f"+quality={style.quality}"
        )
        plans.append(
            VariantPlan(
                variant_id=style_id,
                variant_kind="render_style",
                style_id=style_id,
                render_style=style,
                variation_types=["render_style"],
                variation_group=style_group,
                transforms={},
                source_variant_id=None,
            )
        )

    specs = (
        list(postprocess_specs)
        if postprocess_specs is not None
        else _load_postprocess_variant_specs(config.variants_config)
    )
    for spec in specs:
        transforms = spec["transforms"]
        plans.append(
            VariantPlan(
                variant_id=spec["variant_id"],
                variant_kind="postprocess",
                style_id=spec["variant_id"],
                render_style=reference_style,
                variation_types=[
                    field
                    for field in POSTPROCESS_FIELDS
                    if transforms.get(field)
                ],
                variation_group=spec["variation_group"],
                transforms=transforms,
                source_variant_id=spec["source_variant_id"],
            )
        )

    seen: set[str] = set()
    duplicates: List[str] = []
    for plan in plans:
        if plan.variant_id in seen:
            duplicates.append(plan.variant_id)
            continue
        seen.add(plan.variant_id)
    if duplicates:
        raise ValueError(f"Duplicate variant_id values for {base_id}: {duplicates}")
    return plans


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


def _resample_filter(name: str):
    from PIL import Image

    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping.get(str(name).lower(), Image.Resampling.BICUBIC)


def _apply_blur(image, params: Dict[str, Any]):
    from PIL import ImageFilter

    method = str(params.get("method", "gaussian")).lower()
    radius = float(params.get("radius", params.get("value", 1.5)))
    if method == "box":
        return image.filter(ImageFilter.BoxBlur(radius))
    return image.filter(ImageFilter.GaussianBlur(radius))


def _apply_noise(image, params: Dict[str, Any], seed: int):
    import numpy as np
    from PIL import Image

    method = str(params.get("method", "gaussian")).lower()
    if method != "gaussian":
        raise ValueError(f"Unsupported noise method: {method}")
    std = float(params.get("std", params.get("sigma", params.get("value", 5.0))))
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    arr += rng.normal(0.0, std, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")


def _apply_compression(image, params: Dict[str, Any]):
    from PIL import Image

    method = str(params.get("method", "jpeg")).lower()
    if method not in {"jpg", "jpeg"}:
        raise ValueError(f"Unsupported compression method: {method}")
    quality = int(params.get("quality", params.get("value", 55)))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def _apply_resize(image, params: Dict[str, Any]):
    width, height = image.size
    scale = float(params.get("scale", params.get("factor", params.get("value", 0.5))))
    if scale <= 0:
        raise ValueError("Resize scale must be positive.")
    down_width = max(1, int(round(width * scale)))
    down_height = max(1, int(round(height * scale)))
    down_filter = _resample_filter(str(params.get("downsample", "bilinear")))
    up_filter = _resample_filter(str(params.get("upsample", "bicubic")))
    return image.resize((down_width, down_height), down_filter).resize((width, height), up_filter)


def _apply_enhancement(image, params: Dict[str, Any], kind: str):
    from PIL import ImageEnhance

    factor = float(params.get("factor", params.get("value", 1.0)))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(factor)
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(factor)
    raise ValueError(f"Unsupported enhancement: {kind}")


def apply_postprocess_variant(
    source_path: Path,
    output_path: Path,
    transforms: Dict[str, Dict[str, Any]],
    image_format: str,
    seed: int,
) -> None:
    """Apply a deterministic sequence of post-processing transforms."""

    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source_image:
        image = source_image.convert("RGB")

    for field in POSTPROCESS_FIELDS:
        params = transforms.get(field)
        if not params:
            continue
        if field == "blur":
            image = _apply_blur(image, params)
        elif field == "noise":
            image = _apply_noise(image, params, seed)
        elif field == "compression":
            image = _apply_compression(image, params)
        elif field == "resize":
            image = _apply_resize(image, params)
        elif field in {"brightness", "contrast"}:
            image = _apply_enhancement(image, params, field)

    save_kwargs: Dict[str, Any] = {}
    if image_format.lower() in {"jpg", "jpeg"}:
        save_kwargs["quality"] = 95
    save_format = "JPEG" if image_format.lower() in {"jpg", "jpeg"} else image_format.upper()
    image.save(output_path, format=save_format, **save_kwargs)


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
    variant_plan_map: Dict[str, List[VariantPlan]],
    cfg: DatasetConfig,
) -> Dict[Tuple[str, str], str]:
    """Assign split at base granularity with stratification by chart type.

    All image variants for the same base_id are forced into the same split.
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
            for plan in variant_plan_map[base_id]:
                split_plan[(base_id, plan.variant_id)] = split

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

    postprocess_specs = _load_postprocess_variant_specs(config.variants_config)
    variant_plan_map: Dict[str, List[VariantPlan]] = {
        sample.chart.base_id: build_variant_plans_for_base(
            sample.chart.base_id,
            config,
            postprocess_specs=postprocess_specs,
        )
        for sample in base_samples
    }
    split_plan = _build_split_plan(base_samples, variant_plan_map, config)

    metadata: List[Dict] = []
    images: List[str] = []
    questions: List[str] = []
    gts: List[str] = []
    params: List[Dict] = []

    rendered_paths: Dict[Tuple[str, str], str] = {}
    for base_sample in base_samples:
        variant_plans = variant_plan_map[base_sample.chart.base_id]

        # Render or derive one image for each variant, then attach all QA rows.
        for plan in variant_plans:
            split = split_plan[(base_sample.chart.base_id, plan.variant_id)]
            filename = f"{base_sample.chart.base_id}__{plan.variant_id}.{config.image_format}"
            rel_path = f"{split}/{filename}"
            abs_path = out_dir / rel_path
            source_image_path: Optional[str] = None

            if plan.variant_kind == "postprocess":
                source_variant_id = plan.source_variant_id or "base"
                source_image_path = rendered_paths.get((base_sample.chart.base_id, source_variant_id))
                if source_image_path is None:
                    raise ValueError(
                        f"Variant {plan.variant_id} for {base_sample.chart.base_id} "
                        f"requires source variant {source_variant_id}, but it has not been rendered."
                    )
                apply_postprocess_variant(
                    out_dir / source_image_path,
                    abs_path,
                    plan.transforms,
                    config.image_format,
                    seed=_stable_int_seed(config.seed, base_sample.chart.base_id, plan.variant_id),
                )
            else:
                render_chart(base_sample.chart, plan.render_style, abs_path, config.image_format)

            rendered_paths[(base_sample.chart.base_id, plan.variant_id)] = rel_path
            style_factors = asdict(plan.render_style)
            transform_fields = {
                field: plan.transforms.get(field) or None
                for field in POSTPROCESS_FIELDS
            }

            for qa in base_sample.qa_items:
                sample_id = f"{base_sample.chart.base_id}__{plan.variant_id}__{qa.qa_id}"
                record = {
                    "sample_id": sample_id,
                    "base_id": base_sample.chart.base_id,
                    "style_id": plan.style_id,
                    "variant_id": plan.variant_id,
                    "is_base_variant": plan.variant_kind == "base",
                    "variant_kind": plan.variant_kind,
                    "variation_types": list(plan.variation_types),
                    "variation_group": plan.variation_group,
                    "source_image_path": source_image_path,
                    "render_style": style_factors if plan.variant_kind == "render_style" else None,
                    **transform_fields,
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
                    "style_factors": style_factors,
                    "render_factors": plan.render_style.render_quality,
                }
                metadata.append(record)
                images.append(rel_path)
                questions.append(qa.question)
                gts.append(qa.answer)
                params.append(
                    {
                        "variant_id": plan.variant_id,
                        "variant_kind": plan.variant_kind,
                        "variation_types": list(plan.variation_types),
                        "variation_group": plan.variation_group,
                        "style_factors": style_factors,
                        **transform_fields,
                    }
                )

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
        f"variant={p['variant_id']}, types={'+'.join(p['variation_types']) or 'base'}, group={p['variation_group']}"
        for p in params
    ]
    return images, names, descriptions


def build_summary(metadata: Sequence[Dict]) -> Dict:
    by_chart: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    by_style_id: Dict[str, int] = {}
    by_variant_id: Dict[str, int] = {}
    by_variant_kind: Dict[str, int] = {}
    by_variation_type: Dict[str, int] = {}
    by_variation_group: Dict[str, int] = {}
    by_variation_combination: Dict[str, int] = {}
    variation_field_counts: Dict[str, int] = {field: 0 for field in POSTPROCESS_FIELDS}
    by_task: Dict[str, int] = {}
    by_points: Dict[str, int] = {}

    for row in metadata:
        by_chart[row["chart_type"]] = by_chart.get(row["chart_type"], 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        by_style_id[row["style_id"]] = by_style_id.get(row["style_id"], 0) + 1
        by_variant_id[row["variant_id"]] = by_variant_id.get(row["variant_id"], 0) + 1
        by_variant_kind[row["variant_kind"]] = by_variant_kind.get(row["variant_kind"], 0) + 1
        by_variation_group[row["variation_group"]] = by_variation_group.get(row["variation_group"], 0) + 1
        variation_types = list(row.get("variation_types") or [])
        combination_key = "+".join(sorted(variation_types)) if variation_types else "base"
        by_variation_combination[combination_key] = by_variation_combination.get(combination_key, 0) + 1
        if not variation_types:
            by_variation_type["base"] = by_variation_type.get("base", 0) + 1
        for variation_type in variation_types:
            by_variation_type[variation_type] = by_variation_type.get(variation_type, 0) + 1
        for field in POSTPROCESS_FIELDS:
            if row.get(field):
                variation_field_counts[field] += 1
        by_task[row["task_type"]] = by_task.get(row["task_type"], 0) + 1
        key = f"{row['chart_type']}_{row['num_points']}"
        by_points[key] = by_points.get(key, 0) + 1

    image_paths = {row["image_path"] for row in metadata}
    return {
        "num_records": len(metadata),
        "num_images": len(image_paths),
        "chart_type_distribution": by_chart,
        "split_distribution": by_split,
        "style_id_distribution": by_style_id,
        "variant_id_distribution": by_variant_id,
        "variant_kind_distribution": by_variant_kind,
        "variation_type_distribution": by_variation_type,
        "variation_group_distribution": by_variation_group,
        "variation_combination_distribution": by_variation_combination,
        "variation_field_counts": variation_field_counts,
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
    parser.add_argument(
        "--variants-config",
        type=str,
        default=None,
        help="Optional JSON file listing post-processing variants such as blur/noise/compression.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-format", type=str, default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--metadata-filename", type=str, default="metadata.jsonl")
    parser.add_argument("--summary-filename", type=str, default="summary.json")
    parser.add_argument(
        "--skip-reference-style",
        dest="include_base_variant",
        action="store_false",
        default=True,
        help="Deprecated compatibility flag; omit the unprocessed base image.",
    )
    parser.add_argument(
        "--include-reference-style",
        dest="include_base_variant",
        action="store_true",
        help="Generate the unprocessed base image (default).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(
        output_dir=args.output_dir,
        num_per_type=args.num_per_type,
        style_variants_per_base=args.style_variants_per_base,
        variants_config=args.variants_config,
        include_base_variant=args.include_base_variant,
        seed=args.seed,
        image_format=args.image_format,
        metadata_filename=args.metadata_filename,
        summary_filename=args.summary_filename,
        skip_reference_style=not args.include_base_variant,
    )
    images, _, _, _ = image_generation(cfg)
    print(f"Generated {len(images)} QA records under: {cfg.output_dir}")
    print(f"Metadata: {Path(cfg.output_dir) / cfg.metadata_filename}")
    print(f"Summary: {Path(cfg.output_dir) / cfg.summary_filename}")


if __name__ == "__main__":
    main()
