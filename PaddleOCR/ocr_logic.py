"""Chart parsing, answer generation, and metric aggregation logic.

This module keeps the OCR-to-structure logic and the evaluation logic together
because both operate on the same ParsedChart semantics. The intent is to keep
the main script focused on orchestration while the chart-specific rules live in
one place.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Sequence

from .ocr_types import (
    AXIS_LABELS_BY_CHART,
    TITLE_BY_CHART,
    ImageCase,
    OCRToken,
    ParsedChart,
    QARecord,
    classify_line_trend,
    format_number,
    format_percent,
    normalize_key,
    normalize_text,
    parse_category,
    parse_number,
    parse_percent,
)


def dedupe_best_tokens(
    tokens: Sequence[OCRToken],
    key_builder,
) -> list[OCRToken]:
    """Keep the highest-confidence token for each semantic key."""

    best_by_key: dict[str, OCRToken] = {}
    for token in tokens:
        key = key_builder(token)
        if not key:
            continue
        current = best_by_key.get(key)
        current_score = -1.0 if current is None or current.score is None else current.score
        new_score = -1.0 if token.score is None else token.score
        if current is None or new_score >= current_score:
            best_by_key[key] = token
    return list(best_by_key.values())


def is_title_token(token: OCRToken) -> bool:
    """Return whether the token is one of the chart titles."""

    return normalize_key(token.norm_text) in {value.lower() for value in TITLE_BY_CHART.values()}


def is_axis_label_token(token: OCRToken) -> bool:
    """Return whether the token matches an axis label from any chart type."""

    axis_labels = {label.lower() for labels in AXIS_LABELS_BY_CHART.values() for label in labels}
    return normalize_key(token.norm_text) in axis_labels


def is_probable_left_axis_tick(token: OCRToken, case: ImageCase) -> bool:
    """Coarse left-margin heuristic used for bar-chart numeric filtering."""

    value = parse_number(token.norm_text)
    if value is None:
        return False
    return token.cx <= case.image_width * 0.18


def select_numeric_tokens_by_x(
    number_tokens: Sequence[OCRToken],
    category_tokens: Sequence[OCRToken],
    expected_count: int,
    image_width: int,
) -> list[OCRToken]:
    """Pick up to expected_count numeric tokens using left-to-right alignment."""

    ordered_numbers = sorted(number_tokens, key=lambda token: (token.cx, token.cy))
    ordered_categories = sorted(category_tokens, key=lambda token: (token.cx, token.cy))
    if expected_count <= 0 or not ordered_numbers:
        return []

    if ordered_categories:
        used_indices: set[int] = set()
        matched_tokens: list[OCRToken] = []
        for category_token in ordered_categories[:expected_count]:
            best_index: int | None = None
            best_distance: float | None = None
            for index, number_token in enumerate(ordered_numbers):
                if index in used_indices:
                    continue
                distance = abs(number_token.cx - category_token.cx)
                if number_token.cy > category_token.cy:
                    distance += (number_token.cy - category_token.cy) * 0.5
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index
            if (
                best_index is not None
                and best_distance is not None
                and best_distance <= image_width * 0.24
            ):
                used_indices.add(best_index)
                matched_tokens.append(ordered_numbers[best_index])
        if len(matched_tokens) == expected_count:
            return matched_tokens

    filtered = [token for token in ordered_numbers if token.cx >= image_width * 0.18]
    return filtered[:expected_count]


def estimate_left_axis_tick_tokens(
    number_tokens: Sequence[OCRToken],
    category_tokens: Sequence[OCRToken],
    image_width: int,
) -> set[int]:
    """Detect the numeric column that belongs to the y-axis instead of data points.

    A fixed x-threshold is too brittle for line charts because the first plotted
    value can sit very close to the y-axis. Instead, this function looks for a
    dense left-side numeric column relative to the first category position and
    removes the tokens that belong to that column.
    """

    if not number_tokens or not category_tokens:
        return set()

    ordered_categories = sorted(category_tokens, key=lambda token: token.cx)
    first_category_x = ordered_categories[0].cx
    if len(ordered_categories) >= 2:
        spacings = [
            ordered_categories[index + 1].cx - ordered_categories[index].cx
            for index in range(len(ordered_categories) - 1)
        ]
        min_spacing = min(spacings)
    else:
        min_spacing = image_width * 0.25

    margin = min(80.0, max(45.0, min_spacing * 0.25))
    left_candidates = [
        (index, token)
        for index, token in enumerate(number_tokens)
        if token.cx < first_category_x - margin
    ]
    if len(left_candidates) < 3:
        return set()

    cluster_center = statistics.median(token.cx for _, token in left_candidates)
    cluster_half_width = max(
        35.0,
        statistics.median(max(token.w, 1.0) for _, token in left_candidates) * 1.8,
    )
    return {
        index
        for index, token in enumerate(number_tokens)
        if abs(token.cx - cluster_center) <= cluster_half_width
    }


def assign_numeric_tokens_to_categories(
    number_tokens: Sequence[OCRToken],
    category_tokens: Sequence[OCRToken],
) -> list[tuple[OCRToken, OCRToken | None]]:
    """Assign numeric tokens to categories using midpoint-based x partitions.

    Once left-axis ticks are removed, each remaining value token is matched inside
    the x-span owned by a category. This avoids the global left-to-right shift
    error that happens when one point value is missed by OCR.
    """

    ordered_categories = sorted(category_tokens, key=lambda token: (token.cx, token.cy))
    if not ordered_categories:
        return []

    ordered_numbers = sorted(number_tokens, key=lambda token: (token.cx, token.cy))
    category_centers = [token.cx for token in ordered_categories]
    boundaries = [-float("inf")]
    for index in range(len(category_centers) - 1):
        boundaries.append((category_centers[index] + category_centers[index + 1]) / 2.0)
    boundaries.append(float("inf"))

    assignments: list[tuple[OCRToken, OCRToken | None]] = []
    for index, category_token in enumerate(ordered_categories):
        left_boundary = boundaries[index]
        right_boundary = boundaries[index + 1]
        candidates = [
            token
            for token in ordered_numbers
            if left_boundary < token.cx <= right_boundary
        ]
        if not candidates:
            assignments.append((category_token, None))
            continue

        best_token = min(
            candidates,
            key=lambda token: (
                abs(token.cx - category_token.cx),
                abs(token.cy - category_token.cy),
                -(token.score if token.score is not None else -1.0),
            ),
        )
        assignments.append((category_token, best_token))

    return assignments


def token_angle(token: OCRToken, center_x: float, center_y: float) -> float:
    """Return the token angle around the estimated pie center."""

    angle = math.atan2(token.cy - center_y, token.cx - center_x)
    return angle if angle >= 0 else angle + 2 * math.pi


def circular_angle_distance(a: float, b: float) -> float:
    """Return the shortest angular distance between two angles."""

    delta = abs(a - b)
    return min(delta, 2 * math.pi - delta)


def sort_tokens_by_angle(
    tokens: Sequence[OCRToken],
    center_x: float,
    center_y: float,
) -> list[tuple[OCRToken, float]]:
    """Sort tokens by polar angle around the estimated pie center."""

    with_angles = [(token, token_angle(token, center_x, center_y)) for token in tokens]
    return sorted(with_angles, key=lambda item: item[1])


def pair_tokens_by_circular_shift(
    categories: Sequence[OCRToken],
    percents: Sequence[OCRToken],
    center_x: float,
    center_y: float,
) -> list[tuple[OCRToken, OCRToken]]:
    """Pair pie categories and percentages with a circular-shift search."""

    if not categories or not percents:
        return []

    ordered_categories = sort_tokens_by_angle(categories, center_x, center_y)
    ordered_percents = sort_tokens_by_angle(percents, center_x, center_y)

    if len(ordered_categories) == len(ordered_percents):
        count = len(ordered_categories)
        best_shift = 0
        best_cost: float | None = None
        for shift in range(count):
            cost = 0.0
            for index in range(count):
                _, category_angle = ordered_categories[index]
                _, percent_angle = ordered_percents[(index + shift) % count]
                cost += circular_angle_distance(category_angle, percent_angle)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_shift = shift
        return [
            (ordered_categories[index][0], ordered_percents[(index + best_shift) % count][0])
            for index in range(count)
        ]

    remaining = list(ordered_percents)
    pairs: list[tuple[OCRToken, OCRToken]] = []
    for category_token, category_angle in ordered_categories:
        if not remaining:
            break
        best_index = min(
            range(len(remaining)),
            key=lambda idx: circular_angle_distance(category_angle, remaining[idx][1]),
        )
        pairs.append((category_token, remaining.pop(best_index)[0]))
    return pairs


def build_pie_pairs_from_tokens(
    category_tokens: Sequence[OCRToken],
    percent_tokens: Sequence[OCRToken],
    center_x: float,
    center_y: float,
) -> list[dict[str, Any]]:
    """Build pie category-percent pairs, including one-missing-percent repair.

    If OCR finds exactly one more category than percentage token, the missing
    percent can be inferred from the 100% total. The repaired value is treated
    as a normal recovered pair because it is a deterministic logical fix rather
    than a separate model prediction.
    """

    if not category_tokens or not percent_tokens:
        return []

    ordered_categories = sort_tokens_by_angle(category_tokens, center_x, center_y)
    ordered_percents = sort_tokens_by_angle(percent_tokens, center_x, center_y)

    if len(ordered_categories) == len(ordered_percents) + 1:
        percent_values = [parse_percent(token.norm_text) for token, _ in ordered_percents]
        if all(value is not None for value in percent_values):
            inferred_percent = round(100.0 - sum(float(value) for value in percent_values), 1)
            if 0.0 < inferred_percent < 100.0:
                best_missing_idx: int | None = None
                best_shift = 0
                best_cost: float | None = None
                for missing_idx in range(len(ordered_categories)):
                    remaining_categories = [
                        ordered_categories[index]
                        for index in range(len(ordered_categories))
                        if index != missing_idx
                    ]
                    for shift in range(len(ordered_percents)):
                        cost = 0.0
                        for index, (_, category_angle) in enumerate(remaining_categories):
                            _, percent_angle = ordered_percents[(index + shift) % len(ordered_percents)]
                            cost += circular_angle_distance(category_angle, percent_angle)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_missing_idx = missing_idx
                            best_shift = shift

                if best_missing_idx is not None:
                    built_pairs: list[dict[str, Any]] = []
                    percent_position = 0
                    for category_index, (category_token, _) in enumerate(ordered_categories):
                        category = parse_category(category_token.norm_text)
                        if category is None:
                            continue
                        if category_index == best_missing_idx:
                            built_pairs.append({"category": category, "percent": inferred_percent})
                            continue
                        percent_token, _ = ordered_percents[
                            (percent_position + best_shift) % len(ordered_percents)
                        ]
                        percent = parse_percent(percent_token.norm_text)
                        if percent is not None:
                            built_pairs.append({"category": category, "percent": percent})
                        percent_position += 1
                    return built_pairs

    paired_tokens = pair_tokens_by_circular_shift(category_tokens, percent_tokens, center_x, center_y)
    built_pairs: list[dict[str, Any]] = []
    for category_token, percent_token in paired_tokens:
        category = parse_category(category_token.norm_text)
        percent = parse_percent(percent_token.norm_text)
        if category is None or percent is None:
            continue
        built_pairs.append({"category": category, "percent": percent})
    return built_pairs


def parse_bar_chart(tokens: Sequence[OCRToken], case: ImageCase) -> ParsedChart:
    """Recover bar-chart categories and values from OCR tokens."""

    warnings: list[str] = []
    semantic_tokens = [token for token in tokens if not is_title_token(token) and not is_axis_label_token(token)]
    category_tokens = dedupe_best_tokens(
        [token for token in semantic_tokens if parse_category(token.norm_text) is not None],
        key_builder=lambda token: parse_category(token.norm_text),
    )
    category_tokens = sorted(category_tokens, key=lambda token: (token.cx, token.cy))

    number_tokens = [
        token
        for token in semantic_tokens
        if parse_number(token.norm_text) is not None and not is_probable_left_axis_tick(token, case)
    ]
    value_tokens = select_numeric_tokens_by_x(
        number_tokens,
        category_tokens,
        case.num_points,
        case.image_width,
    )
    values = [parse_number(token.norm_text) for token in value_tokens if parse_number(token.norm_text) is not None]
    categories = [parse_category(token.norm_text) for token in category_tokens[: case.num_points]]
    pairs = [
        {
            "category": categories[index] if index < len(categories) else None,
            "value": values[index],
        }
        for index in range(len(values))
    ]

    if len(values) != case.num_points:
        warnings.append(
            f"Recovered {len(values)} bar values but expected {case.num_points}."
        )
    if len(categories) != case.num_points:
        warnings.append(
            f"Recovered {len(categories)} categories but expected {case.num_points}."
        )

    if len(values) == case.num_points:
        parse_status = "ok"
    elif values:
        parse_status = "partial"
    else:
        parse_status = "failed"

    return ParsedChart(
        image_path=case.image_path,
        chart_type=case.chart_type,
        parse_status=parse_status,
        categories=[category for category in categories if category is not None],
        values=values,
        pairs=pairs,
        warnings=warnings,
    )


def parse_line_chart(tokens: Sequence[OCRToken], case: ImageCase) -> ParsedChart:
    """Recover line-chart categories and point values from OCR tokens."""

    warnings: list[str] = []
    semantic_tokens = [token for token in tokens if not is_title_token(token) and not is_axis_label_token(token)]
    category_tokens = dedupe_best_tokens(
        [token for token in semantic_tokens if parse_category(token.norm_text) is not None],
        key_builder=lambda token: parse_category(token.norm_text),
    )
    category_tokens = sorted(category_tokens, key=lambda token: (token.cx, token.cy))

    number_tokens = [
        token
        for token in semantic_tokens
        if parse_number(token.norm_text) is not None
    ]
    left_axis_tick_indices = estimate_left_axis_tick_tokens(
        number_tokens,
        category_tokens,
        case.image_width,
    )
    filtered_number_tokens = [
        token
        for index, token in enumerate(number_tokens)
        if index not in left_axis_tick_indices
    ]
    aligned_pairs = assign_numeric_tokens_to_categories(
        filtered_number_tokens,
        category_tokens[: case.num_points],
    )
    categories = [parse_category(token.norm_text) for token in category_tokens[: case.num_points]]
    pairs: list[dict[str, Any]] = []
    values: list[float] = []
    for category_token, value_token in aligned_pairs:
        category = parse_category(category_token.norm_text)
        value = parse_number(value_token.norm_text) if value_token is not None else None
        pairs.append({"category": category, "value": value})
        if value is not None:
            values.append(value)
    trend = classify_line_trend(values) if values else None

    if len(values) != case.num_points:
        warnings.append(
            f"Recovered {len(values)} line values but expected {case.num_points}."
        )
    if left_axis_tick_indices:
        warnings.append(
            f"Filtered {len(left_axis_tick_indices)} probable left-axis tick numbers."
        )
    if len(categories) != case.num_points:
        warnings.append(
            f"Recovered {len(categories)} categories but expected {case.num_points}."
        )

    if len(values) == case.num_points:
        parse_status = "ok"
    elif values:
        parse_status = "partial"
    else:
        parse_status = "failed"

    return ParsedChart(
        image_path=case.image_path,
        chart_type=case.chart_type,
        parse_status=parse_status,
        categories=[category for category in categories if category is not None],
        values=values,
        pairs=pairs,
        trend=trend,
        warnings=warnings,
    )


def parse_pie_chart(tokens: Sequence[OCRToken], case: ImageCase) -> ParsedChart:
    """Recover pie categories and percentages from OCR tokens."""

    warnings: list[str] = []
    semantic_tokens = [token for token in tokens if not is_title_token(token)]
    category_tokens = dedupe_best_tokens(
        [token for token in semantic_tokens if parse_category(token.norm_text) is not None],
        key_builder=lambda token: parse_category(token.norm_text),
    )
    percent_tokens = dedupe_best_tokens(
        [token for token in semantic_tokens if parse_percent(token.norm_text) is not None],
        key_builder=lambda token: format_percent(parse_percent(token.norm_text) or 0.0),
    )
    category_tokens = category_tokens[: case.num_points]
    percent_tokens = sorted(
        percent_tokens,
        key=lambda token: (-1.0 if token.score is None else token.score),
        reverse=True,
    )[: case.num_points]

    if category_tokens or percent_tokens:
        all_tokens = [*category_tokens, *percent_tokens]
        center_x = sum(token.cx for token in all_tokens) / len(all_tokens)
        center_y = sum(token.cy for token in all_tokens) / len(all_tokens)
    else:
        center_x = case.image_width / 2.0
        center_y = case.image_height / 2.0

    pairs = build_pie_pairs_from_tokens(category_tokens, percent_tokens, center_x, center_y)

    percentages = [pair["percent"] for pair in pairs]
    categories = [pair["category"] for pair in pairs]

    if len(pairs) != case.num_points:
        warnings.append(
            f"Recovered {len(pairs)} pie pairs but expected {case.num_points}."
        )

    if len(pairs) == case.num_points:
        parse_status = "ok"
    elif pairs:
        parse_status = "partial"
    else:
        parse_status = "failed"

    return ParsedChart(
        image_path=case.image_path,
        chart_type=case.chart_type,
        parse_status=parse_status,
        categories=categories,
        percentages=percentages,
        pairs=pairs,
        warnings=warnings,
    )


def parse_chart(tokens: Sequence[OCRToken], case: ImageCase) -> ParsedChart:
    """Dispatch parsing to the chart-specific baseline."""

    if case.chart_type == "bar":
        return parse_bar_chart(tokens, case)
    if case.chart_type == "line":
        return parse_line_chart(tokens, case)
    if case.chart_type == "pie":
        return parse_pie_chart(tokens, case)
    return ParsedChart(
        image_path=case.image_path,
        chart_type=case.chart_type,
        parse_status="failed",
        warnings=[f"Unsupported chart_type: {case.chart_type}"],
    )


def expected_visible_texts(case: ImageCase) -> dict[str, set[str]]:
    """Return the raw text strings that should be visibly present in the image."""

    title_texts = {normalize_key(TITLE_BY_CHART[case.chart_type])}
    axis_texts = {normalize_key(label) for label in AXIS_LABELS_BY_CHART[case.chart_type]}
    category_texts = set(case.categories_gt)
    number_texts: set[str] = set()
    percent_texts: set[str] = set()

    if case.chart_type in {"bar", "line"}:
        number_texts = {format_number(value) for value in case.values_gt}
    elif case.chart_type == "pie":
        total = sum(case.values_gt) or 1.0
        percent_texts = {format_percent((value / total) * 100.0) for value in case.values_gt}

    return {
        "titles": title_texts,
        "axis_labels": axis_texts,
        "categories": category_texts,
        "numbers": number_texts,
        "percentages": percent_texts,
    }


def recognized_visible_texts(tokens: Sequence[OCRToken]) -> dict[str, set[str]]:
    """Extract raw OCR text buckets for high-level diagnostics.

    This intentionally operates on raw OCR tokens, so line-chart y-axis ticks are
    still counted as recognized numbers here. Use compute_line_point_metrics for
    the point-only line metric.
    """

    titles: set[str] = set()
    axis_labels: set[str] = set()
    categories: set[str] = set()
    numbers: set[str] = set()
    percentages: set[str] = set()

    for token in tokens:
        key = normalize_key(token.norm_text)
        if key in {value.lower() for value in TITLE_BY_CHART.values()}:
            titles.add(key)
            continue
        if key in {label.lower() for labels in AXIS_LABELS_BY_CHART.values() for label in labels}:
            axis_labels.add(key)
            continue
        category = parse_category(token.norm_text)
        if category is not None:
            categories.add(category)
            continue
        percent = parse_percent(token.norm_text)
        if percent is not None:
            percentages.add(format_percent(percent))
            continue
        number = parse_number(token.norm_text)
        if number is not None:
            numbers.add(format_number(number))

    return {
        "titles": titles,
        "axis_labels": axis_labels,
        "categories": categories,
        "numbers": numbers,
        "percentages": percentages,
    }


def compute_visible_text_metrics(case: ImageCase, tokens: Sequence[OCRToken]) -> dict[str, Any]:
    """Compute raw OCR text diagnostics before chart-specific filtering."""

    expected = expected_visible_texts(case)
    recognized = recognized_visible_texts(tokens)

    component_metrics: dict[str, Any] = {}
    total_expected = 0
    total_found = 0
    total_matched = 0

    for component in ("titles", "axis_labels", "categories", "numbers", "percentages"):
        expected_values = expected[component]
        found_values = recognized[component]
        matched_values = expected_values & found_values
        total_expected += len(expected_values)
        total_found += len(found_values)
        total_matched += len(matched_values)
        component_metrics[component] = {
            "expected_count": len(expected_values),
            "found_count": len(found_values),
            "matched_count": len(matched_values),
            "recall": (
                len(matched_values) / len(expected_values)
                if expected_values
                else 1.0
            ),
            "precision": (
                len(matched_values) / len(found_values)
                if found_values
                else 1.0
            ),
        }

    component_metrics["overall"] = {
        "expected_count": total_expected,
        "found_count": total_found,
        "matched_count": total_matched,
        "recall": total_matched / total_expected if total_expected else 1.0,
        "precision": total_matched / total_found if total_found else 1.0,
    }
    return component_metrics


def compute_line_point_metrics(parsed_chart: ParsedChart, case: ImageCase) -> dict[str, Any] | None:
    """Compute the line-specific point recovery metric after parsing.

    This metric ignores unrelated OCR text and only checks whether each expected
    category recovered the correct plotted value.
    """

    if case.chart_type != "line":
        return None

    parsed_by_category = {
        str(pair["category"]): pair.get("value")
        for pair in parsed_chart.pairs
        if pair.get("category") is not None
    }

    point_results: list[dict[str, Any]] = []
    matched_count = 0
    recovered_count = 0
    expected_count = len(case.categories_gt)

    for category, gt_value in zip(case.categories_gt, case.values_gt):
        pred_value = parsed_by_category.get(category)
        pred_value_fmt = format_number(float(pred_value)) if pred_value is not None else None
        gt_value_fmt = format_number(float(gt_value))
        matched = pred_value_fmt == gt_value_fmt
        if pred_value is not None:
            recovered_count += 1
        if matched:
            matched_count += 1
        point_results.append(
            {
                "category": category,
                "gt_value": gt_value_fmt,
                "pred_value": pred_value_fmt,
                "matched": matched,
            }
        )

    return {
        "expected_count": expected_count,
        "recovered_count": recovered_count,
        "matched_count": matched_count,
        "missing_count": max(0, expected_count - recovered_count),
        "recall": matched_count / expected_count if expected_count else 1.0,
        "precision": matched_count / recovered_count if recovered_count else 1.0,
        "full_match": matched_count == expected_count,
        "point_results": point_results,
    }


def compute_task_evidence_metrics(parsed_chart: ParsedChart, qa: QARecord, case: ImageCase) -> dict[str, Any]:
    """Measure whether the parser recovered enough structure for a QA item."""

    if case.chart_type in {"bar", "line"}:
        evidence_type = "all_values"
        recovered_count = len(parsed_chart.values)
        required_count = case.num_points
    else:
        evidence_type = "all_pairs"
        recovered_count = len(parsed_chart.pairs)
        required_count = case.num_points

    evidence_complete = recovered_count >= required_count
    if recovered_count == 0:
        evidence_status = "missing"
    elif evidence_complete:
        evidence_status = "complete"
    else:
        evidence_status = "partial"

    return {
        "evidence_type": evidence_type,
        "recovered_count": recovered_count,
        "required_count": required_count,
        "missing_count": max(0, required_count - recovered_count),
        "evidence_complete": evidence_complete,
        "evidence_status": evidence_status,
    }


def answer_task(parsed_chart: ParsedChart, qa: QARecord) -> str | None:
    """Generate the baseline answer for one QA task from a parsed chart."""

    if qa.chart_type in {"bar", "line"}:
        values = [float(value) for value in parsed_chart.values]
        if not values:
            return None
        if qa.task_type.endswith("_max_value"):
            return format_number(max(values))
        if qa.task_type.endswith("_min_value"):
            return format_number(min(values))
        if qa.task_type.endswith("_avg_value"):
            return format_number(sum(values) / len(values))
        if qa.task_type == "line_trend":
            if len(values) < 2:
                return None
            return classify_line_trend(values)
        return None

    if qa.chart_type == "pie":
        pairs = [
            pair
            for pair in parsed_chart.pairs
            if pair.get("category") is not None and pair.get("percent") is not None
        ]
        if not pairs:
            return None
        ordered_desc = sorted(pairs, key=lambda pair: pair["percent"], reverse=True)
        ordered_asc = sorted(pairs, key=lambda pair: pair["percent"])
        if qa.task_type == "pie_largest_share":
            return ordered_desc[0]["category"]
        if qa.task_type == "pie_second_largest_share":
            return ordered_desc[1]["category"] if len(ordered_desc) >= 2 else None
        if qa.task_type == "pie_smallest_share":
            return ordered_asc[0]["category"]
        if qa.task_type == "pie_second_smallest_share":
            return ordered_asc[1]["category"] if len(ordered_asc) >= 2 else None
    return None


def normalize_answer_for_compare(answer: str | None, answer_type: str) -> str | None:
    """Normalize prediction and ground truth into the same comparison space."""

    if answer is None:
        return None
    if answer_type == "number":
        value = parse_number(answer)
        return format_number(value) if value is not None else normalize_text(answer)
    if answer_type == "category":
        return parse_category(answer) or normalize_text(answer)
    return normalize_key(answer)


def infer_ocr_status(tokens: Sequence[OCRToken]) -> str:
    """Collapse OCR token presence into a lightweight image-level status."""

    if not tokens:
        return "empty"
    return "text_found"


def infer_error_type(
    pred_answer: str | None,
    correct: bool,
    ocr_status: str,
    parsed_chart: ParsedChart,
    evidence: dict[str, Any],
) -> str | None:
    """Assign a coarse error bucket for failure analysis."""

    if correct:
        return None
    if ocr_status == "empty":
        return "ocr_empty"
    if parsed_chart.parse_status == "failed":
        return "parse_failed"
    if pred_answer is None:
        return "missing_prediction"
    if not evidence["evidence_complete"]:
        return "missing_evidence"
    return "wrong_answer"


def aggregate_accuracy(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """Aggregate accuracy by a categorical key."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)

    summary: dict[str, Any] = {}
    for group_key, group_records in sorted(grouped.items()):
        count = len(group_records)
        correct = sum(1 for record in group_records if record["correct"])
        summary[group_key] = {
            "count": count,
            "correct": correct,
            "accuracy": correct / count if count else 0.0,
        }
    return summary


def aggregate_distribution(records: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    """Aggregate value counts for a categorical key."""

    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def aggregate_visible_text_metrics(image_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw OCR text diagnostics across images."""

    totals: dict[str, dict[str, int]] = {
        component: {"expected_count": 0, "found_count": 0, "matched_count": 0}
        for component in ("titles", "axis_labels", "categories", "numbers", "percentages", "overall")
    }
    for record in image_metrics:
        visible = record["raw_visible_text_metrics"]
        for component, stats in visible.items():
            totals[component]["expected_count"] += int(stats["expected_count"])
            totals[component]["found_count"] += int(stats["found_count"])
            totals[component]["matched_count"] += int(stats["matched_count"])

    summary: dict[str, Any] = {}
    for component, stats in totals.items():
        expected_count = stats["expected_count"]
        found_count = stats["found_count"]
        matched_count = stats["matched_count"]
        summary[component] = {
            **stats,
            "recall": matched_count / expected_count if expected_count else 1.0,
            "precision": matched_count / found_count if found_count else 1.0,
        }
    return summary


def aggregate_line_point_metrics(image_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the line-only point recovery metric across images."""

    line_records = [
        record["line_point_metrics"]
        for record in image_metrics
        if record.get("line_point_metrics") is not None
    ]
    if not line_records:
        return {
            "image_count": 0,
            "expected_count": 0,
            "recovered_count": 0,
            "matched_count": 0,
            "missing_count": 0,
            "recall": 1.0,
            "precision": 1.0,
            "full_match_count": 0,
            "full_match_rate": 1.0,
        }

    expected_count = sum(int(record["expected_count"]) for record in line_records)
    recovered_count = sum(int(record["recovered_count"]) for record in line_records)
    matched_count = sum(int(record["matched_count"]) for record in line_records)
    missing_count = sum(int(record["missing_count"]) for record in line_records)
    full_match_count = sum(1 for record in line_records if record["full_match"])

    return {
        "image_count": len(line_records),
        "expected_count": expected_count,
        "recovered_count": recovered_count,
        "matched_count": matched_count,
        "missing_count": missing_count,
        "recall": matched_count / expected_count if expected_count else 1.0,
        "precision": matched_count / recovered_count if recovered_count else 1.0,
        "full_match_count": full_match_count,
        "full_match_rate": full_match_count / len(line_records) if line_records else 1.0,
    }
