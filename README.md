# COMP-646 Project

## Scope
This project builds a chart understanding dataset for:
- OCR baselines
- rule-based baselines
- multimodal LLM baselines

Supported chart types:
- bar chart
- line chart
- pie chart

## Current Status
- Dataset construction is **initially completed**.
- Chart-level QA tasks and style variations are implemented.
- Metadata is exported in JSONL format for downstream benchmarking.

## Deployment (Linux + GPU on NOTS HPC)
All deployment is designed for Linux with GPU acceleration

### 1. Create environment
```bash
conda create -n chartqa python=3.11 -y
conda activate chartqa
pip install -r requirements.txt
```

### 2. GPU runtime check
```bash
python -c "import paddle; print('paddle:', paddle.__version__); print('gpu:', paddle.is_compiled_with_cuda())"
```

### 3. Generate dataset
```bash
python datasets/data.py \
  --output-dir datasets/out \
  --num-per-type 100 \
  --style-variants-per-base 8 \
  --variants-config configs/variants.example.json
```

The generator now creates a `base` image variant by default. Additional render
style variants are controlled by `--style-variants-per-base`, and optional
post-processing variants such as blur/noise/compression are listed in
`--variants-config`. Use `--style-variants-per-base 0` if you only want base
images plus post-processing variants. Each metadata row includes explicit
variant fields such as `variant_id`, `is_base_variant`, `variant_kind`, `variation_types`,
`variation_group`, and per-transform fields like `blur`, `noise`, and
`compression`. A transform field is `null` when that transform was not applied.

## Baselines
This repository currently contains two executable benchmark pipelines:
- `PaddleOCR` chart-QA baseline
- multimodal LLM direct-QA baseline

### PaddleOCR Baseline
Run:
```bash
python PaddleOCR/run_ocr.py \
  --dataset-dir datasets/out \
  --output-dir runs/ocr/latest
```

Principle:
- The pipeline loads `metadata.jsonl`, groups QA rows by image, and runs OCR once per image.
- OCR tokens are normalized into a unified token format with text, score, and geometry.
- Chart structure is then recovered from OCR text instead of directly answering from the image.
- The recovered chart is used to answer each QA item and compare against ground truth.

Chart parsing logic:
- Bar charts: remove title and axis labels, recover `Cat-N` labels, filter likely y-axis ticks, and align numeric values to categories by x-position.
- Line charts: recover category labels, detect and remove left-axis tick numbers, assign remaining values to categories by x partitions, then infer the overall trend from recovered values.
- Pie charts: recover category labels and percentage tokens, pair them by circular order around the pie center, and optionally repair one missing percentage from the 100% total.

Correctness and metrics:
- QA correctness is exact-match after normalization of numbers, labels, and categories.
- The pipeline also reports raw OCR text recall/precision, parse status, evidence completeness, and line-point recovery metrics.
- This baseline measures a multi-stage process: image -> OCR text -> parsed chart -> answer.

Saved outputs under `runs/ocr/latest`:
- `config.json`: runtime arguments
- `ocr_raw.jsonl`: raw OCR engine outputs
- `ocr_tokens.jsonl`: normalized OCR tokens per image
- `parsed_charts.jsonl`: recovered chart structures
- `image_metrics.jsonl`: per-image OCR and parse diagnostics
- `qa_predictions.jsonl`: per-QA predictions and correctness
- `metrics_summary.json`: aggregated metrics
- `base_variant_comparison.json`: base-vs-variant accuracy deltas by variation group
- `report.md`: compact markdown report

### MM-LLM Direct-QA Baseline
Run:
```bash
python src/run.py \
  --dataset-dir datasets/out \
  --output-dir runs/mm_llm/latest
```

Principle:
- This baseline does not do OCR or explicit chart parsing.
- Each QA record is turned into one multimodal request containing exactly one chart image and one question.
- The prompt asks the model to return JSON only, with exactly one field: `{"answer": ...}`.
- The pipeline supports two backend families: OpenAI-compatible APIs and local Hugging Face multimodal models such as Qwen3-VL.

Execution flow:
- Load QA rows from `metadata.jsonl` and build one request per `sample_id`.
- Send the image and question to the configured backend.
- Cache raw responses by `sample_id` so reruns can reuse previous results.
- Parse the returned JSON answer, normalize it, and compare it with the ground truth answer.

Correctness and metrics:
- Numbers are normalized into the dataset format, so values like `31.333333` become `31.33`.
- Category answers are normalized to labels such as `Cat-3`.
- Trend questions are normalized to one of `increasing`, `decreasing`, or `fluctuating`.
- Final correctness is exact-match after this normalization step.
- This baseline measures an end-to-end process: image + question -> model answer.

Saved outputs under `runs/mm_llm/latest`:
- `config.json`: resolved runtime configuration
- `raw_responses.jsonl`: raw model responses and API payload metadata
- `qa_predictions.jsonl`: parsed predictions and correctness
- `metrics_summary.json`: aggregated accuracy and latency metrics
- `base_variant_comparison.json`: base-vs-variant accuracy deltas by variation group
- `report.md`: compact markdown report

### Baseline Difference
- `PaddleOCR` evaluates a structured perception pipeline and exposes where errors happen: OCR text loss, chart parsing failure, missing evidence, or wrong answer.
- `MM-LLM direct QA` evaluates the model end-to-end without an explicit intermediate chart representation.
- Running both is useful because they answer different questions: one measures recoverable chart structure, the other measures direct multimodal reasoning.
