# COMP-646 Project

This project builds and evaluates a synthetic chart question-answering dataset
for three chart types:

- bar chart
- line chart
- pie chart

The repository contains three executable workflows:

- dataset generation: `datasets/data.py`
- OCR + rule-based QA baseline: `PaddleOCR/run_ocr.py`
- multimodal GPT direct-QA baseline: `src/run.py`

There is also a root-level OCR-augmented GPT runner, `run.py`, which first runs
PaddleOCR and then sends image + OCR context to GPT-5.4 nano.

## Environment

Run all commands from the repository root.

```bash
conda create -n chartqa python=3.11 -y
conda activate chartqa
pip install -r requirements.txt
```

Check Paddle runtime:

```bash
python -c "import paddle; print('paddle:', paddle.__version__); print('gpu:', paddle.is_compiled_with_cuda())"
```

For GPT runs, set an OpenAI API key first.

PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

Bash:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Do not commit real API keys. Runtime `config.json` files should store
`"api_key": "<set>"`, not the raw key.

## Generate Dataset

The normal-size dataset is `datasets/out_3000`. It contains 10000 QA records
and is the dataset size to use for the main experiments:

- 125 base charts per chart type
- 8 rendered style variants per base chart
- split: 6960 train, 1520 val, 1520 test

Command used to generate `datasets/out_3000`:

```bash
python datasets/data.py --output-dir datasets/out_3000 --num-per-type 125 --style-variants-per-base 8 --skip-reference-style --seed 42 --image-format png
```

The smaller `datasets/out_504` dataset is for quick runs when we need results
fast. It contains 504 images and 1680 QA records:

- 84 base charts: 28 bar, 28 line, 28 pie
- 6 variants per base chart: `base` plus five post-processing variants
- split: 1200 train, 240 val, 240 test

Command used to generate `datasets/out_504`:

```bash
python datasets/data.py --output-dir datasets/out_504 --num-per-type 28 --style-variants-per-base 0 --variants-config configs/variants.example.json --seed 42 --image-format jpg
```

The post-processing variants are defined in
`configs/variants.example.json`:

- `blur_gaussian_r1`
- `blur_gaussian_r2`
- `noise_gaussian_s5`
- `compression_jpeg_q50`
- `blur_r2_noise_s5`

## PaddleOCR Baseline

This baseline runs OCR once per image, parses the chart structure from OCR
tokens, answers the QA task with rules, and evaluates exact-match accuracy.

Run PaddleOCR on the normal-size `out_3000` dataset:

```bash
python PaddleOCR/run_ocr.py --dataset-dir datasets/out_3000 --output-dir ocr_out_3000_all --split all --text-det-limit-side-len 960 --text-det-limit-type max
```

Run PaddleOCR on the quick `out_504` dataset:

```bash
python PaddleOCR/run_ocr.py --dataset-dir datasets/out_504 --output-dir ocr_out_504_all --split all --reuse-ocr-cache --text-det-limit-side-len 960 --text-det-limit-type max
```

Run only the test split:

```bash
python PaddleOCR/run_ocr.py --dataset-dir datasets/out_504 --output-dir ocr_out_504_test --split test --reuse-ocr-cache --text-det-limit-side-len 960 --text-det-limit-type max
```

Run with OCR visualization images:

```bash
python PaddleOCR/run_ocr.py --dataset-dir datasets/out_504 --output-dir ocr_out_504_test_vis --split test --save-vis --text-det-limit-side-len 960 --text-det-limit-type max
```

Main outputs:

- `config.json`
- `ocr_raw.jsonl`
- `ocr_tokens.jsonl`
- `parsed_charts.jsonl`
- `image_metrics.jsonl`
- `qa_predictions.jsonl`
- `metrics_summary.json`
- `base_variant_comparison.json`
- `report.md`

Latest full `out_504` OCR result:

- output: `ocr_out_504_all`
- images: 504
- QA records: 1680
- QA accuracy: 0.7583

## GPT-5.4 Nano Direct-QA Baseline

This is the `src/run.py` workflow. It sends each chart image and question
directly to the multimodal model. It does not use OCR context.

Run `gpt-5.4-nano` on the quick `out_504` test split:

```bash
python src/run.py --dataset-dir datasets/out_504 --output-dir mm_llm_out_504_gpt54nano_test --split test --backend openai_compatible --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --no-reuse-cache
```

Run `gpt-5.4-nano` on the quick `out_504` full dataset:

```bash
python src/run.py --dataset-dir datasets/out_504 --output-dir mm_llm_out_504_gpt54nano_all --split all --backend openai_compatible --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --reuse-cache
```

If the test split has already been run and you want to reuse those cached
responses before running `all`, copy the cache first:

```bash
mkdir -p mm_llm_out_504_gpt54nano_all
cp mm_llm_out_504_gpt54nano_test/raw_responses.jsonl mm_llm_out_504_gpt54nano_all/raw_responses.jsonl
python src/run.py --dataset-dir datasets/out_504 --output-dir mm_llm_out_504_gpt54nano_all --split all --backend openai_compatible --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --reuse-cache
```

PowerShell equivalent for the cache copy:

```powershell
New-Item -ItemType Directory -Force -Path mm_llm_out_504_gpt54nano_all
Copy-Item mm_llm_out_504_gpt54nano_test\raw_responses.jsonl mm_llm_out_504_gpt54nano_all\raw_responses.jsonl -Force
python src/run.py --dataset-dir datasets/out_504 --output-dir mm_llm_out_504_gpt54nano_all --split all --backend openai_compatible --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --reuse-cache
```

Main outputs:

- `config.json`
- `raw_responses.jsonl`
- `qa_predictions.jsonl`
- `metrics_summary.json`
- `base_variant_comparison.json`
- `report.md`

Latest full `out_504` GPT direct-QA result:

- output: `mm_llm_out_504_gpt54nano_all`
- QA records: 1680
- QA accuracy: 0.8548
- average latency: 1563.34 ms
- request errors: 0

Latest test `out_504` GPT direct-QA result:

- output: `mm_llm_out_504_gpt54nano_test`
- QA records: 240
- QA accuracy: 0.8417
- request errors: 0

## OCR-Augmented GPT Runner

This is the root-level `run.py` workflow. It runs OCR first, builds a compact
OCR context, and sends image + question + OCR context to GPT-5.4 nano.

Run on the quick `out_504` test split:

```bash
python run.py --dataset-dir datasets/out_504 --output-dir ocr_gpt54nano_out_504_test --split test --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --min-ocr-score 0.3 --max-ocr-tokens 80 --reuse-ocr-cache --reuse-response-cache --text-det-limit-side-len 960 --text-det-limit-type max
```

Run on the quick `out_504` full dataset:

```bash
python run.py --dataset-dir datasets/out_504 --output-dir ocr_gpt54nano_out_504_all --split all --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --min-ocr-score 0.3 --max-ocr-tokens 80 --reuse-ocr-cache --reuse-response-cache --text-det-limit-side-len 960 --text-det-limit-type max
```

OCR-augmented GPT runs on the normal-size `out_3000` dataset:

```bash
python run.py --dataset-dir datasets/out_3000 --output-dir ocr_gpt54nano_test --split test --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --min-ocr-score 0.3 --max-ocr-tokens 80 --reuse-ocr-cache --reuse-response-cache --text-det-limit-side-len 960 --text-det-limit-type max
```

```bash
python run.py --dataset-dir datasets/out_3000 --output-dir ocr_gpt54nano_val --split val --model-name gpt-5.4-nano --api-base-url https://api.openai.com/v1 --temperature 0 --max-tokens 128 --timeout-seconds 60 --min-ocr-score 0.3 --max-ocr-tokens 80 --reuse-ocr-cache --reuse-response-cache --text-det-limit-side-len 960 --text-det-limit-type max
```

## Useful Checks

Count predictions:

```bash
python -c "from pathlib import Path; print(sum(1 for _ in Path('mm_llm_out_504_gpt54nano_all/qa_predictions.jsonl').open(encoding='utf-8')))"
```

Read the full GPT summary:

```bash
python -m json.tool mm_llm_out_504_gpt54nano_all/metrics_summary.json
```

Read the full OCR summary:

```bash
python -m json.tool ocr_out_504_all/metrics_summary.json
```

## Output Directories Currently Used

- `datasets/out_3000`: normal-size dataset
- `datasets/out_504`: quick 504-image dataset for faster runs
- `ocr_out_504_all`: PaddleOCR full-dataset baseline
- `mm_llm_out_504_gpt54nano_test`: GPT direct-QA test split
- `mm_llm_out_504_gpt54nano_all`: GPT direct-QA full dataset
- `ocr_gpt54nano_test`: OCR-augmented GPT run on `datasets/out_3000` test
- `ocr_gpt54nano_val`: OCR-augmented GPT run on `datasets/out_3000` val

## Baseline Difference

- PaddleOCR baseline evaluates `image -> OCR text -> parsed chart -> answer`.
- GPT direct-QA evaluates `image + question -> answer`.
- OCR-augmented GPT evaluates `image + OCR context + question -> answer`.

These three workflows answer different questions, so their reports should be
kept separate.
