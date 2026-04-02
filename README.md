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
  --style-variants-per-base 8
```

## Next Step
The next milestone is to run **PaddleOCR benchmark** on the generated dataset:
- build OCR inference pipeline over dataset images
- collect OCR outputs in a unified result format
- report task-wise and chart-wise OCR performance
