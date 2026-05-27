# P&ID Text Grounding POC

Proof of concept for extracting engineering tags with bounding boxes from P&ID documents. Compares three pipelines:

- **OCR-only** — PaddleOCR or EasyOCR
- **VLM-only** — Florence-2 (`<OCR_WITH_REGION>`)
- **Hybrid** — OCR localization + Florence-2 merge on clustered fragments

## Prerequisites

- Python 3.10+
- (Recommended) NVIDIA GPU with CUDA for Florence-2
- **Poppler** for PDF support ([poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) — add `bin/` to PATH on Windows)

## Setup

```bash
cd text_grounding_poc
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch with CUDA 12.4 (`cu124`). For a different CUDA version, change the `--extra-index-url` line per [pytorch.org](https://pytorch.org/get-started/locally/) (e.g. `cu121`, `cu118`).

Place sample P&ID files in `samples/` (PNG, JPG, or PDF).

## Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/docs for Swagger UI.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /extract` | Extract text boxes as JSON |
| `POST /visualize` | Return PNG with bounding box overlay |
| `POST /compare` | Run OCR, VLM, and hybrid; save JSON + overlays to `outputs/` |

### Query parameters

- `pipeline`: `ocr` \| `vlm` \| `hybrid` (default: `hybrid`)
- `ocr_engine`: `paddle` \| `easy` (default: `paddle`)

### Example (curl)

```bash
curl -X POST "http://127.0.0.1:8000/extract?pipeline=hybrid&ocr_engine=paddle" \
  -F "file=@samples/your_pid.png"
```

```bash
curl -X POST "http://127.0.0.1:8000/compare?ocr_engine=paddle" \
  -F "file=@samples/your_pid.png"
```

## Batch evaluation

Runs all three pipelines on every file in `samples/` and writes metrics + overlays:

```bash
python -m app.evaluate samples/
```

Outputs:

- `outputs/metrics.csv` — timing, box counts, confidence (manual quality columns left blank)
- `outputs/{file}_{pipeline}.json` — structured extraction
- `outputs/{file}_{pipeline}.png` — visualization

## Output format

```json
{
  "pipeline": "hybrid",
  "image_size": [2400, 1800],
  "items": [
    {
      "text": "P-101A",
      "bbox": [120, 220, 180, 250],
      "confidence": 0.95,
      "source": "hybrid"
    }
  ],
  "elapsed_ms": 1234.5
}
```

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLORENCE_MODEL_ID` | `microsoft/Florence-2-base` | Hugging Face model id |
| `DEVICE` | `cuda` if available else `cpu` | Inference device |
| `DEFAULT_OCR_ENGINE` | `paddle` | Default OCR backend |
| `MAX_IMAGE_DIM` | `2400` | Max longest image side |
| `ENABLE_ENHANCE` | `false` | CLAHE + denoise preprocessing |
| `HORIZONTAL_GAP_RATIO` | `0.5` | Hybrid cluster horizontal gap threshold |
| `VERTICAL_OVERLAP_RATIO` | `0.6` | Hybrid cluster vertical overlap threshold |

## Project layout

```text
app/
  main.py              # FastAPI
  pipeline_runner.py   # Unified runner
  ocr_pipeline.py      # PaddleOCR / EasyOCR
  vlm_pipeline.py      # Florence-2
  hybrid_pipeline.py   # OCR + VLM merge
  preprocessing.py
  visualization.py
  evaluate.py          # CLI metrics
samples/               # Your P&ID inputs
outputs/               # JSON, PNG, metrics.csv
```

## Troubleshooting (PaddleOCR on Windows)

If you see `OneDnnContext does not have the input Filter` / `fused_conv2d` errors, you likely have **PaddlePaddle 3.3+** installed. Reinstall the pinned stack:

```bash
uv pip install "paddlepaddle==2.6.2" "paddleocr==2.7.3" "numpy>=1.24,<2"
```

`requirements.txt` pins these versions for CPU inference stability on Windows.

## Notes

- First VLM request downloads Florence-2 weights (~500MB for base).
- Hybrid pipeline clusters nearby OCR fragments and re-reads each crop with Florence-2 `<OCR>` to merge tags like `P` + `-101` + `A` → `P-101A`.
- For production, consider async job queues and model warm-up on startup.
