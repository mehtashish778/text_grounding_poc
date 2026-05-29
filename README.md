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

Copy `sample.env` to `.env` and edit as needed:

```bash
copy sample.env .env   # Windows
# cp sample.env .env   # Linux/macOS
```

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

Runs one or more pipelines on every file in `samples/` and writes metrics + overlays.

### Commands

Run **all** pipelines (default: `ocr`, `vlm`, `hybrid`):

```bash
python -m app.evaluate samples/
```

Run a **single** pipeline:

```bash
python -m app.evaluate samples/ --pipeline vlm
python -m app.evaluate samples/ --pipeline ocr --ocr-engine paddle
python -m app.evaluate samples/ --pipeline hybrid --ocr-engine easy
```

Run **multiple** pipelines:

```bash
python -m app.evaluate samples/ --pipeline ocr vlm
```

**VLM + Qwen3 reasoning filter** — set env vars, then evaluate (PowerShell):

```powershell
$env:ENABLE_REASONING_FILTER = "true"
$env:QWEN3_TEXT_MODEL_ID = "Qwen/Qwen3-4B"
$env:HF_TOKEN = "hf_..."   # only if Hugging Face requires authentication
python -m app.evaluate samples\ --pipeline vlm
```

Same via `.env` (copy from `sample.env`, set `ENABLE_REASONING_FILTER=true`, then):

```bash
python -m app.evaluate samples/ --pipeline vlm
```

Linux/macOS (bash):

```bash
export ENABLE_REASONING_FILTER=true
export QWEN3_TEXT_MODEL_ID=Qwen/Qwen3-4B
export HF_TOKEN=hf_...   # optional
python -m app.evaluate samples/ --pipeline vlm
```

### CLI options

| Argument | Default | Description |
|----------|---------|-------------|
| `samples_dir` | `samples/` | Folder of P&ID images (PNG, JPG, PDF) |
| `--pipeline` | `ocr vlm hybrid` | One or more of: `ocr`, `vlm`, `hybrid` |
| `--ocr-engine` | `paddle` (from env) | `paddle` or `easy` (for `ocr` and `hybrid` only) |

### Outputs

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

Configuration is read from a `.env` file at the project root (via `python-dotenv` in `app/config.py`). Copy [`sample.env`](sample.env) to `.env` and edit values there, or export variables in your shell (shell values override `.env`).

**Boolean flags** (`ENABLE_*`, `SAVE_*`): use `true`, `false`, `1`, `0`, `yes`, or `no` (case-insensitive).

### Florence-2 and device

| Variable | Default | Description |
|----------|---------|-------------|
| `FLORENCE_MODEL_ID` | `microsoft/Florence-2-base` | Hugging Face model id for the VLM pipeline |
| `DEVICE` | `cuda` if available, else `cpu` | Inference device (`cuda` or `cpu`). Omit or leave unset in `.env` to auto-detect |

### Tiling (OCR, VLM, and hybrid)

Large P&IDs are processed in overlapping tiles. These settings apply to all tiled pipelines.

| Variable | Default | Description |
|----------|---------|-------------|
| `TILE_SIZE_PX` | `400` | Width and height of each tile in pixels |
| `STRIDE_RATIO` | `0.2` | Stride as a fraction of tile size (`0.2` = 20% overlap between adjacent tiles) |
| `DEDUP_IOU_THRESHOLD` | `0.5` | IoU threshold for merging duplicate boxes across tile boundaries |

### Tile debug overlays

| Variable | Default | Description |
|----------|---------|-------------|
| `SAVE_TILE_OVERLAYS` | `true` | Write per-tile PNG overlays under `outputs/tile_overlays/` |
| `TILE_OVERLAY_MAX_SAVES` | `60` | Maximum number of tile overlay images to save per run (avoids huge disk use) |

### VLM inference (VRAM tuning)

| Variable | Default | Description |
|----------|---------|-------------|
| `TILE_BATCH_SIZE` | `1` | Number of tiles processed per forward pass (`1` is safest for GPU memory) |
| `VLM_NUM_BEAMS` | `1` | Beam search width for Florence-2 (`1` = greedy, lowest VRAM) |
| `VLM_MAX_NEW_TOKENS` | `256` | Max tokens generated per tile (lower reduces decode memory) |

### Qwen3 reasoning filter (optional)

Post-processing step that uses a text-only Qwen3 model to drop false-positive VLM detections. Enable for the VLM pipeline when evaluating or via API.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_REASONING_FILTER` | `false` | Run Qwen3 filter after Florence-2 extraction |
| `QWEN3_TEXT_MODEL_ID` | `Qwen/Qwen3-4B` | Hugging Face model id for the reasoning filter |
| `HF_TOKEN` | *(unset)* | Hugging Face API token (for gated models). **Never commit** |
| `HUGGINGFACE_HUB_TOKEN` | *(unset)* | Alias for `HF_TOKEN` (either one is used) |
| `REASONING_FILTER_PROMPT_FILE` | `app/filter_prompt.txt` | Path to the filter system/user prompt template |
| `REASONING_MAX_NEW_TOKENS` | `2048` | Max tokens for Qwen3 generation per batch |
| `SAVE_REASONING_INTERMEDIATE` | `true` | Save intermediate filter inputs/outputs for debugging |
| `REASONING_INTERMEDIATE_MAX_ITEMS` | `4000` | Cap on items written to reasoning debug artifacts |

See [Batch evaluation](#batch-evaluation) for example commands.

### OCR

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_OCR_ENGINE` | `paddle` | Default OCR backend: `paddle` or `easy` (API query param and `--ocr-engine` can override) |

### Preprocessing

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_IMAGE_DIM` | `2400` | Resize so the longest image side does not exceed this value |
| `ENABLE_ENHANCE` | `false` | Apply CLAHE contrast enhancement and denoising before inference |
| `PDF_DPI` | `200` | Rasterization DPI when converting PDF inputs to images |

### Hybrid clustering

Used when merging nearby OCR fragments before Florence-2 re-reads each cluster.

| Variable | Default | Description |
|----------|---------|-------------|
| `HORIZONTAL_GAP_RATIO` | `0.5` | Max horizontal gap between boxes (as a multiple of average box height) to cluster |
| `VERTICAL_OVERLAP_RATIO` | `0.6` | Minimum vertical overlap ratio required to cluster boxes on the same line |
| `CLUSTER_PADDING_PX` | `4` | Extra padding in pixels around each cluster crop |
| `MIN_CLUSTER_SIZE` | `2` | Minimum number of OCR boxes required to form a cluster |

### Quick reference

A ready-to-copy template with all keys and defaults is in [`sample.env`](sample.env):

```bash
copy sample.env .env   # Windows
# cp sample.env .env   # Linux/macOS
```

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
