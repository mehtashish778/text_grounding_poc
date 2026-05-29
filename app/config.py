"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# Florence-2
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-base")
DEVICE = os.getenv("DEVICE", "cuda" if _has_cuda() else "cpu")

# Unified tiling config (applies to OCR/VLM/Hybrid)
TILE_SIZE_PX = int(os.getenv("TILE_SIZE_PX", "400"))
STRIDE_RATIO = float(os.getenv("STRIDE_RATIO", "0.2"))  # 0.2 => 20% stride
DEDUP_IOU_THRESHOLD = float(os.getenv("DEDUP_IOU_THRESHOLD", "0.5"))

# Save intermediate per-tile overlays for debugging/inspection.
# These can get many files, so we cap them.
SAVE_TILE_OVERLAYS = os.getenv("SAVE_TILE_OVERLAYS", "true").lower() in (
    "1",
    "true",
    "yes",
)
TILE_OVERLAY_MAX_SAVES = int(os.getenv("TILE_OVERLAY_MAX_SAVES", "60"))

# Tile batching size for GPU VLM models (1 = one tile/request at a time; safest for VRAM)
TILE_BATCH_SIZE = int(os.getenv("TILE_BATCH_SIZE", "1"))
# Beam search width for Florence-2 generation (1 = greedy, lowest VRAM)
VLM_NUM_BEAMS = int(os.getenv("VLM_NUM_BEAMS", "1"))
# Max tokens generated per tile (lower = less VRAM during decode)
VLM_MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "256"))

# Qwen3 text-only reasoning filter (post-Florence false-positive removal)
ENABLE_REASONING_FILTER = os.getenv("ENABLE_REASONING_FILTER", "false").lower() in (
    "1",
    "true",
    "yes",
)
_qwen3_text_model = os.getenv("QWEN3_TEXT_MODEL_ID", "Qwen/Qwen3-4B")
# Invalid HF repo id from earlier docs; map to the real model card.
if _qwen3_text_model in ("Qwen/Qwen3-4B-Instruct",):
    _qwen3_text_model = "Qwen/Qwen3-4B"
QWEN3_TEXT_MODEL_ID = _qwen3_text_model
# Optional: set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your shell (never commit tokens).
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
REASONING_FILTER_PROMPT_FILE = Path(
    os.getenv(
        "REASONING_FILTER_PROMPT_FILE",
        str(PROJECT_ROOT / "app" / "filter_prompt.txt"),
    )
)
REASONING_MAX_NEW_TOKENS = int(os.getenv("REASONING_MAX_NEW_TOKENS", "2048"))
SAVE_REASONING_INTERMEDIATE = os.getenv("SAVE_REASONING_INTERMEDIATE", "true").lower() in (
    "1",
    "true",
    "yes",
)
REASONING_INTERMEDIATE_MAX_ITEMS = int(os.getenv("REASONING_INTERMEDIATE_MAX_ITEMS", "4000"))

# OCR defaults
DEFAULT_OCR_ENGINE = os.getenv("DEFAULT_OCR_ENGINE", "paddle")

# Backwards-compatible aliases (older config variable names)
VLM_TILE_SIZE_PX = TILE_SIZE_PX
VLM_STRIDE_RATIO = STRIDE_RATIO
VLM_DEDUP_IOU_THRESHOLD = DEDUP_IOU_THRESHOLD
VLM_SAVE_TILE_OVERLAYS = SAVE_TILE_OVERLAYS
VLM_TILE_OVERLAY_MAX_SAVES = TILE_OVERLAY_MAX_SAVES

# Preprocessing
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "2400"))
ENABLE_ENHANCE = os.getenv("ENABLE_ENHANCE", "false").lower() in ("1", "true", "yes")
PDF_DPI = int(os.getenv("PDF_DPI", "200"))

# Hybrid clustering thresholds
HORIZONTAL_GAP_RATIO = float(os.getenv("HORIZONTAL_GAP_RATIO", "0.5"))
VERTICAL_OVERLAP_RATIO = float(os.getenv("VERTICAL_OVERLAP_RATIO", "0.6"))
CLUSTER_PADDING_PX = int(os.getenv("CLUSTER_PADDING_PX", "4"))
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "2"))

# Visualization colors (BGR for OpenCV)
SOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "paddle": (0, 200, 0),
    "easy": (255, 120, 0),
    "florence": (0, 0, 255),
    "hybrid": (0, 165, 255),
}


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR
