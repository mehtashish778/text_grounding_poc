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

# OCR defaults
DEFAULT_OCR_ENGINE = os.getenv("DEFAULT_OCR_ENGINE", "paddle")

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
