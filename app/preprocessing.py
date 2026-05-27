"""Image and PDF preprocessing for P&ID documents."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from app.config import ENABLE_ENHANCE, MAX_IMAGE_DIM, PDF_DPI

PathLike = Union[str, Path]


def load_image(path: PathLike) -> np.ndarray:
    """Load image file as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def pdf_to_images(path: PathLike, dpi: int = PDF_DPI) -> list[np.ndarray]:
    """Convert PDF pages to BGR numpy arrays."""
    from pdf2image import convert_from_path

    pil_pages = convert_from_path(str(path), dpi=dpi)
    images: list[np.ndarray] = []
    for page in pil_pages:
        rgb = np.array(page.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        images.append(bgr)
    return images


def load_input(path: PathLike) -> list[np.ndarray]:
    """Load PDF (multi-page) or single image file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_images(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return [load_image(path)]
    raise ValueError(f"Unsupported file type: {suffix}")


def bytes_to_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def resize_if_huge(img: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance(img: np.ndarray) -> np.ndarray:
    """Optional contrast enhancement and denoising."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Apply standard preprocessing pipeline."""
    out = resize_if_huge(img)
    if ENABLE_ENHANCE:
        out = enhance(out)
    return out


def image_size(img: np.ndarray) -> tuple[int, int]:
    h, w = img.shape[:2]
    return (w, h)


def crop_with_padding(
    img: np.ndarray,
    bbox: list[int],
    padding: int = 4,
) -> np.ndarray:
    """Crop region with padding, clamped to image bounds."""
    h, w = img.shape[:2]
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)
    return img[y1:y2, x1:x2].copy()


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
