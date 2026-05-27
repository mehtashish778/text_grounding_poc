"""OCR extraction pipelines (PaddleOCR and EasyOCR)."""

from __future__ import annotations

import app.paddle_env  # noqa: F401 — must run before paddle/paddleocr import

from typing import Literal

import numpy as np

from app.schemas import OcrEngine, TextBox
from app.utils import poly_to_bbox

_paddle_ocr = None
_easy_reader = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        from app.paddle_env import apply_paddle_flags
        from paddleocr import PaddleOCR

        apply_paddle_flags()
        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            enable_mkldnn=False,
            use_gpu=False,
        )
    return _paddle_ocr


def _get_easy():
    global _easy_reader
    if _easy_reader is None:
        import easyocr

        from app.config import DEVICE

        _easy_reader = easyocr.Reader(["en"], gpu=DEVICE == "cuda")
    return _easy_reader


def run_paddle(img: np.ndarray) -> list[TextBox]:
    ocr = _get_paddle()
    result = ocr.ocr(img, cls=True)
    items: list[TextBox] = []
    if not result or not result[0]:
        return items

    for line in result[0]:
        poly, (text, conf) = line
        text = (text or "").strip()
        if not text:
            continue
        bbox = poly_to_bbox(poly)
        items.append(
            TextBox(
                text=text,
                bbox=bbox,
                confidence=float(conf),
                source="paddle",
            )
        )
    return items


def run_easy(img: np.ndarray) -> list[TextBox]:
    reader = _get_easy()
    result = reader.readtext(img)
    items: list[TextBox] = []
    for poly, text, conf in result:
        text = (text or "").strip()
        if not text:
            continue
        bbox = poly_to_bbox(poly)
        items.append(
            TextBox(
                text=text,
                bbox=bbox,
                confidence=float(conf),
                source="easy",
            )
        )
    return items


def run_ocr(img: np.ndarray, engine: OcrEngine = "paddle") -> list[TextBox]:
    if engine == "paddle":
        return run_paddle(img)
    if engine == "easy":
        return run_easy(img)
    raise ValueError(f"Unknown OCR engine: {engine}")
