"""Unified pipeline runner for OCR, VLM, and hybrid extraction."""

from __future__ import annotations

import numpy as np

from app.hybrid_pipeline import run_hybrid
from app.ocr_pipeline import run_ocr
from app.preprocessing import image_size, preprocess
from app.schemas import ExtractionResult, OcrEngine, PipelineName, TextBox
from app.utils import timer
from app.vlm_pipeline import run_vlm


def run_pipeline(
    img: np.ndarray,
    pipeline: PipelineName,
    ocr_engine: OcrEngine = "paddle",
) -> ExtractionResult:
    processed = preprocess(img)
    size = image_size(processed)

    with timer() as elapsed:
        if pipeline == "ocr":
            items: list[TextBox] = run_ocr(processed, engine=ocr_engine)
        elif pipeline == "vlm":
            items = run_vlm(processed)
        elif pipeline == "hybrid":
            items = run_hybrid(processed, ocr_engine=ocr_engine)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    return ExtractionResult(
        pipeline=pipeline,
        image_size=size,
        items=items,
        elapsed_ms=elapsed[0],
    )
