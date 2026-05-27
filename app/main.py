"""FastAPI application for P&ID text grounding POC."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.config import DEFAULT_OCR_ENGINE, OUTPUTS_DIR, ensure_outputs_dir
from app.pipeline_runner import run_pipeline
from app.preprocessing import bytes_to_image, load_input, preprocess
from app.schemas import CompareResult, ExtractionResult, OcrEngine, PipelineName
from app.visualization import draw_overlay, save_overlay

app = FastAPI(
    title="P&ID Text Grounding POC",
    description="Extract engineering tags with bounding boxes from P&ID documents.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    ensure_outputs_dir()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _read_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix == ".pdf":
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        images = load_input(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        if not images:
            raise HTTPException(status_code=400, detail="PDF produced no pages")
        return images[0]
    return bytes_to_image(data)


@app.post("/extract", response_model=ExtractionResult)
async def extract(
    file: UploadFile = File(...),
    pipeline: PipelineName = Query("hybrid"),
    ocr_engine: OcrEngine = Query(DEFAULT_OCR_ENGINE),
) -> ExtractionResult:
    img = _read_upload(file)
    return run_pipeline(img, pipeline=pipeline, ocr_engine=ocr_engine)


@app.post("/visualize")
async def visualize(
    file: UploadFile = File(...),
    pipeline: PipelineName = Query("hybrid"),
    ocr_engine: OcrEngine = Query(DEFAULT_OCR_ENGINE),
):
    img = _read_upload(file)
    result = run_pipeline(img, pipeline=pipeline, ocr_engine=ocr_engine)
    processed = preprocess(img)
    overlay = draw_overlay(processed, result.items)

    stem = Path(file.filename or "upload").stem
    out_path = OUTPUTS_DIR / f"{stem}_{pipeline}.png"
    save_overlay(processed, result.items, out_path)

    _, buf = cv2.imencode(".png", overlay)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")


@app.post("/compare", response_model=CompareResult)
async def compare(
    file: UploadFile = File(...),
    ocr_engine: OcrEngine = Query(DEFAULT_OCR_ENGINE),
) -> CompareResult:
    img = _read_upload(file)
    processed = preprocess(img)
    stem = Path(file.filename or "upload").stem
    results: dict[str, ExtractionResult] = {}

    for pipeline in ("ocr", "vlm", "hybrid"):
        result = run_pipeline(img, pipeline=pipeline, ocr_engine=ocr_engine)
        results[pipeline] = result

        json_path = OUTPUTS_DIR / f"{stem}_{pipeline}.json"
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        save_overlay(processed, result.items, OUTPUTS_DIR / f"{stem}_{pipeline}.png")

    first = results["ocr"]
    return CompareResult(
        filename=file.filename or "upload",
        image_size=first.image_size,
        results=results,
    )
