"""Pydantic models for extraction requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OcrEngine = Literal["paddle", "easy"]
# Source name is used for visualization color-coding and comparison.
PipelineName = Literal["ocr", "vlm", "hybrid"]
SourceName = Literal["paddle", "easy", "florence", "hybrid"]


class TextBox(BaseModel):
    text: str
    bbox: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceName


class ExtractionResult(BaseModel):
    pipeline: PipelineName
    image_size: tuple[int, int]
    items: list[TextBox]
    elapsed_ms: float
    # Optional debug metadata for analysis (e.g., reasoning filter deltas).
    intermediate: dict | None = None


class CompareResult(BaseModel):
    filename: str
    image_size: tuple[int, int]
    results: dict[str, ExtractionResult]
