"""Shared utility helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import numpy as np


def poly_to_bbox(poly: list | np.ndarray) -> list[int]:
    """Convert 4-point polygon to axis-aligned [x1, y1, x2, y2]."""
    arr = np.asarray(poly, dtype=np.float32)
    xs = arr[:, 0]
    ys = arr[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def union_bbox(boxes: list[list[int]]) -> list[int]:
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def bbox_width(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0])


def bbox_height(bbox: list[int]) -> int:
    return max(0, bbox[3] - bbox[1])


def vertical_overlap_ratio(a: list[int], b: list[int]) -> float:
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    min_h = min(bbox_height(a), bbox_height(b))
    if min_h <= 0:
        return 0.0
    return overlap / min_h


def horizontal_gap(a: list[int], b: list[int]) -> float:
    """Gap between two boxes on x-axis (0 if overlapping)."""
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0


@contextmanager
def timer() -> Generator[list[float], None, None]:
    elapsed: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed[0] = (time.perf_counter() - start) * 1000.0
