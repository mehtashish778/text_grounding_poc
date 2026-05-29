"""Universal tiling utilities.

This module provides:
- sliding-window iteration over an image
- shifting + stitching tile-level extraction results into full-image coords
- optional per-tile overlay saving for debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.config import (
    DEDUP_IOU_THRESHOLD,
    OUTPUTS_DIR,
    SAVE_TILE_OVERLAYS,
    TILE_OVERLAY_MAX_SAVES,
)
from app.schemas import TextBox
from app.visualization import draw_overlay


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    crop: np.ndarray


def iter_tiles(img: np.ndarray, tile_size_px: int, stride_ratio: float) -> Iterator[Tile]:
    """Yield tiles over `img` as Tile(x0, y0, crop). Coordinates are top-left in pixels."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return

    tile_w = min(tile_size_px, w)
    tile_h = min(tile_size_px, h)

    stride_w = max(1, int(tile_w * stride_ratio))
    stride_h = max(1, int(tile_h * stride_ratio))

    def starts(length: int, tile: int, stride: int) -> list[int]:
        if length <= tile:
            return [0]
        out = list(range(0, length - tile + 1, stride))
        last = length - tile
        if out[-1] != last:
            out.append(last)
        return out

    x_starts = starts(w, tile_w, stride_w)
    y_starts = starts(h, tile_h, stride_h)

    for y0 in y_starts:
        for x0 in x_starts:
            crop = img[y0 : y0 + tile_h, x0 : x0 + tile_w]
            yield Tile(x0=x0, y0=y0, crop=crop)


def _iou(a: TextBox, b: TextBox) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def stitch_tile_results(
    tile_results: list[tuple[int, int, list[TextBox]]],
    img_w: int,
    img_h: int,
    iou_threshold: float = DEDUP_IOU_THRESHOLD,
) -> list[TextBox]:
    """Shift each tile's boxes to full-image coords, then dedup by IoU."""

    all_items: list[TextBox] = []
    for x0, y0, items in tile_results:
        for item in items:
            x1, y1, x2, y2 = item.bbox
            shifted = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
            # clamp to image bounds
            shifted[0] = max(0, min(img_w, shifted[0]))
            shifted[2] = max(0, min(img_w, shifted[2]))
            shifted[1] = max(0, min(img_h, shifted[1]))
            shifted[3] = max(0, min(img_h, shifted[3]))
            if shifted[2] <= shifted[0] or shifted[3] <= shifted[1]:
                continue
            all_items.append(
                TextBox(
                    text=item.text,
                    bbox=[int(shifted[0]), int(shifted[1]), int(shifted[2]), int(shifted[3])],
                    confidence=item.confidence,
                    source=item.source,
                )
            )

    all_items.sort(
        key=lambda it: (it.bbox[2] - it.bbox[0]) * (it.bbox[3] - it.bbox[1]),
        reverse=True,
    )
    kept: list[TextBox] = []
    for item in all_items:
        if any(_iou(item, k) >= iou_threshold for k in kept):
            continue
        kept.append(item)
    return kept


def save_tile_overlay(
    tile_crop: np.ndarray,
    tile_items: list[TextBox],
    *,
    pipeline: str,
    x0: int,
    y0: int,
    tile_run_id: str,
    saved_counter: list[int],
) -> None:
    """Save a debug overlay for one tile crop (best-effort, capped)."""
    if not SAVE_TILE_OVERLAYS:
        return
    if saved_counter[0] >= TILE_OVERLAY_MAX_SAVES:
        return

    out_dir = OUTPUTS_DIR / "tile_overlays" / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{tile_run_id}_x{x0}_y{y0}.png"
    overlay = draw_overlay(tile_crop, tile_items)
    cv2.imwrite(str(out_path), overlay)

    saved_counter[0] += 1

