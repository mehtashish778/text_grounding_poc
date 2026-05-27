"""OCR + Florence-2 hybrid pipeline with spatial clustering."""

from __future__ import annotations

import numpy as np

from app.config import (
    CLUSTER_PADDING_PX,
    HORIZONTAL_GAP_RATIO,
    MIN_CLUSTER_SIZE,
    VERTICAL_OVERLAP_RATIO,
)
from app.ocr_pipeline import run_ocr
from app.preprocessing import bgr_to_pil, crop_with_padding
from app.schemas import OcrEngine, TextBox
from app.utils import bbox_width, horizontal_gap, union_bbox, vertical_overlap_ratio
from app.vlm_pipeline import run_ocr_on_crop


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _median_char_width(items: list[TextBox]) -> float:
    widths = [bbox_width(item.bbox) / max(len(item.text), 1) for item in items]
    if not widths:
        return 10.0
    widths.sort()
    return widths[len(widths) // 2]


def cluster_fragments(items: list[TextBox]) -> list[list[int]]:
    """Group fragment indices that likely belong to the same engineering tag."""
    n = len(items)
    if n == 0:
        return []

    uf = _UnionFind(n)
    median_cw = _median_char_width(items)
    max_gap = median_cw * HORIZONTAL_GAP_RATIO

    sorted_indices = sorted(range(n), key=lambda i: (items[i].bbox[1], items[i].bbox[0]))

    for idx_a in range(n):
        for idx_b in range(idx_a + 1, n):
            a, b = items[idx_a], items[idx_b]
            if vertical_overlap_ratio(a.bbox, b.bbox) < VERTICAL_OVERLAP_RATIO:
                continue
            gap = horizontal_gap(a.bbox, b.bbox)
            if gap <= max_gap:
                uf.union(idx_a, idx_b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    return [sorted(g, key=lambda i: items[i].bbox[0]) for g in groups.values()]


def run_hybrid(
    img: np.ndarray,
    ocr_engine: OcrEngine = "paddle",
) -> list[TextBox]:
    fragments = run_ocr(img, engine=ocr_engine)
    if not fragments:
        return []

    clusters = cluster_fragments(fragments)
    merged_indices: set[int] = set()
    results: list[TextBox] = []

    for cluster_idxs in clusters:
        if len(cluster_idxs) < MIN_CLUSTER_SIZE:
            continue

        cluster_items = [fragments[i] for i in cluster_idxs]
        merged_bbox = union_bbox([item.bbox for item in cluster_items])
        crop = crop_with_padding(img, merged_bbox, padding=CLUSTER_PADDING_PX)
        if crop.size == 0:
            continue

        pil_crop = bgr_to_pil(crop)
        merged_text = run_ocr_on_crop(pil_crop)
        if not merged_text:
            merged_text = "".join(item.text for item in cluster_items)

        avg_conf = sum(item.confidence for item in cluster_items) / len(cluster_items)
        results.append(
            TextBox(
                text=merged_text,
                bbox=merged_bbox,
                confidence=avg_conf,
                source="hybrid",
            )
        )
        merged_indices.update(cluster_idxs)

    for i, item in enumerate(fragments):
        if i not in merged_indices:
            results.append(
                TextBox(
                    text=item.text,
                    bbox=item.bbox,
                    confidence=item.confidence,
                    source="hybrid",
                )
            )

    return results
