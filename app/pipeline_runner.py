"""Unified pipeline runner for OCR, VLM, and hybrid extraction.

All pipelines use the same tiling loop. Each tile is processed as a separate
request (no VLM batching) to keep GPU memory usage bounded.
"""

from __future__ import annotations

import time

import numpy as np

from app.config import (
    ENABLE_REASONING_FILTER,
    QWEN3_TEXT_MODEL_ID,
    REASONING_INTERMEDIATE_MAX_ITEMS,
    SAVE_REASONING_INTERMEDIATE,
    STRIDE_RATIO,
    TILE_SIZE_PX,
)
from app.model_manager import clear_gpu_cache, unload_all_vlm_models
from app.hybrid_pipeline import run_hybrid
from app.ocr_pipeline import run_ocr
from app.preprocessing import image_size, preprocess
from app.schemas import ExtractionResult, OcrEngine, PipelineName, TextBox
from app.tiling import iter_tiles, save_tile_overlay, stitch_tile_results
from app.utils import timer
from app.vlm_pipeline import run_vlm_tile


def run_pipeline(
    img: np.ndarray,
    pipeline: PipelineName,
    ocr_engine: OcrEngine = "paddle",
) -> ExtractionResult:
    processed = preprocess(img)
    size = image_size(processed)
    img_w, img_h = size

    tile_run_id = str(int(timer_now_ms()))
    saved_counter = [0]
    tile_results: list[tuple[int, int, list[TextBox]]] = []

    tiles = list(iter_tiles(processed, TILE_SIZE_PX, STRIDE_RATIO))
    total_tiles = len(tiles)

    with timer() as elapsed:
        for idx, tile in enumerate(tiles, start=1):
            if pipeline in ("vlm", "hybrid") and idx == 1:
                unload_all_vlm_models()  # clean slate before loading VLM

            if pipeline == "ocr":
                tile_items = run_ocr(tile.crop, engine=ocr_engine)
            elif pipeline == "vlm":
                if idx % 10 == 1 or idx == total_tiles:
                    print(f"    VLM tile {idx}/{total_tiles}...")
                tile_items = run_vlm_tile(tile.crop)
            elif pipeline == "hybrid":
                tile_items = run_hybrid(tile.crop, ocr_engine=ocr_engine)
            else:
                raise ValueError(f"Unknown pipeline: {pipeline}")

            tile_results.append((tile.x0, tile.y0, tile_items))
            save_tile_overlay(
                tile.crop,
                tile_items,
                pipeline=pipeline,
                x0=tile.x0,
                y0=tile.y0,
                tile_run_id=tile_run_id,
                saved_counter=saved_counter,
            )

            if pipeline == "vlm":
                clear_gpu_cache()

    items = stitch_tile_results(
        tile_results=tile_results,
        img_w=img_w,
        img_h=img_h,
    )

    intermediate: dict | None = None
    if ENABLE_REASONING_FILTER and pipeline in ("vlm", "hybrid"):
        pre_items = items
        pre_n = len(pre_items)
        unload_all_vlm_models()
        from app.reasoning_filter import filter_items, load_filter_prompt, unload_reasoning_model

        print(f"    Running reasoning filter on {pre_n} items...")
        t0 = time.perf_counter()
        items = filter_items(pre_items, load_filter_prompt())
        reasoning_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        post_n = len(items)
        print(f"    Reasoning filter: kept {post_n}/{pre_n} items ({reasoning_ms} ms)")

        if SAVE_REASONING_INTERMEDIATE:
            # Cap stored items to avoid massive JSONs in extreme cases.
            capped_pre = pre_items[: max(0, REASONING_INTERMEDIATE_MAX_ITEMS)]
            intermediate = {
                "reasoning_filter": {
                    "enabled": True,
                    "model_id": QWEN3_TEXT_MODEL_ID,
                    "pre_num_items": pre_n,
                    "post_num_items": post_n,
                    "reasoning_filter_ms": reasoning_ms,
                    "pre_items_truncated": pre_n > len(capped_pre),
                    "pre_items": [it.model_dump() for it in capped_pre],
                }
            }
        unload_reasoning_model()

    return ExtractionResult(
        pipeline=pipeline,
        image_size=size,
        items=items,
        elapsed_ms=elapsed[0],
        intermediate=intermediate,
    )


def timer_now_ms() -> float:
    import time

    return time.time() * 1000.0
