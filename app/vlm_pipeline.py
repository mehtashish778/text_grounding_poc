"""Florence-2 VLM extraction pipeline."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.config import DEVICE, FLORENCE_MODEL_ID, VLM_MAX_NEW_TOKENS, VLM_NUM_BEAMS
from app.model_manager import clear_gpu_cache
from app.preprocessing import bgr_to_pil
from app.schemas import TextBox
from app.utils import poly_to_bbox

_model = None
_processor = None


def _get_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        _processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            trust_remote_code=True,
            # Florence-2 remote code is missing `_supports_sdpa` required by newer
            # Transformers when SDPA is selected. Force eager attention.
            attn_implementation="eager",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        _model.eval()
    return _model, _processor


def unload_florence_model() -> None:
    """Release Florence-2 from GPU memory."""
    global _model, _processor
    if _model is not None:
        del _model
    _model = None
    _processor = None
    clear_gpu_cache()


def _run_florence_task(
    image: Image.Image,
    task_prompt: str,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    model, processor = _get_model_and_processor()
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    # Florence-2 remote code expects `pixel_values` dtype to match model weights.
    # The processor typically outputs float32 pixel tensors; cast to model dtype.
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=VLM_NUM_BEAMS,
            use_cache=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    del inputs, generated_ids
    clear_gpu_cache()
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    return parsed


def _parse_ocr_with_region(parsed: dict[str, Any]) -> list[TextBox]:
    region_data = parsed.get("<OCR_WITH_REGION>")
    if not region_data:
        return []

    labels = region_data.get("labels", [])
    quad_boxes = region_data.get("quad_boxes", [])

    items: list[TextBox] = []
    for label, quad in zip(labels, quad_boxes):
        text = (label or "").strip()
        if not text:
            continue
        if len(quad) >= 8:
            poly = [
                [quad[0], quad[1]],
                [quad[2], quad[3]],
                [quad[4], quad[5]],
                [quad[6], quad[7]],
            ]
            bbox = poly_to_bbox(poly)
        elif len(quad) == 4:
            bbox = [int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3])]
        else:
            continue
        items.append(
            TextBox(
                text=text,
                bbox=bbox,
                confidence=0.9,
                source="florence",
            )
        )
    return items


def run_ocr_on_crop(pil_image: Image.Image) -> str:
    """Run Florence-2 <OCR> on a cropped region; return merged text."""
    parsed = _run_florence_task(pil_image, "<OCR>", max_new_tokens=256)
    text = parsed.get("<OCR>", "")
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = str(text).strip()
    # Engineering tags are often read with spaces; collapse for tag-like strings
    if re.search(r"[A-Za-z0-9]-[A-Za-z0-9]", text) or re.fullmatch(r"[\w\-./]+", text.replace(" ", "")):
        text = re.sub(r"\s+", "", text)
    return text


def run_vlm(img: np.ndarray) -> list[TextBox]:
    """Backwards compatible alias. Use `run_vlm_tile()` for single-tile behavior."""
    return run_vlm_tile(img)


def run_vlm_tile(img: np.ndarray) -> list[TextBox]:
    """Run Florence-2 on a single image tile."""
    pil_img = bgr_to_pil(img)
    parsed = _run_florence_task(
        pil_img, "<OCR_WITH_REGION>", max_new_tokens=VLM_MAX_NEW_TOKENS
    )
    return _parse_ocr_with_region(parsed)


def run_vlm_tiles(img_tiles: list[np.ndarray]) -> list[list[TextBox]]:
    """Run VLM extraction sequentially (one tile / one request at a time)."""
    return [run_vlm_tile(t) for t in img_tiles]
