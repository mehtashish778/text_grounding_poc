"""Post-processing filter using a local Qwen3 text model to drop false positives."""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from app.config import DEVICE, HF_TOKEN, QWEN3_TEXT_MODEL_ID, REASONING_MAX_NEW_TOKENS
from app.model_manager import clear_gpu_cache
from app.schemas import TextBox

_model = None
_tokenizer = None


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_kwargs: dict = {}
        if HF_TOKEN:
            hf_kwargs["token"] = HF_TOKEN

        print(f"    Loading reasoning model: {QWEN3_TEXT_MODEL_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(QWEN3_TEXT_MODEL_ID, **hf_kwargs)
        _model = AutoModelForCausalLM.from_pretrained(
            QWEN3_TEXT_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="cuda" if DEVICE == "cuda" else None,
            **hf_kwargs,
        )
        _model.eval()
    return _model, _tokenizer


def unload_reasoning_model() -> None:
    """Release local Qwen3 text model from GPU memory."""
    global _model, _tokenizer
    if _model is not None:
        del _model
    _model = None
    _tokenizer = None
    clear_gpu_cache()


def load_filter_prompt() -> str:
    """Load prompt from filter_prompt.txt (lines starting with # are ignored)."""
    from app.config import REASONING_FILTER_PROMPT_FILE

    if not REASONING_FILTER_PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Reasoning filter prompt file not found: {REASONING_FILTER_PROMPT_FILE}"
        )
    lines = [
        line
        for line in REASONING_FILTER_PROMPT_FILE.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("#")
    ]
    return "\n".join(lines).strip()


def _extract_json_value(text: str) -> Any:
    """Parse JSON from model output (fenced block or first array/object)."""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("No JSON found in model response", text, 0)


def _decisions_from_parsed(parsed: Any, n: int) -> list[bool] | None:
    """Convert model JSON to per-item keep decisions (length n)."""
    if isinstance(parsed, list):
        if len(parsed) == n and all(isinstance(x, bool) for x in parsed):
            return list(parsed)
        if parsed and all(isinstance(x, int) for x in parsed):
            keep = [False] * n
            for i in parsed:
                if 0 <= i < n:
                    keep[i] = True
            return keep
        return None

    if isinstance(parsed, dict):
        for key in ("keep", "decisions", "results", "flags"):
            if key in parsed and isinstance(parsed[key], list):
                return _decisions_from_parsed(parsed[key], n)
        if "keep_indices" in parsed and isinstance(parsed["keep_indices"], list):
            return _decisions_from_parsed(parsed["keep_indices"], n)
    return None


def filter_items(items: list[TextBox], prompt_template: str) -> list[TextBox]:
    """Run one batch LLM call to filter false-positive text detections."""
    if not items:
        return items
    if not prompt_template.strip():
        print("    Reasoning filter: empty prompt; keeping all items.")
        return items

    texts = [item.text for item in items]
    user_content = (
        f"{prompt_template.strip()}\n\n"
        f"Extracted text strings (JSON array, {len(texts)} items, same order):\n"
        f"{json.dumps(texts, ensure_ascii=False)}"
    )

    model, tokenizer = _get_model_and_tokenizer()
    messages = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    elif DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=REASONING_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, input_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    try:
        parsed = _extract_json_value(decoded)
        decisions = _decisions_from_parsed(parsed, len(items))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"    Reasoning filter: could not parse response ({exc}); keeping all items.")
        return items

    if decisions is None or len(decisions) != len(items):
        print(
            "    Reasoning filter: unexpected response shape; keeping all items."
        )
        return items

    filtered = [item for item, keep in zip(items, decisions) if keep]
    return filtered
