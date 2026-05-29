"""GPU model lifecycle: load one backend at a time, unload between pipeline runs."""

from __future__ import annotations

import gc

from app.config import DEVICE


def clear_gpu_cache() -> None:
    if DEVICE != "cuda":
        return
    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def unload_all_vlm_models() -> None:
    """Unload Florence-2 and reasoning model weights from GPU."""
    from app.reasoning_filter import unload_reasoning_model
    from app.vlm_pipeline import unload_florence_model

    unload_florence_model()
    unload_reasoning_model()
    clear_gpu_cache()
