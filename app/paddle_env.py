"""Configure PaddlePaddle before first import (Windows OneDNN workaround)."""

from __future__ import annotations

import os

# Paddle 3.3+ on Windows CPU can hit: OneDnnContext does not have the input Filter
# Disable OneDNN/MKLDNN before paddle is loaded (see requirements paddle pin).
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")
os.environ.setdefault("FLAGS_use_dnnl", "0")


def apply_paddle_flags() -> None:
    """Call after `import paddle` to ensure MKLDNN stays off."""
    try:
        import paddle

        paddle.set_flags({"FLAGS_use_mkldnn": False})
    except ImportError:
        pass
