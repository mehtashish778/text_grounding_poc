"""Draw bounding boxes and labels on P&ID images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import OUTPUTS_DIR, SOURCE_COLORS
from app.schemas import TextBox


def draw_overlay(
    img: np.ndarray,
    items: list[TextBox],
    line_thickness: int = 2,
) -> np.ndarray:
    overlay = img.copy()
    for item in items:
        color = SOURCE_COLORS.get(item.source, (255, 255, 255))
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_thickness)
        label = f"{item.text} ({item.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            overlay,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def save_overlay(
    img: np.ndarray,
    items: list[TextBox],
    output_path: Path | str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(img, items)
    cv2.imwrite(str(output_path), overlay)
    return output_path


def default_overlay_path(stem: str, pipeline: str) -> Path:
    return OUTPUTS_DIR / f"{stem}_{pipeline}.png"
