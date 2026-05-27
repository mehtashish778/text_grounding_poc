"""CLI evaluation harness: run all pipelines on samples/."""

from __future__ import annotations

import app.paddle_env  # noqa: F401 — before any paddle import

import argparse
import csv
import json
from pathlib import Path

from app.config import DEFAULT_OCR_ENGINE, OUTPUTS_DIR, SAMPLES_DIR, ensure_outputs_dir
from app.pipeline_runner import run_pipeline
from app.preprocessing import load_input, preprocess
from app.schemas import PipelineName
from app.visualization import save_overlay

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}


def iter_sample_files(samples_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(samples_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def evaluate_file(
    path: Path,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
) -> list[dict]:
    rows: list[dict] = []
    images = load_input(path)
    stem = path.stem

    for page_idx, img in enumerate(images):
        processed = preprocess(img)
        page_suffix = f"_p{page_idx + 1}" if len(images) > 1 else ""

        for pipeline in ("ocr", "vlm", "hybrid"):
            result = run_pipeline(
                img,
                pipeline=pipeline,  # type: ignore[arg-type]
                ocr_engine=ocr_engine,  # type: ignore[arg-type]
            )
            file_key = f"{stem}{page_suffix}"
            json_path = OUTPUTS_DIR / f"{file_key}_{pipeline}.json"
            json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            save_overlay(
                processed,
                result.items,
                OUTPUTS_DIR / f"{file_key}_{pipeline}.png",
            )

            avg_conf = (
                sum(i.confidence for i in result.items) / len(result.items)
                if result.items
                else 0.0
            )
            rows.append(
                {
                    "file": file_key,
                    "source_path": str(path),
                    "pipeline": pipeline,
                    "num_boxes": len(result.items),
                    "avg_confidence": round(avg_conf, 4),
                    "elapsed_ms": round(result.elapsed_ms, 2),
                    "tag_completeness": "",
                    "bbox_accuracy": "",
                    "engineering_tag_quality": "",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate P&ID text grounding pipelines")
    parser.add_argument(
        "samples_dir",
        nargs="?",
        default=str(SAMPLES_DIR),
        help="Directory containing P&ID sample files",
    )
    parser.add_argument(
        "--ocr-engine",
        default=DEFAULT_OCR_ENGINE,
        choices=["paddle", "easy"],
        help="OCR engine for ocr and hybrid pipelines",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    ensure_outputs_dir()

    if not samples_dir.exists():
        raise SystemExit(f"Samples directory not found: {samples_dir}")

    files = iter_sample_files(samples_dir)
    if not files:
        raise SystemExit(f"No sample files found in {samples_dir}")

    all_rows: list[dict] = []
    for path in files:
        print(f"Processing {path.name}...")
        all_rows.extend(evaluate_file(path, ocr_engine=args.ocr_engine))

    metrics_path = OUTPUTS_DIR / "metrics.csv"
    fieldnames = [
        "file",
        "source_path",
        "pipeline",
        "num_boxes",
        "avg_confidence",
        "elapsed_ms",
        "tag_completeness",
        "bbox_accuracy",
        "engineering_tag_quality",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Processed {len(files)} file(s), {len(all_rows)} pipeline runs.")


if __name__ == "__main__":
    main()
