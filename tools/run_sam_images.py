#!/usr/bin/env python3
"""Run SAM on one or more local images and save masks plus visualizations.

Examples:
  python tools/run_sam_images.py image1.jpg image2.png \
      --checkpoint checkpoints/sam_vit_h_4b8939.pth --output-dir sam_results
  python tools/run_sam_images.py my_images/ \
      --checkpoint checkpoints/sam_vit_h_4b8939.pth --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Segment Anything on local images and save masks/overlays."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Image files and/or directories containing images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="SAM checkpoint, for example sam_vit_h_4b8939.pth.",
    )
    parser.add_argument(
        "--model-type",
        choices=("vit_h", "vit_l", "vit_b"),
        default="vit_h",
        help="SAM model family; default: vit_h.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sam_results"),
        help="Directory for outputs; default: sam_results.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device; default: auto.",
    )
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--points-per-batch", type=int, default=144)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--min-mask-region-area", type=int, default=50)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process at most this many images; 0 means all.",
    )
    return parser.parse_args()


def collect_images(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(
                path for path in sorted(item.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif item.is_file():
            if item.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image extension: {item}")
            paths.append(item)
        else:
            raise FileNotFoundError(f"Input image or directory does not exist: {item}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def color_for_index(index: int) -> tuple[int, int, int]:
    # Deterministic BGR colors, deliberately varied enough for adjacent masks.
    rng = np.random.default_rng(1009 + index * 7919)
    color = rng.integers(45, 235, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def save_image_result(image_path: Path, image_bgr: np.ndarray, masks: list[dict], output_dir: Path) -> dict:
    stem_dir = output_dir / image_path.stem
    mask_dir = stem_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    overlay = image_bgr.copy()
    metadata = []
    for index, mask in enumerate(sorted(masks, key=lambda item: item["area"], reverse=True), start=1):
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        color = np.asarray(color_for_index(index), dtype=np.uint8)
        overlay[segmentation] = (
            0.55 * overlay[segmentation].astype(np.float32)
            + 0.45 * color.astype(np.float32)
        ).astype(np.uint8)

        binary_path = mask_dir / f"mask_{index:04d}.png"
        if not cv2.imwrite(str(binary_path), segmentation.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to write {binary_path}")

        x, y, width, height = [int(value) for value in mask["bbox"]]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 1)
        label = f"{index} ({int(mask['area'])} px)"
        text_y = max(14, y + 14)
        cv2.putText(
            overlay,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        record = {
            "mask_id": index,
            "area_px": int(mask["area"]),
            "bbox_xywh": [x, y, width, height],
            "predicted_iou": float(mask.get("predicted_iou", 0.0)),
            "stability_score": float(mask.get("stability_score", 0.0)),
            "crop_box_xyxy": [int(value) for value in mask.get("crop_box", [])],
            "mask_file": str(binary_path.relative_to(stem_dir)),
        }
        metadata.append(record)

    overlay_path = stem_dir / "overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write {overlay_path}")

    # Also save a compact semantic-friendly view: white masks on black background.
    all_masks = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for mask in masks:
        all_masks[np.asarray(mask["segmentation"], dtype=bool)] = 255
    all_masks_path = stem_dir / "all_masks.png"
    if not cv2.imwrite(str(all_masks_path), all_masks):
        raise RuntimeError(f"Failed to write {all_masks_path}")

    info = {
        "source_image": str(image_path),
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "mask_count": len(metadata),
        "overlay_file": "overlay.png",
        "all_masks_file": "all_masks.png",
        "masks": metadata,
    }
    json_path = stem_dir / "masks.json"
    json_path.write_text(json.dumps(info, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return info


def main() -> None:
    args = parse_args()
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ModuleNotFoundError as error:
        missing = error.name or "a required package"
        raise RuntimeError(
            f"Missing dependency {missing!r}. Run this script in the environment "
            "where PyTorch and segment-anything are installed."
        ) from error
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {args.checkpoint}")
    if args.points_per_side <= 0 or args.points_per_batch <= 0:
        raise ValueError("--points-per-side and --points-per-batch must be positive")
    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative")

    images = collect_images(args.inputs)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError("No input images were found")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"[SAM] model={args.model_type} device={device} images={len(images)}", flush=True)

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        points_per_batch=args.points_per_batch,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for position, image_path in enumerate(images, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[SAM][WARN] skip unreadable image={image_path}", flush=True)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"[SAM] processing={position}/{len(images)} image={image_path}", flush=True)
        masks = mask_generator.generate(image_rgb)
        info = save_image_result(image_path, image_bgr, masks, args.output_dir)
        info["source_image"] = str(image_path)
        summary.append(info)
        print(
            f"[SAM][OK] image={image_path.name} masks={info['mask_count']} "
            f"output={args.output_dir / image_path.stem}",
            flush=True,
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"[SAM][DONE] processed={len(summary)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
