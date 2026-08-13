"""Validate ground-family navigation masks directly from an Open3D binary PLY.

This diagnostic intentionally depends only on NumPy and Pillow so it can run on
a review workstation without the HOV-SG/Open3D environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_open3d_binary_ply(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header is missing end_header")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only binary_little_endian Open3D PLY files are supported")
        vertex_line = next(line for line in header if line.startswith("element vertex "))
        count = int(vertex_line.split()[-1])
        expected = [
            "property double x",
            "property double y",
            "property double z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        properties = [line for line in header if line.startswith("property ")]
        if properties != expected:
            raise ValueError(f"Unexpected PLY vertex layout: {properties}")
        dtype = np.dtype(
            [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("rgb", "u1", (3,))]
        )
        vertices = np.fromfile(file, dtype=dtype, count=count)
    if len(vertices) != count:
        raise ValueError(f"Expected {count} vertices, read {len(vertices)}")
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"]))


def select_family(diagnostic: dict) -> list[float]:
    anchor = diagnostic["ground"]
    anchor_height = float(anchor["height"])
    anchor_coverage = max(float(anchor["coverage_area_m2"]), 1e-9)
    family = []
    for candidate in diagnostic["candidates"]:
        if abs(float(candidate["height"]) - anchor_height) > 0.90:
            continue
        if (
            float(candidate["coverage_area_m2"]) < anchor_coverage * 0.35
            or float(candidate["connected_fraction"]) < 0.65
            or float(candidate["camera_height_support"]) < 0.65
            or float(candidate["camera_proximity_support"]) < 0.15
            or float(candidate["verticality"]) < 0.98
        ):
            continue
        family.append(float(candidate["height"]))
    return sorted(family or [anchor_height])


def point_masks(points: np.ndarray, family: list[float], ceiling: float):
    heights = np.asarray(family, dtype=np.float64)
    relative = points[:, 1, None] - heights[None, :]
    ground = np.any((relative >= -0.08) & (relative <= 0.12), axis=1)
    excluded = np.any((relative >= -0.10) & (relative <= 0.15), axis=1)
    relative_below = np.where(relative >= 0.0, relative, np.inf)
    obstacle_height = np.min(relative_below, axis=1)
    obstacle = (
        ~excluded
        & (obstacle_height >= 0.15)
        & (obstacle_height <= 1.80)
        & (points[:, 1] < ceiling - 0.20)
    )
    return ground, obstacle


def occupancy(points: np.ndarray, mask: np.ndarray, minimum: np.ndarray, shape, count=1):
    cells = np.floor((points[mask][:, [0, 2]] - minimum[[0, 2]]) / 0.05).astype(int)
    valid = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < shape[1])
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < shape[0])
    )
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, (cells[valid, 1], cells[valid, 0]), 1)
    return counts >= count


def dilate_ellipse(mask: np.ndarray, diameter: int) -> np.ndarray:
    radius = diameter // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy) <= radius * radius
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask)
    for row, column in np.argwhere(kernel):
        result |= padded[row : row + mask.shape[0], column : column + mask.shape[1]]
    return result


def erode_ellipse(mask: np.ndarray, diameter: int) -> np.ndarray:
    return ~dilate_ellipse(~mask, diameter)


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--ground-json", required=True, type=Path)
    parser.add_argument("--poses-map", type=Path)
    parser.add_argument("--baseline-ground-map", type=Path)
    parser.add_argument("--baseline-obstacle-map", type=Path)
    parser.add_argument("--baseline-inflated-obstacle-map", type=Path)
    parser.add_argument("--top-down-map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    diagnostic = json.loads(args.ground_json.read_text(encoding="utf-8"))
    family = select_family(diagnostic)
    anchor = float(diagnostic["floor_zero_level_m"])
    ceiling = float(diagnostic["floor_upper_level_m"])
    points = read_open3d_binary_ply(args.point_cloud)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    crop = (points[:, 1] >= min(family) - 0.15) & (points[:, 1] <= ceiling)
    points = points[crop]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    shape = tuple(np.ceil((maximum[[2, 0]] - minimum[[2, 0]]) / 0.05 + 1).astype(int))

    ground_points, obstacle_points = point_masks(points, family, ceiling)
    ground_map = occupancy(points, ground_points, minimum, shape)
    obstacle_raw = occupancy(points, obstacle_points, minimum, shape, count=3)
    obstacle_inflated = dilate_ellipse(obstacle_raw, 17)
    save_mask(args.output / "ground_family_region.png", ground_map)
    save_mask(args.output / "ground_family_obstacles_raw.png", obstacle_raw)
    save_mask(args.output / "ground_family_obstacles_inflated.png", obstacle_inflated)

    overlay = np.zeros((*shape, 3), dtype=np.uint8)
    overlay[ground_map] = [45, 180, 75]
    overlay[obstacle_raw] = [220, 50, 47]

    result = {
        "point_cloud": str(args.point_cloud),
        "finite_cropped_points": int(len(points)),
        "ground_height_family_m": family,
        "grid_shape": [int(value) for value in shape],
        "ground_points": int(np.count_nonzero(ground_points)),
        "obstacle_points": int(np.count_nonzero(obstacle_points)),
        "ground_cells": int(np.count_nonzero(ground_map)),
        "raw_obstacle_cells": int(np.count_nonzero(obstacle_raw)),
        "inflated_obstacle_cells": int(np.count_nonzero(obstacle_inflated)),
        "candidate_peak_classification": [],
    }
    for candidate in diagnostic["candidates"]:
        height = float(candidate["height"])
        band = np.abs(points[:, 1] - height) <= 0.08
        band_count = int(np.count_nonzero(band))
        result["candidate_peak_classification"].append(
            {
                "height_m": height,
                "in_ground_family": any(abs(height - item) < 1e-6 for item in family),
                "points_in_0.08m_band": band_count,
                "classified_ground_ratio": float(
                    np.count_nonzero(band & ground_points) / max(band_count, 1)
                ),
                "classified_obstacle_ratio": float(
                    np.count_nonzero(band & obstacle_points) / max(band_count, 1)
                ),
            }
        )
    if args.poses_map:
        poses = np.asarray(Image.open(args.poses_map).convert("L")) > 0
        result["poses_map_shape"] = [int(value) for value in poses.shape]
        if poses.shape == obstacle_raw.shape:
            pose_cells = int(np.count_nonzero(poses))
            result["pose_cells"] = pose_cells
            result["raw_obstacle_pose_overlap_cells"] = int(
                np.count_nonzero(obstacle_raw & poses)
            )
            result["raw_obstacle_pose_overlap_ratio"] = float(
                np.count_nonzero(obstacle_raw & poses) / max(pose_cells, 1)
            )
            result["inflated_obstacle_pose_overlap_ratio"] = float(
                np.count_nonzero(obstacle_inflated & poses) / max(pose_cells, 1)
            )
            # poses_region_map uses a 0.50 m radius in this dataset. Eroding by
            # 0.35 m leaves a 0.15 m center band, which is a more meaningful
            # collision check than counting walls along the corridor boundary.
            pose_center = erode_ellipse(poses, 15)
            pose_center_cells = int(np.count_nonzero(pose_center))
            result["pose_center_cells"] = pose_center_cells
            result["raw_obstacle_pose_center_overlap_ratio"] = float(
                np.count_nonzero(obstacle_raw & pose_center)
                / max(pose_center_cells, 1)
            )
            result["inflated_obstacle_pose_center_overlap_ratio"] = float(
                np.count_nonzero(obstacle_inflated & pose_center)
                / max(pose_center_cells, 1)
            )
            obstacle_cells = np.floor(
                (points[obstacle_points][:, [0, 2]] - minimum[[0, 2]]) / 0.05
            ).astype(int)
            valid_obstacle_cells = (
                (obstacle_cells[:, 0] >= 0)
                & (obstacle_cells[:, 0] < shape[1])
                & (obstacle_cells[:, 1] >= 0)
                & (obstacle_cells[:, 1] < shape[0])
            )
            obstacle_in_pose_center = np.zeros(
                np.count_nonzero(obstacle_points), dtype=bool
            )
            obstacle_in_pose_center[valid_obstacle_cells] = pose_center[
                obstacle_cells[valid_obstacle_cells, 1],
                obstacle_cells[valid_obstacle_cells, 0],
            ]
            obstacle_pose_heights = points[obstacle_points][
                obstacle_in_pose_center, 1
            ]
            bins = np.arange(min(family), ceiling + 0.1001, 0.10)
            counts, edges = np.histogram(obstacle_pose_heights, bins=bins)
            result["pose_center_obstacle_height_histogram"] = [
                {
                    "min_height_m": float(edges[index]),
                    "max_height_m": float(edges[index + 1]),
                    "points": int(count),
                }
                for index, count in enumerate(counts)
                if count
            ]
            save_mask(args.output / "pose_center_region.png", pose_center)
            overlay[poses] = np.maximum(overlay[poses], [40, 110, 235])
        else:
            result["poses_map_alignment_error"] = (
                f"pose map {poses.shape} != computed grid {obstacle_raw.shape}"
            )
    Image.fromarray(overlay).save(args.output / "classification_overlay.png")

    baselines = {
        "ground": (args.baseline_ground_map, ground_map),
        "raw_obstacle": (args.baseline_obstacle_map, obstacle_raw),
        "inflated_obstacle": (
            args.baseline_inflated_obstacle_map,
            obstacle_inflated,
        ),
    }
    result["baseline_comparison"] = {}
    for name, (path, current) in baselines.items():
        if path is None:
            continue
        baseline = np.asarray(Image.open(path).convert("L")) > 0
        if baseline.shape != current.shape:
            result["baseline_comparison"][name] = {
                "alignment_error": f"{baseline.shape} != {current.shape}"
            }
            continue
        old_cells = int(np.count_nonzero(baseline))
        new_cells = int(np.count_nonzero(current))
        result["baseline_comparison"][name] = {
            "old_cells": old_cells,
            "new_cells": new_cells,
            "change_ratio": float((new_cells - old_cells) / max(old_cells, 1)),
        }

    if args.top_down_map:
        top_down = np.asarray(Image.open(args.top_down_map).convert("RGB"))
        if top_down.shape[:2] == shape:
            color = top_down.copy()
            color[ground_map] = (
                0.55 * color[ground_map] + 0.45 * np.array([45, 180, 75])
            ).astype(np.uint8)
            color[obstacle_raw] = (
                0.45 * color[obstacle_raw] + 0.55 * np.array([220, 50, 47])
            ).astype(np.uint8)
            Image.fromarray(color).save(args.output / "classification_top_down.png")
        else:
            result["top_down_alignment_error"] = (
                f"{top_down.shape[:2]} != {shape}"
            )
    (args.output / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
