"""Robust single-floor estimation for gravity-aligned indoor point clouds."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import cv2
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


def _value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _plain_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=True))
    return dict(config)


def _largest_xz_component(points: np.ndarray, cell_size: float) -> tuple[float, float]:
    """Return largest connected X-Z area and its fraction of occupied cells."""
    if len(points) == 0:
        return 0.0, 0.0
    cells = np.floor(points[:, [0, 2]] / cell_size).astype(np.int64)
    cells = np.unique(cells, axis=0)
    if len(cells) == 0:
        return 0.0, 0.0
    cells -= cells.min(axis=0)
    shape = cells.max(axis=0) + 1
    if int(shape[0]) * int(shape[1]) > 25_000_000:
        area = len(cells) * cell_size * cell_size
        return float(area), 1.0
    occupancy = np.zeros((int(shape[1]), int(shape[0])), dtype=np.uint8)
    occupancy[cells[:, 1], cells[:, 0]] = 1
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        occupancy, 8, cv2.CV_32S
    )
    if count <= 1:
        return 0.0, 0.0
    largest_cells = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return (
        float(largest_cells * cell_size * cell_size),
        float(largest_cells / len(cells)),
    )


def _camera_support(
    plane_points: np.ndarray,
    plane_height: float,
    camera_positions: np.ndarray,
    min_camera_height: float,
    max_camera_height: float,
    proximity_m: float,
) -> tuple[float, float]:
    if len(camera_positions) == 0:
        return 1.0, 1.0
    relative_heights = camera_positions[:, 1] - plane_height
    height_support = np.mean(
        np.logical_and(
            relative_heights >= min_camera_height,
            relative_heights <= max_camera_height,
        )
    )
    if len(plane_points) == 0:
        return float(height_support), 0.0
    tree = cKDTree(plane_points[:, [0, 2]])
    distances, _ = tree.query(camera_positions[:, [0, 2]], k=1)
    proximity_support = np.mean(distances <= proximity_m)
    return float(height_support), float(proximity_support)


def estimate_single_floor_geometry(
    point_cloud: o3d.geometry.PointCloud,
    camera_positions: Iterable[np.ndarray],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Estimate ground and ceiling using constrained horizontal height peaks.

    The input must already use a gravity-aligned frame where Y is vertical.
    Candidate peaks are accepted only when a RANSAC plane is horizontal, below
    plausible camera positions, spatially extensive, connected, and close to
    the recorded camera trajectory.
    """
    points = np.asarray(point_cloud.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("Cannot estimate a floor from an empty point cloud")
    config = _plain_config(config)

    camera_positions = np.asarray(list(camera_positions), dtype=np.float64)
    if camera_positions.size == 0:
        camera_positions = np.empty((0, 3), dtype=np.float64)
    else:
        camera_positions = camera_positions.reshape(-1, 3)
        camera_positions = camera_positions[
            np.isfinite(camera_positions).all(axis=1)
        ]

    voxel_size = float(_value(config, "ground_estimation_voxel_m", 0.05))
    normal_radius = float(_value(config, "ground_normal_radius_m", 0.15))
    normal_max_nn = int(_value(config, "ground_normal_max_nn", 50))
    max_tilt_deg = float(_value(config, "ground_normal_max_tilt_deg", 20.0))
    histogram_bin = float(_value(config, "ground_histogram_bin_m", 0.02))
    peak_distance = float(_value(config, "ground_peak_min_distance_m", 0.15))
    peak_band = float(_value(config, "ground_peak_band_m", 0.10))
    plane_threshold = float(
        _value(config, "ground_plane_distance_threshold_m", 0.04)
    )
    min_inliers = int(_value(config, "ground_min_plane_inliers", 100))
    coverage_cell = float(_value(config, "ground_coverage_cell_m", 0.10))
    min_camera_height = float(_value(config, "ground_camera_height_min_m", 0.35))
    max_camera_height = float(_value(config, "ground_camera_height_max_m", 2.20))
    proximity_m = float(_value(config, "ground_pose_proximity_m", 1.00))
    min_height_support = float(
        _value(config, "ground_min_camera_height_support_ratio", 0.50)
    )
    min_pose_support = float(
        _value(config, "ground_min_pose_proximity_ratio", 0.15)
    )

    sampled = point_cloud.voxel_down_sample(voxel_size=max(voxel_size, 1e-3))
    sampled_points = np.asarray(sampled.points, dtype=np.float64)
    if len(sampled_points) < min_inliers:
        sampled = o3d.geometry.PointCloud()
        sampled.points = o3d.utility.Vector3dVector(points)
        sampled_points = points
    sampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(normal_radius, voxel_size * 2.0),
            max_nn=max(normal_max_nn, 10),
        )
    )
    normals = np.asarray(sampled.normals, dtype=np.float64)
    verticality_threshold = float(np.cos(np.deg2rad(max_tilt_deg)))
    horizontal_mask = np.abs(normals[:, 1]) >= verticality_threshold
    horizontal_points = sampled_points[horizontal_mask]
    if len(horizontal_points) < min_inliers:
        horizontal_points = sampled_points

    min_height = float(np.min(horizontal_points[:, 1]))
    max_height = float(np.max(horizontal_points[:, 1]))
    bin_count = max(3, int(np.ceil((max_height - min_height) / histogram_bin)))
    counts, edges = np.histogram(horizontal_points[:, 1], bins=bin_count)
    centers = (edges[:-1] + edges[1:]) * 0.5
    sigma_bins = float(_value(config, "ground_histogram_smoothing_sigma", 2.0))
    if sigma_bins > 0.0:
        smooth_counts = gaussian_filter1d(counts.astype(np.float64), sigma=sigma_bins)
    else:
        smooth_counts = counts.astype(np.float64)
    positive = smooth_counts[smooth_counts > 0]
    height_threshold = (
        max(float(np.percentile(positive, 60)), float(np.max(smooth_counts) * 0.02))
        if len(positive)
        else 0.0
    )
    prominence = max(float(np.max(smooth_counts) * 0.01), 1.0)
    peaks, _ = find_peaks(
        smooth_counts,
        distance=max(1, int(round(peak_distance / histogram_bin))),
        height=height_threshold,
        prominence=prominence,
    )
    if len(smooth_counts):
        peaks = np.unique(np.append(peaks, int(np.argmax(smooth_counts))))

    candidates = []
    for peak_index in peaks:
        peak_height = float(centers[int(peak_index)])
        band_points = horizontal_points[
            np.abs(horizontal_points[:, 1] - peak_height) <= peak_band
        ]
        if len(band_points) < min_inliers:
            continue
        band_cloud = o3d.geometry.PointCloud()
        band_cloud.points = o3d.utility.Vector3dVector(band_points)
        plane_model, inlier_indices = band_cloud.segment_plane(
            distance_threshold=plane_threshold,
            ransac_n=3,
            num_iterations=500,
        )
        if len(inlier_indices) < min_inliers:
            continue
        normal = np.asarray(plane_model[:3], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-9:
            continue
        verticality = float(abs(normal[1]) / normal_norm)
        if verticality < verticality_threshold:
            continue
        inlier_points = band_points[np.asarray(inlier_indices, dtype=np.int64)]
        plane_height = float(np.median(inlier_points[:, 1]))
        plane_residuals = np.abs(
            inlier_points @ normal + float(plane_model[3])
        ) / normal_norm
        residual_median = float(np.median(plane_residuals))
        largest_area, connected_fraction = _largest_xz_component(
            inlier_points, coverage_cell
        )
        height_support, proximity_support = _camera_support(
            inlier_points,
            plane_height,
            camera_positions,
            min_camera_height,
            max_camera_height,
            proximity_m,
        )
        spatial_score = (
            largest_area
            * (0.25 + 0.75 * connected_fraction)
            * verticality
        )
        ground_score = (
            spatial_score
            * (0.20 + 0.80 * height_support)
            * (0.20 + 0.80 * proximity_support)
            * float(np.exp(-residual_median / max(plane_threshold, 1e-6)))
        )
        candidates.append(
            {
                "height": plane_height,
                "peak_height": peak_height,
                "score": float(ground_score),
                "spatial_score": float(spatial_score),
                "inliers": int(len(inlier_points)),
                "coverage_area_m2": largest_area,
                "connected_fraction": connected_fraction,
                "camera_height_support": height_support,
                "camera_proximity_support": proximity_support,
                "verticality": verticality,
                "plane_residual_median_m": residual_median,
                "ground_eligible": bool(
                    not len(camera_positions)
                    or (
                        height_support >= min_height_support
                        and proximity_support >= min_pose_support
                    )
                ),
            }
        )

    ground_candidates = sorted(
        [candidate for candidate in candidates if candidate["ground_eligible"]],
        key=lambda item: item["score"],
        reverse=True,
    )
    fallback_used = not ground_candidates
    if ground_candidates:
        ground = ground_candidates[0]
        floor_zero_level = float(ground["height"])
    else:
        lower_quantile = float(_value(config, "single_floor_lower_quantile", 0.02))
        floor_zero_level = float(np.quantile(points[:, 1], lower_quantile))
        ground = {
            "height": floor_zero_level,
            "score": 0.0,
            "inliers": 0,
            "coverage_area_m2": 0.0,
            "connected_fraction": 0.0,
            "camera_height_support": 0.0,
            "camera_proximity_support": 0.0,
            "verticality": 0.0,
            "plane_residual_median_m": 0.0,
        }

    min_room_height = float(_value(config, "single_floor_min_room_height_m", 2.0))
    max_room_height = float(_value(config, "single_floor_max_room_height_m", 4.0))
    ceiling_min_camera_clearance = float(
        _value(config, "ceiling_min_camera_clearance_m", 0.30)
    )
    ceiling_candidates = []
    for candidate in candidates:
        room_height = candidate["height"] - floor_zero_level
        if room_height < min_room_height or room_height > max_room_height:
            continue
        if len(camera_positions):
            camera_below_ratio = float(
                np.mean(
                    candidate["height"] - camera_positions[:, 1]
                    >= ceiling_min_camera_clearance
                )
            )
            if camera_below_ratio < min_height_support:
                continue
        ceiling_candidates.append(candidate)
    if ceiling_candidates:
        ceiling = max(ceiling_candidates, key=lambda item: item["spatial_score"])
        floor_upper_level = float(ceiling["height"])
        ceiling_source = "horizontal_peak"
    else:
        upper_quantile = float(_value(config, "single_floor_upper_quantile", 0.98))
        robust_upper = float(np.quantile(points[:, 1], upper_quantile))
        floor_upper_level = float(
            np.clip(
                robust_upper,
                floor_zero_level + min_room_height,
                floor_zero_level + max_room_height,
            )
        )
        ceiling = None
        ceiling_source = "bounded_quantile_fallback"

    return {
        "floor_zero_level": floor_zero_level,
        "floor_upper_level": floor_upper_level,
        "ground": ground,
        "ceiling": ceiling,
        "ceiling_source": ceiling_source,
        "fallback_used": fallback_used,
        "fallback_allowed": bool(
            _value(config, "ground_allow_quantile_fallback", True)
        ),
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda item: item["height"]),
        "histogram_centers": centers,
        "histogram_counts": smooth_counts,
    }
