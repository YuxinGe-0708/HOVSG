"""Memory-efficient feature accumulation and geometry-cache helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch


class LowMemoryFeatureAccumulator:
    """Accumulate per-point CLIP means without a second full-size copy.

    The historical implementation used a float32 ``N x D`` tensor and then
    created additional float32 tensors while averaging and normalizing it.
    For a 3.7 million point, 1024-dimensional map that single tensor is about
    14.2 GiB. Store running means in float16, reducing duplicate observations
    in float32 batches. Raw float16 sums lose increments or overflow when many
    pixels/frames associate to the same map point.
    """

    _DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def __init__(
        self,
        point_count: int,
        feature_dim: int,
        dtype: str = "float16",
        normalization_chunk_points: int = 32768,
    ) -> None:
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if dtype not in self._DTYPES:
            raise ValueError(
                f"Unsupported accumulator dtype {dtype!r}; "
                f"choose one of {sorted(self._DTYPES)}"
            )
        if normalization_chunk_points <= 0:
            raise ValueError("normalization_chunk_points must be positive")

        self.dtype_name = dtype
        self.dtype = self._DTYPES[dtype]
        self.normalization_chunk_points = int(normalization_chunk_points)
        self.means = torch.zeros(
            (int(point_count), int(feature_dim)),
            dtype=self.dtype,
            device="cpu",
        )
        # Counts are tiny compared with the feature matrix. float32 keeps the
        # division stable even when a point is observed in many frames.
        self.counts = torch.zeros(
            (int(point_count), 1),
            dtype=torch.float32,
            device="cpu",
        )

    @property
    def allocated_bytes(self) -> int:
        """Return bytes owned by the persistent accumulator tensors."""
        return (
            self.means.numel() * self.means.element_size()
            + self.counts.numel() * self.counts.element_size()
        )

    def add(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        """Add one bounded batch of features to their global point indices."""
        indices = indices.to(device="cpu", dtype=torch.long)
        features = features.to(device="cpu", dtype=torch.float32)
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if features.ndim != 2:
            raise ValueError("features must be two-dimensional")
        if indices.numel() != features.shape[0]:
            raise ValueError("indices and features must contain the same rows")
        if not indices.numel():
            return
        if features.shape[1] != self.means.shape[1]:
            raise ValueError("feature dimension does not match accumulator")
        if indices.min() < 0 or indices.max() >= self.means.shape[0]:
            raise ValueError("point index outside accumulator")

        unique_indices, inverse = torch.unique(indices, return_inverse=True)
        batch_sums = torch.zeros(
            (unique_indices.numel(), features.shape[1]), dtype=torch.float32
        )
        batch_sums.index_add_(0, inverse, features)
        batch_counts = torch.bincount(inverse).to(torch.float32).unsqueeze(1)
        old_counts = self.counts[unique_indices]
        new_counts = old_counts + batch_counts
        means = self.means[unique_indices].to(torch.float32)
        means = (means * old_counts + batch_sums) / new_counts
        self.means[unique_indices] = means.to(self.dtype)
        self.counts[unique_indices] = new_counts

    def finalize(self) -> np.ndarray:
        """L2-normalize means in-place, returning a float16/32 view."""
        point_count = self.means.shape[0]
        for start in range(0, point_count, self.normalization_chunk_points):
            end = min(point_count, start + self.normalization_chunk_points)
            chunk = self.means[start:end].to(dtype=torch.float32)
            chunk = torch.nn.functional.normalize(chunk, p=2, dim=-1)
            self.means[start:end].copy_(chunk.to(dtype=self.dtype))
            del chunk
        return self.means.numpy()

    def release_counts(self) -> None:
        """Release counters after finalization while retaining features."""
        self.counts = torch.empty((0, 1), dtype=torch.float32)


def save_geometry_cache(
    cache_dir: str,
    point_cloud: o3d.geometry.PointCloud,
    metadata: Dict[str, Any],
) -> Tuple[str, str]:
    """Atomically save a geometry point cloud and its validation metadata."""
    if point_cloud is None or point_cloud.is_empty():
        raise ValueError("Cannot cache an empty geometry point cloud")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    point_cloud_path = cache_path / "full_pcd.ply"
    point_cloud_tmp = cache_path / "full_pcd.tmp.ply"
    metadata_path = cache_path / "metadata.json"
    metadata_tmp = cache_path / "metadata.tmp.json"

    if not o3d.io.write_point_cloud(str(point_cloud_tmp), point_cloud):
        raise RuntimeError(f"Failed to write geometry cache: {point_cloud_tmp}")
    os.replace(point_cloud_tmp, point_cloud_path)

    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    os.replace(metadata_tmp, metadata_path)
    return str(point_cloud_path), str(metadata_path)


def load_geometry_cache(
    cache_dir: str,
    expected_signature: Dict[str, Any],
) -> Optional[Tuple[o3d.geometry.PointCloud, Dict[str, Any]]]:
    """Load a cache only when its complete geometry signature matches."""
    cache_path = Path(cache_dir)
    point_cloud_path = cache_path / "full_pcd.ply"
    metadata_path = cache_path / "metadata.json"
    if not point_cloud_path.is_file() or not metadata_path.is_file():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, ValueError, TypeError):
        return None
    if metadata.get("signature") != expected_signature:
        return None

    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    if point_cloud.is_empty():
        return None
    return point_cloud, metadata
