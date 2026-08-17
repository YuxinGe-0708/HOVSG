from collections import defaultdict
import copy
import glob
import json
from multiprocessing import current_process, Process, Queue
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Union, Any

import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    label as connected_components,
    median_filter,
)
from scipy.signal import find_peaks
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import skfmm
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from hovsg.graph.floor import Floor
from hovsg.graph.object import Object
from hovsg.utils.constants import MATTERPORT_LABELS_40
from hovsg.utils.clip_utils import get_text_feats_62_templates
from hovsg.utils.runtime_status import print_status

# change matplotlib backend to a gui one
plt.switch_backend("Agg")


def compute_sdf(boundary_mask, distance_scale=1):
    dx = 1  # downsample factor
    f = distance_scale / dx  # distance function scale

    # cv2.resize(boundary_map, None, fx=1/dx, fy=1/dx, interpolation=cv2.INTER_NEAREST)
    sdf = skfmm.distance(1 - boundary_mask)
    sdf[sdf > f] = f
    sdf = sdf / f
    sdf = 1 - sdf
    return sdf


class NavigationGraph:
    def __init__(self, floor_pcd: o3d.geometry.PointCloud, cell_size: int):
        """Initialization of the NavigationGraph class.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            cell_size (int): the resolution of the cell (m/cell)
        """
        self.pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        self.pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        self.grid_size = np.ceil((self.pcd_max - self.pcd_min) / cell_size + 1).astype(
            np.int32
        )
        self.grid_size = self.grid_size[[0, 2]]
        self.cell_size = cell_size
        self.has_stairs = False
        # self.midpoint = grid_size // 2
        # assert self.grid_size % 2 == 0

        # self.reset()

    def get_floor_poses(
        self,
        floor: Floor,
        poses_list: List[np.ndarray],
        upper_floor_min_height: float = None,
        camera_height: float = 1.5,
        filter_by_height: bool = True,
    ) -> List[np.ndarray]:
        """Return the list of poses that are within the height range of the floor.

        Args:
            floor (Floor): The Floor object.
            poses_list (List[np.ndarray]): The list of camera poses for the exploration of the whole scene (multi-floor).
            upper_floor_min_height (float, optional): The uppebound of the height range that is used to determined the poses
                belonging to the floor. If None, then use the upper bound of the floor point cloud. Defaults to None.
            camera_height (float, optional): Camera height used only for multi-floor assignment.
            filter_by_height (bool, optional): Whether to assign poses by camera height. Single-floor
                handheld recordings should disable this so vertical hand motion and SLAM drift do not
                discard the horizontal trajectory.

        Returns:
            floor_poses_list (List[np.ndarray]): A list of (4, 4) numpy arrays representing the poses that are within the
                height range of the floor.
        """
        valid_poses = [
            np.asarray(pose, dtype=np.float64)
            for pose in poses_list
            if np.asarray(pose).shape == (4, 4) and np.isfinite(pose).all()
        ]
        if not filter_by_height:
            return valid_poses

        floor_min_height = np.min(np.asarray(floor.pcd.points)[:, 1])
        floor_max_height = np.max(np.asarray(floor.pcd.points)[:, 1])
        floor_poses_list = []
        upperbound = (
            upper_floor_min_height
            if upper_floor_min_height is not None
            else floor_max_height
        )
        for pose in valid_poses:
            if (
                pose[1, 3] - camera_height >= floor_min_height
                and pose[1, 3] - camera_height < upperbound
            ):
                floor_poses_list.append(pose)
        return floor_poses_list

    def reset(self):
        """Reset the map size to double the current grid size."""

        self.map = np.zeros([self.grid_size] * 2, dtype=np.float32)

    def to_grid(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert the world coordinates to grid coordinates (integer).

        Args:
            world_coords (np.ndarray): The point in the world coordinate frame.

        Returns:
            np.ndarray: The point in the grid coordinate frame.
        """
        return np.int32((world_coords - self.pcd_min) / self.cell_size)

    def to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """Convert the grid coordinates to world coordinates.

        Args:
            grid_coords (np.ndarray): The point in the grid coordinate frame.

        Returns:
            np.ndarray: The point in the world coordinate frame.
        """
        return grid_coords * self.cell_size + self.pcd_min

    def build_local_ground_model(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        cell_size_m: float = 0.10,
        height_bin_m: float = 0.04,
        normal_radius_m: float = 0.15,
        normal_max_nn: int = 50,
        normal_max_tilt_deg: float = 20.0,
        min_points_per_voxel: int = 2,
        min_component_area_m2: float = 5.0,
        envelope_margin_m: float = 0.20,
        support_radius_cells: int = 1,
        ground_below_m: float = 0.10,
        ground_above_m: float = 0.15,
    ) -> Dict:
        """Build locally supported ground surfaces from horizontal point clusters.

        The global ground-height family is only used to define a conservative
        vertical search envelope. Within that envelope, large connected
        horizontal surfaces are retained in an X-Z-height voxel grid. This
        fills continuous SLAM floor drift without deleting every object whose
        absolute height lies between two drifted floor peaks.
        """
        points = np.asarray(floor_pcd.points, dtype=np.float64)
        if points.size == 0:
            raise RuntimeError("Cannot build a local ground model from no points")
        if cell_size_m <= 0 or height_bin_m <= 0:
            raise ValueError("Local-ground cell and height-bin sizes must be positive")

        floor_zero_level = float(floor_info["floor_zero_level"])
        ground_family = np.asarray(
            floor_info.get("ground_height_family") or [floor_zero_level],
            dtype=np.float64,
        )
        ground_family = ground_family[np.isfinite(ground_family)]
        if ground_family.size == 0:
            ground_family = np.asarray([floor_zero_level], dtype=np.float64)

        cloud_with_normals = copy.deepcopy(floor_pcd)
        cloud_with_normals.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(normal_radius_m),
                max_nn=max(3, int(normal_max_nn)),
            )
        )
        normals = np.asarray(cloud_with_normals.normals, dtype=np.float64)
        horizontal = np.abs(normals[:, 1]) >= np.cos(
            np.deg2rad(float(normal_max_tilt_deg))
        )

        height_min = float(np.min(ground_family) - envelope_margin_m)
        height_max = float(np.max(ground_family) + envelope_margin_m)
        candidate_mask = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(normals).all(axis=1)
            & horizontal
            & (points[:, 1] >= height_min)
            & (points[:, 1] <= height_max)
        )

        x_size = max(
            1,
            int(np.ceil((self.pcd_max[0] - self.pcd_min[0]) / cell_size_m)) + 1,
        )
        z_size = max(
            1,
            int(np.ceil((self.pcd_max[2] - self.pcd_min[2]) / cell_size_m)) + 1,
        )
        height_size = max(
            1,
            int(np.ceil((height_max - height_min) / height_bin_m)) + 1,
        )
        volume_shape = (height_size, z_size, x_size)
        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_points = points[candidate_indices]
        x_index = np.floor(
            (candidate_points[:, 0] - self.pcd_min[0]) / cell_size_m
        ).astype(np.int64)
        z_index = np.floor(
            (candidate_points[:, 2] - self.pcd_min[2]) / cell_size_m
        ).astype(np.int64)
        height_index = np.floor(
            (candidate_points[:, 1] - height_min) / height_bin_m
        ).astype(np.int64)
        valid = (
            (x_index >= 0)
            & (x_index < x_size)
            & (z_index >= 0)
            & (z_index < z_size)
            & (height_index >= 0)
            & (height_index < height_size)
        )
        flat_index = np.ravel_multi_index(
            (height_index[valid], z_index[valid], x_index[valid]),
            volume_shape,
        )
        voxel_counts = np.bincount(
            flat_index,
            minlength=int(np.prod(volume_shape)),
        ).reshape(volume_shape)
        occupied = voxel_counts >= max(1, int(min_points_per_voxel))

        labels, component_count = connected_components(
            occupied,
            structure=np.ones((3, 3, 3), dtype=np.uint8),
        )
        minimum_cells = max(
            1,
            int(np.ceil(min_component_area_m2 / (cell_size_m**2))),
        )
        retained_labels = []
        retained_areas = []
        for component_id in range(1, int(component_count) + 1):
            component_voxels = np.argwhere(labels == component_id)
            if component_voxels.size == 0:
                continue
            xz_cells = np.unique(component_voxels[:, [1, 2]], axis=0)
            if len(xz_cells) < minimum_cells:
                continue
            retained_labels.append(component_id)
            retained_areas.append(float(len(xz_cells) * cell_size_m**2))

        if retained_labels:
            retained = np.isin(labels, retained_labels)
        else:
            retained = np.zeros(volume_shape, dtype=bool)

        spatial_radius = max(0, int(support_radius_cells))
        if spatial_radius > 0 and np.any(retained):
            local_support = binary_dilation(
                retained,
                structure=np.ones(
                    (1, 2 * spatial_radius + 1, 2 * spatial_radius + 1),
                    dtype=bool,
                ),
            )
        else:
            local_support = retained

        vertical_radius = max(
            1,
            int(np.ceil(max(ground_below_m, ground_above_m) / height_bin_m)),
        )
        expanded_ground = binary_dilation(
            local_support,
            structure=np.ones(
                (2 * vertical_radius + 1, 1, 1),
                dtype=bool,
            ),
        )

        all_x = np.floor((points[:, 0] - self.pcd_min[0]) / cell_size_m).astype(
            np.int64
        )
        all_z = np.floor((points[:, 2] - self.pcd_min[2]) / cell_size_m).astype(
            np.int64
        )
        all_h = np.floor((points[:, 1] - height_min) / height_bin_m).astype(
            np.int64
        )
        inside_volume = (
            (all_x >= 0)
            & (all_x < x_size)
            & (all_z >= 0)
            & (all_z < z_size)
            & (all_h >= 0)
            & (all_h < height_size)
        )
        local_ground_mask = np.zeros(len(points), dtype=bool)
        local_ground_mask[inside_volume] = expanded_ground[
            all_h[inside_volume],
            all_z[inside_volume],
            all_x[inside_volume],
        ]

        support_bins = np.where(
            local_support,
            np.arange(height_size, dtype=np.int32)[:, None, None],
            -1,
        )
        highest_support_below = np.maximum.accumulate(support_bins, axis=0)
        lookup_h = np.clip(all_h, 0, height_size - 1)
        local_base_bin = np.full(len(points), -1, dtype=np.int32)
        valid_xz = (
            (all_x >= 0)
            & (all_x < x_size)
            & (all_z >= 0)
            & (all_z < z_size)
        )
        local_base_bin[valid_xz] = highest_support_below[
            lookup_h[valid_xz],
            all_z[valid_xz],
            all_x[valid_xz],
        ]
        local_base_height = np.full(len(points), np.nan, dtype=np.float64)
        has_local_base = local_base_bin >= 0
        local_base_height[has_local_base] = (
            height_min + (local_base_bin[has_local_base] + 0.5) * height_bin_m
        )

        print_status(
            "LOCAL_GROUND_MODEL",
            "OK" if retained_labels else "WARN",
            candidate_points=int(np.count_nonzero(candidate_mask)),
            occupied_voxels=int(np.count_nonzero(occupied)),
            components=int(component_count),
            retained_components=len(retained_labels),
            retained_areas_m2=[round(area, 3) for area in retained_areas],
            locally_supported_points=int(np.count_nonzero(local_ground_mask)),
            points_with_local_base=int(np.count_nonzero(has_local_base)),
            height_envelope_m=[round(height_min, 4), round(height_max, 4)],
            cell_size_m=float(cell_size_m),
            height_bin_m=float(height_bin_m),
        )
        return {
            "ground_mask": local_ground_mask,
            "base_height": local_base_height,
            "has_local_base": has_local_base,
            "retained_components": len(retained_labels),
            "retained_areas_m2": retained_areas,
        }

    def obstacles_vertices(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        height_region: List[float] = None,
        return_3d: bool = False,
    ) -> np.ndarray:
        """Get the point cloud within a certain height range from the floor as the obstacles.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            floor_info (Dict): The floor information that contains the minimum height of the floor.
            height_region (List[float], optional): Height range relative to the
                estimated ground in which points are obstacles. Defaults to
                [0.15, 1.8].

        Returns:
            floor_pcd (np.ndarray): The obstacle point cloud of the floor.
        """
        if height_region is None:
            height_region = [0.15, 1.8]
        floor_pcd = np.array(floor_pcd.points)
        floor_zero_level = float(floor_info["floor_zero_level"])
        ground_height_family = np.asarray(
            floor_info.get("ground_height_family") or [floor_zero_level],
            dtype=np.float64,
        )
        ground_height_family = ground_height_family[
            np.isfinite(ground_height_family)
        ]
        if ground_height_family.size == 0:
            ground_height_family = np.asarray([floor_zero_level])
        ceiling_clearance = float(floor_info.get("ceiling_clearance_m", 0.20))
        ceiling_height = floor_zero_level + float(floor_info["floor_height"])
        exclusion_below = float(
            floor_info.get("ground_family_exclusion_below_m", 0.10)
        )
        exclusion_above = float(
            floor_info.get("ground_family_exclusion_above_m", 0.15)
        )

        # Pose drift may reproduce the same physical floor at several global
        # heights. Exclude points in any credible ground band, then classify an
        # obstacle relative to the highest ground layer not above it. Using the
        # absolute nearest layer would let a higher drifted floor hide a real
        # low obstacle immediately below that layer.
        relative_heights = floor_pcd[:, 1, None] - ground_height_family[None, :]
        inside_ground_family = np.any(
            np.logical_and(
                relative_heights >= -exclusion_below,
                relative_heights <= exclusion_above,
            ),
            axis=1,
        )
        relative_to_layer_below = np.where(
            relative_heights >= 0.0, relative_heights, np.inf
        )
        obstacle_relative_height = np.min(relative_to_layer_below, axis=1)
        local_ground_model = floor_info.get("_local_ground_model")
        if local_ground_model is not None:
            local_ground_mask = np.asarray(
                local_ground_model["ground_mask"], dtype=bool
            )
            local_base_height = np.asarray(
                local_ground_model["base_height"], dtype=np.float64
            )
            has_local_base = np.asarray(
                local_ground_model["has_local_base"], dtype=bool
            )
            if len(local_ground_mask) != len(floor_pcd):
                raise RuntimeError("Local-ground model does not match point cloud")
            inside_ground_family = np.logical_or(
                inside_ground_family,
                local_ground_mask,
            )
            obstacle_relative_height[has_local_base] = (
                floor_pcd[has_local_base, 1] - local_base_height[has_local_base]
            )
        obstacle_height = np.logical_and(
            obstacle_relative_height >= float(height_region[0]),
            obstacle_relative_height <= float(height_region[1]),
        )
        below_ceiling = floor_pcd[:, 1] < ceiling_height - ceiling_clearance
        floor_pcd = floor_pcd[
            ~inside_ground_family & obstacle_height & below_ceiling
        ]
        if return_3d:
            return floor_pcd
        # project to 2D
        floor_pcd = floor_pcd[:, [0, 2]]
        # plot for debugging
        # plt.scatter(floor_pcd[:, 0], floor_pcd[:, 1], s=1)
        # plt.show()
        return floor_pcd

    def filter_sparse_obstacle_points(
        self,
        obstacle_points: np.ndarray,
        enabled: bool = True,
        radius_m: float = 0.10,
        min_neighbors: int = 5,
    ) -> Tuple[np.ndarray, Dict]:
        """Remove isolated 3-D obstacle points before rasterization."""
        obstacle_points = np.asarray(obstacle_points, dtype=np.float64)
        input_count = len(obstacle_points)
        if not enabled or input_count == 0:
            return obstacle_points, {
                "input_points": input_count,
                "retained_points": input_count,
                "removed_points": 0,
            }
        if radius_m <= 0 or min_neighbors < 1:
            raise ValueError("Obstacle radius filter parameters must be positive")
        if input_count < min_neighbors:
            filtered = np.empty((0, 3), dtype=np.float64)
        else:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(obstacle_points)
            _, retained_indices = cloud.remove_radius_outlier(
                nb_points=int(min_neighbors),
                radius=float(radius_m),
            )
            filtered = obstacle_points[np.asarray(retained_indices, dtype=np.int64)]
        return filtered, {
            "input_points": input_count,
            "retained_points": len(filtered),
            "removed_points": input_count - len(filtered),
        }

    def build_obstacle_confidence_maps(
        self,
        obstacle_points: np.ndarray,
        min_points_per_cell: int = 3,
        min_component_cells: int = 4,
        high_confidence_min_points: int = 8,
        high_confidence_min_vertical_span_m: float = 0.12,
        high_confidence_min_component_cells: int = 20,
    ) -> Dict:
        """Rasterize obstacle support and separate robust from weak evidence."""
        obstacle_points = np.asarray(obstacle_points, dtype=np.float64)
        shape = tuple(self.grid_size[::-1])
        counts = np.zeros(shape, dtype=np.int32)
        min_height = np.full(shape, np.inf, dtype=np.float64)
        max_height = np.full(shape, -np.inf, dtype=np.float64)
        if len(obstacle_points):
            x_cells = np.floor(
                (obstacle_points[:, 0] - self.pcd_min[0]) / self.cell_size
            ).astype(np.int64)
            z_cells = np.floor(
                (obstacle_points[:, 2] - self.pcd_min[2]) / self.cell_size
            ).astype(np.int64)
            valid = (
                (x_cells >= 0)
                & (x_cells < shape[1])
                & (z_cells >= 0)
                & (z_cells < shape[0])
                & np.isfinite(obstacle_points).all(axis=1)
            )
            rows = z_cells[valid]
            columns = x_cells[valid]
            heights = obstacle_points[valid, 1]
            np.add.at(counts, (rows, columns), 1)
            np.minimum.at(min_height, (rows, columns), heights)
            np.maximum.at(max_height, (rows, columns), heights)

        candidate = counts >= max(1, int(min_points_per_cell))
        vertical_span = np.where(candidate, max_height - min_height, 0.0)
        labels, component_count = connected_components(
            candidate,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        component_sizes = np.bincount(labels.ravel())
        cell_component_size = component_sizes[labels]
        vertical_structure = (
            vertical_span >= float(high_confidence_min_vertical_span_m)
        )
        retained = candidate & (
            (cell_component_size >= max(1, int(min_component_cells)))
            | vertical_structure
        )
        dense_large_structure = (
            (counts >= max(1, int(high_confidence_min_points)))
            & (
                cell_component_size
                >= max(1, int(high_confidence_min_component_cells))
            )
        )
        high_confidence = retained & (
            vertical_structure | dense_large_structure
        )
        low_confidence = retained & ~high_confidence
        return {
            "candidate": candidate.astype(np.uint8),
            "retained": retained.astype(np.uint8),
            "high_confidence": high_confidence.astype(np.uint8),
            "low_confidence": low_confidence.astype(np.uint8),
            "removed_sparse": (candidate & ~retained).astype(np.uint8),
            "counts": counts,
            "vertical_span": vertical_span,
            "component_count": int(component_count),
        }

    @staticmethod
    def clear_low_confidence_obstacles_near_trajectory(
        retained_map: np.ndarray,
        low_confidence_map: np.ndarray,
        trajectory_map: np.ndarray,
        enabled: bool = True,
        max_carved_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Clear only weak obstacle cells contradicted by a trajectory corridor."""
        retained = np.asarray(retained_map, dtype=bool)
        low_confidence = np.asarray(low_confidence_map, dtype=bool)
        trajectory = np.asarray(trajectory_map, dtype=bool)
        if not (
            retained.shape == low_confidence.shape == trajectory.shape
        ):
            raise ValueError("Obstacle and trajectory maps must have equal shapes")
        if not 0.0 <= max_carved_ratio <= 1.0:
            raise ValueError("max_carved_ratio must be in [0, 1]")
        removed = (
            low_confidence & trajectory
            if enabled
            else np.zeros_like(retained, dtype=bool)
        )
        carved_ratio = float(
            np.count_nonzero(removed) / max(np.count_nonzero(retained), 1)
        )
        if carved_ratio > max_carved_ratio:
            raise RuntimeError(
                "Trajectory noise clearing would remove too much obstacle "
                f"evidence: {carved_ratio:.1%} > {max_carved_ratio:.1%}. "
                "Inspect floor_obstacles_low_confidence.png and "
                "trajectory_noise_clearance_map.png instead of forcing a route."
            )
        return (retained & ~removed).astype(np.uint8), removed, carved_ratio

    def floor_region_vertices(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        ground_band: List[float] = None,
    ):
        """Get the point cloud within a certain height range from the floor.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            floor_info (Dict): The floor information that contains the minimum height of the floor.
            ground_band (List[float], optional): Lower and upper offsets around
                the estimated ground used as floor evidence.

        Returns:
            floor_pcd (np.ndarray): The obstacle point cloud of the floor.
        """
        if ground_band is None:
            ground_band = [-0.08, 0.12]
        floor_pcd = np.array(floor_pcd.points)
        floor_zero_level = float(floor_info["floor_zero_level"])
        ground_height_family = np.asarray(
            floor_info.get("ground_height_family") or [floor_zero_level],
            dtype=np.float64,
        )
        ground_height_family = ground_height_family[
            np.isfinite(ground_height_family)
        ]
        if ground_height_family.size == 0:
            ground_height_family = np.asarray([floor_zero_level])
        relative_heights = floor_pcd[:, 1, None] - ground_height_family[None, :]
        ground_evidence = np.any(
            np.logical_and(
                relative_heights >= float(ground_band[0]),
                relative_heights <= float(ground_band[1]),
            ),
            axis=1,
        )
        local_ground_model = floor_info.get("_local_ground_model")
        if local_ground_model is not None:
            local_ground_mask = np.asarray(
                local_ground_model["ground_mask"], dtype=bool
            )
            if len(local_ground_mask) != len(floor_pcd):
                raise RuntimeError("Local-ground model does not match point cloud")
            ground_evidence = np.logical_or(ground_evidence, local_ground_mask)
        floor_pcd = floor_pcd[ground_evidence]
        floor_pcd = floor_pcd[:, [0, 2]]
        return floor_pcd

    def create_occupancy_grid(
        self,
        point_cloud: np.ndarray,
        dilation_radius: int = 5,
        filter_size: int = 3,
        min_points_per_cell: int = 1,
    ) -> np.ndarray:
        """Create the occupancy grid map from the point cloud and apply dilation and smoothing.

        Args:
            point_cloud (np.ndarray): The point cloud to be converted to the occupancy grid map.
            dilation_radius (int, optional): The steps of dilation (embolding the boundary). Defaults to 5.
            filter_size (int, optional): The filter size for gaussian blurring for smoothing the grid map.
                Defaults to 3.

        Returns:
            occupancy_grid_map (np.ndarray): The occupancy grid map.
        """
        # Initialize an empty grid map with all cells marked as unoccupied (0)
        occupancy_grid_map = np.zeros(self.grid_size[::-1], dtype=np.uint8)
        if point_cloud is None or len(point_cloud) == 0:
            return occupancy_grid_map

        # handle point cloud with negative values
        point_cloud = point_cloud - self.pcd_min[[0, 2]]

        # Iterate through the point cloud and mark the corresponding cells as occupied (1)
        x_cells = np.floor(point_cloud[:, 0] / self.cell_size).astype(int)
        y_cells = np.floor(point_cloud[:, 1] / self.cell_size).astype(int)
        valid = np.logical_and.reduce(
            (
                x_cells >= 0,
                x_cells < occupancy_grid_map.shape[1],
                y_cells >= 0,
                y_cells < occupancy_grid_map.shape[0],
            )
        )
        counts = np.zeros_like(occupancy_grid_map, dtype=np.int32)
        np.add.at(counts, (y_cells[valid], x_cells[valid]), 1)
        occupancy_grid_map = (counts >= max(1, int(min_points_per_cell))).astype(
            np.uint8
        )

        if dilation_radius > 0:
            kernel_size = max(1, int(dilation_radius))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            occupancy_grid_map = cv2.dilate(
                occupancy_grid_map.astype(np.uint8),
                kernel,
                iterations=1,
            )
        # apply guassian filter to smooth the occupancy grid map
        if filter_size > 0:
            kernel_size = max(3, int(filter_size) | 1)
            occupancy_grid_map = (
                cv2.GaussianBlur(
                    occupancy_grid_map.astype(np.float32),
                    (kernel_size, kernel_size),
                    0,
                )
                > 0.25
            ).astype(np.uint8)

        return occupancy_grid_map

    def get_height_map(
        self,
        point_cloud: np.ndarray,
        floor_dir: str = None,
        height_region: np.ndarray = None,
        knn: int = 10,
    ):
        """Get the grid map where each cell stores the maximum height of the cell.

        Args:
            point_cloud (np.ndarray): The point cloud of the grid map.
            floor_dir (str, optional): The directory where the intermediate results are stored. Defaults to None.
            height_region (np.ndarray, optional): A binary mask specifying where to compute the height map. Defaults to None.
            knn (int, optional): When the cell has no points projected to it, apply knn to get the nearest height.
                Defaults to 10.

        Returns:
            _type_: _description_
        """
        # Initialize an empty grid map with all cells marked as unoccupied (0)
        height_map = np.zeros(self.grid_size[::-1], dtype=float)

        # handle point cloud with negative values
        vertices = point_cloud[:, [0, 2]] - self.pcd_min[[0, 2]]

        # Iterate through the point cloud and mark the corresponding cells as occupied (1)
        x_cells = np.floor(vertices[:, 0] / self.cell_size).astype(int)
        y_cells = np.floor(vertices[:, 1] / self.cell_size).astype(int)
        # increment the cell value by 1
        height_map[y_cells, x_cells] = point_cloud[:, 1]
        if height_region is not None:
            rows, cols = np.where(height_map > 0)
            interpolate_region = np.logical_and(height_map == 0, height_region)
            interpolate_rows, interpolate_cols = np.where(interpolate_region)
            inter_pos = np.array([interpolate_rows, interpolate_cols]).T
            height_pos = np.array([rows, cols]).T
            dist_mat = cdist(inter_pos, height_pos)
            height_ids = np.argsort(dist_mat, axis=1)
            for i, (int_row, int_col) in enumerate(
                zip(interpolate_rows, interpolate_cols)
            ):
                knn = min(knn, len(height_ids[0]))
                row_col = height_pos[height_ids[i][:knn]]
                height_list = height_map[row_col[:, 0], row_col[:, 1]]
                height_map[int_row, int_col] = np.mean(height_list)

        height_map = median_filter(height_map, size=3)

        save_height_map = (height_map.copy() / np.max(height_map) * 255).astype(
            np.uint8
        )
        save_height_map = cv2.applyColorMap(save_height_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(floor_dir, "height_map.png"), save_height_map)

        return height_map

    def get_largest_region(self, binary_map: np.ndarray) -> np.ndarray:
        """Get the largest disconnected island region in the binary map.

        Args:
            binary_map (np.ndarray): The binary map.

        Returns:
            np.ndarray: the largest region in the binary map.
        """
        # Threshold it so it becomes binary
        # ret, thresh = cv2.threshold(binary_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        input_map = (binary_map > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            input_map, 8, cv2.CV_32S
        )
        if component_count <= 1:
            return np.zeros_like(input_map, dtype=bool)

        # Label 0 is always the background. Select only among foreground labels;
        # sorting all areas fails whenever foreground happens to be larger than it.
        foreground_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(foreground_areas))
        return labels == largest_label

    def get_seeded_free_region(
        self, binary_map: np.ndarray, seed_map: np.ndarray = None
    ) -> np.ndarray:
        """Select the connected free region best supported by camera poses.

        When poses are available, prefer the component containing the most
        pose-region pixels. Area is used as a tie-breaker and as the fallback.
        This prevents a large but unvisited isolated patch from replacing the
        actually reachable room component.
        """
        input_map = (binary_map > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            input_map, 8, cv2.CV_32S
        )
        if component_count <= 1:
            return np.zeros_like(input_map, dtype=bool)
        seed_map = None if seed_map is None else np.asarray(seed_map) > 0
        best_label = None
        best_key = None
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            seed_overlap = (
                int(np.count_nonzero(np.logical_and(labels == label, seed_map)))
                if seed_map is not None
                else 0
            )
            key = (seed_overlap > 0, seed_overlap, area)
            if best_key is None or key > best_key:
                best_key = key
                best_label = label
        return labels == best_label

    def get_poses_region(
        self,
        poses_list: List[np.ndarray],
        floor_dir: str = None,
        radius: float = 0.5,
        save: bool = True,
        cluster: bool = False,
        height_cluster_eps_m: float = 0.1,
        height_tolerance_m: float = 0.1,
    ) -> np.ndarray:
        """Get the region grid map where the pose positions and their circular surrounding are. Each location is marked with a circle with a certain radius.

        Args:
            poses_list (List[np.ndarray]): The list of poses.
            floor_dir (str, optional): The directory where intermediate results are stored. Defaults to None.
            radius (float, optional): The radius of the circle around the pose location. Defaults to 0.5.
            save (bool, optional): The boolean controlling whether to store the pose region grid map as a file.
                Defaults to True.
            cluster (bool, optional): Whether to keep only the dominant camera-height cluster. This is
                intended only for legacy multi-floor data and must remain disabled for handheld data.
            height_cluster_eps_m (float, optional): DBSCAN epsilon for legacy height clustering.
            height_tolerance_m (float, optional): Height tolerance around the dominant cluster mean.

        Returns:
            pose_map (np.ndarray): The resulting pose region grid map. 1 is the pose region, 0 is the non-pose region.
        """
        poses_map = np.zeros(self.grid_size[::-1], dtype=np.uint8)
        input_pose_count = len(poses_list) if poses_list is not None else 0
        pose_source = [] if poses_list is None else poses_list
        valid_poses = [
            np.asarray(pose, dtype=np.float64)
            for pose in pose_source
            if np.asarray(pose).shape == (4, 4) and np.isfinite(pose).all()
        ]
        finite_pose_count = len(valid_poses)

        if cluster and valid_poses:
            pose_heights = np.array([pose[1, 3] for pose in valid_poses])
            clusters = DBSCAN(
                eps=max(float(height_cluster_eps_m), 1e-6), min_samples=1
            ).fit(pose_heights.reshape(-1, 1))
            labels, counts = np.unique(clusters.labels_, return_counts=True)
            dominant_label = labels[int(np.argmax(counts))]
            dominant_mask = clusters.labels_ == dominant_label
            dominant_height = float(np.mean(pose_heights[dominant_mask]))
            valid_poses = [
                pose
                for pose in valid_poses
                if abs(float(pose[1, 3]) - dominant_height)
                <= float(height_tolerance_m)
            ]

        if valid_poses:
            horizontal_positions = np.array(
                [pose[[0, 2], 3] for pose in valid_poses], dtype=np.float64
            )
            grid_positions = np.rint(
                (horizontal_positions - self.pcd_min[[0, 2]]) / self.cell_size
            ).astype(np.int32)
        else:
            grid_positions = np.empty((0, 2), dtype=np.int32)

        height, width = poses_map.shape
        in_bounds = (
            (grid_positions[:, 0] >= 0)
            & (grid_positions[:, 0] < width)
            & (grid_positions[:, 1] >= 0)
            & (grid_positions[:, 1] < height)
        )
        rasterized_positions = grid_positions[in_bounds]
        radius_cells = max(1, int(np.ceil(float(radius) / self.cell_size)))
        for pose in rasterized_positions:
            cv2.circle(poses_map, tuple(pose), radius_cells, 1, -1)

        print_status(
            "POSE_REGION",
            "OK",
            input_poses=input_pose_count,
            finite_poses=finite_pose_count,
            height_clustering=cluster,
            selected_poses=len(grid_positions),
            rasterized_poses=len(rasterized_positions),
            radius_m=float(radius),
        )
        if save:
            os.makedirs(floor_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(floor_dir, f"poses_region_map.png"), poses_map * 255
            )
        return poses_map

    def get_main_free_map(
        self,
        floor_pcd: o3d.geometry.PointCloud,
        floor_info: Dict,
        floor_dir: str = None,
        floor_poses_list: List[np.ndarray] = None,
        save: bool = True,
        cluster_pose_heights: bool = False,
        pose_region_radius: float = 0.5,
        obstacle_min_height_m: float = 0.15,
        obstacle_max_height_m: float = 1.8,
        ground_band_below_m: float = 0.08,
        ground_band_above_m: float = 0.12,
        ceiling_clearance_m: float = 0.20,
        ground_family_exclusion_below_m: float = 0.10,
        ground_family_exclusion_above_m: float = 0.15,
        obstacle_min_points_per_cell: int = 3,
        obstacle_inflation_radius_m: float = 0.38,
        pose_center_radius_m: float = 0.15,
        max_raw_obstacle_pose_center_overlap_ratio: float = 0.15,
        local_ground_enabled: bool = True,
        local_ground_cell_m: float = 0.10,
        local_ground_height_bin_m: float = 0.04,
        local_ground_normal_radius_m: float = 0.15,
        local_ground_normal_max_nn: int = 50,
        local_ground_normal_max_tilt_deg: float = 20.0,
        local_ground_min_points_per_voxel: int = 2,
        local_ground_min_component_area_m2: float = 5.0,
        local_ground_envelope_margin_m: float = 0.20,
        local_ground_support_radius_cells: int = 1,
        obstacle_radius_filter_enabled: bool = True,
        obstacle_radius_filter_m: float = 0.10,
        obstacle_radius_min_neighbors: int = 5,
        obstacle_min_component_cells: int = 4,
        obstacle_high_confidence_min_points: int = 8,
        obstacle_high_confidence_min_vertical_span_m: float = 0.12,
        obstacle_high_confidence_min_component_cells: int = 20,
        trajectory_noise_clearing_enabled: bool = True,
        trajectory_noise_clearing_radius_m: float = 0.10,
        trajectory_max_carved_obstacle_ratio: float = 0.05,
    ) -> np.ndarray:
        """Get the free region of the floor as a grid map.

        Args:
            floor_pcd (o3d.geometry.PointCloud): The floor point cloud.
            floor_info (Dict): The information dictionary of the floor.
            floor_dir (str, optional): Directory where the intermediate results will be stored. Defaults to None.
            floor_poses_list (List[np.ndarray], optional): When provided, also use the pose region as the free space. Defaults to None.
            save (bool, optional): Boolean to control whether to save the intermediate results. Defaults to True.

        Returns:
            main_free_map (np.ndarray): The free region grid map. 1 is free, 0 is occupied.
        """
        floor_info = dict(floor_info)
        floor_info["ceiling_clearance_m"] = ceiling_clearance_m
        floor_info["ground_family_exclusion_below_m"] = (
            ground_family_exclusion_below_m
        )
        floor_info["ground_family_exclusion_above_m"] = (
            ground_family_exclusion_above_m
        )
        local_ground_model = None
        if local_ground_enabled:
            local_ground_model = self.build_local_ground_model(
                floor_pcd,
                floor_info,
                cell_size_m=local_ground_cell_m,
                height_bin_m=local_ground_height_bin_m,
                normal_radius_m=local_ground_normal_radius_m,
                normal_max_nn=local_ground_normal_max_nn,
                normal_max_tilt_deg=local_ground_normal_max_tilt_deg,
                min_points_per_voxel=local_ground_min_points_per_voxel,
                min_component_area_m2=local_ground_min_component_area_m2,
                envelope_margin_m=local_ground_envelope_margin_m,
                support_radius_cells=local_ground_support_radius_cells,
                ground_below_m=ground_family_exclusion_below_m,
                ground_above_m=ground_family_exclusion_above_m,
            )
            if local_ground_model["retained_components"] > 0:
                floor_info["_local_ground_model"] = local_ground_model
        obstacle_points = self.obstacles_vertices(
            floor_pcd,
            floor_info,
            height_region=[obstacle_min_height_m, obstacle_max_height_m],
            return_3d=True,
        )
        obstacle_points, radius_filter_stats = self.filter_sparse_obstacle_points(
            obstacle_points,
            enabled=obstacle_radius_filter_enabled,
            radius_m=obstacle_radius_filter_m,
            min_neighbors=obstacle_radius_min_neighbors,
        )
        obstacle_confidence = self.build_obstacle_confidence_maps(
            obstacle_points,
            min_points_per_cell=obstacle_min_points_per_cell,
            min_component_cells=obstacle_min_component_cells,
            high_confidence_min_points=obstacle_high_confidence_min_points,
            high_confidence_min_vertical_span_m=(
                obstacle_high_confidence_min_vertical_span_m
            ),
            high_confidence_min_component_cells=(
                obstacle_high_confidence_min_component_cells
            ),
        )
        floor_region_vertices = self.floor_region_vertices(
            floor_pcd,
            floor_info,
            ground_band=[-ground_band_below_m, ground_band_above_m],
        )
        floor_occupancy_map = self.create_occupancy_grid(
            floor_region_vertices,
            dilation_radius=1,
            filter_size=0,
            min_points_per_cell=1,
        )
        if save:
            os.makedirs(floor_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(floor_dir, "floor_region_no_closing.png"),
                floor_occupancy_map.astype(np.uint8) * 255,
            )
            if local_ground_model is not None:
                points = np.asarray(floor_pcd.points)
                local_vertices = points[
                    np.asarray(local_ground_model["ground_mask"], dtype=bool)
                ][:, [0, 2]]
                local_ground_map = self.create_occupancy_grid(
                    local_vertices,
                    dilation_radius=1,
                    filter_size=0,
                    min_points_per_cell=1,
                )
                cv2.imwrite(
                    os.path.join(floor_dir, "local_ground_region.png"),
                    local_ground_map.astype(np.uint8) * 255,
                )
        # floor_occupancy_map = binary_closing(floor_occupancy_map, iterations=5)
        poses_map = self.get_poses_region(
            floor_poses_list,
            floor_dir=floor_dir,
            radius=pose_region_radius,
            save=save,
            cluster=cluster_pose_heights,
        )
        np.logical_or(floor_occupancy_map, poses_map, out=floor_occupancy_map)
        inflation_diameter_cells = max(
            1,
            int(np.ceil(2.0 * obstacle_inflation_radius_m / self.cell_size)) + 1,
        )
        if inflation_diameter_cells % 2 == 0:
            inflation_diameter_cells += 1
        pose_center_map = self.get_poses_region(
            floor_poses_list,
            radius=pose_center_radius_m,
            save=False,
            cluster=cluster_pose_heights,
        )
        trajectory_clearance_map = self.get_poses_region(
            floor_poses_list,
            radius=trajectory_noise_clearing_radius_m,
            save=False,
            cluster=cluster_pose_heights,
        )
        obstacle_before_trajectory = obstacle_confidence["retained"].astype(bool)
        raw_obstacle_map, trajectory_removed, trajectory_carved_ratio = (
            self.clear_low_confidence_obstacles_near_trajectory(
                obstacle_before_trajectory,
                obstacle_confidence["low_confidence"],
                trajectory_clearance_map,
                enabled=trajectory_noise_clearing_enabled,
                max_carved_ratio=trajectory_max_carved_obstacle_ratio,
            )
        )
        trajectory_removed_cells = int(np.count_nonzero(trajectory_removed))
        obstacle_before_trajectory_cells = int(
            np.count_nonzero(obstacle_before_trajectory)
        )
        pose_center_cells = int(np.count_nonzero(pose_center_map))
        raw_obstacle_pose_center_overlap_cells = int(
            np.count_nonzero(
                np.logical_and(raw_obstacle_map > 0, pose_center_map > 0)
            )
        )
        raw_obstacle_pose_center_overlap_ratio = float(
            raw_obstacle_pose_center_overlap_cells / max(pose_center_cells, 1)
        )
        inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inflation_diameter_cells, inflation_diameter_cells),
        )
        self.map = cv2.dilate(
            raw_obstacle_map,
            inflation_kernel,
            iterations=1,
        )
        floor_free_map = (
            floor_occupancy_map.astype(np.int16) - self.map.astype(np.int16)
        )
        floor_free_map = np.where(floor_free_map < 0, 0, floor_free_map).astype(
            np.uint8
        )
        main_free_map = self.get_seeded_free_region(floor_free_map, poses_map)
        print_status(
            "NAVIGATION_GRID",
            "OK",
            grid_width=int(self.grid_size[0]),
            grid_height=int(self.grid_size[1]),
            cell_size_m=float(self.cell_size),
            ground_evidence_cells=int(np.count_nonzero(floor_occupancy_map)),
            pose_region_cells=int(np.count_nonzero(poses_map)),
            obstacle_cells=int(np.count_nonzero(self.map)),
            raw_obstacle_cells=int(np.count_nonzero(raw_obstacle_map)),
            obstacle_candidate_cells=int(
                np.count_nonzero(obstacle_confidence["candidate"])
            ),
            obstacle_sparse_removed_cells=int(
                np.count_nonzero(obstacle_confidence["removed_sparse"])
            ),
            obstacle_high_confidence_cells=int(
                np.count_nonzero(obstacle_confidence["high_confidence"])
            ),
            obstacle_low_confidence_cells=int(
                np.count_nonzero(obstacle_confidence["low_confidence"])
            ),
            obstacle_radius_input_points=radius_filter_stats["input_points"],
            obstacle_radius_retained_points=radius_filter_stats[
                "retained_points"
            ],
            obstacle_radius_removed_points=radius_filter_stats["removed_points"],
            obstacle_temporal_support_available=False,
            trajectory_noise_clearing_enabled=trajectory_noise_clearing_enabled,
            trajectory_removed_cells=trajectory_removed_cells,
            trajectory_carved_ratio=trajectory_carved_ratio,
            trajectory_max_carved_obstacle_ratio=float(
                trajectory_max_carved_obstacle_ratio
            ),
            free_cells=int(np.count_nonzero(floor_free_map)),
            main_free_cells=int(np.count_nonzero(main_free_map)),
            obstacle_min_points_per_cell=int(obstacle_min_points_per_cell),
            obstacle_inflation_radius_m=float(obstacle_inflation_radius_m),
            pose_center_radius_m=float(pose_center_radius_m),
            pose_center_cells=pose_center_cells,
            raw_obstacle_pose_center_overlap_cells=(
                raw_obstacle_pose_center_overlap_cells
            ),
            raw_obstacle_pose_center_overlap_ratio=(
                raw_obstacle_pose_center_overlap_ratio
            ),
            max_raw_obstacle_pose_center_overlap_ratio=float(
                max_raw_obstacle_pose_center_overlap_ratio
            ),
        )
        if save:
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles_candidate.png"),
                obstacle_confidence["candidate"] * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles_sparse_removed.png"),
                obstacle_confidence["removed_sparse"] * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles_high_confidence.png"),
                obstacle_confidence["high_confidence"] * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles_low_confidence.png"),
                obstacle_confidence["low_confidence"] * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "trajectory_noise_clearance_map.png"),
                trajectory_clearance_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles_trajectory_removed.png"),
                trajectory_removed.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles_raw.png"),
                raw_obstacle_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_obstacles.png"),
                self.map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_region.png"),
                floor_occupancy_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_free.png"),
                floor_free_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "floor_free_main.png"),
                main_free_map.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(floor_dir, "poses_center_map.png"),
                pose_center_map.astype(np.uint8) * 255,
            )
        if (
            pose_center_cells > 0
            and raw_obstacle_pose_center_overlap_ratio
            > max_raw_obstacle_pose_center_overlap_ratio
        ):
            raise RuntimeError(
                "Navigation rejected because the reconstructed obstacle cloud "
                "contradicts the recorded camera trajectory: raw obstacles cover "
                f"{raw_obstacle_pose_center_overlap_ratio:.1%} of the "
                f"{pose_center_radius_m:.2f} m trajectory center region, above "
                f"the {max_raw_obstacle_pose_center_overlap_ratio:.1%} limit. "
                "This indicates unresolved SLAM ghosting/pose drift; inspect "
                "poses_center_map.png, floor_obstacles_raw.png, and "
                "ground_estimation.json before using routes."
            )
        return main_free_map

    def get_top_down_rgb_map(
        self, floor_pcd: o3d.geometry.PointCloud, floor_info: Dict, floor_dir: str
    ) -> np.ndarray:
        """Generate the top-down RGB map of the floor.

        Args:
            floor_pcd (o3d.geometry.PointCloud): Floor point cloud.
            floor_info (Dict): Floor information dictionary.
            floor_dir (str): Directory where the intermediate results will be stored.

        Returns:
            top_down_bgr (np.ndarray): (H, W, 3) The top-down BGR map of the floor.
        """
        floor_point_cloud = np.asarray(floor_pcd.points)
        floor_point_color = np.asarray(floor_pcd.colors)
        zero_level = floor_info["floor_zero_level"]
        floor_height = floor_info["floor_height"]
        mask = floor_point_cloud[:, 1] < zero_level + 1.5  # floor_height - 0.5
        floor_point_cloud = floor_point_cloud[mask]
        floor_point_color = floor_point_color[mask]
        floor_grid_vertices = np.int32(
            (floor_point_cloud - self.pcd_min) / self.cell_size
        )
        top_down = np.zeros([self.grid_size[1], self.grid_size[0], 3], dtype=np.uint8)
        top_down_height = -1000 * np.ones(
            [self.grid_size[1], self.grid_size[0]], dtype=np.float32
        )
        for p_i, floor_p in enumerate(floor_grid_vertices):
            col, height, row = floor_p
            if height > top_down_height[row, col]:
                top_down[row, col] = (floor_point_color[p_i] * 255).astype(np.uint8)
                top_down_height[row, col] = height
        top_down = median_filter(top_down, size=3)

        top_down_bgr = cv2.cvtColor(top_down, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(floor_dir, "top_down_rgb.png"), top_down_bgr)
        # cv2.imwrite(os.path.join(floor_dir, "top_down_rgb.png"), top_down)
        return top_down_bgr

    def get_voronoi_graph(
        self,
        main_free_map: np.ndarray,
        map_rgb: np.ndarray,
        floor_dir: str,
        floor_id: str,
        name: str = "",
        height_map: np.ndarray = None,
    ) -> nx.Graph:
        """Generate the Voronoi Graph of the floor based on the free space map substracting obstacle map.

        Args:
            main_free_map (np.ndarray): Free space map.
            map_rgb (np.ndarray): RGB map of the floor.
            floor_dir (str): Directory where the intermediate results will be stored.
            floor_id (str): The floor id. For example, "0" for the first floor.
            name (str, optional): The name is used for saving the intermediate results. Defaults to "".
            height_map (np.ndarray, optional): Height map used for getting the 3D positional attributes of
                the Voronoi node. Defaults to None.

        Returns:
            nx.Graph: Resulting Voronoi graph.
        """
        boundary_map = binary_erosion(main_free_map, iterations=1).astype(np.uint8)
        boundary_map = main_free_map - boundary_map
        cv2.imwrite(
            os.path.join(floor_dir, f"boundary_{name}.png"),
            boundary_map.astype(np.uint8) * 255,
        )
        rows, cols = np.where(boundary_map == 1)
        boundaries = np.array(list(zip(rows, cols)))
        voronoi = Voronoi(boundaries)

        fig_free = main_free_map.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)
        map_bgr = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
        fig = map_bgr.copy()
        vertices = []
        if height_map is None:
            height_map = np.ones_like(boundary_map) * self.pcd_min[1]
        voronoi_graph = nx.Graph()
        for simplex in voronoi.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                continue
            src, tar = voronoi.vertices[simplex]
            if (
                src[0] < 0
                or src[0] >= fig.shape[0]
                or src[1] < 0
                or src[1] >= fig.shape[1]
                or tar[0] < 0
                or tar[0] >= fig.shape[0]
                or tar[1] < 0
                or tar[1] >= fig.shape[1]
            ):
                continue
            if (
                main_free_map[int(src[0]), int(src[1])] == 0
                or main_free_map[int(tar[0]), int(tar[1])] == 0
            ):
                continue
            cv2.line(
                fig,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.line(
                fig_free,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.circle(fig, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)
            cv2.circle(fig_free, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)

            # check if src and tar already exist in the graph
            if (src[0], src[1], floor_id) not in voronoi_graph.nodes:
                height = height_map[int(src[0]), int(src[1])]
                voronoi_graph.add_node(
                    (src[0], src[1], floor_id),
                    pos=(
                        src[1] * self.cell_size + self.pcd_min[0],
                        height,
                        src[0] * self.cell_size + self.pcd_min[2],
                    ),
                    floor_id=floor_id,
                )
            if (tar[0], tar[1], floor_id) not in voronoi_graph.nodes:
                height = height_map[int(tar[0]), int(tar[1])]
                voronoi_graph.add_node(
                    (tar[0], tar[1], floor_id),
                    pos=(
                        tar[1] * self.cell_size + self.pcd_min[0],
                        height,
                        tar[0] * self.cell_size + self.pcd_min[2],
                    ),
                    floor_id=floor_id,
                )
            # check if the edge already exists
            if (src[0], src[1], floor_id) not in voronoi_graph[
                (tar[0], tar[1], floor_id)
            ]:
                voronoi_graph.add_edge(
                    (src[0], src[1], floor_id),
                    (tar[0], tar[1], floor_id),
                    dist=np.linalg.norm(src - tar),
                )

        cv2.imwrite(os.path.join(floor_dir, f"vor_{name}.png"), fig)
        cv2.imwrite(os.path.join(floor_dir, f"vor_free_{name}.png"), fig_free)
        # vertices = np.array(vertices)
        return voronoi_graph

    def sparsify_graph(self, floor_graph: nx.Graph, resampling_dist: float = 0.4):
        """
        Sparsify a topology graph by removing nodes with degree 2.
        This algorithm first starts at degree-one nodes (dead ends) and
        removes all degree-two nodes until confluence nodes are found.
        Next, we find close pairs of higher-order degree nodes and
        delete all nodes if the shortest path between two nodes consists
        only of degree-two nodes.
        Args:
            floor_graph (nx.Graph): graph to sparsify
        Returns:
            nx.Graph: sparsified graph
        """
        graph = copy.deepcopy(floor_graph)

        if len(graph.nodes) < 10:
            return graph
        # all nodes with degree 1 or 3+
        new_node_candidates = [
            node for node in list(graph.nodes) if (graph.degree(node) != 2)
        ]

        new_graph = nx.Graph()
        for i, node in enumerate(new_node_candidates):
            new_graph.add_node(
                node,
                pos=graph.nodes[node]["pos"],
                floor_id=graph.nodes[node]["floor_id"],
            )
        new_nodes = set(new_graph.nodes)
        new_nodes_list = list(new_graph.nodes)

        print(
            f"Getting paths between all nodes. Node number: {len(new_node_candidates)}/{len(graph.nodes)}"
        )

        st = time.time()
        all_path_dense_graph = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))
        ed = time.time()
        print("time for computing all pairs shortest path: ", ed - st, " seconds")
        sampled_edges_to_add = list()
        pbar = tqdm(range(len(new_graph.nodes)), desc="Sparsifying graph")
        for i in pbar:
            inner_pbar = tqdm(
                range(len(new_graph.nodes)), desc="Sparsifying graph", leave=False
            )
            for j in inner_pbar:
                if i >= j:
                    continue
                # Go through all edges along path and extract dist
                node1 = new_nodes_list[i]
                node2 = new_nodes_list[j]
                try:
                    path = all_path_dense_graph[node1][node2]
                    for node in path[1:-1]:
                        if graph.degree(node) > 2:
                            break
                    else:
                        sampled_edges_to_add.append(
                            (
                                path[0],
                                path[-1],
                                np.linalg.norm(np.array(path[0]) - np.array(path[-1])),
                            )
                        )
                        dist = [
                            graph.edges[path[k], path[k + 1]]["dist"]
                            for k in range(len(path) - 1)
                        ]
                        mov_agg_dist = 0
                        predecessor = path[0]
                        # connect the nodes if there is a path between them that does not go through any other of the new nodes
                        if (
                            len(path)
                            and len(set(path[1:-1]).intersection(new_nodes)) == 0
                        ):
                            for cand_idx, cand_node in enumerate(path[1:-1]):
                                mov_agg_dist += dist[cand_idx]
                                if mov_agg_dist * self.cell_size > resampling_dist:
                                    sampled_edges_to_add.append(
                                        (
                                            predecessor,
                                            cand_node,
                                            np.linalg.norm(
                                                np.array(predecessor)
                                                - np.array(cand_node)
                                            ),
                                        )
                                    )
                                    predecessor = cand_node
                                    mov_agg_dist = 0
                                else:
                                    continue
                            sampled_edges_to_add.append(
                                (
                                    predecessor,
                                    path[-1],
                                    np.linalg.norm(
                                        np.array(predecessor) - np.array(path[-1])
                                    ),
                                )
                            )
                except:
                    continue

        for edge_param in sampled_edges_to_add:
            k, l, dist = edge_param
            if k not in new_graph.nodes:
                new_graph.add_node(
                    k, pos=graph.nodes[k]["pos"], floor_id=graph.nodes[k]["floor_id"]
                )
            if l not in new_graph.nodes:
                new_graph.add_node(
                    l, pos=graph.nodes[l]["pos"], floor_id=graph.nodes[l]["floor_id"]
                )
            new_graph.add_edge(k, l, dist=dist)

        self.floor_graph = new_graph
        return new_graph

    def trim_graph(self, graph: nx.Graph, trim_deg: int = 1):
        """
        Trim a graph by removing all nodes with degree 1.
        Args:
            graph (nx.Graph): graph to trim
            trim_deg (int): degree of nodes to remove
        Returns:
            nx.Graph: trimmed graph
        """
        graph = copy.deepcopy(graph)
        while True:
            degree_one_nodes = [
                node for node in list(graph.nodes) if (graph.degree(node) == trim_deg)
            ]
            if len(degree_one_nodes) == 0:
                break
            graph.remove_nodes_from(degree_one_nodes)
        return graph

    def draw_graph_on_map(
        self, map: np.ndarray, graph: nx.Graph, floor_dir: str, name: str
    ) -> np.ndarray:
        """Draw graph on the grid map

        Args:
            map (np.ndarray): The grid map where the graph will be drawn on.
            graph (nx.Graph): The drawn graph.
            floor_dir (str): The directory where the intermediate results are stored.
            name (str): The saved name of the intermediate results.

        Returns:
            fig (np.ndarray): The map where the graph is drawn.
        """
        fig = map.copy().astype(np.uint8)
        if np.max(map) <= 1:
            fig = (map.copy() * 255).astype(np.uint8)
        if len(fig.shape) == 2:
            fig = cv2.cvtColor(fig, cv2.COLOR_GRAY2BGR)
        for edge in graph.edges:
            v1, v2 = edge
            v1 = np.int32(v1)[::-1][1:]
            v2 = np.int32(v2)[::-1][1:]
            cv2.line(fig, tuple(v1), tuple(v2), (0, 0, 255), 1)
            cv2.circle(fig, tuple(v1), 2, (255, 0, 0), -1)
            cv2.circle(fig, tuple(v2), 2, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(floor_dir, f"{name}.png"), fig)
        # cv2.imshow(f"graph on map {name}", fig)
        # cv2.waitKey()
        return fig

    def get_stairs_objects(
        self, objects_list: List[Object], clip_model: Any, clip_feat_dim: int
    ) -> List[Object]:
        """Get the objects that are stairs or staircase.

        Args:
            objects_list (List[Object]): The list of Object instances.
            clip_model (Any): The clip model.
            clip_feat_dim (int): The clip model dimension.

        Returns:
            stairs_list (List[Object]): The list of stairs objects.
        """
        text_feats = get_text_feats_62_templates(
            MATTERPORT_LABELS_40, clip_model, clip_feat_dim
        )
        stair_id = MATTERPORT_LABELS_40.index("stairs")
        print(stair_id)
        stairs_list = []
        for obj in objects_list:
            emb = obj.embedding
            sim_mat = emb @ text_feats.T
            label_id = np.argmax(sim_mat)
            if label_id == stair_id:  # staircase
                print(obj.object_id)
                stairs_list.append(obj)
            elif obj.name == "staircase":
                stairs_list.append(obj)

        return stairs_list

    def get_stairs_graph_with_poses_v2(
        self, floor: Floor, floor_id: str, poses_list: List[np.ndarray], floor_dir: str
    ) -> nx.Graph:
        """A way of getting stairs graph based on the poses in between floors.

        Args:
            floor (Floor): Floor instance.
            floor_id (str): Floor id.
            poses_list (List[np.ndarray]): The list of poses in the whole scene.
            floor_dir (str): The directory where the intermediate results are stored.

        Returns:
            spares_voronoi_graph (nx.Graph): The stairs graph.
        """
        if poses_list is None or len(poses_list) < 2:
            self.has_stairs = False
            return nx.Graph()

        floor_pcd = floor.pcd
        floor_info = {
            "floor_zero_level": floor.floor_zero_level,
            "floor_height": floor.floor_height,
            "ground_height_family": list(
                floor.ground_height_family or [floor.floor_zero_level]
            ),
        }
        top_down = self.get_top_down_rgb_map(floor_pcd, floor_info, floor_dir)
        pose_heights = np.array([pose[1, 3] - 1.5 for pose in poses_list])
        hist = np.histogram(pose_heights, bins=100)
        min_peak_height = 0.3 * np.max(hist[0])
        input_hist = np.concatenate([[np.min(hist[0])], hist[0], [np.min(hist[0])]])
        peaks, _ = find_peaks(input_hist, height=min_peak_height)
        peaks = peaks - 1
        plt.clf()
        plt.plot(hist[1][:-1], hist[0])
        plt.plot(hist[1][peaks], hist[0][peaks], "x")
        plt.savefig(os.path.join(floor_dir, "hist_peaks.png"))
        if floor_id >= len(peaks) - 1:
            self.has_stairs = False
            return nx.Graph()

        height_min = hist[1][peaks[int(floor_id)] + 1]
        height_max = hist[1][peaks[int(floor_id) + 1]]
        if pose_heights[0] > pose_heights[-1]:
            print("reverse pose sequence")
            pose_heights = pose_heights[::-1]
            poses_list = poses_list[::-1]
        min_st_id = 0
        if pose_heights[0] > height_min:
            for i, h in enumerate(pose_heights):
                if h <= height_min:
                    min_st_id = i
                    break

        st_id = None
        ed_id = None
        for hi, h in enumerate(pose_heights):
            if h > height_min and hi > min_st_id:
                st_id = hi
                break
        for hi, h in enumerate(pose_heights):
            if h > height_max:
                ed_id = hi
                break
        if st_id is None or ed_id is None or ed_id <= st_id:
            self.has_stairs = False
            return nx.Graph()
        for hi in range(st_id, ed_id):
            if pose_heights[hi] < height_min:
                st_id = hi

        hist = np.histogram(pose_heights[st_id:ed_id], bins=100)
        plt.clf()
        plt.plot(hist[1][:-1], hist[0])
        plt.savefig(os.path.join(floor_dir, "hist_stairs.png"))
        buffer_down = 5
        buffer_up = 20
        st_id = np.max([st_id - buffer_down, 0])
        ed_id = ed_id + buffer_up
        stairs_poses = poses_list[st_id:ed_id]
        tar_poses = []
        for pose in stairs_poses:
            pos = pose[:3, 3]
            tar_poses.append(pos)
            pos_2d = np.round((pos - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(top_down, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(floor_dir, "top_down_rgb_poses.png"), top_down)

        min_height = np.min([pos[1] for pos in tar_poses])
        # cv2.imshow("stairs poses", top_down)
        # cv2.waitKey()
        voronoi_graph = nx.Graph()
        last_pos = None
        for pos in tar_poses:
            pos_2d = (pos - self.pcd_min) / self.cell_size
            voronoi_graph.add_node(
                (pos_2d[2], pos_2d[0], floor_id),
                pos=(
                    pos[0],
                    pos[1] - min_height + floor_info["floor_zero_level"],
                    pos[2],
                ),
                floor_id=floor_id,
            )
            if last_pos is not None:
                voronoi_graph.add_edge(
                    (pos_2d[2], pos_2d[0], floor_id),
                    (last_pos[2], last_pos[0], floor_id),
                    dist=np.linalg.norm(pos_2d - last_pos),
                )
                last_pos = pos_2d
            else:
                last_pos = pos_2d

        # sparse_voronoi_graph = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
        sparse_voronoi_graph = voronoi_graph
        self.has_stairs = True
        self.draw_graph_on_map(
            top_down, sparse_voronoi_graph, floor_dir, "sparse_vor_stairs"
        )
        return sparse_voronoi_graph

    def get_stairs_graph_with_poses(
        self, floor: Floor, floor_id: str, poses_list: List[np.ndarray], floor_dir: str
    ) -> nx.Graph:
        """A way of getting stairs graph based on the poses in between floors.

        Args:
            floor (Floor): Floor instance.
            floor_id (str): Floor id.
            poses_list (List[np.ndarray]): The list of poses in the whole scene.
            floor_dir (str): The directory where the intermediate results are stored.

        Returns:
            spares_voronoi_graph (nx.Graph): The stairs graph.
        """
        floor_pcd = floor.pcd
        floor_info = {
            "floor_zero_level": floor.floor_zero_level,
            "floor_height": floor.floor_height,
            "ground_height_family": list(
                floor.ground_height_family or [floor.floor_zero_level]
            ),
        }
        top_down = self.get_top_down_rgb_map(floor_pcd, floor_info, floor_dir)
        pose_heights = np.array([pose[1, 3] for pose in poses_list])
        min_height = np.min(pose_heights)
        max_height = np.max(pose_heights)
        if max_height - min_height < 0.5:
            self.has_stairs = False
            return nx.Graph()

        pose_heights = np.array([pose[1, 3] for pose in poses_list])
        clusters = DBSCAN(eps=0.1).fit(pose_heights.reshape(-1, 1))
        labels, counts = np.unique(clusters.labels_, return_counts=True)
        id = np.argmax(counts)
        mask = clusters.labels_ == labels[id]
        major_height = np.mean(pose_heights[mask])
        major_height_poses = [pose[:3, 3] for pose, m in zip(poses_list, mask) if m]
        for l in labels:
            print(np.mean(pose_heights[clusters.labels_ == l]))
        non_min_height_poses = [
            pose[:3, 3] for pose in poses_list if pose[1, 3] > major_height + 0.1
        ]
        # clusters = DBSCAN(eps=0.5).fit(np.array(non_min_height_poses))
        # labels, counts = np.unique(clusters.labels_, return_counts=True)
        # id = np.argmax(counts)
        # mask = clusters.labels_ == labels[id]
        # non_min_height_poses = [pose for pose, m in zip(non_min_height_poses, mask) if m]
        # find the closest pose in major_height_poses to the non_min_height_poses
        dist_mat = cdist(np.array(non_min_height_poses), np.array(major_height_poses))
        row, col = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        non_min_height_poses.append(major_height_poses[col])
        poses_sorted_by_height = sorted(non_min_height_poses, key=lambda x: x[1])
        if len(poses_sorted_by_height) > 0:
            self.has_stairs = True

        all_poses = np.array([pose[:3, 3] for pose in poses_list])
        top_down_copy = top_down.copy()
        top_down_copy_1 = top_down.copy()

        for pose in non_min_height_poses:
            pos_2d = np.round((pose - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(top_down, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(floor_dir, "top_down_rgb_poses.png"), top_down)

        for pose in major_height_poses:
            pos_2d = np.round((pose - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(
                top_down_copy, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1
            )
            cv2.imwrite(
                os.path.join(floor_dir, "top_down_rgb_major_poses.png"), top_down_copy
            )

        rest_poses = []
        for pos in [*major_height_poses, *non_min_height_poses]:
            cdist_mat = cdist(np.array([pos]), np.array(all_poses))
            min_dist = np.min(cdist_mat) if len(cdist_mat) > 0 else np.inf
            if min_dist > 0.1:
                rest_poses.append(pos)
        print(rest_poses)

        for pose in all_poses:
            pos_2d = np.round((pose - self.pcd_min) / self.cell_size).astype(np.int32)
            cv2.circle(
                top_down_copy_1, tuple(np.int32(pos_2d[[0, 2]])), 2, (255, 0, 0), -1
            )
            cv2.imwrite(
                os.path.join(floor_dir, "top_down_rgb_all_poses.png"), top_down_copy_1
            )

        tar_poses = []
        last_pose = np.ones(3) * -np.inf
        for pose in poses_sorted_by_height:
            if pose[1] > last_pose[1]:
                last_pose = pose
                tar_poses.append(last_pose)
                continue
            elif pose[1] < last_pose[1]:
                continue
            if np.linalg.norm(pose[:2] - last_pose[:2]) > 0.1:
                last_pose = pose
                tar_poses.append(last_pose)

        voronoi_graph = nx.Graph()
        last_pos = None
        for pos in tar_poses:
            pos_2d = (pos - self.pcd_min) / self.cell_size
            print(
                pos_2d[2],
                pos_2d[0],
                pos[0],
                pos[1] - min_height + floor_info["floor_zero_level"],
                pos[2],
            )
            voronoi_graph.add_node(
                (pos_2d[2], pos_2d[0], floor_id),
                pos=(
                    pos[0],
                    pos[1] - min_height + floor_info["floor_zero_level"],
                    pos[2],
                ),
                floor_id=floor_id,
            )
            if last_pos is not None:
                voronoi_graph.add_edge(
                    (pos_2d[2], pos_2d[0], floor_id),
                    (last_pos[2], last_pos[0], floor_id),
                    dist=np.linalg.norm(pos_2d - last_pos),
                )
                last_pos = pos_2d
            else:
                last_pos = pos_2d

        # sparse_voronoi_graph = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
        sparse_voronoi_graph = voronoi_graph
        self.has_stairs = True
        self.draw_graph_on_map(
            top_down, sparse_voronoi_graph, floor_dir, "sparse_vor_stairs"
        )
        return sparse_voronoi_graph

    def get_stairs_graph(
        self,
        floor_id: str,
        objects_list: List[Object],
        floor_dir: str,
        clip_model: Any = None,
        clip_feat_dim: int = None,
    ) -> nx.Graph:
        """Get the objects that are stairs or staircase, generate a region of stairs and compute the stairs graph.

        Args:
            floor_id (str): Floor id.
            objects_list (List[Object]): The list of Object instances.
            floor_dir (str): The directory where the intermediate results are stored.
            clip_model (Any): The clip model.
            clip_feat_dim (int): The clip model dimension.

        Returns:
            spares_voronoi_graph (nx.Graph): The stairs graph.
        """
        stairs_list = self.get_stairs_objects(objects_list, clip_model, clip_feat_dim)

        floor_stairs_list = []
        for obj in stairs_list:
            obj_min = np.min(np.array(obj.pcd.points), axis=0)
            obj_max = np.max(np.array(obj.pcd.points), axis=0)
            if obj_min[1] > self.pcd_min[1] and obj_max[1] < self.pcd_max[1]:
                floor_stairs_list.append(obj)

        # compute the stairs navigation graph (voronoi graph)
        if len(floor_stairs_list) == 0:
            self.has_stairs = False
            return None
        self.has_stairs = True

        # merge all points in floor_stairs_list
        floor_stairs_pcd = np.concatenate(
            [obj.pcd.points for obj in floor_stairs_list], axis=0
        )

        # save floor_stairs_pcd to ply
        floor_stairs_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(floor_stairs_pcd)
        )
        save_path = os.path.join(floor_dir, f"floor_{floor_id}_stairs.ply")
        o3d.io.write_point_cloud(save_path, floor_stairs_pcd)

        stairs_map_no_close = self.create_occupancy_grid(
            np.asarray(floor_stairs_pcd.points)[:, [0, 2]]
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(floor_dir, "stairs_map_no_closing.png"),
            stairs_map_no_close.astype(np.uint8) * 255,
        )
        stairs_map_no_erosion = binary_closing(
            stairs_map_no_close, iterations=50, structure=np.ones((3, 3))
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(floor_dir, "stairs_map_no_erosion.png"),
            stairs_map_no_erosion.astype(np.uint8) * 255,
        )
        stairs_map = binary_erosion(stairs_map_no_erosion, iterations=5).astype(
            np.uint8
        )
        cv2.imwrite(
            os.path.join(floor_dir, "stairs_map.png"), stairs_map.astype(np.uint8) * 255
        )
        stairs_height = self.get_height_map(
            np.asarray(floor_stairs_pcd.points), floor_dir, stairs_map_no_erosion, 10
        )
        stairs_voronoi = self.get_voronoi_graph(
            stairs_map,
            stairs_map,
            floor_dir,
            floor_id,
            name="stairs",
            height_map=stairs_height,
        )
        # sparse_stairs_voronoi = self.sparsify_graph(stairs_voronoi, resampling_dist=0.4)
        sparse_stairs_voronoi = stairs_voronoi
        sparse_stairs_voronoi = self.trim_graph(sparse_stairs_voronoi, trim_deg=1)
        self.draw_graph_on_map(
            stairs_map, sparse_stairs_voronoi, floor_dir, "sparse_vor_stairs"
        )
        return sparse_stairs_voronoi

    def get_floor_graph(
        self,
        floor: Floor,
        floor_poses_list: List[np.ndarray],
        floor_dir: str,
        cluster_pose_heights: bool = False,
        pose_region_radius: float = 0.5,
        navigation_config: Dict = None,
    ) -> nx.Graph:
        """Get the floor voronoi graph.

        Args:
            floor (Floor): Floor instance.
            floor_poses_list (List[np.ndarray]): The list of poses on the floor.
            floor_dir (str): The directory where the intermediate results are stored.

        Returns:
            spares_voronoi_graph (nx.Graph): The floor graph.
        """
        # compute the floor navigation graph (voronoi graph)
        floor_info = {
            "floor_zero_level": floor.floor_zero_level,
            "floor_height": floor.floor_height,
            "ground_height_family": list(
                floor.ground_height_family or [floor.floor_zero_level]
            ),
        }
        navigation_config = navigation_config or {}
        main_free_map = self.get_main_free_map(
            floor.pcd,
            floor_info,
            floor_dir,
            floor_poses_list,
            cluster_pose_heights=cluster_pose_heights,
            pose_region_radius=pose_region_radius,
            obstacle_min_height_m=float(
                navigation_config.get("obstacle_min_height_m", 0.15)
            ),
            obstacle_max_height_m=float(
                navigation_config.get("obstacle_max_height_m", 1.8)
            ),
            ground_band_below_m=float(
                navigation_config.get("ground_band_below_m", 0.08)
            ),
            ground_band_above_m=float(
                navigation_config.get("ground_band_above_m", 0.12)
            ),
            ceiling_clearance_m=float(
                navigation_config.get("ceiling_clearance_m", 0.20)
            ),
            ground_family_exclusion_below_m=float(
                navigation_config.get("ground_family_exclusion_below_m", 0.10)
            ),
            ground_family_exclusion_above_m=float(
                navigation_config.get("ground_family_exclusion_above_m", 0.15)
            ),
            obstacle_min_points_per_cell=int(
                navigation_config.get("obstacle_min_points_per_cell", 3)
            ),
            obstacle_inflation_radius_m=float(
                navigation_config.get("obstacle_inflation_radius_m", 0.38)
            ),
            pose_center_radius_m=float(
                navigation_config.get("pose_center_radius_m", 0.15)
            ),
            max_raw_obstacle_pose_center_overlap_ratio=float(
                navigation_config.get(
                    "max_raw_obstacle_pose_center_overlap_ratio", 0.15
                )
            ),
            local_ground_enabled=bool(
                navigation_config.get("local_ground_enabled", True)
            ),
            local_ground_cell_m=float(
                navigation_config.get("local_ground_cell_m", 0.10)
            ),
            local_ground_height_bin_m=float(
                navigation_config.get("local_ground_height_bin_m", 0.04)
            ),
            local_ground_normal_radius_m=float(
                navigation_config.get("local_ground_normal_radius_m", 0.15)
            ),
            local_ground_normal_max_nn=int(
                navigation_config.get("local_ground_normal_max_nn", 50)
            ),
            local_ground_normal_max_tilt_deg=float(
                navigation_config.get("local_ground_normal_max_tilt_deg", 20.0)
            ),
            local_ground_min_points_per_voxel=int(
                navigation_config.get("local_ground_min_points_per_voxel", 2)
            ),
            local_ground_min_component_area_m2=float(
                navigation_config.get("local_ground_min_component_area_m2", 5.0)
            ),
            local_ground_envelope_margin_m=float(
                navigation_config.get("local_ground_envelope_margin_m", 0.20)
            ),
            local_ground_support_radius_cells=int(
                navigation_config.get("local_ground_support_radius_cells", 1)
            ),
            obstacle_radius_filter_enabled=bool(
                navigation_config.get("obstacle_radius_filter_enabled", True)
            ),
            obstacle_radius_filter_m=float(
                navigation_config.get("obstacle_radius_filter_m", 0.10)
            ),
            obstacle_radius_min_neighbors=int(
                navigation_config.get("obstacle_radius_min_neighbors", 5)
            ),
            obstacle_min_component_cells=int(
                navigation_config.get("obstacle_min_component_cells", 4)
            ),
            obstacle_high_confidence_min_points=int(
                navigation_config.get("obstacle_high_confidence_min_points", 8)
            ),
            obstacle_high_confidence_min_vertical_span_m=float(
                navigation_config.get(
                    "obstacle_high_confidence_min_vertical_span_m", 0.12
                )
            ),
            obstacle_high_confidence_min_component_cells=int(
                navigation_config.get(
                    "obstacle_high_confidence_min_component_cells", 20
                )
            ),
            trajectory_noise_clearing_enabled=bool(
                navigation_config.get(
                    "trajectory_noise_clearing_enabled", True
                )
            ),
            trajectory_noise_clearing_radius_m=float(
                navigation_config.get(
                    "trajectory_noise_clearing_radius_m", 0.10
                )
            ),
            trajectory_max_carved_obstacle_ratio=float(
                navigation_config.get(
                    "trajectory_max_carved_obstacle_ratio", 0.05
                )
            ),
        ).astype(np.uint8)
        minimum_navigable_area_m2 = float(
            navigation_config.get("minimum_navigable_area_m2", 0.50)
        )
        navigable_area_m2 = float(
            np.count_nonzero(main_free_map) * self.cell_size * self.cell_size
        )
        if navigable_area_m2 < minimum_navigable_area_m2:
            raise RuntimeError(
                "Reliable navigable area is too small after ground and obstacle "
                f"filtering: {navigable_area_m2:.3f} m^2 < "
                f"{minimum_navigable_area_m2:.3f} m^2. Check "
                "ground_estimation.json, floor_obstacles_raw.png, and "
                "floor_free_main.png before generating routes."
            )
        top_down = self.get_top_down_rgb_map(floor.pcd, floor_info, floor_dir)
        voronoi_graph = self.get_voronoi_graph(
            main_free_map, top_down, floor_dir, int(floor.floor_id)
        )
        sparse_floor_voronoi = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
        # sparse_floor_voronoi = voronoi_graph

        self.draw_graph_on_map(
            top_down, sparse_floor_voronoi, floor_dir, "sparse_vor_rgb"
        )
        self.draw_graph_on_map(
            main_free_map, sparse_floor_voronoi, floor_dir, "sparse_vor"
        )

        # save necessary data
        self.top_down = top_down
        self.sparse_floor_voronoi = sparse_floor_voronoi
        return sparse_floor_voronoi

    def connect_voronoi_graphs(
        self, src_graph: nx.Graph, tar_graph: nx.Graph, floor_dir: str = None
    ) -> nx.Graph:
        """Connect two graphs by finding the closest node in the source graph to the target graph.

        Args:
            src_graph (nx.Graph): The source graph.
            tar_graph (nx.Graph): The target graph.
            floor_dir (str, optional): The directory where intermediate results will be saved. Defaults to None.

        Returns:
            tar_graph (nx.Graph): The resulting graph.
        """
        if src_graph.number_of_nodes() == 0:
            return tar_graph
        if tar_graph.number_of_nodes() == 0:
            return src_graph

        # find closest node from src_graph to tar_graph
        floor_nodes = [
            (i, tar_graph.nodes[node]["pos"])
            for i, node in enumerate(tar_graph.nodes)
            if tar_graph.degree(node) > 1
        ]
        if not floor_nodes:
            floor_nodes = [
                (i, tar_graph.nodes[node]["pos"])
                for i, node in enumerate(tar_graph.nodes)
            ]
        floor_node_ids = [node[0] for node in floor_nodes]
        floor_nodes = np.array([node[1] for node in floor_nodes])
        stairs_nodes = np.array(
            [
                src_graph.nodes[node]["pos"] for node in src_graph.nodes
            ]  # if src_graph.degree(node) == 1]
        )
        print(np.unique(stairs_nodes[:, 1], return_counts=True))
        print(np.unique(floor_nodes[:, 1], return_counts=True))

        dist_mat = cdist(stairs_nodes, floor_nodes)
        row, col = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        col = floor_node_ids[col]
        stair_node = list(src_graph.nodes)[row]
        floor_node = list(tar_graph.nodes)[col]
        print("stairs, floor")
        print(stair_node, floor_node)
        print(src_graph.nodes[stair_node]["pos"], tar_graph.nodes[floor_node]["pos"])
        print(
            src_graph.nodes[stair_node]["floor_id"],
            tar_graph.nodes[floor_node]["floor_id"],
        )

        tar_graph = nx.compose(tar_graph, src_graph)
        tar_graph.add_edge(
            stair_node,
            floor_node,
            dist=np.linalg.norm(np.array(stair_node[:2]) - np.array(floor_node[:2])),
        )
        if floor_dir is not None:
            fig = self.draw_graph_on_map(
                self.top_down, tar_graph, floor_dir, "vor_rgb_combined_highlighted"
            )
            src_pos = np.int32([stair_node[1], stair_node[0]])
            tar_pos = np.int32([floor_node[1], floor_node[0]])
            cv2.circle(fig, tuple(src_pos), 3, (0, 255, 0), -1)
            cv2.circle(fig, tuple(tar_pos), 3, (0, 255, 0), -1)
            cv2.imwrite(
                os.path.join(floor_dir, "vor_rgb_combined_highlighted.png"), fig
            )
        return tar_graph

    def connect_stairs_and_floor_graphs(
        self,
        sparse_stairs_voronoi: nx.Graph,
        sparse_floor_voronoi: nx.Graph,
        floor_dir: str,
    ):
        """Connect stairs graph to the floor graph by finding the closest node in the source graph to the target graph.

        Args:
            sparse_stairs_voronoi (nx.Graph): The stairs graph.
            sparse_floor_voronoi (nx.Graph): The floor graph.
            floor_dir (str, optional): The directory where intermediate results will be saved. Defaults to None.

        Returns:
            sparse_floor_voronoi (nx.Graph): The resulting graph.
        """
        # connect sparse_floor_voronoi and sparse_stairs_voronoi
        if not self.has_stairs:
            return sparse_floor_voronoi
        sparse_floor_voronoi = self.connect_voronoi_graphs(
            sparse_stairs_voronoi, sparse_floor_voronoi, floor_dir
        )
        self.draw_graph_on_map(
            self.top_down, sparse_floor_voronoi, floor_dir, "vor_rgb_combined"
        )
        self.sparse_floor_voronoi = sparse_floor_voronoi
        return sparse_floor_voronoi

    @staticmethod
    def save_voronoi_graph(graph: nx.Graph, floor_dir: str, name: str) -> None:
        """Save the Voronoi graph to a json file.

        Args:
            graph (nx.Graph): The Voronoi graph.
            floor_dir (str): The directory where the intermediate results are stored.
            name (str): The name of the file.
        """
        graph_path = os.path.join(floor_dir, f"{name}_graph.json")
        graph_json = nx.node_link_data(graph)
        with open(graph_path, "w") as f:
            json.dump(graph_json, f, indent=4)
