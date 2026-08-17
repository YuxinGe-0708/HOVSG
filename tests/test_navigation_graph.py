import tempfile
import unittest

import numpy as np
import open3d as o3d

from hovsg.graph.floor import Floor
from hovsg.graph.navigation_graph import NavigationGraph


def make_pose(x, y, z):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [x, y, z]
    return pose


def make_floor_cloud():
    points = []
    for x in np.linspace(0.0, 4.0, 21):
        for z in np.linspace(0.0, 4.0, 21):
            points.append([x, 0.0, z])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
    return cloud


class NavigationGraphHandheldTest(unittest.TestCase):
    def setUp(self):
        self.cloud = make_floor_cloud()
        self.floor = Floor("0", name="floor_0")
        self.floor.pcd = self.cloud
        self.floor.floor_zero_level = 0.0
        self.floor.floor_height = 3.0
        self.nav = NavigationGraph(self.cloud, cell_size=0.1)

    def test_single_floor_keeps_horizontal_trajectory_across_height_changes(self):
        poses = [
            make_pose(0.5 + index * 0.3, index * 0.15, 2.0)
            for index in range(11)
        ]

        selected = self.nav.get_floor_poses(
            self.floor, poses, camera_height=1.5, filter_by_height=False
        )
        pose_map = self.nav.get_poses_region(
            selected, radius=0.15, save=False, cluster=False
        )

        self.assertEqual(len(selected), len(poses))
        for pose in poses:
            grid = self.nav.to_grid(pose[:3, 3])[[0, 2]]
            self.assertEqual(pose_map[grid[1], grid[0]], 1)

    def test_largest_region_never_selects_background(self):
        binary_map = np.zeros((20, 20), dtype=np.uint8)
        binary_map[1:10, 1:10] = 1
        binary_map[15:18, 15:18] = 1

        largest = self.nav.get_largest_region(binary_map)

        self.assertEqual(int(np.count_nonzero(largest)), 81)
        self.assertTrue(largest[5, 5])
        self.assertFalse(largest[16, 16])

    def test_seeded_region_prefers_camera_supported_component(self):
        binary_map = np.zeros((30, 30), dtype=np.uint8)
        binary_map[1:20, 1:20] = 1
        binary_map[22:27, 22:27] = 1
        seed_map = np.zeros_like(binary_map)
        seed_map[23:25, 23:25] = 1

        selected = self.nav.get_seeded_free_region(binary_map, seed_map)

        self.assertEqual(int(np.count_nonzero(selected)), 25)
        self.assertTrue(selected[23, 23])
        self.assertFalse(selected[5, 5])

    def test_obstacle_grid_requires_density_support(self):
        points = np.array(
            [
                [0.51, 0.51],
                [0.52, 0.52],
                [1.01, 1.01],
                [1.02, 1.02],
                [1.03, 1.03],
            ]
        )

        grid = self.nav.create_occupancy_grid(
            points,
            dilation_radius=0,
            filter_size=0,
            min_points_per_cell=3,
        )

        sparse_cell = np.floor((np.array([0.51, 0.51]) - [0.0, 0.0]) / 0.1).astype(
            int
        )
        dense_cell = np.floor((np.array([1.01, 1.01]) - [0.0, 0.0]) / 0.1).astype(
            int
        )
        self.assertEqual(grid[sparse_cell[1], sparse_cell[0]], 0)
        self.assertEqual(grid[dense_cell[1], dense_cell[0]], 1)

    def test_radius_filter_removes_isolated_obstacle_points(self):
        cluster = np.asarray(
            [
                [1.00, 0.50, 1.00],
                [1.01, 0.50, 1.00],
                [1.00, 0.51, 1.01],
                [0.99, 0.50, 1.00],
                [1.00, 0.49, 0.99],
                [1.01, 0.51, 1.01],
            ]
        )
        isolated = np.asarray([[3.0, 0.5, 3.0]])

        filtered, stats = self.nav.filter_sparse_obstacle_points(
            np.vstack((cluster, isolated)),
            radius_m=0.05,
            min_neighbors=3,
        )

        self.assertEqual(len(filtered), len(cluster))
        self.assertEqual(stats["removed_points"], 1)

    def test_trajectory_clears_only_low_confidence_obstacles(self):
        points = []
        for x, z in ((1.01, 1.01), (1.11, 1.01), (1.01, 1.11), (1.11, 1.11)):
            points.extend([[x, 0.40, z], [x + 0.01, 0.40, z], [x, 0.40, z + 0.01]])
        points.extend(
            [
                [2.01, 0.20, 2.01],
                [2.01, 0.40, 2.01],
                [2.01, 0.60, 2.01],
            ]
        )
        points.extend([[3.01, 0.40, 3.01]] * 3)
        confidence = self.nav.build_obstacle_confidence_maps(
            np.asarray(points),
            min_points_per_cell=3,
            min_component_cells=4,
            high_confidence_min_points=8,
            high_confidence_min_vertical_span_m=0.12,
            high_confidence_min_component_cells=20,
        )
        trajectory = np.logical_or(
            confidence["low_confidence"],
            confidence["high_confidence"],
        )

        cleaned, removed, ratio = (
            self.nav.clear_low_confidence_obstacles_near_trajectory(
                confidence["retained"],
                confidence["low_confidence"],
                trajectory,
                max_carved_ratio=1.0,
            )
        )

        self.assertEqual(int(np.count_nonzero(confidence["low_confidence"])), 4)
        self.assertEqual(int(np.count_nonzero(confidence["high_confidence"])), 1)
        self.assertEqual(int(np.count_nonzero(confidence["removed_sparse"])), 1)
        self.assertEqual(int(np.count_nonzero(removed)), 4)
        self.assertEqual(int(np.count_nonzero(cleaned)), 1)
        self.assertAlmostEqual(ratio, 0.8)
        with self.assertRaisesRegex(RuntimeError, "remove too much"):
            self.nav.clear_low_confidence_obstacles_near_trajectory(
                confidence["retained"],
                confidence["low_confidence"],
                trajectory,
                max_carved_ratio=0.1,
            )

    def test_ground_family_excludes_drift_layers_but_keeps_table(self):
        points = np.array(
            [
                [0.0, -0.40, 0.0],
                [1.0, 0.00, 1.0],
                [2.0, 0.36, 2.0],
                [2.5, 0.78, 2.5],
                [2.7, 0.20, 2.7],
                [3.0, 2.70, 3.0],
            ],
            dtype=np.float64,
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        info = {
            "floor_zero_level": 0.0,
            "floor_height": 2.7,
            "ground_height_family": [-0.40, 0.0, 0.36],
            "ground_family_exclusion_below_m": 0.10,
            "ground_family_exclusion_above_m": 0.15,
        }

        ground = self.nav.floor_region_vertices(cloud, info)
        obstacles = self.nav.obstacles_vertices(cloud, info)

        self.assertEqual(len(ground), 3)
        self.assertEqual(len(obstacles), 2)
        np.testing.assert_allclose(obstacles, [[2.5, 2.5], [2.7, 2.7]])

    def test_local_ground_model_follows_continuous_floor_drift(self):
        floor_points = []
        for x in np.linspace(0.0, 4.0, 41):
            for z in np.linspace(0.0, 4.0, 41):
                floor_points.append([x, 0.10 * x, z])
        obstacle_points = np.repeat([[2.0, 0.70, 2.0]], 5, axis=0)
        points = np.vstack((np.asarray(floor_points), obstacle_points))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        nav = NavigationGraph(cloud, cell_size=0.1)
        info = {
            "floor_zero_level": 0.0,
            "floor_height": 3.0,
            "ground_height_family": [0.0, 0.40],
            "ground_family_exclusion_below_m": 0.10,
            "ground_family_exclusion_above_m": 0.15,
        }

        model = nav.build_local_ground_model(
            cloud,
            info,
            cell_size_m=0.10,
            height_bin_m=0.04,
            normal_radius_m=0.25,
            min_points_per_voxel=1,
            min_component_area_m2=5.0,
            envelope_margin_m=0.10,
        )
        info["_local_ground_model"] = model
        ground = nav.floor_region_vertices(cloud, info)
        obstacles = nav.obstacles_vertices(cloud, info)

        floor_ground_ratio = np.mean(model["ground_mask"][: len(floor_points)])
        self.assertGreater(floor_ground_ratio, 0.95)
        self.assertGreaterEqual(model["retained_components"], 1)
        self.assertGreater(len(ground), int(0.95 * len(floor_points)))
        self.assertEqual(len(obstacles), len(obstacle_points))
        np.testing.assert_allclose(obstacles, np.repeat([[2.0, 2.0]], 5, axis=0))

    def test_floor_round_trip_preserves_ground_height_family(self):
        self.floor.vertices = np.asarray(
            self.floor.pcd.get_axis_aligned_bounding_box().get_box_points()
        )
        self.floor.ground_height_family = [-0.40, 0.0, 0.36]

        with tempfile.TemporaryDirectory() as directory:
            self.floor.save(directory)
            loaded = Floor("0")
            loaded.load(directory)

        self.assertEqual(loaded.ground_height_family, [-0.40, 0.0, 0.36])

    def test_navigation_rejects_obstacles_on_recorded_path_center(self):
        points = np.vstack(
            (
                np.asarray(self.cloud.points),
                np.repeat([[2.0, 0.50, 2.0]], 5, axis=0),
            )
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        nav = NavigationGraph(cloud, cell_size=0.1)

        with self.assertRaisesRegex(RuntimeError, "contradicts the recorded"):
            nav.get_main_free_map(
                cloud,
                {
                    "floor_zero_level": 0.0,
                    "floor_height": 3.0,
                    "ground_height_family": [0.0],
                },
                floor_poses_list=[make_pose(2.0, 1.0, 2.0)],
                save=False,
                pose_region_radius=0.2,
                pose_center_radius_m=0.05,
                obstacle_inflation_radius_m=0.0,
                max_raw_obstacle_pose_center_overlap_ratio=0.10,
            )



if __name__ == "__main__":
    unittest.main()
