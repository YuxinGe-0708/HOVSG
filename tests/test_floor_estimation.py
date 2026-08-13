import unittest

import numpy as np
import open3d as o3d

from hovsg.utils.floor_estimation import estimate_single_floor_geometry


def horizontal_plane(y, x_size, z_size, spacing=0.08):
    xs = np.arange(0.0, x_size + spacing, spacing)
    zs = np.arange(0.0, z_size + spacing, spacing)
    xx, zz = np.meshgrid(xs, zs)
    return np.column_stack(
        (xx.ravel(), np.full(xx.size, y, dtype=np.float64), zz.ravel())
    )


class FloorEstimationTest(unittest.TestCase):
    def test_ground_beats_table_and_low_outliers(self):
        rng = np.random.default_rng(7)
        ground = horizontal_plane(0.0, 4.0, 4.0)
        table = horizontal_plane(0.8, 1.5, 1.5, spacing=0.04)
        ceiling = horizontal_plane(2.7, 4.0, 4.0)
        low_outliers = np.column_stack(
            (
                rng.uniform(0.0, 4.0, 300),
                rng.normal(-0.55, 0.02, 300),
                rng.uniform(0.0, 4.0, 300),
            )
        )
        points = np.vstack((ground, table, ceiling, low_outliers))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        poses = [np.array([x, 1.2, 2.0]) for x in np.linspace(0.4, 3.6, 20)]

        result = estimate_single_floor_geometry(
            cloud,
            poses,
            {
                "ground_min_plane_inliers": 50,
                "ground_min_pose_proximity_ratio": 0.10,
                "single_floor_min_room_height_m": 2.0,
                "single_floor_max_room_height_m": 3.5,
            },
        )

        self.assertFalse(result["fallback_used"])
        self.assertAlmostEqual(result["floor_zero_level"], 0.0, delta=0.06)
        self.assertAlmostEqual(result["floor_upper_level"], 2.7, delta=0.08)

    def test_unverified_ground_is_reported_for_fail_closed_caller(self):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.array([[0.0, -0.5, 0.0], [1.0, 0.4, 1.0], [2.0, 1.3, 2.0]])
        )

        result = estimate_single_floor_geometry(
            cloud,
            [np.array([0.0, 1.0, 0.0])],
            {
                "ground_min_plane_inliers": 50,
                "ground_allow_quantile_fallback": False,
            },
        )

        self.assertTrue(result["fallback_used"])
        self.assertFalse(result["fallback_allowed"])

    def test_large_drifted_ground_layers_form_family_but_table_does_not(self):
        ground_layers = [
            horizontal_plane(height, 4.0, 4.0)
            for height in (-0.40, 0.0, 0.36)
        ]
        table = horizontal_plane(0.78, 1.2, 1.2, spacing=0.04)
        ceiling = horizontal_plane(2.7, 4.0, 4.0)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.vstack((*ground_layers, table, ceiling))
        )
        poses = [np.array([x, 1.15, 2.0]) for x in np.linspace(0.4, 3.6, 20)]

        result = estimate_single_floor_geometry(
            cloud,
            poses,
            {
                "ground_min_plane_inliers": 50,
                "ground_min_pose_proximity_ratio": 0.10,
                "ground_family_min_coverage_ratio": 0.35,
                "ground_family_min_connected_fraction": 0.65,
            },
        )

        family = result["ground_height_family"]
        for expected in (-0.40, 0.0, 0.36):
            self.assertTrue(any(abs(height - expected) < 0.06 for height in family))
        self.assertFalse(any(abs(height - 0.78) < 0.06 for height in family))


if __name__ == "__main__":
    unittest.main()
