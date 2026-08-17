import unittest

import numpy as np
import open3d as o3d

from hovsg.utils.graph_utils import merge_point_clouds_list, pcd_denoise_dbscan


def make_cloud(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([[0.2, 0.4, 0.8]]), (len(points), 1))
    )
    return cloud


class GraphUtilsTest(unittest.TestCase):
    def test_keep_all_clusters_preserves_disconnected_object_parts(self):
        cluster_a = np.asarray(
            [[x * 0.01, 0.0, z * 0.01] for x in range(5) for z in range(5)]
        )
        cluster_b = cluster_a + np.asarray([0.0, 0.0, 0.35])
        cloud = make_cloud(np.vstack([cluster_a, cluster_b]))

        largest_only = pcd_denoise_dbscan(
            cloud, eps=0.04, min_points=3, keep_all_clusters=False
        )
        all_supported = pcd_denoise_dbscan(
            cloud,
            eps=0.04,
            min_points=3,
            keep_all_clusters=True,
            min_cluster_points=5,
            min_cluster_ratio=0.02,
        )

        self.assertEqual(len(largest_only.points), 25)
        self.assertEqual(len(all_supported.points), 50)

    def test_merge_downsamples_without_discarding_components(self):
        cluster_a = np.asarray(
            [[x * 0.005, 0.0, z * 0.005] for x in range(4) for z in range(4)]
        )
        cluster_b = cluster_a + np.asarray([0.0, 0.0, 0.20])
        merged = merge_point_clouds_list(
            [make_cloud(cluster_a), make_cloud(cluster_b)], voxel_size=0.01
        )

        self.assertGreaterEqual(len(merged.points), 16)
        bounds = np.asarray(merged.get_max_bound()) - np.asarray(merged.get_min_bound())
        self.assertGreater(bounds[2], 0.19)


if __name__ == "__main__":
    unittest.main()
