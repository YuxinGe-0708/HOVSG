import tempfile
import unittest

import numpy as np
import open3d as o3d
import torch

from hovsg.utils.feature_map_memory import (
    LowMemoryFeatureAccumulator,
    load_geometry_cache,
    save_geometry_cache,
)


class LowMemoryFeatureAccumulatorTests(unittest.TestCase):
    def test_accumulates_duplicates_and_normalizes_in_chunks(self):
        accumulator = LowMemoryFeatureAccumulator(
            point_count=4,
            feature_dim=3,
            dtype="float16",
            normalization_chunk_points=2,
        )
        accumulator.add(
            torch.tensor([0, 0, 2]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ),
        )

        features = accumulator.finalize()

        np.testing.assert_allclose(
            features[0].astype(np.float32),
            np.array([2 ** -0.5, 2 ** -0.5, 0.0]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            features[2].astype(np.float32),
            np.array([0.0, 0.0, 1.0]),
            atol=1e-3,
        )
        np.testing.assert_array_equal(features[1], np.zeros(3))
        self.assertEqual(features.dtype, np.float16)

    def test_float16_allocation_is_half_the_float32_feature_storage(self):
        half = LowMemoryFeatureAccumulator(10, 8, dtype="float16")
        single = LowMemoryFeatureAccumulator(10, 8, dtype="float32")
        # Both instances have the same small float32 count tensor.
        count_bytes = 10 * 4
        self.assertEqual(
            half.allocated_bytes - count_bytes,
            (single.allocated_bytes - count_bytes) // 2,
        )


class GeometryCacheTests(unittest.TestCase):
    def test_cache_load_requires_an_exact_signature(self):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        cloud.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        signature = {"version": 1, "voxel_size": 0.03}
        metadata = {
            "signature": signature,
            "frame_indices": [0, 2],
        }

        with tempfile.TemporaryDirectory() as directory:
            save_geometry_cache(directory, cloud, metadata)
            cached = load_geometry_cache(directory, signature)
            self.assertIsNotNone(cached)
            cached_cloud, cached_metadata = cached
            self.assertEqual(len(cached_cloud.points), 2)
            self.assertEqual(cached_metadata["frame_indices"], [0, 2])
            self.assertIsNone(
                load_geometry_cache(
                    directory,
                    {"version": 1, "voxel_size": 0.05},
                )
            )


if __name__ == "__main__":
    unittest.main()
