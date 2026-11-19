"""Synthetic RGB-D regression cases; no checkpoints, GPU or robot required."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
import torch

from hovsg.dataloader.generic import RGBDDataset
from hovsg.dataloader.real_robot import RealRobotDataset, ROS_Z_UP_TO_HOV_Y_UP
from hovsg.utils.feature_map_memory import LowMemoryFeatureAccumulator
from hovsg.utils.geometry_cache_inputs import dataset_geometry_signature
from robot_capture.camera_geometry import projection_metadata
from hovsg.models.sam_clip_feats_extractor import extract_feats_per_pixel, extract_feats_raw
from hovsg.dataloader.scannet import ScannetDataset
from hovsg.dataloader.hm3dsem import HM3DSemDataset
from hovsg.dataloader.replica import ReplicaDataset
from hovsg.utils.frame_matching import match_frame_paths


class SyntheticDataset(RGBDDataset):
    def _get_data_list(self):
        return []

    def __init__(self):
        super().__init__({"root_dir": "", "transforms": None})
        self.scale = 1000.0
        self.depth_intrinsics = np.eye(3)


def make_scene(root, **overrides):
    camera = dict(width=3, height=2, fx=2.0, fy=2.0, cx=1.0, cy=0.0,
                  depth_units_per_meter=1000.0, min_depth_m=0.0, max_depth_m=3.0,
                  rgb_depth_aligned=True, world_frame="hov_y_up",
                  pose_convention="T_world_camera_optical", image_rectified=True)
    camera.update(overrides)
    (root / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    for name in ("rgb", "depth", "pose"):
        (root / name).mkdir(exist_ok=True)
    Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(root / "rgb/000001.png")
    Image.fromarray(np.full((2, 3), 1000, np.uint16)).save(root / "depth/000001.png")
    np.savetxt(root / "pose/000001.txt", np.eye(4))
    return RealRobotDataset({"root_dir": str(root), "transforms": None})


class ProjectionTests(unittest.TestCase):
    def test_per_pixel_depth_filter_and_feature_indices(self):
        dataset = SyntheticDataset()
        raw = np.array([[1000, np.nan, np.inf], [0, 4000, 2000]])
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        pose = np.eye(4)
        pose[:3, 3] = [10, 20, 30]
        cloud, pixels = dataset.create_pcd(rgb, raw, pose, filter_distance=3,
                                            return_pixel_indices=True)
        np.testing.assert_array_equal(pixels, [0, 5])
        np.testing.assert_allclose(cloud.points, [[10, 20, 31], [14, 22, 32]])
        np.testing.assert_allclose(cloud.colors, rgb.reshape(-1, 3)[pixels] / 255)

    def test_uint8_mask_selects_pixels_without_scaling_depth(self):
        dataset = SyntheticDataset()
        cloud = dataset.create_pcd(np.array([[255, 0]], np.uint8),
                                   np.array([[1000, 2000]]), mask_img=True)
        np.testing.assert_allclose(cloud.points, [[0, 0, 1]])

    def test_mask_resolution_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "resolutions"):
            SyntheticDataset().create_pcd(np.ones((1, 1)), np.ones((2, 2)),
                                           mask_img=True)

    def test_no_pose_and_empty_depth_are_supported(self):
        cloud, pixels = SyntheticDataset().create_pcd(
            np.zeros((1, 2, 3)), np.zeros((1, 2)), return_pixel_indices=True)
        self.assertTrue(cloud.is_empty())
        self.assertEqual(pixels.size, 0)

    def test_two_view_features_associate_to_same_world_point(self):
        dataset = SyntheticDataset()
        dataset.depth_intrinsics[0, 2] = 1
        rgb = np.zeros((1, 3, 3), np.uint8)
        first, first_pixels = dataset.create_pcd(
            rgb, [[0, 1000, 0]], return_pixel_indices=True)
        pose = np.eye(4)
        pose[0, 3] = 1
        second, second_pixels = dataset.create_pcd(
            rgb, [[1000, 0, 0]], pose, return_pixel_indices=True)
        np.testing.assert_array_equal(first_pixels, [1])
        np.testing.assert_array_equal(second_pixels, [0])
        np.testing.assert_allclose(first.points, second.points)
        accumulator = LowMemoryFeatureAccumulator(1, 2, dtype="float32")
        tree = cKDTree(np.asarray(first.points))
        for cloud, pixels, features in (
            (first, first_pixels, [[0, 0], [1, 0], [0, 0]]),
            (second, second_pixels, [[0, 1], [0, 0], [0, 0]]),
        ):
            distances, ids = tree.query(cloud.points)
            self.assertTrue(np.all(distances < 1e-8))
            accumulator.add(torch.from_numpy(ids), torch.tensor(features)[pixels])
        np.testing.assert_allclose(accumulator.finalize(), [[2**-0.5, 2**-0.5]])

    def test_mask_association_rejects_distant_global_points(self):
        dataset = SyntheticDataset()
        global_cloud = o3d.geometry.PointCloud()
        global_cloud.points = o3d.utility.Vector3dVector([[0, 0, 1], [100, 0, 1]])
        masks = [{"segmentation": np.array([[True, True]])},
                 {"segmentation": np.array([[False, False]])}]
        projected = dataset.create_3d_masks(
            masks, np.array([[1000, 1000]]), global_cloud,
            cKDTree(global_cloud.points), np.eye(4), association_distance=0.05)
        self.assertEqual(len(projected), 2)
        np.testing.assert_allclose(projected[0].points, [[0, 0, 1]])
        self.assertTrue(projected[1].is_empty())


class RealRobotAlignmentTests(unittest.TestCase):
    def test_zero_and_nonfinite_depth_never_emit_points(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_scene(Path(directory))
            cloud, ids = dataset.create_pcd(np.zeros((2, 3, 3)),
                [[0, np.inf, np.nan], [1000, 4000, 2000]], return_pixel_indices=True)
            np.testing.assert_array_equal(ids, [3, 5])
            self.assertTrue(np.isfinite(cloud.points).all())

    def test_direct_projection_rejects_resizing_or_stale_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_scene(Path(directory))
            for rgb, depth in ((np.zeros((1, 3, 3)), np.ones((2, 3))),
                               (np.zeros((1, 2, 3)), np.ones((1, 2)))):
                with self.assertRaisesRegex(ValueError, "camera.json resolution"):
                    dataset.create_pcd(rgb, depth)

    def test_bad_calibration_and_duplicate_frame_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for overrides in ({"fx": 0}, {"max_depth_m": float("nan")},
                              {"distortion_coefficients": [0, 0, 0, float("nan")] }):
                with self.assertRaises(ValueError):
                    make_scene(root, **overrides)
            make_scene(root)
            Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(root / "rgb/000001.jpg")
            with self.assertRaisesRegex(ValueError, "Duplicate frame ID"):
                make_scene(root)

    def test_distortion_geometry_and_masks_share_rays(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_scene(Path(directory), image_rectified=False,
                                 distortion_model="plumb_bob",
                                 distortion_coefficients=[0.1, -0.02, 0.01, 0, 0])
            raw = np.full((2, 3), 1000)
            rgb = np.zeros((2, 3, 3), np.uint8)
            cloud = dataset.create_pcd(rgb, raw)
            mask = np.zeros((2, 3), np.uint8)
            mask[1, 2] = 255
            selected = dataset.create_pcd(mask, raw, mask_img=True)
            expected = cv2.undistortPoints(np.array([[[2., 1.]]]),
                dataset.depth_intrinsics, dataset.distortion_coefficients).reshape(2)
            np.testing.assert_allclose(np.asarray(selected.points)[0], [*expected, 1], atol=1e-6)
            np.testing.assert_allclose(np.asarray(cloud.points)[5], selected.points[0])
            empty = dataset.create_pcd(mask * 0, raw, mask_img=True)
            self.assertTrue(empty.is_empty())

    def test_world_frame_conversion_applies_to_translation_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_scene(root, world_frame="ros_z_up")
            pose = np.eye(4)
            pose[:3, 3] = [1, 2, 3]
            np.savetxt(root / "pose/000001.txt", pose)
            np.testing.assert_allclose(dataset[0][2], ROS_Z_UP_TO_HOV_Y_UP @ pose)

    def test_rectified_rays_return_to_tf_optical_frame(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_scene(Path(directory), rectification_rotation=rotation.tolist())
            points = dataset._unproject_pixels(np.array([2]), np.array([0]), np.array([1]))
            np.testing.assert_allclose(points, np.array([[0.5, 0, 1]]) @ rotation)

    def test_cache_changes_on_pose_calibration_selection_and_projection_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_scene(root)
            previous = dataset_geometry_signature(dataset)
            self.assertEqual(previous, dataset_geometry_signature(dataset))
            changes = [
                lambda: np.savetxt(root / "pose/000001.txt", np.eye(4) * 2),
                lambda: (root / "camera.json").write_text("{}", encoding="utf-8"),
                lambda: (root / "valid_frame_ids.txt").write_text("000001\n"),
                lambda: setattr(dataset, "planar_pose_height_m", 2.5),
            ]
            for change in changes:
                change()
                current = dataset_geometry_signature(dataset)
                self.assertNotEqual(previous, current)
                previous = current


class CameraInfoTests(unittest.TestCase):
    def test_raw_uses_k_and_rectified_uses_p(self):
        info = SimpleNamespace(k=[100, 0, 10, 0, 110, 20, 0, 0, 1],
            p=[200, 0, 30, 0, 0, 210, 40, 0, 0, 0, 1, 0], r=np.eye(3).ravel())
        self.assertEqual(projection_metadata(info, image_rectified=False)["fx"], 100)
        self.assertEqual(projection_metadata(info, image_rectified=True)["fx"], 200)
        info.p = [0] * 12
        with self.assertRaises(ValueError):
            projection_metadata(info, image_rectified=True)

    def test_empty_sam_result_skips_clip_without_cuda(self):
        generator = SimpleNamespace(generate=lambda image: [])
        for extractor in (extract_feats_per_pixel, extract_feats_raw):
            result = extractor(np.zeros((2, 3, 3), np.uint8), generator, None, None)
            self.assertEqual(result, (None, None, [], None))


class FrameMatchingTests(unittest.TestCase):
    def test_missing_frames_do_not_shift_sorted_pairs(self):
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            match_frame_paths(["0.jpg", "2.jpg"], ["0.png", "1.png"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            match_frame_paths(["0.jpg", "0.png"], ["0.png"])
        self.assertEqual(match_frame_paths(["10.jpg", "2.jpg"], ["2.png", "10.png"]),
                         [("2.jpg", "2.png"), ("10.jpg", "10.png")])

    def test_scannet_path_without_trailing_slash_and_missing_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("color", "depth", "pose", "intrinsic"):
                (root / folder).mkdir()
            for name in ("intrinsic_color.txt", "intrinsic_depth.txt"):
                np.savetxt(root / "intrinsic" / name, np.eye(4))
            (root / "color/0.jpg").touch()
            (root / "depth/0.png").touch()
            (root / "pose/0.txt").touch()
            dataset = ScannetDataset({"root_dir": str(root), "transforms": None})
            self.assertEqual(len(dataset), 1)
            (root / "pose/0.txt").rename(root / "pose/1.txt")
            with self.assertRaisesRegex(ValueError, "frame IDs"):
                ScannetDataset({"root_dir": str(root), "transforms": None})

    def test_hm3d_rejects_missing_pose_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("rgb", "depth", "pose"):
                (root / folder).mkdir()
            (root / "rgb/scene_000000.png").touch()
            (root / "depth/scene_000000.png").touch()
            with self.assertRaisesRegex(ValueError, "frame IDs"):
                HM3DSemDataset({"root_dir": str(root), "transforms": None})

    def test_replica_uses_source_frame_id_for_trajectory_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            (root / "results").mkdir(parents=True)
            camera = dict(w=1, h=1, fx=1, fy=1, cx=0, cy=0, scale=1000)
            (root.parent / "cam_params.json").write_text(json.dumps({"camera": camera}))
            poses = []
            for frame_id in range(3):
                pose = np.eye(4)
                pose[0, 3] = frame_id * 10
                poses.append(" ".join(map(str, pose.ravel())))
            (root / "traj.txt").write_text("\n".join(poses))
            for frame_id in (0, 2):
                Image.fromarray(np.zeros((1, 1, 3), np.uint8)).save(
                    root / "results" / f"frame{frame_id:06d}.jpg")
                Image.fromarray(np.ones((1, 1), np.uint16)).save(
                    root / "results" / f"depth{frame_id:06d}.png")
            dataset = ReplicaDataset({"root_dir": str(root), "transforms": None})
            self.assertEqual(dataset[1][2][0, 3], 20)


if __name__ == "__main__":
    unittest.main()
