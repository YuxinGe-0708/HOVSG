import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from hovsg.dataloader.generic import RGBDDataset
from hovsg.utils.pose_utils import (
    as_pose_matrix,
    camera_tilt_deg_y_up,
    project_camera_pose_to_planar_y_up,
)
from hovsg.utils.runtime_status import print_status


# ROS world coordinates: (x forward, y left, z up)
# Convert to HOV-SG coordinates: (x left, y up, z forword)
ROS_Z_UP_TO_HOV_Y_UP = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# When RTAB-Map is run with the RGB optical frame itself as ``frame_id`` and
# no base_link exists, its initial map axes are usually the camera optical
# axes: x right, y down, z forward. This maps those axes to ROS z-up axes
# (x forward, y left, z up) for a level, forward-facing camera.
CAMERA_OPTICAL_INITIAL_TO_ROS_Z_UP = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
CAMERA_OPTICAL_INITIAL_TO_HOV_Y_UP = (
    ROS_Z_UP_TO_HOV_Y_UP[:3, :3] @ CAMERA_OPTICAL_INITIAL_TO_ROS_Z_UP
)

# /mnt/data/real_camera/robot_recordings/
# └── test/                              # Corresponds to main.split
#     └── scene_001/                     # Corresponds to main.scene_id
#         ├── camera.json                # Camera intrinsics and depth settings
#         ├── rgb/                       # RGB images
#         │   ├── 000001.png
#         │   ├── 000002.png
#         │   ├── 000003.png
#         │   └── ...
#         ├── depth/                     # Aligned raw depth images
#         │   ├── 000001.png
#         │   ├── 000002.png
#         │   ├── 000003.png
#         │   └── ...
#         └── pose/                      # T_map_camera pose matrices
#             ├── 000001.txt
#             ├── 000002.txt
#             ├── 000003.txt
#             └── ...


class RealRobotDataset(RGBDDataset):
    """
    Read the synchronized and aligned RGB-D sequence captured by the real camera.

    camera.json must define exactly one explicit depth conversion:
    - depth_meters_per_unit: multiply raw depth by this value to get meters.
    - depth_units_per_meter: divide raw depth by this value to get meters.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        camera_path = Path(self.root_dir) / "camera.json"
        if not camera_path.exists():
            raise FileNotFoundError(f"Missing camera config: {camera_path}")

        with camera_path.open("r", encoding="utf-8") as file:
            camera = json.load(file)

        self.width = int(camera["width"])
        self.height = int(camera["height"])
        if "depth_scale" in camera:
            raise ValueError(
                "camera.json field 'depth_scale' is ambiguous. Replace it with "
                "exactly one of 'depth_meters_per_unit' or "
                f"'depth_units_per_meter': {camera_path}"
            )

        depth_conversion_keys = [
            key
            for key in ("depth_meters_per_unit", "depth_units_per_meter")
            if key in camera
        ]
        if len(depth_conversion_keys) != 1:
            raise ValueError(
                "camera.json must define exactly one of "
                "'depth_meters_per_unit' or 'depth_units_per_meter': "
                f"{camera_path}"
            )

        conversion_key = depth_conversion_keys[0]
        conversion_value = float(camera[conversion_key])
        if not np.isfinite(conversion_value) or conversion_value <= 0:
            raise ValueError(f"{conversion_key} must be a finite positive number")

        if conversion_key == "depth_meters_per_unit":
            self.depth_meters_per_unit = conversion_value
        else:
            self.depth_meters_per_unit = 1.0 / conversion_value

        self.depth_units_per_meter = 1.0 / self.depth_meters_per_unit

        # RGBDDataset uses scale when projecting 2D masks. Keep the alias for
        # compatibility; real-robot conversion uses meters_per_unit explicitly.
        self.scale = self.depth_units_per_meter
        configured_min_depth = cfg.get("min_depth_m")
        configured_max_depth = cfg.get("max_depth_m")
        self.min_depth_m = float(
            camera.get("min_depth_m", 0.2)
            if configured_min_depth is None
            else configured_min_depth
        )
        self.max_depth_m = float(
            camera.get("max_depth_m", 5.0)
            if configured_max_depth is None
            else configured_max_depth
        )
        if (not np.isfinite([self.min_depth_m, self.max_depth_m]).all()
                or self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m):
            raise ValueError(
                "Invalid depth range: "
                f"min_depth_m={self.min_depth_m}, max_depth_m={self.max_depth_m}"
            )
        self.world_frame = camera.get("world_frame", "hov_y_up")
        self.pose_convention = camera.get(
            "pose_convention", "T_world_camera_optical"
        )
        if self.pose_convention not in {
            "T_world_camera_optical",
            "T_global_camera_optical",
        }:
            raise ValueError(
                f"Unsupported pose_convention: {self.pose_convention!r}"
            )
        self.pose_mode = str(cfg.get("pose_mode", "raw")).strip().lower()
        if self.pose_mode not in {"raw", "planar"}:
            raise ValueError("pose_mode must be either 'raw' or 'planar'")

        first_pose = self._convert_pose_to_hov(
            self._read_pose_matrix(self.data_list[0][2])
        )
        configured_height = cfg.get("planar_pose_height_m")
        self.planar_pose_height_m = (
            float(configured_height)
            if configured_height is not None
            else float(first_pose[1, 3])
        )
        self.pose_warn_tilt_deg = float(cfg.get("pose_warn_tilt_deg", 5.0))
        self.pose_warn_height_drift_m = float(
            cfg.get("pose_warn_height_drift_m", 0.15)
        )
        self._pose_diagnostics = {
            "poses_loaded": 0,
            "max_raw_tilt_deg": 0.0,
            "max_raw_height_drift_m": 0.0,
            "tilt_warning_frames": 0,
            "height_warning_frames": 0,
        }

        self.distortion_model = str(camera.get("distortion_model", ""))
        distortion = camera.get("distortion_coefficients", [])
        if distortion is None:
            distortion = []
        self.distortion_coefficients = np.asarray(distortion, dtype=np.float64)
        if self.distortion_coefficients.size not in (0, 4, 5, 8, 12, 14):
            raise ValueError(
                "distortion_coefficients must contain a supported CameraInfo D vector"
            )
        self.image_rectified = bool(camera.get("image_rectified", False))
        self.rectification_rotation = np.asarray(
            camera.get("rectification_rotation", np.eye(3)), dtype=np.float64
        )
        rotation_pose = np.eye(4)
        if self.rectification_rotation.shape != (3, 3):
            raise ValueError("rectification_rotation must be a 3x3 matrix")
        rotation_pose[:3, :3] = self.rectification_rotation
        as_pose_matrix(rotation_pose, name="rectification_rotation")
        if (
            not self.image_rectified
            and self.distortion_coefficients.size
            and self.distortion_model not in ("plumb_bob", "rational_polynomial")
        ):
            raise ValueError(
                "Unsupported distortion_model for unrectified images: "
                f"{self.distortion_model!r}"
            )

        if not camera.get("rgb_depth_aligned", False):
            raise ValueError(
                "RGB and depth must be registered before loading into HOV-SG"
            )

        self.depth_intrinsics = np.array(
            [
                [camera["fx"], 0.0, camera["cx"]],
                [0.0, camera["fy"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.rgb_intrinsics = self.depth_intrinsics.copy()
        if (self.width <= 0 or self.height <= 0
                or not np.isfinite(self.depth_intrinsics).all()
                or self.depth_intrinsics[0, 0] <= 0
                or self.depth_intrinsics[1, 1] <= 0
                or not np.isfinite(self.distortion_coefficients).all()):
            raise ValueError("Camera dimensions, focal lengths and distortion must be valid")
        print_status(
            "REAL_ROBOT_CAMERA",
            "OK",
            config=camera_path,
            resolution=f"{self.width}x{self.height}",
            fx=float(self.depth_intrinsics[0, 0]),
            fy=float(self.depth_intrinsics[1, 1]),
            cx=float(self.depth_intrinsics[0, 2]),
            cy=float(self.depth_intrinsics[1, 2]),
            depth_meters_per_unit=self.depth_meters_per_unit,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            world_frame=self.world_frame,
            pose_convention=self.pose_convention,
            pose_mode=self.pose_mode,
            planar_pose_height_m=self.planar_pose_height_m,
            distortion_model=self.distortion_model,
            image_rectified=self.image_rectified,
        )

    def _get_data_list(self):
        root = Path(self.root_dir)
        rgb_dir = root / "rgb"
        depth_dir = root / "depth"
        pose_dir = root / "pose"

        for directory in (rgb_dir, depth_dir, pose_dir):
            if not directory.exists():
                raise FileNotFoundError(f"Missing directory: {directory}")

        rgb_files = self._index_files(rgb_dir, {".png", ".jpg", ".jpeg"})
        depth_files = self._index_files(depth_dir, {".png", ".tiff", ".tif"})
        pose_files = self._index_files(pose_dir, {".txt"})

        rgb_ids = set(rgb_files)
        depth_ids = set(depth_files)
        pose_ids = set(pose_files)

        if not (rgb_ids == depth_ids == pose_ids):
            raise ValueError(
                "RGB/depth/pose frame IDs do not match. "
                f"rgb={len(rgb_ids)}, depth={len(depth_ids)}, pose={len(pose_ids)}"
            )

        valid_ids_path = root / "valid_frame_ids.txt"
        if valid_ids_path.is_file():
            selected_ids = {
                line.strip()
                for line in valid_ids_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            unknown_ids = selected_ids - rgb_ids
            if unknown_ids:
                raise ValueError(
                    f"{valid_ids_path} contains unknown frame IDs: "
                    f"{sorted(unknown_ids)[:5]}"
                )
        else:
            selected_ids = rgb_ids

        frame_ids = sorted(
            selected_ids,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )

        if not frame_ids:
            raise ValueError(f"No RGB-D frames found under {root}")

        data_list = [
            (rgb_files[frame_id], depth_files[frame_id], pose_files[frame_id])
            for frame_id in frame_ids
        ]
        print_status(
            "REAL_ROBOT_DATASET",
            "OK",
            root=root,
            frames=len(data_list),
            first_frame=frame_ids[0],
            last_frame=frame_ids[-1],
            first_rgb=data_list[0][0],
            first_depth=data_list[0][1],
            first_pose=data_list[0][2],
        )
        return data_list

    @staticmethod
    def _index_files(directory, allowed_suffixes):
        files = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                if path.stem in files:
                    raise ValueError(f"Duplicate frame ID {path.stem} in {directory}")
                files[path.stem] = path
        return files

    def __getitem__(self, idx):
        rgb_path, depth_path, pose_path = self.data_list[idx]

        with Image.open(rgb_path) as image:
            rgb_image = image.convert("RGB").copy()

        with Image.open(depth_path) as image:
            depth_image = image.copy()

        if rgb_image.size != depth_image.size:
            raise ValueError(
                f"Unaligned frame {idx}: "
                f"rgb={rgb_image.size}, depth={depth_image.size}"
            )

        if rgb_image.size != (self.width, self.height):
            raise ValueError(
                f"Unexpected image size at frame {idx}: {rgb_image.size}"
            )

        pose = self._load_pose(pose_path)

        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)

        return (
            rgb_image,
            depth_image,
            pose,
            self.rgb_intrinsics,
            self.depth_intrinsics,
        )

    def depth_to_meters(self, depth):
        """Convert the raw depth image to meters."""
        return (
            np.asarray(depth, dtype=np.float32) * self.depth_meters_per_unit
        )

    def get_valid_depth_mask(self, depth, max_distance=None):
        """Return metric depth and a per-pixel validity mask."""
        depth_m = self.depth_to_meters(depth)
        if depth_m.ndim != 2:
            raise ValueError("Depth must be a single-channel image")

        upper_bound = self.max_depth_m
        if max_distance is not None:
            if np.isnan(max_distance) or max_distance <= 0:
                raise ValueError("filter_distance must be positive")
            upper_bound = min(upper_bound, float(max_distance))

        valid = (
            np.isfinite(depth_m)
            & (depth_m > 0)
            & (depth_m >= self.min_depth_m)
            & (depth_m <= upper_bound)
        )
        return depth_m, valid

    def create_pcd(
        self,
        rgb,
        depth,
        camera_pose=None,
        mask_img=False,
        filter_distance=np.inf,
        return_pixel_indices=False,
    ):
        """Project an aligned real-robot RGB-D frame into a point cloud."""
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth)

        if (depth_array.shape != (self.height, self.width)
                or rgb_array.shape[:2] != depth_array.shape):
            raise ValueError("RGB/mask and depth must match camera.json resolution; "
                             "register streams and update calibration before loading")
        return super().create_pcd(
            rgb_array, depth_array, camera_pose, mask_img, filter_distance,
            return_pixel_indices=return_pixel_indices,
        )

    def _unproject_pixels(self, x, y, z):
        """Use the same distortion-aware rays for geometry and semantic masks."""
        if z.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        camera_matrix = self.depth_intrinsics
        if (
            not self.image_rectified
            and self.distortion_coefficients.size >= 4
            and np.any(np.abs(self.distortion_coefficients) > 1e-12)
        ):
            pixels = np.column_stack((x, y)).astype(np.float32)
            normalized = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2),
                camera_matrix,
                self.distortion_coefficients,
            ).reshape(-1, 2)
            point_x = normalized[:, 0] * z
            point_y = normalized[:, 1] * z
        else:
            point_x = (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
            point_y = (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
        points = np.column_stack((point_x, point_y, z))
        if self.image_rectified:
            # R maps raw optical rays into rectified rays. Undo it before TF.
            points = points @ self.rectification_rotation
        return points

    def _load_pose(self, path):
        pose = self._convert_pose_to_hov(self._read_pose_matrix(path))
        raw_tilt_deg = camera_tilt_deg_y_up(pose)
        raw_height_drift_m = abs(
            float(pose[1, 3]) - self.planar_pose_height_m
        )
        diagnostics = self._pose_diagnostics
        diagnostics["poses_loaded"] += 1
        diagnostics["max_raw_tilt_deg"] = max(
            diagnostics["max_raw_tilt_deg"], raw_tilt_deg
        )
        diagnostics["max_raw_height_drift_m"] = max(
            diagnostics["max_raw_height_drift_m"], raw_height_drift_m
        )
        if raw_tilt_deg > self.pose_warn_tilt_deg:
            diagnostics["tilt_warning_frames"] += 1
        if raw_height_drift_m > self.pose_warn_height_drift_m:
            diagnostics["height_warning_frames"] += 1

        if self.pose_mode == "planar":
            pose = project_camera_pose_to_planar_y_up(
                pose,
                height_m=self.planar_pose_height_m,
            )
        return pose

    def load_pose(self, path):
        """Load one source pose using this dataset's frame conversion policy."""
        return self._load_pose(path)

    @staticmethod
    def _read_pose_matrix(path):
        return as_pose_matrix(np.loadtxt(path, dtype=np.float64), name=str(path))

    def _convert_pose_to_hov(self, pose):
        pose = pose.copy()
        if self.world_frame == "ros_z_up":
            pose = ROS_Z_UP_TO_HOV_Y_UP @ pose
        elif self.world_frame == "camera_optical_initial":
            conversion = np.eye(4, dtype=np.float64)
            conversion[:3, :3] = CAMERA_OPTICAL_INITIAL_TO_HOV_Y_UP
            pose = conversion @ pose
        elif self.world_frame != "hov_y_up":
            raise ValueError(f"Unknown world frame: {self.world_frame}")
        return pose

    def get_pose_diagnostics(self):
        """Return accumulated raw-pose drift statistics."""
        return dict(self._pose_diagnostics)
