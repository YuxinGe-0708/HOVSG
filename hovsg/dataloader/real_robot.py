import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from hovsg.dataloader.generic import RGBDDataset
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
        self.min_depth_m = float(
            cfg.get("min_depth_m", camera.get("min_depth_m", 0.2))
        )
        self.max_depth_m = float(
            cfg.get("max_depth_m", camera.get("max_depth_m", 5.0))
        )
        if self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError(
                "Invalid depth range: "
                f"min_depth_m={self.min_depth_m}, max_depth_m={self.max_depth_m}"
            )
        self.world_frame = camera.get("world_frame", "hov_y_up")

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

        frame_ids = sorted(
            rgb_ids,
            key=lambda value: int(value) if value.isdigit() else value,
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
        return {
            path.stem: path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        }

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

        upper_bound = self.max_depth_m
        if max_distance is not None and np.isfinite(max_distance):
            upper_bound = min(upper_bound, float(max_distance))

        valid = (
            np.isfinite(depth_m)
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
    ):
        """Project an aligned real-robot RGB-D frame into a point cloud."""
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth)

        if rgb_array.shape[:2] != depth_array.shape[:2]:
            if mask_img:
                raise ValueError("Mask and depth resolutions must match")
            rgb_array = cv2.resize(
                rgb_array,
                (depth_array.shape[1], depth_array.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        depth_m, valid = self.get_valid_depth_mask(
            depth_array,
            max_distance=filter_distance,
        )

        if mask_img:
            instance_mask = rgb_array.astype(bool)
            if instance_mask.ndim == 3:
                instance_mask = instance_mask[..., 0]
            valid &= instance_mask

        if not np.any(valid):
            return o3d.geometry.PointCloud()

        height, width = depth_m.shape
        y, x = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )

        z = depth_m[valid]
        x = x[valid]
        y = y[valid]
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

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if not mask_img:
            rgb_array = rgb_array.astype(np.uint8)
            pcd.colors = o3d.utility.Vector3dVector(rgb_array[valid] / 255.0)

        if camera_pose is not None:
            pcd.transform(camera_pose)

        return pcd

    def _load_pose(self, path):
        pose = np.loadtxt(path, dtype=np.float64)

        if pose.size != 16:
            raise ValueError(f"Pose must contain 16 numbers: {path}")

        pose = pose.reshape(4, 4)

        if not np.isfinite(pose).all():
            raise ValueError(f"Pose contains NaN or Inf: {path}")

        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"Invalid pose last row: {path}")

        rotation = pose[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-2,
        ):
            raise ValueError(f"Invalid rotation matrix: {path}")

        if self.world_frame == "ros_z_up":
            pose = ROS_Z_UP_TO_HOV_Y_UP @ pose
        elif self.world_frame == "camera_optical_initial":
            conversion = np.eye(4, dtype=np.float64)
            conversion[:3, :3] = CAMERA_OPTICAL_INITIAL_TO_HOV_Y_UP
            pose = conversion @ pose
        elif self.world_frame != "hov_y_up":
            raise ValueError(f"Unknown world frame: {self.world_frame}")

        return pose
