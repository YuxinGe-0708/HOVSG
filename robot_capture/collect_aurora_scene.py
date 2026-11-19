#!/usr/bin/env python3
"""Capture synchronized Aurora 930 RGB/depth frames and camera poses."""

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
from itertools import islice
import json
import os
from pathlib import Path
import time
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from camera_geometry import projection_metadata


FRAME_FIELDS = [
    "frame_id",
    "rgb_timestamp_ns",
    "depth_timestamp_ns",
    "pose_timestamp_ns",
    "tracking_quality",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save synchronized RGB/depth PNGs and T_world_camera matrices."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--part-name", required=True)
    parser.add_argument("--previous-part", default="")
    parser.add_argument("--overlap-description", default="")
    parser.add_argument("--rgb-topic", default="/rgb/image_raw")
    parser.add_argument("--depth-topic", default="/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/rgb/camera_info")
    parser.add_argument("--tf-world-frame", default="map")
    parser.add_argument(
        "--world-frame",
        choices=("ros_z_up", "hov_y_up", "camera_optical_initial"),
        default="ros_z_up",
        help="Coordinate convention of the TF world frame recorded in each pose.",
    )
    parser.add_argument(
        "--camera-frame",
        default="",
        help="TF camera frame. Empty uses the RGB message frame_id.",
    )
    parser.add_argument(
        "--camera-frame-is-optical",
        action="store_true",
        help=(
            "Assert that --camera-frame uses x-right/y-down/z-forward axes, "
            "even when its ROS frame name ends in _link."
        ),
    )
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--rgb-depth-aligned",
        action="store_true",
        help="Assert that the selected depth topic is registered to RGB.",
    )
    parser.add_argument("--sync-slop", type=float, default=0.02)
    parser.add_argument("--max-rgb-depth-skew-ms", type=float, default=20.0)
    parser.add_argument("--max-pose-skew-ms", type=float, default=50.0)
    parser.add_argument("--tf-timeout", type=float, default=0.02)
    parser.add_argument(
        "--pose-queue-timeout",
        type=float,
        default=1.0,
        help="Seconds to retain an RGB-D pair while waiting for its exact-time TF.",
    )
    parser.add_argument("--max-pending-frames", type=int, default=60)
    parser.add_argument(
        "--image-rectified",
        action="store_true",
        help="Set only when the RGB/depth streams are already undistorted.",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="0 saves every pair.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 is unlimited.")
    parser.add_argument("--resume", action="store_true")
    args, ros_args = parser.parse_known_args()
    if args.depth_scale <= 0:
        parser.error("--depth-scale must be positive")
    if args.fps < 0 or args.max_frames < 0:
        parser.error("--fps and --max-frames cannot be negative")
    if args.pose_queue_timeout <= 0 or args.max_pending_frames <= 0:
        parser.error("--pose-queue-timeout and --max-pending-frames must be positive")
    if (
        args.sync_slop <= 0
        or args.max_rgb_depth_skew_ms <= 0
        or args.max_pose_skew_ms <= 0
        or args.tf_timeout < 0
    ):
        parser.error("RGB-D synchronization limits must be positive")
    if not args.camera_frame_is_optical:
        parser.error(
            "--camera-frame-is-optical is required after verifying the camera axes"
        )
    if not args.part_name or "/" in args.part_name or "\\" in args.part_name:
        parser.error("--part-name must be one directory name such as part_01")
    if args.output.name != args.part_name:
        parser.error("--output must end with the same name as --part-name")
    return args, ros_args


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-16:
        raise ValueError("TF quaternion has zero norm")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def transform_matrix(transform) -> np.ndarray:
    q = transform.rotation
    t = transform.translation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class SceneCollector(Node):
    def __init__(self, args):
        super().__init__("aurora_scene_collector")
        self.args = args
        self.camera_info: Optional[CameraInfo] = None
        self.last_saved_stamp_ns: Optional[int] = None
        self.last_selected_stamp_ns: Optional[int] = None
        self.warned_no_info = False
        self.last_tf_warning_time = 0.0
        self.saved_this_run = 0
        self.dropped_pose_frames = 0
        self.dropped_sync_frames = 0
        self.done = False
        self.accepting_frames = True
        self.pending_frames = deque()
        self.capture_started_at_utc = utc_now()
        self.capture_finalized = False

        self.rgb_dir = args.output / "rgb"
        self.depth_dir = args.output / "depth"
        self.pose_dir = args.output / "pose"
        self.frames_path = args.output / "frames.csv"
        self.part_metadata_path = args.output / "part.json"
        self.valid_ids_path = args.output / "valid_frame_ids.txt"
        self.next_index = self._prepare_output()
        self._write_part_metadata("recording")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.rgb_sub = message_filters.Subscriber(
            self, Image, args.rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=args.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self._pair_callback)
        self.pending_timer = self.create_timer(0.02, self._process_pending_frames)
        self.get_logger().info(
            f"Waiting for {args.rgb_topic}, {args.depth_topic}, and "
            f"{args.camera_info_topic}"
        )

    def _prepare_output(self) -> int:
        self.args.output.mkdir(parents=True, exist_ok=True)
        for directory in (self.rgb_dir, self.depth_dir, self.pose_dir):
            directory.mkdir(parents=True, exist_ok=True)

        rgb_ids = {p.stem for p in self.rgb_dir.glob("*.png")}
        depth_ids = {p.stem for p in self.depth_dir.glob("*.png")}
        pose_ids = {p.stem for p in self.pose_dir.glob("*.txt")}
        metadata_ids = set()
        if self.frames_path.is_file() and self.frames_path.stat().st_size:
            with self.frames_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FRAME_FIELDS:
                    raise RuntimeError(
                        f"Unexpected frames.csv columns: {reader.fieldnames}"
                    )
                for row in reader:
                    frame_id = str(row.get("frame_id", "")).strip()
                    if not frame_id or frame_id in metadata_ids:
                        raise RuntimeError("frames.csv has an empty or duplicate frame_id")
                    metadata_ids.add(frame_id)

        if not self.args.resume and (
            rgb_ids or depth_ids or pose_ids or metadata_ids
        ):
            raise RuntimeError(
                f"{self.args.output} already contains frames; choose a new scene or use --resume"
            )
        if self.args.resume and not (rgb_ids == depth_ids == pose_ids):
            raise RuntimeError("Cannot resume: rgb/depth/pose frame sets do not match")
        if self.args.resume and metadata_ids != rgb_ids:
            raise RuntimeError("Cannot resume: frames.csv IDs do not match frame files")
        if self.args.resume and self.valid_ids_path.exists():
            raise RuntimeError("Cannot resume a finalized part with valid_frame_ids.txt")
        if not self.frames_path.exists() or self.frames_path.stat().st_size == 0:
            with self.frames_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                csv.DictWriter(handle, fieldnames=FRAME_FIELDS).writeheader()
                handle.flush()
                os.fsync(handle.fileno())
        if any(not value.isdigit() for value in rgb_ids):
            raise RuntimeError("Existing frame filenames must use numeric IDs")
        return max((int(value) for value in rgb_ids), default=0) + 1

    def _write_part_metadata(self, status: str):
        metadata = {
            "schema_version": 1,
            "part_name": self.args.part_name,
            "status": status,
            "started_at_utc": self.capture_started_at_utc,
            "previous_part": self.args.previous_part or None,
            "overlap_description": self.args.overlap_description,
            "pose_convention": "T_world_camera_optical",
            "recorder": "collect_aurora_scene.py",
            "saved_frames": self.next_index - 1 if hasattr(self, "next_index") else 0,
            "dropped_pose_frames": self.dropped_pose_frames,
            "dropped_sync_frames": self.dropped_sync_frames,
        }
        if status != "recording":
            metadata["finalized_at_utc"] = utc_now()
        atomic_json(self.part_metadata_path, metadata)

    def _validated_frame_ids(self):
        rgb_ids = {path.stem for path in self.rgb_dir.glob("*.png")}
        depth_ids = {path.stem for path in self.depth_dir.glob("*.png")}
        pose_ids = {path.stem for path in self.pose_dir.glob("*.txt")}
        if not (rgb_ids == depth_ids == pose_ids):
            raise RuntimeError(
                "Final RGB/depth/pose frame IDs do not match: "
                f"rgb={len(rgb_ids)}, depth={len(depth_ids)}, pose={len(pose_ids)}"
            )
        with self.frames_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            metadata_id_list = [
                str(row.get("frame_id", "")).strip()
                for row in csv.DictReader(handle)
                if str(row.get("frame_id", "")).strip()
            ]
        metadata_ids = set(metadata_id_list)
        if len(metadata_ids) != len(metadata_id_list):
            raise RuntimeError("Final frames.csv contains duplicate frame IDs")
        if metadata_ids != rgb_ids:
            raise RuntimeError(
                "Final frames.csv IDs do not match frame files: "
                f"metadata={len(metadata_ids)}, files={len(rgb_ids)}"
            )
        if not rgb_ids:
            raise RuntimeError("No complete RGB-D-pose frames were captured")
        return sorted(rgb_ids, key=int)

    def finalize_capture(self):
        if self.capture_finalized:
            return
        frame_ids = self._validated_frame_ids()
        temporary = self.valid_ids_path.with_name(
            self.valid_ids_path.name + ".tmp"
        )
        temporary.write_text(
            "".join(f"{frame_id}\n" for frame_id in frame_ids),
            encoding="utf-8",
        )
        os.replace(temporary, self.valid_ids_path)
        status = "failed" if getattr(self, "failed", False) else "finalized"
        self._write_part_metadata(status)
        self.capture_finalized = True
        self.get_logger().info(
            f"Capture status={status} for {self.args.part_name}: "
            f"{len(frame_ids)} synchronized frames"
        )

    def _camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def _lookup_pose(self, rgb_msg: Image, camera_frame: str):
        rgb_stamp_ns = Time.from_msg(rgb_msg.header.stamp).nanoseconds
        if self.args.tf_world_frame == camera_frame:
            return np.eye(4, dtype=np.float64), rgb_stamp_ns
        try:
            stamped = self.tf_buffer.lookup_transform(
                self.args.tf_world_frame,
                camera_frame,
                Time.from_msg(rgb_msg.header.stamp),
                timeout=Duration(seconds=self.args.tf_timeout),
            )
            pose_stamp_ns = Time.from_msg(stamped.header.stamp).nanoseconds
            if pose_stamp_ns == 0:
                pose_stamp_ns = rgb_stamp_ns
            return transform_matrix(stamped.transform), pose_stamp_ns
        except TransformException:
            return None

    @staticmethod
    def _rows_from_image(
        msg: Image, dtype: np.dtype, elements_per_pixel: int
    ) -> np.ndarray:
        dtype = np.dtype(dtype)
        if msg.step % dtype.itemsize:
            raise ValueError(
                f"Image step {msg.step} is not divisible by item size {dtype.itemsize}"
            )
        row_elements = msg.step // dtype.itemsize
        required_per_row = int(msg.width) * elements_per_pixel
        if row_elements < required_per_row:
            raise ValueError(
                f"Image step is too small: {row_elements} < {required_per_row} elements"
            )
        flat = np.frombuffer(msg.data, dtype=dtype)
        required_total = int(msg.height) * row_elements
        if flat.size < required_total:
            raise ValueError(
                f"Image data is truncated: {flat.size} < {required_total} elements"
            )
        return flat[:required_total].reshape(int(msg.height), row_elements)[
            :, :required_per_row
        ]

    def _convert_rgb(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        if encoding in ("bgr8", "rgb8"):
            image = self._rows_from_image(msg, np.uint8, 3).reshape(
                int(msg.height), int(msg.width), 3
            )
            if encoding == "rgb8":
                image = image[:, :, ::-1]
            return np.ascontiguousarray(image)
        if encoding in ("bgra8", "rgba8"):
            image = self._rows_from_image(msg, np.uint8, 4).reshape(
                int(msg.height), int(msg.width), 4
            )
            code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
            return cv2.cvtColor(image, code)
        if encoding == "mono8":
            image = self._rows_from_image(msg, np.uint8, 1).reshape(
                int(msg.height), int(msg.width)
            )
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        raise ValueError(
            f"Unsupported RGB encoding {msg.encoding}; expected bgr8, rgb8, bgra8, rgba8, or mono8"
        )

    def _convert_depth(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        byte_order = ">" if msg.is_bigendian else "<"
        if encoding in ("mono16", "16uc1"):
            depth = self._rows_from_image(msg, np.dtype(byte_order + "u2"), 1)
            return np.ascontiguousarray(depth.astype(np.uint16, copy=False))
        if encoding == "32fc1":
            depth = self._rows_from_image(msg, np.dtype(byte_order + "f4"), 1)
            finite = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            units = np.rint(finite / self.args.depth_scale)
            return np.clip(units, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        raise ValueError(
            f"Unsupported depth encoding {msg.encoding}; expected mono16, 16UC1, or 32FC1"
        )

    def _write_camera_json(
        self, info: CameraInfo, rgb_msg: Image, depth_msg: Image, camera_frame: str
    ):
        calibration = projection_metadata(info, image_rectified=self.args.image_rectified)
        data = {
            "schema_version": 1,
            "width": int(rgb_msg.width),
            "height": int(rgb_msg.height),
            **calibration,
            "depth_meters_per_unit": float(self.args.depth_scale),
            "min_depth_m": float(self.args.min_depth),
            "max_depth_m": float(self.args.max_depth),
            "rgb_depth_aligned": bool(self.args.rgb_depth_aligned),
            "world_frame": self.args.world_frame,
            "pose_convention": "T_world_camera_optical",
            "tf_world_frame": self.args.tf_world_frame,
            "camera_frame": camera_frame,
            "camera_frame_axes": "optical_x_right_y_down_z_forward",
            "rgb_topic": self.args.rgb_topic,
            "depth_topic": self.args.depth_topic,
            "rgb_encoding": rgb_msg.encoding,
            "depth_encoding": depth_msg.encoding,
            "distortion_model": getattr(info, "distortion_model", ""),
            "distortion_coefficients": [
                float(value) for value in getattr(info, "d", [])
            ],
            "image_rectified": bool(self.args.image_rectified),
            "pose_timestamp_policy": "exact_image_timestamp_with_tf_wait",
            "pose_queue_timeout_s": float(self.args.pose_queue_timeout),
            "max_rgb_depth_skew_ms": float(
                self.args.max_rgb_depth_skew_ms
            ),
            "max_pose_skew_ms": float(self.args.max_pose_skew_ms),
        }
        path = self.args.output / "camera.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            keys = (
                "width",
                "height",
                "fx",
                "fy",
                "cx",
                "cy",
                "depth_meters_per_unit",
                "min_depth_m",
                "max_depth_m",
                "rgb_depth_aligned",
                "tf_world_frame",
                "camera_frame",
                "rgb_topic",
                "depth_topic",
                "rgb_encoding",
                "depth_encoding",
                "world_frame",
                "pose_convention",
                "camera_frame_axes",
                "distortion_model",
                "distortion_coefficients",
                "image_rectified",
                "rectification_rotation",
                "pose_timestamp_policy",
                "max_rgb_depth_skew_ms",
                "max_pose_skew_ms",
            )
            mismatches = [key for key in keys if existing.get(key) != data.get(key)]
            if mismatches:
                raise RuntimeError(
                    "Existing camera.json does not match current stream: "
                    + ", ".join(mismatches)
                )
            self._ensure_room_metadata(data)
            return
        atomic_json(path, data)
        self._ensure_room_metadata(data)

    def _ensure_room_metadata(self, camera_data):
        room_dir = self.args.output.parent
        capture_path = room_dir / "capture.json"
        if capture_path.exists():
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            if capture.get("dataset_type") != "hovsg_multipart_capture":
                raise RuntimeError(f"Unsupported capture.json: {capture_path}")
        else:
            atomic_json(
                capture_path,
                {
                    "schema_version": 1,
                    "dataset_type": "hovsg_multipart_capture",
                    "room_id": room_dir.name,
                    "created_at_utc": utc_now(),
                    "requirements": {
                        "pose_convention": "T_world_camera_optical",
                        "minimum_overlap_ratio": 0.30,
                        "recommended_overlap_ratio": 0.50,
                    },
                },
            )

        signature_keys = (
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "depth_meters_per_unit",
            "min_depth_m",
            "max_depth_m",
            "rgb_depth_aligned",
            "distortion_model",
            "distortion_coefficients",
            "image_rectified",
            "rectification_rotation",
            "camera_frame_axes",
        )
        template_path = room_dir / "camera_template.json"
        if template_path.exists():
            template = json.loads(template_path.read_text(encoding="utf-8"))
            mismatches = [
                key
                for key in signature_keys
                if template.get(key) != camera_data.get(key)
            ]
            if mismatches:
                raise RuntimeError(
                    "Camera parameters differ from the first capture part: "
                    + ", ".join(mismatches)
                )
        else:
            atomic_json(template_path, camera_data)

    @staticmethod
    def _atomic_png(path: Path, image: np.ndarray):
        temporary = path.with_name(path.stem + ".tmp.png")
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"Failed to write {temporary}")
        os.replace(temporary, path)

    @staticmethod
    def _atomic_pose(path: Path, matrix: np.ndarray):
        temporary = path.with_name(path.name + ".tmp")
        np.savetxt(temporary, matrix, fmt="%.9f")
        os.replace(temporary, path)

    def stop_accepting_frames(self):
        self.accepting_frames = False

    @property
    def pending_frame_count(self) -> int:
        return len(self.pending_frames)

    def _save_frame(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        info: CameraInfo,
        camera_frame: str,
        pose: np.ndarray,
        rgb_stamp_ns: int,
        depth_stamp_ns: int,
        pose_stamp_ns: int,
    ):
        frame_paths = []
        metadata_size = self.frames_path.stat().st_size
        try:
            rgb = self._convert_rgb(rgb_msg)
            depth = self._convert_depth(depth_msg)
            if info.width != rgb_msg.width or info.height != rgb_msg.height:
                raise ValueError(
                    "RGB CameraInfo dimensions do not match the RGB image dimensions"
                )
            if self.args.rgb_depth_aligned and depth.shape != rgb.shape[:2]:
                raise ValueError(
                    "Aligned depth assertion failed: depth and RGB dimensions differ"
                )
            self._write_camera_json(info, rgb_msg, depth_msg, camera_frame)
            name = f"{self.next_index:06d}"
            frame_paths = [
                self.rgb_dir / f"{name}.png",
                self.depth_dir / f"{name}.png",
                self.pose_dir / f"{name}.txt",
            ]
            self._atomic_png(frame_paths[0], rgb)
            self._atomic_png(frame_paths[1], depth)
            self._atomic_pose(frame_paths[2], pose)
            row = {
                "frame_id": name,
                "rgb_timestamp_ns": rgb_stamp_ns,
                "depth_timestamp_ns": depth_stamp_ns,
                "pose_timestamp_ns": pose_stamp_ns,
                "tracking_quality": "",
            }
            with self.frames_path.open(
                "a", encoding="utf-8", newline=""
            ) as handle:
                csv.DictWriter(handle, fieldnames=FRAME_FIELDS).writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as error:
            for path in frame_paths:
                path.unlink(missing_ok=True)
                path.with_name(path.stem + ".tmp.png").unlink(missing_ok=True)
                path.with_name(path.name + ".tmp").unlink(missing_ok=True)
            with self.frames_path.open("r+b") as handle:
                handle.truncate(metadata_size)
            self.get_logger().error(f"Frame was not saved: {error}")
            return

        self.last_saved_stamp_ns = rgb_stamp_ns
        self.next_index += 1
        self.saved_this_run += 1
        if self.saved_this_run == 1 or self.saved_this_run % 25 == 0:
            self.get_logger().info(
                f"Saved {self.saved_this_run} frame(s) to {self.args.output}"
            )
        if self.args.max_frames and self.saved_this_run >= self.args.max_frames:
            self.get_logger().info("Reached --max-frames; stopping")
            self.accepting_frames = False
            self.pending_frames.clear()
            self.done = True

    def _process_pending_frames(self):
        processed = 0
        while self.pending_frames and not self.done and processed < 4:
            (
                queued_at,
                rgb_stamp_ns,
                depth_stamp_ns,
                rgb_msg,
                depth_msg,
                info,
                camera_frame,
            ) = (
                self.pending_frames[0]
            )
            pose_result = self._lookup_pose(rgb_msg, camera_frame)
            if pose_result is None:
                age = time.monotonic() - queued_at
                newer_index = None
                newer_pose = None

                # Probe the newest frame first. During a prolonged TF outage
                # this keeps lookup cost bounded; after TF recovers, find the
                # earliest newer frame that can be saved in timestamp order.
                newest = self.pending_frames[-1]
                newest_pose_result = None
                if len(self.pending_frames) > 1:
                    newest_pose_result = self._lookup_pose(newest[3], newest[6])
                if newest_pose_result is not None:
                    for index, pending in enumerate(
                        islice(self.pending_frames, 1, None), start=1
                    ):
                        if pending is newest:
                            candidate_pose_result = newest_pose_result
                        else:
                            candidate_pose_result = self._lookup_pose(
                                pending[3], pending[6]
                            )
                        if candidate_pose_result is not None:
                            newer_index = index
                            newer_pose = candidate_pose_result
                            break

                if newer_index is not None:
                    for _ in range(newer_index):
                        self.pending_frames.popleft()
                    self.dropped_pose_frames += newer_index
                    now = time.monotonic()
                    if now - self.last_tf_warning_time >= 5.0:
                        self.get_logger().warning(
                            f"Dropped {newer_index} earlier RGB-D pair(s) without exact-time "
                            f"TF; continuing from a newer valid frame "
                            f"(total pose drops={self.dropped_pose_frames})"
                        )
                        self.last_tf_warning_time = now

                    (
                        queued_at,
                        rgb_stamp_ns,
                        depth_stamp_ns,
                        rgb_msg,
                        depth_msg,
                        info,
                        camera_frame,
                    ) = self.pending_frames[0]
                    pose_result = newer_pose
                    processed += newer_index
                else:
                    if age < self.args.pose_queue_timeout:
                        break
                    self.pending_frames.popleft()
                    self.dropped_pose_frames += 1
                    now = time.monotonic()
                    if now - self.last_tf_warning_time >= 5.0:
                        self.get_logger().warning(
                            f"Dropped an RGB-D pair after waiting {age:.2f}s for exact-time TF "
                            f"{self.args.tf_world_frame} <- {camera_frame} "
                            f"(total pose drops={self.dropped_pose_frames})"
                        )
                        self.last_tf_warning_time = now
                    processed += 1
                    continue

            self.pending_frames.popleft()
            pose, pose_stamp_ns = pose_result
            pose_skew_ns = abs(rgb_stamp_ns - pose_stamp_ns)
            if pose_skew_ns > int(self.args.max_pose_skew_ms * 1_000_000):
                self.dropped_pose_frames += 1
                self.get_logger().warning(
                    f"Dropped frame with {pose_skew_ns / 1e6:.2f} ms pose skew"
                )
                processed += 1
                continue
            self._save_frame(
                rgb_msg,
                depth_msg,
                info,
                camera_frame,
                pose,
                rgb_stamp_ns,
                depth_stamp_ns,
                pose_stamp_ns,
            )
            processed += 1

    def _pair_callback(self, rgb_msg: Image, depth_msg: Image):
        if not self.accepting_frames or self.done:
            return
        if self.camera_info is None:
            if not self.warned_no_info:
                self.get_logger().warning("Waiting for RGB CameraInfo")
                self.warned_no_info = True
            return

        stamp_ns = Time.from_msg(rgb_msg.header.stamp).nanoseconds
        depth_stamp_ns = Time.from_msg(depth_msg.header.stamp).nanoseconds
        skew_ns = abs(stamp_ns - depth_stamp_ns)
        max_skew_ns = int(self.args.max_rgb_depth_skew_ms * 1_000_000)
        if skew_ns > max_skew_ns:
            self.dropped_sync_frames += 1
            self.get_logger().warning(
                f"Dropped RGB-D pair with {skew_ns / 1e6:.2f} ms timestamp skew"
            )
            return
        if self.args.fps > 0 and self.last_selected_stamp_ns is not None:
            min_interval_ns = int(1e9 / self.args.fps)
            if stamp_ns - self.last_selected_stamp_ns < min_interval_ns:
                return

        camera_frame = self.args.camera_frame or rgb_msg.header.frame_id
        if not camera_frame:
            self.get_logger().error("RGB message has an empty frame_id; pass --camera-frame")
            return
        self.last_selected_stamp_ns = stamp_ns
        self.pending_frames.append(
            (
                time.monotonic(),
                stamp_ns,
                depth_stamp_ns,
                rgb_msg,
                depth_msg,
                self.camera_info,
                camera_frame,
            )
        )
        if len(self.pending_frames) > self.args.max_pending_frames:
            self.pending_frames.popleft()
            self.dropped_pose_frames += 1
            self.get_logger().warning(
                "Pending TF queue reached --max-pending-frames; dropped its oldest pair "
                f"(total pose drops={self.dropped_pose_frames})"
            )

    def destroy_node(self):
        self.stop_accepting_frames()
        self.finalize_capture()
        return super().destroy_node()


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = None
    executor = None
    try:
        node = SceneCollector(args)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None and node is not None:
            executor.remove_node(node)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
