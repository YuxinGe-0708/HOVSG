"""Atomic writer used by a robot RGB-D recorder for one capture part."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from hovsg.utils.pose_utils import as_pose_matrix


FRAME_FIELDS = [
    "frame_id",
    "rgb_timestamp_ns",
    "depth_timestamp_ns",
    "pose_timestamp_ns",
    "tracking_quality",
]


class MultipartFrameWriter:
    """Validate and atomically save synchronized RGB-D-pose frames."""

    def __init__(
        self,
        part_dir,
        *,
        max_rgb_depth_skew_ms=20.0,
        max_pose_skew_ms=50.0,
    ):
        self.part_dir = Path(part_dir).resolve()
        with (self.part_dir / "camera.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.camera = json.load(handle)
        if self.camera.get("pose_convention") != "T_world_camera_optical":
            raise ValueError(
                "camera.json must explicitly set "
                "pose_convention='T_world_camera_optical' while recording"
            )
        self.width = int(self.camera["width"])
        self.height = int(self.camera["height"])
        self.max_rgb_depth_skew_ns = int(max_rgb_depth_skew_ms * 1_000_000)
        self.max_pose_skew_ns = int(max_pose_skew_ms * 1_000_000)
        self.rgb_dir = self.part_dir / "rgb"
        self.depth_dir = self.part_dir / "depth"
        self.pose_dir = self.part_dir / "pose"
        for directory in (self.rgb_dir, self.depth_dir, self.pose_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.part_dir / "frames.csv"
        self._known_frame_ids = self._load_known_frame_ids()

    def _load_known_frame_ids(self):
        if not self.frames_path.is_file() or self.frames_path.stat().st_size == 0:
            self._write_header()
            return set()
        with self.frames_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            return {
                str(row["frame_id"]).strip()
                for row in csv.DictReader(handle)
                if str(row.get("frame_id", "")).strip()
            }

    def _write_header(self):
        with self.frames_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FRAME_FIELDS).writeheader()

    @staticmethod
    def _atomic_write_image(path, image):
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if not cv2.imwrite(str(temporary), image):
            raise IOError(f"failed to write image: {temporary}")
        temporary.replace(path)

    @staticmethod
    def _atomic_write_pose(path, pose):
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        np.savetxt(temporary, pose, fmt="%.9f")
        temporary.replace(path)

    def write_frame(
        self,
        frame_id,
        rgb_bgr,
        depth_raw,
        pose_world_camera_optical,
        *,
        rgb_timestamp_ns,
        depth_timestamp_ns,
        pose_timestamp_ns,
        tracking_quality,
    ):
        """Write one complete frame, then commit its metadata row."""
        frame_id = str(frame_id).strip()
        if not frame_id or not frame_id.isdigit():
            raise ValueError("frame_id must contain decimal digits")
        frame_id = f"{int(frame_id):06d}"
        if frame_id in self._known_frame_ids:
            raise FileExistsError(f"frame already recorded: {frame_id}")

        rgb_bgr = np.asarray(rgb_bgr)
        depth_raw = np.asarray(depth_raw)
        expected_shape = (self.height, self.width)
        if rgb_bgr.shape != (*expected_shape, 3):
            raise ValueError(
                f"RGB shape must be {(*expected_shape, 3)}, got {rgb_bgr.shape}"
            )
        if depth_raw.shape != expected_shape:
            raise ValueError(
                f"depth shape must be {expected_shape}, got {depth_raw.shape}"
            )
        if rgb_bgr.dtype != np.uint8:
            raise ValueError(f"RGB dtype must be uint8, got {rgb_bgr.dtype}")
        if depth_raw.dtype != np.uint16:
            raise ValueError(f"depth dtype must be uint16, got {depth_raw.dtype}")

        pose = as_pose_matrix(
            pose_world_camera_optical,
            name=f"pose for frame {frame_id}",
        )
        rgb_timestamp_ns = int(rgb_timestamp_ns)
        depth_timestamp_ns = int(depth_timestamp_ns)
        pose_timestamp_ns = int(pose_timestamp_ns)
        if abs(rgb_timestamp_ns - depth_timestamp_ns) > self.max_rgb_depth_skew_ns:
            raise ValueError(f"RGB/depth timestamp skew is too large: {frame_id}")
        if abs(rgb_timestamp_ns - pose_timestamp_ns) > self.max_pose_skew_ns:
            raise ValueError(f"RGB/pose timestamp skew is too large: {frame_id}")
        if tracking_quality is not None and float(tracking_quality) <= 0.0:
            raise ValueError(f"tracking quality is not valid: {frame_id}")

        rgb_path = self.rgb_dir / f"{frame_id}.png"
        depth_path = self.depth_dir / f"{frame_id}.png"
        pose_path = self.pose_dir / f"{frame_id}.txt"
        existing = [path for path in (rgb_path, depth_path, pose_path) if path.exists()]
        if existing:
            raise FileExistsError(f"frame files already exist: {existing}")

        row = {
            "frame_id": frame_id,
            "rgb_timestamp_ns": rgb_timestamp_ns,
            "depth_timestamp_ns": depth_timestamp_ns,
            "pose_timestamp_ns": pose_timestamp_ns,
            "tracking_quality": "" if tracking_quality is None else tracking_quality,
        }
        metadata_size = self.frames_path.stat().st_size
        try:
            self._atomic_write_image(rgb_path, rgb_bgr)
            self._atomic_write_image(depth_path, depth_raw)
            self._atomic_write_pose(pose_path, pose)
            with self.frames_path.open(
                "a", encoding="utf-8", newline=""
            ) as handle:
                csv.DictWriter(handle, fieldnames=FRAME_FIELDS).writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            for path in (rgb_path, depth_path, pose_path):
                if path.exists():
                    path.unlink()
            with self.frames_path.open("r+b") as handle:
                handle.truncate(metadata_size)
            raise
        self._known_frame_ids.add(frame_id)
        return rgb_path, depth_path, pose_path
