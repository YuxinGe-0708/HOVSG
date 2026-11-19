"""Fingerprint RGB-D inputs without hashing every image in a large sequence."""

import hashlib
import json
from pathlib import Path

import numpy as np


def dataset_geometry_signature(dataset):
    """Track ordered frames, image stats, pose/calibration contents and policy.

    Images use size/mtime for bounded startup cost. Pose and calibration files
    are small and are hashed by content, including in-place SLAM corrections.
    """
    root = Path(dataset.root_dir)
    digest = hashlib.sha256()

    def record(value):
        digest.update(json.dumps(value, sort_keys=True).encode("utf-8") + b"\n")

    def add_file(path, content=False):
        path = Path(path)
        record(str(path.resolve()))
        if not path.is_file():
            record(None)
            return
        if content:
            digest.update(path.read_bytes())
            digest.update(b"\n")
        else:
            stat = path.stat()
            record([stat.st_size, stat.st_mtime_ns])

    for frame in dataset.data_list:
        for path in frame:
            add_file(path, content=Path(path).suffix.lower() == ".txt")
    for name in ("camera.json", "valid_frame_ids.txt", "traj.txt", "frames.csv",
                 "intrinsic/intrinsic_color.txt", "intrinsic/intrinsic_depth.txt"):
        add_file(root / name, content=True)
    add_file(root.parent / "cam_params.json", content=True)

    policy = {}
    for name in ("scale", "depth_intrinsics", "rgb_intrinsics", "world_frame",
                 "pose_convention", "pose_mode", "planar_pose_height_m",
                 "min_depth_m", "max_depth_m", "image_rectified",
                 "distortion_coefficients", "rectification_rotation"):
        value = getattr(dataset, name, None)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        policy[name] = value
    record(policy)
    return digest.hexdigest()
