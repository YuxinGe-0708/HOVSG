"""Validation and metadata helpers for multipart RGB-D datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hovsg.utils.pose_utils import as_pose_matrix


RGB_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEPTH_SUFFIXES = {".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class PartData:
    name: str
    root: Path
    camera: dict
    frame_ids: list[str]
    rgb_files: dict[str, Path]
    depth_files: dict[str, Path]
    pose_files: dict[str, Path]


def _index_files(directory, suffixes):
    return {
        path.stem: path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _frame_sort_key(frame_id):
    return (0, int(frame_id)) if frame_id.isdigit() else (1, frame_id)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_camera_metadata(camera, *, source="camera.json"):
    required_camera_keys = {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "rgb_depth_aligned",
        "world_frame",
        "pose_convention",
    }
    missing_camera = sorted(required_camera_keys - set(camera))
    if missing_camera:
        raise ValueError(f"{source} is missing keys: {missing_camera}")
    if camera["pose_convention"] not in {
        "T_world_camera_optical",
        "T_global_camera_optical",
    }:
        raise ValueError(
            f"unsupported pose_convention in {source}: "
            f"{camera['pose_convention']!r}"
        )
    if camera["world_frame"] not in {
        "ros_z_up",
        "camera_optical_initial",
        "hov_y_up",
    }:
        raise ValueError(
            f"unsupported world_frame in {source}: {camera['world_frame']!r}"
        )
    if (
        camera["pose_convention"] == "T_global_camera_optical"
        and camera["world_frame"] != "hov_y_up"
    ):
        raise ValueError(
            f"global multipart poses must use world_frame='hov_y_up': {source}"
        )
    for key in ("width", "height"):
        if int(camera[key]) <= 0:
            raise ValueError(f"{key} must be positive in {source}")
    for key in ("fx", "fy", "cx", "cy"):
        if not np.isfinite(float(camera[key])):
            raise ValueError(f"{key} must be finite in {source}")
    if float(camera["fx"]) <= 0 or float(camera["fy"]) <= 0:
        raise ValueError(f"fx and fy must be positive in {source}")
    depth_keys = {
        key
        for key in ("depth_meters_per_unit", "depth_units_per_meter")
        if key in camera
    }
    if len(depth_keys) != 1:
        raise ValueError(
            f"{source} must define exactly one explicit depth conversion"
        )
    depth_value = float(camera[next(iter(depth_keys))])
    if not np.isfinite(depth_value) or depth_value <= 0:
        raise ValueError(f"depth conversion must be positive in {source}")
    if not camera["rgb_depth_aligned"]:
        raise ValueError(f"RGB and depth are not aligned in {source}")


def validate_part(part_dir, *, require_frame_metadata=False):
    """Validate one part without silently dropping or re-pairing frames."""
    root = Path(part_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"part directory does not exist: {root}")

    camera_path = root / "camera.json"
    if not camera_path.is_file():
        raise FileNotFoundError(f"missing camera.json: {camera_path}")
    camera = load_json(camera_path)

    validate_camera_metadata(camera, source=str(camera_path))

    rgb_files = _index_files(root / "rgb", RGB_SUFFIXES)
    depth_files = _index_files(root / "depth", DEPTH_SUFFIXES)
    pose_files = _index_files(root / "pose", {".txt"})
    rgb_ids = set(rgb_files)
    depth_ids = set(depth_files)
    pose_ids = set(pose_files)
    if not rgb_ids:
        raise ValueError(f"no frames found in {root}")
    if not (rgb_ids == depth_ids == pose_ids):
        raise ValueError(
            "frame IDs must match exactly: "
            f"rgb={len(rgb_ids)}, depth={len(depth_ids)}, pose={len(pose_ids)}, "
            f"rgb_only={sorted(rgb_ids - depth_ids - pose_ids)[:5]}, "
            f"depth_only={sorted(depth_ids - rgb_ids - pose_ids)[:5]}, "
            f"pose_only={sorted(pose_ids - rgb_ids - depth_ids)[:5]}"
        )

    valid_ids_path = root / "valid_frame_ids.txt"
    if valid_ids_path.is_file():
        requested_ids = {
            line.strip()
            for line in valid_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        unknown_ids = requested_ids - rgb_ids
        if unknown_ids:
            raise ValueError(
                f"{valid_ids_path} contains unknown IDs: {sorted(unknown_ids)[:5]}"
            )
        frame_ids = sorted(requested_ids, key=_frame_sort_key)
    else:
        frame_ids = sorted(rgb_ids, key=_frame_sort_key)

    if not frame_ids:
        raise ValueError(f"no valid frames selected in {root}")
    for frame_id in frame_ids:
        as_pose_matrix(
            np.loadtxt(pose_files[frame_id], dtype=np.float64),
            name=str(pose_files[frame_id]),
        )

    metadata_path = root / "frames.csv"
    if require_frame_metadata and not metadata_path.is_file():
        raise FileNotFoundError(f"missing frame metadata: {metadata_path}")
    if metadata_path.is_file():
        metadata = read_frame_metadata(metadata_path)
        unknown_metadata = set(metadata) - rgb_ids
        if unknown_metadata:
            raise ValueError(
                f"{metadata_path} contains unknown frame IDs: "
                f"{sorted(unknown_metadata, key=_frame_sort_key)[:5]}"
            )
        missing_metadata = set(frame_ids) - set(metadata)
        if require_frame_metadata and missing_metadata:
            raise ValueError(
                f"{metadata_path} is missing frame IDs: "
                f"{sorted(missing_metadata, key=_frame_sort_key)[:5]}"
            )
        if require_frame_metadata:
            timestamp_fields = (
                "rgb_timestamp_ns",
                "depth_timestamp_ns",
                "pose_timestamp_ns",
            )
            for frame_id in frame_ids:
                row = metadata[frame_id]
                for field in timestamp_fields:
                    value = str(row.get(field, "")).strip()
                    if not value:
                        raise ValueError(
                            f"{metadata_path} has an empty {field} for {frame_id}"
                        )
                    try:
                        timestamp_ns = int(value)
                    except ValueError as error:
                        raise ValueError(
                            f"{metadata_path} has a non-integer {field} for "
                            f"{frame_id}: {value!r}"
                        ) from error
                    if timestamp_ns < 0:
                        raise ValueError(
                            f"{metadata_path} has a negative {field} for {frame_id}"
                        )

    return PartData(
        name=root.name,
        root=root,
        camera=camera,
        frame_ids=frame_ids,
        rgb_files=rgb_files,
        depth_files=depth_files,
        pose_files=pose_files,
    )


def camera_signature(camera):
    """Return fields that must match before parts can share one camera.json."""
    keys = [
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "depth_meters_per_unit",
        "depth_units_per_meter",
        "min_depth_m",
        "max_depth_m",
        "rgb_depth_aligned",
        "distortion_model",
        "distortion_coefficients",
        "image_rectified",
        "rectification_rotation",
        "camera_frame_axes",
        "pose_convention",
    ]
    return {key: camera.get(key) for key in keys}


def validate_camera_compatibility(parts):
    if not parts:
        raise ValueError("no parts were provided")
    reference = camera_signature(parts[0].camera)
    for part in parts[1:]:
        current = camera_signature(part.camera)
        if current != reference:
            differing = [key for key in reference if current[key] != reference[key]]
            raise ValueError(
                f"camera parameters differ for {part.name}: {differing}. "
                "Rectify/resample explicitly before merging."
            )


def read_frame_metadata(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        frame_id = str(row.get("frame_id", "")).strip()
        if not frame_id:
            raise ValueError(f"row without frame_id in {path}")
        if frame_id in result:
            raise ValueError(f"duplicate frame_id {frame_id} in {path}")
        result[frame_id] = row
    return result


def ensure_new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}. Choose a new output path."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
