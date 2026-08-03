#!/usr/bin/env python3
"""Convert each part to global HOV-SG poses using registration transforms."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hovsg.dataloader.real_robot import RealRobotDataset
from hovsg.utils.pose_utils import as_pose_matrix
from multipart_common import (
    ensure_new_directory,
    load_json,
    validate_camera_compatibility,
    validate_part,
    write_json,
)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_selected_frame_metadata(source, destination, frame_ids):
    if not source.is_file():
        return
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "frame_id" not in fieldnames:
            raise ValueError(f"invalid frame metadata header: {source}")
        rows_by_id = {
            str(row["frame_id"]).strip(): row
            for row in reader
            if str(row.get("frame_id", "")).strip()
        }
    selected_rows = []
    for frame_id in frame_ids:
        row = rows_by_id.get(frame_id, {key: "" for key in fieldnames})
        row["frame_id"] = frame_id
        selected_rows.append(row)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--part-names", nargs="+", required=True)
    parser.add_argument("--transformations", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    registration = load_json(args.transformations)
    if registration.get("coordinate_frame") != "hov_y_up":
        raise ValueError("registration transforms must use hov_y_up coordinates")
    if registration.get("pose_convention") != "T_global_camera_optical":
        raise ValueError("unsupported registration pose convention")

    registered_parts = registration.get("parts", {})
    missing_transforms = set(args.part_names) - set(registered_parts)
    if missing_transforms:
        raise ValueError(f"missing part transforms: {sorted(missing_transforms)}")

    parts = [
        validate_part(Path(args.parts_dir) / name) for name in args.part_names
    ]
    validate_camera_compatibility(parts)
    output_root = ensure_new_directory(args.output_dir)
    pose_mode = registration.get("pose_mode", "planar")

    for part in parts:
        part_output = output_root / part.name
        for directory in ("rgb", "depth", "pose"):
            (part_output / directory).mkdir(parents=True, exist_ok=True)

        dataset = RealRobotDataset(
            {
                "root_dir": str(part.root),
                "transforms": None,
                "pose_mode": pose_mode,
                "planar_pose_height_m": None,
            }
        )
        transform = as_pose_matrix(
            registered_parts[part.name]["T_global_part"],
            name=f"T_global_{part.name}",
        )
        data_by_id = {
            pose_path.stem: (rgb_path, depth_path, pose_path)
            for rgb_path, depth_path, pose_path in dataset.data_list
        }

        for frame_id in part.frame_ids:
            rgb_path, depth_path, pose_path = data_by_id[frame_id]
            part_pose = dataset.load_pose(pose_path)
            global_pose = as_pose_matrix(
                transform @ part_pose,
                name=f"global pose {part.name}/{frame_id}",
            )
            copy_file(
                rgb_path,
                part_output / "rgb" / rgb_path.name,
            )
            copy_file(
                depth_path,
                part_output / "depth" / depth_path.name,
            )
            np.savetxt(
                part_output / "pose" / f"{frame_id}.txt",
                global_pose,
                fmt="%.9f",
            )

        camera = dict(part.camera)
        camera.update(
            {
                "world_frame": "hov_y_up",
                "pose_convention": "T_global_camera_optical",
                "source_world_frame": part.camera.get("world_frame"),
                "multipart_pose_mode": pose_mode,
                "multipart_source_part": part.name,
                "T_global_part": transform.tolist(),
            }
        )
        write_json(part_output / "camera.json", camera)
        copy_selected_frame_metadata(
            part.root / "frames.csv",
            part_output / "frames.csv",
            part.frame_ids,
        )
        for metadata_name in ("part.json", "valid_frame_ids.txt"):
            metadata_path = part.root / metadata_name
            if metadata_path.is_file():
                copy_file(metadata_path, part_output / metadata_name)

        print(
            f"transformed {part.name}: {len(part.frame_ids)} frames -> "
            f"{part_output}"
        )


if __name__ == "__main__":
    main()
