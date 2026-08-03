#!/usr/bin/env python3
"""Strictly merge globally aligned RGB-D parts into one HOV-SG dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multipart_common import (
    ensure_new_directory,
    read_frame_metadata,
    validate_camera_compatibility,
    validate_part,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformed-dir", required=True)
    parser.add_argument("--part-names", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    parts = [
        validate_part(Path(args.transformed_dir) / name)
        for name in args.part_names
    ]
    validate_camera_compatibility(parts)
    for part in parts:
        if part.camera.get("world_frame") != "hov_y_up":
            raise ValueError(f"{part.name} is not transformed to hov_y_up")
        if part.camera.get("pose_convention") != "T_global_camera_optical":
            raise ValueError(f"{part.name} has an unsupported pose convention")

    output = ensure_new_directory(args.output_dir)
    rgb_output = output / "rgb"
    depth_output = output / "depth"
    pose_output = output / "pose"
    for directory in (rgb_output, depth_output, pose_output):
        directory.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    part_records = []
    next_frame = 1
    for part in parts:
        metadata = read_frame_metadata(part.root / "frames.csv")
        output_start = next_frame
        for source_id in part.frame_ids:
            output_id = f"{next_frame:06d}"
            rgb_source = part.rgb_files[source_id]
            depth_source = part.depth_files[source_id]
            pose_source = part.pose_files[source_id]
            shutil.copy2(
                rgb_source,
                rgb_output / f"{output_id}{rgb_source.suffix.lower()}",
            )
            shutil.copy2(
                depth_source,
                depth_output / f"{output_id}{depth_source.suffix.lower()}",
            )
            shutil.copy2(pose_source, pose_output / f"{output_id}.txt")

            source_metadata = metadata.get(source_id, {})
            frame_rows.append(
                {
                    "frame_id": output_id,
                    "source_part": part.name,
                    "source_frame_id": source_id,
                    "rgb_timestamp_ns": source_metadata.get(
                        "rgb_timestamp_ns", ""
                    ),
                    "depth_timestamp_ns": source_metadata.get(
                        "depth_timestamp_ns", ""
                    ),
                    "pose_timestamp_ns": source_metadata.get(
                        "pose_timestamp_ns", ""
                    ),
                    "tracking_quality": source_metadata.get(
                        "tracking_quality", ""
                    ),
                }
            )
            next_frame += 1

        part_records.append(
            {
                "part_name": part.name,
                "source_frames": len(part.frame_ids),
                "output_first_frame": f"{output_start:06d}",
                "output_last_frame": f"{next_frame - 1:06d}",
                "T_global_part": part.camera.get("T_global_part"),
            }
        )
        print(f"merged {part.name}: {len(part.frame_ids)} frames")

    camera = dict(parts[0].camera)
    for key in (
        "multipart_source_part",
        "source_world_frame",
        "T_global_part",
    ):
        camera.pop(key, None)
    camera["multipart"] = True
    camera["multipart_parts"] = args.part_names
    write_json(output / "camera.json", camera)

    fieldnames = [
        "frame_id",
        "source_part",
        "source_frame_id",
        "rgb_timestamp_ns",
        "depth_timestamp_ns",
        "pose_timestamp_ns",
        "tracking_quality",
    ]
    with (output / "frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    manifest = {
        "schema_version": 1,
        "dataset_type": "hovsg_multipart_rgbd",
        "world_frame": "hov_y_up",
        "pose_convention": "T_global_camera_optical",
        "total_frames": len(frame_rows),
        "parts": part_records,
    }
    write_json(output / "merge_manifest.json", manifest)

    validated = validate_part(output, require_frame_metadata=True)
    print(f"merged dataset verified: {len(validated.frame_ids)} frames -> {output}")


if __name__ == "__main__":
    main()
