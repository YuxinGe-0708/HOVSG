#!/usr/bin/env python3
"""Initialize and validate multipart capture directories.

This tool does not subscribe to ROS topics. It defines the filesystem contract
that an existing robot recorder must write and validates each part immediately
after capture.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multipart_common import (
    camera_signature,
    load_json,
    read_frame_metadata,
    validate_camera_compatibility,
    validate_camera_metadata,
    validate_part,
    write_json,
)


FRAME_FIELDS = [
    "frame_id",
    "rgb_timestamp_ns",
    "depth_timestamp_ns",
    "pose_timestamp_ns",
    "tracking_quality",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_frame_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def init_room(args):
    room = Path(args.room_dir).resolve()
    room.mkdir(parents=True, exist_ok=True)
    camera = load_json(args.camera_json)
    validate_camera_metadata(camera, source=str(args.camera_json))
    metadata = {
        "schema_version": 1,
        "dataset_type": "hovsg_multipart_capture",
        "room_id": args.room_id or room.name,
        "created_at_utc": utc_now(),
        "camera_signature": camera_signature(camera),
        "requirements": {
            "pose_convention": "T_world_camera_optical",
            "minimum_overlap_ratio": 0.30,
            "recommended_overlap_ratio": 0.50,
        },
    }
    capture_path = room / "capture.json"
    if capture_path.exists():
        raise FileExistsError(f"capture already initialized: {capture_path}")
    write_json(capture_path, metadata)
    shutil.copy2(args.camera_json, room / "camera_template.json")
    print(f"initialized multipart room: {room}")


def start_part(args):
    room = Path(args.room_dir).resolve()
    if not (room / "capture.json").is_file():
        raise FileNotFoundError("run init-room before start-part")
    camera_source = Path(args.camera_json or room / "camera_template.json")
    camera = load_json(camera_source)
    validate_camera_metadata(camera, source=str(camera_source))
    template = load_json(room / "camera_template.json")
    if camera_signature(camera) != camera_signature(template):
        raise ValueError("part camera parameters differ from the room template")

    part = room / args.part_name
    if part.exists():
        raise FileExistsError(f"part already exists: {part}")
    for directory in ("rgb", "depth", "pose"):
        (part / directory).mkdir(parents=True, exist_ok=False)

    shutil.copy2(camera_source, part / "camera.json")
    write_frame_rows(part / "frames.csv", [])
    write_json(
        part / "part.json",
        {
            "schema_version": 1,
            "part_name": args.part_name,
            "started_at_utc": utc_now(),
            "status": "recording",
            "previous_part": args.previous_part,
            "overlap_description": args.overlap_description,
            "pose_convention": "T_world_camera_optical",
            "recorder_output": {
                "rgb_dir": "rgb",
                "depth_dir": "depth",
                "pose_dir": "pose",
                "frame_metadata": "frames.csv",
            },
        },
    )
    print(f"part ready: {part}")
    print("configure the recorder to write matching frame IDs into rgb/depth/pose")
    print("append one frames.csv row per saved frame using nanosecond timestamps")


def finalize_part(args):
    part_dir = Path(args.room_dir).resolve() / args.part_name
    part = validate_part(part_dir, require_frame_metadata=False)
    selected_ids = sorted(
        part.rgb_files,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if args.last_valid_frame is not None:
        if args.last_valid_frame not in selected_ids:
            raise ValueError(
                f"last-valid-frame does not exist: {args.last_valid_frame}"
            )
        selected_ids = selected_ids[
            : selected_ids.index(args.last_valid_frame) + 1
        ]
    if args.trim_tail_frames:
        if args.trim_tail_frames >= len(selected_ids):
            raise ValueError("trim-tail-frames would remove every frame")
        selected_ids = selected_ids[: -args.trim_tail_frames]

    (part_dir / "valid_frame_ids.txt").write_text(
        "".join(f"{frame_id}\n" for frame_id in selected_ids),
        encoding="utf-8",
    )

    existing_metadata = read_frame_metadata(part_dir / "frames.csv")
    missing_metadata = [
        frame_id for frame_id in selected_ids if frame_id not in existing_metadata
    ]
    if missing_metadata and args.require_timestamps:
        raise ValueError(
            "frames.csv is missing selected frame IDs: "
            f"{missing_metadata[:5]}"
        )
    rows = []
    for frame_id in sorted(
        part.rgb_files,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        row = existing_metadata.get(frame_id, {})
        rows.append(
            {
                "frame_id": frame_id,
                "rgb_timestamp_ns": row.get("rgb_timestamp_ns", ""),
                "depth_timestamp_ns": row.get("depth_timestamp_ns", ""),
                "pose_timestamp_ns": row.get("pose_timestamp_ns", ""),
                "tracking_quality": row.get("tracking_quality", ""),
            }
        )
    write_frame_rows(part_dir / "frames.csv", rows)

    part_metadata_path = part_dir / "part.json"
    metadata = (
        load_json(part_metadata_path)
        if part_metadata_path.is_file()
        else {"schema_version": 1, "part_name": args.part_name}
    )
    metadata.update(
        {
            "status": "finalized",
            "finalized_at_utc": utc_now(),
            "captured_frames": len(part.rgb_files),
            "valid_frames": len(selected_ids),
            "timestamps_complete": not missing_metadata,
            "last_valid_frame": selected_ids[-1],
        }
    )
    write_json(part_metadata_path, metadata)
    validated = validate_part(part_dir, require_frame_metadata=True)
    print(
        f"finalized {args.part_name}: {len(validated.frame_ids)} valid frames; "
        f"metadata_complete={not missing_metadata}"
    )


def validate_room(args):
    room = Path(args.room_dir).resolve()
    names = args.part_names or sorted(
        path.name
        for path in room.iterdir()
        if path.is_dir() and (path / "camera.json").is_file()
    )
    if not names:
        raise ValueError(f"no capture parts found under {room}")
    parts = [
        validate_part(room / name, require_frame_metadata=args.require_timestamps)
        for name in names
    ]
    validate_camera_compatibility(parts)
    for part in parts:
        print(f"{part.name}: {len(part.frame_ids)} valid synchronized frames")
    print(f"room capture valid: {len(parts)} parts")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-room")
    init_parser.add_argument("--room-dir", required=True)
    init_parser.add_argument("--camera-json", required=True)
    init_parser.add_argument("--room-id")
    init_parser.set_defaults(handler=init_room)

    start_parser = subparsers.add_parser("start-part")
    start_parser.add_argument("--room-dir", required=True)
    start_parser.add_argument("--part-name", required=True)
    start_parser.add_argument("--camera-json")
    start_parser.add_argument("--previous-part")
    start_parser.add_argument("--overlap-description", default="")
    start_parser.set_defaults(handler=start_part)

    finalize_parser = subparsers.add_parser("finalize-part")
    finalize_parser.add_argument("--room-dir", required=True)
    finalize_parser.add_argument("--part-name", required=True)
    finalize_parser.add_argument("--last-valid-frame")
    finalize_parser.add_argument("--trim-tail-frames", type=int, default=0)
    finalize_parser.add_argument("--require-timestamps", action="store_true")
    finalize_parser.set_defaults(handler=finalize_part)

    validate_parser = subparsers.add_parser("validate-room")
    validate_parser.add_argument("--room-dir", required=True)
    validate_parser.add_argument("--part-names", nargs="+")
    validate_parser.add_argument("--require-timestamps", action="store_true")
    validate_parser.set_defaults(handler=validate_room)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
