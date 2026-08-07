"""Register multiple real-robot capture parts and create one geometry map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hovsg.utils.pose_utils import as_pose_matrix
from hovsg.utils.runtime_status import print_status
from tools.align_pointclouds import (
    align_parts,
    build_submap,
    load_manual_initials,
)
from tools.multipart_common import ensure_new_directory, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-dir", required=True, type=Path)
    parser.add_argument(
        "--part-names",
        nargs="+",
        default=["part_01", "part_02"],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registration-voxel-size", type=float, default=0.05)
    parser.add_argument("--geometry-voxel-size", type=float, default=0.03)
    parser.add_argument("--overlap-frames", type=int, default=120)
    parser.add_argument("--registration-skip-frames", type=int, default=3)
    parser.add_argument("--geometry-skip-frames", type=int, default=1)
    parser.add_argument(
        "--pose-mode",
        choices=("raw", "planar"),
        default="raw",
        help="Handheld RGB-D+IMU capture must use raw.",
    )
    parser.add_argument(
        "--registration-dof",
        choices=("planar", "gravity_aligned", "full"),
        default="gravity_aligned",
        help=(
            "gravity_aligned keeps XYZ translation and yaw while removing "
            "inter-part roll/pitch"
        ),
    )
    parser.add_argument("--min-fitness", type=float, default=0.30)
    parser.add_argument("--max-rmse", type=float, default=0.10)
    parser.add_argument("--max-vertical-offset", type=float, default=1.0)
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--manual-initials", type=Path)
    args = parser.parse_args()

    for name, value in (
        ("registration-voxel-size", args.registration_voxel_size),
        ("geometry-voxel-size", args.geometry_voxel_size),
        ("max-rmse", args.max_rmse),
        ("max-vertical-offset", args.max_vertical_offset),
        ("outlier-std-ratio", args.outlier_std_ratio),
    ):
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.overlap_frames < 10:
        parser.error("--overlap-frames must be at least 10")
    if args.registration_skip_frames < 1 or args.geometry_skip_frames < 1:
        parser.error("frame skip values must be positive")
    if args.outlier_nb_neighbors < 2:
        parser.error("--outlier-nb-neighbors must be at least 2")
    return args


def validate_vertical_offsets(registration, max_vertical_offset):
    """Reject implausible vertical shifts before fusing full-resolution data."""
    if registration["registration_dof"] != "gravity_aligned":
        return
    for pair in registration["pairwise"]:
        transform = as_pose_matrix(
            pair["T_target_source"],
            name=f"{pair['source_part']} registration",
        )
        vertical_offset = abs(float(transform[1, 3]))
        pair["vertical_offset_m"] = vertical_offset
        if vertical_offset > max_vertical_offset:
            raise RuntimeError(
                f"registration rejected for {pair['source_part']}: vertical "
                f"offset {vertical_offset:.4f} m exceeds "
                f"{max_vertical_offset:.4f} m"
            )


def create_full_geometry(
    datasets,
    part_names,
    registration,
    *,
    voxel_size,
    skip_frames,
):
    """Fuse every selected part into the registration reference frame."""
    full_cloud = o3d.geometry.PointCloud()
    part_statistics = []
    for part_name, dataset in zip(part_names, datasets):
        indices = range(0, len(dataset), skip_frames)
        part_cloud, used_frames = build_submap(dataset, indices, voxel_size)
        transform = as_pose_matrix(
            registration["parts"][part_name]["T_global_part"],
            name=f"T_global_{part_name}",
        )
        part_cloud.transform(transform)
        full_cloud += part_cloud
        full_cloud = full_cloud.voxel_down_sample(voxel_size)
        statistics = {
            "part_name": part_name,
            "dataset_frames": len(dataset),
            "used_frames": used_frames,
            "part_points": len(part_cloud.points),
            "resident_points": len(full_cloud.points),
        }
        part_statistics.append(statistics)
        print_status("MULTIPART_GEOMETRY_PART", "OK", **statistics)
    if full_cloud.is_empty():
        raise RuntimeError("multipart fusion generated an empty point cloud")
    return full_cloud, part_statistics


def main():
    args = parse_args()
    output_dir = ensure_new_directory(args.output_dir.resolve())
    manual_initials = load_manual_initials(args.manual_initials)

    print_status(
        "MULTIPART_GEOMETRY",
        "START",
        room=args.room_dir.resolve(),
        parts=args.part_names,
        output=output_dir,
        pose_mode=args.pose_mode,
        registration_dof=args.registration_dof,
    )
    registration, _, datasets, preview = align_parts(
        parts_dir=args.room_dir,
        part_names=args.part_names,
        voxel_size=args.registration_voxel_size,
        overlap_frames=args.overlap_frames,
        skip_frames=args.registration_skip_frames,
        pose_mode=args.pose_mode,
        registration_dof=args.registration_dof,
        min_fitness=args.min_fitness,
        max_rmse=args.max_rmse,
        manual_initials=manual_initials,
    )
    validate_vertical_offsets(registration, args.max_vertical_offset)

    preview_path = output_dir / "registration_preview.ply"
    if not o3d.io.write_point_cloud(str(preview_path), preview):
        raise RuntimeError(f"failed to save {preview_path}")

    full_cloud, part_statistics = create_full_geometry(
        datasets,
        args.part_names,
        registration,
        voxel_size=args.geometry_voxel_size,
        skip_frames=args.geometry_skip_frames,
    )
    voxel_points = len(full_cloud.points)
    if voxel_points > args.outlier_nb_neighbors:
        full_cloud, _ = full_cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )

    output_path = output_dir / "full_pcd.ply"
    if not o3d.io.write_point_cloud(str(output_path), full_cloud):
        raise RuntimeError(f"failed to save {output_path}")

    registration["geometry"] = {
        "voxel_size_m": args.geometry_voxel_size,
        "skip_frames": args.geometry_skip_frames,
        "outlier_nb_neighbors": args.outlier_nb_neighbors,
        "outlier_std_ratio": args.outlier_std_ratio,
        "voxel_points": voxel_points,
        "final_points": len(full_cloud.points),
        "parts": part_statistics,
    }
    transformations_path = output_dir / "transformations.json"
    write_json(transformations_path, registration)

    print_status(
        "MULTIPART_GEOMETRY",
        "DONE",
        output=output_path,
        registration_preview=preview_path,
        transformations=transformations_path,
        voxel_points=voxel_points,
        final_points=len(full_cloud.points),
        min_bound=np.asarray(full_cloud.get_min_bound()).round(4).tolist(),
        max_bound=np.asarray(full_cloud.get_max_bound()).round(4).tolist(),
    )


if __name__ == "__main__":
    main()
