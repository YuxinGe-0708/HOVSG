#!/usr/bin/env python3
"""Register overlapping RGB-D parts and estimate planar global transforms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hovsg.dataloader.real_robot import RealRobotDataset
from hovsg.utils.pose_utils import project_world_transform_to_planar_y_up
from multipart_common import (
    load_json,
    validate_camera_compatibility,
    validate_part,
    write_json,
)


def make_dataset(part, pose_mode):
    return RealRobotDataset(
        {
            "root_dir": str(part.root),
            "transforms": None,
            "pose_mode": pose_mode,
            "planar_pose_height_m": None,
        }
    )


def build_submap(dataset, indices, voxel_size):
    pointcloud = o3d.geometry.PointCloud()
    used_frames = 0
    for position, index in enumerate(indices, start=1):
        rgb, depth, pose, _, _ = dataset[index]
        frame = dataset.create_pcd(rgb, depth, camera_pose=pose)
        if frame.is_empty():
            continue
        pointcloud += frame
        used_frames += 1
        if position % 10 == 0:
            pointcloud = pointcloud.voxel_down_sample(voxel_size)
    if pointcloud.is_empty():
        raise RuntimeError("overlap window generated an empty point cloud")
    return pointcloud.voxel_down_sample(voxel_size), used_frames


def preprocess(pointcloud, voxel_size):
    down = pointcloud.voxel_down_sample(voxel_size)
    normal_radius = voxel_size * 2.5
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=40)
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0,
            max_nn=100,
        ),
    )
    return down, feature


def register_pair(source, target, voxel_size, manual_initial=None):
    source_down, source_feature = preprocess(source, voxel_size)
    target_down, target_feature = preprocess(target, voxel_size)
    coarse_distance = voxel_size * 2.0

    if manual_initial is None:
        coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            True,
            coarse_distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    coarse_distance
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        initial = coarse.transformation
        coarse_fitness = float(coarse.fitness)
        coarse_rmse = float(coarse.inlier_rmse)
    else:
        initial = np.asarray(manual_initial, dtype=np.float64).reshape(4, 4)
        coarse_eval = o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, coarse_distance, initial
        )
        coarse_fitness = float(coarse_eval.fitness)
        coarse_rmse = float(coarse_eval.inlier_rmse)

    initial = project_world_transform_to_planar_y_up(initial, vertical_m=0.0)
    fine_distance = voxel_size * 0.8
    refined = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        fine_distance,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    transform = project_world_transform_to_planar_y_up(
        refined.transformation,
        vertical_m=0.0,
    )
    final = o3d.pipelines.registration.evaluate_registration(
        source_down,
        target_down,
        fine_distance,
        transform,
    )
    return transform, {
        "coarse_fitness": coarse_fitness,
        "coarse_rmse_m": coarse_rmse,
        "fitness": float(final.fitness),
        "rmse_m": float(final.inlier_rmse),
        "source_points": len(source_down.points),
        "target_points": len(target_down.points),
    }


def load_manual_initials(path):
    if path is None:
        return {}
    data = load_json(path)
    return {
        name: np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        for name, matrix in data.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--part-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--overlap-frames", type=int, default=120)
    parser.add_argument("--skip-frames", type=int, default=3)
    parser.add_argument("--pose-mode", choices=("raw", "planar"), default="planar")
    parser.add_argument("--min-fitness", type=float, default=0.30)
    parser.add_argument("--max-rmse", type=float, default=0.10)
    parser.add_argument("--manual-initials")
    parser.add_argument("--save-aligned-pcd")
    args = parser.parse_args()

    if len(args.part_names) < 2:
        raise ValueError("at least two parts are required")
    if args.overlap_frames < 10 or args.skip_frames < 1:
        raise ValueError("overlap-frames must be >= 10 and skip-frames >= 1")

    parts = [
        validate_part(Path(args.parts_dir) / name) for name in args.part_names
    ]
    validate_camera_compatibility(parts)
    datasets = [make_dataset(part, args.pose_mode) for part in parts]
    manual_initials = load_manual_initials(args.manual_initials)

    transformations = {args.part_names[0]: np.eye(4, dtype=np.float64)}
    pairwise = []
    aligned_cloud = o3d.geometry.PointCloud()

    first_indices = range(0, len(datasets[0]), args.skip_frames)
    first_cloud, _ = build_submap(datasets[0], first_indices, args.voxel_size)
    aligned_cloud += first_cloud

    for index in range(1, len(parts)):
        previous = parts[index - 1]
        current = parts[index]
        previous_dataset = datasets[index - 1]
        current_dataset = datasets[index]

        previous_start = max(0, len(previous_dataset) - args.overlap_frames)
        target_indices = range(
            previous_start,
            len(previous_dataset),
            args.skip_frames,
        )
        source_indices = range(
            0,
            min(len(current_dataset), args.overlap_frames),
            args.skip_frames,
        )
        target, target_frames = build_submap(
            previous_dataset, target_indices, args.voxel_size
        )
        source, source_frames = build_submap(
            current_dataset, source_indices, args.voxel_size
        )

        transform_previous_current, quality = register_pair(
            source,
            target,
            args.voxel_size,
            manual_initial=manual_initials.get(current.name),
        )
        quality.update(
            {
                "source_part": current.name,
                "target_part": previous.name,
                "source_overlap_frames": source_frames,
                "target_overlap_frames": target_frames,
            }
        )
        print(
            f"{current.name} -> {previous.name}: "
            f"fitness={quality['fitness']:.4f}, rmse={quality['rmse_m']:.4f} m"
        )
        if quality["fitness"] < args.min_fitness:
            raise RuntimeError(
                f"registration rejected for {current.name}: fitness "
                f"{quality['fitness']:.4f} < {args.min_fitness:.4f}"
            )
        if quality["rmse_m"] > args.max_rmse:
            raise RuntimeError(
                f"registration rejected for {current.name}: RMSE "
                f"{quality['rmse_m']:.4f} m > {args.max_rmse:.4f} m"
            )

        transformations[current.name] = (
            transformations[previous.name] @ transform_previous_current
        )
        quality["T_target_source"] = transform_previous_current.tolist()
        pairwise.append(quality)

        full_indices = range(0, len(current_dataset), args.skip_frames)
        current_cloud, _ = build_submap(
            current_dataset, full_indices, args.voxel_size
        )
        current_cloud.transform(transformations[current.name])
        aligned_cloud += current_cloud
        aligned_cloud = aligned_cloud.voxel_down_sample(args.voxel_size)

    output = {
        "schema_version": 1,
        "coordinate_frame": "hov_y_up",
        "pose_convention": "T_global_camera_optical",
        "pose_mode": args.pose_mode,
        "reference_part": args.part_names[0],
        "parameters": {
            "voxel_size_m": args.voxel_size,
            "overlap_frames": args.overlap_frames,
            "skip_frames": args.skip_frames,
            "min_fitness": args.min_fitness,
            "max_rmse_m": args.max_rmse,
        },
        "parts": {
            name: {"T_global_part": transform.tolist()}
            for name, transform in transformations.items()
        },
        "pairwise": pairwise,
    }
    write_json(args.output, output)
    print(f"saved transformations: {Path(args.output).resolve()}")

    if args.save_aligned_pcd:
        output_pcd = Path(args.save_aligned_pcd)
        output_pcd.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_pcd), aligned_cloud)
        print(f"saved aligned preview: {output_pcd.resolve()}")


if __name__ == "__main__":
    main()
