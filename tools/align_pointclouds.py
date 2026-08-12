#!/usr/bin/env python3
"""Register overlapping RGB-D parts and estimate global part transforms."""

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
from hovsg.utils.pose_utils import (
    project_world_transform_to_gravity_aligned_y_up,
)
from tools.multipart_common import (
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


def constrain_registration_transform(transform, registration_dof):
    """Apply the requested world-to-world motion constraint."""
    if registration_dof == "full":
        return np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if registration_dof == "gravity_aligned":
        return project_world_transform_to_gravity_aligned_y_up(
            transform,
            vertical_m=None,
        )
    if registration_dof == "planar":
        return project_world_transform_to_gravity_aligned_y_up(
            transform,
            vertical_m=0.0,
        )
    raise ValueError(f"unknown registration_dof: {registration_dof}")


def register_pair(
    source,
    target,
    voxel_size,
    manual_initial=None,
    registration_dof="planar",
):
    source_down, source_feature = preprocess(source, voxel_size)
    target_down, target_feature = preprocess(target, voxel_size)
    coarse_distance = voxel_size * 2.0

    if manual_initial is None:
        coarse = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_feature,
                target_feature,
                True,
                coarse_distance,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    False
                ),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        coarse_distance
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    100000,
                    0.999,
                ),
            )
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

    initial = constrain_registration_transform(initial, registration_dof)
    fine_distance = voxel_size * 0.8
    refined = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        fine_distance,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    transform = constrain_registration_transform(
        refined.transformation,
        registration_dof,
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


def align_parts(
    *,
    parts_dir,
    part_names,
    voxel_size=0.05,
    overlap_frames=120,
    skip_frames=3,
    pose_mode="raw",
    registration_dof="gravity_aligned",
    min_fitness=0.30,
    max_rmse=0.10,
    manual_initials=None,
):
    """Register capture parts and return metadata, datasets, and a preview."""
    if len(part_names) < 2:
        raise ValueError("at least two parts are required")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if overlap_frames < 10 or skip_frames < 1:
        raise ValueError("overlap-frames must be >= 10 and skip-frames >= 1")
    if not 0.0 <= min_fitness <= 1.0:
        raise ValueError("min_fitness must be between 0 and 1")
    if max_rmse <= 0:
        raise ValueError("max_rmse must be positive")
    if registration_dof not in {"planar", "gravity_aligned", "full"}:
        raise ValueError(f"unknown registration_dof: {registration_dof}")

    parts = [validate_part(Path(parts_dir) / name) for name in part_names]
    validate_camera_compatibility(parts)
    datasets = [make_dataset(part, pose_mode) for part in parts]
    manual_initials = manual_initials or {}

    transformations = {part_names[0]: np.eye(4, dtype=np.float64)}
    pairwise = []
    aligned_cloud = o3d.geometry.PointCloud()

    first_indices = range(0, len(datasets[0]), skip_frames)
    first_cloud, _ = build_submap(datasets[0], first_indices, voxel_size)
    aligned_cloud += first_cloud

    for index in range(1, len(parts)):
        previous = parts[index - 1]
        current = parts[index]
        previous_dataset = datasets[index - 1]
        current_dataset = datasets[index]

        previous_start = max(0, len(previous_dataset) - overlap_frames)
        target_indices = range(
            previous_start,
            len(previous_dataset),
            skip_frames,
        )
        source_indices = range(
            0,
            min(len(current_dataset), overlap_frames),
            skip_frames,
        )
        target, target_frames = build_submap(
            previous_dataset, target_indices, voxel_size
        )
        source, source_frames = build_submap(
            current_dataset, source_indices, voxel_size
        )

        transform_previous_current, quality = register_pair(
            source,
            target,
            voxel_size,
            manual_initial=manual_initials.get(current.name),
            registration_dof=registration_dof,
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
        if quality["fitness"] < min_fitness:
            raise RuntimeError(
                f"registration rejected for {current.name}: fitness "
                f"{quality['fitness']:.4f} < {min_fitness:.4f}"
            )
        if quality["rmse_m"] > max_rmse:
            raise RuntimeError(
                f"registration rejected for {current.name}: RMSE "
                f"{quality['rmse_m']:.4f} m > {max_rmse:.4f} m"
            )

        transformations[current.name] = (
            transformations[previous.name] @ transform_previous_current
        )
        quality["T_target_source"] = transform_previous_current.tolist()
        pairwise.append(quality)

        full_indices = range(0, len(current_dataset), skip_frames)
        current_cloud, _ = build_submap(
            current_dataset, full_indices, voxel_size
        )
        current_cloud.transform(transformations[current.name])
        aligned_cloud += current_cloud
        aligned_cloud = aligned_cloud.voxel_down_sample(voxel_size)

    output = {
        "schema_version": 2,
        "coordinate_frame": "hov_y_up",
        "pose_convention": "T_global_camera_optical",
        "pose_mode": pose_mode,
        "registration_dof": registration_dof,
        "reference_part": part_names[0],
        "parameters": {
            "voxel_size_m": voxel_size,
            "overlap_frames": overlap_frames,
            "skip_frames": skip_frames,
            "min_fitness": min_fitness,
            "max_rmse_m": max_rmse,
        },
        "parts": {
            name: {"T_global_part": transform.tolist()}
            for name, transform in transformations.items()
        },
        "pairwise": pairwise,
    }
    return output, parts, datasets, aligned_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--part-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--overlap-frames", type=int, default=120)
    parser.add_argument("--skip-frames", type=int, default=3)
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
    )
    parser.add_argument("--min-fitness", type=float, default=0.30)
    parser.add_argument("--max-rmse", type=float, default=0.10)
    parser.add_argument("--manual-initials")
    parser.add_argument("--save-aligned-pcd")
    args = parser.parse_args()

    manual_initials = load_manual_initials(args.manual_initials)
    output, _, _, aligned_cloud = align_parts(
        parts_dir=args.parts_dir,
        part_names=args.part_names,
        voxel_size=args.voxel_size,
        overlap_frames=args.overlap_frames,
        skip_frames=args.skip_frames,
        pose_mode=args.pose_mode,
        registration_dof=args.registration_dof,
        min_fitness=args.min_fitness,
        max_rmse=args.max_rmse,
        manual_initials=manual_initials,
    )
    write_json(args.output, output)
    print(f"saved transformations: {Path(args.output).resolve()}")

    if args.save_aligned_pcd:
        output_pcd = Path(args.save_aligned_pcd)
        output_pcd.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_pcd), aligned_cloud)
        print(f"saved aligned preview: {output_pcd.resolve()}")


if __name__ == "__main__":
    main()
