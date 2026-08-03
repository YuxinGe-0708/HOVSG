"""Create a geometry-only point cloud from a real-robot RGB-D recording."""

import os

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig
from tqdm import tqdm

from hovsg.dataloader.real_robot import RealRobotDataset
from hovsg.utils.runtime_status import print_status


def _validate_pose(pose):
    """Validate a homogeneous camera pose before using it for fusion."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        return False, f"invalid_shape_{pose.shape}"
    if not np.isfinite(pose).all():
        return False, "contains_non_finite_values"
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        return False, "invalid_homogeneous_last_row"

    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        return False, "rotation_not_orthonormal"
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        return False, "invalid_rotation_determinant"
    return True, ""


def _pose_delta(previous_pose, current_pose):
    """Return translation distance in metres and rotation distance in degrees."""
    translation_m = float(
        np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3])
    )
    relative_rotation = previous_pose[:3, :3].T @ current_pose[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_deg


@hydra.main(version_base=None, config_path="../config", config_name="real_robot")
def main(params: DictConfig):
    """Generate and save only the fused scene point cloud."""
    scene_path = os.path.join(
        params.main.dataset_path,
        params.main.split,
        params.main.scene_id,
    )
    save_dir = os.path.join(
        params.main.save_path,
        params.main.dataset,
        params.main.scene_id,
    )
    os.makedirs(save_dir, exist_ok=True)

    dataset = RealRobotDataset(
        {
            "root_dir": scene_path,
            "transforms": None,
            "min_depth_m": params.pipeline.get("min_depth_m"),
            "max_depth_m": params.pipeline.get("max_depth_m"),
            "pose_mode": params.pipeline.get("pose_mode", "raw"),
            "planar_pose_height_m": params.pipeline.get(
                "planar_pose_height_m"
            ),
            "pose_warn_tilt_deg": params.pipeline.get(
                "pose_warn_tilt_deg", 5.0
            ),
            "pose_warn_height_drift_m": params.pipeline.get(
                "pose_warn_height_drift_m", 0.15
            ),
        }
    )

    skip_frames = max(1, int(params.pipeline.get("skip_frames", 1)))
    voxel_size = float(params.pipeline.get("voxel_size", 0.03))
    fusion_batch_size = max(
        1,
        int(params.pipeline.get("pointcloud_fusion_batch_size", 10)),
    )
    translation_threshold = max(
        0.0,
        float(params.pipeline.get("keyframe_translation_m", 0.0)),
    )
    rotation_threshold = max(
        0.0,
        float(params.pipeline.get("keyframe_rotation_deg", 0.0)),
    )
    max_translation_jump = float(
        params.pipeline.get("max_pose_translation_jump_m", np.inf)
    )
    max_rotation_jump = float(
        params.pipeline.get("max_pose_rotation_jump_deg", np.inf)
    )
    log_every = max(1, int(params.pipeline.get("log_every_n_frames", 10)))

    candidate_indices = list(range(0, len(dataset), skip_frames))
    full_pcd = o3d.geometry.PointCloud()
    accepted_indices = []
    last_accepted_pose = None
    frames_since_downsample = 0
    invalid_poses = 0
    pose_jumps = 0
    non_keyframes = 0

    print_status(
        "GEOMETRY_MAP",
        "START",
        scene=scene_path,
        output=save_dir,
        dataset_frames=len(dataset),
        candidate_frames=len(candidate_indices),
        voxel_size=voxel_size,
        translation_threshold_m=translation_threshold,
        rotation_threshold_deg=rotation_threshold,
    )

    for candidate_position, frame_index in enumerate(
        tqdm(candidate_indices, desc="Creating geometry map")
    ):
        rgb, depth, pose, _, _ = dataset[frame_index]
        pose = np.asarray(pose, dtype=np.float64)
        pose_valid, pose_error = _validate_pose(pose)
        if not pose_valid:
            invalid_poses += 1
            print_status(
                "GEOMETRY_FRAME",
                "WARN",
                frame_index=frame_index,
                reason=pose_error,
            )
            continue

        translation_delta = 0.0
        rotation_delta = 0.0
        if last_accepted_pose is not None:
            translation_delta, rotation_delta = _pose_delta(
                last_accepted_pose,
                pose,
            )
            if (
                translation_delta > max_translation_jump
                or rotation_delta > max_rotation_jump
            ):
                pose_jumps += 1
                print_status(
                    "GEOMETRY_FRAME",
                    "WARN",
                    frame_index=frame_index,
                    reason="pose_jump",
                    translation_delta_m=translation_delta,
                    rotation_delta_deg=rotation_delta,
                )
                continue

            keyframes_enabled = (
                translation_threshold > 0.0 or rotation_threshold > 0.0
            )
            translation_reached = (
                translation_threshold > 0.0
                and translation_delta >= translation_threshold
            )
            rotation_reached = (
                rotation_threshold > 0.0
                and rotation_delta >= rotation_threshold
            )
            if keyframes_enabled and not (
                translation_reached or rotation_reached
            ):
                non_keyframes += 1
                continue

        frame_pcd = dataset.create_pcd(rgb, depth, camera_pose=pose)
        if frame_pcd.is_empty():
            print_status(
                "GEOMETRY_FRAME",
                "WARN",
                frame_index=frame_index,
                reason="empty_point_cloud",
            )
            continue

        full_pcd += frame_pcd
        accepted_indices.append(frame_index)
        last_accepted_pose = pose.copy()
        frames_since_downsample += 1

        if frames_since_downsample >= fusion_batch_size:
            full_pcd = full_pcd.voxel_down_sample(voxel_size)
            frames_since_downsample = 0

        if (
            len(accepted_indices) == 1
            or len(accepted_indices) % log_every == 0
            or candidate_position == len(candidate_indices) - 1
        ):
            print_status(
                "GEOMETRY_FRAME",
                "OK",
                frame_index=frame_index,
                progress=f"{candidate_position + 1}/{len(candidate_indices)}",
                accepted_keyframes=len(accepted_indices),
                frame_points=len(frame_pcd.points),
                resident_points=len(full_pcd.points),
                translation_delta_m=translation_delta,
                rotation_delta_deg=rotation_delta,
            )

    if full_pcd.is_empty():
        raise RuntimeError("No valid 3D points were generated")

    full_pcd = full_pcd.voxel_down_sample(voxel_size)
    voxel_points = len(full_pcd.points)
    outlier_neighbors = max(
        2,
        int(params.pipeline.get("outlier_nb_neighbors", 20)),
    )
    outlier_std_ratio = float(
        params.pipeline.get("outlier_std_ratio", 2.0)
    )
    if voxel_points > outlier_neighbors:
        full_pcd, _ = full_pcd.remove_statistical_outlier(
            nb_neighbors=outlier_neighbors,
            std_ratio=outlier_std_ratio,
        )

    output_path = os.path.join(save_dir, "full_pcd.ply")
    saved = o3d.io.write_point_cloud(output_path, full_pcd)
    if not saved:
        raise RuntimeError(f"Failed to save point cloud: {output_path}")

    print_status(
        "GEOMETRY_MAP",
        "DONE",
        output=output_path,
        dataset_frames=len(dataset),
        accepted_keyframes=len(accepted_indices),
        rejected_invalid_pose=invalid_poses,
        rejected_pose_jump=pose_jumps,
        rejected_non_keyframe=non_keyframes,
        voxel_points=voxel_points,
        final_points=len(full_pcd.points),
        min_bound=np.asarray(full_pcd.get_min_bound()).round(4).tolist(),
        max_bound=np.asarray(full_pcd.get_max_bound()).round(4).tolist(),
        raw_pose_diagnostics=dataset.get_pose_diagnostics(),
    )


if __name__ == "__main__":
    main()
