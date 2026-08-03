"""Rigid-pose helpers shared by real-robot and multipart pipelines."""

from __future__ import annotations

import numpy as np


def as_pose_matrix(value, *, name="pose"):
    """Return a validated 4x4 rigid transform."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.size != 16:
        raise ValueError(f"{name} must contain 16 numbers")
    pose = pose.reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")

    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-2):
        raise ValueError(f"{name} rotation determinant is not +1")
    return pose


def camera_tilt_deg_y_up(pose):
    """Measure optical-camera up-vector tilt from a Y-up world axis."""
    pose = as_pose_matrix(pose)
    camera_up = -pose[:3, 1]
    cosine = np.clip(camera_up[1] / np.linalg.norm(camera_up), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def project_camera_pose_to_planar_y_up(pose, height_m=None):
    """Keep camera X/Z translation and yaw while removing roll and pitch.

    The input maps optical-camera coordinates (x right, y down, z forward)
    into a HOV-SG Y-up world. The output preserves the projected optical
    forward direction and uses a fixed vertical translation when requested.
    """
    pose = as_pose_matrix(pose).copy()
    forward = pose[:3, 2].copy()
    forward[1] = 0.0
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("camera forward axis is vertical; yaw is undefined")
    forward /= forward_norm

    down = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, down)
    forward /= np.linalg.norm(forward)

    pose[:3, :3] = np.column_stack((right, down, forward))
    if height_m is not None:
        pose[1, 3] = float(height_m)
    return pose


def project_world_transform_to_planar_y_up(transform, vertical_m=0.0):
    """Project a Y-up world-to-world transform onto SE(2)."""
    transform = as_pose_matrix(transform, name="transform").copy()
    forward = transform[:3, 2].copy()
    forward[1] = 0.0
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("transform forward axis is vertical; yaw is undefined")
    forward /= forward_norm

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, forward)
    x_axis /= np.linalg.norm(x_axis)
    forward = np.cross(x_axis, up)
    forward /= np.linalg.norm(forward)

    transform[:3, :3] = np.column_stack((x_axis, up, forward))
    transform[1, 3] = float(vertical_m)
    return transform


def pose_delta(previous_pose, current_pose):
    """Return translation distance in metres and rotation distance in degrees."""
    previous_pose = as_pose_matrix(previous_pose, name="previous_pose")
    current_pose = as_pose_matrix(current_pose, name="current_pose")
    translation_m = float(
        np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3])
    )
    relative_rotation = previous_pose[:3, :3].T @ current_pose[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_deg
