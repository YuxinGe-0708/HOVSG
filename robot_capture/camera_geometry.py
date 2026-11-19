"""CameraInfo geometry helpers that can run without ROS installed."""

import numpy as np


def projection_metadata(info, *, image_rectified):
    """Select K for raw images, P[:, :3] for rectified images.

    Record R so back-projected rectified rays can be returned to the original
    optical frame used by TF. ROI/binning require explicit upstream handling.
    """
    if getattr(info, "binning_x", 0) > 1 or getattr(info, "binning_y", 0) > 1:
        raise ValueError("Binned CameraInfo requires adjusted calibration")
    roi = getattr(info, "roi", None)
    if roi is not None and (roi.x_offset or roi.y_offset
                            or (roi.width and roi.width != info.width)
                            or (roi.height and roi.height != info.height)):
        raise ValueError("Cropped CameraInfo requires adjusted calibration")
    matrix = np.asarray(info.p if image_rectified else info.k, dtype=np.float64)
    expected_size = 12 if image_rectified else 9
    if matrix.size != expected_size:
        raise ValueError("CameraInfo must contain a complete K/P matrix")
    matrix = matrix.reshape(3, -1)[:, :3]
    if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)):
        raise ValueError("CameraInfo K/P is not a calibrated pinhole matrix")
    rotation = np.eye(3)
    if image_rectified:
        rotation = np.asarray(info.r, dtype=np.float64).reshape(3, 3)
        if (not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError("CameraInfo R must be a valid rectification rotation")
    return {
        "fx": float(matrix[0, 0]), "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2]),
        "rectification_rotation": rotation.tolist(),
    }
