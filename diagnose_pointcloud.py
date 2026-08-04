#!/usr/bin/env python3
"""Diagnose point cloud generation issues for real robot data."""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
import cv2

def analyze_dataset(dataset_path):
    """Analyze real robot dataset and generate diagnostic point cloud."""
    dataset_path = Path(dataset_path)

    # Load camera config
    camera_json = dataset_path / "camera.json"
    with open(camera_json) as f:
        cam = json.load(f)

    print("=" * 60)
    print("DATASET DIAGNOSIS")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Resolution: {cam['width']} x {cam['height']}")
    print(f"Focal length: fx={cam['fx']:.2f}, fy={cam['fy']:.2f}")
    print(f"Principal point: cx={cam['cx']:.2f}, cy={cam['cy']:.2f}")
    print(f"Depth range: {cam['min_depth_m']} to {cam['max_depth_m']} m")
    print(f"World frame: {cam['world_frame']}")
    print(f"Image rectified: {cam['image_rectified']}")
    print(f"Distortion model: {cam['distortion_model']}")

    # Count frames
    rgb_files = sorted((dataset_path / "rgb").glob("*.png"))
    depth_files = sorted((dataset_path / "depth").glob("*.png"))
    pose_files = sorted((dataset_path / "pose").glob("*.txt"))

    print(f"\nFrame counts:")
    print(f"  RGB: {len(rgb_files)}")
    print(f"  Depth: {len(depth_files)}")
    print(f"  Pose: {len(pose_files)}")

    # Analyze depth quality
    print(f"\nDepth quality analysis (sampling 10 frames):")
    sample_indices = np.linspace(0, len(depth_files)-1, 10, dtype=int)
    valid_ratios = []

    for idx in sample_indices:
        depth = np.array(Image.open(depth_files[idx]))
        depth_m = depth * cam['depth_meters_per_unit']
        valid = (depth_m >= cam['min_depth_m']) & (depth_m <= cam['max_depth_m'])
        ratio = valid.sum() / valid.size
        valid_ratios.append(ratio)

    print(f"  Valid depth ratio: {np.mean(valid_ratios)*100:.1f}% ± {np.std(valid_ratios)*100:.1f}%")
    print(f"  Min/Max: {np.min(valid_ratios)*100:.1f}% / {np.max(valid_ratios)*100:.1f}%")

    # Analyze camera motion
    print(f"\nCamera motion analysis:")
    poses = [np.loadtxt(f) for f in pose_files[::10]]  # Sample every 10 frames
    translations = np.array([p[:3, 3] for p in poses])

    print(f"  Translation range:")
    print(f"    X: {translations[:, 0].min():.4f} to {translations[:, 0].max():.4f} m")
    print(f"    Y: {translations[:, 1].min():.4f} to {translations[:, 1].max():.4f} m")
    print(f"    Z: {translations[:, 2].min():.4f} to {translations[:, 2].max():.4f} m")

    total_distance = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
    print(f"  Total camera path: {total_distance:.3f} m")

    # Identify the issue: translations are TOO SMALL (cm scale not meter scale)
    max_range = np.max([
        translations[:, 0].max() - translations[:, 0].min(),
        translations[:, 1].max() - translations[:, 1].min(),
        translations[:, 2].max() - translations[:, 2].min()
    ])

    print(f"\n{'='*60}")
    print("ISSUE IDENTIFICATION:")
    print(f"{'='*60}")

    if max_range < 0.2:
        print("⚠️  PROBLEM DETECTED: Camera movement is TOO SMALL!")
        print(f"   Maximum range across all axes: {max_range:.4f} m = {max_range*100:.2f} cm")
        print("\nThis explains why the point cloud looks like a blob:")
        print("  - The robot only rotated ~10° in place")
        print("  - All frames are captured from nearly the same location")
        print("  - Point clouds overlap almost perfectly")
        print("  - No spatial separation between viewpoints")

        print("\n💡 SOLUTIONS:")
        print("  1. For scene reconstruction: Perform 90° rotation test")
        print("     or capture data while moving through the room")
        print("  2. For object-centric capture: This 10° data is OK,")
        print("     but need higher resolution and closer distance")

    # Generate a sample point cloud
    print(f"\n{'='*60}")
    print("Generating diagnostic point cloud...")
    print(f"{'='*60}")

    K = np.array([
        [cam['fx'], 0, cam['cx']],
        [0, cam['fy'], cam['cy']],
        [0, 0, 1]
    ])

    distortion = np.array(cam.get('distortion_coefficients', []))
    image_rectified = cam.get('image_rectified', False)

    full_pcd = o3d.geometry.PointCloud()

    # Sample every 10 frames
    for i, (rgb_file, depth_file, pose_file) in enumerate(
        zip(rgb_files[::10], depth_files[::10], pose_files[::10])
    ):
        rgb = np.array(Image.open(rgb_file))
        depth_raw = np.array(Image.open(depth_file))
        pose = np.loadtxt(pose_file)

        # Apply camera_optical_initial conversion
        if cam['world_frame'] == 'camera_optical_initial':
            conversion = np.array([
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            pose = conversion @ pose

        depth_m = depth_raw * cam['depth_meters_per_unit']
        valid = (depth_m >= cam['min_depth_m']) & (depth_m <= cam['max_depth_m'])

        if valid.sum() < 100:
            continue

        h, w = depth_m.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = depth_m[valid]

        # Undistort if needed
        if not image_rectified and distortion.size >= 4:
            pixels = np.column_stack((x_valid, y_valid)).astype(np.float32)
            normalized = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2),
                K,
                distortion
            ).reshape(-1, 2)
            point_x = normalized[:, 0] * z_valid
            point_y = normalized[:, 1] * z_valid
        else:
            point_x = (x_valid - K[0, 2]) * z_valid / K[0, 0]
            point_y = (y_valid - K[1, 2]) * z_valid / K[1, 1]

        points_camera = np.column_stack((point_x, point_y, z_valid))

        # Transform to world
        points_world = (pose[:3, :3] @ points_camera.T).T + pose[:3, 3]

        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(points_world)
        pcd_frame.colors = o3d.utility.Vector3dVector(rgb[valid] / 255.0)

        full_pcd += pcd_frame

        if i == 0:
            print(f"Sample frame 0:")
            print(f"  Valid pixels: {valid.sum()} / {valid.size} ({100*valid.sum()/valid.size:.1f}%)")
            print(f"  Points range in camera frame:")
            print(f"    X: {point_x.min():.3f} to {point_x.max():.3f} m")
            print(f"    Y: {point_y.min():.3f} to {point_y.max():.3f} m")
            print(f"    Z: {z_valid.min():.3f} to {z_valid.max():.3f} m")

    # Downsample
    full_pcd = full_pcd.voxel_down_sample(voxel_size=0.02)

    print(f"\nFinal point cloud:")
    print(f"  Total points: {len(full_pcd.points)}")

    if len(full_pcd.points) > 0:
        bounds_min = np.asarray(full_pcd.get_min_bound())
        bounds_max = np.asarray(full_pcd.get_max_bound())
        print(f"  Bounding box (world frame):")
        print(f"    X: {bounds_min[0]:.3f} to {bounds_max[0]:.3f} m (range: {bounds_max[0]-bounds_min[0]:.3f} m)")
        print(f"    Y: {bounds_min[1]:.3f} to {bounds_max[1]:.3f} m (range: {bounds_max[1]-bounds_min[1]:.3f} m)")
        print(f"    Z: {bounds_min[2]:.3f} to {bounds_max[2]:.3f} m (range: {bounds_max[2]-bounds_min[2]:.3f} m)")

    # Save
    output_path = dataset_path / "diagnostic_pointcloud.ply"
    o3d.io.write_point_cloud(str(output_path), full_pcd)
    print(f"\n✓ Saved diagnostic point cloud to: {output_path}")

    return full_pcd


if __name__ == "__main__":
    dataset = r"C:\Users\Lenovo\Desktop\scene_rotation_10_test_20260730_112319"
    pcd = analyze_dataset(dataset)

    print("\n" + "="*60)
    print("Opening visualization...")
    print("="*60)
    o3d.visualization.draw_geometries([pcd], window_name="Diagnostic Point Cloud")
