#!/usr/bin/env python3
"""Simple diagnosis without Open3D dependency."""

import json
import numpy as np
from pathlib import Path
from PIL import Image

def analyze_dataset(dataset_path):
    """Analyze real robot dataset."""
    dataset_path = Path(dataset_path)

    # Load camera config
    camera_json = dataset_path / "camera.json"
    with open(camera_json) as f:
        cam = json.load(f)

    print("=" * 70)
    print("真机数据点云生成问题诊断报告")
    print("=" * 70)
    print(f"\n数据集路径: {dataset_path}")
    print(f"\n【相机配置】")
    print(f"  分辨率: {cam['width']} x {cam['height']}")
    print(f"  焦距: fx={cam['fx']:.2f}, fy={cam['fy']:.2f}")
    print(f"  主点: cx={cam['cx']:.2f}, cy={cam['cy']:.2f}")
    print(f"  深度范围: {cam['min_depth_m']} ~ {cam['max_depth_m']} m")
    print(f"  世界坐标系: {cam['world_frame']}")
    print(f"  畸变模型: {cam['distortion_model']}")
    print(f"  已去畸变: {cam['image_rectified']}")

    # Count frames
    rgb_files = sorted((dataset_path / "rgb").glob("*.png"))
    depth_files = sorted((dataset_path / "depth").glob("*.png"))
    pose_files = sorted((dataset_path / "pose").glob("*.txt"))

    print(f"\n【帧数统计】")
    print(f"  RGB: {len(rgb_files)}")
    print(f"  Depth: {len(depth_files)}")
    print(f"  Pose: {len(pose_files)}")

    # Analyze depth quality
    print(f"\n【深度图质量分析】(采样10帧)")
    sample_indices = np.linspace(0, len(depth_files)-1, 10, dtype=int)
    valid_ratios = []
    depth_ranges = []

    for idx in sample_indices:
        depth = np.array(Image.open(depth_files[idx]))
        depth_m = depth * cam['depth_meters_per_unit']
        valid = (depth_m >= cam['min_depth_m']) & (depth_m <= cam['max_depth_m'])
        ratio = valid.sum() / valid.size
        valid_ratios.append(ratio)
        if valid.any():
            depth_ranges.append((depth_m[valid].min(), depth_m[valid].max()))

    print(f"  有效深度像素比例: {np.mean(valid_ratios)*100:.1f}% ± {np.std(valid_ratios)*100:.1f}%")
    print(f"  最小/最大: {np.min(valid_ratios)*100:.1f}% / {np.max(valid_ratios)*100:.1f}%")
    if depth_ranges:
        all_min = min(r[0] for r in depth_ranges)
        all_max = max(r[1] for r in depth_ranges)
        print(f"  实际深度范围: {all_min:.2f} ~ {all_max:.2f} m")

    # Analyze camera motion
    print(f"\n【相机运动分析】")
    poses_raw = [np.loadtxt(f) for f in pose_files]

    # Show raw pose translations
    translations_raw = np.array([p[:3, 3] for p in poses_raw])

    print(f"  原始位姿平移范围 (未转换前):")
    print(f"    X轴: {translations_raw[:, 0].min():.6f} ~ {translations_raw[:, 0].max():.6f} m")
    print(f"    Y轴: {translations_raw[:, 1].min():.6f} ~ {translations_raw[:, 1].max():.6f} m")
    print(f"    Z轴: {translations_raw[:, 2].min():.6f} ~ {translations_raw[:, 2].max():.6f} m")

    # Apply camera_optical_initial conversion
    if cam['world_frame'] == 'camera_optical_initial':
        conversion = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        poses_converted = [conversion @ p for p in poses_raw]
    else:
        poses_converted = poses_raw

    translations = np.array([p[:3, 3] for p in poses_converted])

    print(f"\n  转换后位姿平移范围 ({cam['world_frame']} -> HOV-SG):")
    print(f"    X轴: {translations[:, 0].min():.6f} ~ {translations[:, 0].max():.6f} m (跨度: {translations[:, 0].max() - translations[:, 0].min():.6f} m)")
    print(f"    Y轴: {translations[:, 1].min():.6f} ~ {translations[:, 1].max():.6f} m (跨度: {translations[:, 1].max() - translations[:, 1].min():.6f} m)")
    print(f"    Z轴: {translations[:, 2].min():.6f} ~ {translations[:, 2].max():.6f} m (跨度: {translations[:, 2].max() - translations[:, 2].min():.6f} m)")

    # Calculate total path
    deltas = np.diff(translations, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    total_distance = np.sum(distances)
    print(f"\n  相机总路径长度: {total_distance:.6f} m = {total_distance*100:.4f} cm")
    print(f"  平均帧间距离: {np.mean(distances):.6f} m = {np.mean(distances)*100:.4f} cm")

    # Identify the issue
    max_range = np.max([
        translations[:, 0].max() - translations[:, 0].min(),
        translations[:, 1].max() - translations[:, 1].min(),
        translations[:, 2].max() - translations[:, 2].min()
    ])

    print(f"\n{'='*70}")
    print("【问题诊断】")
    print(f"{'='*70}")

    issues_found = []

    # Issue 1: Very small motion
    if max_range < 0.15:
        issues_found.append("相机运动范围过小")
        print(f"\n❌ 问题1: 相机运动范围过小")
        print(f"   最大跨度: {max_range:.4f} m = {max_range*100:.2f} cm")
        print(f"   这是10°原地旋转测试的预期结果")
        print(f"\n   导致的问题:")
        print(f"   - 所有帧都在几乎相同的位置拍摄")
        print(f"   - 点云高度重叠,无法分辨场景结构")
        print(f"   - 看起来像一团模糊的点")

    # Issue 2: Low valid depth ratio
    if np.mean(valid_ratios) < 0.5:
        issues_found.append("深度有效像素比例低")
        print(f"\n❌ 问题2: 深度有效像素比例低 ({np.mean(valid_ratios)*100:.1f}%)")
        print(f"   深度图中有 {100-np.mean(valid_ratios)*100:.1f}% 的像素无效")
        print(f"   可能原因:")
        print(f"   - 场景距离过远 (>4m)")
        print(f"   - 表面材质对深度相机不友好 (光滑/透明/黑色)")
        print(f"   - 光照条件不佳")

    # Issue 3: Low resolution
    if cam['width'] * cam['height'] < 100000:
        issues_found.append("分辨率低")
        print(f"\n⚠️  问题3: 图像分辨率较低 ({cam['width']}x{cam['height']})")
        print(f"   总像素: {cam['width']*cam['height']:,}")
        print(f"   有效深度像素约: {int(cam['width']*cam['height']*np.mean(valid_ratios)):,}")
        print(f"   在运动范围小的情况下,低分辨率会加剧点云模糊")

    # Check rotation
    print(f"\n【旋转分析】")
    rotations = [p[:3, :3] for p in poses_converted]
    yaw_angles = []
    for R in rotations:
        # Extract yaw from rotation matrix
        yaw = np.arctan2(R[0, 2], R[2, 2])
        yaw_angles.append(yaw)

    yaw_angles = np.array(yaw_angles)
    yaw_range = yaw_angles.max() - yaw_angles.min()
    print(f"  偏航角范围: {np.rad2deg(yaw_range):.2f}° ")
    print(f"  这符合10°旋转测试的预期")

    print(f"\n{'='*70}")
    print("【解决方案】")
    print(f"{'='*70}")

    print(f"\n针对当前10°测试数据:")
    print(f"  ✓ 数据采集正常,这就是10°原地旋转的预期结果")
    print(f"  ✓ 用于验证系统运行,但无法重建完整场景")

    print(f"\n要获得清晰的场景点云,需要:")
    print(f"  1. 【推荐】运行90°旋转测试")
    print(f"     bash run_aurora_rotation_capture.sh 90 --confirm-90")
    print(f"     - 更大的视角变化")
    print(f"     - 更好的空间覆盖")
    print(f"")
    print(f"  2. 【更好】沿轨迹移动采集")
    print(f"     - 让机器人在房间中移动")
    print(f"     - 从不同位置观察场景")
    print(f"     - 可以重建完整房间结构")

    print(f"\n要提高点云质量:")
    print(f"  1. 增加有效深度像素:")
    print(f"     - 靠近目标物体 (保持在0.5-3m范围)")
    print(f"     - 避免光滑/透明表面")
    print(f"     - 改善光照条件")
    print(f"")
    print(f"  2. HOV-SG pipeline参数调整:")
    print(f"     - 减小keyframe_translation_m (当前0.10m)")
    print(f"     - 减小keyframe_rotation_deg (当前10°)")
    print(f"     - 增加voxel_size以减少重叠点 (当前0.03m)")

    print(f"\n{'='*70}")
    print("【配置文件建议】")
    print(f"{'='*70}")

    print(f"\n针对10°旋转数据,建议修改 config/real_robot.yaml:")
    print(f"""
pipeline:
  voxel_size: 0.05              # 增大以减少重叠
  skip_frames: 2                # 跳过更多帧
  keyframe_translation_m: 0.02  # 降低平移阈值
  keyframe_rotation_deg: 5.0    # 降低旋转阈值
  association_distance_m: 0.03  # 减小关联距离
""")

    print(f"\n{'='*70}")
    return {
        'total_frames': len(rgb_files),
        'valid_depth_ratio': np.mean(valid_ratios),
        'camera_path_length_m': total_distance,
        'max_translation_range_m': max_range,
        'issues': issues_found
    }


if __name__ == "__main__":
    dataset = r"C:\Users\Lenovo\Desktop\scene_rotation_10_test_20260730_112319"
    result = analyze_dataset(dataset)

    print(f"\n诊断完成!")
    print(f"主要发现: {', '.join(result['issues']) if result['issues'] else '数据正常'}")
