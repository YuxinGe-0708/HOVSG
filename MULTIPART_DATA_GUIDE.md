# 多段采集数据配准和合并完整指南

## 📖 概述

本指南用于将多段独立采集的机器人RGB-D数据配准并合并为单个完整数据集，以解决RTAB-Map在转弯处位姿丢失的问题。

---

## 🎯 核心思路

### 问题
- RTAB-Map纯视觉里程计在转弯处容易失效
- 每次重启RTAB-Map，`map`坐标系会重新初始化
- 直接合并多段数据会导致点云错位或分离

### 解决方案
1. **分段采集**：将完整轨迹分成多个短而稳定的子场景
2. **独立重建**：每段生成独立的子点云
3. **点云配准**：利用重叠区域对齐点云，获得变换矩阵
4. **位姿转换**：用变换矩阵统一所有相机位姿到全局坐标系
5. **数据合并**：合并RGB、Depth和转换后的Pose

---

## 📂 数据采集要求

### 目录结构

```
room_capture/
├── part_01/                    # 第1段
│   ├── camera.json
│   ├── rgb/
│   │   ├── 000000.png
│   │   └── ...
│   ├── depth/
│   │   ├── 000000.png
│   │   └── ...
│   └── pose/
│       ├── 000000.txt
│       └── ...
├── part_02/                    # 第2段
│   ├── camera.json
│   ├── rgb/
│   ├── depth/
│   └── pose/
└── part_03/                    # 第3段
    ├── camera.json
    ├── rgb/
    ├── depth/
    └── pose/
```

---

### 采集注意事项

#### ✅ 必须保证的重叠区域

每两段之间必须有**30%～50%的重叠区域**，且包含：
- ✅ 门框、桌角、柜子、墙角等**明显几何结构**
- ✅ 尽量相近的**观察方向**
- ❌ **不要**只让重叠部分包含白墙或平整地面

#### ✅ 采集建议

**每段开始**：
- 静止3～5秒，让RTAB-Map初始化
- 选择纹理丰富的区域

**采集过程**：
- 保持稳定速度（0.2-0.5 m/s）
- 避免快速转弯
- 尽量覆盖房间主要区域

**每段结束**：
- 停在纹理丰富的区域
- 与下一段的起点有明显重叠

#### ✅ 采集示例

```
第1段：门口 → 房间左侧 → 桌子（停在桌子附近）
         └─────── 重叠区域：桌子 ───────┐
                                          ▼
第2段：   从桌子附近开始 → 房间右侧 → 柜子（停在柜子附近）
                      └─────── 重叠区域：柜子 ───────┐
                                                      ▼
第3段：            从柜子附近开始 → 房间后侧 → 门口
```

#### ⚠️ 失效数据处理

每段中，**只保留`Registration failed`之前的数据**：

1. **第一次连续出现`quality=0`后的数据不要使用**
2. **向前裁掉约1～2秒**，防止保留视觉里程计刚开始发散的位姿
3. **确保RGB、Depth、Pose三个目录文件数量一致**

验证命令：
```bash
SCENE=/path/to/part_01

echo "RGB:   $(find "$SCENE/rgb" -name '*.png' | wc -l)"
echo "Depth: $(find "$SCENE/depth" -name '*.png' | wc -l)"
echo "Pose:  $(find "$SCENE/pose" -name '*.txt' | wc -l)"
```

---

## 🔧 工具说明

### 工具1: align_pointclouds.py
**功能**：从多段RGB-D数据生成点云并配准

**输入**：
- 多段独立采集的RGB-D-Pose数据

**输出**：
- 变换矩阵JSON文件（记录每段到全局坐标系的变换）

---

### 工具2: transform_poses.py
**功能**：将配准矩阵应用到每段的相机位姿

**输入**：
- 原始多段数据
- 变换矩阵JSON文件

**输出**：
- 转换后的多段数据（位姿已统一到全局坐标系）

---

### 工具3: merge_multipart_data.py
**功能**：合并转换后的多段数据为单个数据集

**输入**：
- 转换后的多段数据

**输出**：
- 单个完整数据集（重新编号0, 1, 2, ...）

---

## 🚀 完整流程

### 步骤1: 点云配准

```bash
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev-multi-part

python tools/align_pointclouds.py \
    --parts-dir room_capture \
    --part-names part_01 part_02 part_03 \
    --output transformations.json \
    --voxel-size 0.05 \
    --skip-frames 5 \
    --visualize
```

**参数说明**：
- `--parts-dir`: 包含多段数据的根目录
- `--part-names`: 各段的目录名（按采集顺序）
- `--output`: 输出的变换矩阵文件
- `--voxel-size`: 体素降采样大小（米），默认0.05
- `--skip-frames`: 跳帧数（加速），默认5
- `--max-frames`: 每段最大帧数（测试用），可选
- `--visualize`: 可视化配准结果

**预期输出**：
```
Loading part_01...
  Processing 797 frames with skip=5
  Generated point cloud with 145230 points
  After voxel downsampling: 28456 points

Loading part_02...
  Generated point cloud with 138921 points
  After voxel downsampling: 27184 points

Loading part_03...
  Generated point cloud with 152387 points
  After voxel downsampling: 29877 points

=======================================================================
Aligning 3 point cloud segments
=======================================================================

part_01 set as global coordinate system

--- Aligning part_02 to part_01 ---
  Executing RANSAC global registration...
    RANSAC fitness: 0.7823
    RANSAC RMSE: 0.0234
  Refining with Point-to-Plane ICP...
    ICP fitness: 0.8912
    ICP RMSE: 0.0187

✓ part_02 aligned:
    Fitness: 0.8912
    RMSE: 0.0187

--- Aligning part_03 to part_02 ---
  RANSAC fitness: 0.8156
  ICP fitness: 0.9034
  ICP RMSE: 0.0165

✓ part_03 aligned:
    Fitness: 0.9034
    RMSE: 0.0165

Transformations saved to: transformations.json
```

**重要指标**：
- **Fitness > 0.8**：配准质量良好
- **RMSE < 0.03m**：误差在3cm以内，可接受
- **Fitness < 0.5 或 RMSE > 0.05m**：配准失败，检查重叠区域

---

### 步骤2: 转换位姿

```bash
python tools/transform_poses.py \
    --parts-dir room_capture \
    --transformations transformations.json \
    --output-dir room_capture_transformed \
    --verbose
```

**参数说明**：
- `--parts-dir`: 原始多段数据目录
- `--transformations`: 步骤1生成的变换矩阵文件
- `--output-dir`: 输出的转换后数据目录
- `--verbose`: 显示详细进度

**预期输出**：
```
Loading transformations from transformations.json
Loaded 3 transformations

Verifying transformations...
  ✓ part_01 det(R) = 1.0000
  ✓ part_01 orthogonality error = 0.000000
  ✓ part_02 det(R) = 0.9998
  ✓ part_02 orthogonality error = 0.000124
  ✓ part_03 det(R) = 1.0001
  ✓ part_03 orthogonality error = 0.000089

=======================================================================
Transforming poses for each part
=======================================================================

Processing part_01...
  Transforming 797 poses...
  ✓ Saved to room_capture_transformed/part_01/pose
  Copying RGB images...
    ✓ Copied 797 RGB images
  Copying depth images...
    ✓ Copied 797 depth images
  ✓ Copied camera.json
  ✓ part_01 completed

Processing part_02...
  Transforming 823 poses...
  ✓ part_02 completed

Processing part_03...
  Transforming 891 poses...
  ✓ part_03 completed

=======================================================================
All poses transformed successfully!
Output directory: room_capture_transformed
=======================================================================
```

---

### 步骤3: 合并数据

```bash
python tools/merge_multipart_data.py \
    --transformed-dir room_capture_transformed \
    --part-names part_01 part_02 part_03 \
    --output-dir room_capture_merged
```

**参数说明**：
- `--transformed-dir`: 步骤2生成的转换后数据目录
- `--part-names`: 要合并的数据段（按顺序）
- `--output-dir`: 合并后的完整数据集目录

**预期输出**：
```
=======================================================================
Multi-Part Data Merger
=======================================================================

Merging part_01...
  Processing 797 frames...
    Copied 600/797 frames
  ✓ Added 797 frames (total: 797)

Merging part_02...
  Processing 823 frames...
  ✓ Added 823 frames (total: 1620)

Merging part_03...
  Processing 891 frames...
  ✓ Added 891 frames (total: 2511)

✓ Copied camera.json from part_01

=======================================================================
Verifying merged data
=======================================================================
File counts:
  RGB:   2511
  Depth: 2511
  Pose:  2511

✓ All file counts match: 2511 frames
✓ Frame indices are continuous (0 to 2510)
✓ Sampled 5 pose files - format OK
✓ camera.json is valid

=======================================================================
✓ All verification checks passed!
=======================================================================

✓ Merge report saved to: room_capture_merged/merge_report.txt

=======================================================================
✓ Data merge completed successfully!
=======================================================================

Merged dataset location: room_capture_merged
Total frames: 2511

You can now run HOV-SG mapping on this merged dataset.
```

---

### 步骤4: 运行HOV-SG建图

```bash
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev-multi-part

python application/create_graph.py \
    main.dataset=real_robot \
    main.scene_id=room_capture_merged \
    main.split="" \
    main.dataset_path=C:/path/to \
    main.save_path=C:/path/to/outputs \
    main.package_path=C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev-multi-part/hovsg \
    pipeline.skip_frames=5 \
    pipeline.extract_semantics=true \
    pipeline.create_graph=true \
    pipeline.create_navigation=true
```

---

## 📊 配准质量评估

### 好的配准结果
```
Fitness: 0.85-0.95
RMSE: 0.01-0.03m
```
- 重叠区域充足
- 几何特征明显
- 配准成功

### 一般的配准结果
```
Fitness: 0.70-0.85
RMSE: 0.03-0.05m
```
- 重叠区域一般
- 可以使用，但建议检查配准结果
- 可能需要手动微调

### 失败的配准结果
```
Fitness: <0.70
RMSE: >0.05m
```
- 重叠区域不足或特征不明显
- 需要重新采集数据
- 增加重叠区域或选择更好的特征区域

---

## 🔍 故障排除

### 问题1: 配准失败（Fitness < 0.5）

**原因**：
- 重叠区域不足（<30%）
- 重叠区域只有白墙或地面
- 观察角度差异太大

**解决**：
- 重新采集，增加重叠区域到40-50%
- 在重叠区域包含桌子、柜子等明显特征
- 保持相近的观察方向

---

### 问题2: 配准RMSE过大（>0.05m）

**原因**：
- 点云噪声大
- 深度数据质量差

**解决**：
```bash
# 增大voxel_size进行更强的降采样
python tools/align_pointclouds.py \
    --voxel-size 0.08 \  # 从0.05增加到0.08
    ...
```

---

### 问题3: 合并后点云有明显错位

**检查**：
```bash
# 1. 验证变换矩阵
python tools/transform_poses.py --transformations transformations.json

# 应该看到:
# ✓ det(R) ≈ 1.0
# ✓ orthogonality error < 0.001
```

**解决**：
- 如果det(R)偏离1.0较多，说明配准质量差
- 重新运行配准或调整参数

---

### 问题4: 内存不足

**解决**：
```bash
# 方法1: 增大skip_frames
python tools/align_pointclouds.py \
    --skip-frames 10 \  # 从5增加到10
    ...

# 方法2: 限制max_frames
python tools/align_pointclouds.py \
    --max-frames 500 \  # 每段只用前500帧
    ...
```

---

## 💡 最佳实践

### 采集建议

1. **规划采集路径**
   - 提前规划好每段的起点和终点
   - 确保重叠区域有明显特征

2. **控制采集速度**
   - 保持0.2-0.5 m/s的稳定速度
   - 避免快速转弯

3. **及时检查数据**
   - 每段采集后立即检查`quality`
   - 发现问题立即重新采集

4. **预留冗余**
   - 每段多采集一些，后期可以裁剪
   - 重叠区域宁可多不可少

---

### 处理建议

1. **先测试后完整**
   ```bash
   # 先用少量帧测试配准
   python tools/align_pointclouds.py \
       --max-frames 200 \
       --skip-frames 10 \
       --visualize
   
   # 确认配准质量后再处理完整数据
   ```

2. **逐步检查**
   - 每个步骤完成后都要验证结果
   - 使用`--visualize`查看点云配准效果

3. **保留中间结果**
   - 保留原始数据
   - 保留变换矩阵文件
   - 保留转换后数据

---

## 📖 参考资料

- **Open3D配准教程**: http://www.open3d.org/docs/latest/tutorial/pipelines/registration.html
- **ICP算法原理**: https://en.wikipedia.org/wiki/Iterative_closest_point
- **FPFH特征**: Fast Point Feature Histograms

---

**多段数据配准和合并工具已准备就绪！开始采集您的数据吧！**
