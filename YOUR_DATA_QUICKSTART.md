# 您的数据建图快速指南

## 📊 您的数据信息

✅ **数据已确认**：
- **路径**：`C:\Users\Lenovo\Desktop\scene_manual_20260731_113213`
- **帧数**：3982帧（RGB + Depth + Pose）
- **相机**：Aurora 930 (320×200)
- **坐标系**：camera_optical_initial
- **数据状态**：完整且一致 ✅

---

## 🚀 最简单方案：本地运行（推荐）

### 一键运行（3步）

```bash
# 1. 进入项目目录
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# 2. 激活conda环境（重要！）
conda activate hovsg

# 3. 运行建图（正确命令）
python application/create_graph.py --config-name scene_manual_local

# 4. 等待完成（预计45-70分钟）
```

**重要**：
- ⚠️ 必须先激活conda环境：`conda activate hovsg`
- ⚠️ 正确命令是 `python application/create_graph.py`（不是 `python -m hovsg.main`）
- ✅ 如果环境未创建，参考 `ENVIRONMENT_SETUP.md` 或 `CORRECTED_QUICKSTART.md`

---

## 📋 配置说明

已为您创建 `config/scene_manual_local.yaml`，关键配置：

```yaml
# 数据路径（已配置好）
dataset_path: C:/Users/Lenovo/Desktop
scene_id: scene_manual_20260731_113213
split: ""  # 留空，直接读取

# 输出路径
save_path: C:/Users/Lenovo/Desktop/outputs_manual

# 优化参数（针对3982帧）
skip_frames: 5  # 跳帧处理，约797帧候选
voxel_size: 0.04
keyframe_translation_m: 0.05
keyframe_rotation_deg: 5.0

# 所有建图功能已开启
extract_semantics: true
create_graph: true
create_navigation: true
```

---

## ⏱️ 处理时间估算

| 项目 | 数值 |
|------|------|
| 原始帧数 | 3982帧 |
| 跳帧设置 | skip_frames=5 |
| 候选帧数 | 约797帧 |
| 每帧处理时间 | 约4秒 |
| **估算总时间** | **45-70分钟** |

---

## 📁 输出文件位置

建图完成后，输出将保存在：

```
C:/Users/Lenovo/Desktop/outputs_manual/
├── full_pcd.ply              # 全局点云
├── masked_pcd.ply            # 物体mask点云
├── full_feats.pt             # 点云CLIP特征
├── mask_feats.pt             # 物体特征
└── graph/
    ├── floors/
    │   └── 0.ply             # 楼层点云
    ├── rooms/
    │   ├── 0_0.ply           # 房间点云
    │   └── ...
    ├── objects/
    │   ├── 0_0_0.ply         # 物体点云
    │   └── ...
    └── nav_graph/
        ├── global_nav_graph.pkl  # 导航图
        └── sparse_voronoi.pkl    # Voronoi图
```

---

## 🔍 验证结果

建图完成后，运行以下命令验证：

```python
# 查看点云
import open3d as o3d

pcd = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/outputs_manual/full_pcd.ply')
print(f'Total points: {len(pcd.points):,}')
o3d.visualization.draw_geometries([pcd], window_name="Manual Trajectory Map")

# 查看导航图
import pickle
import networkx as nx

with open('C:/Users/Lenovo/Desktop/outputs_manual/graph/nav_graph/global_nav_graph.pkl', 'rb') as f:
    nav = pickle.load(f)

print(f'Navigation nodes: {nav.number_of_nodes()}')
print(f'Navigation edges: {nav.number_of_edges()}')
print(f'Connected: {nx.is_connected(nav)}')
```

---

## 🔧 如果需要上传到服务器

### 方法1：使用SCP（命令行）

```bash
# 上传数据到服务器
scp -r "C:/Users/Lenovo/Desktop/scene_manual_20260731_113213" \
    username@server_ip:/mnt/data/real_camera/robot_recordings/test/scene_manual

# 或者压缩后上传（更快）
cd C:/Users/Lenovo/Desktop
tar -czf scene_manual.tar.gz scene_manual_20260731_113213
scp scene_manual.tar.gz username@server_ip:/mnt/data/real_camera/robot_recordings/test/

# 在服务器上解压
ssh username@server_ip
cd /mnt/data/real_camera/robot_recordings/test/
tar -xzf scene_manual.tar.gz
mv scene_manual_20260731_113213 scene_manual
```

### 方法2：使用Rsync（支持断点续传）

```bash
# 如果数据较大，使用rsync更可靠
rsync -avz --progress \
    "C:/Users/Lenovo/Desktop/scene_manual_20260731_113213/" \
    username@server_ip:/mnt/data/real_camera/robot_recordings/test/scene_manual/
```

### 服务器配置文件

如果在服务器运行，修改配置：

```yaml
main:
  device: cuda
  dataset: real_robot
  scene_id: scene_manual  # 服务器上的目录名
  split: test

  package_path: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/hovsg
  dataset_path: /mnt/data/real_camera/robot_recordings
  save_path: /mnt/data/real_camera/outputs/scene_manual

models:
  clip:
    checkpoint: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/checkpoints/laion2b_s32b_b79k.bin
  sam:
    checkpoint: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/checkpoints/sam_vit_h_4b8939.pth

# 其他参数同 scene_manual_local.yaml
```

### 服务器运行命令

```bash
# SSH登录服务器
ssh username@server_ip

# 进入项目目录
cd /mnt/data/real_camera/isaac/HOV-SG-graph-dev

# 后台运行建图
nohup python -m hovsg.main --config config/scene_manual_server.yaml > build_manual.log 2>&1 &

# 查看进度
tail -f build_manual.log

# 或者使用screen（推荐）
screen -S hovsg_build
python -m hovsg.main --config config/scene_manual_server.yaml
# 按 Ctrl+A, D 退出screen
# 重新连接：screen -r hovsg_build
```

---

## ⚠️ 注意事项

### 运行前检查

- [ ] 确保有可用的CUDA GPU
- [ ] 确保模型checkpoint已下载（CLIP + SAM）
- [ ] 确保有足够磁盘空间（至少20GB）
- [ ] 如果内存不足，可以增大 `skip_frames` 到 7 或 10

### 常见问题

**Q1: 提示找不到模型checkpoint**
```bash
# 需要下载模型文件到 checkpoints/ 目录
# CLIP: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
# SAM: https://github.com/facebookresearch/segment-anything
```

**Q2: 内存不足（OOM）**
```yaml
# 修改 config/scene_manual_local.yaml
skip_frames: 7  # 从5改为7
voxel_size: 0.05  # 从0.04改为0.05
```

**Q3: GPU不可用**
```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📞 需要帮助？

- **配置问题**：查看 `DATA_DEPLOYMENT_GUIDE.md`
- **技术细节**：查看 `REAL_ROBOT_MODIFICATIONS.md`
- **快速开始**：查看 `QUICK_START_GUIDE.md`
- **验证修改**：运行 `python verify_modifications.py`

---

## 🎯 推荐方案总结

**对于您的情况**（3982帧本地数据）：

✅ **推荐：本地运行**
```bash
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev
python -m hovsg.main --config config/scene_manual_local.yaml
```

**原因**：
- 数据已在本地，无需传输
- 可以实时查看进度
- 配置已准备好，直接运行

**如果本地GPU不够**：使用服务器，参考上面的"上传到服务器"部分

---

**准备好了吗？直接运行上面的命令即可开始建图！**
