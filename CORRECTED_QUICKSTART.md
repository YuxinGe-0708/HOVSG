# ✅ 更正后的快速启动指南

## ⚠️ 重要更正

之前的指南有误，**正确的运行方式**如下：

---

## 🚀 正确的运行步骤

### 第1步：创建并激活conda环境

```bash
# 进入项目目录
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# 创建hovsg环境（首次运行）
conda env create -f environment.yaml

# 激活环境
conda activate hovsg

# 安装habitat-sim（单独安装）
conda install habitat-sim -c conda-forge -c aihabitat

# 安装项目包
pip install -e .
```

**环境说明**：
- Python 3.9（不是3.12）
- PyTorch + CUDA
- Open3D, Hydra, SAM等依赖

---

### 第2步：下载模型文件

**CLIP模型** (~1.7GB)：
```bash
# 创建checkpoints目录
mkdir -p checkpoints

# 下载（Linux/Mac）
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O checkpoints/laion2b_s32b_b79k.bin
```

**SAM模型** (~2.4GB)：
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

**Windows用户**（浏览器下载）：
1. CLIP: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
2. SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

保存到 `checkpoints/` 目录并重命名：
- `laion2b_s32b_b79k.bin`
- `sam_vit_h_4b8939.pth`

---

### 第3步：验证环境

```bash
# 激活环境
conda activate hovsg

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 应该输出：
# PyTorch: 2.x.x
# CUDA: True

# 检查模型文件
ls -lh checkpoints/
# 应该看到两个文件：
# laion2b_s32b_b79k.bin (~1.7GB)
# sam_vit_h_4b8939.pth (~2.4GB)
```

---

### 第4步：运行建图（✅ 正确命令）

```bash
# 激活环境
conda activate hovsg

# 进入项目目录
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# ✅ 使用配置文件运行（推荐）
python application/create_graph.py --config-name scene_manual_local

# 或者命令行覆盖参数
python application/create_graph.py \
    main.dataset=real_robot \
    main.scene_id=scene_manual_20260731_113213 \
    main.split="" \
    main.dataset_path=C:/Users/Lenovo/Desktop \
    main.save_path=C:/Users/Lenovo/Desktop/outputs_manual \
    pipeline.skip_frames=5
```

---

## 📊 处理进度

建图过程分为3个阶段：

### 阶段1：特征提取（约45-70分钟）
```
📊 FEATURE MAP CREATION - PROGRESS ESTIMATION
  Candidate frames: 797
  ⏱️  Total estimated time: 53.1 minutes
```

### 阶段2：点云保存（约2-5分钟）
```
✅ FEATURE_OUTPUTS OK path=C:/Users/Lenovo/Desktop/outputs_manual/...
```

### 阶段3：场景图构建（约5-10分钟）
```
✅ HIERARCHICAL_GRAPH OK floors=1 rooms=1 objects=XX
```

---

## 📁 输出结果

完成后，输出在：
```
C:/Users/Lenovo/Desktop/outputs_manual/real_robot/scene_manual_20260731_113213/
├── full_pcd.ply              # 全局点云（可用Open3D查看）
├── masked_pcd.ply            # 物体mask点云
├── full_feats.pt             # CLIP特征
├── mask_feats.pt             # 物体特征
└── graph/
    ├── floors/
    │   ├── 0.ply             # 楼层点云
    │   └── 0.json            # 楼层信息
    ├── rooms/
    │   ├── 0_0.ply           # 房间点云
    │   └── 0_0.json          # 房间信息
    ├── objects/              # 识别的物体
    │   ├── 0_0_0.ply
    │   ├── 0_0_0.json
    │   └── ...
    └── nav_graph/
        ├── global_nav_graph.pkl    # 全局导航图
        └── sparse_voronoi.pkl      # Voronoi图
```

---

## 🔍 验证结果

### 查看点云

```python
import open3d as o3d

# 读取全局点云
pcd = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/outputs_manual/real_robot/scene_manual_20260731_113213/full_pcd.ply')
print(f'Total points: {len(pcd.points):,}')

# 可视化
o3d.visualization.draw_geometries([pcd], window_name="Manual Trajectory Map")
```

### 检查导航图

```python
import pickle
import networkx as nx

# 读取导航图
nav_path = 'C:/Users/Lenovo/Desktop/outputs_manual/real_robot/scene_manual_20260731_113213/graph/nav_graph/global_nav_graph.pkl'
with open(nav_path, 'rb') as f:
    nav = pickle.load(f)

print(f'Navigation nodes: {nav.number_of_nodes()}')
print(f'Navigation edges: {nav.number_of_edges()}')
print(f'Connected: {nx.is_connected(nav)}')
```

---

## 🐛 故障排除

### 问题1: `No module named 'torch'`

**原因**：未激活conda环境

**解决**：
```bash
conda activate hovsg
```

### 问题2: `FileNotFoundError: checkpoints/...`

**原因**：模型文件未下载或路径错误

**解决**：
```bash
# 检查文件
ls -lh checkpoints/

# 应该看到两个模型文件（总计约4GB）
```

### 问题3: `CUDA out of memory`

**原因**：RTX 4060 Laptop VRAM不足（8GB）

**解决**：增大跳帧数
```bash
python application/create_graph.py --config-name scene_manual_local \
    pipeline.skip_frames=7  # 从5改为7，减少处理帧数
```

### 问题4: 进程卡住不动

**检查**：
1. GPU是否可用：`nvidia-smi`
2. 内存是否充足：任务管理器查看RAM使用
3. 是否在下载模型：首次运行CLIP/SAM会自动下载权重

---

## ⏱️ 时间估算

| 跳帧设置 | 处理帧数 | 预计时间 |
|---------|---------|---------|
| skip_frames=3 | ~1327帧 | 75-90分钟 |
| skip_frames=5 | ~797帧 | 45-70分钟 ⭐ 推荐 |
| skip_frames=7 | ~569帧 | 35-50分钟 |
| skip_frames=10 | ~398帧 | 25-35分钟 |

---

## 📝 错误更正说明

### ❌ 之前的错误命令
```bash
python -m hovsg.main --config config/scene_manual_local.yaml
```

**问题**：
1. HOV-SG不使用 `hovsg.main` 模块
2. 项目使用Hydra配置系统，入口是 `application/create_graph.py`

### ✅ 正确的命令
```bash
python application/create_graph.py --config-name scene_manual_local
```

---

## 💡 命令对比

| 错误命令 | 正确命令 |
|---------|---------|
| `python -m hovsg.main` | `python application/create_graph.py` |
| `--config config/file.yaml` | `--config-name file` (无需`.yaml`后缀) |
| 使用系统Python | 使用conda环境：`conda activate hovsg` |

---

## ✅ 检查清单

运行前请确认：

- [ ] conda环境已创建：`conda env create -f environment.yaml`
- [ ] conda环境已激活：`conda activate hovsg`
- [ ] PyTorch已安装且CUDA可用
- [ ] CLIP模型已下载（~1.7GB）
- [ ] SAM模型已下载（~2.4GB）
- [ ] 数据目录存在且包含3982帧
- [ ] GPU可用（nvidia-smi检查）
- [ ] 配置文件路径正确

---

## 🎯 一键启动（完整命令）

```bash
# 1. 激活环境
conda activate hovsg

# 2. 进入项目目录
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# 3. 运行建图
python application/create_graph.py --config-name scene_manual_local

# 4. 等待完成（45-70分钟）
```

---

**准备好了？直接运行上面的命令即可！**
