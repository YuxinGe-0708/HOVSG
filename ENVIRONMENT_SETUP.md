# 环境配置指南

## ❌ 问题：运行命令错误

您遇到的错误：
```
D:\Programs\Python\Python312\python.exe: No module named hovsg.main
```

**原因**：
1. ✅ HOV-SG项目使用的是 `application/create_graph.py`，不是 `hovsg.main`
2. ✅ 需要先激活conda虚拟环境（hovsg），当前使用的是系统Python 3.12

---

## 🔧 正确的运行步骤

### 步骤1：创建并激活conda环境

```bash
# 1. 创建hovsg环境
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev
conda env create -f environment.yaml

# 2. 激活环境
conda activate hovsg

# 3. 安装habitat-sim（单独安装以避免错误）
conda install habitat-sim -c conda-forge -c aihabitat

# 4. 安装项目包
pip install -e .
```

### 步骤2：下载模型checkpoint

```bash
# 创建checkpoints目录
mkdir -p checkpoints

# 下载CLIP模型（约1.7GB）
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O checkpoints/laion2b_s32b_b79k.bin

# 下载SAM模型（约2.4GB）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

**Windows用户**：如果没有wget，可以用浏览器下载：
- CLIP: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
- SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

保存到 `checkpoints/` 目录，并重命名为：
- `laion2b_s32b_b79k.bin`
- `sam_vit_h_4b8939.pth`

### 步骤3：验证环境

```bash
# 激活环境后验证
conda activate hovsg

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查其他依赖
python -c "import open3d; import hydra; import omegaconf; print('Dependencies OK')"

# 检查模型文件
ls -lh checkpoints/
```

### 步骤4：运行建图（正确命令）

```bash
# 激活环境
conda activate hovsg

# 进入项目目录
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# ✅ 正确的运行命令（使用Hydra配置系统）
python application/create_graph.py \
    main.dataset=real_robot \
    main.scene_id=scene_manual_20260731_113213 \
    main.split="" \
    main.dataset_path=C:/Users/Lenovo/Desktop \
    main.save_path=C:/Users/Lenovo/Desktop/outputs_manual \
    main.package_path=C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev/hovsg \
    pipeline.extract_semantics=true \
    pipeline.create_graph=true \
    pipeline.create_navigation=true \
    pipeline.skip_frames=5 \
    pipeline.voxel_size=0.04 \
    pipeline.force_single_floor=true \
    pipeline.min_floor_height=0.5
```

**或者使用配置文件**：

由于参数太多，更简单的方法是使用配置文件：

```bash
conda activate hovsg
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# 使用我们创建的配置文件
python application/create_graph.py --config-name scene_manual_local
```

---

## 📝 关于Hydra配置系统

HOV-SG使用 [Hydra](https://hydra.cc/) 进行配置管理。有两种使用方式：

### 方法1：命令行覆盖（推荐用于测试）

```bash
python application/create_graph.py main.dataset=real_robot main.scene_id=your_scene
```

### 方法2：配置文件（推荐用于生产）

```bash
# Hydra会在 config/ 目录查找配置文件
python application/create_graph.py --config-name scene_manual_local
```

**注意**：我们已经为您创建了 `config/scene_manual_local.yaml`，但**需要修改一个地方**：

原配置文件使用相对路径，Hydra可能无法正确解析。我已经更新了配置文件使用绝对路径。

---

## 🔍 环境检查清单

运行前请确认：

- [ ] conda环境已创建并激活（`conda activate hovsg`）
- [ ] PyTorch已安装且CUDA可用
- [ ] Open3D、Hydra等依赖已安装
- [ ] CLIP模型已下载（checkpoints/laion2b_s32b_b79k.bin，约1.7GB）
- [ ] SAM模型已下载（checkpoints/sam_vit_h_4b8939.pth，约2.4GB）
- [ ] 数据目录存在（C:/Users/Lenovo/Desktop/scene_manual_20260731_113213）
- [ ] GPU可用（nvidia-smi显示RTX 4060）

---

## ⚡ 快速启动（一键运行）

```bash
# 1. 激活环境
conda activate hovsg

# 2. 进入项目目录
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# 3. 运行建图（使用配置文件）
python application/create_graph.py --config-name scene_manual_local
```

**预计时间**：45-70分钟（3982帧，skip_frames=5 → 约797帧）

---

## 📂 输出文件说明

建图完成后，输出在：`C:/Users/Lenovo/Desktop/outputs_manual/real_robot/scene_manual_20260731_113213/`

```
outputs_manual/real_robot/scene_manual_20260731_113213/
├── full_pcd.ply              # 全局点云
├── masked_pcd.ply            # 物体mask点云
├── full_feats.pt             # CLIP特征
├── mask_feats.pt             # 物体特征
└── graph/
    ├── floors/               # 楼层分割
    │   └── 0.ply
    ├── rooms/                # 房间分割
    │   ├── 0_0.ply
    │   └── 0_0.json
    ├── objects/              # 物体识别
    │   ├── 0_0_0.ply
    │   └── 0_0_0.json
    └── nav_graph/            # 导航图
        ├── global_nav_graph.pkl
        └── sparse_voronoi.pkl
```

---

## 🐛 常见问题

### Q1: `ModuleNotFoundError: No module named 'torch'`

**原因**：没有激活conda环境

**解决**：
```bash
conda activate hovsg
python -c "import torch; print(torch.__version__)"
```

### Q2: `FileNotFoundError: checkpoints/laion2b_s32b_b79k.bin`

**原因**：模型文件未下载或路径错误

**解决**：
```bash
# 检查文件是否存在
ls -lh checkpoints/

# 如果不存在，下载模型（见步骤2）
```

### Q3: `CUDA out of memory`

**原因**：RTX 4060 Laptop (8GB VRAM) 可能内存不足

**解决**：
```bash
# 增大跳帧数，减少处理帧数
python application/create_graph.py --config-name scene_manual_local \
    pipeline.skip_frames=7  # 从5改为7
```

### Q4: Hydra配置覆盖不生效

**检查**：
```bash
# 查看Hydra实际使用的配置
python application/create_graph.py --config-name scene_manual_local --cfg job
```

---

## 💡 下一步

环境配置完成后：

1. ✅ **运行建图**：使用上面的快速启动命令
2. ✅ **监控进度**：观察终端输出的进度信息
3. ✅ **验证结果**：建图完成后使用Open3D查看点云
4. ✅ **检查导航图**：验证导航图的连通性

---

**需要帮助？**

如果遇到其他问题：
1. 检查 `environment.yaml` 中的依赖版本
2. 查看项目README中的安装说明
3. 提供完整的错误日志以便诊断
