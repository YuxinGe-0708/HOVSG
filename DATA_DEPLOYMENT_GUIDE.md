# 真机数据部署指南

## 📂 您的数据情况

**本地数据路径**：`C:\Users\Lenovo\Desktop\scene_manual_20260731_113213`

**数据结构**：
```
scene_manual_20260731_113213/
├── camera.json       ✅ 已检查，配置正确
├── rgb/              (PNG files)
├── depth/            (PNG files)
└── pose/             (TXT files)
```

---

## 🎯 两种部署方案

### 方案A：在本地Windows运行（推荐，最简单）

**优点**：
- ✅ 无需传输数据到服务器
- ✅ 可以实时查看进度
- ✅ 配置简单

**步骤**：

#### 1. 创建本地配置文件

```bash
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev
cp config/real_robot_trajectory.yaml config/scene_manual.yaml
```

#### 2. 修改配置文件

用文本编辑器打开 `config/scene_manual.yaml`，修改以下部分：

```yaml
main:
  device: cuda  # 如果本地有GPU
  dataset: real_robot
  scene_id: scene_manual_20260731_113213  # 数据文件夹名
  split: ""  # 留空，因为数据直接在桌面

  # 本地路径（Windows路径）
  package_path: C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev/hovsg
  
  # 数据集路径：桌面目录
  dataset_path: C:/Users/Lenovo/Desktop
  
  # 输出路径：保存到桌面outputs文件夹
  save_path: C:/Users/Lenovo/Desktop/outputs

models:
  clip:
    type: ViT-H-14
    # 修改为实际的checkpoint路径
    checkpoint: C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev/checkpoints/laion2b_s32b_b79k.bin

  sam:
    type: vit_h
    # 修改为实际的checkpoint路径
    checkpoint: C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev/checkpoints/sam_vit_h_4b8939.pth

pipeline:
  extract_semantics: true
  create_graph: true
  create_navigation: true
  
  # 其他优化参数保持不变...
```

#### 3. 运行建图

```bash
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev

# 运行建图
python -m hovsg.main --config config/scene_manual.yaml
```

**注意**：
- 确保本地有CUDA GPU（建图需要GPU）
- 确保已安装所有依赖（PyTorch, Open3D, CLIP, SAM等）
- 确保模型checkpoint文件已下载

---

### 方案B：上传到服务器运行

**优点**：
- ✅ 服务器GPU性能更强
- ✅ 不占用本地资源
- ✅ 可以后台运行

**服务器目录结构要求**：

根据配置文件，HOV-SG期望的目录结构是：
```
dataset_path/split/scene_id/
```

例如，如果配置是：
```yaml
dataset_path: /mnt/data/real_camera/robot_recordings
split: test
scene_id: scene_manual_20260731
```

则实际读取路径是：
```
/mnt/data/real_camera/robot_recordings/test/scene_manual_20260731/
├── camera.json
├── rgb/
├── depth/
└── pose/
```

#### 步骤1：确定服务器路径

**标准方案**（推荐）：
```bash
# 服务器目录
/mnt/data/real_camera/robot_recordings/
├── test/                    # split目录
│   └── scene_manual_20260731/  # scene_id目录
│       ├── camera.json
│       ├── rgb/
│       ├── depth/
│       └── pose/
```

**简化方案**（如果不想要split层级）：
```bash
# 服务器目录
/mnt/data/real_camera/robot_recordings/
└── scene_manual_20260731/  # 直接放scene_id目录
    ├── camera.json
    ├── rgb/
    ├── depth/
    └── pose/
```

#### 步骤2：传输数据到服务器

**方法1：使用SCP（推荐）**

```bash
# 在本地Windows PowerShell或Git Bash中运行

# 标准方案（有split层级）
scp -r "C:/Users/Lenovo/Desktop/scene_manual_20260731_113213" \
    username@server_ip:/mnt/data/real_camera/robot_recordings/test/scene_manual_20260731

# 简化方案（无split层级）
scp -r "C:/Users/Lenovo/Desktop/scene_manual_20260731_113213" \
    username@server_ip:/mnt/data/real_camera/robot_recordings/scene_manual_20260731
```

**方法2：使用WinSCP（图形界面）**

1. 下载并安装 WinSCP
2. 连接到服务器
3. 导航到 `/mnt/data/real_camera/robot_recordings/test/`
4. 拖拽上传 `scene_manual_20260731_113213` 文件夹
5. 重命名为 `scene_manual_20260731`

**方法3：使用rsync（更快，支持断点续传）**

```bash
# 需要安装WSL或使用Git Bash
rsync -avz --progress \
    "/c/Users/Lenovo/Desktop/scene_manual_20260731_113213/" \
    username@server_ip:/mnt/data/real_camera/robot_recordings/test/scene_manual_20260731/
```

#### 步骤3：创建服务器配置文件

登录服务器，创建配置文件：

```bash
# SSH登录服务器
ssh username@server_ip

# 进入项目目录
cd /mnt/data/real_camera/isaac/HOV-SG-graph-dev

# 创建配置文件
nano config/scene_manual.yaml
```

**标准方案配置**（有split层级）：
```yaml
main:
  device: cuda
  dataset: real_robot
  scene_id: scene_manual_20260731  # 场景名（不含日期时间戳后缀）
  split: test  # split目录名

  package_path: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/hovsg
  dataset_path: /mnt/data/real_camera/robot_recordings
  save_path: /mnt/data/real_camera/outputs/scene_manual_20260731

models:
  clip:
    type: ViT-H-14
    checkpoint: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/checkpoints/laion2b_s32b_b79k.bin
  sam:
    type: vit_h
    checkpoint: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/checkpoints/sam_vit_h_4b8939.pth

pipeline:
  extract_semantics: true
  create_graph: true
  create_navigation: true
  
  voxel_size: 0.04
  skip_frames: 3
  keyframe_translation_m: 0.05
  keyframe_rotation_deg: 5.0
  
  # ... 其他参数参考 real_robot_trajectory.yaml
```

**简化方案配置**（无split层级）：
```yaml
main:
  device: cuda
  dataset: real_robot
  scene_id: scene_manual_20260731
  split: ""  # 留空

  package_path: /mnt/data/real_camera/isaac/HOV-SG-graph-dev/hovsg
  dataset_path: /mnt/data/real_camera/robot_recordings  # 直接指向父目录
  save_path: /mnt/data/real_camera/outputs/scene_manual_20260731

  # ... 其余配置同上
```

#### 步骤4：验证数据路径

```bash
# 验证数据存在
# 标准方案
ls -la /mnt/data/real_camera/robot_recordings/test/scene_manual_20260731/
ls /mnt/data/real_camera/robot_recordings/test/scene_manual_20260731/rgb/ | wc -l
ls /mnt/data/real_camera/robot_recordings/test/scene_manual_20260731/depth/ | wc -l
ls /mnt/data/real_camera/robot_recordings/test/scene_manual_20260731/pose/ | wc -l

# 简化方案
ls -la /mnt/data/real_camera/robot_recordings/scene_manual_20260731/
```

#### 步骤5：运行建图

```bash
# 进入项目目录
cd /mnt/data/real_camera/isaac/HOV-SG-graph-dev

# 激活虚拟环境（如果有）
source venv/bin/activate

# 运行建图
python -m hovsg.main --config config/scene_manual.yaml

# 或者后台运行（推荐）
nohup python -m hovsg.main --config config/scene_manual.yaml > build_map.log 2>&1 &

# 查看日志
tail -f build_map.log
```

---

## 🔍 数据路径解析逻辑

HOV-SG的数据加载逻辑（在 `hovsg/dataloader/real_robot.py`）：

```python
# 配置文件中：
dataset_path = "/mnt/data/real_camera/robot_recordings"
split = "test"
scene_id = "scene_manual_20260731"

# 实际读取路径：
actual_path = os.path.join(dataset_path, split, scene_id)
# 结果：/mnt/data/real_camera/robot_recordings/test/scene_manual_20260731
```

**如果不想要split层级**：
```python
# 配置文件中：
dataset_path = "/mnt/data/real_camera/robot_recordings"
split = ""  # 留空
scene_id = "scene_manual_20260731"

# 实际读取路径：
actual_path = os.path.join(dataset_path, split, scene_id)
# 结果：/mnt/data/real_camera/robot_recordings/scene_manual_20260731
```

---

## 📋 快速决策表

| 情况 | 推荐方案 | 原因 |
|------|---------|------|
| 本地有GPU | 方案A（本地运行） | 最简单，无需传输 |
| 本地无GPU | 方案B（服务器） | 必须用服务器GPU |
| 数据量<5GB | 方案A | 传输时间短 |
| 数据量>10GB | 方案B | 避免占用本地空间 |
| 想实时查看 | 方案A | 更方便监控 |
| 多任务并行 | 方案B | 服务器性能更强 |

---

## ⚡ 快速命令模板

### 本地运行（最简单）

```bash
# 1. 创建配置
cd C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev
cp config/real_robot_trajectory.yaml config/scene_manual.yaml

# 2. 编辑 config/scene_manual.yaml
#    - dataset_path: C:/Users/Lenovo/Desktop
#    - scene_id: scene_manual_20260731_113213
#    - split: ""

# 3. 运行
python -m hovsg.main --config config/scene_manual.yaml
```

### 服务器运行（标准方案）

```bash
# 1. 上传数据
scp -r "C:/Users/Lenovo/Desktop/scene_manual_20260731_113213" \
    user@server:/mnt/data/real_camera/robot_recordings/test/scene_manual

# 2. SSH到服务器
ssh user@server

# 3. 创建配置
cd /mnt/data/real_camera/isaac/HOV-SG-graph-dev
nano config/scene_manual.yaml
# 配置：
#   - dataset_path: /mnt/data/real_camera/robot_recordings
#   - split: test
#   - scene_id: scene_manual

# 4. 运行
nohup python -m hovsg.main --config config/scene_manual.yaml > build.log 2>&1 &
tail -f build.log
```

---

## 💡 建议

基于您的情况，我建议：

1. **首先尝试方案A（本地运行）**
   - 数据已在本地
   - 配置更简单
   - 可以实时看到进度

2. **如果本地GPU不足或想用服务器**
   - 使用 **rsync** 传输数据（支持断点续传）
   - 使用 **简化方案**（不需要split层级）
   - 使用 **nohup** 后台运行

---

需要我帮您创建具体的配置文件吗？请告诉我：
1. 您想在本地运行还是服务器运行？
2. 如果是服务器，服务器IP和用户名是什么？
3. 本地是否有GPU？
