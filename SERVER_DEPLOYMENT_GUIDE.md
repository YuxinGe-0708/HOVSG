# 服务器部署完整指南

## 📍 服务器信息

**服务器路径**: `/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2`

---

## 📂 服务器目录结构规划

```
/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/
├── hovsg/                        # 项目代码
├── config/                       # 配置文件
│   └── scene_manual_server.yaml  # 已创建的服务器配置
├── checkpoints/                  # 模型文件
│   ├── laion2b_s32b_b79k.bin    # CLIP模型 (~1.7GB)
│   └── sam_vit_h_4b8939.pth     # SAM模型 (~2.4GB)
├── data/                         # 数据目录
│   └── test/                     # split目录
│       └── scene_manual/         # 场景数据
│           ├── camera.json
│           ├── rgb/
│           ├── depth/
│           └── pose/
└── outputs/                      # 输出目录
    └── scene_manual/             # 建图结果
```

---

## 🚀 部署步骤

### 步骤1: 上传数据到服务器

#### 方法1: 使用rsync（推荐，支持断点续传）

```bash
# 在本地Windows运行（Git Bash或WSL）
rsync -avz --progress \
    "/c/Users/Lenovo/Desktop/scene_manual_20260731_113213/" \
    username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/scene_manual/
```

**参数说明**:
- `-a`: 归档模式（保留权限、时间戳）
- `-v`: 显示详细信息
- `-z`: 压缩传输
- `--progress`: 显示进度条

#### 方法2: 先压缩再上传（更快，适合首次传输）

```bash
# 1. 在本地压缩（Windows PowerShell或Git Bash）
cd C:/Users/Lenovo/Desktop
tar -czf scene_manual.tar.gz scene_manual_20260731_113213

# 2. 上传压缩包
scp scene_manual.tar.gz username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/

# 3. SSH登录服务器
ssh username@server_ip

# 4. 解压
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/
tar -xzf scene_manual.tar.gz
mv scene_manual_20260731_113213 scene_manual

# 5. 清理压缩包
rm scene_manual.tar.gz
```

#### 方法3: 使用WinSCP（Windows图形界面）

1. 下载安装 WinSCP
2. 连接到服务器
3. 导航到 `/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/`
4. 拖拽上传 `scene_manual_20260731_113213` 文件夹
5. 重命名为 `scene_manual`

---

### 步骤2: 上传配置文件到服务器

```bash
# 在本地运行
scp C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev/config/scene_manual_server.yaml \
    username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/config/
```

---

### 步骤3: 在服务器上验证数据

```bash
# SSH登录服务器
ssh username@server_ip

# 验证数据路径
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/scene_manual

# 检查文件结构
ls -la

# 应该看到:
# camera.json
# rgb/ (目录)
# depth/ (目录)
# pose/ (目录)

# 统计帧数
echo "RGB frames: $(ls rgb | wc -l)"
echo "Depth frames: $(ls depth | wc -l)"
echo "Pose frames: $(ls pose | wc -l)"

# 应该都是 3982
```

---

### 步骤4: 检查服务器环境

```bash
# 检查conda环境
conda env list
# 应该看到 hovsg 环境

# 激活环境
conda activate hovsg

# 检查GPU
nvidia-smi

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查模型文件
ls -lh /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/checkpoints/
# 应该看到:
# laion2b_s32b_b79k.bin (~1.7GB)
# sam_vit_h_4b8939.pth (~2.4GB)
```

---

### 步骤5: 运行建图

#### 方法A: 使用screen（推荐，支持断线重连）

```bash
# 1. SSH登录服务器
ssh username@server_ip

# 2. 创建screen会话
screen -S hovsg_build

# 3. 激活环境
conda activate hovsg

# 4. 进入项目目录
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2

# 5. 运行建图
python application/create_graph.py --config-name scene_manual_server

# 6. 等待开始处理...

# 7. 退出screen（保持运行）
# 按 Ctrl+A，然后按 D

# 8. 重新连接screen
screen -r hovsg_build

# 9. 终止screen
# 在screen内按 Ctrl+C，然后输入 exit
```

#### 方法B: 使用nohup（后台运行）

```bash
# 1. SSH登录服务器
ssh username@server_ip

# 2. 激活环境
conda activate hovsg

# 3. 进入项目目录
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2

# 4. 后台运行
nohup python application/create_graph.py --config-name scene_manual_server > build_manual.log 2>&1 &

# 5. 记录进程ID
echo $! > build_manual.pid

# 6. 查看日志（实时）
tail -f build_manual.log

# 7. 查看进度（新终端）
grep "PROGRESS" build_manual.log | tail -20

# 8. 停止进程（如果需要）
kill $(cat build_manual.pid)
```

---

## 📊 监控进度

### 实时查看日志

```bash
# 查看完整日志
tail -f build_manual.log

# 只看进度信息
tail -f build_manual.log | grep --line-buffered "PROGRESS\|OK\|DONE"

# 查看错误
tail -f build_manual.log | grep --line-buffered "ERROR\|FAIL"
```

### 检查GPU使用

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者
nvidia-smi dmon -i 0 -s mu
```

### 估算完成时间

```bash
# 查看已处理帧数
grep "Processing frame" build_manual.log | wc -l

# 估算剩余时间
# 总帧数: ~797帧
# 已处理: X帧
# 剩余: 797 - X
# 预计时间: (797 - X) * 4秒 / 60 = Y分钟
```

---

## 📁 输出文件位置

建图完成后，输出在：

```
/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual/real_robot/scene_manual/
├── full_pcd.ply              # 全局点云
├── masked_pcd.ply            # 物体mask点云
├── full_feats.pt             # CLIP特征
├── mask_feats.pt             # 物体特征
└── graph/
    ├── floors/
    │   ├── 0.ply             # 楼层点云
    │   └── 0.json
    ├── rooms/
    │   ├── 0_0.ply           # 房间点云
    │   └── 0_0.json
    ├── objects/              # 识别的物体
    │   ├── 0_0_0.ply
    │   └── ...
    └── nav_graph/
        ├── global_nav_graph.pkl
        └── sparse_voronoi.pkl
```

---

## 🔍 结果验证

### 在服务器上验证

```bash
# 检查输出文件
ls -lh /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual/real_robot/scene_manual/

# 统计点云点数
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual/real_robot/scene_manual/full_pcd.ply'); print(f'Points: {len(pcd.points):,}')"

# 检查导航图
python -c "import pickle; nav = pickle.load(open('/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual/real_robot/scene_manual/graph/nav_graph/global_nav_graph.pkl', 'rb')); print(f'Nodes: {nav.number_of_nodes()}, Edges: {nav.number_of_edges()}')"
```

### 下载结果到本地查看

```bash
# 下载完整输出目录
scp -r username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual \
    C:/Users/Lenovo/Desktop/outputs_manual_from_server

# 或只下载点云文件
scp username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual/real_robot/scene_manual/full_pcd.ply \
    C:/Users/Lenovo/Desktop/
```

在本地用Open3D查看：

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/full_pcd.ply')
print(f'Total points: {len(pcd.points):,}')
o3d.visualization.draw_geometries([pcd], window_name="Server Build Result")
```

---

## ⏱️ 处理时间估算

| 项目 | 数值 |
|------|------|
| 原始帧数 | 3982帧 |
| 跳帧设置 | skip_frames=5 |
| 候选帧数 | ~797帧 |
| 每帧处理 | ~4秒 |
| **估算总时间** | **45-70分钟** |

实际时间取决于服务器GPU性能。

---

## 🐛 故障排除

### 问题1: 数据路径错误

**错误**: `FileNotFoundError: /mnt/data/.../test/scene_manual/camera.json`

**检查**:
```bash
# 验证完整路径
ls /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/scene_manual/camera.json

# 检查配置文件中的路径
grep "dataset_path\|scene_id\|split" config/scene_manual_server.yaml
```

### 问题2: 模型文件未找到

**错误**: `FileNotFoundError: checkpoints/laion2b_s32b_b79k.bin`

**解决**:
```bash
# 检查模型文件
ls -lh /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/checkpoints/

# 如果缺失，从其他位置复制或重新下载
```

### 问题3: GPU内存不足

**错误**: `CUDA out of memory`

**解决**: 修改配置文件增大跳帧
```bash
# 编辑配置
nano /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/config/scene_manual_server.yaml

# 修改: skip_frames: 5 -> skip_frames: 7
```

### 问题4: conda环境未激活

**错误**: `ModuleNotFoundError: No module named 'torch'`

**解决**:
```bash
conda activate hovsg
```

---

## 📋 快速命令清单

### 上传数据（rsync）
```bash
rsync -avz --progress \
    "/c/Users/Lenovo/Desktop/scene_manual_20260731_113213/" \
    username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/data/test/scene_manual/
```

### 运行建图（screen）
```bash
ssh username@server_ip
screen -S hovsg_build
conda activate hovsg
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2
python application/create_graph.py --config-name scene_manual_server
# Ctrl+A, D 退出screen
```

### 查看进度
```bash
screen -r hovsg_build
# 或
tail -f build_manual.log
```

### 下载结果
```bash
scp -r username@server_ip:/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-v2/outputs/scene_manual \
    C:/Users/Lenovo/Desktop/
```

---

## ✅ 部署检查清单

上传前：
- [ ] 本地数据完整（3982帧RGB+Depth+Pose）
- [ ] 配置文件已创建（scene_manual_server.yaml）

上传后：
- [ ] 数据已传输到服务器 `/data/test/scene_manual/`
- [ ] 配置文件已上传到 `/config/`
- [ ] 服务器conda环境已激活
- [ ] GPU可用（nvidia-smi检查）
- [ ] 模型文件存在（checkpoints/）

运行中：
- [ ] screen/nohup已启动
- [ ] 日志正常输出
- [ ] GPU被占用
- [ ] 无错误信息

完成后：
- [ ] 输出目录包含所有文件
- [ ] 点云文件可读取
- [ ] 导航图生成成功

---

**准备好了吗？按照步骤1开始上传数据！**
