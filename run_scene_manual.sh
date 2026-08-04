#!/bin/bash
# 真机数据建图 - 一键运行脚本
# 数据：scene_manual_20260731_113213 (3982帧)

echo "========================================================================"
echo "HOV-SG 真机完整轨迹建图"
echo "========================================================================"
echo ""
echo "数据信息："
echo "  - 数据路径: C:/Users/Lenovo/Desktop/scene_manual_20260731_113213"
echo "  - 总帧数: 3982帧"
echo "  - 跳帧设置: skip_frames=5"
echo "  - 候选帧数: 约797帧"
echo "  - 估算时间: 45-70分钟"
echo ""

# 检查数据目录
DATA_DIR="C:/Users/Lenovo/Desktop/scene_manual_20260731_113213"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误：数据目录不存在: $DATA_DIR"
    exit 1
fi

echo "✅ 数据目录检查通过"
echo ""

# 检查camera.json
if [ ! -f "$DATA_DIR/camera.json" ]; then
    echo "❌ 错误：camera.json不存在"
    exit 1
fi

echo "✅ camera.json检查通过"
echo ""

# 统计帧数
RGB_COUNT=$(ls "$DATA_DIR/rgb" | wc -l)
DEPTH_COUNT=$(ls "$DATA_DIR/depth" | wc -l)
POSE_COUNT=$(ls "$DATA_DIR/pose" | wc -l)

echo "帧数统计："
echo "  - RGB: $RGB_COUNT"
echo "  - Depth: $DEPTH_COUNT"
echo "  - Pose: $POSE_COUNT"

if [ "$RGB_COUNT" != "$DEPTH_COUNT" ] || [ "$RGB_COUNT" != "$POSE_COUNT" ]; then
    echo "⚠️  警告：RGB/Depth/Pose帧数不一致"
    echo "   建图仍会继续，但可能影响结果"
else
    echo "✅ 帧数一致性检查通过"
fi

echo ""
echo "========================================================================"
echo "开始建图..."
echo "========================================================================"
echo ""

# 切换到项目目录
cd "C:/Users/Lenovo/Documents/Codex/2026-07-24/a-m/HOV-SG-graph-dev" || exit

# 检查配置文件
CONFIG="config/scene_manual_local.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误：配置文件不存在: $CONFIG"
    echo "   请先运行: cp config/real_robot_trajectory.yaml $CONFIG"
    exit 1
fi

echo "✅ 配置文件: $CONFIG"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息："
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  警告：未检测到nvidia-smi，请确保有可用的CUDA GPU"
    echo ""
fi

# 提示用户
echo "即将运行建图，预计需要45-70分钟"
echo "输出将保存到: C:/Users/Lenovo/Desktop/outputs_manual"
echo ""
read -p "是否继续？(y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "========================================================================"
echo "运行建图..."
echo "========================================================================"
echo ""

# 创建输出目录
mkdir -p "C:/Users/Lenovo/Desktop/outputs_manual"

# 运行建图
python -m hovsg.main --config "$CONFIG"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 建图完成！"
    echo ""
    echo "输出文件："
    echo "  - 全局点云: C:/Users/Lenovo/Desktop/outputs_manual/full_pcd.ply"
    echo "  - 语义地图: C:/Users/Lenovo/Desktop/outputs_manual/graph/"
    echo "  - 导航图: C:/Users/Lenovo/Desktop/outputs_manual/graph/nav_graph/"
    echo ""
    echo "验证建图结果："
    echo "  python -c \"import open3d as o3d; pcd = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/outputs_manual/full_pcd.ply'); print(f'Points: {len(pcd.points)}'); o3d.visualization.draw_geometries([pcd])\""
else
    echo "❌ 建图失败，退出码: $EXIT_CODE"
    echo ""
    echo "请检查："
    echo "  1. GPU是否可用"
    echo "  2. 模型checkpoint是否下载"
    echo "  3. 查看错误日志"
fi
echo "========================================================================"
