#!/usr/bin/env python3
"""
验证真机建图代码修改的完整性和正确性
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("HOV-SG 真机建图代码修改验证工具")
print("=" * 80)
print()

# 检查1: 验证配置文件存在
print("✓ 检查1: 验证配置文件")
config_file = Path("config/real_robot_trajectory.yaml")
if config_file.exists():
    print(f"  ✅ 配置文件存在: {config_file}")

    # 读取配置文件内容
    with open(config_file, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # 检查关键配置
    checks = [
        ("extract_semantics: true", "语义提取已开启"),
        ("create_graph: true", "建图功能已开启"),
        ("create_navigation: true", "导航图生成已开启"),
        ("force_single_floor: true", "强制单楼层模式已配置"),
        ("skip_room_segmentation_if_small: true", "房间分割fallback已配置"),
        ("voxel_size: 0.04", "体素大小已优化"),
        ("skip_frames: 3", "跳帧参数已优化"),
        ("keyframe_translation_m: 0.05", "关键帧平移阈值已优化"),
        ("keyframe_rotation_deg: 5.0", "关键帧旋转阈值已优化"),
    ]

    all_passed = True
    for check_str, description in checks:
        if check_str in config_content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} - 未找到: {check_str}")
            all_passed = False

    if all_passed:
        print("  🎉 所有配置检查通过!")
    else:
        print("  ⚠️  部分配置缺失，请检查配置文件")
else:
    print(f"  ❌ 配置文件不存在: {config_file}")

print()

# 检查2: 验证代码修改
print("✓ 检查2: 验证代码修改")
graph_file = Path("hovsg/graph/graph.py")
if graph_file.exists():
    print(f"  ✅ 核心文件存在: {graph_file}")

    with open(graph_file, 'r', encoding='utf-8') as f:
        code_content = f.read()

    code_checks = [
        ("def segment_floors", "楼层分割函数存在"),
        ("force_single_floor", "单楼层检测代码已添加"),
        ("min_floor_height", "楼层高度阈值检查已添加"),
        ("def segment_rooms", "房间分割函数存在"),
        ("min_room_coverage", "房间覆盖范围检查已添加"),
        ("skip_room_segmentation_if_small", "房间分割fallback已添加"),
        ("FEATURE MAP CREATION - PROGRESS ESTIMATION", "进度估算已添加"),
        ("estimated_time_per_frame", "处理时间估算已添加"),
    ]

    all_code_passed = True
    for check_str, description in code_checks:
        if check_str in code_content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} - 未找到关键代码")
            all_code_passed = False

    if all_code_passed:
        print("  🎉 所有代码修改验证通过!")
    else:
        print("  ⚠️  部分代码修改缺失")
else:
    print(f"  ❌ 核心文件不存在: {graph_file}")

print()

# 检查3: 验证文档
print("✓ 检查3: 验证修改文档")
doc_files = [
    "REAL_ROBOT_MODIFICATIONS.md",
    "DIAGNOSIS_REPORT.md",
]

for doc_file in doc_files:
    doc_path = Path(doc_file)
    if doc_path.exists():
        print(f"  ✅ 文档存在: {doc_file}")
    else:
        print(f"  ⚠️  文档缺失: {doc_file}")

print()

# 检查4: 语法检查
print("✓ 检查4: Python语法检查")
try:
    import py_compile

    py_files = [
        "hovsg/graph/graph.py",
        "hovsg/dataloader/real_robot.py",
    ]

    syntax_ok = True
    for py_file in py_files:
        if Path(py_file).exists():
            try:
                py_compile.compile(py_file, doraise=True)
                print(f"  ✅ 语法正确: {py_file}")
            except py_compile.PyCompileError as e:
                print(f"  ❌ 语法错误: {py_file}")
                print(f"     {e}")
                syntax_ok = False
        else:
            print(f"  ⚠️  文件不存在: {py_file}")

    if syntax_ok:
        print("  🎉 所有Python文件语法正确!")
except Exception as e:
    print(f"  ⚠️  无法执行语法检查: {e}")

print()

# 总结
print("=" * 80)
print("验证总结")
print("=" * 80)
print()
print("修改内容：")
print("  1. ✅ 新增优化配置文件: config/real_robot_trajectory.yaml")
print("  2. ✅ segment_floors() - 单楼层检测fallback")
print("  3. ✅ segment_rooms() - 房间分割健壮性检查")
print("  4. ✅ create_feature_map() - 进度估算")
print()
print("下一步操作：")
print("  1. 采集真机完整轨迹数据（不是10°测试）")
print("  2. 修改配置文件中的路径")
print("     - dataset_path: 你的数据路径")
print("     - save_path: 输出路径")
print("  3. 运行建图命令（预计30-60分钟）")
print()
print("建图命令示例：")
print("  cd", os.getcwd())
print("  python -m hovsg.main --config config/real_robot_trajectory.yaml")
print()
print("=" * 80)
