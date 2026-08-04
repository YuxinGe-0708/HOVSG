#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 REAL_ROBOT_MODIFICATIONS.md 中的代码修改
"""

import os
import sys
import re
from pathlib import Path

def check_modification(file_path, pattern, description):
    """检查文件中是否包含指定的修改"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f"[OK] {description}")
            return True
        else:
            print(f"[FAIL] {description}")
            return False
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False

def main():
    print("=" * 80)
    print("验证真机建图代码修改 - REAL_ROBOT_MODIFICATIONS.md")
    print("=" * 80)
    print()

    base_path = Path(__file__).parent
    graph_py = base_path / "hovsg" / "graph" / "graph.py"

    if not graph_py.exists():
        print(f"[ERROR] 找不到文件: {graph_py}")
        return False

    print(f"检查文件: {graph_py}")
    print()

    checks = []

    # 修改1: segment_floors() - 单楼层检测
    print("【修改1】segment_floors() - 单楼层检测fallback")
    checks.append(check_modification(
        graph_py,
        r'def segment_floors.*?y_min.*?y_max.*?y_range.*?min_floor_height.*?force_single_floor',
        "  检查Y轴范围计算"
    ))
    checks.append(check_modification(
        graph_py,
        r'if force_single_floor or y_range < min_floor_height:',
        "  检查单楼层条件判断"
    ))
    checks.append(check_modification(
        graph_py,
        r'floor = Floor\("0", name="floor_0"\).*?floor\.pcd = self\.full_pcd',
        "  检查单楼层创建逻辑"
    ))
    checks.append(check_modification(
        graph_py,
        r'print_status\(\s*"FLOOR_DETECTION"',
        "  检查楼层检测状态输出"
    ))
    print()

    # 修改2: segment_rooms() - 覆盖范围检查
    print("【修改2】segment_rooms() - 覆盖范围检查")
    checks.append(check_modification(
        graph_py,
        r'def segment_rooms.*?x_range.*?z_range.*?coverage.*?point_count',
        "  检查覆盖范围计算"
    ))
    checks.append(check_modification(
        graph_py,
        r'min_room_coverage.*?skip_if_small',
        "  检查房间分割配置读取"
    ))
    checks.append(check_modification(
        graph_py,
        r'if skip_if_small and \(coverage < min_room_coverage or point_count < 1000\):',
        "  检查覆盖不足条件判断"
    ))
    checks.append(check_modification(
        graph_py,
        r'room = Room\(f"\{floor\.floor_id\}_0", floor\.floor_id, name="room_0"\)',
        "  检查单房间创建逻辑"
    ))
    checks.append(check_modification(
        graph_py,
        r'print_status\(\s*"ROOM_COVERAGE_CHECK"',
        "  检查房间覆盖检测状态输出"
    ))
    print()

    # 修改3: create_feature_map() - 进度估算
    print("【修改3】create_feature_map() - 进度估算")
    checks.append(check_modification(
        graph_py,
        r'estimated_time_per_frame.*?total_estimated_time',
        "  检查时间估算计算"
    ))
    checks.append(check_modification(
        graph_py,
        r'FEATURE MAP CREATION - PROGRESS ESTIMATION',
        "  检查进度估算标题"
    ))
    checks.append(check_modification(
        graph_py,
        r'Total dataset frames.*?Skip frames.*?Candidate frames.*?Total estimated time',
        "  检查进度信息输出"
    ))
    checks.append(check_modification(
        graph_py,
        r'Tip: Increase skip_frames to reduce processing time',
        "  检查优化建议输出"
    ))
    print()

    # 检查import
    print("【依赖检查】")
    checks.append(check_modification(
        graph_py,
        r'from hovsg\.utils\.runtime_status import print_status',
        "  检查 print_status 导入"
    ))
    print()

    # 统计结果
    print("=" * 80)
    total = len(checks)
    passed = sum(checks)
    failed = total - passed

    print(f"验证结果: {passed}/{total} 通过")
    if failed > 0:
        print(f"  失败: {failed} 项")
        return False
    else:
        print("  [OK] 所有修改已正确实施")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
