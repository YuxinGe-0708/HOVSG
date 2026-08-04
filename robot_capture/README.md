# Aurora 930 多段 RGB-D 采集

本目录基于机器人现有 ROS 2 采集脚本改造。手动巡航应使用
`run_aurora_manual_capture.sh`；自动旋转脚本只用于 10 度和 90 度安全测试，
不用于完整房间巡航。

## 输出格式

第一次采集会创建：

```text
/home/creasen/dataset/room_capture_YYYYMMDD_HHMMSS/
├── capture.json
├── camera_template.json
└── part_01/
    ├── camera.json
    ├── part.json
    ├── frames.csv
    ├── valid_frame_ids.txt
    ├── rgb/000001.png
    ├── depth/000001.png
    └── pose/000001.txt
```

后续采集向同一个房间目录增加 `part_02`、`part_03`。每一段都会重新启动
RTAB-Map，因此每段 pose 使用独立的 map 原点；不能直接把 pose 文件放在一起，
必须在服务器运行多段配准流水线。

每一行 `frames.csv` 记录：

```csv
frame_id,rgb_timestamp_ns,depth_timestamp_ns,pose_timestamp_ns,tracking_quality
000001,1785800000000000000,1785800000005000000,1785800000000000000,
```

当前链路没有可靠的 RTAB-Map tracking quality 数值话题，所以
`tracking_quality` 留空，不会伪造为 1。只有 RGB/Depth 时间差不超过 20 ms、
TF 返回时间与 RGB 时间差不超过 50 ms，并成功取得对应 TF 的帧才会保存。

`camera.json` 会明确保存：

```json
{
  "world_frame": "camera_optical_initial",
  "tf_world_frame": "map",
  "camera_frame": "rgb_camera_link",
  "camera_frame_axes": "optical_x_right_y_down_z_forward",
  "pose_convention": "T_world_camera_optical"
}
```

Aurora 驱动的 frame 名称虽然是 `rgb_camera_link`，当前已确认其轴为 optical：
X 向右、Y 向下、Z 向前。因此这里的 pose 可以用于变换深度反投影得到的
optical 点。`--camera-frame-is-optical` 是显式确认，不是坐标转换操作。

## 部署到机器人

将本目录下五个文件放到机器人 `/home/creasen/`：

```text
collect_aurora_scene.py
capture_rotation_scene.py
sync_rgb_camera_info.py
run_aurora_manual_capture.sh
run_aurora_rotation_capture.sh
```

然后执行：

```bash
chmod +x /home/creasen/collect_aurora_scene.py \
  /home/creasen/capture_rotation_scene.py \
  /home/creasen/sync_rgb_camera_info.py \
  /home/creasen/run_aurora_manual_capture.sh \
  /home/creasen/run_aurora_rotation_capture.sh
```

## 采集第一段

确保没有其他 Aurora、RTAB-Map 或旧采集节点运行，然后执行：

```bash
bash /home/creasen/run_aurora_manual_capture.sh
```

脚本启动相机、CameraInfo 时间戳同步节点和 RTAB-Map。提示可以巡航后：

1. 在有门框、桌角、墙角等结构的位置静止 3 至 5 秒。
2. 缓慢遥控机器人完成第一段。
3. 在计划作为下一段起点的重叠区域静止 3 至 5 秒。
4. 回到运行脚本的终端按 `Ctrl+C`。

脚本会完成剩余写入、生成 `valid_frame_ids.txt` 并检查五类帧记录数量。
房间目录保存在：

```bash
cat /home/creasen/dataset/latest_room_capture.txt
```

## 采集第二段

第二段必须复用第一段的房间目录：

```bash
ROOM_DIR="$(cat /home/creasen/dataset/latest_room_capture.txt)" \
PART_NAME=part_02 \
PREVIOUS_PART=part_01 \
OVERLAP_DESCRIPTION="same door frame, desk corner and wall corner" \
bash /home/creasen/run_aurora_manual_capture.sh
```

第三段示例：

```bash
ROOM_DIR="$(cat /home/creasen/dataset/latest_room_capture.txt)" \
PART_NAME=part_03 \
PREVIOUS_PART=part_02 \
OVERLAP_DESCRIPTION="same cabinet, rear wall corner and table edge" \
bash /home/creasen/run_aurora_manual_capture.sh
```

相邻段不要求相机回到完全相同的 pose。必须重新观察同一块真实三维区域，
建议保留 30% 至 50% 的重叠，并尽量让下一段起点与上一段终点相距不超过
约 1 至 2 米、朝向差不超过约 30 至 45 度。只有白墙、地板或外观相似但
位置不同的家具不能作为可靠重叠。

## 检查结果

```bash
ROOM_DIR="$(cat /home/creasen/dataset/latest_room_capture.txt)"

for PART in part_01 part_02; do
  echo "===== $PART ====="
  wc -l "$ROOM_DIR/$PART/valid_frame_ids.txt"
  tail -n +2 "$ROOM_DIR/$PART/frames.csv" | wc -l
  find "$ROOM_DIR/$PART/rgb" -maxdepth 1 -name '*.png' | wc -l
  find "$ROOM_DIR/$PART/depth" -maxdepth 1 -name '*.png' | wc -l
  find "$ROOM_DIR/$PART/pose" -maxdepth 1 -name '*.txt' | wc -l
  grep -E 'world_frame|camera_frame|camera_frame_axes|pose_convention' \
    "$ROOM_DIR/$PART/camera.json"
done
```

同一段的五个数量必须完全相同。

## 跟踪失效后的裁剪

采集器只能确认对应时间 TF 是否存在，不能保证 RTAB-Map 的位姿没有逐渐漂移。
若日志或点云表明 `part_02` 在 `001250` 后开始发散，应在服务器项目目录执行：

```bash
python tools/multipart_capture.py finalize-part \
  --room-dir /path/to/room_capture_YYYYMMDD_HHMMSS \
  --part-name part_02 \
  --last-valid-frame 001250 \
  --trim-tail-frames 20 \
  --require-timestamps
```

该命令只重写有效帧清单，不删除原始 RGB、Depth 和 Pose。

## 仍需注意

- 当前 `camera_optical_initial` 是兼容现有无 `base_link` 链路的方案。
- 平面位姿修正会在 HOV-SG 加载阶段去除 roll、pitch 和高度漂移。
- 它不能修复一段内部的 X/Z 平移漂移、yaw 漂移或错误闭环。
- 每段应保持短而稳定；一旦定位持续失败，应立即结束并重新开始下一段。
- 长期正式方案仍应标定 `base_link -> rgb_camera_link`，融合轮速和 IMU，
  并让 SLAM 工作在真实的 ROS Z-up 底盘坐标系中。
