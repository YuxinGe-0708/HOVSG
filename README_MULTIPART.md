# HOV-SG 多段采集与离线拼接

## 结论

可以把一个房间分成多段采集，但不能直接拼目录，也不能只要求“画面看起来差不多”。相邻两段必须看见同一块真实空间中的稳定三维结构，算法才能求出两个独立 `map` 坐标系之间的刚体变换。

不要求两段存在完全相同的相机位姿。相机可以相差一定距离和角度，但需要满足：

- 建议有 30% 至 50% 的真实三维重叠，50% 更稳妥。
- 下一段从上一段结束区域附近开始，建议相距不超过约 1 至 2 米。
- 观察方向尽量相近，建议偏航差不超过约 30 至 45 度。
- 重叠区要包含门框、墙角、桌角、柜体等多个不同方向的几何平面。
- 只有白墙、空地板、玻璃、镜子或重复排列的相同家具不够可靠。
- “长得相似但不是同一个位置”会产生错误闭合，比配准失败更危险。

## 数据目录

```text
room_capture/
├── capture.json
├── camera_template.json
├── part_01/
│   ├── camera.json
│   ├── part.json
│   ├── frames.csv
│   ├── valid_frame_ids.txt
│   ├── rgb/000001.png
│   ├── depth/000001.png
│   └── pose/000001.txt
├── part_02/
└── part_03/
```

三个帧文件必须使用完全相同的 `frame_id`。工具会拒绝缺失或错配，不再按排序静默截断。

每个 pose 必须是 4x4 的 `T_world_camera_optical`，即把相机 optical 坐标中的点变换到该段 world/map 坐标。推荐机器人直接保存 ROS Z-up 坐标中的：

```text
T_map_rgb_camera_optical_frame
```

对应 `camera.json`：

```json
{
  "world_frame": "ros_z_up",
  "tf_world_frame": "map",
  "camera_frame": "rgb_camera_optical_frame",
  "pose_convention": "T_world_camera_optical"
}
```

如果只保存 `T_map_base_link`，采集程序必须先计算：

```text
T_map_camera_optical = T_map_base_link * T_base_link_camera_optical
```

不能把 optical 深度点直接乘 `T_map_base_link` 或 `T_map_rgb_camera_link`。

`frames.csv` 每帧一行：

```csv
frame_id,rgb_timestamp_ns,depth_timestamp_ns,pose_timestamp_ns,tracking_quality
000001,1785740000000000000,1785740000000000000,1785740000000000000,1
```

RGB、深度和 TF pose 必须对应同一采样时间。文件名顺序一致不能代替时间同步。

已有采集目录如果缺少 `pose_convention`，必须先确认 `pose/*.txt` 确实是“相机 optical 到该段 map/world”的变换，再向 `camera.json` 增加：

```json
"pose_convention": "T_world_camera_optical"
```

不能仅根据矩阵是 4x4 就补该字段。尤其要检查当前采集脚本查询的是 `rgb_camera_optical_frame`，而不是 `base_link` 或 `rgb_camera_link`。相机驱动不会自动替本项目生成这些约定。

## 采集会话

初始化房间：

```bash
python tools/multipart_capture.py init-room \
  --room-dir /home/creasen/dataset/room_capture \
  --camera-json /path/to/camera.json \
  --room-id room_01
```

创建第一段：

```bash
python tools/multipart_capture.py start-part \
  --room-dir /home/creasen/dataset/room_capture \
  --part-name part_01 \
  --overlap-description "door, left desk, wall corner"
```

创建下一段时记录上一段：

```bash
python tools/multipart_capture.py start-part \
  --room-dir /home/creasen/dataset/room_capture \
  --part-name part_02 \
  --previous-part part_01 \
  --overlap-description "same left desk and wall corner"
```

`start-part` 只创建标准目录。现有机器人录制器需要把数据写入它打印出的 `rgb/`、`depth/`、`pose/`，并追加 `frames.csv`。仓库中没有 ROS 录制器源码，因此本工具不会假装控制相机或 RTAB-Map。

机器人录制回调可以复用 `MultipartFrameWriter`，避免半帧和时间错配：

```python
from hovsg.utils.recording_writer import MultipartFrameWriter

writer = MultipartFrameWriter("/home/creasen/dataset/room_capture/part_01")
writer.write_frame(
    frame_id,
    rgb_bgr,
    depth_uint16,
    T_map_camera_optical,
    rgb_timestamp_ns=rgb_stamp_ns,
    depth_timestamp_ns=depth_stamp_ns,
    pose_timestamp_ns=tf_stamp_ns,
    tracking_quality=tracking_quality,
)
```

`rgb_bgr` 必须是与深度对齐的 `uint8 [H,W,3]`，`depth_uint16` 必须保留相机原始深度单位；例如毫米深度配合 `depth_meters_per_unit: 0.001`。传入的位姿必须是图像时间戳处的 `T_map_camera_optical`。如果当前 TF 查询目标是 `rgb_camera_link`，先通过静态外参转换成 optical frame，不能只修改 JSON 名字。

每段开始和结束时建议在重叠区静止 3 至 5 秒，并在 `tracking_quality <= 0` 后停止当前段。下一段不是要求回到完全相同的相机位置，而是要求重新看见同一块真实几何区域。相机位置和朝向越接近越容易配准，但两段 pose 数值无需相同，因为它们本来就在两个独立 map 坐标系中。

如果第 `001250` 帧之后跟踪失败，并希望再向前裁掉 30 帧：

```bash
python tools/multipart_capture.py finalize-part \
  --room-dir /home/creasen/dataset/room_capture \
  --part-name part_01 \
  --last-valid-frame 001250 \
  --trim-tail-frames 30 \
  --require-timestamps
```

该操作写 `valid_frame_ids.txt`，不会删除原图，后续工具只使用有效帧。

## 位姿漂移处理

`config/real_robot.yaml` 默认：

```yaml
pipeline:
  pose_mode: planar
  planar_pose_height_m: null
  pose_warn_tilt_deg: 5.0
  pose_warn_height_drift_m: 0.15
```

平面模式在 HOV-SG Y-up 坐标中保留 X/Z 平移和 yaw，固定首帧相机高度并去除 roll/pitch。它可以修复平地机器人视觉里程计造成的地板倾斜，但不能修复水平位置漂移或错误闭环，也不能用于坡道、楼梯或相机确实会俯仰的机器人。长期方案仍然是轮速里程计、IMU 与 SLAM 融合，并让 RTAB-Map 使用平面约束。

## 离线拼接

服务器环境示例：

```bash
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-multi-part
export PYTHONPATH="$PWD"
export PYTHON_BIN=/mnt/data/huangyixuan/conda_envs/pfp_saliency_env/bin/python

bash run_multipart_pipeline.sh \
  /mnt/data/huangyixuan/isaac/robot_recordings/test/room_capture \
  part_01 part_02 part_03
```

默认每一对相邻段使用“上一段最后 120 帧”和“下一段最前 120 帧”，再按每 3 帧取一帧生成配准子地图。可以覆盖：

```bash
OVERLAP_FRAMES=180 SKIP_FRAMES=2 VOXEL_SIZE=0.04 \
  bash run_multipart_pipeline.sh /path/to/room_capture part_01 part_02
```

输出：

```text
room_capture_transformations.json
room_capture_aligned_preview.ply
room_capture_transformed/
room_capture_merged/
```

配准使用 FPFH/RANSAC 初始对齐和 point-to-plane ICP 精配准，最终变换强制为 Y-up 的 SE(2)。低于 fitness 门槛或高于 RMSE 门槛会直接停止，不能忽略警告继续生成。

先用 CloudCompare 检查 `room_capture_aligned_preview.ply`。确认相邻段的墙角、桌角和地面连续后，再对 `room_capture_merged` 运行几何或语义建图。

只生成场景点云：

```bash
"$PYTHON_BIN" -m application.create_geometry_map \
  --config-name real_robot \
  main.dataset_path=/mnt/data/huangyixuan/isaac/robot_recordings \
  main.split=test \
  main.scene_id=room_capture_merged \
  main.save_path=/mnt/data/huangyixuan/isaac/hovsg_outputs
```

确认点云正确并准备好 SAM、CLIP 权重后，再生成语义特征和场景图：

```bash
"$PYTHON_BIN" -m application.create_graph \
  --config-name real_robot \
  main.dataset_path=/mnt/data/huangyixuan/isaac/robot_recordings \
  main.split=test \
  main.scene_id=room_capture_merged \
  main.save_path=/mnt/data/huangyixuan/isaac/hovsg_outputs \
  pipeline.extract_semantics=true \
  pipeline.create_graph=true
```
