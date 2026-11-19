# RGB-D、位姿与三维语义关联检查及修复

检查基线：`863396931f2a506360c3b22fc344851b0999e413`。

## 已修复的问题

| 位置 | 原有问题及影响 | 本次修改 |
| --- | --- | --- |
| ScanNet/HM3D/Replica 加载器 | 独立排序后 zip 会在缺帧时截断或错配；Replica 用筛选后的数组位置读取轨迹行；ScanNet 路径拼接依赖末尾斜杠 | 按源帧 ID 严格配对，拒绝缺失/重复 ID；Replica 按源帧编号读取对应轨迹；修复路径拼接 |
| `hovsg/dataloader/generic.py` | 用整块掩码的平均 Z 判断距离，会保留掩码内过远点，也会整块丢掉有效近处点；0/255 掩码通过乘法改变深度；Inf 可进入 KD-tree | 按像素过滤有限、正值、米制深度；掩码只用于布尔选择；未提供位姿时返回相机坐标点 |
| `hovsg/dataloader/real_robot.py` | 直接调用投影时仍允许缩放 RGB，无法保证标定、掩码、深度一致；最小深度设为零会接受零深度；非法标定或同名不同扩展名帧可被接受 | 所有投影入口严格检查标定分辨率；拒绝零深度、非有限深度范围/内参/畸变参数和重复帧 ID |
| `hovsg/graph/graph.py` | 语义融合另外计算有效像素，未来深度过滤变化后可能与点云行顺序不一致；行数相同也不能保证语义位置相同 | `create_pcd(..., return_pixel_indices=True)` 返回投影实际采用的行优先像素索引；直接索引对应语义特征，并检查特征/深度分辨率 |
| `robot_capture/collect_aurora_scene.py` | `--image-rectified` 仍写原始 K；校正后的射线若直接应用原光学坐标 TF，也会引入旋转错误 | 新增可独立测试的 `camera_geometry.py`：原图使用 K，校正图使用 P 的左 3×3；保存 R，加载端用逆校正旋转还原到 TF 光学坐标；ROI/binning 暂不隐式适配，要求上游提供调整后的标定 |
| 几何缓存 | 原签名主要包含帧数和配置，修改 camera.json、位姿、有效帧清单、平面高度等后可能复用旧地图 | 缓存版本升为 2；对有序帧列表、位姿/标定/清单内容、图像大小及修改时间和实际投影策略计算签名 |
| `hovsg/models/sam_clip_feats_extractor.py` | SAM 返回空列表时，CLIP 空批次的 stack 在原来的空结果判断前崩溃；返回值数量不一致 | 在 CLIP 调用前处理空掩码，统一返回四项，语义主流程跳过该帧 |
| `hovsg/utils/feature_map_memory.py` | 多对一关联使用 float16 原始累加，大量观测可能丢失增量/溢出，导致融合方向错误 | 在受限批次内以 float32 聚合重复索引，float16 持久保存运行均值，最后分块归一化；维持原有低内存设计 |

校正模型遵循 [ROS 2 CameraInfo 定义](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg)；米制反投影可对照 [Open3D PointCloud API](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)。

## 坐标与数据约定

深度图必须已经注册到 RGB 像素平面，且采用对应 RGB 相机的内参；同样的分辨率本身不能证明完成了注册。`rgb_depth_aligned=true` 是数据提供者对已完成注册的声明。

本次严格分辨率检查用于 real_robot。传统 benchmark 通用加载器仍保留既有 RGB 缩放兼容逻辑；若 ScanNet 导出的是两个不同光学坐标系的原始流，应先用标定外参完成注册，缩放本身不能替代注册。本次没有在缺少原始传感器数据的情况下推断其外参。

位姿为 **T_world_camera_optical**，将光学坐标 X 右、Y 下、Z 前的点变换到世界坐标，平移单位为米。若采集的是底盘位姿，需先计算 `T_world_base @ T_base_camera_optical`。传入逆变换或遗漏相机外参不能靠调整轴方向补救。

普通图像：`z = raw / depth_units_per_meter`，然后通过 K 反投影。畸变原图使用 `undistortPoints` 取得归一化射线。校正图使用 P 的内参反投影，再应用 R 的转置回到原光学坐标，最后应用相机位姿。`rectification_rotation` 缺省为单位矩阵，兼容已有普通单目数据。

真实手持数据应保留 `pipeline.pose_mode=raw`。逐帧去除相机俯仰、横滚和高度会改变每帧的真实观测方向，不能作为 SLAM 漂移修复方法。导航的水平规划应在地图坐标中独立处理。

## 使用与迁移

1. 拉取本分支后，机器人采集目录应同时更新 `collect_aurora_scene.py` 和新增的 **`camera_geometry.py`**。其他部署文件见 `robot_capture/README.md`。
2. 已校正相机流需要重新导出正确 P/R 标定。旧 camera.json 若使用 K 冒充校正内参，应更新其 fx/fy/cx/cy 和 rectification_rotation，再重建点云和语义特征。
3. 版本 1 几何缓存自动失效；不需要手动删除。旧的最终 `mask_feats.pt`、对象图和导航图不会自动重算，必须重新执行建图入口。
4. 图像文件以大小和修改时间做轻量校验。若外部工具改写图像后刻意保持两者都相同，应设置 `pipeline.resume_geometry_cache=false` 强制重建。

在原项目完整 Linux/CUDA 环境中，根据实际路径运行，例如：

```bash
python application/create_graph.py --config-name real_robot \
  main.dataset_path=/data/robot_recordings main.split=test main.scene_id=scene_001 \
  main.save_path=/data/hovsg_results \
  main.package_path=/path/to/HOVSG/hovsg \
  models.clip.checkpoint=/models/laion2b_s32b_b79k.bin \
  models.sam.checkpoint=/models/sam_vit_h_4b8939.pth \
  pipeline.pose_mode=raw pipeline.extract_semantics=true \
  pipeline.create_graph=true pipeline.create_navigation=true
```

`main.dataset_path` 会追加 `split/scene_id`。现有 `main.package_path` 也应改为本机 HOV-SG 包目录。正式运行前应检查 `navigation.pose_region_is_free`：当前实机配置设为 true，这依赖采集轨迹确实能让目标机器人通行的前提；手持轨迹不能自动提供这个保证。

## 验证范围

新增 `tests/test_rgbd_alignment.py`：逐像素距离过滤、NaN/Inf/零深度、0/255 掩码、无位姿/空点云、已知位姿下的双视角语义融合、距离门限、掩码与几何共用畸变射线、分辨率/标定检查、坐标转换、P/R 标定、缓存失效、空 SAM 结果、缺帧配对、ScanNet 路径与 Replica 轨迹行。

`tests/test_feature_map_memory.py` 新增 10 万个像素观测关联到同一点的回归用例，验证 float16 下的特征方向和计数。

```bash
python -m pytest tests/test_rgbd_alignment.py tests/test_feature_map_memory.py tests/test_semantic_route.py -q
```

本地 Windows / Python 3.11 / NumPy 1.26.4 / Open3D 0.18.0 / CPU PyTorch 环境运行结果：**26 passed**。同时通过修改文件的 Python 编译检查和 `git diff --check`。没有执行其余依赖完整图构建环境的全部测试。

当前环境没有用户真实 RGB-D 数据、SAM/CLIP 权重及机器人 ROS 运行环境，因此这些测试只验证投影、对齐和融合逻辑以及已有语义路由用例；未执行完整神经网络建图和实机导航，不能据此宣称实际定位/导航精度提高。

时间同步仍依赖采集端在图像时间查询 TF，并通过 frames.csv 保存时间戳。对外部导入的数据，文件 ID 对齐不等同于时间同步，应先确认时钟、RGB/深度时间差与位姿来源。几何最近邻距离门限保留，但不能消除错误位姿、动态物体和相邻表面之间的语义污染；这仍需实际录制数据验证。
