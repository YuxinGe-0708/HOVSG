#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/creasen/dataset}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/creasen}"
ROOM_DIR="${ROOM_DIR:-}"
PART_NAME="${PART_NAME:-part_01}"
PREVIOUS_PART="${PREVIOUS_PART:-}"
OVERLAP_DESCRIPTION="${OVERLAP_DESCRIPTION:-}"
ROS_SETUP="/opt/ros/humble/setup.bash"
DEPTRUM_SETUP="/home/creasen/deptrum_ws/install/local_setup.bash"

CAMERA_PID=""
SYNC_PID=""
RTABMAP_PID=""
SCENE_DIR=""
LOG_DIR=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "错误：$*"
  exit 1
}

stop_group() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 0.1
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  stop_group "$RTABMAP_PID"
  stop_group "$SYNC_PID"
  stop_group "$CAMERA_PID"
  [[ -n "$SCENE_DIR" ]] && log "数据目录：$SCENE_DIR"
  [[ -n "$LOG_DIR" ]] && log "日志目录：$LOG_DIR"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM HUP

wait_for_topic() {
  local topic="$1"
  local timeout_seconds="$2"
  local start=$SECONDS
  while (( SECONDS - start < timeout_seconds )); do
    if timeout 3 ros2 topic echo "$topic" --once >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_tf() {
  local target="$1"
  local source="$2"
  local timeout_seconds="$3"
  local start=$SECONDS
  local tf_output=""
  while (( SECONDS - start < timeout_seconds )); do
    tf_output="$(
      timeout 4 ros2 run tf2_ros tf2_echo "$target" "$source" 2>/dev/null || true
    )"
    if grep -q 'Translation:' <<<"$tf_output"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

[[ -r "$ROS_SETUP" ]] || die "找不到 $ROS_SETUP"
[[ -r "$DEPTRUM_SETUP" ]] || die "找不到 $DEPTRUM_SETUP"
for script in collect_aurora_scene.py sync_rgb_camera_info.py; do
  [[ -r "$SCRIPT_ROOT/$script" ]] || die "缺少 $SCRIPT_ROOT/$script"
done
grep -q 'T_world_camera_optical' "$SCRIPT_ROOT/collect_aurora_scene.py" || \
  die "collect_aurora_scene.py不是多段格式版本"

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$DEPTRUM_SETUP"
set -u

command -v ros2 >/dev/null || die "找不到 ros2"
command -v timeout >/dev/null || die "找不到 timeout"
command -v setsid >/dev/null || die "找不到 setsid"

if ros2 node list 2>/dev/null | grep -Eq \
  '^/aurora$|^/rtabmap/|^/rgb_camera_info_stamp_synchronizer$|^/aurora_scene_collector$'; then
  die "检测到旧的相机、RTAB-Map或采集节点；请先停止旧流程"
fi
if pgrep -f '[c]apture_rotation_scene.py' >/dev/null 2>&1; then
  die "检测到自动旋转采集脚本仍在运行；请先停止它"
fi

mkdir -p "$DATASET_ROOT"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
[[ "$PART_NAME" =~ ^part_[0-9]+$ ]] || \
  die "PART_NAME必须采用part_01、part_02这样的格式"
if [[ -z "$ROOM_DIR" ]]; then
  [[ "$PART_NAME" == "part_01" ]] || \
    die "采集part_02及以后时必须设置已有的ROOM_DIR"
  ROOM_DIR="$DATASET_ROOT/room_capture_${RUN_STAMP}"
fi
mkdir -p "$ROOM_DIR"
if [[ "$PART_NAME" != "part_01" ]]; then
  [[ -n "$PREVIOUS_PART" ]] || die "后续段必须设置PREVIOUS_PART"
  [[ -f "$ROOM_DIR/$PREVIOUS_PART/valid_frame_ids.txt" ]] || \
    die "上一段不存在或未完成：$ROOM_DIR/$PREVIOUS_PART"
fi
SCENE_DIR="$ROOM_DIR/$PART_NAME"
[[ ! -e "$SCENE_DIR" ]] || die "目标分段目录已存在：$SCENE_DIR"
LOG_DIR="$DATASET_ROOT/logs_${PART_NAME}_${RUN_STAMP}"
RTABMAP_DB="$DATASET_ROOT/rtabmap_${PART_NAME}_${RUN_STAMP}.db"
mkdir -p "$LOG_DIR"
printf '%s\n' "$ROOM_DIR" > "$DATASET_ROOT/latest_room_capture.txt"
printf '%s\n' "$SCENE_DIR" > "$DATASET_ROOT/latest_capture_part.txt"

log "房间目录：$ROOM_DIR"
log "当前分段：$PART_NAME"
log "输出目录：$SCENE_DIR"
log "此流程不会控制底盘；请使用遥控器驾驶"

setsid ros2 run deptrum-ros-driver-aurora930 aurora930_node \
  --ros-args -p align_mode:=true \
  >"$LOG_DIR/camera.log" 2>&1 &
CAMERA_PID=$!
wait_for_topic /rgb/image_raw 30 || die "RGB启动失败，查看 $LOG_DIR/camera.log"
wait_for_topic /depth/image_raw 30 || die "深度启动失败，查看 $LOG_DIR/camera.log"

setsid python3 "$SCRIPT_ROOT/sync_rgb_camera_info.py" \
  --image-topic /rgb/image_raw \
  --input-info-topic /rgb/camera_info \
  --output-info-topic /rgb/camera_info_synced \
  >"$LOG_DIR/camera_info_sync.log" 2>&1 &
SYNC_PID=$!
wait_for_topic /rgb/camera_info_synced 20 || \
  die "CameraInfo同步失败，查看 $LOG_DIR/camera_info_sync.log"

setsid ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/rgb/image_raw \
  depth_topic:=/depth/image_raw \
  camera_info_topic:=/rgb/camera_info_synced \
  frame_id:=rgb_camera_link \
  approx_sync:=true \
  qos:=2 \
  rviz:=false \
  rtabmap_viz:=false \
  database_path:="$RTABMAP_DB" \
  >"$LOG_DIR/rtabmap.log" 2>&1 &
RTABMAP_PID=$!

wait_for_topic /rtabmap/odom 60 || \
  die "RTAB-Map里程计无输出，查看 $LOG_DIR/rtabmap.log"
wait_for_tf odom rgb_camera_link 30 || die "缺少 odom <- rgb_camera_link TF"
wait_for_tf map rgb_camera_link 30 || die "缺少 map <- rgb_camera_link TF"

log "检查通过。先保持静止5秒，然后开始遥控巡航"
sleep 5
log "正在采集；完成巡航后回到此tmux窗口按Ctrl+C"

python3 "$SCRIPT_ROOT/collect_aurora_scene.py" \
  --output "$SCENE_DIR" \
  --part-name "$PART_NAME" \
  --previous-part "$PREVIOUS_PART" \
  --overlap-description "$OVERLAP_DESCRIPTION" \
  --rgb-topic /rgb/image_raw \
  --depth-topic /depth/image_raw \
  --camera-info-topic /rgb/camera_info_synced \
  --tf-world-frame map \
  --world-frame camera_optical_initial \
  --camera-frame rgb_camera_link \
  --camera-frame-is-optical \
  --depth-scale 0.001 \
  --min-depth 0.15 \
  --max-depth 4.0 \
  --rgb-depth-aligned \
  --sync-slop 0.02 \
  --max-rgb-depth-skew-ms 20 \
  --max-pose-skew-ms 50 \
  --tf-timeout 0.02 \
  --fps 10 \
  --pose-queue-timeout 1.0

RGB_COUNT="$(find "$SCENE_DIR/rgb" -maxdepth 1 -type f -name '*.png' | wc -l)"
DEPTH_COUNT="$(find "$SCENE_DIR/depth" -maxdepth 1 -type f -name '*.png' | wc -l)"
POSE_COUNT="$(find "$SCENE_DIR/pose" -maxdepth 1 -type f -name '*.txt' | wc -l)"
METADATA_COUNT="$(tail -n +2 "$SCENE_DIR/frames.csv" | wc -l)"
VALID_COUNT="$(grep -c '^[0-9]' "$SCENE_DIR/valid_frame_ids.txt")"
log "文件数量：RGB=$RGB_COUNT，depth=$DEPTH_COUNT，pose=$POSE_COUNT"
[[ "$RGB_COUNT" -eq "$DEPTH_COUNT" && "$RGB_COUNT" -eq "$POSE_COUNT" ]] || \
  die "RGB、深度和位姿数量不一致"
[[ "$RGB_COUNT" -eq "$METADATA_COUNT" && "$RGB_COUNT" -eq "$VALID_COUNT" ]] || \
  die "帧文件、frames.csv和valid_frame_ids.txt数量不一致"
[[ "$RGB_COUNT" -gt 0 ]] || die "没有保存任何数据帧"
grep -q '"pose_convention": "T_world_camera_optical"' \
  "$SCENE_DIR/camera.json" || die "camera.json缺少正确的pose_convention"
log "分段采集完成：$SCENE_DIR"
