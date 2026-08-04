#!/usr/bin/env bash
set -Eeuo pipefail

ANGLE_MODE="${1:-10}"
SERIAL_DEVICE="${SERIAL_DEVICE:-/dev/ttyUSB0}"
DATASET_ROOT="${DATASET_ROOT:-/home/creasen/dataset}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/creasen}"
ROOM_DIR="${ROOM_DIR:-}"
PART_NAME="${PART_NAME:-part_01}"
PREVIOUS_PART="${PREVIOUS_PART:-}"
OVERLAP_DESCRIPTION="${OVERLAP_DESCRIPTION:-rotation test overlap}"
ROS_SETUP="/opt/ros/humble/setup.bash"
DEPTRUM_SETUP="/home/creasen/deptrum_ws/install/local_setup.bash"
LOG_DIR=""
SCENE_DIR=""
CAMERA_PID=""
SYNC_PID=""
RTABMAP_PID=""
MOTION_STARTED=0

usage() {
  printf '%s\n' \
    '用法：' \
    '  bash /home/creasen/run_aurora_rotation_capture.sh 10' \
    '  bash /home/creasen/run_aurora_rotation_capture.sh 90 --confirm-90' \
    '' \
    '默认采用当前兼容链路：' \
    '  RTAB-Map frame_id=rgb_camera_link' \
    '  world_frame=camera_optical_initial' \
    '  CCW yaw sign=-1' \
    '' \
    '环境变量（可选）：' \
    '  SERIAL_DEVICE=/dev/ttyUSB0' \
    '  DATASET_ROOT=/home/creasen/dataset' \
    '  SCRIPT_ROOT=/home/creasen'
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "错误：$*"
  exit 1
}

send_stop() {
  if [[ -w "$SERIAL_DEVICE" ]]; then
    printf '\xA5\x5A\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' \
      > "$SERIAL_DEVICE" 2>/dev/null || true
  fi
}

stop_pid() {
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
  trap - EXIT INT TERM
  if (( MOTION_STARTED )); then
    send_stop
    sleep 0.05
    send_stop
    sleep 0.05
    send_stop
  fi
  stop_pid "$RTABMAP_PID"
  stop_pid "$SYNC_PID"
  stop_pid "$CAMERA_PID"
  if [[ -n "$SCENE_DIR" ]]; then
    log "数据目录：$SCENE_DIR"
  fi
  if [[ -n "$LOG_DIR" ]]; then
    log "日志目录：$LOG_DIR"
  fi
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
    # tf2_echo streams forever. Capture one bounded sample first so grep -q
    # cannot close the pipe early and make pipefail report a false failure.
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

if [[ "$ANGLE_MODE" == "-h" || "$ANGLE_MODE" == "--help" ]]; then
  usage
  exit 0
fi

case "$ANGLE_MODE" in
  10)
    ROTATION_RADIANS="0.17453292519943295"
    SCENE_PREFIX="scene_rotation_10_test"
    ;;
  90)
    [[ "${2:-}" == "--confirm-90" ]] || {
      usage
      die "90°运动必须附加 --confirm-90；应先完成10°测试"
    }
    ROTATION_RADIANS="1.5707963267948966"
    SCENE_PREFIX="scene_rotation_90"
    ;;
  *)
    usage
    die "角度只能是10或90"
    ;;
esac

[[ -r "$ROS_SETUP" ]] || die "找不到 $ROS_SETUP"
[[ -r "$DEPTRUM_SETUP" ]] || die "找不到 $DEPTRUM_SETUP"
for script in collect_aurora_scene.py capture_rotation_scene.py sync_rgb_camera_info.py; do
  [[ -r "$SCRIPT_ROOT/$script" ]] || die "缺少 $SCRIPT_ROOT/$script"
done

# shellcheck disable=SC1090
set +u
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$DEPTRUM_SETUP"
set -u

grep -q 'camera_optical_initial' "$SCRIPT_ROOT/collect_aurora_scene.py" || \
  die "collect_aurora_scene.py不是坐标修正版"
grep -q 'T_world_camera_optical' "$SCRIPT_ROOT/collect_aurora_scene.py" || \
  die "collect_aurora_scene.py不是多段格式版本"
grep -q 'ccw-yaw-sign' "$SCRIPT_ROOT/capture_rotation_scene.py" || \
  die "capture_rotation_scene.py不是方向修正版"
grep -q 'camera-frame-is-optical' "$SCRIPT_ROOT/capture_rotation_scene.py" || \
  die "capture_rotation_scene.py不是多段格式版本"

command -v ros2 >/dev/null || die "找不到ros2"
command -v timeout >/dev/null || die "找不到timeout"
command -v setsid >/dev/null || die "找不到setsid"
[[ -e "$SERIAL_DEVICE" ]] || die "串口不存在：$SERIAL_DEVICE"

if command -v fuser >/dev/null; then
  SERIAL_USERS="$(fuser "$SERIAL_DEVICE" 2>/dev/null || true)"
  [[ -z "$SERIAL_USERS" ]] || \
    die "串口被进程占用，PID：$SERIAL_USERS。请先确认并关闭该进程"
fi
[[ -r "$SERIAL_DEVICE" && -w "$SERIAL_DEVICE" ]] || \
  die "无串口读写权限，请先执行：sudo chmod 666 $SERIAL_DEVICE"

if ros2 node list 2>/dev/null | grep -Eq \
  '^/aurora$|^/rtabmap/|^/rgb_camera_info_stamp_synchronizer$|^/aurora_scene_collector$'; then
  die "检测到相机、RTAB-Map或同步节点仍在运行；请先Ctrl+C停止旧终端"
fi

mkdir -p "$DATASET_ROOT"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
[[ "$PART_NAME" =~ ^part_[0-9]+$ ]] || \
  die "PART_NAME必须采用part_01、part_02这样的格式"
if [[ -z "$ROOM_DIR" ]]; then
  [[ "$PART_NAME" == "part_01" ]] || \
    die "测试part_02及以后时必须设置已有的ROOM_DIR"
  ROOM_DIR="$DATASET_ROOT/${SCENE_PREFIX}_${RUN_STAMP}"
fi
mkdir -p "$ROOM_DIR"
if [[ "$PART_NAME" != "part_01" ]]; then
  [[ -n "$PREVIOUS_PART" ]] || die "后续段必须设置PREVIOUS_PART"
  [[ -f "$ROOM_DIR/$PREVIOUS_PART/valid_frame_ids.txt" ]] || \
    die "上一段不存在或未完成：$ROOM_DIR/$PREVIOUS_PART"
fi
SCENE_DIR="$ROOM_DIR/$PART_NAME"
[[ ! -e "$SCENE_DIR" ]] || die "目标分段目录已存在：$SCENE_DIR"
LOG_DIR="$DATASET_ROOT/logs_${RUN_STAMP}"
RTABMAP_DB="$DATASET_ROOT/rtabmap_rotation_${RUN_STAMP}.db"
mkdir -p "$LOG_DIR"
printf '%s\n' "$ROOM_DIR" > "$DATASET_ROOT/latest_rotation_room.txt"

log "房间目录：$ROOM_DIR"
log "测试分段：$PART_NAME"
log "输出目录：$SCENE_DIR"
log "按Ctrl+C或硬件急停可随时中止"

setsid ros2 run deptrum-ros-driver-aurora930 aurora930_node \
  --ros-args -p align_mode:=true \
  >"$LOG_DIR/camera.log" 2>&1 &
CAMERA_PID=$!
wait_for_topic /rgb/image_raw 30 || die "RGB话题启动失败，查看 $LOG_DIR/camera.log"
wait_for_topic /depth/image_raw 30 || die "深度话题启动失败，查看 $LOG_DIR/camera.log"
ALIGN_VALUE="$(ros2 param get /aurora align_mode 2>/dev/null || true)"
[[ "$ALIGN_VALUE" == *"True"* ]] || \
  die "相机align_mode未生效：$ALIGN_VALUE"

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

wait_for_topic /rtabmap/odom 60 || die "RTAB-Map里程计无输出，查看 $LOG_DIR/rtabmap.log"
wait_for_tf odom rgb_camera_link 30 || die "缺少 odom <- rgb_camera_link TF"
wait_for_tf map rgb_camera_link 30 || die "缺少 map <- rgb_camera_link TF"

log "检查通过，即将执行：逆时针${ANGLE_MODE}°，再顺时针${ANGLE_MODE}°返回"
log "请确认四周无人、无障碍，手放在硬件急停附近"
read -r -p "确认安全后输入 MOVE 并回车：" MOTION_CONFIRMATION
[[ "$MOTION_CONFIRMATION" == "MOVE" ]] || die "用户取消运动"
sleep 3

MOTION_STARTED=1
python3 "$SCRIPT_ROOT/capture_rotation_scene.py" \
  --output "$SCENE_DIR" \
  --part-name "$PART_NAME" \
  --previous-part "$PREVIOUS_PART" \
  --overlap-description "$OVERLAP_DESCRIPTION" \
  --rgb-topic /rgb/image_raw \
  --depth-topic /depth/image_raw \
  --camera-info-topic /rgb/camera_info_synced \
  --tf-world-frame map \
  --world-frame camera_optical_initial \
  --control-frame odom \
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
  --pose-queue-timeout 1.0 \
  --rotation-speed-units 1 \
  --rotation-radians "$ROTATION_RADIANS" \
  --ccw-yaw-sign -1 \
  --warmup-seconds 5 \
  --settle-seconds 2 \
  --pose-stale-timeout 0.75 \
  --max-leg-seconds 120 \
  --i-understand-robot-will-move
MOTION_STARTED=0

RGB_COUNT="$(find "$SCENE_DIR/rgb" -maxdepth 1 -type f -name '*.png' | wc -l)"
DEPTH_COUNT="$(find "$SCENE_DIR/depth" -maxdepth 1 -type f -name '*.png' | wc -l)"
POSE_COUNT="$(find "$SCENE_DIR/pose" -maxdepth 1 -type f -name '*.txt' | wc -l)"
METADATA_COUNT="$(tail -n +2 "$SCENE_DIR/frames.csv" | wc -l)"
VALID_COUNT="$(grep -c '^[0-9]' "$SCENE_DIR/valid_frame_ids.txt")"
[[ "$RGB_COUNT" -eq "$DEPTH_COUNT" && "$RGB_COUNT" -eq "$POSE_COUNT" ]] || \
  die "文件数量不一致：RGB=$RGB_COUNT depth=$DEPTH_COUNT pose=$POSE_COUNT"
[[ "$RGB_COUNT" -eq "$METADATA_COUNT" && "$RGB_COUNT" -eq "$VALID_COUNT" ]] || \
  die "帧文件、frames.csv和valid_frame_ids.txt数量不一致"
[[ "$RGB_COUNT" -gt 0 ]] || die "没有保存任何数据帧"
grep -q '"world_frame": "camera_optical_initial"' "$SCENE_DIR/camera.json" || \
  die "camera.json中的world_frame不正确"
grep -q '"distortion_model": "plumb_bob"' "$SCENE_DIR/camera.json" || \
  die "camera.json中没有保存plumb_bob畸变模型"
grep -q '"pose_convention": "T_world_camera_optical"' \
  "$SCENE_DIR/camera.json" || die "camera.json缺少正确的pose_convention"

log "采集成功：RGB=$RGB_COUNT depth=$DEPTH_COUNT pose=$POSE_COUNT"
log "camera.json：$SCENE_DIR/camera.json"
