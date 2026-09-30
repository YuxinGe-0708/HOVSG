#!/usr/bin/env bash
set -Eeuo pipefail

# One-command handheld RGB-D + IMU capture for an Orbbec Gemini 336L.
#
# Output layout:
#   $ROOM_DIR/
#     capture.json
#     camera_template.json
#     rtabmap_part_01.db
#     logs/part_01_YYYYmmdd_HHMMSS/*.log
#     part_01/{camera.json,frames.csv,valid_frame_ids.txt,rgb,depth,pose}

DATASET_ROOT="${DATASET_ROOT:-/home/creasen/dataset}"
ROOM_DIR="${ROOM_DIR:-}"
PART_NAME="${PART_NAME:-part_01}"
PREVIOUS_PART="${PREVIOUS_PART:-}"
OVERLAP_DESCRIPTION="${OVERLAP_DESCRIPTION:-}"
CAPTURE_FPS="${CAPTURE_FPS:-5}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ORBBEC_SETUP="${ORBBEC_SETUP:-/home/creasen/orbbec_main_ws/install/setup.bash}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${COLLECTOR:-$SCRIPT_DIR/collect_aurora_scene.py}"

CAMERA_PID=""
IMU_TF_PID=""
IMU_FILTER_PID=""
RTABMAP_PID=""
SCENE_DIR=""
LOG_DIR=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

stop_group() {
  local pid="${1:-}"
  local attempts="${2:-50}"
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for ((i = 0; i < attempts; i++)); do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 0.2
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  # Stop RTAB-Map first so that its database is flushed while camera TF exists.
  stop_group "$RTABMAP_PID" 100
  stop_group "$IMU_FILTER_PID"
  stop_group "$IMU_TF_PID"
  stop_group "$CAMERA_PID"
  [[ -n "$SCENE_DIR" ]] && log "Capture directory: $SCENE_DIR"
  [[ -n "$LOG_DIR" ]] && log "Log directory: $LOG_DIR"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM HUP

wait_for_topic() {
  local topic="$1"
  local timeout_seconds="$2"
  local start=$SECONDS
  while ((SECONDS - start < timeout_seconds)); do
    if timeout 4 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
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
  local output=""
  while ((SECONDS - start < timeout_seconds)); do
    output="$(
      timeout 4 ros2 run tf2_ros tf2_echo "$target" "$source" 2>/dev/null || true
    )"
    if grep -q 'Translation:' <<<"$output"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

count_files() {
  local directory="$1"
  local pattern="$2"
  find "$directory" -maxdepth 1 -type f -name "$pattern" | wc -l
}

[[ -r "$ROS_SETUP" ]] || die "Missing ROS setup: $ROS_SETUP"
[[ -r "$ORBBEC_SETUP" ]] || die "Missing Orbbec workspace setup: $ORBBEC_SETUP"
[[ -r "$COLLECTOR" ]] || die "Missing collector: $COLLECTOR"
[[ "$PART_NAME" =~ ^part_[0-9]+$ ]] || \
  die "PART_NAME must look like part_01 or part_02"

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$ORBBEC_SETUP"
set -u
export ROS_DOMAIN_ID

for command_name in ros2 python3 timeout setsid find; do
  command -v "$command_name" >/dev/null || die "Missing command: $command_name"
done

if ros2 node list 2>/dev/null | grep -Eq \
  '^/camera/|^/rtabmap/|imu_filter|static_transform|aurora_scene_collector'; then
  die "Camera, RTAB-Map, IMU filter, static TF, or collector node is already running"
fi

mkdir -p "$DATASET_ROOT"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$ROOM_DIR" ]]; then
  [[ "$PART_NAME" == "part_01" ]] || \
    die "ROOM_DIR is required when capturing part_02 or later"
  ROOM_DIR="$DATASET_ROOT/room_orbbec_${RUN_STAMP}"
fi
mkdir -p "$ROOM_DIR"

if [[ "$PART_NAME" != "part_01" ]]; then
  [[ -n "$PREVIOUS_PART" ]] || \
    die "PREVIOUS_PART is required when capturing part_02 or later"
  [[ -f "$ROOM_DIR/$PREVIOUS_PART/valid_frame_ids.txt" ]] || \
    die "Previous part is missing or not finalized: $ROOM_DIR/$PREVIOUS_PART"
fi

SCENE_DIR="$ROOM_DIR/$PART_NAME"
[[ ! -e "$SCENE_DIR" ]] || die "Capture directory already exists: $SCENE_DIR"
LOG_DIR="$ROOM_DIR/logs/${PART_NAME}_${RUN_STAMP}"
RTABMAP_DB="$ROOM_DIR/rtabmap_${PART_NAME}.db"
mkdir -p "$LOG_DIR"

printf '%s\n' "$ROOM_DIR" >"$DATASET_ROOT/latest_orbbec_room.txt"
printf '%s\n' "$SCENE_DIR" >"$DATASET_ROOT/latest_orbbec_part.txt"

log "Room directory: $ROOM_DIR"
log "Capture part: $SCENE_DIR"
log "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

setsid ros2 launch orbbec_camera gemini_330_series.launch.py \
  serial_number:=CP84B410007Y \
  depth_registration:=true \
  enable_point_cloud:=false \
  enable_accel:=true \
  enable_gyro:=true \
  enable_sync_output_accel_gyro:=true \
  publish_tf:=true \
  >"$LOG_DIR/camera.log" 2>&1 &
CAMERA_PID=$!

wait_for_topic /camera/color/image_raw 45 || \
  die "No RGB stream; inspect $LOG_DIR/camera.log"
wait_for_topic /camera/depth/image_raw 45 || \
  die "No depth stream; inspect $LOG_DIR/camera.log"
wait_for_topic /camera/color/camera_info 20 || \
  die "No CameraInfo stream; inspect $LOG_DIR/camera.log"
wait_for_topic /camera/gyro_accel/sample 20 || \
  die "No combined IMU stream; inspect $LOG_DIR/camera.log"

setsid ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 \
  --qx 0 --qy 0 --qz 0 --qw 1 \
  --frame-id camera_accel_optical_frame \
  --child-frame-id camera_accel_gyro_optical_frame \
  >"$LOG_DIR/imu_static_tf.log" 2>&1 &
IMU_TF_PID=$!

setsid ros2 run imu_filter_madgwick imu_filter_madgwick_node \
  --ros-args \
  -p use_mag:=false \
  -p publish_tf:=false \
  -p world_frame:=enu \
  -p orientation_stddev:=0.05 \
  -r imu/data_raw:=/camera/gyro_accel/sample \
  -r imu/data:=/camera/imu_filtered \
  >"$LOG_DIR/imu_filter.log" 2>&1 &
IMU_FILTER_PID=$!

wait_for_topic /camera/imu_filtered 30 || \
  die "No filtered IMU; inspect $LOG_DIR/imu_filter.log"
wait_for_tf camera_link camera_color_optical_frame 30 || \
  die "Missing camera_link <- camera_color_optical_frame TF"
wait_for_tf camera_link camera_accel_gyro_optical_frame 30 || \
  die "Missing camera_link <- camera_accel_gyro_optical_frame TF"

setsid ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  imu_topic:=/camera/imu_filtered \
  frame_id:=camera_link \
  odom_frame_id:=odom \
  map_frame_id:=map \
  publish_tf_map:=true \
  rgbd_sync:=true \
  approx_sync:=true \
  approx_sync_max_interval:=0.02 \
  queue_size:=50 \
  qos:=2 \
  wait_imu_to_init:=true \
  odom_args:="--Odom/ImageDecimation 2" \
  rviz:=false \
  rtabmap_viz:=false \
  database_path:="$RTABMAP_DB" \
  >"$LOG_DIR/rtabmap.log" 2>&1 &
RTABMAP_PID=$!

wait_for_topic /rtabmap/odom 60 || \
  die "No RTAB-Map odometry; inspect $LOG_DIR/rtabmap.log"
wait_for_tf odom camera_link 60 || \
  die "RTAB-Map odometry is not valid; inspect $LOG_DIR/rtabmap.log"
wait_for_tf map camera_color_optical_frame 60 || \
  die "RTAB-Map map pose is not valid; inspect $LOG_DIR/rtabmap.log"

if grep -Eq \
  "Could not transform IMU|didn't receive IMU|IMU.*failed" \
  "$LOG_DIR/rtabmap.log"; then
  die "RTAB-Map reported an IMU error; inspect $LOG_DIR/rtabmap.log"
fi

log "All checks passed. Keep the camera still for 5 seconds."
sleep 5
log "Capture started. Move slowly; return to the start and hold still before Ctrl+C."

python3 "$COLLECTOR" \
  --output "$SCENE_DIR" \
  --part-name "$PART_NAME" \
  --previous-part "$PREVIOUS_PART" \
  --overlap-description "$OVERLAP_DESCRIPTION" \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --tf-world-frame map \
  --world-frame ros_z_up \
  --camera-frame camera_color_optical_frame \
  --camera-frame-is-optical \
  --depth-scale 0.001 \
  --min-depth 0.15 \
  --max-depth 5.0 \
  --rgb-depth-aligned \
  --sync-slop 0.02 \
  --max-rgb-depth-skew-ms 20 \
  --max-pose-skew-ms 50 \
  --tf-timeout 0.02 \
  --fps "$CAPTURE_FPS" \
  --pose-queue-timeout 1.0

RGB_COUNT="$(count_files "$SCENE_DIR/rgb" '*.png')"
DEPTH_COUNT="$(count_files "$SCENE_DIR/depth" '*.png')"
POSE_COUNT="$(count_files "$SCENE_DIR/pose" '*.txt')"
METADATA_COUNT="$(tail -n +2 "$SCENE_DIR/frames.csv" | wc -l)"
VALID_COUNT="$(grep -c '^[0-9]' "$SCENE_DIR/valid_frame_ids.txt" || true)"

log "Saved counts: RGB=$RGB_COUNT depth=$DEPTH_COUNT pose=$POSE_COUNT metadata=$METADATA_COUNT valid=$VALID_COUNT"
[[ "$RGB_COUNT" -gt 0 ]] || die "No frames were saved"
[[ "$RGB_COUNT" -eq "$DEPTH_COUNT" && "$RGB_COUNT" -eq "$POSE_COUNT" ]] || \
  die "RGB, depth, and pose counts do not match"
[[ "$RGB_COUNT" -eq "$METADATA_COUNT" && "$RGB_COUNT" -eq "$VALID_COUNT" ]] || \
  die "Frame files, frames.csv, and valid_frame_ids.txt do not match"
grep -q '"pose_convention": "T_world_camera_optical"' \
  "$SCENE_DIR/camera.json" || die "camera.json has the wrong pose convention"
grep -q '"world_frame": "ros_z_up"' \
  "$SCENE_DIR/camera.json" || die "camera.json has the wrong world convention"

QUALITY_ZERO_COUNT="$(grep -c 'quality=0' "$LOG_DIR/rtabmap.log" || true)"
log "RTAB-Map quality=0 log lines: $QUALITY_ZERO_COUNT"
log "Capture finalized successfully: $SCENE_DIR"
log "RTAB-Map database retained for trajectory review: $RTABMAP_DB"
