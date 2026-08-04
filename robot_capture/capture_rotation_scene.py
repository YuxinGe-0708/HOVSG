#!/usr/bin/env python3
"""Capture a scene while rotating the robot 90 degrees CCW and back CW."""

import argparse
import math
import os
from pathlib import Path
import termios
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from tf2_ros import TransformException

from collect_aurora_scene import SceneCollector, quaternion_matrix


STOP_DIRECTION = 0x00
LEFT_DIRECTION = 0x0C
RIGHT_DIRECTION = 0x03


def command_packet(direction: int, speed: int) -> bytes:
    if not 0 <= speed <= 0xFFFF:
        raise ValueError("speed must fit in uint16")
    motor_speed = speed.to_bytes(2, byteorder="big")
    if speed == 0:
        direction = STOP_DIRECTION
    return (
        bytes((0xA5, 0x5A, 0x00, direction))
        + motor_speed * 4
        + bytes((0x00, 0x00, 0x00))
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-closed-loop CCW/CW rotation with synchronized RGB-D capture."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--part-name", required=True)
    parser.add_argument("--previous-part", default="")
    parser.add_argument("--overlap-description", default="")
    parser.add_argument("--rgb-topic", default="/rgb/image_raw")
    parser.add_argument("--depth-topic", default="/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/rgb/camera_info")
    parser.add_argument("--tf-world-frame", default="map")
    parser.add_argument(
        "--world-frame",
        choices=("ros_z_up", "hov_y_up", "camera_optical_initial"),
        default="ros_z_up",
    )
    parser.add_argument("--control-frame", default="odom")
    parser.add_argument("--camera-frame", default="rgb_camera_link")
    parser.add_argument("--camera-frame-is-optical", action="store_true")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--rgb-depth-aligned", action="store_true")
    parser.add_argument("--sync-slop", type=float, default=0.02)
    parser.add_argument("--max-rgb-depth-skew-ms", type=float, default=20.0)
    parser.add_argument("--max-pose-skew-ms", type=float, default=50.0)
    parser.add_argument("--tf-timeout", type=float, default=0.02)
    parser.add_argument("--pose-queue-timeout", type=float, default=1.0)
    parser.add_argument("--max-pending-frames", type=int, default=60)
    parser.add_argument("--image-rectified", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--serial-device", default="/dev/ttyUSB0")
    parser.add_argument(
        "--rotation-speed-units",
        type=int,
        default=1,
        help="Motor speed units from the chassis protocol; keep at 1 for the first test.",
    )
    parser.add_argument("--rotation-radians", type=float, default=math.pi / 2.0)
    parser.add_argument("--angle-tolerance", type=float, default=0.03)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--pose-wait-timeout", type=float, default=30.0)
    parser.add_argument("--pose-stale-timeout", type=float, default=0.75)
    parser.add_argument("--max-leg-seconds", type=float, default=120.0)
    parser.add_argument("--max-pose-step", type=float, default=0.35)
    parser.add_argument("--min-frames-before-motion", type=int, default=10)
    parser.add_argument(
        "--ccw-yaw-sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="Sign reported by control TF during physical CCW motion; this robot reports -1.",
    )
    parser.add_argument(
        "--i-understand-robot-will-move",
        action="store_true",
        help="Required safety acknowledgement before opening the serial controller.",
    )
    args, ros_args = parser.parse_known_args()
    if not args.i_understand_robot_will_move:
        parser.error("--i-understand-robot-will-move is required")
    if not args.rgb_depth_aligned:
        parser.error("--rgb-depth-aligned is required for this dataset")
    if not 1 <= args.rotation_speed_units <= 3:
        parser.error("--rotation-speed-units must be 1, 2, or 3")
    if args.rotation_radians <= 0 or args.angle_tolerance <= 0:
        parser.error("rotation angle and tolerance must be positive")
    if args.angle_tolerance >= args.rotation_radians:
        parser.error("--angle-tolerance must be smaller than the rotation angle")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.pose_queue_timeout <= 0 or args.max_pending_frames <= 0:
        parser.error("pose queue limits must be positive")
    if (
        args.sync_slop <= 0
        or args.max_rgb_depth_skew_ms <= 0
        or args.max_pose_skew_ms <= 0
        or args.tf_timeout < 0
    ):
        parser.error("RGB-D synchronization limits must be positive")
    if not args.camera_frame_is_optical:
        parser.error("--camera-frame-is-optical is required")
    if not args.part_name or "/" in args.part_name or "\\" in args.part_name:
        parser.error("--part-name must be one directory name such as part_01")
    if args.output.name != args.part_name:
        parser.error("--output must end with the same name as --part-name")
    args.max_frames = 0
    return args, ros_args


class RotationSceneCollector(SceneCollector):
    def __init__(self, args):
        super().__init__(args)
        self.motion_args = args
        self.failed = False
        self.phase = "WAIT_POSE"
        self.phase_started = time.monotonic()
        self.deadline = self.phase_started + args.pose_wait_timeout
        self.last_control_rotation = None
        self.accumulated_yaw = 0.0
        self.last_tf_stamp_ns = None
        self.last_tf_update_time = None
        self.serial_fd = None
        self.stop_packet = command_packet(STOP_DIRECTION, 0)
        self.left_packet = command_packet(LEFT_DIRECTION, args.rotation_speed_units)
        self.right_packet = command_packet(RIGHT_DIRECTION, args.rotation_speed_units)
        try:
            self.serial_fd = os.open(args.serial_device, os.O_RDWR | os.O_NOCTTY)
            self._configure_serial()
            self._send_stop(repeat=3)
        except Exception:
            if self.serial_fd is not None:
                os.close(self.serial_fd)
                self.serial_fd = None
            raise
        self.motion_timer = self.create_timer(0.02, self._motion_tick)
        self.get_logger().warning(
            "Base motion is armed but stopped. Waiting for valid dataset/control TF and "
            f"{args.min_frames_before_motion} saved frames. Keep the emergency stop ready."
        )

    def _configure_serial(self):
        attrs = termios.tcgetattr(self.serial_fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.B115200 | termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.serial_fd, termios.TCSANOW, attrs)
        termios.tcflush(self.serial_fd, termios.TCIOFLUSH)

    def _write_packet(self, packet: bytes):
        if self.serial_fd is None:
            raise RuntimeError("Serial controller is closed")
        written = os.write(self.serial_fd, packet)
        if written != len(packet):
            raise RuntimeError(f"Short serial write: {written}/{len(packet)} bytes")

    def _send_stop(self, repeat: int = 1):
        if self.serial_fd is None:
            return
        for _ in range(repeat):
            try:
                self._write_packet(self.stop_packet)
            except Exception as error:
                self.get_logger().error(f"Failed to send stop packet: {error}")
            if repeat > 1:
                time.sleep(0.02)

    def _abort(self, message: str):
        if self.done:
            return
        self.failed = True
        self._send_stop(repeat=3)
        self.get_logger().error(f"MOTION ABORTED: {message}")
        self.done = True

    def _latest_control_pose(self):
        stamped = self.tf_buffer.lookup_transform(
            self.motion_args.control_frame,
            self.motion_args.camera_frame,
            Time(),
            timeout=Duration(seconds=0.0),
        )
        stamp_ns = (
            int(stamped.header.stamp.sec) * 1_000_000_000
            + int(stamped.header.stamp.nanosec)
        )
        rotation = stamped.transform.rotation
        return stamp_ns, quaternion_matrix(
            rotation.x, rotation.y, rotation.z, rotation.w
        )

    def _update_rotation(self, stamp_ns: int, rotation, now: float):
        if stamp_ns == self.last_tf_stamp_ns:
            return
        if self.last_control_rotation is not None:
            relative = rotation @ self.last_control_rotation.T
            delta = math.atan2(relative[1, 0], relative[0, 0])
            if abs(delta) > self.motion_args.max_pose_step:
                raise RuntimeError(
                    f"TF yaw jumped by {delta:.3f} rad; localization is not stable"
                )
            if self.phase in ("ROTATE_CCW", "ROTATE_CW"):
                self.accumulated_yaw += delta
        self.last_control_rotation = rotation
        self.last_tf_stamp_ns = stamp_ns
        self.last_tf_update_time = now

    def _begin_phase(self, phase: str, now: float):
        self.phase = phase
        self.phase_started = now
        self.accumulated_yaw = 0.0
        if phase in ("WARMUP", "SETTLE_CCW", "FINAL_SETTLE"):
            seconds = (
                self.motion_args.warmup_seconds
                if phase == "WARMUP"
                else self.motion_args.settle_seconds
            )
            self.deadline = now + seconds
        elif phase in ("ROTATE_CCW", "ROTATE_CW"):
            self.deadline = now + self.motion_args.max_leg_seconds
        self.get_logger().warning(f"Motion phase: {phase}")

    def _motion_tick(self):
        if self.done:
            self._send_stop()
            return
        now = time.monotonic()
        try:
            stamp_ns, rotation = self._latest_control_pose()
            self._update_rotation(stamp_ns, rotation, now)
        except TransformException as error:
            self._send_stop()
            if self.phase == "WAIT_POSE" and now < self.deadline:
                return
            self._abort(
                f"Missing TF {self.motion_args.control_frame} <- "
                f"{self.motion_args.camera_frame}: {error}"
            )
            return
        except Exception as error:
            self._abort(str(error))
            return

        if self.phase == "WAIT_POSE":
            self._send_stop()
            if self.saved_this_run >= self.motion_args.min_frames_before_motion:
                self._begin_phase("WARMUP", now)
            elif now >= self.deadline:
                self._abort(
                    "Dataset TF did not produce enough saved frames before the pose timeout"
                )
            return

        if self.last_tf_update_time is None or (
            now - self.last_tf_update_time > self.motion_args.pose_stale_timeout
        ):
            self._abort("Control TF became stale")
            return

        if self.phase == "WARMUP":
            self._send_stop()
            if now >= self.deadline:
                self._begin_phase("ROTATE_CCW", now)
            return

        if self.phase == "ROTATE_CCW":
            if now >= self.deadline:
                self._abort("CCW rotation exceeded --max-leg-seconds")
                return
            progress = self.motion_args.ccw_yaw_sign * self.accumulated_yaw
            if progress < -self.motion_args.angle_tolerance:
                self._abort("Pose yaw moved clockwise while commanding CCW")
                return
            if progress >= (
                self.motion_args.rotation_radians - self.motion_args.angle_tolerance
            ):
                self._send_stop(repeat=3)
                self._begin_phase("SETTLE_CCW", now)
            else:
                self._write_packet(self.left_packet)
            return

        if self.phase == "SETTLE_CCW":
            self._send_stop()
            if now >= self.deadline:
                self._begin_phase("ROTATE_CW", now)
            return

        if self.phase == "ROTATE_CW":
            if now >= self.deadline:
                self._abort("CW rotation exceeded --max-leg-seconds")
                return
            progress = -self.motion_args.ccw_yaw_sign * self.accumulated_yaw
            if progress < -self.motion_args.angle_tolerance:
                self._abort("Pose yaw moved counter-clockwise while commanding CW")
                return
            if progress >= (
                self.motion_args.rotation_radians - self.motion_args.angle_tolerance
            ):
                self._send_stop(repeat=3)
                self._begin_phase("FINAL_SETTLE", now)
            else:
                self._write_packet(self.right_packet)
            return

        if self.phase == "FINAL_SETTLE":
            self._send_stop()
            if now >= self.deadline:
                self.stop_accepting_frames()
                self.phase = "DRAIN_DATA"
                self.deadline = now + self.motion_args.pose_queue_timeout + 1.0
                self.get_logger().warning(
                    f"Motion phase: DRAIN_DATA ({self.pending_frame_count} pending frame(s))"
                )
            return

        if self.phase == "DRAIN_DATA":
            self._send_stop()
            if self.pending_frame_count == 0 or now >= self.deadline:
                self.get_logger().info(
                    "Rotation sequence complete; scene capture finished"
                )
                self.done = True

    def destroy_node(self):
        self._send_stop(repeat=5)
        if self.serial_fd is not None:
            os.close(self.serial_fd)
            self.serial_fd = None
        return super().destroy_node()


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = None
    executor = None
    exit_code = 0
    try:
        node = RotationSceneCollector(args)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.02)
        if node.failed:
            exit_code = 2
    except KeyboardInterrupt:
        if node is not None:
            node._send_stop(repeat=5)
            node.get_logger().warning("Interrupted by user; base stop sent")
        exit_code = 130
    finally:
        if executor is not None and node is not None:
            executor.remove_node(node)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
