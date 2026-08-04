#!/usr/bin/env python3
"""Republish CameraInfo with the current RGB image timestamp."""

import argparse
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix a camera driver whose CameraInfo and Image use different clocks."
    )
    parser.add_argument("--image-topic", default="/rgb/image_raw")
    parser.add_argument("--input-info-topic", default="/rgb/camera_info")
    parser.add_argument("--output-info-topic", default="/rgb/camera_info_synced")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


class CameraInfoStampSynchronizer(Node):
    def __init__(self, args):
        super().__init__("rgb_camera_info_stamp_synchronizer")
        self.args = args
        self.latest_info = None
        self.published = 0
        self.publisher = self.create_publisher(
            CameraInfo, args.output_info_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            args.input_info_topic,
            self._info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            args.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Republishing {args.input_info_topic} as {args.output_info_topic} "
            f"with timestamps from {args.image_topic}"
        )

    def _info_callback(self, msg: CameraInfo):
        self.latest_info = msg

    def _image_callback(self, image: Image):
        if self.latest_info is None:
            return
        info = copy.deepcopy(self.latest_info)
        info.header.stamp = image.header.stamp
        info.header.frame_id = image.header.frame_id
        self.publisher.publish(info)
        self.published += 1
        if self.published == 1:
            self.get_logger().info("Published first synchronized CameraInfo")


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = CameraInfoStampSynchronizer(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

