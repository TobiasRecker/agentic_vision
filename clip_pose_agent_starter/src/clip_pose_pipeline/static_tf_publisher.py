from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .capture_geometry import quaternion_from_matrix
from .hybrid_io import read_yaml


class ClipPoseStaticTfPublisher(Node):
    def __init__(self) -> None:
        super().__init__("clip_pose_static_tf")
        self.declare_parameter("pose_file", "")
        self.declare_parameter("child_frame", "clip_task")
        pose_file = Path(str(self.get_parameter("pose_file").value)).expanduser().resolve()
        if not pose_file.is_file():
            raise FileNotFoundError(f"accepted clip pose file not found: {pose_file}")
        data = read_yaml(pose_file)
        if not bool(data.get("accepted", False)):
            raise ValueError(f"refusing to publish unaccepted clip pose: {pose_file}")
        T = np.asarray(data["T_base_clip"], dtype=np.float64).reshape(4, 4)
        parent = str(data.get("parent_frame", "mur620d/UR10_r/base_link"))
        child = str(self.get_parameter("child_frame").value or data.get("child_frame", "clip_task"))
        q = quaternion_from_matrix(T[:3, :3])
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = float(T[0, 3])
        message.transform.translation.y = float(T[1, 3])
        message.transform.translation.z = float(T[2, 3])
        message.transform.rotation.x = float(q[0])
        message.transform.rotation.y = float(q[1])
        message.transform.rotation.z = float(q[2])
        message.transform.rotation.w = float(q[3])
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(message)
        self.get_logger().info(f"Published accepted static TF {parent}->{child} from {pose_file}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ClipPoseStaticTfPublisher()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
