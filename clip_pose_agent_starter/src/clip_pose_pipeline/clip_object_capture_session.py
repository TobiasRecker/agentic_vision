#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .capture_geometry import (
    center_camera_target,
    estimate_anchor_from_xyz_image,
    generate_spiral_hemisphere_targets,
    invert_transform,
    quaternion_from_matrix,
    slew_vector,
    target_motion_metrics,
    transform_from_translation_quaternion,
    transform_point,
)

try:
    from mur_control.action import JparseMove
except ImportError:  # pragma: no cover - lets non-robot dev shells import the module.
    JparseMove = None


@dataclass
class Observation:
    image: np.ndarray
    image_msg: Image | CompressedImage
    camera_info: CameraInfo
    cloud_msg: PointCloud2 | None
    T_base_tcp: np.ndarray
    T_base_camera: np.ndarray
    T_tcp_camera: np.ndarray
    camera_frame: str


class ClipObjectCaptureSession(Node):
    def __init__(self) -> None:
        super().__init__("clip_object_capture_session")

        self.declare_parameter("image_topic", "/oak/rgb/image_raw/compressed")
        self.declare_parameter("image_compressed", True)
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")
        self.declare_parameter("pointcloud_topic", "/oak/rgbd/points")
        self.declare_parameter("output_root", "~/clip_pose_sessions")
        self.declare_parameter("session_name", "")
        self.declare_parameter("sample_prefix", "sample")
        self.declare_parameter("robot_base_frame", "mur620/UR10_r/base_link")
        self.declare_parameter("robot_tcp_frame", "mur620/UR10_r/tool0")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("planning_frame", "mur620/UR10_r/base_link")
        self.declare_parameter("action_name", "/mur620/jparse_move_r")
        self.declare_parameter("move_enabled", False)
        self.declare_parameter("keyboard_jog_enabled", False)
        self.declare_parameter("jog_twist_topic", "/mur620/jparse_velocity_controller_r/twist_cmd")
        self.declare_parameter("jog_frame", "UR10_r/base_link")
        self.declare_parameter("jog_linear_velocity", 0.03)
        self.declare_parameter("jog_angular_velocity", 0.25)
        self.declare_parameter("jog_linear_acceleration", 0.12)
        self.declare_parameter("jog_angular_acceleration", 1.0)
        self.declare_parameter("jog_hold_timeout", 0.8)
        self.declare_parameter("display_max_side", 1600)
        self.declare_parameter("click_window_radius", 4)
        self.declare_parameter("click_min_points", 8)
        self.declare_parameter("click_max_mad_m", 0.02)
        self.declare_parameter("samples", 18)
        self.declare_parameter("sphere_radius_m", 0.0)
        self.declare_parameter("sphere_polar_span_deg", 50.0)
        self.declare_parameter("sphere_spiral_turns", 1.25)
        self.declare_parameter("camera_look_axis", "plus_z")
        self.declare_parameter("center_camera_xy_only", True)
        self.declare_parameter("target_min_camera_delta_m", 0.04)
        self.declare_parameter("target_max_tcp_delta_m", 0.25)
        self.declare_parameter("target_max_camera_delta_m", 0.30)
        self.declare_parameter("target_max_rotation_deg", 35.0)
        self.declare_parameter("max_linear_velocity", 0.025)
        self.declare_parameter("max_angular_velocity", 0.10)
        self.declare_parameter("move_timeout", 30.0)
        self.declare_parameter("save_anchor_approx_tf", True)
        self.declare_parameter("log_key_codes", True)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.action_client = (
            ActionClient(self, JparseMove, self.param_str("action_name"))
            if JparseMove is not None
            else None
        )
        self.jog_pub = self.create_publisher(TwistStamped, self.param_str("jog_twist_topic"), 10)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.latest_image_msg: Image | CompressedImage | None = None
        self.latest_camera_info: CameraInfo | None = None
        self.latest_cloud_msg: PointCloud2 | None = None
        self.image_compressed = self.param_bool("image_compressed") or self.param_str("image_topic").endswith(
            "/compressed"
        )
        image_msg_type = CompressedImage if self.image_compressed else Image
        self.create_subscription(image_msg_type, self.param_str("image_topic"), self.on_image, qos)
        self.create_subscription(CameraInfo, self.param_str("camera_info_topic"), self.on_camera_info, qos)
        self.create_subscription(PointCloud2, self.param_str("pointcloud_topic"), self.on_cloud, qos)

        session_name = self.param_str("session_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(os.path.expanduser(self.param_str("output_root")))
        self.output_dir = output_root / session_name
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.window_name = "clip_object_capture_session"
        self.display_scale = 1.0
        self.last_status = "waiting for ROS data"
        self.last_tf_error = ""
        self.rotation_jog_mode = False
        self.last_key_text = "none"
        self.last_jog_text = "jog idle"
        self.jog_target_linear = np.zeros(3, dtype=np.float64)
        self.jog_target_angular = np.zeros(3, dtype=np.float64)
        self.jog_current_linear = np.zeros(3, dtype=np.float64)
        self.jog_current_angular = np.zeros(3, dtype=np.float64)
        self.last_jog_key_time = 0.0
        self.last_jog_update_time = time.monotonic()

        self.initial_T_base_tcp: np.ndarray | None = None
        self.anchor: dict[str, Any] | None = None
        self.gui_target: np.ndarray | None = None
        self.gui_target_source = ""
        self.target_cache: list[np.ndarray] = []
        self.target_cursor = 0
        self.target_history: list[dict[str, Any]] = []
        self.camera_pose_records: list[dict[str, Any]] = []
        self.anchor_sample_records: list[dict[str, Any]] = []
        self.sample_index = 1

        self.write_metadata()
        self.get_logger().info(f"Writing clip capture session to {self.output_dir}")
        self.get_logger().info(
            "GUI keys: click=set anchor, z=center, n=next target, g=go, "
            "c=save, b=back, arrows/PgUp/PgDn=jog, m=toggle rotation jog, .=stop, q=quit"
        )

    def param_str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def param_bool(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def run_gui_session(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                cv2.imshow(self.window_name, self.render_gui_frame())
                self.update_jog_output()
                key = cv2.waitKeyEx(20)
                if key != -1 and not self.handle_key(key):
                    break
        finally:
            self.stop_jog(force=True)
            cv2.destroyWindow(self.window_name)

    def on_image(self, msg: Image | CompressedImage) -> None:
        self.latest_image_msg = msg

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def on_cloud(self, msg: PointCloud2) -> None:
        self.latest_cloud_msg = msg

    def render_gui_frame(self) -> np.ndarray:
        if self.latest_image_msg is None:
            view = np.zeros((520, 960, 3), dtype=np.uint8)
            self.draw_text_lines(view, ["Waiting for OAK RGB image", self.last_status])
            return view

        try:
            image = self.decode_image_msg(self.latest_image_msg)
        except Exception as exc:  # noqa: BLE001
            view = np.zeros((520, 960, 3), dtype=np.uint8)
            self.last_status = f"image decode failed: {exc}"
            self.draw_text_lines(view, [self.last_status])
            return view

        view, self.display_scale = self.resize_for_display(image)
        if self.anchor is not None:
            pixel = self.anchor["pixel_uv"]
            center = (int(round(pixel[0] * self.display_scale)), int(round(pixel[1] * self.display_scale)))
            cv2.drawMarker(view, center, (0, 255, 255), cv2.MARKER_CROSS, 42, 3, cv2.LINE_AA)
            cv2.circle(view, center, 20, (0, 180, 255), 2, cv2.LINE_AA)

        status = self.status_lines(image.shape[:2])
        self.draw_text_lines(view, status)
        return view

    def status_lines(self, image_shape: tuple[int, int]) -> list[str]:
        anchor_text = "anchor: none"
        if self.anchor is not None:
            p = self.anchor["p_base_anchor"]
            q = self.anchor["quality"]
            anchor_text = (
                f"anchor base [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
                f"mad={q['median_abs_deviation_m'] * 1000.0:.1f}mm"
            )
        cloud_text = "cloud: no"
        if self.latest_cloud_msg is not None:
            cloud_text = f"cloud: {self.latest_cloud_msg.width}x{self.latest_cloud_msg.height}"
        target_text = "target: none" if self.gui_target is None else f"target: {self.gui_target_source}, press g"
        return [
            self.last_status,
            f"image: {image_shape[1]}x{image_shape[0]}  {cloud_text}",
            anchor_text,
            target_text,
            f"samples saved: {len(self.camera_pose_records)}  output: {self.output_dir}",
            (
                f"move={'on' if self.param_bool('move_enabled') else 'off'} "
                f"jog={'on' if self.param_bool('keyboard_jog_enabled') else 'off'} "
                f"mode={'rot' if self.rotation_jog_mode else 'xyz'}"
            ),
            f"last key: {self.last_key_text}  {self.last_jog_text}",
            "click anchor | z center | n next | g go | c save | b back | . stop | q quit",
        ]

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.display_scale <= 0.0:
            return
        pixel = (float(x) / self.display_scale, float(y) / self.display_scale)
        self.set_anchor_from_click(pixel)

    def set_anchor_from_click(self, pixel_uv: tuple[float, float]) -> None:
        observation = self.current_observation(require_cloud=True)
        if observation is None:
            return
        xyz = self.pointcloud_xyz_image(observation.cloud_msg)
        if xyz is None:
            return
        cloud_frame = observation.cloud_msg.header.frame_id if observation.cloud_msg is not None else ""
        if cloud_frame and cloud_frame != observation.camera_frame:
            self.last_status = (
                f"blocked: pointcloud frame '{cloud_frame}' differs from camera frame "
                f"'{observation.camera_frame}'"
            )
            self.get_logger().warn(self.last_status)
            return
        image_h, image_w = observation.image.shape[:2]
        cloud_h, cloud_w = xyz.shape[:2]
        if (image_w, image_h) != (cloud_w, cloud_h):
            self.last_status = (
                f"blocked: image {image_w}x{image_h} and pointcloud {cloud_w}x{cloud_h} differ"
            )
            self.get_logger().warn(self.last_status)
            return

        estimate = estimate_anchor_from_xyz_image(
            xyz,
            pixel_uv,
            window_radius=int(self.get_parameter("click_window_radius").value),
            min_points=int(self.get_parameter("click_min_points").value),
            max_mad_m=float(self.get_parameter("click_max_mad_m").value),
        )
        if not estimate.valid or estimate.point_camera is None:
            self.last_status = f"anchor rejected: {estimate.reason}"
            self.get_logger().warn(f"{self.last_status}; quality={estimate.as_dict()}")
            return

        p_base_anchor = transform_point(observation.T_base_camera, estimate.point_camera)
        self.anchor = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "pixel_uv": [float(pixel_uv[0]), float(pixel_uv[1])],
            "camera_frame": observation.camera_frame,
            "base_frame": self.param_str("robot_base_frame"),
            "p_camera_anchor": estimate.point_camera.astype(float).tolist(),
            "p_base_anchor": p_base_anchor.astype(float).tolist(),
            "quality": estimate.as_dict(),
            "source": "manual_click_rgbd_pointcloud",
            "image_stamp": stamp_dict(observation.image_msg.header.stamp),
            "cloud_stamp": stamp_dict(observation.cloud_msg.header.stamp) if observation.cloud_msg is not None else None,
            "T_base_camera_at_click": matrix_to_list(observation.T_base_camera),
        }
        self.target_cache = []
        self.target_cursor = 0
        self.gui_target = None
        self.gui_target_source = ""
        self.last_status = (
            "anchor set: "
            f"base=[{p_base_anchor[0]:.3f}, {p_base_anchor[1]:.3f}, {p_base_anchor[2]:.3f}]"
        )
        self.get_logger().info(self.last_status)
        self.write_anchors()
        self.write_metadata()
        if self.param_bool("save_anchor_approx_tf"):
            self.write_anchor_approx_tf(observation.T_base_camera)

    def current_observation(self, require_cloud: bool = False) -> Observation | None:
        if self.latest_image_msg is None:
            self.last_status = f"waiting for image on {self.param_str('image_topic')}"
            return None
        if self.latest_camera_info is None:
            self.last_status = f"waiting for CameraInfo on {self.param_str('camera_info_topic')}"
            return None
        if require_cloud and self.latest_cloud_msg is None:
            self.last_status = f"waiting for PointCloud2 on {self.param_str('pointcloud_topic')}"
            return None

        try:
            image = self.decode_image_msg(self.latest_image_msg)
        except Exception as exc:  # noqa: BLE001
            self.last_status = f"image decode failed: {exc}"
            return None

        camera_frame = self.param_str("camera_frame")
        if not camera_frame:
            camera_frame = self.latest_camera_info.header.frame_id
        if not camera_frame and self.latest_cloud_msg is not None:
            camera_frame = self.latest_cloud_msg.header.frame_id
        if not camera_frame:
            self.last_status = "waiting for camera frame id"
            return None

        T_base_tcp = self.lookup_transform_matrix(self.param_str("robot_base_frame"), self.param_str("robot_tcp_frame"))
        if T_base_tcp is None:
            return None
        T_base_camera = self.lookup_transform_matrix(self.param_str("robot_base_frame"), camera_frame)
        if T_base_camera is None:
            return None
        T_tcp_camera = self.lookup_transform_matrix(self.param_str("robot_tcp_frame"), camera_frame)
        if T_tcp_camera is None:
            return None

        if self.initial_T_base_tcp is None:
            self.initial_T_base_tcp = T_base_tcp.copy()
            p = self.initial_T_base_tcp[:3, 3]
            self.get_logger().info(f"Captured initial TCP pose [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

        self.last_status = "observation ok"
        return Observation(
            image=image,
            image_msg=self.latest_image_msg,
            camera_info=self.latest_camera_info,
            cloud_msg=self.latest_cloud_msg,
            T_base_tcp=T_base_tcp,
            T_base_camera=T_base_camera,
            T_tcp_camera=T_tcp_camera,
            camera_frame=camera_frame,
        )

    def lookup_transform_matrix(self, target_frame: str, source_frame: str) -> np.ndarray | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.last_tf_error = str(exc)
            self.last_status = f"waiting for TF {target_frame}->{source_frame}: {exc}"
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return transform_from_translation_quaternion(
            [translation.x, translation.y, translation.z],
            [rotation.x, rotation.y, rotation.z, rotation.w],
        )

    def pointcloud_xyz_image(self, cloud_msg: PointCloud2 | None) -> np.ndarray | None:
        if cloud_msg is None:
            self.last_status = f"waiting for PointCloud2 on {self.param_str('pointcloud_topic')}"
            return None
        if cloud_msg.height <= 1:
            self.last_status = "blocked: PointCloud2 is not organized"
            self.get_logger().warn(self.last_status)
            return None
        try:
            points = point_cloud2.read_points_numpy(
                cloud_msg,
                field_names=["x", "y", "z"],
                skip_nans=False,
                reshape_organized_cloud=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_status = f"pointcloud decode failed: {exc}"
            self.get_logger().warn(self.last_status)
            return None
        arr = np.asarray(points)
        if arr.dtype.fields:
            arr = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            try:
                arr = arr.reshape(int(cloud_msg.height), int(cloud_msg.width), -1)
            except ValueError:
                self.last_status = f"pointcloud has unexpected shape {arr.shape}"
                self.get_logger().warn(self.last_status)
                return None
        return np.asarray(arr[:, :, :3], dtype=np.float64)

    def handle_key(self, key: int) -> bool:
        char = ascii_char(key)
        name = key_name(key) or char
        self.last_key_text = f"{key} ({name or 'unknown'})"
        if self.param_bool("log_key_codes"):
            self.get_logger().info(f"Capture GUI key: {self.last_key_text}")

        if char == "q" or key == 27:
            return False
        if char == "." or char == " ":
            self.stop_jog(force=True)
            self.gui_target = None
            self.gui_target_source = ""
            return True
        if char == "m":
            self.rotation_jog_mode = not self.rotation_jog_mode
            self.last_status = f"jog mode {'rotation' if self.rotation_jog_mode else 'translation'}"
            return True
        if char == "z":
            self.prepare_center_target()
            return True
        if char == "n":
            self.propose_next_target()
            return True
        if char == "g":
            self.go_to_gui_target()
            return True
        if char == "c":
            self.save_sample()
            return True
        if char == "b":
            self.go_to_initial_pose()
            return True

        linear, angular = self.jog_command_from_key(key)
        if linear is not None:
            self.set_jog_target(linear, angular)
        return True

    def prepare_center_target(self) -> None:
        if self.anchor is None:
            self.last_status = "center blocked: click an anchor first"
            return
        observation = self.current_observation(require_cloud=False)
        if observation is None:
            return
        anchor_base = np.asarray(self.anchor["p_base_anchor"], dtype=np.float64)
        T_base_camera_target = center_camera_target(
            observation.T_base_camera,
            anchor_base,
            xy_only=self.param_bool("center_camera_xy_only"),
            look_axis=self.param_str("camera_look_axis"),
        )
        T_base_tcp_target = T_base_camera_target @ invert_transform(observation.T_tcp_camera)
        self.set_gui_target("center", T_base_tcp_target, observation)

    def propose_next_target(self) -> None:
        if self.anchor is None:
            self.last_status = "target blocked: click an anchor first"
            return
        observation = self.current_observation(require_cloud=False)
        if observation is None:
            return
        anchor_base = np.asarray(self.anchor["p_base_anchor"], dtype=np.float64)
        if not self.target_cache or self.target_cursor >= len(self.target_cache):
            radius = float(self.get_parameter("sphere_radius_m").value)
            self.target_cache = generate_spiral_hemisphere_targets(
                anchor_base,
                observation.T_base_camera,
                sample_count=max(1, int(self.get_parameter("samples").value)),
                radius_m=radius if radius > 0.0 else None,
                polar_span_deg=float(self.get_parameter("sphere_polar_span_deg").value),
                spiral_turns=float(self.get_parameter("sphere_spiral_turns").value),
                look_axis=self.param_str("camera_look_axis"),
            )
            self.target_cursor = 0

        while self.target_cursor < len(self.target_cache):
            T_base_camera_target = self.target_cache[self.target_cursor]
            self.target_cursor += 1
            T_base_tcp_target = T_base_camera_target @ invert_transform(observation.T_tcp_camera)
            metrics = target_motion_metrics(
                observation.T_base_tcp,
                T_base_tcp_target,
                observation.T_tcp_camera,
                anchor_base,
                look_axis=self.param_str("camera_look_axis"),
            )
            if metrics["camera_delta_norm"] < float(self.get_parameter("target_min_camera_delta_m").value):
                continue
            if not self.target_is_safe(metrics):
                continue
            self.set_gui_target(f"sphere {self.target_cursor}/{len(self.target_cache)}", T_base_tcp_target, observation)
            return

        self.last_status = "no safe target left; press n again to regenerate from current pose"
        self.target_cache = []
        self.target_cursor = 0

    def set_gui_target(self, source: str, T_base_tcp_target: np.ndarray, observation: Observation) -> None:
        if self.anchor is None:
            return
        anchor_base = np.asarray(self.anchor["p_base_anchor"], dtype=np.float64)
        metrics = target_motion_metrics(
            observation.T_base_tcp,
            T_base_tcp_target,
            observation.T_tcp_camera,
            anchor_base,
            look_axis=self.param_str("camera_look_axis"),
        )
        if not self.target_is_safe(metrics):
            return
        self.gui_target = T_base_tcp_target.copy()
        self.gui_target_source = source
        self.target_history.append(
            {
                "event": "target_prepared",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": source,
                "T_base_tcp": matrix_to_list(T_base_tcp_target),
                "metrics": metrics_to_json(metrics),
            }
        )
        self.last_status = (
            f"target {source}: tcp {metrics['tcp_delta_norm']:.3f}m, "
            f"cam {metrics['camera_delta_norm']:.3f}m, rot {metrics['tcp_rotation_deg']:.1f}deg"
        )
        self.get_logger().info(self.last_status)
        self.write_metadata()

    def target_is_safe(self, metrics: dict[str, Any]) -> bool:
        max_tcp = float(self.get_parameter("target_max_tcp_delta_m").value)
        max_camera = float(self.get_parameter("target_max_camera_delta_m").value)
        max_rotation = float(self.get_parameter("target_max_rotation_deg").value)
        if (
            metrics["tcp_delta_norm"] > max_tcp
            or metrics["camera_delta_norm"] > max_camera
            or metrics["tcp_rotation_deg"] > max_rotation
        ):
            self.last_status = (
                f"target refused: tcp {metrics['tcp_delta_norm']:.3f}/{max_tcp:.3f}m, "
                f"cam {metrics['camera_delta_norm']:.3f}/{max_camera:.3f}m, "
                f"rot {metrics['tcp_rotation_deg']:.1f}/{max_rotation:.1f}deg"
            )
            self.get_logger().warn(self.last_status)
            return False
        return True

    def go_to_gui_target(self) -> None:
        if self.gui_target is None:
            self.last_status = "go blocked: no target prepared"
            return
        if not self.param_bool("move_enabled"):
            self.last_status = "go blocked: move_enabled is false"
            self.get_logger().warn(self.last_status)
            return
        observation = self.current_observation(require_cloud=False)
        if observation is None:
            return
        if self.anchor is not None:
            metrics = target_motion_metrics(
                observation.T_base_tcp,
                self.gui_target,
                observation.T_tcp_camera,
                np.asarray(self.anchor["p_base_anchor"], dtype=np.float64),
                look_axis=self.param_str("camera_look_axis"),
            )
            if not self.target_is_safe(metrics):
                return
        self.stop_jog(force=True)
        success = self.send_pose_goal(self.gui_target)
        self.target_history.append(
            {
                "event": "target_move",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": self.gui_target_source,
                "success": bool(success),
            }
        )
        self.last_status = "target move done" if success else "target move failed"
        if success:
            self.gui_target = None
            self.gui_target_source = ""
        self.write_metadata()

    def go_to_initial_pose(self) -> None:
        if self.initial_T_base_tcp is None:
            self.last_status = "back blocked: no initial pose captured"
            return
        if not self.param_bool("move_enabled"):
            self.last_status = "back blocked: move_enabled is false"
            self.get_logger().warn(self.last_status)
            return
        self.stop_jog(force=True)
        success = self.send_pose_goal(self.initial_T_base_tcp)
        self.last_status = "back to start done" if success else "back to start failed"

    def send_pose_goal(self, T_base_tcp: np.ndarray) -> bool:
        if self.action_client is None or JparseMove is None:
            self.get_logger().error("JparseMove action type is not available in this environment.")
            return False
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(f"Action server not available: {self.param_str('action_name')}")
            return False

        goal = JparseMove.Goal()
        goal.mode = "task_space"
        goal.accuracy = "approach"
        goal.target_pose = pose_stamped_from_matrix(
            T_base_tcp,
            self.param_str("planning_frame"),
            self.get_clock().now().to_msg(),
        )
        goal.max_linear_velocity = float(self.get_parameter("max_linear_velocity").value)
        goal.max_angular_velocity = float(self.get_parameter("max_angular_velocity").value)
        goal.timeout = float(self.get_parameter("move_timeout").value)

        future = self.action_client.send_goal_async(goal)
        self.wait_for_future(future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        self.wait_for_future(result_future)
        result = result_future.result().result
        self.get_logger().info(
            f"Move result: success={result.success}, message={result.message}, "
            f"pos_err={result.final_position_error:.4f}, ori_err={result.final_orientation_error:.4f}"
        )
        return bool(result.success)

    def wait_for_future(self, future: Any) -> None:
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)

    def jog_command_from_key(self, key: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        name = key_name(key)
        if name is None:
            return None, None
        v = float(self.get_parameter("jog_linear_velocity").value)
        w = float(self.get_parameter("jog_angular_velocity").value)
        zero = np.zeros(3, dtype=np.float64)
        translations = {
            "left": np.array([-v, 0.0, 0.0]),
            "right": np.array([v, 0.0, 0.0]),
            "up": np.array([0.0, v, 0.0]),
            "down": np.array([0.0, -v, 0.0]),
            "page_up": np.array([0.0, 0.0, v]),
            "page_down": np.array([0.0, 0.0, -v]),
            "a": np.array([-v, 0.0, 0.0]),
            "d": np.array([v, 0.0, 0.0]),
            "w": np.array([0.0, v, 0.0]),
            "x": np.array([0.0, -v, 0.0]),
            "r": np.array([0.0, 0.0, v]),
            "f": np.array([0.0, 0.0, -v]),
        }
        rotations = {
            "left": np.array([w, 0.0, 0.0]),
            "right": np.array([-w, 0.0, 0.0]),
            "up": np.array([0.0, w, 0.0]),
            "down": np.array([0.0, -w, 0.0]),
            "page_up": np.array([0.0, 0.0, w]),
            "page_down": np.array([0.0, 0.0, -w]),
            "j": np.array([w, 0.0, 0.0]),
            "l": np.array([-w, 0.0, 0.0]),
            "i": np.array([0.0, w, 0.0]),
            "k": np.array([0.0, -w, 0.0]),
            "u": np.array([0.0, 0.0, w]),
            "o": np.array([0.0, 0.0, -w]),
        }
        direct_rotation = name in ("i", "j", "k", "l", "u", "o")
        if self.rotation_jog_mode or direct_rotation:
            angular = rotations.get(name)
            return (zero, angular) if angular is not None else (None, None)
        linear = translations.get(name)
        return (linear, zero) if linear is not None else (None, None)

    def set_jog_target(self, linear: np.ndarray, angular: np.ndarray) -> None:
        if not self.param_bool("keyboard_jog_enabled"):
            self.last_jog_text = "jog disabled"
            self.get_logger().warn("Keyboard jog is disabled. Start with keyboard_jog_enabled:=true.")
            return
        self.jog_target_linear = np.asarray(linear, dtype=np.float64)
        self.jog_target_angular = np.asarray(angular, dtype=np.float64)
        self.last_jog_key_time = time.monotonic()
        self.last_jog_text = (
            f"jog lin=[{linear[0]:.3f},{linear[1]:.3f},{linear[2]:.3f}] "
            f"ang=[{angular[0]:.3f},{angular[1]:.3f},{angular[2]:.3f}]"
        )

    def update_jog_output(self) -> None:
        if not self.param_bool("keyboard_jog_enabled"):
            return
        now = time.monotonic()
        dt = max(1.0e-4, now - self.last_jog_update_time)
        self.last_jog_update_time = now
        if now - self.last_jog_key_time > float(self.get_parameter("jog_hold_timeout").value):
            self.jog_target_linear = np.zeros(3, dtype=np.float64)
            self.jog_target_angular = np.zeros(3, dtype=np.float64)

        self.jog_current_linear = slew_vector(
            self.jog_current_linear,
            self.jog_target_linear,
            float(self.get_parameter("jog_linear_acceleration").value) * dt,
        )
        self.jog_current_angular = slew_vector(
            self.jog_current_angular,
            self.jog_target_angular,
            float(self.get_parameter("jog_angular_acceleration").value) * dt,
        )
        if (
            np.linalg.norm(self.jog_current_linear) > 1.0e-5
            or np.linalg.norm(self.jog_current_angular) > 1.0e-5
            or np.linalg.norm(self.jog_target_linear) > 1.0e-5
            or np.linalg.norm(self.jog_target_angular) > 1.0e-5
        ):
            self.publish_jog_twist(self.jog_current_linear, self.jog_current_angular)

    def publish_jog_twist(self, linear: np.ndarray, angular: np.ndarray) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.param_str("jog_frame")
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self.jog_pub.publish(msg)

    def stop_jog(self, force: bool = False) -> None:
        self.jog_target_linear = np.zeros(3, dtype=np.float64)
        self.jog_target_angular = np.zeros(3, dtype=np.float64)
        self.jog_current_linear = np.zeros(3, dtype=np.float64)
        self.jog_current_angular = np.zeros(3, dtype=np.float64)
        self.last_jog_text = "jog stopped"
        if force or self.param_bool("keyboard_jog_enabled"):
            for _ in range(3):
                self.publish_jog_twist(self.jog_current_linear, self.jog_current_angular)
                time.sleep(0.02)

    def save_sample(self) -> None:
        if self.anchor is None:
            self.last_status = "save blocked: click an anchor first"
            return
        observation = self.current_observation(require_cloud=False)
        if observation is None:
            return

        image_rel = f"images/{self.sample_index:06d}.png"
        image_path = self.output_dir / image_rel
        cv2.imwrite(str(image_path), observation.image)
        self.write_intrinsics(observation.camera_info)

        record = {
            "image": image_rel,
            "timestamp": stamp_float(observation.image_msg.header.stamp),
            "image_stamp": stamp_dict(observation.image_msg.header.stamp),
            "camera_frame": observation.camera_frame,
            "T_base_camera": matrix_to_list(observation.T_base_camera),
            "T_base_tcp": matrix_to_list(observation.T_base_tcp),
        }
        self.camera_pose_records.append(record)

        anchor_base = np.asarray(self.anchor["p_base_anchor"], dtype=np.float64)
        p_camera_at_capture = transform_point(invert_transform(observation.T_base_camera), anchor_base)
        self.anchor_sample_records.append(
            {
                "image": image_rel,
                "p_base_anchor": anchor_base.astype(float).tolist(),
                "p_camera_anchor_at_capture": p_camera_at_capture.astype(float).tolist(),
                "camera_frame": observation.camera_frame,
            }
        )
        self.sample_index += 1
        self.write_camera_poses()
        self.write_anchors()
        self.write_metadata()
        self.last_status = f"saved {image_rel}"
        self.get_logger().info(self.last_status)

    def write_intrinsics(self, msg: CameraInfo) -> None:
        data = {
            "camera_name": msg.header.frame_id or "camera",
            "image_width": int(msg.width),
            "image_height": int(msg.height),
            "K": np.asarray(msg.k, dtype=float).reshape(3, 3).tolist(),
            "distortion": [float(value) for value in msg.d],
        }
        write_yaml(self.output_dir / "intrinsics.yaml", data)

    def write_camera_poses(self) -> None:
        write_json(
            self.output_dir / "camera_poses.json",
            {
                "frames": {
                    "base": self.param_str("robot_base_frame"),
                    "camera": None if self.anchor is None else self.anchor["camera_frame"],
                },
                "poses": self.camera_pose_records,
            },
        )

    def write_anchors(self) -> None:
        write_json(
            self.output_dir / "anchors.json",
            {
                "anchor": self.anchor,
                "samples": self.anchor_sample_records,
            },
        )

    def write_metadata(self) -> None:
        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "topics": {
                "image": self.param_str("image_topic"),
                "image_compressed": self.image_compressed,
                "camera_info": self.param_str("camera_info_topic"),
                "pointcloud": self.param_str("pointcloud_topic"),
            },
            "frames": {
                "base": self.param_str("robot_base_frame"),
                "tcp": self.param_str("robot_tcp_frame"),
                "camera_override": self.param_str("camera_frame") or None,
                "planning": self.param_str("planning_frame"),
            },
            "motion": {
                "move_enabled": self.param_bool("move_enabled"),
                "keyboard_jog_enabled": self.param_bool("keyboard_jog_enabled"),
                "action_name": self.param_str("action_name"),
                "jog_twist_topic": self.param_str("jog_twist_topic"),
                "jog_frame": self.param_str("jog_frame"),
            },
            "capture": {
                "click_window_radius": int(self.get_parameter("click_window_radius").value),
                "click_min_points": int(self.get_parameter("click_min_points").value),
                "click_max_mad_m": float(self.get_parameter("click_max_mad_m").value),
                "samples": int(self.get_parameter("samples").value),
                "sphere_polar_span_deg": float(self.get_parameter("sphere_polar_span_deg").value),
                "sphere_spiral_turns": float(self.get_parameter("sphere_spiral_turns").value),
                "camera_look_axis": self.param_str("camera_look_axis"),
            },
            "target_history": self.target_history,
        }
        write_yaml(self.output_dir / "capture_metadata.yaml", data)

    def write_anchor_approx_tf(self, T_base_camera_at_click: np.ndarray) -> None:
        if self.anchor is None:
            return
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = T_base_camera_at_click[:3, :3]
        T[:3, 3] = np.asarray(self.anchor["p_base_anchor"], dtype=np.float64)
        write_yaml(
            self.output_dir / "anchor_approx_tf.yaml",
            {
                "warning": "Approximate debug transform only. Do not use as final clip pose.",
                "parent_frame": self.param_str("robot_base_frame"),
                "child_frame": "clip_anchor_approx",
                "T_base_clip_anchor_approx": matrix_to_list(T),
                "translation_m": T[:3, 3].astype(float).tolist(),
                "quaternion_xyzw": quaternion_from_matrix(T[:3, :3]).astype(float).tolist(),
            },
        )

    def resize_for_display(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        max_side = int(self.get_parameter("display_max_side").value)
        height, width = image.shape[:2]
        scale = min(1.0, float(max_side) / float(max(height, width)))
        if scale >= 1.0:
            return image.copy(), 1.0
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized, scale

    def decode_image_msg(self, msg: Image | CompressedImage) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def draw_text_lines(self, image: np.ndarray, lines: list[str]) -> None:
        y = 28
        for line in lines:
            if not line:
                continue
            cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
            y += 27


def key_name(key: int) -> str | None:
    arrow_codes = {
        81: "left",
        82: "up",
        83: "right",
        84: "down",
        85: "page_up",
        86: "page_down",
        2424832: "left",
        2490368: "up",
        2555904: "right",
        2621440: "down",
        2162688: "page_up",
        2228224: "page_down",
        65361: "left",
        65362: "up",
        65363: "right",
        65364: "down",
        65365: "page_up",
        65366: "page_down",
    }
    candidates = {key, key & 0xFFFF, key & 0xFFFFFF, key & 0x01FFFFFF}
    for candidate in candidates:
        if candidate in arrow_codes:
            return arrow_codes[candidate]
    char = ascii_char(key)
    if char in ("w", "a", "d", "x", "r", "f", "i", "j", "k", "l", "u", "o"):
        return char
    return None


def ascii_char(key: int) -> str:
    return chr(key).lower() if 0 <= key < 128 else ""


def pose_stamped_from_matrix(T: np.ndarray, frame_id: str, stamp: Any) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    q = quaternion_from_matrix(T[:3, :3])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def matrix_to_list(T: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(T, dtype=np.float64)]


def metrics_to_json(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            result[key] = value.astype(float).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def stamp_dict(stamp: Any) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def stamp_float(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ClipObjectCaptureSession()
    try:
        node.run_gui_session()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
