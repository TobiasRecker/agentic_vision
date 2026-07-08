from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnchorEstimate:
    valid: bool
    point_camera: np.ndarray | None
    samples_total: int
    samples_finite: int
    samples_inlier: int
    median_abs_deviation_m: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "point_camera": None
            if self.point_camera is None
            else self.point_camera.astype(float).tolist(),
            "samples_total": int(self.samples_total),
            "samples_finite": int(self.samples_finite),
            "samples_inlier": int(self.samples_inlier),
            "median_abs_deviation_m": float(self.median_abs_deviation_m),
            "reason": self.reason,
        }


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0e-12:
        return vector / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(fallback, dtype=np.float64).reshape(3)


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def transform_point(T_A_B: np.ndarray, point_B: np.ndarray) -> np.ndarray:
    point_B = np.asarray(point_B, dtype=np.float64).reshape(3)
    homogeneous = np.r_[point_B, 1.0]
    return (np.asarray(T_A_B, dtype=np.float64) @ homogeneous)[:3]


def transform_from_translation_quaternion(
    translation: list[float] | tuple[float, float, float] | np.ndarray,
    quaternion_xyzw: list[float] | tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = q / norm
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray | None = None) -> float:
    if R_b is None:
        R_rel = np.asarray(R_a, dtype=np.float64)
    else:
        R_rel = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cos_angle = 0.5 * (float(np.trace(R_rel)) - 1.0)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def slew_vector(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_delta or norm <= 1.0e-12:
        return target.copy()
    return current + delta * (float(max_delta) / norm)


def estimate_anchor_from_xyz_image(
    xyz_image: np.ndarray,
    pixel_uv: tuple[float, float] | list[float] | np.ndarray,
    window_radius: int = 4,
    min_points: int = 8,
    max_mad_m: float = 0.02,
) -> AnchorEstimate:
    xyz = np.asarray(xyz_image, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        return AnchorEstimate(False, None, 0, 0, 0, float("inf"), "point cloud is not HxWx3")

    height, width = xyz.shape[:2]
    u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or vi < 0 or ui >= width or vi >= height:
        return AnchorEstimate(False, None, 0, 0, 0, float("inf"), "pixel outside point cloud")

    radius = max(0, int(window_radius))
    x0 = max(0, ui - radius)
    x1 = min(width, ui + radius + 1)
    y0 = max(0, vi - radius)
    y1 = min(height, vi + radius + 1)
    window = xyz[y0:y1, x0:x1, :3].reshape(-1, 3)
    samples_total = int(window.shape[0])
    finite = window[np.all(np.isfinite(window), axis=1)]
    finite = finite[np.linalg.norm(finite, axis=1) > 1.0e-9]
    samples_finite = int(finite.shape[0])
    if samples_finite < int(min_points):
        return AnchorEstimate(
            False,
            None,
            samples_total,
            samples_finite,
            0,
            float("inf"),
            "too few finite 3D samples",
        )

    median = np.median(finite, axis=0)
    distances = np.linalg.norm(finite - median, axis=1)
    mad = float(np.median(distances))
    inlier_threshold = max(float(max_mad_m), 3.0 * mad)
    inliers = finite[distances <= inlier_threshold]
    samples_inlier = int(inliers.shape[0])
    if samples_inlier < int(min_points):
        return AnchorEstimate(
            False,
            None,
            samples_total,
            samples_finite,
            samples_inlier,
            mad,
            "too few inlier 3D samples",
        )

    refined = np.median(inliers, axis=0)
    refined_mad = float(np.median(np.linalg.norm(inliers - refined, axis=1)))
    if refined_mad > float(max_mad_m):
        return AnchorEstimate(
            False,
            refined,
            samples_total,
            samples_finite,
            samples_inlier,
            refined_mad,
            "3D samples are too noisy",
        )

    return AnchorEstimate(
        True,
        refined,
        samples_total,
        samples_finite,
        samples_inlier,
        refined_mad,
        "ok",
    )


def center_camera_target(
    T_base_camera_current: np.ndarray,
    anchor_base: np.ndarray,
    xy_only: bool = True,
    look_axis: str = "plus_z",
) -> np.ndarray:
    T_base_camera_current = np.asarray(T_base_camera_current, dtype=np.float64)
    anchor_base = np.asarray(anchor_base, dtype=np.float64).reshape(3)
    camera_position = T_base_camera_current[:3, 3]
    forward = camera_forward_vector(T_base_camera_current, look_axis)
    anchor_vector = anchor_base - camera_position
    axial_distance = float(np.dot(anchor_vector, forward))
    centered_position = anchor_base - axial_distance * forward
    if xy_only:
        centered_position[2] = camera_position[2]
    T_target = T_base_camera_current.copy()
    T_target[:3, 3] = centered_position
    return T_target


def camera_point_from_pixel(
    pixel_uv: tuple[float, float] | list[float] | np.ndarray,
    K: np.ndarray,
    depth_m: float,
) -> np.ndarray:
    u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if abs(fx) <= 1.0e-12 or abs(fy) <= 1.0e-12:
        raise ValueError("Camera matrix has invalid focal length")
    z = float(depth_m)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def center_camera_target_from_pixel(
    T_base_camera_current: np.ndarray,
    pixel_uv: tuple[float, float] | list[float] | np.ndarray,
    K: np.ndarray,
    depth_m: float,
    xy_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    T_base_camera_current = np.asarray(T_base_camera_current, dtype=np.float64)
    point_camera = camera_point_from_pixel(pixel_uv, K, depth_m)
    anchor_base = transform_point(T_base_camera_current, point_camera)
    lateral_camera = point_camera.copy()
    lateral_camera[2] = 0.0
    target_position = T_base_camera_current[:3, 3] + T_base_camera_current[:3, :3] @ lateral_camera
    if xy_only:
        target_position[2] = T_base_camera_current[2, 3]
    T_target = T_base_camera_current.copy()
    T_target[:3, 3] = target_position
    return T_target, anchor_base


def camera_forward_vector(T_base_camera: np.ndarray, look_axis: str = "plus_z") -> np.ndarray:
    z_axis = normalize(np.asarray(T_base_camera, dtype=np.float64)[:3, 2])
    return -z_axis if str(look_axis).lower() == "minus_z" else z_axis


def look_at_camera_pose(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    reference_rotation: np.ndarray | None = None,
    look_axis: str = "plus_z",
) -> np.ndarray:
    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    forward = normalize(target_position - camera_position, fallback=np.array([0.0, 0.0, 1.0]))
    z_axis = -forward if str(look_axis).lower() == "minus_z" else forward

    if reference_rotation is not None:
        reference_x = np.asarray(reference_rotation, dtype=np.float64)[:3, 0]
    else:
        reference_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = reference_x - z_axis * float(np.dot(reference_x, z_axis))
    if np.linalg.norm(x_axis) <= 1.0e-9:
        for fallback in (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ):
            x_axis = fallback - z_axis * float(np.dot(fallback, z_axis))
            if np.linalg.norm(x_axis) > 1.0e-9:
                break
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0]))
    x_axis = normalize(np.cross(y_axis, z_axis), fallback=x_axis)

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = camera_position
    return T


def generate_spiral_hemisphere_targets(
    anchor_base: np.ndarray,
    T_base_camera_start: np.ndarray,
    sample_count: int,
    radius_m: float | None = None,
    polar_span_deg: float = 50.0,
    spiral_turns: float = 1.25,
    look_axis: str = "plus_z",
) -> list[np.ndarray]:
    anchor_base = np.asarray(anchor_base, dtype=np.float64).reshape(3)
    T_base_camera_start = np.asarray(T_base_camera_start, dtype=np.float64)
    start_position = T_base_camera_start[:3, 3]
    start_rotation = T_base_camera_start[:3, :3]
    radius = float(radius_m) if radius_m is not None and radius_m > 0.0 else float(np.linalg.norm(start_position - anchor_base))
    if radius <= 1.0e-6:
        raise ValueError("Cannot generate hemisphere targets with near-zero radius")

    count = max(1, int(sample_count))
    polar_span = math.radians(float(np.clip(polar_span_deg, 1.0, 85.0)))
    turns = max(0.1, float(spiral_turns))
    zenith = normalize(start_position - anchor_base, fallback=np.array([0.0, 0.0, 1.0]))
    tangent_x = start_rotation[:3, 0] - zenith * float(np.dot(start_rotation[:3, 0], zenith))
    tangent_x = normalize(tangent_x, fallback=np.array([1.0, 0.0, 0.0]))
    tangent_y = normalize(np.cross(zenith, tangent_x), fallback=np.array([0.0, 1.0, 0.0]))

    targets: list[np.ndarray] = []
    for index in range(count):
        t = (index + 1) / float(count)
        polar = polar_span * math.sqrt(t)
        azimuth = 2.0 * math.pi * turns * t
        tangent = math.cos(azimuth) * tangent_x + math.sin(azimuth) * tangent_y
        direction = normalize(math.cos(polar) * zenith + math.sin(polar) * tangent)
        position = anchor_base + radius * direction
        targets.append(
            look_at_camera_pose(
                position,
                anchor_base,
                reference_rotation=start_rotation,
                look_axis=look_axis,
            )
        )
    return targets


def target_motion_metrics(
    T_base_tcp_current: np.ndarray,
    T_base_tcp_target: np.ndarray,
    T_tcp_camera: np.ndarray,
    anchor_base: np.ndarray,
    look_axis: str = "plus_z",
) -> dict[str, Any]:
    T_current_camera = np.asarray(T_base_tcp_current, dtype=np.float64) @ np.asarray(T_tcp_camera, dtype=np.float64)
    T_target_camera = np.asarray(T_base_tcp_target, dtype=np.float64) @ np.asarray(T_tcp_camera, dtype=np.float64)
    T_current_target_tcp = invert_transform(T_base_tcp_current) @ T_base_tcp_target
    T_current_target_camera = invert_transform(T_current_camera) @ T_target_camera
    anchor_base = np.asarray(anchor_base, dtype=np.float64).reshape(3)

    target_camera_position = T_target_camera[:3, 3]
    target_to_anchor = normalize(anchor_base - target_camera_position)
    target_forward = camera_forward_vector(T_target_camera, look_axis)
    look_cos = float(np.clip(np.dot(target_forward, target_to_anchor), -1.0, 1.0))
    return {
        "tcp_delta_base": (T_base_tcp_target[:3, 3] - T_base_tcp_current[:3, 3]).astype(float),
        "tcp_delta_norm": float(np.linalg.norm(T_base_tcp_target[:3, 3] - T_base_tcp_current[:3, 3])),
        "camera_delta_base": (T_target_camera[:3, 3] - T_current_camera[:3, 3]).astype(float),
        "camera_delta_norm": float(np.linalg.norm(T_target_camera[:3, 3] - T_current_camera[:3, 3])),
        "tcp_rotation_deg": rotation_angle_deg(T_current_target_tcp[:3, :3]),
        "camera_rotation_deg": rotation_angle_deg(T_current_target_camera[:3, :3]),
        "distance_to_anchor_m": float(np.linalg.norm(target_camera_position - anchor_base)),
        "look_angle_deg": math.degrees(math.acos(look_cos)),
        "camera_position": target_camera_position.astype(float),
        "anchor_base": anchor_base.astype(float),
    }
