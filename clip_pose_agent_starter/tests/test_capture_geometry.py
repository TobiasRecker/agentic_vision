from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clip_pose_pipeline.capture_geometry import (  # noqa: E402
    center_camera_target_from_pixel,
    center_camera_target,
    estimate_anchor_from_xyz_image,
    generate_spiral_hemisphere_targets,
    invert_transform,
    project_camera_point_to_pixel,
    target_motion_metrics,
    transform_from_translation_quaternion,
    transform_point,
    xyz_image_from_organized_points,
)


def test_estimate_anchor_from_organized_pointcloud_window() -> None:
    xyz = np.zeros((20, 30, 3), dtype=np.float64)
    xyz[:, :, 0] = 0.10
    xyz[:, :, 1] = -0.02
    xyz[:, :, 2] = 0.42
    xyz[9:12, 14:17] += np.array([0.002, -0.001, 0.003])
    xyz[10, 15] = np.array([np.nan, np.nan, np.nan])

    estimate = estimate_anchor_from_xyz_image(
        xyz,
        (15, 10),
        window_radius=2,
        min_points=8,
        max_mad_m=0.02,
    )

    assert estimate.valid
    assert estimate.point_camera is not None
    np.testing.assert_allclose(estimate.point_camera, [0.10, -0.02, 0.42], atol=0.004)
    assert estimate.samples_finite >= 8


def test_estimate_anchor_accepts_degraded_noisy_depth_window() -> None:
    xyz = np.zeros((20, 30, 3), dtype=np.float64)
    xyz[:, :, 0] = 0.12
    xyz[:, :, 1] = 0.05
    xyz[:, :, 2] = 0.62
    for offset, z_delta in enumerate(np.linspace(-0.10, 0.10, 9)):
        xyz[6 + offset, 11:20, 2] += z_delta

    estimate = estimate_anchor_from_xyz_image(
        xyz,
        (15, 10),
        window_radius=4,
        min_points=8,
        max_mad_m=0.02,
    )

    assert estimate.valid
    assert estimate.reason == "ok_noisy"
    assert estimate.point_camera is not None
    np.testing.assert_allclose(estimate.point_camera, [0.12, 0.05, 0.62], atol=0.02)


def test_xyz_image_from_organized_points_keeps_native_layout() -> None:
    points = np.zeros((720, 1280, 3), dtype=np.float64)
    points[421, 704] = [0.1, -0.02, 0.42]

    xyz, layout = xyz_image_from_organized_points(points, width=1280, height=720)

    assert layout == "native"
    assert xyz is not None
    assert xyz.shape == (720, 1280, 3)
    np.testing.assert_allclose(xyz[421, 704], [0.1, -0.02, 0.42])


def test_xyz_image_from_organized_points_recovers_sensor_msgs_py_layout() -> None:
    native = np.zeros((720, 1280, 3), dtype=np.float64)
    native[421, 704] = [0.1, -0.02, 0.42]
    points = native.reshape(1280, 720, 3)

    xyz, layout = xyz_image_from_organized_points(points, width=1280, height=720)

    assert layout == "width_height_reshape"
    assert xyz is not None
    assert xyz.shape == (720, 1280, 3)
    np.testing.assert_allclose(xyz[421, 704], [0.1, -0.02, 0.42])


def test_project_camera_point_round_trips_pixel() -> None:
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    point = np.array([0.04, -0.02, 0.5])

    pixel = project_camera_point_to_pixel(point, K)

    np.testing.assert_allclose(pixel, [712.0, 324.0])


def test_xyz_image_from_organized_points_reshapes_flat_layout() -> None:
    points = np.zeros((720 * 1280, 3), dtype=np.float64)
    points[421 * 1280 + 704] = [0.1, -0.02, 0.42]

    xyz, layout = xyz_image_from_organized_points(points, width=1280, height=720)

    assert layout == "flat"
    assert xyz is not None
    assert xyz.shape == (720, 1280, 3)
    np.testing.assert_allclose(xyz[421, 704], [0.1, -0.02, 0.42])


def test_transform_point_camera_to_base() -> None:
    T_base_camera = transform_from_translation_quaternion(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    p_base = transform_point(T_base_camera, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(p_base, [1.1, 1.8, 3.3])


def test_center_camera_target_places_anchor_on_optical_axis() -> None:
    T_base_camera = np.eye(4, dtype=np.float64)
    anchor = np.array([0.12, -0.08, 0.5])

    target = center_camera_target(T_base_camera, anchor, xy_only=False)
    centered_anchor = transform_point(invert_transform(target), anchor)

    np.testing.assert_allclose(centered_anchor[:2], [0.0, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(target[:3, :3], np.eye(3), atol=1.0e-9)


def test_center_camera_target_from_pixel_uses_intrinsics_and_fallback_depth() -> None:
    T_base_camera = np.eye(4, dtype=np.float64)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    target, anchor = center_camera_target_from_pixel(
        T_base_camera,
        (370.0, 220.0),
        K,
        depth_m=0.5,
        xy_only=False,
    )

    np.testing.assert_allclose(anchor, [0.05, -0.02, 0.5])
    np.testing.assert_allclose(target[:3, 3], [0.05, -0.02, 0.0])
    centered_anchor = transform_point(invert_transform(target), anchor)
    np.testing.assert_allclose(centered_anchor[:2], [0.0, 0.0], atol=1.0e-9)


def test_spiral_hemisphere_targets_keep_radius_and_look_at_anchor() -> None:
    anchor = np.array([0.0, 0.0, 0.0])
    T_start = np.eye(4, dtype=np.float64)
    T_start[:3, 3] = [0.0, 0.0, 0.5]

    targets = generate_spiral_hemisphere_targets(anchor, T_start, sample_count=6)

    assert len(targets) == 6
    for target in targets:
        distance = np.linalg.norm(target[:3, 3] - anchor)
        np.testing.assert_allclose(distance, 0.5, atol=1.0e-9)
        forward = target[:3, 2]
        direction_to_anchor = (anchor - target[:3, 3]) / distance
        assert float(np.dot(forward, direction_to_anchor)) > 0.999


def test_target_motion_metrics_reports_camera_delta() -> None:
    T_tcp_camera = np.eye(4, dtype=np.float64)
    current = np.eye(4, dtype=np.float64)
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [0.01, 0.02, 0.03]

    metrics = target_motion_metrics(current, target, T_tcp_camera, np.array([0.0, 0.0, 1.0]))

    np.testing.assert_allclose(metrics["tcp_delta_base"], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(metrics["camera_delta_base"], [0.01, 0.02, 0.03])
    assert metrics["tcp_delta_norm"] > 0.0
