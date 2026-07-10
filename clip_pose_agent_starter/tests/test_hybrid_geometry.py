from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clip_pose_pipeline.hybrid_geometry import (  # noqa: E402
    HullView,
    carve_visual_hull,
    construct_task_frame,
    decode_depth_mm,
    encode_depth_mm,
    fit_plane_ransac,
    triangulate_pixels,
    voxel_downsample,
)
from clip_pose_pipeline.hybrid_annotation import automatic_dark_background_mask  # noqa: E402


def camera_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(reference_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, forward])
    T[:3, 3] = position
    return T


def project(point: np.ndarray, T_base_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    T_camera_base = np.linalg.inv(T_base_camera)
    camera = T_camera_base @ np.r_[point, 1.0]
    return np.array([K[0, 0] * camera[0] / camera[2] + K[0, 2], K[1, 1] * camera[1] / camera[2] + K[1, 2]])


def test_depth_uint16_roundtrip_has_millimeter_precision() -> None:
    depth = np.array([[0.4214, np.nan], [1.2346, 0.0]])
    encoded = encode_depth_mm(depth)
    decoded = decode_depth_mm(encoded)

    assert encoded.dtype == np.uint16
    np.testing.assert_allclose(decoded[[0, 1], [0, 0]], [0.421, 1.235], atol=0.00051)
    assert np.isnan(decoded[0, 1])
    assert np.isnan(decoded[1, 1])


def test_robust_triangulation_recovers_known_point() -> None:
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    point = np.array([0.012, -0.008, 0.03])
    poses = [
        camera_pose(np.array([0.18, -0.08, 0.32]), point),
        camera_pose(np.array([-0.16, -0.06, 0.30]), point),
        camera_pose(np.array([0.02, 0.16, 0.34]), point),
        camera_pose(np.array([-0.10, 0.12, 0.29]), point),
    ]
    pixels = np.asarray([project(point, pose, K) for pose in poses])
    pixels[-1] += [3.0, -2.0]

    result = triangulate_pixels(pixels, [K] * len(poses), poses, [640] * len(poses))

    np.testing.assert_allclose(result.point_base, point, atol=0.001)
    assert float(np.median(result.reprojection_errors_px)) < 2.0
    assert float(np.sqrt(np.mean(result.reprojection_errors_px**2))) < 3.1


def test_plane_normal_points_toward_cameras() -> None:
    rng = np.random.default_rng(4)
    xy = rng.uniform(-0.15, 0.15, size=(500, 2))
    points = np.column_stack([xy, rng.normal(0.0, 0.0003, size=len(xy))])
    points = np.vstack([points, rng.uniform(-0.2, 0.2, size=(30, 3))])
    cameras = np.array([[0.0, 0.0, 0.4], [0.1, -0.1, 0.35]])

    plane = fit_plane_ransac(points, cameras, threshold_m=0.0015)

    assert float(np.dot(plane.normal, [0.0, 0.0, 1.0])) > 0.999
    assert plane.rms_m < 0.0005
    assert plane.inlier_count >= 480


def test_task_frame_uses_semantic_lip_direction() -> None:
    T = construct_task_frame(
        seat_center_base=np.array([0.4, 0.1, 0.2]),
        left_lip_base=np.array([0.4, 0.08, 0.2]),
        right_lip_base=np.array([0.4, 0.12, 0.2]),
        outward_normal_base=np.array([0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1.0e-9)
    np.testing.assert_allclose(T[:3, 3], [0.4, 0.1, 0.2])
    assert np.linalg.det(T[:3, :3]) > 0.999


def test_voxel_downsample_averages_points() -> None:
    points = np.array([[0.0001, 0.0, 0.0], [0.0004, 0.0, 0.0], [0.0021, 0.0, 0.0]])
    downsampled = voxel_downsample(points, voxel_size_m=0.001)

    assert len(downsampled) == 2
    np.testing.assert_allclose(downsampled[0], [0.00025, 0.0, 0.0])


def test_visual_hull_keeps_only_points_inside_masks() -> None:
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 10, 255, -1)
    points = np.array([[0.0, 0.0, 1.0], [0.4, 0.0, 1.0]])
    view = HullView(T, K, mask, (0, 0, 100, 100), (100, 100))

    keep = carve_visual_hull(points, [view], min_views=1, inside_fraction=1.0)

    np.testing.assert_array_equal(keep, [True, False])


def test_dark_background_presegmentation_selects_center_object() -> None:
    image = np.full((160, 200, 3), 20, dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (120, 100), (230, 230, 230), -1)
    cv2.rectangle(image, (0, 0), (25, 25), (255, 255, 255), -1)

    mask = automatic_dark_background_mask(image)

    assert mask[80, 100] == 255
    assert mask[10, 10] == 0
