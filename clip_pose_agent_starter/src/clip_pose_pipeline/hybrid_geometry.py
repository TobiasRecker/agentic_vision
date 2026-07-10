from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PlaneFit:
    point: np.ndarray
    normal: np.ndarray
    rms_m: float
    inlier_count: int


@dataclass(frozen=True)
class TriangulationResult:
    point_base: np.ndarray
    ray_errors_m: np.ndarray
    reprojection_errors_px: np.ndarray


@dataclass(frozen=True)
class HullView:
    T_base_camera: np.ndarray
    K: np.ndarray
    mask: np.ndarray
    roi_xywh: tuple[int, int, int, int]
    image_size: tuple[int, int]
    distortion: np.ndarray | None = None


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def invert_transform(T_A_B: np.ndarray) -> np.ndarray:
    T_A_B = np.asarray(T_A_B, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T_A_B[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T_A_B[:3, 3]
    return result


def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    points = np.asarray(points_B, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(T_A_B, dtype=np.float64).reshape(4, 4) @ homogeneous.T).T[:, :3]


def project_camera_points(
    points_camera: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 1.0e-9)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    coefficients = np.asarray([] if distortion is None else distortion, dtype=np.float64).reshape(-1)
    if coefficients.size and not np.allclose(coefficients, 0.0):
        projected, _ = cv2.projectPoints(
            points[valid],
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            coefficients,
        )
        pixels[valid] = projected.reshape(-1, 2)
    else:
        pixels[valid, 0] = K[0, 0] * points[valid, 0] / points[valid, 2] + K[0, 2]
        pixels[valid, 1] = K[1, 1] * points[valid, 1] / points[valid, 2] + K[1, 2]
    return pixels, valid


def project_base_points(
    points_base: np.ndarray,
    T_base_camera: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = transform_points(invert_transform(T_base_camera), points_base)
    return project_camera_points(points_camera, K, distortion)


def pixel_ray_base(
    pixel_uv: Iterable[float],
    K: np.ndarray,
    T_base_camera: np.ndarray,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.asarray(tuple(pixel_uv), dtype=np.float64).reshape(2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    coefficients = np.asarray([] if distortion is None else distortion, dtype=np.float64).reshape(-1)
    if coefficients.size and not np.allclose(coefficients, 0.0):
        normalized = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float64),
            K,
            coefficients,
        ).reshape(2)
        direction_camera = normalize(np.array([normalized[0], normalized[1], 1.0]))
    else:
        direction_camera = normalize(np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64))
    T = np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4)
    return T[:3, 3].copy(), normalize(T[:3, :3] @ direction_camera)


def triangulate_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    huber_scale_m: float = 0.003,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(origins) < 2 or len(origins) != len(directions):
        raise ValueError("triangulation needs at least two matching rays")
    directions = np.asarray([normalize(direction) for direction in directions])
    weights = np.ones(len(origins), dtype=np.float64)
    point = np.mean(origins, axis=0)
    identity = np.eye(3, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        projectors = identity[None, :, :] - directions[:, :, None] * directions[:, None, :]
        A = np.sum(weights[:, None, None] * projectors, axis=0)
        b = np.sum(weights[:, None] * np.einsum("nij,nj->ni", projectors, origins), axis=0)
        if np.linalg.cond(A) > 1.0e10:
            raise ValueError("camera rays do not provide a stable triangulation baseline")
        point = np.linalg.solve(A, b)
        residuals = np.linalg.norm(np.einsum("nij,nj->ni", projectors, point - origins), axis=1)
        scale = max(float(huber_scale_m), 1.0e-6)
        weights = np.where(residuals <= scale, 1.0, scale / np.maximum(residuals, 1.0e-12))
    return point, residuals


def triangulate_pixels(
    pixels_uv: np.ndarray,
    camera_matrices: list[np.ndarray],
    camera_poses: list[np.ndarray],
    reference_widths: Iterable[int] | None = None,
    distortions: list[np.ndarray] | None = None,
) -> TriangulationResult:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    if len(pixels) != len(camera_matrices) or len(pixels) != len(camera_poses):
        raise ValueError("pixels, intrinsics, and camera poses must have equal lengths")
    distortion_values = distortions if distortions is not None else [np.zeros(0)] * len(pixels)
    if len(distortion_values) != len(pixels):
        raise ValueError("distortions must match the number of pixels")
    rays = [
        pixel_ray_base(pixel, K, T, distortion)
        for pixel, K, T, distortion in zip(pixels, camera_matrices, camera_poses, distortion_values)
    ]
    point, ray_errors = triangulate_rays(
        np.asarray([ray[0] for ray in rays]),
        np.asarray([ray[1] for ray in rays]),
    )
    reprojection_errors = []
    widths = list(reference_widths) if reference_widths is not None else [1280] * len(pixels)
    for pixel, K, T, width, distortion in zip(
        pixels, camera_matrices, camera_poses, widths, distortion_values
    ):
        projected, valid = project_base_points(point[None, :], T, K, distortion)
        error = float(np.linalg.norm(projected[0] - pixel)) if valid[0] else float("inf")
        reprojection_errors.append(error * 1280.0 / float(width))
    return TriangulationResult(point, ray_errors, np.asarray(reprojection_errors))


def fit_plane_ransac(
    points_base: np.ndarray,
    camera_positions_base: np.ndarray,
    threshold_m: float = 0.002,
    iterations: int = 500,
    seed: int = 7,
) -> PlaneFit:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError("plane fitting needs at least three finite points")
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(points), dtype=bool)
    best_median = float("inf")
    for _ in range(max(1, int(iterations))):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-9:
            continue
        normal /= norm
        distances = np.abs((points - sample[0]) @ normal)
        mask = distances <= float(threshold_m)
        median = float(np.median(distances[mask])) if np.any(mask) else float("inf")
        if int(mask.sum()) > int(best_mask.sum()) or (int(mask.sum()) == int(best_mask.sum()) and median < best_median):
            best_mask = mask
            best_median = median
    if int(best_mask.sum()) < 3:
        raise ValueError("RANSAC could not find a support plane")
    inliers = points[best_mask]
    centroid = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = normalize(vh[-1])
    cameras = np.asarray(camera_positions_base, dtype=np.float64).reshape(-1, 3)
    if len(cameras) and float(np.dot(normal, np.mean(cameras, axis=0) - centroid)) < 0.0:
        normal = -normal
    residuals = (inliers - centroid) @ normal
    return PlaneFit(
        point=centroid,
        normal=normal,
        rms_m=float(np.sqrt(np.mean(residuals**2))),
        inlier_count=int(len(inliers)),
    )


def construct_task_frame(
    seat_center_base: np.ndarray,
    left_lip_base: np.ndarray,
    right_lip_base: np.ndarray,
    outward_normal_base: np.ndarray,
) -> np.ndarray:
    origin = np.asarray(seat_center_base, dtype=np.float64).reshape(3)
    z_axis = normalize(outward_normal_base)
    lateral = np.asarray(right_lip_base, dtype=np.float64).reshape(3) - np.asarray(
        left_lip_base, dtype=np.float64
    ).reshape(3)
    y_axis = normalize(lateral - float(np.dot(lateral, z_axis)) * z_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    T_base_clip = np.eye(4, dtype=np.float64)
    T_base_clip[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T_base_clip[:3, 3] = origin
    return T_base_clip


def encode_depth_mm(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 65.535)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[valid] = np.clip(np.rint(depth[valid] * 1000.0), 1, 65535).astype(np.uint16)
    return encoded


def decode_depth_mm(depth_mm: np.ndarray) -> np.ndarray:
    encoded = np.asarray(depth_mm, dtype=np.uint16)
    depth = encoded.astype(np.float64) * 0.001
    depth[encoded == 0] = np.nan
    return depth


def backproject_depth(depth_m: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rows, cols = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0.0)
    z = depth[valid]
    x = (cols[valid] - K[0, 2]) * z / K[0, 0]
    y = (rows[valid] - K[1, 2]) * z / K[1, 1]
    return np.column_stack([x, y, z]), np.column_stack([cols[valid], rows[valid]])


def sample_roi_mask(mask: np.ndarray, pixels_full: np.ndarray, roi_xywh: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    pixels = np.asarray(pixels_full, dtype=np.float64).reshape(-1, 2)
    x, y, width, height = (int(value) for value in roi_xywh)
    u = np.rint(pixels[:, 0] - x).astype(np.int64)
    v = np.rint(pixels[:, 1] - y).astype(np.int64)
    inside = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    selected = np.zeros(len(pixels), dtype=bool)
    selected[inside] = mask[v[inside], u[inside]] > 0
    return selected


def voxel_downsample(points: np.ndarray, voxel_size_m: float = 0.0005) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points.copy()
    voxel = max(float(voxel_size_m), 1.0e-6)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=points[:, axis]) for axis in range(3)])
    return sums / counts[:, None]


def statistical_outlier_filter(points: np.ndarray, neighbors: int = 12, std_ratio: float = 2.5) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= max(3, int(neighbors)):
        return points.copy()
    k = min(len(points), max(2, int(neighbors) + 1))
    distances, _ = cKDTree(points).query(points, k=k)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = float(np.mean(mean_distances) + float(std_ratio) * np.std(mean_distances))
    return points[mean_distances <= threshold]


def local_voxel_centers(
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    voxel_size_m: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    voxel = float(voxel_size_m)
    axes = [np.arange(low + 0.5 * voxel, high, voxel, dtype=np.float64) for low, high in bounds_xyz]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([component.reshape(-1) for component in grid])
    return points, tuple(len(axis) for axis in axes)


def carve_visual_hull(
    points_base: np.ndarray,
    views: list[HullView],
    min_views: int = 3,
    inside_fraction: float = 0.8,
    chunk_size: int = 200000,
) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if not views or not len(points):
        return np.zeros(len(points), dtype=bool)
    kept = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), max(1, int(chunk_size))):
        stop = min(len(points), start + max(1, int(chunk_size)))
        chunk = points[start:stop]
        visible_count = np.zeros(len(chunk), dtype=np.int16)
        inside_count = np.zeros(len(chunk), dtype=np.int16)
        for view in views:
            pixels, positive_depth = project_base_points(
                chunk, view.T_base_camera, view.K, view.distortion
            )
            width, height = view.image_size
            visible = (
                positive_depth
                & (pixels[:, 0] >= 0.0)
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] < height)
            )
            inside = visible & sample_roi_mask(view.mask, pixels, view.roi_xywh)
            visible_count += visible.astype(np.int16)
            inside_count += inside.astype(np.int16)
        fraction = inside_count / np.maximum(visible_count, 1)
        kept[start:stop] = (inside_count >= int(min_views)) & (fraction >= float(inside_fraction))
    return kept


def rasterize_silhouette(
    points_base: np.ndarray,
    view: HullView,
    dilation_px: int = 2,
) -> np.ndarray:
    pixels, valid = project_base_points(points_base, view.T_base_camera, view.K, view.distortion)
    x, y, width, height = view.roi_xywh
    u = np.rint(pixels[valid, 0] - x).astype(np.int64)
    v = np.rint(pixels[valid, 1] - y).astype(np.int64)
    inside = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    rendered = np.zeros((height, width), dtype=np.uint8)
    rendered[v[inside], u[inside]] = 255
    if dilation_px > 0:
        size = 2 * int(dilation_px) + 1
        rendered = cv2.dilate(rendered, np.ones((size, size), dtype=np.uint8))
    return rendered


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first) > 0
    b = np.asarray(second) > 0
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0
