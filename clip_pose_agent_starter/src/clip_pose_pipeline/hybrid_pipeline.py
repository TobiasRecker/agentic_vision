from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .hybrid_geometry import (
    HullView,
    PlaneFit,
    backproject_depth,
    carve_visual_hull,
    construct_task_frame,
    decode_depth_mm,
    fit_plane_ransac,
    invert_transform,
    local_voxel_centers,
    mask_iou,
    project_camera_points,
    rasterize_silhouette,
    sample_roi_mask,
    statistical_outlier_filter,
    transform_points,
    triangulate_pixels,
    voxel_downsample,
)
from .hybrid_io import (
    load_annotations,
    load_manifest,
    read_yaml,
    write_json,
    write_point_cloud_ply,
    write_yaml,
)
from .transforms import rotation_angle_deg


KEYPOINT_NAMES = ("left_lip", "right_lip", "seat_center")


@dataclass
class LoadedView:
    sample_id: str
    image_path: Path
    image_size: tuple[int, int]
    roi_path: Path
    roi_xywh: tuple[int, int, int, int]
    mask: np.ndarray
    K_rgb: np.ndarray
    distortion_rgb: np.ndarray
    T_base_camera: np.ndarray
    keypoints: dict[str, np.ndarray]
    depth_m: np.ndarray | None
    K_depth: np.ndarray | None
    T_base_camera_depth: np.ndarray | None

    def hull_view(self) -> HullView:
        return HullView(
            T_base_camera=self.T_base_camera,
            K=self.K_rgb,
            mask=self.mask,
            roi_xywh=self.roi_xywh,
            image_size=self.image_size,
            distortion=self.distortion_rgb,
        )


@dataclass(frozen=True)
class SubsetEstimate:
    T_base_clip: np.ndarray
    plane: PlaneFit
    points: dict[str, np.ndarray]
    normalized_reprojection_errors_px: np.ndarray


DEFAULT_CONFIG: dict[str, Any] = {
    "minimum_views": 6,
    "minimum_keypoint_views": 4,
    "plane_threshold_m": 0.002,
    "plane_max_rms_m": 0.002,
    "clip_min_plane_distance_m": 0.001,
    "clip_max_plane_distance_m": 0.060,
    "depth_stride": 2,
    "voxel_size_m": 0.0005,
    "outlier_neighbors": 12,
    "outlier_std_ratio": 2.5,
    "hull_bounds_clip_m": [[-0.04, 0.04], [-0.04, 0.04], [-0.005, 0.05]],
    "hull_voxel_size_m": 0.0005,
    "hull_min_views": 3,
    "hull_inside_fraction": 0.8,
    "max_reprojection_rmse_px_1280": 3.0,
    "max_split_translation_m": 0.003,
    "max_split_rotation_deg": 2.0,
    "max_leave_one_out_origin_m": 0.003,
    "minimum_heldout_mask_iou": 0.70,
}


def merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULT_CONFIG, **(config or {})}


def load_hybrid_views(session_dir: str | Path, manifest_path: str | Path | None = None) -> list[LoadedView]:
    session = Path(session_dir).expanduser().resolve()
    manifest_file = Path(manifest_path).resolve() if manifest_path else session / "hybrid" / "manifest.json"
    manifest = load_manifest(manifest_file)
    annotations = load_annotations(session).get("views", {})
    views: list[LoadedView] = []
    for record in manifest["views"]:
        sample_id = str(record["sample_id"])
        annotation = annotations.get(sample_id, {})
        mask_path = session / str(record["mask"])
        if not mask_path.is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        roi_xywh = tuple(int(value) for value in record["roi_xywh"])
        if mask.shape != (roi_xywh[3], roi_xywh[2]):
            raise ValueError(f"mask {mask_path} has shape {mask.shape}, expected {(roi_xywh[3], roi_xywh[2])}")
        keypoints = {
            name: np.asarray(value, dtype=np.float64).reshape(2)
            for name, value in annotation.get("keypoints_full", {}).items()
            if name in KEYPOINT_NAMES
        }
        depth_m = None
        K_depth = None
        T_base_camera_depth = None
        if record.get("depth") and record.get("depth_metadata"):
            depth_path = session / str(record["depth"])
            metadata_path = session / str(record["depth_metadata"])
            encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if encoded is None or encoded.dtype != np.uint16:
                raise ValueError(f"depth image must be uint16 PNG: {depth_path}")
            metadata = read_yaml(metadata_path)
            depth_m = decode_depth_mm(encoded)
            K_depth = np.asarray(metadata["K"], dtype=np.float64).reshape(3, 3)
            T_base_camera_depth = np.asarray(
                metadata.get("T_base_camera", record["T_base_camera"]), dtype=np.float64
            ).reshape(4, 4)
        views.append(
            LoadedView(
                sample_id=sample_id,
                image_path=session / str(record["image"]),
                image_size=tuple(int(value) for value in record["image_size"]),
                roi_path=session / str(record["roi"]),
                roi_xywh=roi_xywh,
                mask=mask,
                K_rgb=np.asarray(record["K_rgb"], dtype=np.float64).reshape(3, 3),
                distortion_rgb=np.asarray(record.get("distortion_rgb", []), dtype=np.float64),
                T_base_camera=np.asarray(record["T_base_camera"], dtype=np.float64).reshape(4, 4),
                keypoints=keypoints,
                depth_m=depth_m,
                K_depth=K_depth,
                T_base_camera_depth=T_base_camera_depth,
            )
        )
    return views


def depth_cloud_for_view(view: LoadedView, stride: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if view.depth_m is None or view.K_depth is None or view.T_base_camera_depth is None:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    points_camera, _ = backproject_depth(view.depth_m, view.K_depth)
    if stride > 1:
        points_camera = points_camera[:: int(stride)]
    pixels_rgb, valid_projection = project_camera_points(
        points_camera, view.K_rgb, view.distortion_rgb
    )
    points_camera = points_camera[valid_projection]
    pixels_rgb = pixels_rgb[valid_projection]
    selected_clip = sample_roi_mask(view.mask, pixels_rgb, view.roi_xywh)
    x, y, width, height = view.roi_xywh
    inside_roi = (
        (pixels_rgb[:, 0] >= x)
        & (pixels_rgb[:, 1] >= y)
        & (pixels_rgb[:, 0] < x + width)
        & (pixels_rgb[:, 1] < y + height)
    )
    points_base = transform_points(view.T_base_camera_depth, points_camera)

    colors = np.zeros((len(points_base), 3), dtype=np.uint8)
    roi_image = cv2.imread(str(view.roi_path), cv2.IMREAD_COLOR)
    if roi_image is not None:
        u = np.rint(pixels_rgb[:, 0] - x).astype(np.int64)
        v = np.rint(pixels_rgb[:, 1] - y).astype(np.int64)
        valid_color = inside_roi & (u >= 0) & (v >= 0) & (u < roi_image.shape[1]) & (v < roi_image.shape[0])
        colors[valid_color] = roi_image[v[valid_color], u[valid_color], ::-1]
    return points_base[selected_clip], points_base[inside_roi & ~selected_clip], colors[selected_clip]


def estimate_subset(views: list[LoadedView], config: dict[str, Any]) -> SubsetEstimate:
    cfg = merged_config(config)
    support_points = []
    camera_positions = []
    for view in views:
        _, support, _ = depth_cloud_for_view(view, int(cfg["depth_stride"]))
        if len(support):
            support_points.append(support)
            camera_positions.append(view.T_base_camera[:3, 3])
    if not support_points:
        raise ValueError("no depth support points are available for plane fitting")
    plane = fit_plane_ransac(
        np.vstack(support_points),
        np.asarray(camera_positions),
        threshold_m=float(cfg["plane_threshold_m"]),
    )

    points: dict[str, np.ndarray] = {}
    reprojection_errors = []
    for name in KEYPOINT_NAMES:
        observations = [view for view in views if name in view.keypoints]
        if len(observations) < int(cfg["minimum_keypoint_views"]):
            raise ValueError(f"keypoint {name} has only {len(observations)} annotated views")
        result = triangulate_pixels(
            np.asarray([view.keypoints[name] for view in observations]),
            [view.K_rgb for view in observations],
            [view.T_base_camera for view in observations],
            [view.image_size[0] for view in observations],
            [view.distortion_rgb for view in observations],
        )
        points[name] = result.point_base
        reprojection_errors.extend(result.reprojection_errors_px.tolist())
    T_base_clip = construct_task_frame(
        points["seat_center"],
        points["left_lip"],
        points["right_lip"],
        plane.normal,
    )
    return SubsetEstimate(T_base_clip, plane, points, np.asarray(reprojection_errors))


def pose_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = float(rotation_angle_deg(first[:3, :3], second[:3, :3]))
    return translation, rotation


def build_geometry(
    views: list[LoadedView],
    estimate: SubsetEstimate,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], list[float]]:
    cfg = merged_config(config)
    clip_points = []
    clip_colors = []
    for view in views:
        points, _, colors = depth_cloud_for_view(view, int(cfg["depth_stride"]))
        if not len(points):
            continue
        signed_distance = (points - estimate.plane.point) @ estimate.plane.normal
        keep = (
            (signed_distance >= float(cfg["clip_min_plane_distance_m"]))
            & (signed_distance <= float(cfg["clip_max_plane_distance_m"]))
        )
        clip_points.append(points[keep])
        clip_colors.append(colors[keep])
    fused_base = np.vstack(clip_points) if clip_points else np.empty((0, 3), dtype=np.float64)
    fused_colors = np.vstack(clip_colors) if clip_colors else np.empty((0, 3), dtype=np.uint8)
    if len(fused_base):
        raw_base = fused_base
        raw_colors = fused_colors
        filtered = statistical_outlier_filter(
            fused_base,
            neighbors=int(cfg["outlier_neighbors"]),
            std_ratio=float(cfg["outlier_std_ratio"]),
        )
        fused_base = voxel_downsample(filtered, float(cfg["voxel_size_m"]))
        if len(raw_colors) == len(raw_base) and len(fused_base):
            _, nearest = cKDTree(raw_base).query(fused_base, k=1)
            fused_colors = raw_colors[np.atleast_1d(np.asarray(nearest, dtype=np.int64))]
        else:
            fused_colors = np.empty((0, 3), dtype=np.uint8)

    bounds = tuple(tuple(float(value) for value in axis) for axis in cfg["hull_bounds_clip_m"])
    local_grid, hull_shape = local_voxel_centers(bounds, float(cfg["hull_voxel_size_m"]))
    grid_base = transform_points(estimate.T_base_clip, local_grid)
    hull_views = [view.hull_view() for view in views]
    occupancy = carve_visual_hull(
        grid_base,
        hull_views,
        min_views=min(int(cfg["hull_min_views"]), len(hull_views)),
        inside_fraction=float(cfg["hull_inside_fraction"]),
    )
    return fused_base, fused_colors, local_grid[occupancy], hull_shape, [value for axis in bounds for value in axis]


def heldout_hull_iou(
    views: list[LoadedView],
    T_base_clip: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    cfg = merged_config(config)
    if len(views) < 4:
        return 0.0, {}
    heldout_indices = set(range(2, len(views), 3)) or {len(views) - 1}
    training = [view for index, view in enumerate(views) if index not in heldout_indices]
    heldout = [view for index, view in enumerate(views) if index in heldout_indices]
    bounds = tuple(tuple(float(value) for value in axis) for axis in cfg["hull_bounds_clip_m"])
    local_grid, _ = local_voxel_centers(bounds, float(cfg["hull_voxel_size_m"]))
    grid_base = transform_points(T_base_clip, local_grid)
    keep = carve_visual_hull(
        grid_base,
        [view.hull_view() for view in training],
        min_views=min(int(cfg["hull_min_views"]), len(training)),
        inside_fraction=float(cfg["hull_inside_fraction"]),
    )
    hull_base = grid_base[keep]
    scores = {}
    for view in heldout:
        rendered = rasterize_silhouette(hull_base, view.hull_view())
        scores[view.sample_id] = mask_iou(rendered, view.mask)
    return (float(np.mean(list(scores.values()))) if scores else 0.0), scores


def write_overlays(results_dir: Path, views: list[LoadedView], points: dict[str, np.ndarray]) -> None:
    overlay_dir = results_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    colors = {"left_lip": (255, 100, 40), "right_lip": (40, 220, 255), "seat_center": (80, 255, 80)}
    for view in views:
        image = cv2.imread(str(view.roi_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        contours, _ = cv2.findContours(view.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 200, 255), 2)
        x, y, _, _ = view.roi_xywh
        for name, point in points.items():
            projected, valid = project_camera_points(
                transform_points(invert_transform(view.T_base_camera), point[None, :]),
                view.K_rgb,
                view.distortion_rgb,
            )
            if valid[0]:
                center = (int(round(projected[0, 0] - x)), int(round(projected[0, 1] - y)))
                cv2.drawMarker(image, center, colors[name], cv2.MARKER_CROSS, 24, 2)
                cv2.putText(image, name, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)
        cv2.imwrite(str(overlay_dir / f"{view.sample_id}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_failure_report(results_dir: Path, valid_views: int, failures: list[str]) -> dict[str, Any]:
    report = {"accepted": False, "valid_views": int(valid_views), "failures": failures}
    write_json(results_dir / "report.json", report)
    (results_dir / "FAILED_report.md").write_text(
        "# Hybrid reconstruction failed\n\n"
        + "\n".join(f"- {failure}" for failure in failures)
        + "\n",
        encoding="utf-8",
    )
    return report


def run_hybrid_reconstruction(
    session_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir).expanduser().resolve()
    cfg = merged_config(config)
    results_dir = session / "hybrid" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        views = load_hybrid_views(session)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return write_failure_report(results_dir, 0, [str(exc)])
    failures = []
    if len(views) < int(cfg["minimum_views"]):
        failures.append(f"valid views {len(views)} < {int(cfg['minimum_views'])}")
    if failures:
        return write_failure_report(results_dir, len(views), failures)

    try:
        estimate = estimate_subset(views, cfg)
        even = estimate_subset(views[::2], {**cfg, "minimum_keypoint_views": 2})
        odd = estimate_subset(views[1::2], {**cfg, "minimum_keypoint_views": 2})
    except (FileNotFoundError, ValueError) as exc:
        return write_failure_report(results_dir, len(views), [str(exc)])
    split_translation, split_rotation = pose_difference(even.T_base_clip, odd.T_base_clip)
    leave_one_out = []
    for index in range(len(views)):
        subset = views[:index] + views[index + 1 :]
        try:
            candidate = estimate_subset(subset, cfg)
        except ValueError:
            continue
        leave_one_out.append(float(np.linalg.norm(candidate.T_base_clip[:3, 3] - estimate.T_base_clip[:3, 3])))
    loo_origin_max = max(leave_one_out, default=float("inf"))
    reprojection_rmse = float(np.sqrt(np.mean(estimate.normalized_reprojection_errors_px**2)))
    heldout_iou, heldout_scores = heldout_hull_iou(views, estimate.T_base_clip, cfg)
    fused_base, fused_colors, hull_clip, hull_shape, hull_bounds = build_geometry(views, estimate, cfg)

    if reprojection_rmse > float(cfg["max_reprojection_rmse_px_1280"]):
        failures.append(f"normalized reprojection RMSE {reprojection_rmse:.3f}px exceeds limit")
    if split_translation > float(cfg["max_split_translation_m"]):
        failures.append(f"split translation {split_translation:.6f}m exceeds limit")
    if split_rotation > float(cfg["max_split_rotation_deg"]):
        failures.append(f"split rotation {split_rotation:.3f}deg exceeds limit")
    if loo_origin_max > float(cfg["max_leave_one_out_origin_m"]):
        failures.append(f"leave-one-out origin {loo_origin_max:.6f}m exceeds limit")
    if estimate.plane.rms_m > float(cfg["plane_max_rms_m"]):
        failures.append(f"support plane RMS {estimate.plane.rms_m:.6f}m exceeds limit")
    if heldout_iou < float(cfg["minimum_heldout_mask_iou"]):
        failures.append(f"held-out mask IoU {heldout_iou:.3f} below limit")

    accepted = not failures
    report = {
        "accepted": accepted,
        "valid_views": len(views),
        "normalized_reprojection_rmse_px_1280": reprojection_rmse,
        "split_translation_m": split_translation,
        "split_rotation_deg": split_rotation,
        "leave_one_out_origin_max_m": loo_origin_max,
        "support_plane_rms_m": estimate.plane.rms_m,
        "support_plane_inliers": estimate.plane.inlier_count,
        "heldout_mask_iou": heldout_iou,
        "heldout_mask_iou_per_view": heldout_scores,
        "fused_depth_points": int(len(fused_base)),
        "visual_hull_voxels": int(len(hull_clip)),
        "failures": failures,
    }
    write_json(results_dir / "report.json", report)
    write_yaml(
        results_dir / "support_plane.yaml",
        {
            "point_base_m": estimate.plane.point.tolist(),
            "normal_base_outward": estimate.plane.normal.tolist(),
            "rms_m": estimate.plane.rms_m,
            "inlier_count": estimate.plane.inlier_count,
        },
    )
    points_clip = (
        transform_points(invert_transform(estimate.T_base_clip), fused_base)
        if len(fused_base)
        else fused_base
    )
    np.savez_compressed(
        results_dir / "clip_geometry.npz",
        depth_points_base=fused_base,
        depth_points_clip=points_clip,
        depth_colors=fused_colors,
        visual_hull_points_clip=hull_clip,
        visual_hull_shape=np.asarray(hull_shape),
        visual_hull_bounds=np.asarray(hull_bounds),
        T_base_clip=estimate.T_base_clip,
    )
    combined_clip = np.vstack([points_clip, hull_clip]) if len(points_clip) else hull_clip
    write_point_cloud_ply(results_dir / "clip_points.ply", combined_clip)
    write_overlays(results_dir, views, estimate.points)

    if accepted:
        write_yaml(
            results_dir / "clip_pose.yaml",
            {
                "accepted": True,
                "parent_frame": load_manifest(session / "hybrid" / "manifest.json")["base_frame"],
                "child_frame": "clip_task",
                "T_base_clip": estimate.T_base_clip.tolist(),
                "metrics": report,
                "functional_points_base_m": {name: point.tolist() for name, point in estimate.points.items()},
            },
        )
    else:
        (results_dir / "FAILED_report.md").write_text(
            "# Hybrid reconstruction failed\n\n" + "\n".join(f"- {failure}" for failure in failures) + "\n",
            encoding="utf-8",
        )
    return report
