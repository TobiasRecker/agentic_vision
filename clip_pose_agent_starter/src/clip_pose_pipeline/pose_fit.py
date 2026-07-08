from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .io import CameraIntrinsics, CameraView, DetectionObservation
from .transforms import invert_transform, pose_from_rvec_t, rvec_t_from_pose


@dataclass(frozen=True)
class MatchedView:
    view: CameraView
    object_points: np.ndarray
    image_points: np.ndarray
    confidences: np.ndarray


@dataclass(frozen=True)
class PoseFitResult:
    T_base_clip: np.ndarray
    reprojection_rmse_px: float
    median_view_error_px: float
    per_view_rmse_px: dict[str, float]
    valid_views: int
    valid_observations: int
    success: bool
    message: str


def build_matched_views(
    views: list[CameraView],
    detections: dict[str, DetectionObservation],
    model_keypoints: dict[str, np.ndarray],
    confidence_threshold: float,
    min_keypoints_per_view: int,
) -> list[MatchedView]:
    matched: list[MatchedView] = []
    for view in views:
        det = detections.get(view.image)
        if det is None:
            det = detections.get(view.image.split("/")[-1])
        if det is None:
            continue

        object_points = []
        image_points = []
        confidences = []
        for name, point_3d in model_keypoints.items():
            if name not in det.keypoints:
                continue
            u, v, conf = det.keypoints[name]
            if conf < confidence_threshold:
                continue
            object_points.append(point_3d)
            image_points.append([u, v])
            confidences.append(conf)

        if len(object_points) >= min_keypoints_per_view:
            matched.append(
                MatchedView(
                    view=view,
                    object_points=np.asarray(object_points, dtype=float),
                    image_points=np.asarray(image_points, dtype=float),
                    confidences=np.asarray(confidences, dtype=float),
                )
            )
    return matched


def project_points(
    object_points_clip: np.ndarray,
    T_base_clip: np.ndarray,
    T_base_camera: np.ndarray,
    intr: CameraIntrinsics,
) -> np.ndarray:
    T_camera_base = invert_transform(T_base_camera)
    T_camera_clip = T_camera_base @ T_base_clip
    rvec, tvec = rvec_t_from_pose(T_camera_clip)
    projected, _ = cv2.projectPoints(
        object_points_clip.astype(float),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intr.K,
        intr.distortion,
    )
    return projected.reshape(-1, 2)


def initial_pose_from_view(matched: MatchedView, intr: CameraIntrinsics) -> np.ndarray | None:
    if matched.object_points.shape[0] < 4:
        return None
    ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
        matched.object_points.astype(float),
        matched.image_points.astype(float),
        intr.K,
        intr.distortion,
        flags=cv2.SOLVEPNP_SQPNP if matched.object_points.shape[0] >= 3 else cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None
    T_camera_clip = pose_from_rvec_t(rvec, tvec)
    return matched.view.T_base_camera @ T_camera_clip


def median_initial_pose(matched_views: list[MatchedView], intr: CameraIntrinsics) -> np.ndarray:
    candidates = []
    for mv in matched_views:
        T = initial_pose_from_view(mv, intr)
        if T is not None and np.all(np.isfinite(T)):
            candidates.append(T)
    if not candidates:
        raise RuntimeError("Could not compute initial pose from any view. Need >=4 reliable keypoints per view.")

    # Translation median; rotation from first candidate. This is only an initialization.
    translations = np.asarray([T[:3, 3] for T in candidates])
    T0 = candidates[0].copy()
    T0[:3, 3] = np.median(translations, axis=0)
    return T0


def residual_vector(params: np.ndarray, matched_views: list[MatchedView], intr: CameraIntrinsics) -> np.ndarray:
    T_base_clip = pose_from_rvec_t(params[:3], params[3:6])
    residuals = []
    for mv in matched_views:
        projected = project_points(mv.object_points, T_base_clip, mv.view.T_base_camera, intr)
        r = (projected - mv.image_points).reshape(-1)
        weights = np.repeat(np.sqrt(np.clip(mv.confidences, 1e-3, 1.0)), 2)
        residuals.append(r * weights)
    if not residuals:
        return np.zeros(0)
    return np.concatenate(residuals)


def compute_view_errors(T_base_clip: np.ndarray, matched_views: list[MatchedView], intr: CameraIntrinsics) -> dict[str, float]:
    errors = {}
    for mv in matched_views:
        projected = project_points(mv.object_points, T_base_clip, mv.view.T_base_camera, intr)
        err = np.linalg.norm(projected - mv.image_points, axis=1)
        errors[mv.view.image] = float(np.sqrt(np.mean(err**2)))
    return errors


def fit_clip_pose(
    matched_views: list[MatchedView],
    intr: CameraIntrinsics,
    robust_loss: str = "soft_l1",
    f_scale_px: float = 3.0,
    max_nfev: int = 500,
) -> PoseFitResult:
    if not matched_views:
        raise ValueError("No matched views available")

    T0 = median_initial_pose(matched_views, intr)
    r0, t0 = rvec_t_from_pose(T0)
    x0 = np.r_[r0, t0]

    result = least_squares(
        residual_vector,
        x0,
        args=(matched_views, intr),
        loss=robust_loss,
        f_scale=f_scale_px,
        max_nfev=max_nfev,
    )

    T_base_clip = pose_from_rvec_t(result.x[:3], result.x[3:6])
    per_view = compute_view_errors(T_base_clip, matched_views, intr)
    all_residuals = residual_vector(result.x, matched_views, intr).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(all_residuals**2, axis=1)))) if len(all_residuals) else float("inf")
    median_view = float(np.median(list(per_view.values()))) if per_view else float("inf")

    return PoseFitResult(
        T_base_clip=T_base_clip,
        reprojection_rmse_px=rmse,
        median_view_error_px=median_view,
        per_view_rmse_px=per_view,
        valid_views=len(matched_views),
        valid_observations=int(sum(len(mv.object_points) for mv in matched_views)),
        success=bool(result.success),
        message=str(result.message),
    )
