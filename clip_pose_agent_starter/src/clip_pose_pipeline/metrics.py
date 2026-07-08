from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import CameraIntrinsics
from .pose_fit import MatchedView, fit_clip_pose
from .transforms import rotation_angle_deg


@dataclass(frozen=True)
class SplitConsistency:
    translation_error_m: float
    rotation_error_deg: float
    success: bool
    message: str


def split_half_consistency(
    matched_views: list[MatchedView],
    intr: CameraIntrinsics,
    robust_loss: str,
    f_scale_px: float,
    max_nfev: int,
) -> SplitConsistency:
    if len(matched_views) < 4:
        return SplitConsistency(float("inf"), float("inf"), False, "Need at least 4 views for split-half consistency")
    even = matched_views[::2]
    odd = matched_views[1::2]
    if len(even) < 2 or len(odd) < 2:
        return SplitConsistency(float("inf"), float("inf"), False, "Split has too few views")
    try:
        fit_a = fit_clip_pose(even, intr, robust_loss, f_scale_px, max_nfev)
        fit_b = fit_clip_pose(odd, intr, robust_loss, f_scale_px, max_nfev)
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        return SplitConsistency(float("inf"), float("inf"), False, str(exc))
    dt = float(np.linalg.norm(fit_a.T_base_clip[:3, 3] - fit_b.T_base_clip[:3, 3]))
    dR = rotation_angle_deg(fit_a.T_base_clip[:3, :3], fit_b.T_base_clip[:3, :3])
    return SplitConsistency(dt, dR, True, "ok")


def accepted(report: dict, acceptance: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = [
        ("reprojection_rmse_px", "reprojection_rmse_px"),
        ("median_view_error_px", "median_view_error_px"),
        ("split_translation_error_m", "split_translation_error_m"),
        ("split_rotation_error_deg", "split_rotation_error_deg"),
    ]
    for report_key, limit_key in checks:
        value = float(report.get(report_key, float("inf")))
        limit = float(acceptance.get(limit_key, float("inf")))
        if value > limit:
            failures.append(f"{report_key}={value:.6g} > {limit:.6g}")
    if int(report.get("valid_views", 0)) < int(acceptance.get("min_valid_views", 1)):
        failures.append(
            f"valid_views={report.get('valid_views', 0)} < {acceptance.get('min_valid_views', 1)}"
        )
    return len(failures) == 0, failures
