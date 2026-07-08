#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running directly from a fresh checkout before installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clip_pose_pipeline.detectors import build_detector  # noqa: E402
from clip_pose_pipeline.io import (  # noqa: E402
    DetectionObservation,
    load_camera_views,
    load_clip_keypoints,
    load_intrinsics,
    load_yaml,
    write_json,
    write_yaml,
)
from clip_pose_pipeline.metrics import accepted, split_half_consistency  # noqa: E402
from clip_pose_pipeline.pose_fit import build_matched_views, fit_clip_pose  # noqa: E402


def matrix_to_list(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in T]


def resolve_session_path(session_dir: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else session_dir / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate T_base_clip from multi-view keypoints.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()

    session_dir = resolve_session_path(project_root, config["session_dir"]).resolve()
    output_dir = resolve_session_path(project_root, config.get("output_dir", "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    intr = load_intrinsics(session_dir / config["intrinsics_file"])
    views = load_camera_views(session_dir / config["camera_poses_file"])
    model_path = resolve_session_path(session_dir, config["clip_model_file"])
    model_keypoints = load_clip_keypoints(model_path)

    detector = build_detector(config.get("detector", {}), session_dir)
    detections: dict[str, DetectionObservation] = {}
    for view in views:
        obs = detector.detect(session_dir / view.image)
        if obs is not None:
            detections[view.image] = obs

    opt_cfg = config.get("optimizer", {})
    matched = build_matched_views(
        views=views,
        detections=detections,
        model_keypoints=model_keypoints,
        confidence_threshold=float(opt_cfg.get("confidence_threshold", 0.5)),
        min_keypoints_per_view=int(opt_cfg.get("min_keypoints_per_view", 4)),
    )

    if len(matched) < int(opt_cfg.get("min_views", 1)):
        report = {
            "success": False,
            "message": "Too few valid views",
            "valid_views": len(matched),
            "required_views": int(opt_cfg.get("min_views", 1)),
        }
        write_json(output_dir / "report.json", report)
        (output_dir / "FAILED_report.md").write_text(
            f"# Clip pose estimation failed\n\nToo few valid views: {len(matched)}.\n",
            encoding="utf-8",
        )
        print(f"FAILED: too few valid views ({len(matched)})")
        return 2

    fit = fit_clip_pose(
        matched_views=matched,
        intr=intr,
        robust_loss=str(opt_cfg.get("robust_loss", "soft_l1")),
        f_scale_px=float(opt_cfg.get("f_scale_px", 3.0)),
        max_nfev=int(opt_cfg.get("max_nfev", 500)),
    )
    split = split_half_consistency(
        matched,
        intr,
        robust_loss=str(opt_cfg.get("robust_loss", "soft_l1")),
        f_scale_px=float(opt_cfg.get("f_scale_px", 3.0)),
        max_nfev=int(opt_cfg.get("max_nfev", 500)),
    )

    report = {
        "success": fit.success,
        "message": fit.message,
        "T_base_clip": matrix_to_list(fit.T_base_clip),
        "reprojection_rmse_px": fit.reprojection_rmse_px,
        "median_view_error_px": fit.median_view_error_px,
        "split_translation_error_m": split.translation_error_m,
        "split_rotation_error_deg": split.rotation_error_deg,
        "split_success": split.success,
        "split_message": split.message,
        "valid_views": fit.valid_views,
        "valid_observations": fit.valid_observations,
        "per_view_rmse_px": fit.per_view_rmse_px,
        "detector_backend": config.get("detector", {}).get("backend", "manual_keypoints"),
        "model_keypoints": list(model_keypoints.keys()),
    }
    ok, failures = accepted(report, config.get("acceptance", {}))
    report["accepted"] = ok
    report["acceptance_failures"] = failures
    write_json(output_dir / "report.json", report)

    if ok:
        write_yaml(
            output_dir / "clip_pose.yaml",
            {
                "T_base_clip": matrix_to_list(fit.T_base_clip),
                "metrics": {
                    "reprojection_rmse_px": fit.reprojection_rmse_px,
                    "median_view_error_px": fit.median_view_error_px,
                    "split_translation_error_m": split.translation_error_m,
                    "split_rotation_error_deg": split.rotation_error_deg,
                    "valid_views": fit.valid_views,
                    "valid_observations": fit.valid_observations,
                },
            },
        )
        print(f"OK: wrote {output_dir / 'clip_pose.yaml'}")
        return 0

    (output_dir / "FAILED_report.md").write_text(
        "# Clip pose estimation failed\n\n"
        + "\n".join(f"- {failure}" for failure in failures)
        + "\n\nSee `report.json` for details.\n",
        encoding="utf-8",
    )
    print("FAILED:")
    for failure in failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
