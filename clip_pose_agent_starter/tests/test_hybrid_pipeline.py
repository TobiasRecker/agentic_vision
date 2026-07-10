from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clip_pose_pipeline.hybrid_geometry import encode_depth_mm  # noqa: E402
from clip_pose_pipeline.hybrid_io import prepare_hybrid_manifest, read_yaml, save_depth_artifact  # noqa: E402
from clip_pose_pipeline.hybrid_pipeline import run_hybrid_reconstruction  # noqa: E402
from clip_pose_pipeline.transforms import rotation_angle_deg  # noqa: E402


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, forward])
    T[:3, 3] = position
    return T


def project(point: np.ndarray, T_base_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = np.linalg.inv(T_base_camera) @ np.r_[point, 1.0]
    return np.array([K[0, 0] * camera[0] / camera[2] + K[0, 2], K[1, 1] * camera[1] / camera[2] + K[1, 2]])


def plane_depth(T_base_camera: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    rays_camera = np.stack(
        [(cols - K[0, 2]) / K[0, 0], (rows - K[1, 2]) / K[1, 1], np.ones_like(cols)], axis=-1
    )
    rays_base = rays_camera @ T_base_camera[:3, :3].T
    distance = -T_base_camera[2, 3] / rays_base[:, :, 2]
    depth = distance.astype(np.float64)
    depth[(distance <= 0.0) | ~np.isfinite(distance)] = np.nan
    return depth


def create_synthetic_session(session: Path) -> np.ndarray:
    width, height = 400, 300
    K = np.array([[520.0, 0.0, 200.0], [0.0, 520.0, 150.0], [0.0, 0.0, 1.0]])
    seat = np.array([0.0, 0.0, 0.025])
    functional = {
        "left_lip": np.array([0.0, -0.018, 0.020]),
        "right_lip": np.array([0.0, 0.018, 0.020]),
        "seat_center": seat,
    }
    image_dir = session / "images"
    depth_dir = session / "depth"
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    poses = []
    annotations: dict[str, object] = {"version": 1, "views": {}}
    for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False), start=1):
        sample_id = f"{index:06d}"
        position = np.array([0.14 * np.cos(angle), 0.14 * np.sin(angle), 0.32])
        T = look_at(position, seat)
        image = np.full((height, width, 3), 25, dtype=np.uint8)
        center = project(seat, T, K)
        cv2.circle(image, tuple(np.rint(center).astype(int)), 28, (220, 220, 220), -1)
        image_path = image_dir / f"{sample_id}.jpg"
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 97])
        depth_path = depth_dir / f"{sample_id}.png"
        cv2.imwrite(str(depth_path), encode_depth_mm(plane_depth(T, K, width, height)))
        with (depth_dir / f"{sample_id}.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                {
                    "K": K.tolist(),
                    "T_base_camera": T.tolist(),
                    "image_width": width,
                    "image_height": height,
                    "depth_scale_m": 0.001,
                },
                stream,
            )
        poses.append(
            {
                "image": f"images/{sample_id}.jpg",
                "camera_frame": "synthetic_camera",
                "T_base_camera": T.tolist(),
                "focus_roi_xywh": [0, 0, width, height],
            }
        )
    with (session / "camera_poses.json").open("w", encoding="utf-8") as stream:
        json.dump({"frames": {"base": "base", "camera": "synthetic_camera"}, "poses": poses}, stream)
    with (session / "intrinsics.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"image_width": width, "image_height": height, "K": K.tolist()}, stream)
    prepare_hybrid_manifest(session, roi_size_px=400)
    for index, pose in enumerate(poses, start=1):
        sample_id = f"{index:06d}"
        mask = np.zeros((height, width), dtype=np.uint8)
        center = project(seat, np.asarray(pose["T_base_camera"]), K)
        cv2.circle(mask, tuple(np.rint(center).astype(int)), 28, 255, -1)
        cv2.imwrite(str(session / "hybrid" / "masks" / f"{sample_id}.png"), mask)
        annotations["views"][sample_id] = {
            "mask": f"hybrid/masks/{sample_id}.png",
            "roi_xywh": [0, 0, width, height],
            "keypoints_full": {
                name: project(point, np.asarray(pose["T_base_camera"]), K).tolist()
                for name, point in functional.items()
            },
        }
    with (session / "hybrid" / "annotations.json").open("w", encoding="utf-8") as stream:
        json.dump(annotations, stream)
    expected = np.eye(4)
    expected[:3, 3] = seat
    return expected


def test_synthetic_end_to_end_recovers_task_frame(tmp_path: Path) -> None:
    expected = create_synthetic_session(tmp_path)
    report = run_hybrid_reconstruction(
        tmp_path,
        {
            "depth_stride": 8,
            "hull_bounds_clip_m": [[-0.025, 0.025], [-0.035, 0.035], [-0.002, 0.045]],
            "hull_voxel_size_m": 0.004,
            "minimum_heldout_mask_iou": 0.0,
        },
    )

    assert report["accepted"], report["failures"]
    with (tmp_path / "hybrid" / "results" / "clip_pose.yaml").open("r", encoding="utf-8") as stream:
        pose = np.asarray(yaml.safe_load(stream)["T_base_clip"], dtype=np.float64)
    translation_error = np.linalg.norm(pose[:3, 3] - expected[:3, 3])
    rotation_error = rotation_angle_deg(expected[:3, :3], pose[:3, :3])
    assert translation_error < 0.002
    assert rotation_error < 1.0
    assert (tmp_path / "hybrid" / "results" / "clip_geometry.npz").is_file()
    assert (tmp_path / "hybrid" / "results" / "clip_points.ply").is_file()


def test_incomplete_session_writes_failure_report(tmp_path: Path) -> None:
    create_synthetic_session(tmp_path)
    (tmp_path / "depth" / "000001.png").unlink()

    report = run_hybrid_reconstruction(tmp_path, {"hull_voxel_size_m": 0.01})

    assert not report["accepted"]
    assert report["failures"]
    assert (tmp_path / "hybrid" / "results" / "FAILED_report.md").is_file()


def test_capture_depth_artifact_writes_png_and_calibration(tmp_path: Path) -> None:
    xyz = np.zeros((4, 5, 3), dtype=np.float64)
    xyz[:, :, 2] = 0.4564
    K = np.array([[200.0, 0.0, 2.0], [0.0, 200.0, 1.5], [0.0, 0.0, 1.0]])

    depth_path, metadata_path = save_depth_artifact(
        tmp_path,
        "000007",
        xyz,
        K,
        np.eye(4),
        "camera",
        {"sec": 1, "nanosec": 2},
        {"sec": 1, "nanosec": 3},
    )

    encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    metadata = read_yaml(metadata_path)
    assert encoded.dtype == np.uint16
    assert int(encoded[0, 0]) == 456
    np.testing.assert_allclose(metadata["K"], K)
    assert metadata["cloud_stamp"]["nanosec"] == 3
