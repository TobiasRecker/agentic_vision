from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .hybrid_geometry import encode_depth_mm


@dataclass(frozen=True)
class HybridViewRecord:
    sample_id: str
    image: str
    image_size: tuple[int, int]
    roi: str
    roi_xywh: tuple[int, int, int, int]
    mask: str
    depth: str | None
    depth_metadata: str | None
    K_rgb: list[list[float]]
    distortion_rgb: list[float]
    T_base_camera: list[list[float]]
    camera_frame: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def camera_matrix(data: dict[str, Any]) -> np.ndarray:
    raw = data.get("K", data.get("k", []))
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.size != 9:
        raise ValueError("camera metadata must contain a 3x3 K matrix")
    return matrix.reshape(3, 3)


def scaled_camera_matrix(K: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    result = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    result[0, :] *= float(target_width) / float(source_width)
    result[1, :] *= float(target_height) / float(source_height)
    result[2, :] = [0.0, 0.0, 1.0]
    return result


def resolve_rgb_intrinsics(
    session_dir: Path,
    pose: dict[str, Any],
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    sidecar = Path(str(image_path) + ".camera_info.yaml")
    if sidecar.is_file():
        data = read_yaml(sidecar)
        return camera_matrix(data), np.asarray(data.get("D", data.get("distortion", [])), dtype=np.float64)
    intrinsics_path = session_dir / "intrinsics.yaml"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"missing RGB intrinsics: {intrinsics_path}")
    intrinsics = read_yaml(intrinsics_path)
    K = camera_matrix(intrinsics)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    target_size = (int(image.shape[1]), int(image.shape[0]))
    source_size = (
        int(intrinsics.get("image_width", target_size[0])),
        int(intrinsics.get("image_height", target_size[1])),
    )
    matrix = K if source_size == target_size else scaled_camera_matrix(K, source_size, target_size)
    return matrix, np.asarray(intrinsics.get("D", intrinsics.get("distortion", [])), dtype=np.float64)


def choose_roi(pose: dict[str, Any], image_size: tuple[int, int], default_size_px: int) -> tuple[int, int, int, int]:
    width, height = image_size
    if "focus_roi_xywh" in pose:
        x, y, roi_width, roi_height = (int(value) for value in pose["focus_roi_xywh"])
        return (
            max(0, min(x, width - 1)),
            max(0, min(y, height - 1)),
            max(1, min(roi_width, width - x)),
            max(1, min(roi_height, height - y)),
        )
    pixel = pose.get("fullres_pixel_uv", [0.5 * width, 0.5 * height])
    roi_width = min(int(default_size_px), width)
    roi_height = min(int(default_size_px), height)
    x = int(np.clip(round(float(pixel[0]) - 0.5 * roi_width), 0, width - roi_width))
    y = int(np.clip(round(float(pixel[1]) - 0.5 * roi_height), 0, height - roi_height))
    return x, y, roi_width, roi_height


def prepare_hybrid_manifest(
    session_dir: str | Path,
    roi_size_px: int = 1200,
    overwrite: bool = False,
) -> Path:
    session = Path(session_dir).expanduser().resolve()
    pose_path = session / "camera_poses.json"
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    raw_poses = read_json(pose_path)
    poses = raw_poses.get("poses", [])
    if not poses:
        raise ValueError(f"{pose_path} contains no camera poses")

    hybrid_dir = session / "hybrid"
    roi_dir = hybrid_dir / "roi"
    mask_dir = hybrid_dir / "masks"
    roi_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[HybridViewRecord] = []

    for index, pose in enumerate(poses, start=1):
        image_rel = str(pose["image"])
        image_path = session / image_rel
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image_size = (int(image.shape[1]), int(image.shape[0]))
        sample_id = Path(image_rel).stem or f"{index:06d}"
        roi_xywh = choose_roi(pose, image_size, roi_size_px)
        x, y, width, height = roi_xywh
        captured_roi = session / str(pose.get("rgb_roi", "")) if pose.get("rgb_roi") else None
        roi_path = captured_roi if captured_roi is not None and captured_roi.is_file() else roi_dir / f"{sample_id}.png"
        if roi_path.parent == roi_dir and (overwrite or not roi_path.is_file()):
            if not cv2.imwrite(str(roi_path), image[y : y + height, x : x + width], [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise RuntimeError(f"could not write {roi_path}")
        mask_path = mask_dir / f"{sample_id}.png"
        depth_candidates = [session / "depth" / f"{sample_id}.png", hybrid_dir / "depth" / f"{sample_id}.png"]
        depth_path = next((candidate for candidate in depth_candidates if candidate.is_file()), None)
        metadata_candidates = [
            session / "depth" / f"{sample_id}.yaml",
            hybrid_dir / "depth" / f"{sample_id}.yaml",
        ]
        depth_metadata = next((candidate for candidate in metadata_candidates if candidate.is_file()), None)
        K_rgb, distortion_rgb = resolve_rgb_intrinsics(session, pose, image_path)
        records.append(
            HybridViewRecord(
                sample_id=sample_id,
                image=image_rel,
                image_size=image_size,
                roi=relative_path(roi_path, session),
                roi_xywh=roi_xywh,
                mask=relative_path(mask_path, session),
                depth=relative_path(depth_path, session) if depth_path else None,
                depth_metadata=relative_path(depth_metadata, session) if depth_metadata else None,
                K_rgb=K_rgb.tolist(),
                distortion_rgb=distortion_rgb.reshape(-1).tolist(),
                T_base_camera=np.asarray(pose["T_base_camera"], dtype=np.float64).reshape(4, 4).tolist(),
                camera_frame=str(pose.get("camera_frame", raw_poses.get("frames", {}).get("camera", "camera"))),
            )
        )

    manifest = {
        "version": 1,
        "session_dir": str(session),
        "base_frame": str(raw_poses.get("frames", {}).get("base", "base_link")),
        "camera_frame": str(raw_poses.get("frames", {}).get("camera", records[0].camera_frame)),
        "reference_width_px": 1280,
        "views": [asdict(record) for record in records],
    }
    manifest_path = hybrid_dir / "manifest.json"
    write_json(manifest_path, manifest)
    annotations_path = hybrid_dir / "annotations.json"
    if not annotations_path.exists():
        write_json(annotations_path, {"version": 1, "views": {}})
    return manifest_path


def save_depth_artifact(
    session_dir: str | Path,
    sample_id: str,
    xyz_image: np.ndarray,
    K_depth: np.ndarray,
    T_base_camera: np.ndarray,
    camera_frame: str,
    image_stamp: dict[str, int],
    cloud_stamp: dict[str, int],
) -> tuple[Path, Path]:
    session = Path(session_dir).expanduser().resolve()
    depth_dir = session / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz_image, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        raise ValueError("xyz_image must have shape HxWx3")
    depth_path = depth_dir / f"{sample_id}.png"
    metadata_path = depth_dir / f"{sample_id}.yaml"
    if not cv2.imwrite(str(depth_path), encode_depth_mm(xyz[:, :, 2]), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"could not write {depth_path}")
    write_yaml(
        metadata_path,
        {
            "encoding": "16UC1",
            "depth_scale_m": 0.001,
            "invalid_value": 0,
            "image_width": int(xyz.shape[1]),
            "image_height": int(xyz.shape[0]),
            "K": np.asarray(K_depth, dtype=np.float64).reshape(3, 3).tolist(),
            "T_base_camera": np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4).tolist(),
            "camera_frame": camera_frame,
            "image_stamp": image_stamp,
            "cloud_stamp": cloud_stamp,
        },
    )
    return depth_path, metadata_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    data = read_json(manifest_path)
    if int(data.get("version", 0)) != 1 or not isinstance(data.get("views"), list):
        raise ValueError(f"unsupported hybrid manifest: {manifest_path}")
    return data


def load_annotations(session_dir: str | Path) -> dict[str, Any]:
    path = Path(session_dir).expanduser().resolve() / "hybrid" / "annotations.json"
    return read_json(path) if path.is_file() else {"version": 1, "views": {}}


def write_point_cloud_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rgb = None if colors is None else np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype(np.uint8)
    if rgb is not None and len(rgb) != len(xyz):
        raise ValueError("points and colors must have the same length")
    with output.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for index, point in enumerate(xyz):
            values = f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}"
            if rgb is not None:
                values += f" {rgb[index, 0]} {rgb[index, 1]} {rgb[index, 2]}"
            stream.write(values + "\n")
