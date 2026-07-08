from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .transforms import as_transform


@dataclass(frozen=True)
class CameraIntrinsics:
    K: np.ndarray
    distortion: np.ndarray
    image_width: int
    image_height: int
    camera_name: str = "camera"


@dataclass(frozen=True)
class CameraView:
    image: str
    T_base_camera: np.ndarray


@dataclass(frozen=True)
class DetectionObservation:
    image: str
    keypoints: dict[str, tuple[float, float, float]]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = load_yaml(path)
    K = np.asarray(data["K"], dtype=float)
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")
    distortion = np.asarray(data.get("distortion", [0, 0, 0, 0, 0]), dtype=float).reshape(-1)
    return CameraIntrinsics(
        K=K,
        distortion=distortion,
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        camera_name=str(data.get("camera_name", "camera")),
    )


def load_camera_views(path: str | Path) -> list[CameraView]:
    data = load_json(path)
    poses = data.get("poses", []) if isinstance(data, dict) else []
    views: list[CameraView] = []
    for item in poses:
        views.append(
            CameraView(
                image=str(item["image"]),
                T_base_camera=as_transform(item["T_base_camera"]),
            )
        )
    return views


def load_clip_keypoints(path: str | Path) -> dict[str, np.ndarray]:
    data = load_yaml(path)
    units = data.get("units", "m")
    scale = 1.0
    if units == "mm":
        scale = 1e-3
    elif units != "m":
        raise ValueError(f"Unsupported clip model units: {units}")
    keypoints = data.get("keypoints", {})
    if not keypoints:
        raise ValueError("clip_model.yaml must define at least four keypoints")
    return {name: np.asarray(value, dtype=float).reshape(3) * scale for name, value in keypoints.items()}


def load_manual_detections(path: str | Path) -> dict[str, DetectionObservation]:
    raw = load_json(path)
    observations: dict[str, DetectionObservation] = {}
    for image, entry in raw.items():
        kp_raw = entry.get("keypoints", {})
        keypoints: dict[str, tuple[float, float, float]] = {}
        for name, uv_conf in kp_raw.items():
            if len(uv_conf) == 2:
                u, v = uv_conf
                conf = 1.0
            else:
                u, v, conf = uv_conf[:3]
            keypoints[name] = (float(u), float(v), float(conf))
        observations[str(image)] = DetectionObservation(image=str(image), keypoints=keypoints)
    return observations
