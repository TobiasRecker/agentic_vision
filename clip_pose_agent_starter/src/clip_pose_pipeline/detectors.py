from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .io import DetectionObservation, load_manual_detections


class Detector(Protocol):
    def detect(self, image_path: Path) -> DetectionObservation | None:
        ...


@dataclass
class ManualKeypointDetector:
    detections: dict[str, DetectionObservation]

    @classmethod
    def from_file(cls, path: Path) -> "ManualKeypointDetector":
        return cls(load_manual_detections(path))

    def detect(self, image_path: Path) -> DetectionObservation | None:
        # Match both full relative path and filename because camera_poses.json may use either.
        candidates = [str(image_path), image_path.as_posix(), image_path.name]
        for key in candidates:
            if key in self.detections:
                return self.detections[key]
        return None


@dataclass
class ClassicalMaskDetector:
    """Simple HSV/edge mask detector placeholder.

    This backend returns masks only. Convert mask to task-specific keypoints before using it for pose.
    It is intentionally conservative; tune it with debug overlays before trusting it.
    """

    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    morph_kernel: int = 3
    roi: tuple[int, int, int, int] | None = None

    def mask(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        if self.roi is not None:
            x, y, w, h = self.roi
            crop = image[y : y + h, x : x + w]
        else:
            crop = image
            x = y = 0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.hsv_lower), np.array(self.hsv_upper))
        if self.morph_kernel > 1:
            kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if self.roi is None:
            return mask
        full = np.zeros(image.shape[:2], dtype=np.uint8)
        full[y : y + h, x : x + w] = mask
        return full

    def detect(self, image_path: Path) -> DetectionObservation | None:
        raise NotImplementedError(
            "ClassicalMaskDetector currently produces masks only. Add mask-to-keypoint logic for your clip geometry."
        )


class GroundedSamDetector:
    """Optional detector adapter for Grounding DINO + SAM/SAM2.

    Keep this as an adapter so the core geometry pipeline remains usable without GPU/model checkpoints.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "Install and wire optional GroundingDINO/SAM2 dependencies in this adapter. "
            "The estimator should still be validated with manual_keypoints first."
        )


def build_detector(config: dict, session_dir: Path) -> Detector:
    backend = config.get("backend", "manual_keypoints")
    if backend == "manual_keypoints":
        return ManualKeypointDetector.from_file(session_dir / config["detections_file"])
    if backend == "classical":
        roi = config.get("roi")
        return ClassicalMaskDetector(
            hsv_lower=tuple(config.get("hsv_lower", [0, 0, 0])),
            hsv_upper=tuple(config.get("hsv_upper", [179, 255, 255])),
            morph_kernel=int(config.get("morph_kernel", 3)),
            roi=tuple(roi) if roi is not None else None,
        )
    if backend == "grounded_sam":
        return GroundedSamDetector(config)
    raise ValueError(f"Unknown detector backend: {backend}")
