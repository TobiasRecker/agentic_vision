from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .hybrid_io import load_annotations, load_manifest, write_json


KEY_BINDINGS = {"l": "left_lip", "r": "right_lip", "s": "seat_center"}
KEYPOINT_COLORS = {
    "left_lip": (255, 120, 40),
    "right_lip": (40, 220, 255),
    "seat_center": (80, 255, 80),
}


def automatic_dark_background_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    blurred = cv2.GaussianBlur(value, (5, 5), 0)
    _threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return binary
    center = np.array([0.5 * image.shape[1], 0.5 * image.shape[0]], dtype=np.float64)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 20 or area > int(0.5 * image.shape[0] * image.shape[1]):
            continue
        distance = float(np.linalg.norm(centroids[label] - center))
        candidates.append((distance - 0.01 * np.sqrt(area), label))
    if not candidates:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    selected = min(candidates)[1]
    return np.where(labels == selected, 255, 0).astype(np.uint8)


class HybridAnnotationGui:
    def __init__(self, session_dir: str | Path) -> None:
        self.session = Path(session_dir).expanduser().resolve()
        self.manifest = load_manifest(self.session / "hybrid" / "manifest.json")
        self.annotations_path = self.session / "hybrid" / "annotations.json"
        self.annotations = load_annotations(self.session)
        self.views = self.manifest["views"]
        if not self.views:
            raise ValueError("hybrid manifest contains no views")
        self.index = 0
        self.window = "clip_hybrid_annotation"
        self.image = np.empty((0, 0, 3), dtype=np.uint8)
        self.mask = np.empty((0, 0), dtype=np.uint8)
        self.polygon: list[tuple[int, int]] = []
        self.pending_keypoint: str | None = None
        self.keypoints_full: dict[str, list[float]] = {}
        self.load_view()

    @property
    def view(self) -> dict[str, Any]:
        return self.views[self.index]

    def load_view(self) -> None:
        roi_path = self.session / str(self.view["roi"])
        image = cv2.imread(str(roi_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(roi_path)
        self.image = image
        mask_path = self.session / str(self.view["mask"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.mask = mask if mask is not None else automatic_dark_background_mask(image)
        entry = self.annotations.get("views", {}).get(str(self.view["sample_id"]), {})
        self.keypoints_full = {
            name: [float(value[0]), float(value[1])]
            for name, value in entry.get("keypoints_full", {}).items()
            if name in KEYPOINT_COLORS
        }
        self.polygon = []
        self.pending_keypoint = None

    def save_view(self) -> None:
        mask_path = self.session / str(self.view["mask"])
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), self.mask, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise RuntimeError(f"could not write {mask_path}")
        sample_id = str(self.view["sample_id"])
        self.annotations.setdefault("views", {})[sample_id] = {
            "mask": str(self.view["mask"]),
            "roi_xywh": [int(value) for value in self.view["roi_xywh"]],
            "keypoints_full": self.keypoints_full,
        }
        write_json(self.annotations_path, self.annotations)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.polygon:
                self.polygon.pop()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.pending_keypoint:
            roi_x, roi_y, _, _ = self.view["roi_xywh"]
            self.keypoints_full[self.pending_keypoint] = [float(x + roi_x), float(y + roi_y)]
            self.pending_keypoint = None
        else:
            self.polygon.append((int(x), int(y)))

    def commit_polygon(self) -> None:
        if len(self.polygon) < 3:
            return
        self.mask[:] = 0
        cv2.fillPoly(self.mask, [np.asarray(self.polygon, dtype=np.int32)], 255)
        self.polygon = []

    def render(self) -> np.ndarray:
        view = self.image.copy()
        tint = np.zeros_like(view)
        tint[:, :, 1] = self.mask
        view = cv2.addWeighted(view, 1.0, tint, 0.25, 0.0)
        if self.polygon:
            cv2.polylines(view, [np.asarray(self.polygon, dtype=np.int32)], False, (0, 255, 255), 2)
            for point in self.polygon:
                cv2.circle(view, point, 4, (0, 255, 255), -1)
        roi_x, roi_y, _, _ = self.view["roi_xywh"]
        for name, pixel in self.keypoints_full.items():
            center = (int(round(pixel[0] - roi_x)), int(round(pixel[1] - roi_y)))
            cv2.drawMarker(view, center, KEYPOINT_COLORS[name], cv2.MARKER_CROSS, 28, 2)
            cv2.putText(
                view,
                name,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                KEYPOINT_COLORS[name],
                1,
            )
        lines = [
            f"{self.index + 1}/{len(self.views)} sample {self.view['sample_id']}",
            f"mode: {self.pending_keypoint or 'mask polygon'}",
        ]
        for row, text in enumerate(lines):
            y = 28 + row * 28
            cv2.putText(view, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        try:
            while True:
                cv2.imshow(self.window, self.render())
                key = cv2.waitKeyEx(20)
                if key < 0:
                    continue
                char = chr(key).lower() if 0 <= key < 128 else ""
                if char == "q" or key == 27:
                    self.save_view()
                    break
                if key in (10, 13):
                    self.commit_polygon()
                elif char == "a":
                    self.mask = automatic_dark_background_mask(self.image)
                elif char == "x":
                    self.mask[:] = 0
                    self.polygon = []
                elif char in KEY_BINDINGS:
                    self.pending_keypoint = KEY_BINDINGS[char]
                elif char == "w":
                    self.save_view()
                elif char in ("n", "p"):
                    self.save_view()
                    self.index = (self.index + (1 if char == "n" else -1)) % len(self.views)
                    self.load_view()
        finally:
            cv2.destroyWindow(self.window)
