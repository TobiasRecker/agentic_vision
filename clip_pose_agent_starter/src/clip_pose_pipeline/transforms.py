from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Pose:
    """Homogeneous pose with convention T_A_B transforms points from B to A."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=float)
        if mat.shape != (4, 4):
            raise ValueError(f"Pose matrix must be 4x4, got {mat.shape}")
        object.__setattr__(self, "matrix", mat)

    @property
    def R(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "Pose":
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = t_inv
        return Pose(T)

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.matrix @ other.matrix)


def as_transform(matrix: object) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {mat.shape}")
    if not np.allclose(mat[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("Last row of homogeneous transform must be [0, 0, 0, 1]")
    return mat


def pose_from_rvec_t(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return T


def rvec_t_from_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = as_transform(T)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3), T[:3, 3].copy()


def invert_transform(T: np.ndarray) -> np.ndarray:
    return Pose(as_transform(T)).inverse().matrix


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    T_A_B = as_transform(T_A_B)
    points_B = np.asarray(points_B, dtype=float)
    if points_B.ndim != 2 or points_B.shape[1] != 3:
        raise ValueError("points_B must have shape (N, 3)")
    homo = np.c_[points_B, np.ones(len(points_B))]
    return (T_A_B @ homo.T).T[:, :3]
