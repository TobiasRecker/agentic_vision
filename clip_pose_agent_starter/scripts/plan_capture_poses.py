#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def look_at(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return T_base_camera with OpenCV optical frame roughly looking at target."""
    z = target - camera_position
    z = z / np.linalg.norm(z)
    down_guess = np.array([0.0, 0.0, -1.0])
    x = np.cross(down_guess, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = camera_position
    return T


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate nominal camera look-at poses around a target.")
    parser.add_argument("--target", nargs=3, type=float, default=[0.45, 0.0, 0.30])
    parser.add_argument("--radius", type=float, default=0.40)
    parser.add_argument("--out", default="results/capture_plan.json")
    args = parser.parse_args()

    target = np.asarray(args.target, dtype=float)
    poses = []
    idx = 0
    for az_deg in [-45, -25, 0, 25, 45]:
        for el_deg in [-20, 0, 20]:
            az = math.radians(az_deg)
            el = math.radians(el_deg)
            pos = target + args.radius * np.array(
                [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
            )
            T = look_at(pos, target)
            poses.append({"name": f"view_{idx:03d}", "T_base_camera_nominal": T.tolist()})
            idx += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"poses": poses}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
