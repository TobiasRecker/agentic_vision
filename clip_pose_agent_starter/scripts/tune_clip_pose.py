#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple parameter sweep for detector/optimizer config.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base = load_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    output_dir = project_root / base.get("output_dir", "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_log = output_dir / "tuning_log.md"

    # Keep this small initially. Expand when automatic detectors exist.
    f_scales = [2.0, 3.0, 5.0]
    confidence_thresholds = [0.3, 0.5, 0.7]

    best = None
    trials = []
    for idx, (f_scale, conf) in enumerate(itertools.product(f_scales, confidence_thresholds)):
        cfg = deepcopy(base)
        cfg.setdefault("optimizer", {})["f_scale_px"] = f_scale
        cfg.setdefault("optimizer", {})["confidence_threshold"] = conf
        tmp_cfg = output_dir / f"_trial_{idx:03d}.yaml"
        write_yaml(tmp_cfg, cfg)
        cmd = [sys.executable, "scripts/estimate_clip_pose.py", "--config", str(tmp_cfg)]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True, check=False)
        report_path = output_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        score = float(report.get("reprojection_rmse_px", "inf"))
        accepted = bool(report.get("accepted", False))
        trial = {
            "idx": idx,
            "f_scale_px": f_scale,
            "confidence_threshold": conf,
            "score": score,
            "accepted": accepted,
            "returncode": proc.returncode,
        }
        trials.append(trial)
        if best is None or (accepted, -score) > (best["accepted"], -best["score"]):
            best = trial

    lines = ["# Tuning log", ""]
    for trial in trials:
        lines.append(
            f"- trial {trial['idx']}: f_scale={trial['f_scale_px']}, "
            f"conf={trial['confidence_threshold']}, rmse={trial['score']:.4g}, "
            f"accepted={trial['accepted']}"
        )
    if best is not None:
        lines.extend(["", f"Best trial: `{best}`"])
    tuning_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {tuning_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
