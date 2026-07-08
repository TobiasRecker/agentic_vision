# AGENTS.md

## Mission
Build and maintain a deterministic multi-view calibration pipeline that estimates the fixed pose of a plastic conduit clip / pipe clamp in the robot base frame.

The agent must **not** use an LLM/VLM as the source of the metric pose. The agent may use vision foundation models only to generate masks, boxes, or keypoint proposals. The final pose must be produced by calibrated geometry and validated numerically.

## Problem statement
Given:

- calibrated camera intrinsics `K` and distortion coefficients,
- one image per wrist-camera pose,
- known camera poses `T_base_camera_i` from robot forward kinematics and hand-eye calibration,
- a known or approximate clip model with 3D keypoints or silhouette geometry,

estimate:

- `T_base_clip`, the pose of the clip task frame in the robot/world/base frame.

The clip task frame should be defined functionally:

- origin: center of the pipe seat / insertion target,
- `x_clip`: pipe axis in the inserted state,
- `z_clip`: insertion / press-in direction,
- `y_clip`: lateral correction axis completing a right-handed frame.

## Frame conventions
Use these conventions everywhere:

- `T_A_B` transforms homogeneous points from frame `B` into frame `A`.
- Composition: `p_A = T_A_B @ p_B`.
- Camera frame is OpenCV optical frame: `x` right, `y` down, `z` forward.
- Units are SI: meters, radians, seconds, Newtons.
- Rotation vectors are OpenCV Rodrigues vectors unless explicitly stated otherwise.
- YAML matrices are row-major 4x4 homogeneous transforms.

Never silently change these conventions. If input files use another convention, add explicit conversion code and document it.

## Repository layout
Expected layout:

```text
configs/default.yaml                  # main configuration
src/clip_pose_pipeline/               # importable Python package
scripts/estimate_clip_pose.py         # main offline estimator
scripts/tune_clip_pose.py             # automatic parameter sweep / repair loop
scripts/plan_capture_poses.py         # helper to generate camera viewpoints
examples/                             # tiny data-format examples only
results/                              # generated outputs; do not commit large data
```

## Required pipeline stages
1. Load images, camera intrinsics, distortion, and `T_base_camera_i`.
2. Detect the clip in each image using one selected backend:
   - `manual_keypoints`: read 2D keypoints from JSON; use this first for validation.
   - `classical`: OpenCV threshold/contour/edge pipeline with tunable parameters.
   - `grounded_sam`: Grounding DINO box proposal plus SAM/SAM2 segmentation; optional dependency.
3. Convert detections into either:
   - 2D keypoints corresponding to known 3D model points, or
   - masks/silhouette constraints for a CAD/model-based objective.
4. Estimate `T_base_clip` with multi-view optimization.
5. Validate by reprojection error, split-view consistency, and per-view pose consistency.
6. Write outputs only if acceptance criteria are met.

## Acceptance criteria
Default acceptance gates:

- multi-view reprojection RMSE: `< 3.0 px`,
- median per-view reprojection error: `< 4.0 px`,
- split-half translation difference: `< 0.002 m`,
- split-half rotation difference: `< 2.0 deg`,
- at least `6` valid views,
- at least `4` valid keypoints per accepted view.

If these fail, do not overwrite the last valid `clip_pose.yaml`. Write `results/FAILED_report.md` explaining what failed and what to try next.

## Agent repair loop
When asked to process a new data session:

1. Inspect `configs/default.yaml` and the session metadata.
2. Run:
   ```bash
   python scripts/estimate_clip_pose.py --config configs/default.yaml
   ```
3. Inspect `results/report.json` and generated overlays.
4. If validation fails, tune only the detector parameters first: ROI, thresholds, contour filters, prompt phrases, mask postprocessing, and confidence thresholds.
5. Re-run the estimator after each change.
6. Only modify geometry/model/frame code after detector tuning is clearly insufficient.
7. Add or update tests for every geometry or transform bug fixed.
8. Keep a log of parameter changes in `results/tuning_log.md`.

## Safety and correctness rules
- Never fabricate a pose from visual inspection.
- Never accept a result without numerical validation.
- Never mix millimeters and meters.
- Never overwrite raw input data.
- Never commit large images, model checkpoints, ROS bags, or generated result folders.
- Prefer explicit errors over silent fallback behavior.
- When uncertain about hand-eye or pose convention, create a small diagnostic script and verify with a known marker or calibration target.

## Development commands
Install local package:

```bash
python -m pip install -e .
```

Run estimator:

```bash
python scripts/estimate_clip_pose.py --config configs/default.yaml
```

Run parameter sweep:

```bash
python scripts/tune_clip_pose.py --config configs/default.yaml
```

Run tests if present:

```bash
python -m pytest -q
```
