# Clip Pose Agent Starter

Starter scaffold for estimating the pose of a plastic conduit clip from multiple calibrated wrist-camera views.

Main entry point:

```bash
python -m pip install -e .
python scripts/estimate_clip_pose.py --config configs/default.yaml
```

Start with `manual_keypoints` detections to validate geometry. Add automatic detection only after the multi-view optimization works with manual labels.

ROS2 capture GUI:

```bash
colcon build --packages-select clip_pose_agent_starter
source install/setup.bash
ros2 launch clip_pose_agent_starter mur620_clip_capture.launch.py
```

The capture GUI writes RGB images, intrinsics, `T_base_camera`, and the clicked RGBD anchor to `~/clip_pose_sessions/<timestamp>/`.

The first anchor click also creates a visible default object ROI. Press `r`, then drag a new
rectangle in the live image to adjust it. The ROI is retained for the session and follows the
projected 3D anchor in later views. Every sample records the effective ROI coordinates and saves
the crop losslessly below `roi/`.

Convenience runner with build, source, launch, and logging:

```bash
./src/agentic_vision/clip_pose_agent_starter/scripts/run_clip_capture.sh
```

Override launch values with environment variables, for example:

```bash
IMAGE_TOPIC=/oak/rgb/image_raw IMAGE_COMPRESSED=false MOVE_ENABLED=true \
  ./src/agentic_vision/clip_pose_agent_starter/scripts/run_clip_capture.sh
```

### Full-resolution capture and focus

The launch file also starts the idle `oak4_fullres_capture` helper. Pressing `c` freezes the
current RGB-D observation and robot pose, maps the tracked preview ROI into the full sensor
image, and uses it as the autofocus ROI. The requested photo, its full-resolution
intrinsics, the RGB ROI, and the synchronized depth artifact are committed only when every
required file has been validated. Manual focus is available through `fullres_focus_mode: manual`;
the requested lens position is estimated from the clicked point's camera-frame depth.
The running `/oak` driver currently exposes the manual-focus enable flag but not the required
integer `rgb.r_focus` position. Click focus therefore performs a capability check and leaves the
live preview focus unchanged on this driver instead of partially applying an invalid request.


Direct `Camera` pipelines currently crash this OAK 4 D R9 reproducibly in device firmware with
Luxonis OS RVC4 1.32.1 / Agent 0.20.0, including tests at 8000x6000, 4000x3000, and 1920x1080.
Therefore `fullres_hardware_enabled` and `fullres_allow_fallback` default to `false`. With
`fullres_required: true`, `c` reports a clear error and does not silently save the 1280x720
preview. Enable `fullres_hardware_enabled` only after the direct pipeline has been validated with
an updated Luxonis stack. To explicitly return to preview-only capture, set both
`fullres_enabled: false` and `fullres_required: false`; resulting samples are marked
`preview_explicit`.

As a temporary 4K path, the capture node can save the running ROS RGB stream. First set the
anchor while OAK is in `RGBD + PointCloud` mode. Then switch OAK Settings to `RGB`,
`3840x2160 @ 5 Hz`, compressed, with a high transport quality and restart OAK. The capture node
keeps the anchor. Pressing `c` writes the original JPEG payload without recompression, a lossless
ROI PNG, intrinsics, and the current camera pose. Images below 3840x2160 are rejected explicitly
while `ros_stream_fallback_enabled` is active.

Each accepted image also gets a matching `poses/000001.json` sidecar. It contains
`T_base_camera`, translation/quaternion, the TCP pose, frame names, and timestamps. The aggregate
`camera_poses.json` references the sidecar through `pose_json`.

## Hybrid clip reconstruction

The hybrid pipeline estimates a functional clip task frame without CAD. It combines manually
verified full-resolution masks/keypoints, the organized OAK depth image, known camera poses, and
the visible planar mounting surface.

Prepare a captured session, annotate it, and reconstruct it with the logged runner:

```bash
SESSION_DIR=~/clip_pose_sessions/<timestamp> \
  ./src/agentic_vision/clip_pose_agent_starter/scripts/run_hybrid_clip_pipeline.sh prepare

SESSION_DIR=~/clip_pose_sessions/<timestamp> \
  ./src/agentic_vision/clip_pose_agent_starter/scripts/run_hybrid_clip_pipeline.sh annotate

SESSION_DIR=~/clip_pose_sessions/<timestamp> \
  ./src/agentic_vision/clip_pose_agent_starter/scripts/run_hybrid_clip_pipeline.sh reconstruct
```

Annotation controls:

- Left-click adds mask polygon vertices; right-click removes the last vertex; Enter applies it.
- `a` proposes a foreground mask for a bright clip on a dark surface; `x` clears it.
- `l`, `r`, `s` select `left_lip`, `right_lip`, or `seat_center`; the next click places it.
- `n` / `p` changes view, `w` saves, and `q` saves and exits.

Generated inputs stay below `<session>/hybrid/`; raw captures are never overwritten. Accepted
results are written to `<session>/hybrid/results/clip_pose.yaml`, `clip_geometry.npz`, and
`clip_points.ply`. Failed quality gates produce `report.json` and `FAILED_report.md` without
replacing a previously accepted pose.

Publish an accepted result as a ROS2 static transform:

```bash
ros2 run clip_pose_agent_starter clip_pose_static_tf --ros-args \
  -p pose_file:=~/clip_pose_sessions/<timestamp>/hybrid/results/clip_pose.yaml
```

The equivalent build/source/log runner is:

```bash
POSE_FILE=~/clip_pose_sessions/<timestamp>/hybrid/results/clip_pose.yaml \
  ./src/agentic_vision/clip_pose_agent_starter/scripts/run_clip_pose_static_tf.sh
```
