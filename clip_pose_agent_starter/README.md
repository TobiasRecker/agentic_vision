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
