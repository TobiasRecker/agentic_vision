# Data format

## `camera_poses.json`

`T_base_camera` must transform points from the OpenCV optical camera frame into the robot base frame.

```json
{
  "frames": {
    "base": "robot_base",
    "camera": "wrist_camera_optical"
  },
  "poses": [
    {
      "image": "images/000001.png",
      "timestamp": 123.456,
      "T_base_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ]
}
```

## `detections.json`

For first validation, manually annotate keypoints. Confidence is optional and defaults to 1.0.

```json
{
  "images/000001.png": {
    "keypoints": {
      "left_lip": [620.0, 330.0, 1.0],
      "right_lip": [690.0, 332.0, 1.0],
      "saddle": [655.0, 355.0, 1.0],
      "back_left": [625.0, 390.0, 1.0],
      "back_right": [685.0, 391.0, 1.0]
    }
  }
}
```

## `clip_model.yaml`

Keypoints are in the clip task frame.

```yaml
frame: clip
units: m
keypoints:
  left_lip: [-0.0125, 0.0000, 0.0000]
  right_lip: [0.0125, 0.0000, 0.0000]
  saddle: [0.0000, 0.0000, 0.0000]
  back_left: [-0.0125, 0.0000, -0.0180]
  back_right: [0.0125, 0.0000, -0.0180]
```
