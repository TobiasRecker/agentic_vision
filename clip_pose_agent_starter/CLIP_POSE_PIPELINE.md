# Multi-view Clip Pose Pipeline

## Core idea

The clip pose should not be estimated by a language model. The robust version is:

```text
robot FK + hand-eye  ->  T_base_camera_i
camera image_i       ->  2D clip mask/keypoints
clip geometry        ->  3D keypoints / silhouette model
--------------------------------------------------------
multi-view optimizer ->  T_base_clip + validation report
```

The agent's job is to improve detection parameters and code until the geometric estimate becomes self-consistent across all camera poses.

## Why this works

The clip is static during one calibration run. Each image gives a constraint on the same unknown transform `T_base_clip`. If the image detections and robot/camera poses are correct, all views should agree on the same clip pose. This gives a useful automatic quality criterion:

```text
high-quality result:  all views explain one common T_base_clip
bad result:           each view implies a different clip pose
```

## Data capture protocol

1. Move the wrist camera to 12-30 poses around the clip.
2. Keep the clip fixed during the capture.
3. Vary azimuth, elevation, distance, and roll slightly.
4. Avoid all views being coplanar or from only one side.
5. Store for every image:
   - image path,
   - timestamp,
   - `T_base_camera`,
   - camera intrinsics version,
   - robot and hand-eye calibration version.

Recommended view distribution:

```text
- distance: 0.25-0.60 m
- azimuth: left / center / right
- elevation: above / level / below if feasible
- roll: 0 deg and +/- 20 deg
- at least 6 valid views after filtering
```

## Required inputs

### `intrinsics.yaml`

```yaml
camera_name: wrist_left
image_width: 1280
image_height: 720
K:
  - [900.0, 0.0, 640.0]
  - [0.0, 900.0, 360.0]
  - [0.0, 0.0, 1.0]
distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
```

### `camera_poses.json`

`T_base_camera` transforms points from camera frame into base frame.

```json
{
  "frames": {
    "base": "robot_base",
    "camera": "wrist_camera_optical"
  },
  "poses": [
    {
      "image": "images/000001.png",
      "T_base_camera": [[1,0,0,0.4],[0,1,0,0.0],[0,0,1,0.3],[0,0,0,1]]
    }
  ]
}
```

### `clip_model.yaml`

Start with a sparse functional model. Replace the example values with measured/CAD values.

```yaml
frame: clip
units: m
keypoints:
  left_lip:  [-0.0125, 0.0000, 0.0000]
  right_lip: [ 0.0125, 0.0000, 0.0000]
  saddle:    [ 0.0000, 0.0000, 0.0000]
  back_left: [-0.0125, 0.0000, -0.0180]
  back_right:[ 0.0125, 0.0000, -0.0180]
```

### `detections.json`

This is the first backend to implement because it isolates geometry from detection.

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

## Estimation objective

For each image `i` and keypoint `j`:

```text
p_clip_j        known 3D model point
T_base_camera_i known from robot + hand-eye
T_base_clip     unknown
u_ij            detected 2D keypoint
```

Projection:

```text
T_camera_clip_i = inverse(T_base_camera_i) @ T_base_clip
u_hat_ij = project(K, distortion, T_camera_clip_i, p_clip_j)
```

Optimize:

```text
min_T_base_clip  sum_i,j  rho( || u_ij - u_hat_ij || )
```

Use a robust loss to reduce the effect of bad detections.

## Detector strategy

Recommended development order:

1. `manual_keypoints`: hand-click keypoints for 10-20 images; validate geometry.
2. `classical`: fixed setup, controlled background, HSV/edge/contour detector.
3. `grounded_sam`: optional robust detector using text-guided box proposal and promptable segmentation.
4. `cad_silhouette`: use masks to align a CAD or simplified mesh model.

For the first real experiments, the manual/keypoint mode is enough to verify the full calibration chain.

## Agent tuning parameters

The agent may tune:

```yaml
roi: [x, y, width, height]
hsv_lower: [h, s, v]
hsv_upper: [h, s, v]
canny_thresholds: [low, high]
contour_min_area: 100
contour_max_area: 20000
morph_kernel: 3
box_threshold: 0.30
text_threshold: 0.25
prompt: "plastic pipe clip. conduit clamp. cable conduit holder."
```

The agent should prefer narrower ROIs and better lighting over complex model changes.

## Outputs

### `clip_pose.yaml`

```yaml
T_base_clip:
  - [1.0, 0.0, 0.0, 0.450]
  - [0.0, 1.0, 0.0, 0.020]
  - [0.0, 0.0, 1.0, 0.310]
  - [0.0, 0.0, 0.0, 1.000]
metrics:
  reprojection_rmse_px: 1.8
  split_translation_error_m: 0.0011
  split_rotation_error_deg: 0.7
  valid_views: 14
```

### `report.json`

Contains per-view errors, accepted/rejected views, and parameter settings.

### overlays

For every image, write a debug overlay with detected keypoints/mask and projected model keypoints.

## Failure interpretation

- High error in all images: wrong intrinsics, wrong hand-eye, wrong frame convention, or wrong clip model scale.
- Good per-view PnP but bad global consistency: wrong `T_base_camera_i` or timestamp mismatch.
- Some views fail strongly: detection/occlusion issue; reject those views.
- Good reprojection but wrong physical pose: model frame definition or sign convention wrong.

## Best first experiment

1. Use a fixed marker or manually clicked keypoints to validate camera intrinsics and hand-eye.
2. Capture 15 views.
3. Click 5 functional keypoints in each image.
4. Run multi-view optimizer.
5. Check whether the projected keypoints align in all views.
6. Only then add automatic detection.
