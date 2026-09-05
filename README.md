# Headlight Angle Estimation via Homography + PnP

Estimates the 3D angle (from a roadside camera's optical axis) to a
vehicle's headlights, using ground-plane homography and camera pose
estimation.

## Pipeline

1. **Camera calibration** (checkerboard, not included here) →
   produces `calibration_data.npz` containing:
   - `camera_matrix` (K) — intrinsic matrix
   - `dist_coeffs` — lens distortion coefficients

2. **`homography_setup.py`**
   - Click 4 known points (A, B, C, D) forming a rectangle of known
     real-world dimensions on the ground plane.
   - Undistorts the clicked points using `K`/`dist_coeffs`.
   - Solves for the homography `H` (image plane → ground plane) via
     `cv2.findHomography`.
   - Solves for the camera's full 3D pose (`R`, `tvec`) relative to
     the rectangle via `cv2.solvePnP`.
   - Saves `H`, `R`, `tvec`, `rvec` to `homography_data.npz`.

3. **`theta_from_object_height.py`**
   - Click the vehicle's ground contact point (directly below the
     headlight).
   - Maps that pixel to a ground-plane `(X, Z)` position using `H`.
   - Given the known headlight height above ground, builds the full
     3D world point and transforms it into the camera's coordinate
     frame using `R`, `tvec`.
   - Computes the angle between that point and the camera's optical
     axis (`[0, 0, 1]` in camera coordinates), plus the 3D distance.

## Requirements

See `requirements.txt`. Also needs `calibration_data.npz` (from your
own checkerboard calibration) present in the working directory.

## Usage

```bash
python homography_setup.py           # creates homography_data.npz
python theta_from_object_height.py   # prints angle + distance
```

Update the constants at the top of each script (`img_path`,
`AB_width`, `BC_depth`, `camera_height_m`, `object_height_m`) for
your specific setup.

## Notes

- All computer-vision steps use documented, stock OpenCV functions
  (`undistortPoints`, `findHomography`, `solvePnP`, `Rodrigues`,
  `perspectiveTransform`). The only non-library code is elementary
  vector geometry (dot product / norm) applied to their outputs.
- `.dng` (RAW) images are decoded via `rawpy`, since OpenCV can't
  read RAW formats natively.
