import cv2
import numpy as np
import rawpy
import os

img_path = "trialroad_photo.dng"
camera_height_m = 1.65   # only used as a sanity-check print now, not in the math
object_height_m = 0.78

data = np.load("calibration_data.npz")
K, dist = data["camera_matrix"], data["dist_coeffs"]

hdata = np.load("homography_data.npz")
H = hdata["H"]
R = hdata["R"]        # world -> camera rotation, from solvePnP (in homography_setup.py)
tvec = hdata["tvec"]  # world -> camera translation, from solvePnP


def load_image_as_bgr(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dng":
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(path)


def pixel_to_ground_position(u, v):
    """Uses cv2.undistortPoints + cv2.perspectiveTransform (official calls only)."""
    pt = np.array([[[u, v]]], dtype=np.float32)
    undist = cv2.undistortPoints(pt, K, dist, P=K)          # official: distortion correction
    mapped = cv2.perspectiveTransform(undist, H)             # official: homography application
    X, Z = mapped[0, 0]
    return float(X), float(Z)


def theta_from_object_height(X, Z, h_obj, R, tvec):
    """
    Correct version: transforms the object's world point into the camera's
    OWN coordinate frame using the solvePnP pose (R, tvec), then measures
    the angle to [0,0,1] -- which is always the true optical axis in that
    frame, regardless of how the camera is offset/tilted relative to the
    world rectangle. Replaces the old forward=[0,0,1]-in-world-frame guess.
    """
    P_world = np.array([X, h_obj, Z], dtype=np.float64)
    P_cam = (R @ P_world.reshape(3, 1) + tvec).ravel()

    forward = np.array([0.0, 0.0, 1.0])
    distance = np.linalg.norm(P_cam)
    cos_theta = np.dot(P_cam, forward) / distance
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return theta_deg, distance


img = load_image_as_bgr(img_path)
print(f"Loaded image: {img_path}, shape: {img.shape}")

# Sanity check: recovered camera height vs. your tape-measured value
cam_pos_world = (-R.T @ tvec).ravel()
print(f"Recovered camera height from calibration: {cam_pos_world[1]:.3f} m "
      f"(you measured: {camera_height_m} m)")

max_w = 1200
scale = min(1.0, max_w / img.shape[1])
disp = cv2.resize(img, None, fx=scale, fy=scale)
clicked_point = []


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_point) == 0:
        ox, oy = int(x / scale), int(y / scale)
        clicked_point.append((ox, oy))
        print(f"Clicked ground contact point: u={ox}, v={oy}")
        cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click the object's GROUND CONTACT point", disp)


cv2.imshow("Click the object's GROUND CONTACT point", disp)
cv2.setMouseCallback("Click the object's GROUND CONTACT point", click)
print("Click where the object touches the ground.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if clicked_point:
    u, v = clicked_point[0]
    X, Z = pixel_to_ground_position(u, v)
    theta, distance = theta_from_object_height(X, Z, object_height_m, R, tvec)

    print(f"\nGround position: X={X:.3f} m, Z={Z:.3f} m")
    print(f"Theta (angle from camera's TRUE optical axis): {theta:.4f} deg")
    print(f"Distance: {distance:.3f} m")