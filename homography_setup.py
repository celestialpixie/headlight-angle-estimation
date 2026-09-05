import cv2
import numpy as np
import rawpy
import os

# ---- CHANGE THIS ----
img_path = "trialroad_photo.dng"

# ---- Known rectangle dimensions ----
AB_width = 0.96   # meters, A to B
BC_depth = 1.94   # meters, B to C

world_points = np.array([
    [0.0,        0.0],
    [AB_width,   0.0],
    [AB_width,   BC_depth],
    [0.0,        BC_depth],
], dtype=np.float32)

# 3D versions of the same points, ground plane => Y = 0
# (X, Z) from world_points map to (X, 0, Z) in 3D world coords.
object_points_3d = np.hstack([
    world_points[:, 0:1],
    np.zeros((4, 1), dtype=np.float32),
    world_points[:, 1:2],
]).astype(np.float32)

# ---- Load calibration ----
data = np.load("calibration_data.npz")
K, dist = data["camera_matrix"], data["dist_coeffs"]


# ---- Load image (handles DNG and standard formats) ----
def load_image_as_bgr(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    ext = os.path.splitext(abs_path)[1].lower()
    if ext == ".dng":
        with rawpy.imread(abs_path) as raw:
            rgb = raw.postprocess()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.imread(abs_path)
    if img is None:
        raise IOError(f"OpenCV could not decode file: {abs_path}")
    return img


img = load_image_as_bgr(img_path)

max_w = 1200
scale = min(1.0, max_w / img.shape[1])
disp = cv2.resize(img, None, fx=scale, fy=scale)
clicked_points = []
labels = ["A (closest-left)", "B (closest-right)", "C (farthest-right)", "D (farthest-left)"]


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        ox, oy = int(x / scale), int(y / scale)
        clicked_points.append((ox, oy))
        print(f"{labels[len(clicked_points)-1]}: u={ox}, v={oy}")
        cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(disp, labels[len(clicked_points)-1][0], (x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Click A, B, C, D in order", disp)


cv2.imshow("Click A, B, C, D in order", disp)
cv2.setMouseCallback("Click A, B, C, D in order", click)
print("Click points in order: A (closest-left), B (closest-right), C (farthest-right), D (farthest-left)")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) < 4:
    print("Need all 4 points. Exiting.")
else:
    image_points_raw = np.array(clicked_points, dtype=np.float32)

    pts = image_points_raw.reshape(-1, 1, 2)
    undistorted_pts = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)

    # ---- Homography (still useful for mapping any ground pixel -> (X,Z)) ----
    H, status = cv2.findHomography(undistorted_pts, world_points)
    print("\nHomography matrix:\n", H)
    print("Inlier status:", status.ravel())

    # ---- Full extrinsic pose (R, t): world -> camera ----
    # This is what actually fixes the "camera not aligned with world Z" problem.
    # undistorted_pts are already in pixel coords (P=K was used above), and since
    # they're already undistorted we pass dist=None here.
    ok, rvec, tvec = cv2.solvePnP(object_points_3d, undistorted_pts, K, None)
    if not ok:
        raise RuntimeError("solvePnP failed to find a pose. Check your point correspondences.")
    R, _ = cv2.Rodrigues(rvec)

    # Sanity check: recover camera's position in world coordinates.
    cam_pos_world = (-R.T @ tvec).ravel()
    print(f"\nRecovered camera position in world frame (X, Y=height, Z): {cam_pos_world}")
    print("Compare cam_pos_world[1] (height) to your tape-measured camera height as a sanity check.")

    np.savez("homography_data.npz", H=H, R=R, tvec=tvec, rvec=rvec)
    print("\nSaved homography + extrinsics to homography_data.npz")

    def pixel_to_road_position(u, v):
        pt = np.array([[[u, v]]], dtype=np.float32)
        undist = cv2.undistortPoints(pt, K, dist, P=K)
        ux, uy = undist[0, 0]
        vec = np.array([ux, uy, 1.0])
        mapped = H @ vec
        return mapped[0] / mapped[2], mapped[1] / mapped[2]

    print("\nSanity check (recovered vs expected):")
    for i, (u, v) in enumerate(clicked_points):
        X, Z = pixel_to_road_position(u, v)
        print(f"{labels[i]}: got ({X:.3f}, {Z:.3f})  expected ({world_points[i][0]:.3f}, {world_points[i][1]:.3f})")


    # ---- Independent validation point (not used to build H) ----
    print("\n--- Independent validation ---")
    print("Click one more point on the same road plane with a KNOWN real-world position")
    print("(e.g. midpoint of edge AB, or another measured spot)")

    val_points = []

    def click_val(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(val_points) < 1:
            ox, oy = int(x / scale), int(y / scale)
            val_points.append((ox, oy))
            print(f"Validation point: u={ox}, v={oy}")
            cv2.destroyAllWindows()

    disp2 = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow("Click validation point", disp2)
    cv2.setMouseCallback("Click validation point", click_val)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if val_points:
        vu, vv = val_points[0]
        X_val, Z_val = pixel_to_road_position(vu, vv)
        print(f"Homography says: X={X_val:.3f} m, Z={Z_val:.3f} m")
        print("Compare this to what you actually measured for that point.")