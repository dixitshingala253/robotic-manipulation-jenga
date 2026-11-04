import cv2
import numpy as np
import glob
import os

# Calibration settings
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/camera2/flat_new"
OUTPUT_FILE = "calibration/camera2/camera_parameters_new.npz"

# Create ChArUco board
dictionary = cv2.aruco.Dictionary_get(
    cv2.aruco.DICT_4X4_100
)

board = cv2.aruco.CharucoBoard_create(
    COLS,
    ROWS,
    SQUARE_SIZE,
    MARKER_SIZE,
    dictionary
)

parameters = cv2.aruco.DetectorParameters_create()

all_corners = []
all_ids = []
image_size = None

images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

print("Total images:", len(images))

for filename in images:
    image = cv2.imread(filename)

    if image is None:
        print("Could not read:", filename)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=parameters
    )

    if ids is None:
        print("No markers detected:", filename)
        continue

    _, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
    )

    if charuco_corners is not None and len(charuco_corners) >= 8:
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        image_size = gray.shape[::-1]
        print(filename, ":", len(charuco_corners), "corners")
    else:
        print("Not enough corners:", filename)

print("\nValid calibration images:", len(all_corners))

if len(all_corners) < 10:
    print("ERROR: Not enough valid images.")
    raise SystemExit

# Calculate intrinsic parameters
retval, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None
    )
)

print("\n========== CAMERA 1 RESULTS ==========")
print("Reprojection error:", retval)
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)

# Save calibration parameters
np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_width=image_size[0],
    image_height=image_size[1],
    reprojection_error=retval
)

print("\nCalibration saved to:", OUTPUT_FILE)
