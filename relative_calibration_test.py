import cv2
import numpy as np
import os

# ChArUco board configuration
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/four_camera_pairs"

# ZED SDK rectified LEFT camera matrices
K1 = np.array([
    [723.456665, 0, 972.615051],
    [0, 723.456665, 596.518066],
    [0, 0, 1]
], dtype=np.float64)

K2 = np.array([
    [723.755676, 0, 945.402527],
    [0, 723.755676, 608.364807],
    [0, 0, 1]
], dtype=np.float64)

# Rectified images have zero distortion
DIST = np.zeros((5, 1), dtype=np.float64)

dictionary = cv2.aruco.Dictionary_get(
    cv2.aruco.DICT_4X4_100
)

board = cv2.aruco.CharucoBoard_create(
    COLS, ROWS, SQUARE_SIZE, MARKER_SIZE, dictionary
)

parameters = cv2.aruco.DetectorParameters_create()


def detect_charuco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=parameters
    )

    if ids is None:
        return {}

    _, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
    )

    if charuco_ids is None:
        return {}

    return {
        int(corner_id): point[0]
        for corner_id, point in zip(
            charuco_ids.flatten(), charuco_corners
        )
    }


def estimate_pose(points, camera_matrix, object_points):
    image_points = np.array(
        [points[i] for i in object_points.keys()],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    board_points = np.array(
        [object_points[i] for i in object_points],
        dtype=np.float32
    )

    success, rvec, tvec = cv2.solvePnP(
        board_points,
        image_points,
        camera_matrix,
        DIST,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    projected, _ = cv2.projectPoints(
        board_points, rvec, tvec, camera_matrix, DIST
    )

    error = np.sqrt(
        np.mean(np.sum(
            (projected.reshape(-1, 2) -
             image_points.reshape(-1, 2)) ** 2,
            axis=1
        ))
    )

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T, error


print("\nStarting relative pose estimation...\n")

for number in range(1, 23):

    if number == 5:
        continue

    pair = f"pair_{number:03d}"

    path1 = os.path.join(
        IMAGE_DIR, f"{pair}_cam1_left.png"
    )
    path2 = os.path.join(
        IMAGE_DIR, f"{pair}_cam2_left.png"
    )

    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    if image1 is None or image2 is None:
        print(pair, "- MISSING IMAGE")
        continue

    corners1 = detect_charuco(image1)
    corners2 = detect_charuco(image2)

    common = sorted(
        set(corners1) & set(corners2)
    )

    if len(common) < 8:
        print(pair, "- SKIPPED: only", len(common),
              "common corners")
        continue

    # Board coordinates indexed by ChArUco ID
    object_points = {
        i: board.chessboardCorners[i]
        for i in common
    }

    pose1 = estimate_pose(
        corners1, K1, object_points
    )

    pose2 = estimate_pose(
        corners2, K2, object_points
    )

    if pose1 is None or pose2 is None:
        print(pair, "- PnP FAILED")
        continue

    T1, error1 = pose1
    T2, error2 = pose2

    # Camera 1 coordinates -> Camera 2 coordinates
    T21 = T2 @ np.linalg.inv(T1)

    R21 = T21[:3, :3]
    t21 = T21[:3, 3]

    rotation_vector, _ = cv2.Rodrigues(R21)
    rotation_angle = np.degrees(
        np.linalg.norm(rotation_vector)
    )

    print(f"\n{pair}")
    print("Common corners:", len(common))
    print(f"Cam1 reprojection error: {error1:.3f} px")
    print(f"Cam2 reprojection error: {error2:.3f} px")
    print(f"Relative translation (m): {t21}")
    print(f"Relative rotation angle: {rotation_angle:.3f} degrees")

print("\nPose estimation finished.")

