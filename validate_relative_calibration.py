import cv2
import numpy as np
import os

# ChArUco board configuration
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/four_camera_pairs"

# Camera 1 rectified LEFT intrinsic matrix
K1 = np.array([
    [723.456665, 0, 972.615051],
    [0, 723.456665, 596.518066],
    [0, 0, 1]
], dtype=np.float64)

# Camera 2 rectified LEFT intrinsic matrix
K2 = np.array([
    [723.755676, 0, 945.402527],
    [0, 723.755676, 608.364807],
    [0, 0, 1]
], dtype=np.float64)

DIST = np.zeros((5, 1), dtype=np.float64)

# Provisional transformation: Camera 1 -> Camera 2
T21 = np.array([
    [-0.20792769,  0.92219979, -0.32605771,  0.31309176],
    [-0.79109617,  0.03749828,  0.61054135, -0.19287956],
    [ 0.57526770,  0.38489145,  0.72175178, -0.18246710],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float64)

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


results = []
all_errors = []

print("\nStarting cross-camera validation...\n")

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

    common = sorted(set(corners1) & set(corners2))

    if len(common) < 8:
        print(pair, "- SKIPPED:", len(common), "common corners")
        continue

    object_points = np.array(
        [board.chessboardCorners[i] for i in common],
        dtype=np.float32
    )

    image_points1 = np.array(
        [corners1[i] for i in common],
        dtype=np.float32
    )

    image_points2 = np.array(
        [corners2[i] for i in common],
        dtype=np.float32
    )

    # Estimate board pose in Camera 1
    success, rvec1, tvec1 = cv2.solvePnP(
        object_points,
        image_points1,
        K1,
        DIST,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print(pair, "- PnP FAILED")
        continue

    R1, _ = cv2.Rodrigues(rvec1)

    T1B = np.eye(4)
    T1B[:3, :3] = R1
    T1B[:3, 3] = tvec1.flatten()

    # Transform board pose from Camera 1 to Camera 2
    T2B = T21 @ T1B

    R2 = T2B[:3, :3]
    t2 = T2B[:3, 3].reshape(3, 1)

    rvec2, _ = cv2.Rodrigues(R2)

    # Predict ChArUco corner positions in Camera 2
    projected, _ = cv2.projectPoints(
        object_points,
        rvec2,
        t2,
        K2,
        DIST
    )

    projected = projected.reshape(-1, 2)

    # Compare predictions with actual Camera 2 detections
    differences = projected - image_points2

    point_errors = np.linalg.norm(
        differences, axis=1
    )

    pair_rmse = np.sqrt(
        np.mean(point_errors ** 2)
    )

    pair_mean = np.mean(point_errors)
    pair_max = np.max(point_errors)

    all_errors.extend(point_errors.tolist())

    results.append([
        pair,
        len(common),
        pair_rmse,
        pair_mean,
        pair_max
    ])

    print(
        f"{pair} | Corners: {len(common)} | "
        f"RMSE: {pair_rmse:.3f} px | "
        f"Mean: {pair_mean:.3f} px | "
        f"Maximum: {pair_max:.3f} px"
    )

print("\n========== VALIDATION SUMMARY ==========")

if all_errors:

    all_errors = np.array(all_errors)

    overall_rmse = np.sqrt(
        np.mean(all_errors ** 2)
    )

    print("Validated pairs:", len(results))
    print("Total common corners:", len(all_errors))
    print(f"Overall RMSE: {overall_rmse:.3f} px")
    print(f"Overall mean error: {np.mean(all_errors):.3f} px")
    print(f"Overall median error: {np.median(all_errors):.3f} px")
    print(f"Overall maximum error: {np.max(all_errors):.3f} px")

    with open(
        "calibration/cross_camera_validation.csv",
        "w"
    ) as file:

        file.write(
            "Pair,Common Corners,RMSE px,Mean Error px,Maximum Error px\n"
        )

        for row in results:
            file.write(
                f"{row[0]},{row[1]},{row[2]:.4f},"
                f"{row[3]:.4f},{row[4]:.4f}\n"
            )

    print("\nReport saved:")
    print("calibration/cross_camera_validation.csv")

else:
    print("No valid image pairs.")

print("\nValidation finished.")
