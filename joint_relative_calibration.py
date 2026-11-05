import cv2
import numpy as np
import glob
import os
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_DIR = "calibration/four_camera_pairs"
OUTPUT_DIR = "calibration"

CAM1_SERIAL = 44264443
CAM2_SERIAL = 45762804

# ZED SDK rectified LEFT camera parameters
K1 = np.array([
    [723.4566650390625, 0.0, 972.6150512695312],
    [0.0, 723.4566650390625, 596.51806640625],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

K2 = np.array([
    [723.7556762695312, 0.0, 945.4025268554688],
    [0.0, 723.7556762695312, 608.3648071289062],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST = np.zeros((5, 1), dtype=np.float64)

# Previous provisional Camera 1 -> Camera 2 transformation
T_INITIAL = np.array([
    [-0.20792769,  0.92219979, -0.32605771,  0.31309176],
    [-0.79109617,  0.03749828,  0.61054135, -0.19287956],
    [ 0.57526770,  0.38489145,  0.72175178, -0.18246710],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float64)

# ChArUco board
SQUARES_X = 18
SQUARES_Y = 11
SQUARE_LENGTH = 0.021
MARKER_LENGTH = 0.015
DICTIONARY = cv2.aruco.DICT_4X4_100

MIN_COMMON_CORNERS = 8

# --------------------------------------------------
# 2. CREATE CHARUCO BOARD
# --------------------------------------------------

aruco_dict = cv2.aruco.Dictionary_get(DICTIONARY)

board = cv2.aruco.CharucoBoard_create(
    SQUARES_X,
    SQUARES_Y,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

board_corners = np.asarray(
    board.chessboardCorners,
    dtype=np.float64
)

# --------------------------------------------------
# 3. DETECT CHARUCO CORNERS
# --------------------------------------------------

def detect_charuco(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError("Could not read image: " + image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        gray,
        aruco_dict
    )

    if marker_ids is None:
        return {}, 0

    count, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )
    )

    if charuco_ids is None or charuco_corners is None:
        return {}, 0

    observations = {}

    for i, corner_id in enumerate(charuco_ids.flatten()):

        observations[int(corner_id)] = (
            charuco_corners[i, 0].astype(np.float64)
        )

    return observations, len(observations)


# --------------------------------------------------
# 4. LOAD IMAGE PAIRS
# --------------------------------------------------

print("\nLoading calibration image pairs...")

pair_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "pair_*_cam1_left.png"))
)

datasets = []

for cam1_path in pair_files:

    pair_name = os.path.basename(cam1_path).replace(
        "_cam1_left.png", ""
    )

    # Pair 020 has only seven common corners
    if pair_name == "pair_020":
        print(pair_name, "SKIPPED")
        continue

    cam2_path = os.path.join(
        DATA_DIR,
        pair_name + "_cam2_left.png"
    )

    if not os.path.exists(cam2_path):
        print(pair_name, "MISSING CAMERA 2 IMAGE")
        continue

    obs1, count1 = detect_charuco(cam1_path)
    obs2, count2 = detect_charuco(cam2_path)

    common_ids = sorted(set(obs1.keys()) & set(obs2.keys()))

    if len(common_ids) < MIN_COMMON_CORNERS:
        print(pair_name, "SKIPPED - too few common corners")
        continue

    object_points = np.array(
        [board_corners[i] for i in common_ids],
        dtype=np.float64
    )

    image_points1 = np.array(
        [obs1[i] for i in common_ids],
        dtype=np.float64
    )

    image_points2 = np.array(
        [obs2[i] for i in common_ids],
        dtype=np.float64
    )

    # Initial board pose estimated from Camera 1
    success, rvec1, tvec1 = cv2.solvePnP(
        object_points,
        image_points1,
        K1,
        DIST,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print(pair_name, "SKIPPED - solvePnP failed")
        continue

    datasets.append({
        "name": pair_name,
        "object_points": object_points,
        "image_points1": image_points1,
        "image_points2": image_points2,
        "rvec1": rvec1.flatten(),
        "tvec1": tvec1.flatten()
    })

    print(
        pair_name,
        "Camera 1:", count1,
        "Camera 2:", count2,
        "Common:", len(common_ids)
    )

if len(datasets) < 5:
    raise RuntimeError("Not enough valid image pairs for optimization.")

print("\nValid image pairs:", len(datasets))


# --------------------------------------------------
# 5. PREPARE INITIAL PARAMETERS
# --------------------------------------------------

R_initial = T_INITIAL[:3, :3]
t_initial = T_INITIAL[:3, 3]

rvec_initial = Rotation.from_matrix(
    R_initial
).as_rotvec()

# Parameter vector:
# First 6 values = Camera 1 to Camera 2 transformation
# Remaining 6 values per pair = board pose in Camera 1

initial_parameters = np.concatenate([
    rvec_initial,
    t_initial,
    *[
        np.concatenate([d["rvec1"], d["tvec1"]])
        for d in datasets
    ]
])


# --------------------------------------------------
# 6. CALCULATE REPROJECTION ERRORS
# --------------------------------------------------

def calculate_residuals(parameters):

    rvec21 = parameters[0:3]
    tvec21 = parameters[3:6]

    R21 = Rotation.from_rotvec(rvec21).as_matrix()

    residuals = []

    for i, data in enumerate(datasets):

        start = 6 + i * 6

        rvec1 = parameters[start:start + 3]
        tvec1 = parameters[start + 3:start + 6]

        # Project board into Camera 1
        projected1, _ = cv2.projectPoints(
            data["object_points"],
            rvec1,
            tvec1,
            K1,
            DIST
        )

        projected1 = projected1.reshape(-1, 2)

        # Transform board pose from Camera 1 to Camera 2
        R_board1 = Rotation.from_rotvec(rvec1).as_matrix()

        R_board2 = R21 @ R_board1
        t_board2 = R21 @ tvec1 + tvec21

        rvec2 = Rotation.from_matrix(R_board2).as_rotvec()

        # Project board into Camera 2
        projected2, _ = cv2.projectPoints(
            data["object_points"],
            rvec2,
            t_board2,
            K2,
            DIST
        )

        projected2 = projected2.reshape(-1, 2)

        error1 = projected1 - data["image_points1"]
        error2 = projected2 - data["image_points2"]

        residuals.extend(error1.flatten())
        residuals.extend(error2.flatten())

    return np.asarray(residuals)


# --------------------------------------------------
# 7. RUN JOINT OPTIMIZATION
# --------------------------------------------------

print("\nStarting joint optimization...")
print("Please wait. This may take some time.")

result = least_squares(
    calculate_residuals,
    initial_parameters,
    method="trf",
    loss="soft_l1",
    f_scale=2.0,
    max_nfev=300,
    verbose=1
)

if not result.success:
    print("\nWARNING: Optimizer did not report successful convergence.")
    print(result.message)


# --------------------------------------------------
# 8. EXTRACT OPTIMIZED TRANSFORMATION
# --------------------------------------------------

optimized_parameters = result.x

R_optimized = Rotation.from_rotvec(
    optimized_parameters[0:3]
).as_matrix()

t_optimized = optimized_parameters[3:6]

T_optimized = np.eye(4)
T_optimized[:3, :3] = R_optimized
T_optimized[:3, 3] = t_optimized

# Calculate final combined reprojection error
final_residuals = calculate_residuals(optimized_parameters)

overall_rmse = np.sqrt(
    np.mean(final_residuals ** 2)
)

overall_mean = np.mean(np.abs(final_residuals))


# --------------------------------------------------
# 9. PRINT RESULTS
# --------------------------------------------------

print("\n======================================")
print(" JOINT CALIBRATION RESULTS")
print("======================================")

print("\nTransformation: Camera 1 LEFT -> Camera 2 LEFT")

print("\nOptimized transformation matrix:")
print(T_optimized)

print("\nTranslation in metres:")
print(t_optimized)

print("\nTranslation in millimetres:")
print(t_optimized * 1000.0)

print("\nCombined reprojection RMSE:")
print(overall_rmse, "pixels")

print("\nMean absolute pixel error:")
print(overall_mean, "pixels")

print("\nOptimizer success:", result.success)
print("Optimizer message:", result.message)

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "joint_relative_calibration.npz"
)

np.savez(
    output_path,
    T_C2_from_C1=T_optimized,
    K1=K1,
    K2=K2,
    distortion1=DIST,
    distortion2=DIST,
    rmse=overall_rmse,
    mean_absolute_error=overall_mean,
    successful_pairs=len(datasets),
    optimizer_success=result.success
)

print("\nSaved optimized calibration to:")
print(output_path)

print("\nJoint calibration finished.")
