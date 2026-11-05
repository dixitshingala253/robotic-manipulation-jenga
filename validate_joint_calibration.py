import cv2
import numpy as np
import os
import glob

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_DIR = "calibration/four_camera_pairs"

CALIBRATION_FILE = (
    "calibration/joint_relative_calibration.npz"
)

# These pairs are reserved for independent testing.
TEST_PAIRS = [
    "pair_003",
    "pair_008",
    "pair_013",
    "pair_018",
    "pair_022"
]

# Camera 1 LEFT rectified intrinsics
K1 = np.array([
    [723.4566650390625, 0.0, 972.6150512695312],
    [0.0, 723.4566650390625, 596.51806640625],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Camera 2 LEFT rectified intrinsics
K2 = np.array([
    [723.7556762695312, 0.0, 945.4025268554688],
    [0.0, 723.7556762695312, 608.3648071289062],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST = np.zeros((5, 1), dtype=np.float64)

# ChArUco board parameters
SQUARES_X = 18
SQUARES_Y = 11
SQUARE_LENGTH = 0.021
MARKER_LENGTH = 0.015

DICTIONARY = cv2.aruco.DICT_4X4_100

# --------------------------------------------------
# 2. LOAD OPTIMIZED TRANSFORMATION
# --------------------------------------------------

if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(
        "Optimized calibration file not found: "
        + CALIBRATION_FILE
    )

calibration = np.load(CALIBRATION_FILE)

T21 = calibration["T_C2_from_C1"]

R21 = T21[:3, :3]
t21 = T21[:3, 3]

print("\nLoaded optimized camera transformation:")
print(T21)

# --------------------------------------------------
# 3. CREATE CHARUCO BOARD
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
# 4. DETECT CHARUCO CORNERS
# --------------------------------------------------

def detect_charuco(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            "Could not read image: " + image_path
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    marker_corners, marker_ids, _ = (
        cv2.aruco.detectMarkers(
            gray,
            aruco_dict
        )
    )

    if marker_ids is None:
        return {}

    count, corners, ids = (
        cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )
    )

    if ids is None or corners is None:
        return {}

    observations = {}

    for i, corner_id in enumerate(ids.flatten()):

        observations[int(corner_id)] = (
            corners[i, 0].astype(np.float64)
        )

    return observations


# --------------------------------------------------
# 5. VALIDATE ONE IMAGE PAIR
# --------------------------------------------------

def validate_pair(pair_name):

    path1 = os.path.join(
        DATA_DIR,
        pair_name + "_cam1_left.png"
    )

    path2 = os.path.join(
        DATA_DIR,
        pair_name + "_cam2_left.png"
    )

    if not os.path.exists(path1):
        print(pair_name, "Camera 1 image missing")
        return None

    if not os.path.exists(path2):
        print(pair_name, "Camera 2 image missing")
        return None

    obs1 = detect_charuco(path1)
    obs2 = detect_charuco(path2)

    common_ids = sorted(
        set(obs1.keys()) & set(obs2.keys())
    )

    if len(common_ids) < 8:
        print(
            pair_name,
            "SKIPPED - too few common corners:",
            len(common_ids)
        )
        return None

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

    # Estimate board pose independently using Camera 1
    success, rvec1, tvec1 = cv2.solvePnP(
        object_points,
        image_points1,
        K1,
        DIST,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print(pair_name, "solvePnP failed")
        return None

    # Predict board position in Camera 2
    R1B, _ = cv2.Rodrigues(rvec1)

    R2B = R21 @ R1B
    t2B = R21 @ tvec1 + t21.reshape(3, 1)

    rvec2, _ = cv2.Rodrigues(R2B)

    # Project predicted board corners into Camera 2
    projected2, _ = cv2.projectPoints(
        object_points,
        rvec2,
        t2B,
        K2,
        DIST
    )

    projected2 = projected2.reshape(-1, 2)

    # Compare predictions with actual Camera 2 corners
    errors = projected2 - image_points2

    corner_errors = np.linalg.norm(
        errors,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    mean_error = np.mean(corner_errors)

    median_error = np.median(corner_errors)

    max_error = np.max(corner_errors)

    return {
        "pair": pair_name,
        "corners": len(common_ids),
        "rmse": rmse,
        "mean": mean_error,
        "median": median_error,
        "maximum": max_error
    }


# --------------------------------------------------
# 6. RUN INDEPENDENT VALIDATION
# --------------------------------------------------

print("\n======================================")
print(" INDEPENDENT CALIBRATION VALIDATION")
print("======================================")

print("\nTest pairs:", TEST_PAIRS)

results = []

for pair_name in TEST_PAIRS:

    print("\nTesting", pair_name)

    result = validate_pair(pair_name)

    if result is not None:

        results.append(result)

        print("Common corners:", result["corners"])
        print("RMSE:", round(result["rmse"], 4), "pixels")
        print("Mean error:", round(result["mean"], 4), "pixels")
        print("Median error:", round(result["median"], 4), "pixels")
        print("Maximum error:", round(result["maximum"], 4), "pixels")


# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

if len(results) == 0:
    raise RuntimeError(
        "No valid test pairs. Validation could not be completed."
    )

all_errors = []

for result in results:
    all_errors.append(result["rmse"])

overall_rmse = np.sqrt(
    np.mean([
        result["rmse"] ** 2
        for result in results
    ])
)

average_rmse = np.mean(all_errors)
median_rmse = np.median(all_errors)

total_corners = sum(
    result["corners"]
    for result in results
)

print("\n======================================")
print(" VALIDATION SUMMARY")
print("======================================")

print("Successful test pairs:", len(results))
print("Total common corners:", total_corners)

print("\nAverage pair RMSE:", average_rmse)
print("Median pair RMSE:", median_rmse)
print("Overall RMSE:", overall_rmse)

print("\nPrevious joint optimization RMSE: 0.5238 pixels")
print("Previous cross-camera RMSE: 2.904 pixels")

# --------------------------------------------------
# 8. SAVE VALIDATION RESULTS
# --------------------------------------------------

csv_path = os.path.join(
    "calibration",
    "joint_calibration_independent_validation.csv"
)

with open(csv_path, "w") as file:

    file.write(
        "Pair,Common_Corners,RMSE,Mean,Median,Maximum\n"
    )

    for result in results:

        file.write(
            f'{result["pair"]},'
            f'{result["corners"]},'
            f'{result["rmse"]:.6f},'
            f'{result["mean"]:.6f},'
            f'{result["median"]:.6f},'
            f'{result["maximum"]:.6f}\n'
        )

print("\nValidation results saved to:")
print(csv_path)

print("\nIndependent validation finished.")
