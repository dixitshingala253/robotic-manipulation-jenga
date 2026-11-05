import cv2
import numpy as np
import os
import glob
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ==================================================
# 1. SETTINGS
# ==================================================

DATA_DIR = "calibration/four_camera_pairs"
OUTPUT_DIR = "calibration"

# Five pairs reserved for final testing
TEST_PAIRS = [
    "pair_003",
    "pair_008",
    "pair_013",
    "pair_018",
    "pair_022"
]

MIN_COMMON_CORNERS = 8

K1 = np.array([
    [723.4566650390625, 0, 972.6150512695312],
    [0, 723.4566650390625, 596.51806640625],
    [0, 0, 1]
], dtype=np.float64)

K2 = np.array([
    [723.7556762695312, 0, 945.4025268554688],
    [0, 723.7556762695312, 608.3648071289062],
    [0, 0, 1]
], dtype=np.float64)

DIST = np.zeros((5, 1), dtype=np.float64)

# Previous transformation used only as an initial estimate
T_INITIAL = np.array([
    [-0.20792769,  0.92219979, -0.32605771,  0.31309176],
    [-0.79109617,  0.03749828,  0.61054135, -0.19287956],
    [ 0.57526770,  0.38489145,  0.72175178, -0.18246710],
    [0, 0, 0, 1]
], dtype=np.float64)

SQUARES_X = 18
SQUARES_Y = 11
SQUARE_LENGTH = 0.021
MARKER_LENGTH = 0.015

# ==================================================
# 2. CREATE CHARUCO BOARD
# ==================================================

aruco_dict = cv2.aruco.Dictionary_get(
    cv2.aruco.DICT_4X4_100
)

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

# ==================================================
# 3. DETECT CHARUCO CORNERS
# ==================================================

def detect_charuco(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            "Cannot read image: " + image_path
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


# ==================================================
# 4. LOAD ALL VALID IMAGE PAIRS
# ==================================================

print("\nLoading calibration image pairs...")

pair_files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "pair_*_cam1_left.png"
        )
    )
)

all_datasets = []

for cam1_path in pair_files:

    pair_name = os.path.basename(
        cam1_path
    ).replace(
        "_cam1_left.png", ""
    )

    if pair_name == "pair_020":
        print(pair_name, "SKIPPED")
        continue

    cam2_path = os.path.join(
        DATA_DIR,
        pair_name + "_cam2_left.png"
    )

    if not os.path.exists(cam2_path):
        continue

    obs1 = detect_charuco(cam1_path)
    obs2 = detect_charuco(cam2_path)

    common_ids = sorted(
        set(obs1.keys()) & set(obs2.keys())
    )

    if len(common_ids) < MIN_COMMON_CORNERS:

        print(
            pair_name,
            "SKIPPED - too few common corners"
        )

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

    success, rvec1, tvec1 = cv2.solvePnP(
        object_points,
        image_points1,
        K1,
        DIST,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        continue

    all_datasets.append({
        "name": pair_name,
        "object_points": object_points,
        "image_points1": image_points1,
        "image_points2": image_points2,
        "rvec1": rvec1.flatten(),
        "tvec1": tvec1.flatten()
    })

    print(
        pair_name,
        "Common corners:",
        len(common_ids)
    )


# ==================================================
# 5. SPLIT TRAINING AND TESTING DATA
# ==================================================

training_data = [
    d for d in all_datasets
    if d["name"] not in TEST_PAIRS
]

testing_data = [
    d for d in all_datasets
    if d["name"] in TEST_PAIRS
]

print("\n======================================")
print(" TRAINING AND TESTING SPLIT")
print("======================================")

print("\nTraining pairs:")

for d in training_data:
    print(d["name"], len(d["object_points"]))

print("\nTesting pairs:")

for d in testing_data:
    print(d["name"], len(d["object_points"]))

if len(training_data) < 10:
    raise RuntimeError("Not enough training pairs.")

if len(testing_data) != len(TEST_PAIRS):
    raise RuntimeError(
        "Not all five test pairs are available."
    )

print("\nTraining pair count:", len(training_data))
print("Testing pair count:", len(testing_data))


# ==================================================
# 6. INITIAL PARAMETERS
# ==================================================

R_initial = T_INITIAL[:3, :3]
t_initial = T_INITIAL[:3, 3]

rvec_initial = Rotation.from_matrix(
    R_initial
).as_rotvec()

initial_parameters = np.concatenate([
    rvec_initial,
    t_initial,
    *[
        np.concatenate([
            d["rvec1"],
            d["tvec1"]
        ])
        for d in training_data
    ]
])


# ==================================================
# 7. TRAINING RESIDUAL FUNCTION
# ==================================================

def calculate_training_residuals(parameters):

    rvec21 = parameters[:3]
    tvec21 = parameters[3:6]

    R21 = Rotation.from_rotvec(
        rvec21
    ).as_matrix()

    residuals = []

    for i, data in enumerate(training_data):

        start = 6 + i * 6

        rvec1 = parameters[start:start + 3]
        tvec1 = parameters[start + 3:start + 6]

        # Camera 1 projection
        projected1, _ = cv2.projectPoints(
            data["object_points"],
            rvec1,
            tvec1,
            K1,
            DIST
        )

        projected1 = projected1.reshape(-1, 2)

        # Transform board pose into Camera 2
        R1B = Rotation.from_rotvec(
            rvec1
        ).as_matrix()

        R2B = R21 @ R1B
        t2B = R21 @ tvec1 + tvec21

        rvec2 = Rotation.from_matrix(
            R2B
        ).as_rotvec()

        # Camera 2 projection
        projected2, _ = cv2.projectPoints(
            data["object_points"],
            rvec2,
            t2B,
            K2,
            DIST
        )

        projected2 = projected2.reshape(-1, 2)

        error1 = (
            projected1 - data["image_points1"]
        )

        error2 = (
            projected2 - data["image_points2"]
        )

        residuals.extend(error1.flatten())
        residuals.extend(error2.flatten())

    return np.asarray(residuals)


# ==================================================
# 8. RUN OPTIMIZATION ON TRAINING DATA ONLY
# ==================================================

print("\n======================================")
print(" STARTING TRAINING OPTIMIZATION")
print("======================================")

result = least_squares(
    calculate_training_residuals,
    initial_parameters,
    method="trf",
    loss="soft_l1",
    f_scale=2.0,
    max_nfev=300,
    verbose=1
)

print("\nTraining optimization finished.")
print("Success:", result.success)
print("Message:", result.message)


# ==================================================
# 9. EXTRACT NEW CAMERA TRANSFORMATION
# ==================================================

parameters = result.x

R21 = Rotation.from_rotvec(
    parameters[:3]
).as_matrix()

t21 = parameters[3:6]

T21 = np.eye(4)

T21[:3, :3] = R21
T21[:3, 3] = t21

print("\nNew trained transformation:")
print(T21)

print("\nTranslation in millimetres:")
print(t21 * 1000.0)


# ==================================================
# 10. VALIDATION FUNCTION
# ==================================================

def validate_dataset(data, T):

    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)

    object_points = data["object_points"]

    image_points1 = data["image_points1"]
    image_points2 = data["image_points2"]

    # Estimate board pose using Camera 1 only
    success, rvec1, tvec1 = cv2.solvePnP(
        object_points,
        image_points1,
        K1,
        DIST,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Predict Camera 2 board pose
    R1B, _ = cv2.Rodrigues(rvec1)

    R2B = R @ R1B
    t2B = R @ tvec1 + t

    rvec2, _ = cv2.Rodrigues(R2B)

    projected2, _ = cv2.projectPoints(
        object_points,
        rvec2,
        t2B,
        K2,
        DIST
    )

    projected2 = projected2.reshape(-1, 2)

    errors = projected2 - image_points2

    corner_errors = np.linalg.norm(
        errors,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    return {
        "name": data["name"],
        "corners": len(object_points),
        "rmse": rmse,
        "mean": np.mean(corner_errors),
        "median": np.median(corner_errors),
        "maximum": np.max(corner_errors)
    }


# ==================================================
# 11. TRAINING ERROR
# ==================================================

train_residuals = calculate_training_residuals(
    parameters
)

training_rmse = np.sqrt(
    np.mean(train_residuals ** 2)
)

print("\nTraining RMSE:", training_rmse, "pixels")


# ==================================================
# 12. TEST ON UNSEEN PAIRS
# ==================================================

print("\n======================================")
print(" FINAL HELD-OUT TEST")
print("======================================")

test_results = []

for data in testing_data:

    result_test = validate_dataset(
        data,
        T21
    )

    if result_test is None:
        print(data["name"], "FAILED")
        continue

    test_results.append(result_test)

    print("\n", result_test["name"])
    print("Common corners:", result_test["corners"])
    print("RMSE:", result_test["rmse"])
    print("Mean error:", result_test["mean"])
    print("Median error:", result_test["median"])
    print("Maximum error:", result_test["maximum"])


# ==================================================
# 13. FINAL SUMMARY
# ==================================================

if len(test_results) != len(TEST_PAIRS):
    raise RuntimeError(
        "Some held-out test pairs could not be validated."
    )

test_rmse = np.sqrt(
    np.mean([
        r["rmse"] ** 2
        for r in test_results
    ])
)

average_test_rmse = np.mean([
    r["rmse"] for r in test_results
])

total_test_corners = sum(
    r["corners"] for r in test_results
)

print("\n======================================")
print(" FINAL TRAINING / TEST RESULTS")
print("======================================")

print("Training pairs:", len(training_data))
print("Testing pairs:", len(test_results))
print("Training RMSE:", training_rmse)
print("Held-out test RMSE:", test_rmse)
print("Average test pair RMSE:", average_test_rmse)
print("Total test corners:", total_test_corners)


# ==================================================
# 14. SAVE NEW RESULTS
# ==================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "relative_camera_transform_train_test.npz"
)

np.savez(
    output_file,
    T_C2_from_C1=T21,
    K1=K1,
    K2=K2,
    training_pairs=np.array(
        [d["name"] for d in training_data]
    ),
    testing_pairs=np.array(
        [d["name"] for d in testing_data]
    ),
    training_rmse=training_rmse,
    test_rmse=test_rmse,
    test_average_rmse=average_test_rmse,
    test_corners=total_test_corners,
    optimizer_success=result.success
)

# Save test report
csv_file = os.path.join(
    OUTPUT_DIR,
    "train_test_validation_results.csv"
)

with open(csv_file, "w") as file:

    file.write(
        "Pair,Corners,RMSE,Mean,Median,Maximum\n"
    )

    for r in test_results:

        file.write(
            f'{r["name"]},'
            f'{r["corners"]},'
            f'{r["rmse"]:.6f},'
            f'{r["mean"]:.6f},'
            f'{r["median"]:.6f},'
            f'{r["maximum"]:.6f}\n'
        )

print("\nNew calibration saved:")
print(output_file)

print("\nTest report saved:")
print(csv_file)

print("\nTraining and testing experiment completed.")

