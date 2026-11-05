import cv2
import numpy as np
import os

ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/four_camera_pairs"

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
        [points[i] for i in object_points],
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

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T


def rotation_difference(R1, R2):
    relative = R1 @ R2.T
    value = (np.trace(relative) - 1) / 2
    value = np.clip(value, -1.0, 1.0)
    return np.degrees(np.arccos(value))


transforms = []
pair_names = []

print("\nCalculating transformation consistency...\n")

for number in range(1, 23):

    if number == 5:
        continue

    pair = f"pair_{number:03d}"

    image1 = cv2.imread(
        os.path.join(IMAGE_DIR, f"{pair}_cam1_left.png")
    )
    image2 = cv2.imread(
        os.path.join(IMAGE_DIR, f"{pair}_cam2_left.png")
    )

    if image1 is None or image2 is None:
        continue

    corners1 = detect_charuco(image1)
    corners2 = detect_charuco(image2)

    common = sorted(set(corners1) & set(corners2))

    if len(common) < 8:
        print(pair, "SKIPPED:", len(common), "common corners")
        continue

    object_points = {
        i: board.chessboardCorners[i] for i in common
    }

    T1 = estimate_pose(corners1, K1, object_points)
    T2 = estimate_pose(corners2, K2, object_points)

    if T1 is None or T2 is None:
        print(pair, "PnP FAILED")
        continue

    T21 = T2 @ np.linalg.inv(T1)

    transforms.append(T21)
    pair_names.append(pair)

print("\nSuccessful pairs:", len(transforms))

# Robust translation centre
translations = np.array([T[:3, 3] for T in transforms])
translation_centre = np.median(translations, axis=0)

# Rotation centre using quaternion averaging
quaternions = []

for T in transforms:
    q = cv2.RQDecomp3x3(T[:3, :3])
    R = T[:3, :3]

    from scipy.spatial.transform import Rotation
    quaternions.append(Rotation.from_matrix(R).as_quat())

quaternions = np.array(quaternions)

# Align quaternion signs
reference = quaternions[0]

for i in range(len(quaternions)):
    if np.dot(quaternions[i], reference) < 0:
        quaternions[i] *= -1

average_q = np.mean(quaternions, axis=0)
average_q /= np.linalg.norm(average_q)

from scipy.spatial.transform import Rotation

average_R = Rotation.from_quat(average_q).as_matrix()

# Build representative transformation
T_average = np.eye(4)
T_average[:3, :3] = average_R
T_average[:3, 3] = translation_centre

print("\n========== PAIR DEVIATIONS ==========")

rotation_errors = []
translation_errors = []

for name, T in zip(pair_names, transforms):

    r_error = rotation_difference(
        T[:3, :3], average_R
    )

    t_error = np.linalg.norm(
        T[:3, 3] - translation_centre
    )

    rotation_errors.append(r_error)
    translation_errors.append(t_error)

    print(
        f"{name}: rotation difference = {r_error:.4f} deg, "
        f"translation difference = {t_error * 1000:.2f} mm"
    )

print("\n========== REPRESENTATIVE TRANSFORM ==========")
print(T_average)

print("\nTranslation centre (mm):")
print(translation_centre * 1000)

print("\nMean rotation deviation (degrees):",
      np.mean(rotation_errors))

print("Maximum rotation deviation (degrees):",
      np.max(rotation_errors))

print("Mean translation deviation (mm):",
      np.mean(translation_errors) * 1000)

print("Maximum translation deviation (mm):",
      np.max(translation_errors) * 1000)

print("\nConsistency analysis finished.")
