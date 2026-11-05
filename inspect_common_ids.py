import cv2
import os
import numpy as np

# ChArUco board settings
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/four_camera_pairs"

dictionary = cv2.aruco.Dictionary_get(
    cv2.aruco.DICT_4X4_100
)

board = cv2.aruco.CharucoBoard_create(
    COLS, ROWS, SQUARE_SIZE, MARKER_SIZE, dictionary
)

parameters = cv2.aruco.DetectorParameters_create()

print("\nChecking common ChArUco IDs...\n")

total_common = []

for pair_number in range(1, 23):

    if pair_number == 5:
        continue

    pair_name = f"pair_{pair_number:03d}"

    file1 = os.path.join(
        IMAGE_DIR, f"{pair_name}_cam1_left.png"
    )
    file2 = os.path.join(
        IMAGE_DIR, f"{pair_name}_cam2_left.png"
    )

    image1 = cv2.imread(file1)
    image2 = cv2.imread(file2)

    if image1 is None or image2 is None:
        print(pair_name, "MISSING IMAGE")
        continue

    def detect_charuco(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

        if ids is None:
            return {}

        result, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
        )

        if result is None or charuco_ids is None:
            return {}

        return {
            int(corner_id): point[0]
            for corner_id, point in zip(
                charuco_ids.flatten(),
                charuco_corners
            )
        }

    corners1 = detect_charuco(image1)
    corners2 = detect_charuco(image2)

    common_ids = sorted(
        set(corners1.keys()) & set(corners2.keys())
    )

    total_common.append(len(common_ids))

    print(
        f"{pair_name} | "
        f"Cam1: {len(corners1)} | "
        f"Cam2: {len(corners2)} | "
        f"Common IDs: {len(common_ids)}"
    )

print("\n--------------------------------")
print("Total pairs checked:", len(total_common))

if total_common:
    print("Average common IDs:", np.mean(total_common))
    print("Minimum common IDs:", min(total_common))
    print("Maximum common IDs:", max(total_common))

print("--------------------------------")
