import cv2
import csv
import os

# ChArUco board settings
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/four_camera_pairs"
REPORT = "calibration/four_camera_review.csv"

dictionary = cv2.aruco.Dictionary_get(
    cv2.aruco.DICT_4X4_100
)

board = cv2.aruco.CharucoBoard_create(
    COLS, ROWS, SQUARE_SIZE, MARKER_SIZE, dictionary
)

parameters = cv2.aruco.DetectorParameters_create()

cameras = [
    "cam1_left",
    "cam1_right",
    "cam2_left",
    "cam2_right"
]

results = []

print("\nStarting ChArUco inspection...\n")

# Check 21 sets, skipping deleted pair 005
for pair_number in range(1, 23):

    if pair_number == 5:
        continue

    pair_name = f"pair_{pair_number:03d}"

    pair_results = []

    for camera in cameras:

        filename = f"{pair_name}_{camera}.png"
        path = os.path.join(IMAGE_DIR, filename)

        image = cv2.imread(path)

        if image is None:
            print(filename, "- MISSING")
            pair_results.append((camera, 0, 0, "MISSING"))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

        marker_count = 0 if ids is None else len(ids)
        charuco_count = 0

        if marker_count > 0:

            _, charuco_corners, charuco_ids = (
                cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
            )

            if charuco_corners is not None:
                charuco_count = len(charuco_corners)

        status = "OK" if charuco_count >= 8 else "CHECK"

        print(
            f"{pair_name} | {camera} | "
            f"Markers: {marker_count} | "
            f"ChArUco corners: {charuco_count} | {status}"
        )

        pair_results.append(
            (camera, marker_count, charuco_count, status)
        )

    for camera, markers, corners, status in pair_results:
        results.append([
            pair_name, camera, markers, corners, status
        ])

    valid = all(item[3] == "OK" for item in pair_results)

    print(
        f"{pair_name} OVERALL: "
        f"{'ALL FOUR DETECTED' if valid else 'REVIEW REQUIRED'}\n"
    )

# Save inspection report
with open(REPORT, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Pair", "Camera", "Detected Markers",
        "ChArUco Corners", "Status"
    ])

    writer.writerows(results)

print("--------------------------------")
print("Inspection finished.")
print("Total image sets:", 21)
print("Total images checked:", len(results))
print("Report saved:", REPORT)
