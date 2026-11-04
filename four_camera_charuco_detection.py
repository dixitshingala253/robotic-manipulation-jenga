
import cv2
import numpy as np
import os

# ChArUco board settings
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

IMAGE_DIR = "calibration/four_camera_pairs"
PAIR_NUMBER = 21

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

camera_names = [
    "cam1_left",
    "cam1_right",
    "cam2_left",
    "cam2_right"
]

images = []

for camera_name in camera_names:

    filename = os.path.join(
        IMAGE_DIR,
        f"pair_{PAIR_NUMBER:03d}_{camera_name}.png"
    )

    image = cv2.imread(filename)

    if image is None:
        print("Could not load:", filename)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=parameters
    )

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        count, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
        )

        if charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                image, charuco_corners, charuco_ids
            )

            print(
                camera_name,
                "- Markers:", len(ids),
                "- ChArUco corners:", len(charuco_corners)
            )
        else:
            print(camera_name, "- No ChArUco corners")

    else:
        print(camera_name, "- No markers detected")

    image = cv2.resize(image, (640, 400))
    cv2.putText(
        image,
        camera_name,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    images.append(image)

if len(images) == 4:
    top = np.hstack(images[:2])
    bottom = np.hstack(images[2:])
    result = np.vstack([top, bottom])

    cv2.imshow("Four Camera ChArUco Detection", result)
    print("\nPress Q to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("\nERROR: All four camera images are required.")

