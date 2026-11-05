
import cv2
import os

# Folder containing the four-camera calibration images
IMAGE_FOLDER = "calibration/four_camera_pairs"

# Inspect pair 020
PAIR_NUMBER = 20

CAMERAS = [
    ("cam1_left", "Camera 1 - LEFT"),
    ("cam1_right", "Camera 1 - RIGHT"),
    ("cam2_left", "Camera 2 - LEFT"),
    ("cam2_right", "Camera 2 - RIGHT"),
]

# ChArUco board settings
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

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

detector_params = cv2.aruco.DetectorParameters_create()

images = []

for camera_name, display_name in CAMERAS:

    filename = f"pair_{PAIR_NUMBER:03d}_{camera_name}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)

    image = cv2.imread(filepath)

    if image is None:
        print(f"Could not load: {filepath}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=detector_params
    )

    # Draw detected markers
    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            image,
            marker_corners,
            marker_ids
        )

        # Interpolate ChArUco corners
        _, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board
            )
        )

        corner_count = (
            0 if charuco_corners is None
            else len(charuco_corners)
        )

        if charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                image,
                charuco_corners,
                charuco_ids
            )

        marker_count = len(marker_ids)

    else:
        marker_count = 0
        corner_count = 0

    print(
        f"{display_name}: "
        f"{marker_count} markers, "
        f"{corner_count} ChArUco corners"
    )

    # Resize for display
    display_image = cv2.resize(
        image,
        (960, 600)
    )

    cv2.putText(
        display_image,
        display_name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    images.append(display_image)

if len(images) == 4:

    top = cv2.hconcat([images[0], images[1]])
    bottom = cv2.hconcat([images[2], images[3]])

    combined = cv2.vconcat([top, bottom])

    cv2.imshow("Pair 020 - Four Camera Inspection", combined)

    print("\nPress Q or ESC to close the window.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

else:
    print("Error: All four images were not loaded.")
