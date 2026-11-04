import cv2
import pyzed.sl as sl
import numpy as np
import os
import time

# Camera 1 serial number
CAMERA_SERIAL = 45762804

# ChArUco board configuration
ROWS = 11
COLS = 18
SQUARE_SIZE = 0.021
MARKER_SIZE = 0.015

SAVE_DIR = "calibration/camera2/flat_new"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# Open ZED camera
zed = sl.Camera()

init = sl.InitParameters()
init.set_from_serial_number(CAMERA_SERIAL)
init.camera_resolution = sl.RESOLUTION.AUTO
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.NONE

status = zed.open(init)

if status != sl.ERROR_CODE.SUCCESS:
    print("Camera could not be opened:", status)
    exit()

runtime = sl.RuntimeParameters()
image = sl.Mat()

count = len([
    f for f in os.listdir(SAVE_DIR)
    if f.endswith(".jpg")
])

print("Camera 1 calibration collection started")
print("Move the board to different positions and angles.")
print("Press S to save an image, Q to quit.")

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

        display = frame.copy()
        charuco_count = 0

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            _, charuco_corners, charuco_ids = (
                cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
            )

            if charuco_corners is not None:
                charuco_count = len(charuco_corners)
                cv2.aruco.drawDetectedCornersCharuco(
                    display, charuco_corners, charuco_ids
                )

        cv2.putText(
            display,
            f"Detected ChArUco corners: {charuco_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Saved images: {count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera 1 Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if charuco_count >= 8:
                filename = os.path.join(
                    SAVE_DIR, f"image_{count:03d}.jpg"
                )
                cv2.imwrite(filename, frame)
                count += 1
                print("Saved:", filename)
            else:
                print("Not enough ChArUco corners. Adjust board position.")

        elif key == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()

print("Calibration image collection finished.")
print("Total saved images:", count)
