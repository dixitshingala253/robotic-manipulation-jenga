
import pyzed.sl as sl
import cv2
import numpy as np

# Open ZED camera
zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1200
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT

status = zed.open(init)
1
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera error:", status)
    exit(1)

print("ZED camera opened successfully!")

image = sl.Mat()
runtime = sl.RuntimeParameters()

# HSV range for natural wooden colour
lower_wood = np.array([5, 30, 70], dtype=np.uint8)
upper_wood = np.array([35, 255, 255], dtype=np.uint8)

kernel = np.ones((5, 5), np.uint8)

try:
    while True:

        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image, sl.VIEW.LEFT)

            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect wooden colour
            mask = cv2.inRange(
                hsv, lower_wood, upper_wood
            )

            # Remove noise
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel
            )

            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel
            )

            # Find contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            block_count = 0

            for contour in contours:

                area = cv2.contourArea(contour)

                if area > 700:

                    x, y, w, h = cv2.boundingRect(contour)

                    block_count += 1

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                    # Find centre
                    cx = x + w // 2
                    cy = y + h // 2

                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"Block {block_count}",
                        (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            cv2.putText(
                frame,
                f"Blocks detected: {block_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            cv2.imshow("Jenga Block Detection", frame)
            cv2.imshow("Wood Colour Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    zed.close()
    cv2.destroyAllWindows()
