
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

if status != sl.ERROR_CODE.SUCCESS:
    print("Camera error:", status)
    exit(1)

print("ZED camera opened successfully!")

image = sl.Mat()
runtime = sl.RuntimeParameters()

# HSV colour ranges
colors = {
    "Yellow": ([20, 80, 80], [40, 255, 255]),
    "Blue": ([90, 80, 50], [130, 255, 255]),
    "Green": ([40, 70, 50], [85, 255, 255])
}

# Kernel for noise removal
kernel = np.ones((5, 5), np.uint8)

try:
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image, sl.VIEW.LEFT)

            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for name, (lower, upper) in colors.items():

                # Create colour mask
                mask = cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8)
                )

                # Remove noise and fill small gaps
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, kernel
                )
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, kernel
                )

                # Find object contours
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:

                    area = cv2.contourArea(contour)

                    # Ignore small regions
                    if area > 500:

                        x, y, w, h = cv2.boundingRect(contour)

                        # Draw bounding box
                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            3
                        )

                        # Display colour name
                        cv2.putText(
                            frame,
                            name,
                            (x, max(y - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

            cv2.imshow("Multiple Colour Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    zed.close()
    cv2.destroyAllWindows()

