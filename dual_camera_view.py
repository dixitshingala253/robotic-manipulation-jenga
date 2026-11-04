import pyzed.sl as sl
import cv2
import numpy as np

serial_numbers = [44264443, 45762804]
cameras = []

for serial in serial_numbers:
    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = sl.RESOLUTION.AUTO
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.NONE

    status = cam.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Could not open camera:", serial, status)
        for opened in cameras:
            opened.close()
        raise SystemExit

    print("Camera opened:", serial)
    cameras.append(cam)

runtime = sl.RuntimeParameters()
image = sl.Mat()

try:
    while True:
        frames = []

        for cam in cameras:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (640, 360))
                frames.append(frame)

        if len(frames) == 2:
            combined = np.hstack(frames)

            cv2.putText(combined, "CAM0", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.putText(combined, "CAM1", (655, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.imshow("Dual ZED X Camera View", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cam in cameras:
        cam.close()

    cv2.destroyAllWindows()
    print("Both cameras closed.")
