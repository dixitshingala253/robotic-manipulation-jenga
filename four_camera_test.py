import pyzed.sl as sl
import cv2
import numpy as np

serial_numbers = [44264443, 45762804]
cameras = []

try:
    # Open both ZED X stereo cameras
    for serial in serial_numbers:
        cam = sl.Camera()

        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.AUTO
        init.camera_fps = 15
        init.depth_mode = sl.DEPTH_MODE.NONE

        status = cam.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Could not open camera {serial}: {status}")

        print(f"Camera opened: {serial}")
        cameras.append(cam)

    runtime = sl.RuntimeParameters()

    # Four separate image containers
    images = [sl.Mat() for _ in range(4)]

    print("\nAll cameras opened.")
    print("Press Q in the image window to exit.")

    while True:
        frames = []
        all_grabbed = True

        for cam in cameras:
            status = cam.grab(runtime)

            if status != sl.ERROR_CODE.SUCCESS:
                all_grabbed = False
                break

            # Retrieve left image
            cam.retrieve_image(images[0], sl.VIEW.LEFT)
            left = images[0].get_data()
            left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)

            # Retrieve right image
            cam.retrieve_image(images[1], sl.VIEW.RIGHT)
            right = images[1].get_data()
            right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

            frames.append(left)
            frames.append(right)

        if all_grabbed and len(frames) == 4:

            display_frames = []

            for frame in frames:
                frame = cv2.resize(frame, (640, 360))
                display_frames.append(frame)

            # Arrange four images in a 2x2 layout
            top = np.hstack(display_frames[0:2])
            bottom = np.hstack(display_frames[2:4])
            combined = np.vstack((top, bottom))

            labels = [
                "CAMERA 1 - LEFT",
                "CAMERA 1 - RIGHT",
                "CAMERA 2 - LEFT",
                "CAMERA 2 - RIGHT"
            ]

            positions = [
                (15, 30),
                (655, 30),
                (15, 390),
                (655, 390)
            ]

            for label, position in zip(labels, positions):
                cv2.putText(
                    combined,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Four ZED Camera Streams", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cam in cameras:
        cam.close()

    cv2.destroyAllWindows()
    print("All opened cameras closed.")
