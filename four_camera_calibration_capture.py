import pyzed.sl as sl
import cv2
import numpy as np
import os

# Camera serial numbers
serial_numbers = [44264443, 45762804]

# Folder for captured calibration images
save_folder = "calibration/four_camera_pairs"
os.makedirs(save_folder, exist_ok=True)

cameras = []
capture_count = 0

try:
    # Open both stereo cameras
    for serial in serial_numbers:
        cam = sl.Camera()

        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = sl.RESOLUTION.AUTO
        init.camera_fps = 15
        init.depth_mode = sl.DEPTH_MODE.NONE

        status = cam.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Could not open camera {serial}: {status}"
            )

        print("Camera opened:", serial)
        cameras.append(cam)

    runtime = sl.RuntimeParameters()

    # Separate image containers
    cam1_left = sl.Mat()
    cam1_right = sl.Mat()
    cam2_left = sl.Mat()
    cam2_right = sl.Mat()

    print("\nFour-camera calibration capture ready.")
    print("S = Save one image set")
    print("Q = Quit")

    while True:

        # Grab Camera 1
        status1 = cameras[0].grab(runtime)

        # Grab Camera 2
        status2 = cameras[1].grab(runtime)

        if (status1 == sl.ERROR_CODE.SUCCESS and
                status2 == sl.ERROR_CODE.SUCCESS):

            # Camera 1 images
            cameras[0].retrieve_image(cam1_left, sl.VIEW.LEFT)
            cameras[0].retrieve_image(cam1_right, sl.VIEW.RIGHT)

            # Camera 2 images
            cameras[1].retrieve_image(cam2_left, sl.VIEW.LEFT)
            cameras[1].retrieve_image(cam2_right, sl.VIEW.RIGHT)

            # Convert to OpenCV BGR images
            frames = [
                cv2.cvtColor(cam1_left.get_data(),
                             cv2.COLOR_BGRA2BGR).copy(),

                cv2.cvtColor(cam1_right.get_data(),
                             cv2.COLOR_BGRA2BGR).copy(),

                cv2.cvtColor(cam2_left.get_data(),
                             cv2.COLOR_BGRA2BGR).copy(),

                cv2.cvtColor(cam2_right.get_data(),
                             cv2.COLOR_BGRA2BGR).copy()
            ]

            # Resize for display
            display_frames = [
                cv2.resize(frame, (640, 360))
                for frame in frames
            ]

            top = np.hstack(display_frames[0:2])
            bottom = np.hstack(display_frames[2:4])
            combined = np.vstack((top, bottom))

            labels = [
                "CAM1 LEFT",
                "CAM1 RIGHT",
                "CAM2 LEFT",
                "CAM2 RIGHT"
            ]

            positions = [
                (15, 30), (655, 30),
                (15, 390), (655, 390)
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

            cv2.putText(
                combined,
                f"Saved sets: {capture_count} | Press S to save",
                (15, 755),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Four Camera Calibration Capture", combined)

            key = cv2.waitKey(1) & 0xFF

            # Save all four images together
            if key == ord('s') or key == ord('S'):

                capture_count += 1
                filename = f"pair_{capture_count:03d}"

                names = [
                    f"{filename}_cam1_left.png",
                    f"{filename}_cam1_right.png",
                    f"{filename}_cam2_left.png",
                    f"{filename}_cam2_right.png"
                ]

                for frame, name in zip(frames, names):
                    path = os.path.join(save_folder, name)
                    cv2.imwrite(path, frame)

                print(f"\nSaved image set {capture_count}:")
                for name in names:
                    print(" ", name)

            elif key == ord('q') or key == ord('Q'):
                break

finally:
    for cam in cameras:
        cam.close()

    cv2.destroyAllWindows()
    print("\nAll cameras closed.")
    print("Total saved image sets:", capture_count)
