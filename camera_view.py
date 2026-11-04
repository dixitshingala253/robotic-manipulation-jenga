import pyzed.sl as sl
import cv2

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

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        cv2.imshow("ZED Camera - Left Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
cv2.destroyAllWindows()
