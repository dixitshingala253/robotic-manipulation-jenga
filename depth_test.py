import pyzed.sl as sl
import time

zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1200
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
init.coordinate_units = sl.UNIT.METER

status = zed.open(init)

if status != sl.ERROR_CODE.SUCCESS:
    print("Camera error:", status)
    exit(1)

print("Camera opened successfully")

image = sl.Mat()
depth = sl.Mat()
runtime = sl.RuntimeParameters()

# Allow camera to stabilize
time.sleep(2)

for i in range(10):
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        x = depth.get_width() // 2
        y = depth.get_height() // 2

        err, distance = depth.get_value(x, y)

        if err == sl.ERROR_CODE.SUCCESS:
            print(f"Measurement {i+1}: {distance:.3f} m")
        else:
            print(f"Measurement {i+1}: Invalid depth")

zed.close()
