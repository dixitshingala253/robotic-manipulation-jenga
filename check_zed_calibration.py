import pyzed.sl as sl

serial_numbers = [44264443, 45762804]

for serial in serial_numbers:
    zed = sl.Camera()

    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = sl.RESOLUTION.AUTO
    init.depth_mode = sl.DEPTH_MODE.NONE

    status = zed.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Could not open camera:", serial, status)
        continue

    info = zed.get_camera_information()
    calibration = info.camera_configuration.calibration_parameters

    print("\nCamera serial:", serial)
    print("Image width:", info.camera_configuration.resolution.width)
    print("Image height:", info.camera_configuration.resolution.height)

    print("LEFT camera:")
    print("fx:", calibration.left_cam.fx)
    print("fy:", calibration.left_cam.fy)
    print("cx:", calibration.left_cam.cx)
    print("cy:", calibration.left_cam.cy)

    print("RIGHT camera:")
    print("fx:", calibration.right_cam.fx)
    print("fy:", calibration.right_cam.fy)
    print("cx:", calibration.right_cam.cx)
    print("cy:", calibration.right_cam.cy)

    zed.close()
