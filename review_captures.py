import cv2
import os
import glob
import math
import numpy as np

folder = "calibration/four_camera_pairs"
files = sorted(glob.glob(folder + "/pair_*_cam1_left.png"))

thumb_w = 240
thumb_h = 150
label_h = 30
columns = 4
rows = math.ceil(len(files) / columns)

sheet = np.ones(
    (rows * (thumb_h + label_h),
     columns * thumb_w, 3),
    dtype=np.uint8
) * 255

for i, path in enumerate(files):
    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (thumb_w, thumb_h))

    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)

    sheet[y:y+thumb_h, x:x+thumb_w] = image

    number = os.path.basename(path).split("_")[1]

    cv2.putText(
        sheet,
        "PAIR " + number,
        (x + 10, y + thumb_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

cv2.imwrite(folder + "/contact_sheet.png", sheet)

print("Contact sheet saved:", folder + "/contact_sheet.png")
print("Total captures:", len(files))
