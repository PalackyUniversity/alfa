import numpy as np
import glob
import cv2
import os

SCALE = 1/6  # Fraction
BIT_CONVERSION = True
SCALE_EXCEPT = ["compressed", "blocks", "fit", "histogram", "svg", "terrain", "sigmoid"]

for i in glob.glob("docs/*"):
    if sum([j in i for j in SCALE_EXCEPT]) != 0:
        continue

    if BIT_CONVERSION:
        image_16bit = cv2.imread(i, cv2.CV_16U)

        if "count" in i:
            image_16bit *= 2560
            image_8bit = np.clip(image_16bit / 2 ** 8, 0, 255)
        else:
            image_8bit = np.uint8(image_16bit / 2**8)
    else:
        image_8bit = cv2.imread(i)

    if "draw" in i or "count" in i:
        image_gray = image_8bit
    else:
        image_gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, None, fx=SCALE, fy=SCALE)

    cv2.imwrite(os.path.splitext(i)[0] + "_compressed.jpg", image_resized)
