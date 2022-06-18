import numpy as np
import glob
import cv2
import os

SCALE = 1/2  # Fraction
BIT_CONVERSION = False

for i in glob.glob("../docs/*"):
    if "compressed" in i or "blocks" in i or "fit" in i:
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

    if "draw" in i or len(image_8bit.shape) == 3:
        image_gray = image_8bit
    else:
        image_gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, None, fx=SCALE, fy=SCALE)

    cv2.imwrite(os.path.splitext(i)[0] + "_compressed.png", image_resized)
