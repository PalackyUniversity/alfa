import numpy as np
import glob
import cv2
import os

SCALE = 1/6  # Fraction

for i in glob.glob("../results/210624_065427 - Scanner 1 - 210624_073321_Scanner_1 - originalpoints/*.png"):
    if "compressed" in i:
        continue

    image_16bit = cv2.imread(i, cv2.CV_16U)

    if "count" in i:
        image_16bit *= 2560
        image_8bit = np.clip(image_16bit / 2 ** 8, 0, 255)
    else:
        image_8bit = np.uint8(image_16bit / 2**8)

    image_gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY) if len(image_8bit.shape) == 3 else image_8bit
    image_resized = cv2.resize(image_gray, None, fx=SCALE, fy=SCALE)

    cv2.imwrite(os.path.splitext(i)[0] + "_compressed.png", image_resized)
