from init import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import glob
import cv2

# Set plot theme and size
sns.set_theme()
plt.figure(figsize=(11, 6))

# Load cached information
with open(PATH_RESULTS + "/info.json") as f:
    info = json.load(f)

# Load images and crop them to fix place where is the terrain that does not grow
images = [cv2.imread(file, cv2.CV_16U)[4_200:4_300, 800:2_400] for file in sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_MEDIAN}{EXTENSION}"))]

# Calculate mean and std of cropped images
mean = np.array([np.mean(im.astype(float) / Z_MAX * (info["z_max"] - info["z_min"]) + info["z_min"]) for im in images])
std = np.array([np.std(im.astype(float) / Z_MAX * (info["z_max"] - info["z_min"]) + info["z_min"]) for im in images])

# mean -= np.mean(mean)

plt.errorbar(dates, mean, yerr=std, capsize=10)
plt.title("Average altitude of cropped fix area based on time")
plt.ylabel("Average altitude of cropped fix area [m]")
plt.xlabel("Time")
plt.savefig(f"{PATH_RESULTS}/terrain.png")
plt.show()

cv2.imwrite(f"{PATH_RESULTS}/terrain_image.png", images[-1])
