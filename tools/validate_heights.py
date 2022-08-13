from scipy.ndimage import maximum_filter
from init import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import json
import cv2

MAX_FILTER_SIZE = 10
FONT_SCALE = .7
ROUND = 4

sns.set_theme()

images = [cv2.imread(file, cv2.CV_16U).astype(int) for file in sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_CROPPED}{EXTENSION}"))]
uncropped = cv2.imread(sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_MEDIAN}{EXTENSION}"))[-1], cv2.CV_16U)
image_first, image_last = images[1], images[-1]

image = np.clip(image_last - image_first, 0, Z_MAX)
image = maximum_filter(image, size=MAX_FILTER_SIZE)
median = np.uint16(cv2.medianBlur(np.uint8(image / 255), MAX_FILTER_SIZE * 2 + 1)) * 255
median_max = maximum_filter(median, size=MAX_FILTER_SIZE)
where = np.where(image >= np.quantile(image, QUANTILE_BLOCK_THRESH_HIGH))
image[where] = median_max[where]

with open(f"{PATH_RESULTS}/info.json") as f:
    info = json.load(f)

angle = math.radians(info["angle"])


def rotate(origin, point):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


df = pd.read_csv("calibration/210624-holice_utm_spravne.txt", delimiter="\t", header=None)

# Rename columns and delete old ones
df["height"] = df[3] / 100
df[["x", "y", "z"]] = df[1].str.split(' ', expand=True)
df = df.drop(columns=[0, 1, 2, 3])

df["x"] = (df["x"].astype(float) - info["x_min"]) / info["scale_factor"]
df["y"] = (df["y"].astype(float) - info["y_min"]) / info["scale_factor"]
df["z"] = (df["z"].astype(float) - info["z_min"]) / (info["z_max"] - info["z_min"]) * info["z_scale"]

# Calculate image rotation padding
mx, my = 0, 0
for extreme_point in [(0, 0), (uncropped.shape[1]-1, 0), (uncropped.shape[1]-1, uncropped.shape[0]-1), (0, uncropped.shape[0]-1)]:
    mnx, mny = rotate((uncropped.shape[1] / 2, uncropped.shape[0] / 2), extreme_point)

    if mnx < mx:
        mx = mnx

    if mny < my:
        my = mny


# Calculate point positions after rotation
def calculate_point(row):
    cx, cy = rotate((uncropped.shape[1] / 2, uncropped.shape[0] / 2), (row["x"], row["y"]))

    if mx < 0:
        cx = cx + abs(mx)

    if my < 0:
        cy = cy + abs(my)

    row["x"] = round(cx) - info["point_lu"][1]
    row["y"] = round(cy) - info["point_lu"][0]
    return row


df = df.apply(calculate_point, axis=1)
df = df[(df["x"].astype(int).isin(range(image.shape[1]))) & (df["y"].astype(int).isin(range(image.shape[0])))]

df["height_lidar"] = image[df["y"].astype(int), df["x"].astype(int)] / Z_MAX * (info["z_max"] - info["z_min"])
df["height_diff"] = df["height"] - df["height_lidar"]

image_last = cv2.cvtColor(np.uint8(image / 255), cv2.COLOR_GRAY2RGB)

for _, row in df.iterrows():
    px = round(row["x"])
    py = round(row["y"])

    if px in range(image.shape[1]) and py in range(image.shape[0]):
        c = image[py, px] / Z_MAX * (info["z_max"] - info["z_min"])

        cv2.circle(image_last, (px, py), 5, (0, 0, 255), 1)
        cv2.putText(image_last, f"{row['height']:.2f}", (px + 8, py - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, Z_MAX // 2, Z_MAX), 2)
        cv2.putText(image_last, f"{c:.2f}", (px + 8, py + 20), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, Z_MAX // 2, Z_MAX), 2)

for df_what, df_name in [(df["height_diff"], ""), (np.abs(df["height_diff"]), " of abs")]:
    print("Mean" + df_name, round(df_what.mean(), ROUND))
    print(" - STD", round(df_what.std(), ROUND))
    print("Median" + df_name, round(np.median(df_what), ROUND))
    print(" - Q01", round(np.quantile(df_what, 0.01), ROUND))
    print(" - Q05", round(np.quantile(df_what, 0.05), ROUND))
    print(" - Q25", round(np.quantile(df_what, 0.25), ROUND))
    print(" - Q75", round(np.quantile(df_what, 0.75), ROUND))
    print(" - Q95", round(np.quantile(df_what, 0.95), ROUND))
    print(" - Q99", round(np.quantile(df_what, 0.99), ROUND))
    print("Min" + df_name, round(df_what.min(), ROUND))
    print("Max" + df_name, round(df_what.max(), ROUND))
    print("RMSD" + df_name, round(np.sqrt(np.mean(df_what ** 2)), ROUND))
    print()

# Draw plot
plt.figure(figsize=(11, 6))
df["height_diff"].hist(bins=30)
plt.axvline(df["height_diff"].mean(), color="k")
plt.title("Histogram of difference between LiDAR and reference height")
plt.xlabel("Difference [m]")
plt.ylabel("Count")
plt.savefig(f"{PATH_RESULTS}/height_diff_histogram.png")
plt.show()
cv2.imwrite(f"{PATH_RESULTS}/height_diff_image.png", image_last)
