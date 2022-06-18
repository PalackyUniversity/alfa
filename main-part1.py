from init import *
from tqdm import tqdm
import numpy as np
import os.path
import ctypes
import laspy
import json
import glob
import math
import cv2

# Load C algo file
cdll = ctypes.CDLL("main.so")
cdll.createImage.argtypes = [
    np.ctypeslib.ndpointer(int, ndim=1, flags='aligned'),
    np.ctypeslib.ndpointer(int, ndim=1, flags='aligned'),
    np.ctypeslib.ndpointer(int, ndim=1, flags='aligned'),
    np.ctypeslib.ndpointer(int, ndim=2, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer(int, ndim=2, flags='aligned, contiguous, writeable'),
    ctypes.c_int,
    ctypes.POINTER(np.ctypeslib.c_intp)
]

cdll.medianImage.argtypes = [
    np.ctypeslib.ndpointer(int, ndim=2, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer(int, ndim=2, flags='aligned, contiguous, writeable'),
    ctypes.POINTER(np.ctypeslib.c_intp)
]

# List all LiDAR files and sort them by date
las_files = glob.glob(f"{PATH_DATA}/*.las")
las_files.sort()

assert len(las_files) != 0, "Missing las files!"

las_data = []

# Load all LiDAR files and find minimums and maximums to normalize
x_min, x_max = np.inf, 0
y_min, y_max = np.inf, 0
z_min, z_max = np.inf, 0

for n in tqdm(range(len(las_files)), desc="Loading LiDAR files..."):
    las = laspy.file.File(las_files[n])

    x, y, z = las.x, las.y, las.z

    # Get low and high quantiles
    x_low, x_high = x > np.quantile(x, QUANTILE_THRESH_LOW), x < np.quantile(x, QUANTILE_THRESH_HIGH)
    y_low, y_high = y > np.quantile(y, QUANTILE_THRESH_LOW), y < np.quantile(y, QUANTILE_THRESH_HIGH)
    z_low, z_high = z > np.quantile(z, QUANTILE_THRESH_LOW), z < np.quantile(z, QUANTILE_THRESH_HIGH)

    # Get intersections of low and high quantiles
    xu = np.logical_and(x_low, x_high)
    yu = np.logical_and(y_low, y_high)
    zu = np.logical_and(z_low, z_high)

    # Get intersection of x, y and z
    condition = np.logical_and(xu, np.logical_and(yu, zu))

    xn = x[condition]
    yn = y[condition]
    zn = z[condition]

    x_min, x_max = min(x_min, xn.min()), max(x_max, xn.max())
    y_min, y_max = min(y_min, yn.min()), max(y_max, yn.max())
    z_min, z_max = min(z_min, zn.min()), max(z_max, zn.max())

    las_data.append({"x": xn, "y": yn, "z": zn})

    las_files[n] = os.path.splitext(os.path.split(las_files[n])[1])[0]
    dir_path = os.path.join(PATH_RESULTS, las_files[n])

    os.makedirs(dir_path, exist_ok=True)

# Calculate width-height scale factor
scale_factor = max((x_max - x_min), (y_max - y_min)) / WH_SCALE

# Normalize LiDAR data
for n in tqdm(range(len(las_data)), desc="Normalizing data..."):
    las_data[n]["x"] = np.around((las_data[n]["x"] - x_min) / scale_factor)
    las_data[n]["y"] = np.around((las_data[n]["y"] - y_min) / scale_factor)
    las_data[n]["z"] = np.around((las_data[n]["z"] - z_min) / (z_max - z_min) * Z_MAX)

image_shape = math.ceil((y_max - y_min) / scale_factor) + 1, math.ceil((x_max - x_min) / scale_factor) + 1

with open(f"{PATH_RESULTS}/info.json", "w") as f:
    json.dump({
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "wh_scale": WH_SCALE,
        "z_scale": Z_MAX,
    }, f)


for n in tqdm(range(len(las_data)), desc="Creating image 2D data from 3D data..."):
    # Create image 2D data from 3D data
    count = np.require(np.zeros(image_shape), int, ["ALIGNED"])
    image = np.require(np.zeros(image_shape), int, ["ALIGNED"])

    cdll.createImage(
        las_data[n]["x"].astype(int),
        las_data[n]["y"].astype(int),
        las_data[n]["z"].astype(int),
        count, image, len(las_data[n]['z']), image.ctypes.shape
    )

    # Create median ignoring zeros from image
    image = np.require(image, int, ['ALIGNED'])
    median = image.copy()

    cdll.medianImage(image, median, image.ctypes.shape)

    # Save images
    cv2.imwrite(f"{PATH_RESULTS}/{las_files[n]}/{IMAGE_MEDIAN}{EXTENSION}", np.uint16(median))
    cv2.imwrite(f"{PATH_RESULTS}/{las_files[n]}/{IMAGE_COUNT}{EXTENSION}", np.uint16(count))
    cv2.imwrite(f"{PATH_RESULTS}/{las_files[n]}/{IMAGE_ORIGINAL}{EXTENSION}", np.uint16(image))

    # Debug info
    # print("Count", np.amax(count), count.mean())
    # print(f"Information compression: {100 - np.count_nonzero(image) / len(las_data[n]['z']) * 100:.2f} %")
