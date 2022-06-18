import datetime
import os.path
import glob
import cv2

# Paths and filenames
PATH_DATA = "data"
PATH_RESULTS = "results"

IMAGE_ORIGINAL = "1_original"
IMAGE_COUNT = "1_count"
IMAGE_MEDIAN = "2_median"
IMAGE_ROTATED = "3_rotated"
IMAGE_CROPPED = "4_cropped"
IMAGE_CROPPED_DRAW = "4_cropped_draw"

# Extension for all image exports
EXTENSION = ".png"

# Quantile thresholds for removing outliers
QUANTILE_THRESH_LOW = 0.001
QUANTILE_THRESH_HIGH = 0.999

# Scale constants for images
# - the bigger value the bigger shape of image
WH_SCALE = 7_000
Z_MAX = 65_535

# GUI constants
GUI_STEP_PX = 1
GUI_STEP_ANGLE = 0.2
GUI_CROP_INIT = 5  # fraction
GUI_CROP_STEP = 100

# Estimated parameters for simplifying analysis
FIELD_COUNT = 5
BLOCK_COUNT = 48

# Split block to parts to calculate deformation
BLOCK_PART = 3  # fraction

# Limits for simplifying guessed parameters above
LIM_INTERVAL = 1.3
LIM_HEIGHT = 10

# Create tree if not exists
os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_RESULTS, exist_ok=True)

# Load images, files and get dates
images = [cv2.imread(file, cv2.CV_16U) for file in sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_MEDIAN}{EXTENSION}"))]
files = []
dates = []

for i in sorted(glob.glob(f"{PATH_RESULTS}/*")):
    if not os.path.isdir(i):
        continue

    filename = os.path.splitext(os.path.split(i)[1])[0]
    date, time = os.path.basename(filename).split()[0].split("_")[:2]

    files.append(filename)
    dates.append(datetime.datetime(
        year=int("20" + date[:2]), month=int(date[2:4]), day=int(date[4:6]),
        hour=int(time[:2]), minute=int(time[2:4]), second=int(time[4:6]))
    )

assert len(images) == len(files) == len(dates), "Length of images and file and date does not match"
