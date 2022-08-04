from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import maximum_filter
from tqdm import tqdm
from init import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statistics
import os.path
import json
import cv2

# Set plot theme and size
sns.set_theme()
plt.figure(figsize=(11, 6))

# Load cached information
with open(PATH_RESULTS + "/info.json") as f:
    info = json.load(f)

# Create dataframe
df = pd.DataFrame()

# Sum images
images = [cv2.imread(file, cv2.CV_16U) for file in sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_CROPPED}{EXTENSION}"))]
image_sum_z = np.zeros_like(images[0])

for image in images:
    image_sum_z += np.uint16(cv2.absdiff(image, images[0]) / IMAGE_SUM_DIVIDE)

# image_sum_z = images[-1]

# def monotone_increasing(lst):
#    pairs = zip(lst, lst[1:])
#    return all(itertools.starmap(operator.le, pairs))


def find_diff_peaks(diff: np.ndarray, count: int):
    # TODO diff = np.insert(diff, 0, 0)
    diff = np.append(diff, 0)

    return find_peaks(diff, height=max(diff) / LIM_HEIGHT, distance=len(diff) / (count * LIM_INTERVAL))[0]


def find_blocks(image_block: np.ndarray, count: int, name: str, path: str, lim: float = LIM_INTERVAL):
    image_sum = np.sum(image_block, axis=0)
    image_sum = np.log(image_sum)  # / np.max(image_sum)

    # image_sum = savgol_filter(image_sum, 51, 5)

    diff = np.diff(image_sum)
    # diff[np.abs(diff) <= 1 * diff.std()] = 0

    upper_peaks = find_diff_peaks(diff, count)
    lower_peaks = find_diff_peaks(diff * (-1), count)

    positions = []

    # Find all blocks
    for upper_peak in upper_peaks:
        for lower_peak in lower_peaks:
            if lower_peak > upper_peak:
                if lower_peak - upper_peak >= len(diff) / count / lim:
                    positions.append((upper_peak, lower_peak))
                    break

    # Visualize up part
    plt.plot(diff)
    plt.plot(upper_peaks, diff[upper_peaks], "x", label="Start")
    plt.plot(lower_peaks, diff[lower_peaks], "x", label="End")
    plt.title(f"{name} - derivative on axis 0 (X)")
    plt.xlabel("Pixel index on axis X")
    plt.ylabel("Derivative of divided sum")
    plt.legend()
    plt.savefig(f"{PATH_RESULTS}/{path}_derivative{EXTENSION}")
    plt.cla()

    # TODO sum graph error

    plt.plot(image_sum, label="Sum")
    plt.plot(upper_peaks, image_sum[upper_peaks], "x", label="Start")
    plt.plot(lower_peaks, image_sum[lower_peaks], "x", label="End")
    plt.title(f"{name} - sum on axis 0 (X)")
    plt.xlabel("Pixel index on axis X")
    plt.ylabel("Pixel value sum")
    plt.legend()
    plt.savefig(f"{PATH_RESULTS}/{path}_sum{EXTENSION}")
    plt.cla()

    return positions

block_positions = find_blocks(image_sum_z, FIELD_COUNT, "Detect peaks from derivative", "blocks")
block_parts = []
cropped_draw = cv2.cvtColor(np.uint8(image_sum_z / 255), cv2.COLOR_GRAY2BGR)

for n, (block_start, block_end) in tqdm(enumerate(block_positions), desc="Analyzing..."):
    # If filtering is enabled
    if FILTER_BLOCKS is not None and n in FILTER_BLOCKS:
        continue

    # Create directories
    folder_path = os.path.join(PATH_RESULTS, f"block_{n + 1}")
    os.makedirs(folder_path, exist_ok=True)

    block_parts.append([])

    block = image_sum_z[:, block_start:block_end].T

    # Split to parts because of deformation
    block_up = block[:round(block.shape[0] / BLOCK_PART), :]
    block_dw = block[block.shape[0] - round(block.shape[0] / BLOCK_PART):block.shape[0], :]

    blocks_up = find_blocks(block_up, BLOCK_COUNT, "Splits - upper part", f"block_{n + 1}/fit_up", lim=1.8)
    blocks_dw = find_blocks(block_dw, BLOCK_COUNT, "Splits - bottom part", f"block_{n + 1}/fit_dw", lim=1.8)

    assert len(blocks_up) == len(blocks_dw), "sub-block count does not match"

    # Set deformation
    warp_shift = statistics.median(np.array(blocks_up).flatten() - np.array(blocks_dw).flatten())

    src_tri = np.array([[0, 0], [block.shape[1] - 1, 0], [0, block.shape[0] - 1]]).astype(np.float32)
    dst_tri = np.array([[0, 0], [block.shape[1] - 1, 0], [warp_shift, block.shape[0] - 1]]).astype(np.float32)

    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    warp_maxes = []

    max_filter_size = round(block.shape[1] / BLOCK_COUNT / 10)

    for image in images:
        block_parts[n].append([])

        block = image[:, block_start:block_end].T

        # Remove deformation
        warp_dst = cv2.warpAffine(block, warp_mat, (block.shape[1], block.shape[0]))
        warp_max = maximum_filter(warp_dst, size=max_filter_size)

        # Remove outliers
        median = np.uint16(cv2.medianBlur(np.uint8(warp_max / 255), max_filter_size * 2 + 1)) * 255
        median_max = maximum_filter(median, size=max_filter_size)

        where = np.where(warp_max >= np.quantile(warp_max, QUANTILE_BLOCK_THRESH_HIGH))
        warp_max[where] = median_max[where]

        warp_maxes.append(warp_max)

        for start, end in blocks_up[:BLOCK_COUNT]:
            # Calculate height by subtracting first LiDAR file
            height = (warp_max.astype(int) - warp_maxes[0].astype(int))[:, start:end] \
                     / Z_MAX * (info["z_max"] - info["z_min"])

            # plt.hist(height.flatten(), bins=50)

            # TODO Python class
            height = {
                "height_median": np.median(height),
                "height_mean": np.mean(height),
                "height_std": np.std(height),
                "height_min": np.min(height),
                "height_max": np.max(height),
                "height_quantile_1%": np.quantile(height, 0.01),
                "height_quantile_5%": np.quantile(height, 0.05),
                "height_quantile_25%": np.quantile(height, 0.25),
                "height_quantile_75%": np.quantile(height, 0.75),
                "height_quantile_95%": np.quantile(height, 0.95),
                "height_quantile_99%": np.quantile(height, 0.99),
            }

            block_parts[n][-1].append(height)

    cv2.imwrite(f"{folder_path}/0_block{EXTENSION}", block)
    cv2.imwrite(f"{folder_path}/1_warped{EXTENSION}", warp_dst)
    cv2.imwrite(f"{folder_path}/1_warped_max{EXTENSION}", warp_maxes[-1])

    # Draw
    block = cv2.cvtColor(np.uint8(block / Z_MAX * 255), cv2.COLOR_GRAY2RGB)
    warp_dst = cv2.cvtColor(np.uint8(warp_dst / Z_MAX * 255), cv2.COLOR_GRAY2RGB)

    for j, ((up_start, up_end), (dw_start, dw_end)) in enumerate(zip(blocks_up, blocks_dw)):
        cv2.line(block, (up_start, 0), (dw_start, block.shape[0]), (0, 255, 255), 1)
        cv2.line(block, (up_end, 0), (dw_end, block.shape[0]), (0, 200, 200), 1)

        cv2.line(warp_dst, (up_start, 0), (up_start, block.shape[0]), (0, 255, 0), 1)
        cv2.line(warp_dst, (up_end, 0), (up_end, block.shape[0]), (0, 200, 0), 1)

        cv2.putText(
            cropped_draw, str(j + 1),
            (block_start + block.shape[0] // 20, up_start + round(block.shape[1] / BLOCK_COUNT / 1.8)),
            cv2.FONT_HERSHEY_SIMPLEX, WH_SCALE // 6000, (0, 255, 255), WH_SCALE // 2000, cv2.LINE_AA
        )

        cv2.line(cropped_draw, (block_start, up_start), (block_end, dw_start), (0, 255, 255), 2)
        cv2.line(cropped_draw, (block_start, up_end), (block_end, dw_end), (0, 150, 200), 2)

    cv2.line(cropped_draw, (block_start, 0), (block_start, block.shape[1]), (80, 80, 255), WH_SCALE // 1000)
    cv2.line(cropped_draw, (block_end, 0), (block_end, block.shape[1]), (20, 20, 230), WH_SCALE // 1000)

    cv2.putText(
        cropped_draw, str(n + 1),
        (block_start + block.shape[0] // 3, round(block.shape[1] / BLOCK_COUNT * 3)),
        cv2.FONT_HERSHEY_SIMPLEX, WH_SCALE // 1_500, (80, 80, 255), WH_SCALE // 600, cv2.LINE_AA
    )

    cv2.imwrite(f"{folder_path}/0_block_draw{EXTENSION}", block)
    cv2.imwrite(f"{folder_path}/1_warped_draw{EXTENSION}", warp_dst)

cv2.imwrite(f"{PATH_RESULTS}/{IMAGE_CROPPED_DRAW}{EXTENSION}", cropped_draw)
cv2.imwrite(f"{PATH_RESULTS}/{IMAGE_SUM}{EXTENSION}", image_sum_z)

values = np.array(block_parts).transpose((0, 2, 1))
data = []

for x, block in enumerate(values):
    # For each block create line

    for y, block_part in enumerate(block):
        height_median = [v["height_median"] for v in block_part]

        row = {"x": x + 1, "y": y + 1}
        row.update(dict(zip(dates, height_median)))

        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

        plt.plot(dates, height_median, "b", alpha=0.3)

    # Format and save plot
    plt.title(f"Block {x + 1}")
    plt.xlabel("Time")
    plt.ylabel("Height [m]")
    plt.savefig(f"results/block_{x + 1}{EXTENSION}")
    plt.cla()

# Save dataframe to csv
df["x"] = df["x"].astype(int)
df["y"] = df["y"].astype(int)

# Calculate relative growth rate
for n in range(len(dates) - 1):
    time_delta = (dates[n+1] - dates[n]) / datetime.timedelta(days=1)
    first_height = np.log(df[dates[n]])
    second_height = np.log(df[dates[n+1]])

    df["rgr_" + str(n)] = (second_height - first_height) / time_delta

df.to_csv(f"{PATH_RESULTS}/statistics.csv", index=False)
