from scipy.signal import find_peaks, savgol_filter
from init import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os.path
import glob
import json
import cv2

OFFSET = 10
MAX = 250

sns.set_theme()
plt.figure(figsize=(11, 6))

with open(PATH_RESULTS + "/info.json") as f:
    info = json.load(f)

df = pd.read_csv(f"{PATH_RESULTS}/statistics.csv")
df["Ca"] = abs(df["C"])

# Load images and filenames
images = [cv2.imread(file, cv2.CV_16U) for file in sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_CROPPED}{EXTENSION}"))]
files = [os.path.splitext(os.path.split(file)[1])[0] for file in sorted(glob.glob(f"{PATH_DATA}/*.las"))]

image_sum_z = np.zeros_like(images[0])

for image in images:
    image_sum_z += np.uint16(cv2.absdiff(image, images[0]) / 2)

image_sum_z = images[-1]


def find_diff_peaks(diff: np.ndarray, count: int):
    # TODO diff = np.insert(diff, 0, 0)
    diff = np.append(diff, 0)

    return find_peaks(diff, height=max(diff) / LIM_HEIGHT, distance=len(diff) / (count * LIM_INTERVAL))[0]


def find_blocks(image_block: np.ndarray, count: int, lim: float = LIM_INTERVAL):
    # TODO duplicite code
    image_sum = np.sum(image_block, axis=0)
    # image_sum = np.log(image_sum)  # / np.max(image_sum)
    image_sum = image_sum / np.max(image_sum)

    image_sum = savgol_filter(image_sum, 51, 10)  # TODO two constants

    diff = np.diff(image_sum)
    # diff[np.abs(diff) <= 1 * diff.std()] = 0

    upper_peaks = find_diff_peaks(diff, count)
    lower_peaks = find_diff_peaks(diff * (-1), count)

    positions = []
    positions_heights = []

    # Find all blocks
    for upper_peak in upper_peaks:
        for lower_peak in lower_peaks:
            if lower_peak > upper_peak:
                if lower_peak - upper_peak >= len(diff) / count / lim:
                    positions.append((upper_peak, lower_peak))
                    positions_heights.append((diff[upper_peak] * 50, abs(diff[lower_peak]) * 50))
                    break

    assert len(set([m[0] for m in positions])) == len(set([m[1] for m in positions])), "Something went wrong - peaks"

    return positions


block_positions = find_blocks(image_sum_z, FIELD_COUNT)

block_parts = []
cropped_draw = cv2.cvtColor(np.uint8(image_sum_z / 255), cv2.COLOR_GRAY2BGR)

draw_a = cropped_draw.copy()
draw_b = cropped_draw.copy()
draw_c = cropped_draw.copy()

for n, (block_start, block_end) in enumerate(block_positions):
    if FILTER_BLOCKS is not None and n in FILTER_BLOCKS:
        continue

    folder_path = os.path.join(PATH_RESULTS, f"block_{n + 1}")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    block_parts.append([])

    block = image_sum_z[:, block_start:block_end].T

    block_up = block[:round(block.shape[0] / BLOCK_PART), :]
    block_dw = block[block.shape[0] - round(block.shape[0] / BLOCK_PART):block.shape[0], :]

    blocks_up = find_blocks(block_up, BLOCK_COUNT, 1.8)
    blocks_dw = find_blocks(block_dw, BLOCK_COUNT, 1.8)

    assert len(blocks_up) == len(blocks_dw), "sub-block count does not match"

    for j, ((up_start, up_end), (dw_start, dw_end)) in enumerate(zip(blocks_up, blocks_dw)):
        selected = df[df["y"] == j+1][df["x"] == n+1]

        a = round((selected["A"].tolist()[0] - df["A"].min()) * MAX / (df["A"].max() - df["A"].min()))
        b = round((selected["B"].tolist()[0] - df["B"].min()) * MAX / (df["B"].max() - df["B"].min()))
        c = round((selected["Ca"].tolist()[0] - df["Ca"].min()) * MAX / (df["Ca"].max() - df["Ca"].min()))

        for image_draw, color, letter in [(draw_a, a, "A"), (draw_b, b, "B"), (draw_c, c, "C")]:
            cv2.rectangle(image_draw, (block_start, up_start), (block_end, dw_end), (0, color, 0), -1)

            cv2.putText(
                image_draw, str(round(selected[letter].tolist()[0], 2)),
                (block_start + block.shape[0] - block.shape[0] // 3, up_start + round(block.shape[1] / BLOCK_COUNT / 1.8)),
                cv2.FONT_HERSHEY_SIMPLEX, WH_SCALE // 6000, (0, 255, 255), WH_SCALE // 2000, cv2.LINE_AA
            )
            cv2.putText(
                image_draw, str(selected["label"].tolist()[0]),
                (block_start + block.shape[0] // 20, up_start + round(block.shape[1] / BLOCK_COUNT / 1.8)),
                cv2.FONT_HERSHEY_SIMPLEX, WH_SCALE // 6000, (0, 255, 255), WH_SCALE // 2000, cv2.LINE_AA
            )

    for image_draw in [draw_a, draw_b, draw_c]:
        cv2.line(image_draw, (block_start, 0), (block_start, block.shape[1]), (80, 80, 255), WH_SCALE // 1000)
        cv2.line(image_draw, (block_end, 0), (block_end, block.shape[1]), (20, 20, 230), WH_SCALE // 1000)

        cv2.putText(
            image_draw, str(n + 1),
            (block_start + block.shape[0] // 3, round(block.shape[1] / BLOCK_COUNT * 3)),
            cv2.FONT_HERSHEY_SIMPLEX, WH_SCALE // 1_500, (80, 80, 255), WH_SCALE // 600, cv2.LINE_AA
        )

cv2.imwrite(f"{PATH_RESULTS}/A{EXTENSION}", draw_a)
cv2.imwrite(f"{PATH_RESULTS}/B{EXTENSION}", draw_b)
cv2.imwrite(f"{PATH_RESULTS}/C{EXTENSION}", draw_c)
