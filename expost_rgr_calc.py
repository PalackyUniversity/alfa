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
df = pd.read_csv(PATH_RESULTS + "/statistics.csv")

date_times_str = [col for col in df.columns if "rgr" not in col and col not in ["x", "y", "A", "B", "C"]]
date_times_str = date_times_str[:1] + date_times_str[2:]
dates = [datetime.datetime.strptime(i, "%d/%m/%y %H:%M") for i in date_times_str]

# Save dataframe to csv
df["x"] = df["x"].astype(int)
df["y"] = df["y"].astype(int)

# Calculate relative growth rate
for n in range(len(dates) - 1):
    time_delta = (dates[n+1] - dates[n]) / datetime.timedelta(days=1)
    first_height = np.log(df[date_times_str[n]])
    second_height = np.log(df[date_times_str[n + 1]])

    df["rgr_" + str(n)] = (second_height - first_height) / time_delta

df.to_csv(f"{PATH_RESULTS}/statistics.csv", index=False)
