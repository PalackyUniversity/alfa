from init import PATH_RESULTS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime

COLORS = ["blue", "orange", "green"]

sns.set_theme()

df = pd.read_csv(f"{PATH_RESULTS}/statistics.csv")

date_times = [datetime.datetime.strptime(i, "%d/%m/%y %H:%M") for i in df.columns[3:-4]]

t = np.linspace(0, (date_times[-1] - date_times[0]).days + 1, 1_000)
x = [date_times[0] + datetime.timedelta(days=i) for i in t]

for block in df["x"].unique():
    plt.figure(figsize=(11, 6))
    for _, row in df[df["x"] == block].iterrows():
        y = row["A"] / (1 + np.exp(-(row["B"]*t + row["C"])))

        plt.plot(x, y, color=COLORS[int(block) - 1], alpha=0.1)

    plt.title(f"Sigmoid fits of sub-blocks from block {int(block)}")
    plt.xlabel("Datetime")
    plt.ylabel("Height [m]")
    plt.savefig(f"docs/sigmoid-{int(block)}.png")
    plt.show()
