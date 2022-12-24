from init import PATH_RESULTS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime

COLORS = ["blue", "orange", "green"]

sns.set_theme()

df = pd.read_csv(f"{PATH_RESULTS}/statistics.csv")
print(df)
df = df[df["x"] == 1][df["y"].isin(range(24, 30))]

date_times_str = [col for col in df.columns if "rgr" not in col and col not in ["x", "y", "A", "B", "C"]]
date_times_str = date_times_str[:1] + date_times_str[2:]
date_times = [datetime.datetime.strptime(i, "%d/%m/%y %H:%M") for i in date_times_str]
print(date_times)
t = np.linspace(0, (date_times[-1] - date_times[0]).days + 1, 1_000)
x = [date_times[0] + datetime.timedelta(days=i) for i in t]

for block in df["x"].unique():
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 10), sharex=True)

    for _, row in df[df["x"] == block].iterrows():
        y = row["A"] / (1 + np.exp(-(row["B"]*t + row["C"])))
        # ax1.scatter(date_times, row[3:-4])
        ax1.plot(date_times, row[date_times_str])
        ax2.plot(date_times[1:], row[[col for col in df.columns if "rgr" in col]])
        ax3.plot(x, y, label="A" + str(int(row["y"])))

    # fig.suptitle("Barley growth during vegetation season in selected representative field-plots")
    ax1.set_title("Height")
    ax2.set_title("Relative growth rate")
    ax3.set_title("Logistic curve")

    ax1.set_ylabel("Height [m]")
    ax2.set_ylabel("RGR")
    ax3.set_ylabel("Height [m]")

    plt.xlabel("Datetime")

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")

    plt.savefig(f"docs/paper-output.png")
    plt.show()
