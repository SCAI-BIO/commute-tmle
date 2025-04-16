import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_histogram(df: pd.DataFrame, feature_name: str, endpoint_name: str):
    hist_input = [
        df.loc[df["exposed"], feature_name],
        df.loc[~df["exposed"], feature_name],
    ]
    values, bin_edges, _ = plt.hist(hist_input, bins=50)

    sns.set_style("white")
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1],
        values[0],
        width=np.diff(bin_edges),
        color="#c00000",
        alpha=0.4,
        label="exposed",
        edgecolor="black",
    )
    ax.bar(
        bin_edges[:-1],
        values[1],
        width=np.diff(bin_edges),
        color="#699aaf",
        alpha=0.4,
        label="not exposed",
        edgecolor="black",
    )

    ax.set_xlabel(feature_name.replace("_", " "))
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Distribution of {feature_name.replace('_', ' ')} ({endpoint_name})")
    ax.legend()
    fig.autofmt_xdate()
    if pd.api.types.is_datetime64_any_dtype(df[feature_name]):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
