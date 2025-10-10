from lifelines import AalenJohansenFitter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict


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


def plot_eval_metrics(save_path: str, metrics_dict: Dict, endpoint_name: str = "NDD"):
    """A method to plot evaluation metrics (summarized over folds)

    Args:
        save_path (Path): path of output plots.
        metrics_dict (Dict): dictionary with all metrics per fold.
        endpoint_name (str): name of the endpoint.
    """
    # Transform the data into a long format DataFrame
    df = pd.DataFrame(metrics_dict).T
    df = df.reset_index().melt(id_vars="index", var_name="metric", value_name="value")

    # Extract endpoint and metric name
    df["suffix"] = df["metric"].apply(lambda x: x.split("_")[-1])
    df["metric_name"] = df["metric"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df.loc[df["suffix"] == "0", "endpoint"] = "censoring"
    df.loc[df["suffix"] == "1", "endpoint"] = endpoint_name
    df.loc[df["suffix"] == "2", "endpoint"] = "death"
    df.loc[df["suffix"] == "all", "endpoint"] = "overall survival"

    # Drop NaN values
    df = df.drop(columns=["suffix"]).dropna()

    c_df = df[df["metric_name"].str.startswith("c_")]
    if not c_df.empty:
        g = sns.FacetGrid(
            c_df,
            col="metric_name",
            sharey=True,
            height=4,
        )
        g.map_dataframe(
            sns.boxplot,
            x="endpoint",
            y="value",
            palette="Set2",
            hue="endpoint",
            width=0.6,
        )
        g.add_legend()
        g.set_axis_labels("Endpoint", "Value")
        g.set_titles(col_template="{col_name}")
        plt.savefig(f"{save_path}/concordance.svg")
        plt.close()

    # Plot IBS
    ibs_df = df[df["metric_name"] == "ibs"]
    if not ibs_df.empty:
        ax = sns.boxplot(
            data=ibs_df,
            x="endpoint",
            y="value",
            palette="Set2",
            hue="endpoint",
        )
        ax.set_title("ibs")
        ax.set_xlabel("Endpoint")
        ax.set_ylabel("Value")
        plt.savefig(f"{save_path}/integrated_brier_score.svg")
        plt.close()

    # Plot AUC(t) mean
    auc_mean_df = df[df["metric_name"] == "auc_mean"]
    if not auc_mean_df.empty:
        ax = sns.boxplot(
            data=auc_mean_df,
            x="endpoint",
            y="value",
            palette="Set2",
            hue="endpoint",
        )
        ax.set_title("auc_mean")
        ax.set_xlabel("Endpoint")
        ax.set_ylabel("Value")
        plt.savefig(f"{save_path}/auc_t_mean.svg")
        plt.close()

    # Plot AUC(t)
    auc_t_df = df[df["metric_name"] == "auc_t"]
    t_df = df[df["metric_name"] == "times"]
    if not auc_t_df.empty and not t_df.empty:
        grouped_t = t_df.groupby("endpoint", sort=False)
        grouped_auc_t = auc_t_df.groupby("endpoint", sort=False)

        plot_dict = {}
        for (function, group_t), (_, group_auc) in zip(grouped_t, grouped_auc_t):
            to_concatenate = []
            for row_t, row_auc in zip(group_t.iterrows(), group_auc.iterrows()):
                to_concatenate.append(
                    pd.Series(row_auc[1]["value"], index=row_t[1]["value"])
                )
            df_concat = pd.concat(to_concatenate, axis=1).interpolate()
            df_concat["mean"] = df_concat.mean(axis=1, skipna=False)
            df_concat["std"] = df_concat.std(axis=1, skipna=False)
            plot_dict[function] = df_concat[["mean", "std"]].reset_index()

        plt.figure(figsize=(10, 5))
        for i, k in enumerate(plot_dict.keys()):
            ax = sns.lineplot(
                x="index",
                y="mean",
                data=plot_dict[k],
                label=k,
                color=sns.color_palette("Set2")[i],
            )
            plt.fill_between(
                plot_dict[k]["index"],
                plot_dict[k]["mean"] - plot_dict[k]["std"],
                plot_dict[k]["mean"] + plot_dict[k]["std"],
                color=sns.color_palette("Set2")[i],
                alpha=0.2,
            )
        ax.set_title("Time-dependent area under the curve")
        ax.set_xlim(0)
        ax.set_xlabel("t")
        ax.set_ylabel("AUC(t)")
        plt.savefig(f"{save_path}/auc_t.svg")
        plt.close()

def plot_aalen_johansen(save_path: str,
                        T: pd.Series, 
                        E: pd.Series, 
                        exposed: pd.Series,
                        event_of_interest: int = 1):
    sns.set_style("white")

    T_1 = T[exposed]
    E_1 = E[exposed]
    T_0 = T[~exposed]
    E_0 = E[~exposed]

    ajf_1 = AalenJohansenFitter(calculate_variance=True)
    ajf_1 = ajf_1.fit(T_1, E_1, event_of_interest=event_of_interest)
    ajf_1.plot(color="#c00000", 
                label="Everyone infected",
                xlabel="Days since index", 
                ylabel="Counterfactual cumulative incidence",
                title="Aalen-Johansen estimates")

    ajf_0 = AalenJohansenFitter(calculate_variance=True)
    ajf_0 = ajf_0.fit(T_0, E_0, event_of_interest=event_of_interest)
    ajf_0.plot(color="#699aaf", 
                label="No one infected",
                xlabel="Days since index", 
                ylabel="Counterfactual cumulative incidence",
                title="Aalen-Johansen estimates")

    plt.savefig(f"{save_path}/aalen_johansen.svg", bbox_inches='tight')