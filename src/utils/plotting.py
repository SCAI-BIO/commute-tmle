from lifelines import AalenJohansenFitter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
from typing import Dict, List, Optional


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

def plot_hte_distribution(save_path: str,
                          hte_df: pd.DataFrame,
                          quantile: float,
                          target_time: Optional[float] = None):
    sns.set_style("white")
    plt.figure(figsize=(6,4))

    # colors for the quantiles
    palette = {f'bottom {quantile*100}%': '#984ea3', f'middle {100 - 2*quantile*100}%': '#9e9e9e', f'top {quantile*100}%': '#e69f00'}

    ax = pt.RainCloud(x='group', y='hte', data=hte_df, hue='quantile',
                    palette=palette, bw=0.25, width_viol=0.5)
    ax.get_xaxis().set_ticks([])
    ax.set_xlabel("")
    ax.set_ylabel("HTE Estimate")

    if target_time is not None:
        plt.savefig(f"{save_path}/hte_{target_time}.svg", bbox_inches='tight')
    else:
        plt.savefig(f"{save_path}/hte.svg", bbox_inches='tight')
    plt.close()

def annotate_brackets(num1, num2, data, center, height, ax, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate plot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param ax: matplotlib axes instance
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

def plot_quantile_distributions_continuous(input_df: pd.DataFrame, 
                                           hte_df: pd.DataFrame, 
                                           quantile: float,
                                           continuous_features: List[str],
                                           save_path: str,
                                           target_time: Optional[float] = None):
    df = pd.concat([input_df, hte_df], axis=1)
    df["quantile"] = df["quantile"].cat.remove_categories([f'middle {100 - 2*quantile*100}%'])
    df = df[(df["quantile"].str.startswith("top")) | (df["quantile"].str.startswith("bottom"))]

    palette = {f'bottom {quantile*100}%': '#984ea3', f'middle {100 - 2*quantile*100}%': '#9e9e9e', f'top {quantile*100}%': '#e69f00'}

    sns.set_style(style="white")
    fig, axes = plt.subplots(ncols=len(continuous_features), figsize=(4*len(continuous_features), 4))
    axes = axes.flatten()
    for i, f in enumerate(continuous_features):
        pt.RainCloud(x="quantile", y=f, data=df,
                        palette=palette, bw=0.25, width_viol=0.5, ax=axes[i])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_title(f)
        heights = [df[df["quantile"] == f"bottom {quantile*100}%"][f].max(),
                    df[df["quantile"] == f"top {quantile*100}%"][f].max()]

        bottom = df[df["quantile"] == f"bottom {quantile*100}%"][f]
        top = df[df["quantile"] == f"top {quantile*100}%"][f]
        _, p = ttest_ind(bottom, top, nan_policy='omit')
        annotate_brackets(0, 1, p, np.arange(2), heights, ax=axes[i], dh=max(heights)*0.02, barh=max(heights)*0.01, maxasterix=3)
    if target_time is not None:
        plt.savefig(f"{save_path}/distributions_continuous_features_{target_time}.svg", bbox_inches='tight')
    else:
        plt.savefig(f"{save_path}/distributions_continuous_features.svg", bbox_inches='tight')
    plt.close()

def plot_quantile_distributions_binary(input_df: pd.DataFrame, 
                                           hte_df: pd.DataFrame, 
                                           quantile: float,
                                           binary_features: List[str],
                                           save_path: str,
                                           target_time: Optional[float] = None):
    df = pd.concat([input_df, hte_df], axis=1)
    df["quantile"] = df["quantile"].cat.remove_categories([f'middle {100 - 2*quantile*100}%'])
    df = df[(df["quantile"].str.startswith("top")) | (df["quantile"].str.startswith("bottom"))]

    palette = {f'bottom {quantile*100}%': '#984ea3', f'middle {100 - 2*quantile*100}%': '#9e9e9e', f'top {quantile*100}%': '#e69f00'}

    sns.set_style(style="white")
    fig, axes = plt.subplots(ncols=len(binary_features), figsize=(4*len(binary_features), 4))
    axes = axes.flatten()

    for i, f in enumerate(binary_features):
        frequencies = df.groupby(["quantile"])[f].sum()
        heights = list(frequencies / df.groupby(["quantile"])[f].count() * 100)
        axes[i].bar(x=[f"bottom {quantile*100}%", f"top {quantile*100}%"], height=heights, color=[palette[f"bottom {quantile*100}%"], palette[f"top {quantile*100}%"]])
        axes[i].set_ylim([0,100])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("%")
        axes[i].set_title(f)

        try:
            contingency = np.array([frequencies.values, len(df) - frequencies.values])
            _, p, _, _ = chi2_contingency(contingency)
            annotate_brackets(
                0,
                1,
                p,
                np.arange(len(heights)),
                heights,
                ax=axes[i],
                dh=5,
                barh=2,
                maxasterix=3,
            )
        except ValueError:
            continue

    if target_time is not None:
        plt.savefig(f"{save_path}/distributions_binary_features_{target_time}.svg", bbox_inches='tight')
    else:
        plt.savefig(f"{save_path}/distributions_binary_features.svg", bbox_inches='tight')
    plt.close()
