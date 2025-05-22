import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from .utils.plotting import make_histogram
from .utils.utils import parse_path_for_experiment
from conf.config import RunConfig

# Set up the config store
cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)


# filter functions for infection waves
waves_fns = {
    None: lambda x: x >= "2020-01-01",
    "first_wave": lambda x: (x >= "2020-01-01") & (x <= "2020-08-31"),
    "second_wave": lambda x: (x >= "2020-09-01") & (x <= "2021-05-31"),
    "third_wave": lambda x: (x >= "2021-06-01") & (x <= "2021-11-31"),
    "omicron_wave": lambda x: (x >= "2021-12-01") & (x <= "2022-06-30"),
}

# filter functions for subset (hospitalized or tested)
subset_fns = {
    None: lambda x: x["index_date"].notna(),
    "hospital": lambda x: (x["index_date"].notna()) & (x["hospitalized_due_to_covid"]),
    "tested": lambda x: (x["index_date"].notna())
    & (x["date_first_tested_positive"] == x["index_date"]),
}

endpoint_fns = {
    0: lambda x: "censored",
    1: lambda x: x,
    2: lambda x: "death",
}


def filter_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where the index date is after the event, death or censoring date.
    Also force censoring_date to censoring_global if it would come later in time.
    """
    df = df[
        (df["index_date"] < df["censoring_global"])
        & (df["event_date"].isna() | (df["index_date"] < df["event_date"]))
        & (df["death_date"].isna() | (df["index_date"] < df["death_date"]))
    ].copy()
    df.loc[df["censoring_global"] < df["censoring_date"], "censoring_date"] = df[
        "censoring_global"
    ]
    return df


def sample_control_index_dates(
    covid_df: pd.DataFrame, control_pool_df: pd.DataFrame, sample_age: bool
):
    """
    Sample index dates for the control group based on the COVID group.
    If sample_age is True, sample the age at index date from the COVID group and
    set the index at the date at which a control patient was as old.
    """
    sample_indices = np.random.randint(0, len(covid_df), size=len(control_pool_df))
    if sample_age:
        age_samples = covid_df.age_at_index.values[sample_indices]
        index_samples = control_pool_df["birth_date"] + pd.to_timedelta(
            age_samples, unit="D"
        )
    else:
        index_samples = covid_df.index_date.values[sample_indices]
    return index_samples


def sample_control_censoring_dates(
    covid_df: pd.DataFrame, control_pool_df: pd.DataFrame
):
    """
    Sample follow-up times from COVID group to set censoring dates accordingly for the control group.
    """
    covid_df_only_censored = covid_df[covid_df["event_indicator"] == 0]
    sample_indices = np.random.randint(
        0, len(covid_df_only_censored), size=len(control_pool_df)
    )
    fu_samples = covid_df_only_censored.event_time.values[sample_indices]
    censoring_samples = control_pool_df["index_date"] + pd.to_timedelta(
        fu_samples, unit="D"
    )
    return censoring_samples


def get_event_indicator_and_time(df):
    """
    Get event indicator and time to event for the given DataFrame.
    The event indicator is 0 for censoring, 1 for event and 2 for death.
    The event time is the time from index date to the earliest of the three dates.
    """
    # get the earliest date of censoring, event and death
    earliest_date = df[["censoring_date", "event_date", "death_date"]].min(axis=1)
    event_time = (earliest_date - df["index_date"]).dt.days
    # get the event indicator
    event_indicator = (
        df[["censoring_date", "event_date", "death_date"]]
        .idxmin(axis=1)
        .map({"censoring_date": 0, "event_date": 1, "death_date": 2})
    )

    return pd.concat(
        (event_indicator, event_time), axis=1, keys=["event_indicator", "event_time"]
    )


def prepandemic_control_fu(
    covid_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    wave: Optional[str],
    subset: Optional[str],
) -> pd.DataFrame:
    """
    Control group design where the control group is sampled from the prepandemic
    population, with index dates set at the dates at which controls were as old as sampled
    COVID patients at the time of their first documented infection.
    The follow-up time is set with samples from the COVID group.

    Parameters
    ----------
    covid_df : pd.DataFrame
        DataFrame containing the COVID group data.
    control_pool_df : pd.DataFrame
        DataFrame containing the control pool data.
    wave : Optional[str]
        The wave of infection to filter the COVID group.
    subset : Optional[str]
        The subset of the population to filter the COVID group (hospitalized or tested).
    Returns
    -------
    -------
    pd.DataFrame
        DataFrame containing the combined data of the COVID group and control group.
    """
    # filter COVID patients based on the subset and wave
    covid_df = covid_df[
        (subset_fns[subset](covid_df)) & ((waves_fns[wave](covid_df["index_date"])))
    ]
    covid_df.loc[:, ["event_indicator", "event_time"]] = get_event_indicator_and_time(
        covid_df
    )
    # no control follow-up in 2020 or later
    control_pool_df["censoring_global"] = pd.to_datetime("2019-12-31")
    # index dates will be set at the dates at which controls were as old as sampled
    # COVID patients at the time of their first documented infection
    control_pool_df["index_date"] = sample_control_index_dates(
        covid_df, control_pool_df, sample_age=True
    )
    # censoring dates will be set after follow-up corresponding to that of a sampled COVID patient
    control_pool_df["censoring_date"] = sample_control_censoring_dates(
        covid_df, control_pool_df
    )
    control_pool_df["age_at_index"] = (
        control_pool_df["index_date"] - control_pool_df["birth_date"]
    ).dt.days
    control_pool_df = filter_invalid_dates(control_pool_df)
    control_pool_df[["event_indicator", "event_time"]] = get_event_indicator_and_time(
        control_pool_df
    )

    df = pd.concat([covid_df, control_pool_df], ignore_index=True)
    return df


def equal_control_fu(
    covid_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    wave: Optional[str],
    subset: Optional[str],
) -> pd.DataFrame:
    """
    Control group design where the control group is sampled from the same time period as the
    COVID group, with index dates directly sampled from the COVID group.
    The follow-up time is set with samples from the COVID group.

    Parameters
    ----------
    covid_df : pd.DataFrame
        DataFrame containing the COVID group data.
    control_pool_df : pd.DataFrame
        DataFrame containing the control pool data.
    wave : Optional[str]
        The wave of infection to filter the COVID group.
    subset : Optional[str]
        The subset of the population to filter the COVID group (hospitalized or tested).
    Returns
    -------
    -------
    pd.DataFrame
        DataFrame containing the combined data of the COVID group and control group.
    """
    # filter COVID patients based on the subset and wave
    covid_df = covid_df[
        subset_fns[subset](covid_df) & (waves_fns[wave](covid_df["index_date"]))
    ]
    covid_df.loc[:, ["event_indicator", "event_time"]] = get_event_indicator_and_time(
        covid_df
    )
    # control index dates are directly sampled from the COVID group
    control_pool_df["index_date"] = sample_control_index_dates(
        covid_df, control_pool_df, sample_age=False
    )
    # set censoring_date to global censoring
    control_pool_df["censoring_date"] = control_pool_df["censoring_global"]
    control_pool_df["age_at_index"] = (
        control_pool_df["index_date"] - control_pool_df["birth_date"]
    ).dt.days
    control_pool_df = filter_invalid_dates(control_pool_df)
    control_pool_df[["event_indicator", "event_time"]] = get_event_indicator_and_time(
        control_pool_df
    )

    df = pd.concat([covid_df, control_pool_df], ignore_index=True)
    return df


def tested_positive_vs_negative(
    covid_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    wave: Optional[str],
) -> pd.DataFrame:
    """
    Control group design where the control group are patients that got first tested in the same
    time period as the COVID group, with index dates directly sampled from the COVID group.
    The follow-up time is set with samples from the COVID group.
    Parameters
    ----------
    covid_df : pd.DataFrame
        DataFrame containing the COVID group data.
    control_pool_df : pd.DataFrame
        DataFrame containing the control pool data.
    wave : Optional[str]
        The wave of infection to filter the COVID group.
    Returns
    -------
    -------
    pd.DataFrame
        DataFrame containing the combined data of the COVID group and control group.
    """
    # only use COVID patients that were tested positive
    covid_df = covid_df[
        subset_fns["tested"](covid_df) & (waves_fns[wave](covid_df["index_date"]))
    ]
    covid_df.loc[:, ["event_indicator", "event_time"]] = get_event_indicator_and_time(
        covid_df
    )
    # only use control patients that were tested (negative) in the same time period as the COVID group
    control_pool_df = control_pool_df[
        (~control_pool_df["date_first_tested"].isna())
        & (waves_fns[wave](control_pool_df["date_first_tested"]))
    ].copy()
    # control index dates are directly sampled from the COVID group
    control_pool_df["index_date"] = sample_control_index_dates(
        covid_df, control_pool_df, sample_age=False
    )
    # set censoring_date to global censoring
    control_pool_df["censoring_date"] = control_pool_df["censoring_global"]
    control_pool_df["age_at_index"] = (
        control_pool_df["index_date"] - control_pool_df["birth_date"]
    ).dt.days
    control_pool_df = filter_invalid_dates(control_pool_df)
    control_pool_df[["event_indicator", "event_time"]] = get_event_indicator_and_time(
        control_pool_df
    )

    df = pd.concat([covid_df, control_pool_df], ignore_index=True)
    return df


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: RunConfig):

    np.random.seed(cfg.general.seed)

    df = pd.read_csv(
        cfg.general.input_csv,
        parse_dates=[
            "birth_date",
            "date_first_tested_positive",
            "date_first_covid_diagnosis",
            "date_first_tested",
            "date_first_ad_diagnosis",
            "date_first_pd_diagnosis",
            "date_first_unspecified_dementia_diagnosis",
            "death_date",
            "censoring_global",
        ],
    )

    # for COVID group, set index to date of first reported COVID infection and filter based on min_age_at_index
    df["index_date"] = df[
        ["date_first_tested_positive", "date_first_covid_diagnosis"]
    ].min(axis=1)
    df["age_at_index"] = (df["index_date"] - df["birth_date"]).dt.days

    # set the event date of the chosen endpoint
    if cfg.experiment.endpoint == "composite":
        df["event_date"] = df[
            [
                "date_first_ad_diagnosis",
                "date_first_pd_diagnosis",
                "date_first_unspecified_dementia_diagnosis",
            ]
        ].min(axis=1)
    elif cfg.experiment.endpoint == "ad":
        df["event_date"] = df["date_first_ad_diagnosis"]
    elif cfg.experiment.endpoint == "pd":
        df["event_date"] = df["date_first_pd_diagnosis"]
    elif cfg.experiment.endpoint == "unspecified_dementia":
        df["event_date"] = df["date_first_unspecified_dementia_diagnosis"]
    else:
        raise ValueError(f"Unknown endpoint: {cfg.experiment.endpoint}")

    if (
        cfg.experiment.subset == "hospital"
        and df["hospitalized_due_to_covid"].isnull().all()
    ):
        raise RuntimeError("No information available on hospitalization status.")

    # split to COVID group and control pool
    covid_df = df.loc[
        (~df["index_date"].isna())
        & (df["age_at_index"] >= cfg.general.min_age_at_index)
    ].copy()
    covid_df["exposed"] = True
    covid_df["censoring_date"] = covid_df["censoring_global"]
    covid_df = filter_invalid_dates(covid_df).reset_index(drop=True)
    # control pool are all patients that have no COVID diagnosis
    control_pool_df = df.loc[df["index_date"].isna()].copy().reset_index(drop=True)
    control_pool_df["exposed"] = False

    # filter data, set index and censoring dates according to chosen experiment
    if cfg.experiment.control_group_design == "prepandemic_control_fu":
        output = prepandemic_control_fu(
            covid_df=covid_df,
            control_pool_df=control_pool_df,
            wave=cfg.experiment.wave,
            subset=cfg.experiment.subset,
        )
    elif cfg.experiment.control_group_design == "equal_control_fu":
        output = equal_control_fu(
            covid_df=covid_df,
            control_pool_df=control_pool_df,
            wave=cfg.experiment.wave,
            subset=cfg.experiment.subset,
        )
    elif cfg.experiment.control_group_design == "tested_positive_vs_negative":
        output = tested_positive_vs_negative(
            covid_df=covid_df,
            control_pool_df=control_pool_df,
            wave=cfg.experiment.wave,
        )
    else:
        raise ValueError(
            f"Unknown control group design: {cfg.experiment.control_group_design}"
        )

    # keep only some variables and transform age to years
    output["age_at_index"] /= 365.25
    output = output[
        [
            "patient_id",
            "exposed",
            "index_date",
            "age_at_index",
            "event_time",
            "event_indicator",
        ]
    ]

    # save output
    save_path = parse_path_for_experiment(cfg.general.dates_set_path, cfg.experiment)
    output.to_csv(save_path, index=False, float_format="%.2f")

    # store the event counts per exposure group
    event_counts = (
        output.groupby("exposed")["event_indicator"]
        .value_counts()
        .unstack(fill_value=0)
    )
    with open(f"{cfg.general.output_path}/event_counts.txt", "w") as f:
        print(event_counts, file=f)
    # save plots
    # plot histogragram of index dates and age at index
    for feature_name in ["index_date", "age_at_index"]:
        make_histogram(
            output,
            feature_name=feature_name,
            endpoint_name="all",
        )
        plt.savefig(f"{cfg.general.output_path}/histogram_{feature_name}.png")
        plt.close()
    # plot histogram of event time for each endpoint (0, 1, 2)
    for endpoint in [0, 1, 2]:
        endpoint_name = endpoint_fns[endpoint](cfg.experiment.endpoint)
        make_histogram(
            output[output["event_indicator"] == endpoint],
            feature_name="event_time",
            endpoint_name=endpoint_name,
        )
        plt.savefig(
            f"{cfg.general.output_path}/histogram_event_time_{endpoint_name}.png"
        )
        plt.close()


if __name__ == "__main__":
    main()
