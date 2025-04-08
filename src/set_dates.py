import pandas as pd
from dateutil.relativedelta import relativedelta
import hydra
import numpy as np
from omegaconf import DictConfig
from typing import Optional

waves_fns = {
    None: lambda x: x,
    "first_wave": lambda x: x <= "2020-08-31",
    "second_wave": lambda x: (x >= "2020-09-01") & (x <= "2021-05-31"),
    "third_wave": lambda x: (x >= "2021-06-01") & (x <= "2021-11-31"),
    "omicron_wave": lambda x: (x >= "2021-12-01") & (x <= "2022-06-30"),
}

subset_fns = {
    None: lambda x: x,
    "hospital": lambda x: x["hospitalized_due_to_covid"],
    "tested": lambda x: x["date_first_tested_positive"] == x["index_date"],
}


def filter_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where the index date is after the event date or the censoring date.
    """
    df = df[
        (df["index_date"] < df["censoring_global"])
        & (df["event_date"].isna() | (df["index_date"] < df["event_date"]))
        & (df["death_date"].isna() | (df["index_date"] < df["death_date"]))
        & (df["censoring_date"] <= df["censoring_global"])
    ].copy()
    return df


def sample_control_index_dates(
    covid_df: pd.DataFrame, control_pool_df: pd.DataFrame, sample_age: bool
):
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
    sample_indices = np.random.randint(0, len(covid_df), size=len(control_pool_df))
    fu_samples = covid_df.event_time.values[sample_indices]
    censoring_samples = control_pool_df["index_date"] + pd.to_timedelta(
        fu_samples, unit="D"
    )
    return censoring_samples


def get_event_indicator_and_time(df):
    earliest_date = df[["censoring_date", "event_date", "death_date"]].min(axis=1)
    event_time = (earliest_date - df["index_date"]).dt.days
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
    covid_df = covid_df[
        subset_fns[subset](covid_df) & (waves_fns[wave](covid_df["index_date"]))
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
    covid_df = covid_df[
        subset_fns[subset](covid_df) & (waves_fns[wave](covid_df["index_date"]))
    ]
    covid_df.loc[:, ["event_indicator", "event_time"]] = get_event_indicator_and_time(
        covid_df
    )
    control_pool_df["index_date"] = sample_control_index_dates(
        covid_df, control_pool_df, sample_age=False
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


def tested_vs_untested(
    covid_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    wave: Optional[str],
) -> pd.DataFrame:
    covid_df = covid_df[
        subset_fns["tested"](covid_df) & (waves_fns[wave](covid_df["index_date"]))
    ]
    covid_df.loc[:, ["event_indicator", "event_time"]] = get_event_indicator_and_time(
        covid_df
    )
    control_pool_df = control_pool_df[
        (~control_pool_df["date_first_tested"].isna())
        & (waves_fns[wave](control_pool_df["date_first_tested"]))
    ].copy()
    control_pool_df["index_date"] = sample_control_index_dates(
        covid_df, control_pool_df, sample_age=False
    )
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    np.random.seed(cfg.seed)

    df = pd.read_csv(
        cfg.input_csv,
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

    # split to COVID group and control pool
    covid_df = df.loc[
        (~df["index_date"].isna()) & (df["age_at_index"] >= cfg.min_age_at_index)
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
    elif cfg.experiment.control_group_design == "tested_vs_untested":
        output = tested_vs_untested(
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
    save_path = f"./data/b_dates_set/{cfg.experiment.endpoint}_"
    if cfg.experiment.wave is not None:
        save_path += f"{cfg.experiment.wave}_"
    save_path += f"{cfg.experiment.control_group_design}"
    if cfg.experiment.subset is not None:
        save_path += f"_{cfg.experiment.subset}"
    save_path += ".csv"

    output.to_csv(save_path, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
