import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict


def get_latest_available_value_from_main_dataset(
    row: pd.Series, target: str = "index_date"
) -> pd.Series:
    dates = [
        row["visit_date-0.0"],
        row["visit_date-1.0"],
        row["visit_date-2.0"],
        row["visit_date-3.0"],
    ]
    # Filter dates that are before the target date
    valid_dates = [date for date in dates if date <= row[target]]

    def get_group_key(column_name):
        # Split by '-' and then by '.' to get the necessary parts
        prefix, suffix = column_name.split("-", 1)  # Split only on the first '-'
        suffix_after_dot = (
            suffix.split(".")[1] if "." in suffix else None
        )  # Get part after '.'
        return (prefix, suffix_after_dot)

    # forward fill so that older values can be used if newer ones are missing
    row = row.groupby(lambda col: get_group_key(col)).ffill()

    if len(valid_dates) == 0:
        row = row.filter(regex="-0.")
        row = np.nan
    else:
        visit = np.argmax(valid_dates)
        row = row.filter(regex=f"-{visit}.")
        row.index = row.index.str.replace(f"-{visit}.", ".")

    return row


def special_cases(df: pd.DataFrame):
    # education (minimum reported level should count --> the lower the number, the hihger the qualification)
    education = df.filter(regex="^education")
    df["education"] = education.min(axis=1)
    df.drop(columns=education.columns, inplace=True)

    # blood pressure(use manual measurement if available, or automatic otherwise; get mean of two repeated measurements)
    blood_pressure = df.filter(regex="^blood_pressure")
    blood_pressure_systolic_manual = (
        blood_pressure["blood_pressure_systolic_manual.0"]
        + blood_pressure["blood_pressure_systolic_manual.1"]
    ) / 2
    blood_pressure_diastolic_manual = (
        blood_pressure["blood_pressure_diastolic_manual.0"]
        + blood_pressure["blood_pressure_diastolic_manual.1"]
    ) / 2
    blood_pressure_systolic_automatic = (
        blood_pressure["blood_pressure_systolic_automatic.0"]
        + blood_pressure["blood_pressure_systolic_automatic.1"]
    ) / 2
    blood_pressure_diastolic_automatic = (
        blood_pressure["blood_pressure_diastolic_automatic.0"]
        + blood_pressure["blood_pressure_diastolic_automatic.1"]
    ) / 2
    df["blood_pressure_systolic"] = np.where(
        blood_pressure_systolic_manual.isna(),
        blood_pressure_systolic_automatic,
        blood_pressure_systolic_manual,
    )
    df["blood_pressure_diastolic"] = np.where(
        blood_pressure_diastolic_manual.isna(),
        blood_pressure_diastolic_automatic,
        blood_pressure_diastolic_manual,
    )
    df.drop(columns=blood_pressure.columns, inplace=True)

    # deprivation index (every patient has only one for england, scotland or wales)
    deprivation = df.filter(regex="^deprivation")
    df["deprivation"] = (
        deprivation["deprivation_england"]
        .fillna(deprivation["deprivation_scotland"])
        .fillna(deprivation["deprivation_wales"])
    )
    df.drop(columns=deprivation.columns, inplace=True)
    return df


def get_diagnosis_information(
    diag_codes: pd.DataFrame,
    diag_dates: pd.DataFrame,
    index_dates: pd.Series,
    diagnoses_wildcards: Dict,
    history_years: int = 5,
) -> pd.DataFrame:

    diag_dates = diag_dates.to_numpy()
    index_dates = index_dates.to_numpy()

    df = pd.DataFrame()
    history_start = index_dates - np.timedelta64(history_years * 365, "D")
    mask_before_index = diag_dates <= index_dates[:, np.newaxis]
    mask_in_history = mask_before_index & (diag_dates > history_start[:, np.newaxis])
    df["num_diagnoses"] = mask_in_history.sum(axis=1)
    diag_dates[~mask_before_index] = np.datetime64("1700-01-01")
    df["days_since_last_diagnosis"] = (
        index_dates.astype("datetime64[D]")
        - diag_dates.max(axis=1).astype("datetime64[D]")
    ).astype(int)

    # TODO: Also store diagnoses and their dates

    # diag_codes = np.where(pd.isnull(diag_codes), "", diag_codes).astype(str)
    for diag, diag_code in diagnoses_wildcards.items():
        tqdm.pandas(desc=f"Map wildcard {diag_code}")
        mapped = diag_codes.progress_apply(
            lambda row: row.str.match(diag_code),
            axis=1,
        )
        df[diag] = np.any(mapped & mask_before_index, axis=1)
    return df


def merge_covariates_ukbiobank(df: pd.DataFrame, **kwargs) -> pd.DataFrame:

    # read from main dataset
    path_main_dataset = kwargs.pop("path_main_dataset")
    path_inpatient_diagnoses = kwargs.pop("path_inpatient_diagnoses")
    features_dict = kwargs.pop("features_dict")
    diagnoses_wildcards = kwargs.pop("diagnoses_wildcards")
    set_to_missing = kwargs.pop("set_to_missing")
    one_hot_encode = kwargs.pop("one_hot_encode")

    main_dataset = pd.read_csv(
        path_main_dataset,
        parse_dates=["53-0.0", "53-1.0", "53-2.0", "53-3.0"],
    )
    df = main_dataset  # TODO: replace with the merged df

    # rename the main dataset features
    for feat in features_dict:
        k, v = list(feat.items())[0]
        df.columns = df.columns.str.replace(
            r"^" + str(k) + "-", str(v) + "-", regex=True
        )
    # 53 is the visit date
    df.columns = df.columns.str.replace(r"^53-", "visit_date-", regex=True)
    # remove those that do not match any given feature (still starting with a number after the renaming)
    df = df.filter(regex="^[^0-9]")
    # set missing codes to NaN
    df = df.replace(set_to_missing, np.nan)

    def get_prefix(column_name):
        return column_name.split("-")[0]

    grouped_columns = df.columns.to_series().groupby(get_prefix).agg(list)
    unique_columns = []
    repeated_columns = []
    for _, group_list in grouped_columns.items():
        if len(group_list) > 1:
            repeated_columns.extend(group_list)
        else:
            unique_columns.extend(group_list)

    # df.merge(main_dataset, left_on="patient_id", right_on="eid", how="outer")
    # if df.empty:
    #     raise RuntimeError(
    #         "It seems like the CSV with input dates does not match the UK Biobank main dataset."
    #     )

    df_only_unique = df[unique_columns]
    df_only_unique.columns = df_only_unique.columns.str.replace(
        r"-(.*)", "", regex=True
    )
    df_only_repeated = df[repeated_columns]

    tqdm.pandas(desc="Merge latest available covariates")
    df_latest_available = df_only_repeated.progress_apply(
        get_latest_available_value_from_main_dataset,
        args=("visit_date-0.0",),  # TODO: Change to index date!
        axis=1,
    )

    df = pd.concat([df_only_unique, df_latest_available], axis=1)
    # address special cases like education and blood pressure
    df = special_cases(df)
    # remove .0 suffixes
    df.columns = df.columns.str.replace(".0", "")
    # TODO: Check if any other suffixes are still there??

    # one-hot encode selected features
    # df[one_hot_encode] = df[one_hot_encode].astype(str)
    df = pd.get_dummies(df, dummy_na=True, columns=one_hot_encode)

    inpatient_diagnoses = pd.read_csv(
        path_inpatient_diagnoses,
        parse_dates=[
            col
            for col in pd.read_csv(path_inpatient_diagnoses, nrows=0).columns
            if col.startswith("41280") or col.startswith("41281")
        ],
    )
    diag_codes = inpatient_diagnoses.filter(regex="^41270|41271").astype(str)
    diag_dates = inpatient_diagnoses.filter(regex="^41280|41281")

    date_range = pd.date_range(start="2017-01-01", end="2022-04-30")
    random_dates = pd.Series(
        np.random.choice(date_range, size=len(diag_codes), replace=True)
    )  # TODO: replace with the real index dates!
    df_diagnoses = get_diagnosis_information(
        diag_codes=diag_codes,
        diag_dates=diag_dates,
        index_dates=random_dates,
        diagnoses_wildcards=diagnoses_wildcards,
    )

    df = pd.concat([df, df_diagnoses], axis=1)
    # TODO: remove patients without any diagnosis?
    df = df[df["num_diagnoses"] > 0]
    df = df.rename(columns={"eid": "patient_id"})
    return df
