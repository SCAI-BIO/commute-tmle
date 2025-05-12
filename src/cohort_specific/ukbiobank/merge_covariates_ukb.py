import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List


"""The merging of all relevant covariates, the diagnoses and prescriptions
specifically for the UK Biobank. The data processing is hard-coded for this
dataset and cannot be applied to any other cohort.
"""


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
    row = row.drop(
        [target, "visit_date-0.0", "visit_date-1.0", "visit_date-2.0", "visit_date-3.0"]
    )

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
        # if there is no valid assessment date before the index,
        # create an empty row with column names corresponding to the others
        row = pd.Series(
            [np.nan] * len(row.filter(regex="-0.")),
            index=[r.replace("-0.", ".") for r in row.index if "-0." in r],
        )
    else:
        # filter only the values for the last visit before the index
        visit = np.argmax(valid_dates)
        row = row.filter(regex=f"-{visit}.")
        row.index = row.index.str.replace(f"-{visit}.", ".")
    return row


def special_cases(df: pd.DataFrame):
    """Some variables need special processing steps."""
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


def reformat_icd_codes(row: pd.Series) -> List:
    row_reformated = row.dropna().tolist()

    def reformat(code):
        # insert '.' after third position (if length>3)
        if len(code) > 3:
            code = code[:3] + "." + code[3:]
        # trim A, D, S and X (encounter and filler symbols)
        code = code.rstrip("A").rstrip("D").rstrip("S").strip("X")
        return code

    row_reformated = [reformat(r) for r in row_reformated]
    return row_reformated


def get_diagnosis_information(
    diag_codes: pd.DataFrame,
    diag_dates: pd.DataFrame,
    index_dates: pd.Series,
    diagnoses_wildcards: Dict,
    history_years: int = 5,
) -> pd.DataFrame:

    diag_dates_array = diag_dates.to_numpy()
    index_dates_array = index_dates.to_numpy()

    df = pd.DataFrame(index=diag_codes.index)
    history_start = index_dates_array - np.timedelta64(history_years * 365, "D")
    mask_before_index = diag_dates_array <= index_dates_array[:, np.newaxis]
    mask_in_history = mask_before_index & (
        diag_dates_array > history_start[:, np.newaxis]
    )
    # count diagnoses in the history period
    df["num_diagnoses"] = mask_in_history.sum(axis=1)
    diag_dates_array[~mask_before_index] = np.datetime64("1700-01-01")
    # count days since last diagnosis before index
    df["days_since_last_diagnosis"] = (
        index_dates_array.astype("datetime64[D]")
        - diag_dates_array.max(axis=1).astype("datetime64[D]")
    ).astype(int)

    # get boolean variables indicating if specified diagnoses are present before index or not
    for diag, diag_code in diagnoses_wildcards.items():
        tqdm.pandas(desc=f"Map wildcard {diag_code}")
        mapped = diag_codes.progress_apply(
            lambda row: row.str.match(diag_code),
            axis=1,
        )
        df[diag] = np.any(mapped & mask_before_index, axis=1)

    # store diagnoses and their dates
    tqdm.pandas(desc="Retrieve diagnoses")
    diag_codes[~mask_in_history] = np.nan
    df["diagnoses"] = diag_codes.progress_apply(reformat_icd_codes, axis=1)
    tqdm.pandas(desc="Retrieve diagnosis dates")
    diag_dates_array[diag_codes.isnull()] = np.datetime64("NaT")
    df["diagnosis_dates"] = pd.DataFrame(diag_dates, index=df.index).progress_apply(
        lambda row: row.dropna().dt.strftime("%Y%m%d").tolist(), axis=1
    )
    return df


def reformat_medication_codes(row: pd.Series, atc_map: Dict) -> List:
    """Translate UKB medication codes to ATC codes using the dictionary,
    and map to fourth level"""
    row_reformated = row.dropna().tolist()
    row_reformated = [
        atc_map[r][:5] if r in atc_map.keys() else "UNK" for r in row_reformated
    ]
    return row_reformated


def get_self_reported_drug_information(
    drug_codes: pd.DataFrame,
    drug_dates: pd.DataFrame,
    index_dates: pd.Series,
    atc_map: Dict,
    history_years: int = 5,
):
    drug_dates_array = np.repeat(
        drug_dates.to_numpy(), drug_codes.shape[1] // drug_dates.shape[1], axis=1
    )
    index_dates_array = index_dates.to_numpy()
    df = pd.DataFrame(index=drug_codes.index)

    history_start = index_dates_array - np.timedelta64(history_years * 365, "D")
    mask_before_index = ~pd.isnull(drug_dates_array) & (
        drug_dates_array <= index_dates_array[:, np.newaxis]
    )
    mask_in_history = mask_before_index & (
        drug_dates_array > history_start[:, np.newaxis]
    )
    drug_codes[~mask_in_history] = np.nan
    drug_dates_array[drug_codes.isna()] = np.datetime64("NaT")
    df["num_drugs"] = (~drug_codes.isnull()).sum(axis=1)
    # store presciptions and their dates
    tqdm.pandas(desc="Retrieve drugs")
    df["drugs"] = drug_codes.progress_apply(
        reformat_medication_codes, args=(atc_map,), axis=1
    )
    tqdm.pandas(desc="Retrieve prescription dates")
    df["prescription_dates"] = pd.DataFrame(
        drug_dates_array, index=df.index
    ).progress_apply(lambda row: row.dropna().dt.strftime("%Y%m%d").tolist(), axis=1)
    return df


def merge_covariates_ukbiobank(df: pd.DataFrame, **kwargs) -> pd.DataFrame:

    # get the necessary variables from kwargs
    path_main_dataset = kwargs.pop("path_main_dataset")
    path_inpatient_diagnoses = kwargs.pop("path_inpatient_diagnoses")
    path_self_reported_drugs = kwargs.pop("path_self_reported_drugs")
    path_atc_map = kwargs.pop("path_atc_map")
    features_dict = kwargs.pop("features_dict")
    diagnoses_wildcards = kwargs.pop("diagnoses_wildcards")
    set_to_missing = kwargs.pop("set_to_missing")
    one_hot_encode = kwargs.pop("one_hot_encode")

    # read main dataset from csv
    main_dataset = pd.read_csv(
        path_main_dataset,
        parse_dates=["53-0.0", "53-1.0", "53-2.0", "53-3.0"],
    )
    # merge with the dates df to keep only the patients needed in the experiment
    df = df.merge(
        main_dataset,
        left_on="patient_id",
        right_on="eid",
        how="left",
    ).sort_values(by="patient_id")
    if df.empty:
        raise RuntimeError(
            "It seems like the CSV with input dates does not match the UK Biobank main dataset."
        )

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
            # repeated measures are present (e.g., lab features)
            repeated_columns.extend(group_list)
        else:
            # only one instance of this variable is present (e.g., biological sex)
            unique_columns.extend(group_list)

    df_only_unique = df[unique_columns]
    df_only_unique.columns = df_only_unique.columns.str.replace(
        r"-(.*)", "", regex=True
    )
    df_only_repeated = df[["index_date"] + repeated_columns]

    # merge latest available covariates before the index
    tqdm.pandas(desc="Merge latest available covariates")
    df_latest_available = df_only_repeated.progress_apply(
        get_latest_available_value_from_main_dataset,
        args=("index_date",),
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
    df.set_index("eid", inplace=True)

    inpatient_diagnoses = pd.read_csv(
        path_inpatient_diagnoses,
        parse_dates=[
            col
            for col in pd.read_csv(path_inpatient_diagnoses, nrows=0).columns
            if col.startswith("41280") or col.startswith("41281")
        ],
    )
    # only keep patients that are also present in df
    inpatient_diagnoses.sort_values(by="eid", inplace=True)
    inpatient_diagnoses.set_index("eid", inplace=True)
    inpatient_diagnoses = inpatient_diagnoses[inpatient_diagnoses.index.isin(df.index)]
    # store ICD-9 and ICD-10 diagnoses and corresponding dates in separate data frames
    diag_codes = inpatient_diagnoses.filter(regex="^41270|41271").astype(str)
    diag_dates = inpatient_diagnoses.filter(regex="^41280|41281")

    # get diagnosis counts, days since last diagnosis,
    # and boolean variables for the presence of selected diagnoses before the index
    df_diagnoses = get_diagnosis_information(
        diag_codes=diag_codes,
        diag_dates=diag_dates,
        index_dates=df["index_date"],
        diagnoses_wildcards=diagnoses_wildcards,
    )

    # read the drugs information
    self_reported_drugs = pd.read_csv(
        path_self_reported_drugs,
        parse_dates=["53-0.0", "53-1.0", "53-2.0", "53-3.0"],
    )
    # get self-reported drug counts
    # only keep patients that are also present in df
    self_reported_drugs.sort_values(by="eid", inplace=True)
    self_reported_drugs.set_index("eid", inplace=True)
    self_reported_drugs = self_reported_drugs[self_reported_drugs.index.isin(df.index)]
    # reat the ATC map and turn it into a dictionary
    atc_map_df = pd.read_csv(path_atc_map)
    atc_map = atc_map_df.set_index("medication_code")["atc"].to_dict()
    df_drugs = get_self_reported_drug_information(
        drug_codes=self_reported_drugs.drop(
            columns=["53-0.0", "53-1.0", "53-2.0", "53-3.0"]
        ),
        drug_dates=self_reported_drugs[["53-0.0", "53-1.0", "53-2.0", "53-3.0"]],
        index_dates=df["index_date"],
        atc_map=atc_map,
    )
    df_diagnoses_drugs = pd.concat([df_diagnoses, df_drugs], axis=1)
    df = df.merge(df_diagnoses_drugs, how="left", left_index=True, right_index=True)
    # remove patients without any diagnosis
    df = df[df["num_diagnoses"] > 0]
    return df
