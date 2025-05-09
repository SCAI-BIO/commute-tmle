# %%
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# %%
path_main_dataset = "~/COMMUTE/data/ukbiobank/ukb678750_main_dataset.csv"
path_inpatient_diagnoses = "~/COMMUTE/data/ukbiobank/ukb678750_inpatient_diagnoses.csv"
path_covid_tests = "~/COMMUTE/data/ukbiobank/covid_tests/covid19_result*.txt"
# %%
df = pd.DataFrame(
    columns=[
        "patient_id",
        "birth_date",
        "date_first_tested_positive",
        "date_first_covid_diagnosis",
        "hospitalized_due_to_covid",
        "date_first_tested",
        "date_first_ad_diagnosis",
        "date_first_pd_diagnosis",
        "date_first_unspecified_dementia_diagnosis",
        "death_date",
        "censoring_global",
    ]
)

# %%
# read patient_id, birth_date and first dates of AD, PD and unspecified dementia from main dataset
main_dataset = pd.read_csv(path_main_dataset)[
    ["eid", "34-0.0", "131036-0.0", "131022-0.0", "130842-0.0"]
]
df["patient_id"] = main_dataset["eid"]
# pick Jan 1 because the exact date of birth is confidential
df["birth_date"] = pd.to_datetime(main_dataset["34-0.0"].astype(str) + "-01-01")
df["date_first_ad_diagnosis"] = pd.to_datetime(main_dataset["131036-0.0"])
df["date_first_pd_diagnosis"] = pd.to_datetime(main_dataset["131022-0.0"])
df["date_first_unspecified_dementia_diagnosis"] = pd.to_datetime(
    main_dataset["130842-0.0"]
)

# %%
# read death date and global censoring date from inpatient diagnoses
inpatient_diagnoses = pd.read_csv(path_inpatient_diagnoses)
df["death_date"] = pd.to_datetime(inpatient_diagnoses["40000-0.0"])
# longest global censoring for England
df["censoring_global"] = pd.to_datetime("2022-10-31")
# set to "2022-08-31" for Scotland
df.loc[
    (
        (inpatient_diagnoses["40022-0.0"] == "SMR")
        | (inpatient_diagnoses["40022-0.1"] == "SMR")
        | (inpatient_diagnoses["40022-0.2"] == "SMR")
    ).values,
    "censoring_global",
] = pd.to_datetime("2022-08-31")
# set to "2022-05-31" for Wales
df.loc[
    (
        (inpatient_diagnoses["40022-0.0"] == "PEDW")
        | (inpatient_diagnoses["40022-0.1"] == "PEDW")
        | (inpatient_diagnoses["40022-0.2"] == "PEDW")
    ).values,
    "censoring_global",
] = pd.to_datetime("2022-05-31")

# %%
# find the date of the first COVID diagnosis (the way this is implemented here is very memory-intensive)
# only look in ICD10 codes
diag_codes = inpatient_diagnoses.filter(regex="^41270").astype(str)
diag_dates = inpatient_diagnoses.filter(regex="^41280")

tqdm.pandas(desc="Mapping COVID diagnosis dates")
mask = diag_codes.progress_apply(
    lambda row: row.str.match("U071"),
    axis=1,
)
first_mask = np.argmax(mask, axis=1)
first_mask[~mask.any(axis=1)] = -1
first_diagnosis = np.array(
    [
        diag_dates.iloc[i, idx] if idx != -1 else np.nan
        for i, idx in enumerate(first_mask)
    ]
)
df["date_first_covid_diagnosis"] = pd.to_datetime(first_diagnosis)


# %%
# read COVID test data
dfs = []
for path in glob.glob(os.path.expanduser(path_covid_tests)):
    dfs.append(pd.read_csv(path, delimiter="\t")[["eid", "specdate", "result"]])
covid_tests_df = pd.concat(dfs)
covid_tests_df["specdate"] = pd.to_datetime(covid_tests_df["specdate"], format="mixed")
covid_tests_df.sort_values(by=["eid", "specdate"], inplace=True)
first_tested = covid_tests_df.groupby("eid").aggregate("first").reset_index()
first_tested_positive = (
    covid_tests_df[covid_tests_df["result"] == 1]
    .groupby("eid")
    .aggregate("first")
    .reset_index()
)
first_tested = df[["patient_id"]].merge(
    first_tested, how="left", left_on="patient_id", right_on="eid"
)
first_tested_positive = df[["patient_id"]].merge(
    first_tested_positive, how="left", left_on="patient_id", right_on="eid"
)
df["date_first_tested"] = first_tested["specdate"]
df["date_first_tested_positive"] = first_tested_positive["specdate"]

# hospitalized_due_to_covid is NaN in ukbiobank because we do not have the information if a patient was hospitalized due to COVID

# %%
# control pool: individuals who never had a reported COVID infection and were 65+ years old on January 1, 2015
inclusion_control_pool = (
    (df["date_first_tested_positive"].isnull())
    & (df["date_first_covid_diagnosis"].isnull())
    & ((pd.to_datetime("2015-01-01") - df["birth_date"]).dt.days / 365 >= 65)
)
# covid group: individuals who had a reported COVID infection at some point and were at least 70 years old then
inclusion_covid_group = (
    ~df["date_first_tested_positive"].isnull()
    | ~df["date_first_covid_diagnosis"].isnull()
) & (
    (
        df[["date_first_tested_positive", "date_first_covid_diagnosis"]].min(axis=1)
        - df["birth_date"]
    ).dt.days
    / 365
    >= 70
)
# exclude diagnoses before March 2020 (probably errors?)
exclusion_early_infection = (df["date_first_tested_positive"] < "2020-03-01") | (
    df["date_first_covid_diagnosis"] < "2020-03-01"
)

df = df[inclusion_control_pool | (inclusion_covid_group & ~exclusion_early_infection)]

# %%
# save the dataframe
df.to_csv("../../../data/a_inputs/ukbiobank.csv", index=False)
