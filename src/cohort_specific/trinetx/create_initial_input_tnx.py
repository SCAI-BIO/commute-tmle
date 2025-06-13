# Databricks notebook source
# COMMAND ---------- [markdown]
# #Create initial input

# COMMAND ---------- [markdown]
# #Setup

# COMMAND ---------- [markdown]
# ##Import Packages

# COMMAND ----------
import pandas as pd
import trinetx as tnx  # automatically installed on the LUCID platform
from pyspark.sql.functions import col, concat, lit, datediff, least, to_timestamp


# COMMAND ---------- [markdown]
# ##Define Database

# COMMAND ---------- [markdown]
# Replace the below command with your dataset database and your time window for analysis

# COMMAND ----------
input_db = "das_fraunhofer_covid19ndd_dwusa_20250515"
output_db = "commute_extracts"
output_table = "initial_input"
censoring_global = "2025-04-30"

# COMMAND ---------- [markdown]
# ## Create the output schema
spark.sql(f"create schema if not exists {output_db}")

# COMMAND ---------- [markdown]
# ## Find all relevant dates

# COMMAND ---------- [markdown]
# Query patients table for birth and death date

# COMMAND ----------
dems = spark.sql(
    f"select patient_id, year_of_birth, month_year_death from {input_db}.patient"
)
dems = dems.withColumn(
    "birth_date",
    to_timestamp(concat(dems.year_of_birth.cast("string"), lit("-01-01"))),
)
dems = dems.withColumn(
    "death_date",
    to_timestamp(
        concat(
            dems.month_year_death.substr(0, 4),
            lit("-"),
            dems.month_year_death.substr(5, 2),
            lit("-01"),
        )
    ),
)
dems = dems.drop("year_of_birth", "month_year_death")
# drop patients without birth year (patients over 90 who would be easily identifiable)
dems = dems.dropna(subset=["birth_date"])

# COMMAND ---------- [markdown]
# Find dates on which patients were first tested positive

# COMMAND ----------
index_parent_tested_positive = pd.DataFrame(
    {
        "code": ["9088", "9089"],
        "code_system": ["TNX", "TNX"],
        "feature": ["date_first_tested_positive", "date_first_tested_positive"],
        "qualifier_text": ["Positive", "Positive"],
    }
)

index_children_tested_positive = tnx.find_children(
    database=input_db, code_list=index_parent_tested_positive
)

date_first_tested_positive = tnx.find_date(
    database=input_db,
    code_list=index_children_tested_positive,
    tables=["lab_result"],
    function="first",
)

# COMMAND ---------- [markdown]
# Find date on which patients were first tested in general

# COMMAND ----------
index_parent_tested = pd.DataFrame(
    {
        "code": ["9088", "9089"],
        "code_system": ["TNX", "TNX"],
        "feature": ["date_first_tested", "date_first_tested"],
    }
)

index_children_tested = tnx.find_children(
    database=input_db, code_list=index_parent_tested
)

date_first_tested = tnx.find_date(
    database=input_db,
    code_list=index_children_tested,
    tables=["lab_result"],
    function="first",
)

# COMMAND ---------- [markdown]
# Find date of first COVID diagnosis

# COMMAND ----------
index_parent_covid_diagnosis = pd.DataFrame(
    {
        "code": ["U07.1", "U07.2", "J12.82", "B34.2"],
        "code_system": ["ICD-10-CM", "ICD-10-CM", "ICD-10-CM", "ICD-10-CM"],
        "feature": [
            "date_first_covid_diagnosis",
            "date_first_covid_diagnosis",
            "date_first_covid_diagnosis",
            "date_first_covid_diagnosis",
        ],
    }
)

index_children_covid_diagnosis = tnx.find_children(
    database=input_db, code_list=index_parent_covid_diagnosis
)

date_first_covid_diagnosis = tnx.find_date(
    database=input_db,
    code_list=index_children_covid_diagnosis,
    tables=["diagnosis"],
    function="first",
)


# COMMAND ---------- [markdown]
# Find date of first AD diagnosis

# COMMAND ----------
index_parent_ad_diagnosis = pd.DataFrame(
    {
        "code": ["G30"],
        "code_system": ["ICD-10-CM"],
        "feature": ["date_first_ad_diagnosis"],
    }
)

index_children_ad_diagnosis = tnx.find_children(
    database=input_db, code_list=index_parent_ad_diagnosis
)

date_first_ad_diagnosis = tnx.find_date(
    database=input_db,
    code_list=index_children_ad_diagnosis,
    tables=["diagnosis"],
    function="first",
)


# COMMAND ---------- [markdown]
# Find date of first PD diagnosis

# COMMAND ----------
index_parent_pd_diagnosis = pd.DataFrame(
    {
        "code": ["G20"],
        "code_system": ["ICD-10-CM"],
        "feature": ["date_first_pd_diagnosis"],
    }
)

index_children_pd_diagnosis = tnx.find_children(
    database=input_db, code_list=index_parent_pd_diagnosis
)
date_first_pd_diagnosis = tnx.find_date(
    database=input_db,
    code_list=index_children_pd_diagnosis,
    tables=["diagnosis"],
    function="first",
)

# COMMAND ---------- [markdown]
# Find date of first unspecified dementia diagnosis

# COMMAND ----------
index_parent_unspecified_dementia_diagnosis = pd.DataFrame(
    {
        "code": ["F03"],
        "code_system": ["ICD-10-CM"],
        "feature": ["date_first_unspecified_dementia_diagnosis"],
    }
)

index_children_unspecified_dementia_diagnosis = tnx.find_children(
    database=input_db, code_list=index_parent_unspecified_dementia_diagnosis
)
date_first_unspecified_dementia_diagnosis = tnx.find_date(
    database=input_db,
    code_list=index_children_unspecified_dementia_diagnosis,
    tables=["diagnosis"],
    function="first",
)

# COMMAND ---------- [markdown]
# ## Merge all the dataframes

# COMMAND ----------
df = (
    dems.join(date_first_tested_positive, on=["patient_id"], how="left")
    .join(date_first_tested, on=["patient_id"], how="left")
    .join(date_first_covid_diagnosis, on=["patient_id"], how="left")
    .join(date_first_ad_diagnosis, on=["patient_id"], how="left")
    .join(date_first_pd_diagnosis, on=["patient_id"], how="left")
    .join(date_first_unspecified_dementia_diagnosis, on=["patient_id"], how="left")
)

# COMMAND ---------- [markdown]
# Cast to correct types and add invariant columns

# COMMAND ----------
df = (
    df.withColumn(
        "date_first_tested_positive",
        to_timestamp(df.date_first_tested_positive),
    )
    .withColumn("date_first_tested", to_timestamp(df.date_first_tested))
    .withColumn(
        "date_first_covid_diagnosis",
        to_timestamp(df.date_first_covid_diagnosis),
    )
    .withColumn("date_first_ad_diagnosis", to_timestamp(df.date_first_ad_diagnosis))
    .withColumn("date_first_pd_diagnosis", to_timestamp(df.date_first_pd_diagnosis))
    .withColumn(
        "date_first_unspecified_dementia_diagnosis",
        to_timestamp(df.date_first_unspecified_dementia_diagnosis),
    )
    .withColumn("censoring_global", to_timestamp(lit(censoring_global)))
    # hospitalized_due_to_covid is all null because admitting_diagnosis is not populated in the diagnosis table for COVID
    .withColumn("hospitalized_due_to_covid", lit(None).cast("boolean"))
)

# COMMAND ---------- [markdown]
# ## Inclusion / exclusion criteria

# COMMAND ----------
# control pool: individuals who never had a reported COVID infection and were 65+ years old on January 1, 2015
inclusion_control_pool = (
    col("date_first_tested_positive").isNull()
    & col("date_first_covid_diagnosis").isNull()
    & ((datediff(lit("2015-01-01"), col("birth_date")) / 365) >= 65)
)

# covid group: individuals who had a reported COVID infection at some point and were at least 70 years old then
first_covid_date = least(
    col("date_first_tested_positive"), col("date_first_covid_diagnosis")
)
inclusion_covid_group = (
    col("date_first_tested_positive").isNotNull()
    | col("date_first_covid_diagnosis").isNotNull()
) & ((datediff(first_covid_date, col("birth_date")) / 365) >= 70)

# exclude diagnoses before March 2020 (probably artifacts?)
exclusion_early_infection = (col("date_first_tested_positive") < lit("2020-03-01")) | (
    col("date_first_covid_diagnosis") < lit("2020-03-01")
)

df = df.filter(
    inclusion_control_pool | (inclusion_covid_group & ~exclusion_early_infection)
)


# COMMAND ---------- [markdown]
# Write initial input to database

# COMMAND ----------
df.write.mode("overwrite").option("overwriteSchema", "True").saveAsTable(
    f"{output_db}.{output_table}"
)
