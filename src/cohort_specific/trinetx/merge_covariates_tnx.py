import pandas as pd
import pyspark
import trinetx as tnx  # automatically installed on the LUCID platform

from typing import List

"""The merging of all relevant covariates, the diagnoses and prescriptions
specifically for a TriNetX cohort. The data processing is hard-coded for this
dataset and cannot be applied to any other cohort.
"""


def find_code_presence(index_date_df: pyspark.sql.DataFrame, 
                       db: str,
                       diseases: List[str], 
                       codes: List[str], 
                       code_systems: List[str], 
                       tables: List[str],
                       history_years: int = 5) -> pd.DataFrame:
    index_parent = pd.DataFrame({
    'code': codes,
    'code_system': code_systems,
    'feature': diseases,
    })
    index_children = tnx.find_children(
                        database=db, 
                        code_list=index_parent
                    )
    
    present_absent_table = tnx.find_presence(
                database=db, 
                index_date=index_date_df,
                code_list=index_children, 
                tables=tables, 
                function='boolean',
                index_days_start=-(history_years*365),
                index_days_end=0,
                allow_subset=True
            )
    
    present_absent_table = present_absent_table.toPandas()
    present_absent_table = present_absent_table.set_index("patient_id").astype(bool)
    return present_absent_table

def compute_code_metadata(index_date_df: pyspark.sql.DataFrame, 
                    db: str, 
                    code_systems: List[str],
                    tables: str,
                    suffix: str,
                    history_years: int = 5) -> pd.DataFrame:
    """
    Get number of codes and days since last code for given code system (e.g., diagnoses).
    """
    code_systems_query = f"code_system='{code_systems[0]}'"
    for code_system in code_systems[1:]:
        code_systems_query += f" or code_system='{code_system}'"
    vocab = spark.sql(
        f"select distinct code, code_system, 'all' as feature from {db}.standardized_terminology where {code_systems_query};"
    ).toPandas()

    # counts
    counts_table = tnx.find_presence(
                database=db, 
                index_date=index_date_df,
                code_list=vocab, 
                tables=tables, 
                function='count',
                index_days_start=-(history_years*365),
                index_days_end=0,
                allow_subset=True
            )
    counts_table = counts_table.toPandas().set_index("patient_id").rename(columns={"all": f"num_{suffix}"})

    # days since last code
    relative_table = tnx.find_presence(
                database=db, 
                index_date=index_date_df,
                code_list=vocab, 
                tables=tables, 
                function='relative',
                index_days_start=-(history_years*365),
                index_days_end=-1,
                allow_subset=True
            )
    relative_table = relative_table.toPandas().set_index("patient_id").rename(columns={"all": f"days_since_last_{suffix}"})
    relative_table = -relative_table.dropna().astype(int)
    return counts_table.merge(relative_table, how="left", left_index=True, right_index=True)

def collect_last_lab_results(index_date_df: pyspark.sql.DataFrame, 
                    db: str, 
                    features: List[str], 
                    codes: List[str],
                    history_years: int = 5) -> pd.DataFrame:
    index_parent = pd.DataFrame({
        'code': codes,
        'code_system': len(codes) * ["TNX"], # for simplification, only TNX codes are currently supported
        'feature': features,
        })

    index_children = tnx.find_children(
                        database=db, 
                        code_list=index_parent
                    )
    
    latest_lab_results = tnx.dstats(database=db, 
               tables=["vitals_signs", "lab_result"], 
               code_list=index_children, 
               index_date=index_date_df, 
               index_days_start=-(history_years*365), 
               index_days_end=0, 
               allow_subset=True)
    latest_lab_results = latest_lab_results.toPandas()
    latest_lab_results = latest_lab_results.pivot(index='patient_id', columns='feature', values='last')
    return latest_lab_results
    
def merge_covariates_trinetx(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Merge covariates from TriNetX.
    """
    df.set_index("patient_id", inplace=True)

    global spark
    spark = kwargs.pop("spark", pyspark.SparkContext.getOrCreate())
    global db
    db = kwargs.pop("db")
    features_dict = kwargs.pop("features_dict")
    diagnoses_wildcards = kwargs.pop("diagnoses_wildcards")

    schema = pyspark.sql.types.StructType([
    pyspark.sql.types.StructField("patient_id", pyspark.sql.types.StringType(), True),
    pyspark.sql.types.StructField("index_date", pyspark.sql.types.DateType(), True)])
    index_df = spark.createDataFrame(df.reset_index()[["patient_id", "index_date"]], schema)

    # demographics
    print(f"Collecting demographics...")
    dems = spark.sql(
    f"select patient_id, case when sex = 'M' then 1 when sex = 'F' then 0 else null end as sex, race from {db}.patient"
    ).toPandas()
    dems = dems.set_index("patient_id")
    dems = pd.get_dummies(dems, dummy_na=True, columns=["race"], drop_first=True)
    df = df.merge(dems, how="left", left_index=True, right_index=True)

    # diagnoses
    diseases = []
    codes = []
    code_systems = []
    for k, v in diagnoses_wildcards.items():
        query_result = spark.sql(f"select code from {db}.standardized_terminology where code rlike '{v}' and code_system='ICD-10-CM'").toPandas()['code'].tolist()
        diseases.extend(len(query_result)*[k])
        codes.extend(query_result)
        code_systems.extend(len(query_result)*["ICD-10-CM"])

    # presence of diagnoses in history window 
    print(f"Computing diagnosis presence...")
    diagnoses_presence = find_code_presence(index_date_df=index_df, 
                                  db=db, 
                                  diseases=diseases, 
                                  codes=codes, 
                                  code_systems=code_systems, 
                                  tables=["diagnosis"])
    df = df.merge(diagnoses_presence, how="left", left_index=True, right_index=True)

    # number of diagnoses and days since last diagnosis
    print(f"Computing diagnosis code metadata...")
    diagnoses_metadata = compute_code_metadata(index_date_df=index_df,
                                      db=db,
                                      code_systems=["ICD-10-CM", "ICD-9-CM"], 
                                      tables=["diagnosis"],
                                      suffix="diag")
    
    df = df.merge(diagnoses_metadata, how="left", left_index=True, right_index=True)

    # lab results and vitals signs
    print(f"Collecting latest available lab results...")
    features = []
    codes = []
    for k, v in features_dict.items():
        features.append(k)
        codes.append(v)
    latest_lab_results = collect_last_lab_results(index_date_df=index_df, 
                    db=db, 
                    features=features, 
                    codes=codes)
    df = df.merge(latest_lab_results, how="left", left_index=True, right_index=True)

    # remove patients without any diagnosis
    df = df[df["num_diag"] > 0]
    df.reset_index(inplace=True)
    return df