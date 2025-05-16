import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import pandas as pd

from conf.config import RunConfig

# Set up the config store
cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)


def generate_random_date(start: str, end: str, n: int, missing_prob: float = 0.0):
    """Generate n random dates between start and end, missing with a given probability."""
    dates = pd.to_datetime(
        pd.Series(pd.Series(pd.date_range(start, end)).sample(n=n, replace=True).values)
    )
    mask = np.random.rand(n) < missing_prob
    dates[mask] = pd.NaT
    return dates


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: RunConfig):

    np.random.seed(cfg.general.seed)

    # Create a data frame with the following columns:
    n = cfg.mock_data.n
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
    # patient ID is just a number from 1 to n
    df["patient_id"] = range(1, n + 1)

    # all mock patients should be at least 70 years old in 2020
    df["birth_date"] = generate_random_date(
        start="1930-01-01", end="1949-12-31", n=n, missing_prob=0.0
    )

    # date_first_tested_positive is a random date between 2020-01-01 and 2022-12-31
    # and should be missing with a probability of 70%
    df["date_first_tested_positive"] = generate_random_date(
        "2020-01-01", "2022-12-31", n=n, missing_prob=0.7
    )

    # date_first_covid_diagnosis is a random date between 2020-01-01 and 2022-12-31
    # and should be missing with a probability of 70%
    df["date_first_covid_diagnosis"] = generate_random_date(
        "2020-01-01", "2022-12-31", n=n, missing_prob=0.7
    )

    # hospitalized_due_to_covid is a boolean value that is True if the patient was hospitalized due to COVID-19
    mask = np.random.rand(n) < 0.5
    df["hospitalized_due_to_covid"] = df["date_first_covid_diagnosis"].notna() & mask

    # date on that a patient was first tested for COVID-19
    df["date_first_tested"] = generate_random_date(
        "2020-01-01", "2022-12-31", n=n, missing_prob=0.5
    )
    # if first date_first_tested is before date_first_tested_positive, set date_first_tested to date_first_tested_positive
    df.loc[
        df["date_first_tested_positive"] < df["date_first_tested"], "date_first_tested"
    ] = df["date_first_tested_positive"]
    df["date_first_tested"] = df.date_first_tested.combine_first(
        df.date_first_tested_positive
    )

    # event dates are independent random dates between 2017-01-01 and 2023-12-31
    # and should each be missing with a probability of 80%-90%
    df["date_first_ad_diagnosis"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.9
    )
    df["date_first_pd_diagnosis"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.9
    )
    df["date_first_unspecified_dementia_diagnosis"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.9
    )
    df["death_date"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.8
    )

    # censoring_global is the 31 December 2022 for all patients
    df["censoring_global"] = pd.to_datetime("2022-12-31")

    df.to_csv(cfg.general.input_csv, index=False)


if __name__ == "__main__":
    main()
