import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd


def generate_random_date(start: str, end: str, n: int, missing_prob: float = 0.0):
    """Generate a random date between start and end."""
    dates = pd.to_datetime(
        pd.Series(pd.Series(pd.date_range(start, end)).sample(n=n, replace=True).values)
    )
    mask = np.random.rand(n) < missing_prob
    dates[mask] = pd.NaT
    return dates


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    np.random.seed(cfg.seed)

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
    df.loc[
        df["date_first_tested_positive"] < df["date_first_tested"], "date_first_tested"
    ] = df["date_first_tested_positive"]
    df["date_first_tested"] = df.date_first_tested.combine_first(
        df.date_first_tested_positive
    )

    # event dates are independent random dates between 2017-01-01 and 2023-12-31
    # and should each be missing with a probability of 95%
    df["date_first_ad_diagnosis"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.95
    )
    df["date_first_pd_diagnosis"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.95
    )
    df["date_first_unspecified_dementia_diagnosis"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.95
    )
    df["death_date"] = generate_random_date(
        "2017-01-01", "2023-12-31", n=n, missing_prob=0.95
    )

    # censoring_global is the 31 December 2022 for all patients
    df["censoring_global"] = pd.to_datetime("2022-12-31")

    df.to_csv(cfg.input_csv, index=False)


if __name__ == "__main__":
    main()
