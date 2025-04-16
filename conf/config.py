from dataclasses import dataclass
from typing import List, Optional

"""
Data classes for typing. 
"""


@dataclass
class General:
    """
    General configuration for the project.
    """

    seed: int
    min_age_at_index: float
    input_csv: str
    dates_set_path: str
    covariates_merged_path: str
    output_path: str


@dataclass
class Fit:
    exclude_columns: List[str]
    undersample_exposure_groups: bool
    subset_condition: Optional[str]
    n_folds_outer: int
    n_folds_inner: int
    n_trials: int
    n_jobs: int
    eval_metric: str
    target_times: List[float]
    max_updates: int


@dataclass
class MockData:
    n: int


@dataclass
class Experiment:
    endpoint: str
    wave: str
    control_group_design: str
    subset: str


@dataclass
class RunConfig:
    general: General
    fit: Fit
    mock_data: MockData
    experiment: Experiment
