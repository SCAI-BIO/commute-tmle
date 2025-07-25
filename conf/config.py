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
    store_json: bool
    pickle_tmle: bool


@dataclass
class Fit:
    exclude_columns: List[str]
    perform_propensity_score_matching: bool
    propensity_score_matching_caliper: float
    subset_condition: Optional[str]
    n_folds_outer: int
    n_folds_inner: int
    n_trials: int
    n_jobs: int
    eval_metric: str
    target_times: List[float]
    max_updates: int
    optuna_storage: Optional[str]
    tune_hyperparameters: bool
    run_evalues_benchmark: bool


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
class Cohort:
    name: str
    female_coded_as_one: bool


@dataclass
class SurvivalBoostParams:
    n_iter: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int


@dataclass
class RunConfig:
    general: General
    fit: Fit
    mock_data: MockData
    experiment: Experiment
    cohort: Cohort
    survivalboost_params: SurvivalBoostParams
