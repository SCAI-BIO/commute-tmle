from hazardous import metrics
import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv
import sys
from typing import Dict
from unittest.mock import MagicMock

from conf.config import Experiment

# a hack to avoid torch requirement in the environment (it's never really needed, but the import crashes if it's not found)
try:
    import torch

    torch_is_magic_mock = False
except ImportError:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch"].__version__ = "2.7.0"
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.optim"] = MagicMock()
    sys.modules["torch.utils"] = MagicMock()
    sys.modules["torch.utils.data"] = MagicMock()
    torch_is_magic_mock = True

from pycox.evaluation import EvalSurv

# remove the mock to avoid issues with other imports
if torch_is_magic_mock:
    sys.modules.pop("torch")


def parse_path_for_experiment(
    parent_dir: str, experiment: Experiment, file_extension: str = "csv"
):
    path = f"{parent_dir}/{experiment.endpoint}_"
    if experiment.wave is not None:
        path += f"{experiment.wave}_"
    path += f"{experiment.control_group_design}"
    if experiment.subset is not None:
        path += f"_{experiment.subset}"
    path += "." + file_extension
    return path


def get_metrics_for_one_endpoint(
    suffix: str,
    pred: np.ndarray,
    event_test: pd.Series,
    duration_test: pd.Series,
    event_train: pd.Series,
    duration_train: pd.Series,
    times: np.ndarray,
    all_times: np.ndarray,
) -> Dict[str, float]:
    metrics_dict = {}

    cum_haz = -np.log(pred + 1e-10)  # add small constant to avoid log(0)
    risk_scores = np.sum(cum_haz, axis=1)
    survival_train = Surv.from_arrays(event_train, time=duration_train)
    survival_test = Surv.from_arrays(event_test, time=duration_test)

    # Harrel's C from scikit-survival
    metrics_dict[f"c_harrel_{suffix}"] = concordance_index_censored(
        event_indicator=event_test,
        event_time=duration_test,
        estimate=risk_scores,
    )[
        0
    ].item()  # type: ignore

    # Uno's C from scikit-survival
    metrics_dict[f"c_uno_{suffix}"] = concordance_index_ipcw(
        survival_train=survival_train,
        survival_test=survival_test,
        estimate=risk_scores,
        tau=times.max(),
    )[
        0
    ].item()  # type: ignore

    metrics_dict[f"c_antolini_{suffix}"] = EvalSurv(
        pd.DataFrame(pred.T),
        duration_test.values,
        event_test.values,
        censor_surv="km",
    ).concordance_td()

    # AUC(t) from scikit-survival
    auc_t, mean_auc_t = cumulative_dynamic_auc(
        survival_train=survival_train,
        survival_test=survival_test,
        estimate=risk_scores,
        times=times,
    )
    metrics_dict[f"auc_mean_{suffix}"] = mean_auc_t.item()
    metrics_dict[f"auc_t_{suffix}"] = auc_t.tolist()
    metrics_dict[f"times_{suffix}"] = times.tolist()

    # Integrated Brier Score from hazardous
    metrics_dict[f"ibs_{suffix}"] = metrics.integrated_brier_score_survival(
        y_test={"event": event_test, "duration": duration_test},
        y_pred=pred,
        y_train={"event": event_train, "duration": duration_train},
        times=all_times,
    ).item()

    return metrics_dict


def get_metrics(
    y_test,
    y_pred,
    y_pred_cens,
    y_train,
    times,
):
    y_pred = y_pred[y_test["duration"] < max(y_train["duration"]), :]
    y_pred_cens = y_pred_cens[y_test["duration"] < max(y_train["duration"]), :]
    y_test = y_test[y_test["duration"] < max(y_train["duration"])]
    # Uno's C and AUC(t) will only be evaluated for times that are in the range of event times (per event type)
    test_times = times[
        (times > y_test.loc[:, "duration"].min())
        & (times < y_test.loc[:, "duration"].max())
    ]

    metrics_dict = {}
    # for censoring endpoint
    metrics_dict.update(
        get_metrics_for_one_endpoint(
            suffix="0",
            pred=y_pred_cens,
            event_test=y_test["event"] == 0,
            duration_test=y_test["duration"],
            event_train=y_train["event"] == 0,
            duration_train=y_train["duration"],
            times=test_times,
            all_times=times,
        )
    )

    # for event 1 endpoint (NDD diagnosis)
    metrics_dict.update(
        get_metrics_for_one_endpoint(
            suffix="1",
            pred=1 - y_pred[:, 1, :],
            event_test=y_test["event"] == 1,
            duration_test=y_test["duration"],
            event_train=y_train["event"] == 1,
            duration_train=y_train["duration"],
            times=test_times,
            all_times=times,
        )
    )

    # for event 2 endpoint (death)
    metrics_dict.update(
        get_metrics_for_one_endpoint(
            suffix="2",
            pred=1 - y_pred[:, 2, :],
            event_test=y_test["event"] == 2,
            duration_test=y_test["duration"],
            event_train=y_train["event"] == 2,
            duration_train=y_train["duration"],
            times=test_times,
            all_times=times,
        )
    )

    # for overall survival endpoint
    metrics_dict.update(
        get_metrics_for_one_endpoint(
            suffix="all",
            pred=y_pred[:, 0, :],
            event_test=y_test["event"] > 0,
            duration_test=y_test["duration"],
            event_train=y_train["event"] > 0,
            duration_train=y_train["duration"],
            times=test_times,
            all_times=times,
        )
    )

    return metrics_dict


def ensure_monotonicity(arr: np.ndarray, nonincreasing: bool) -> np.ndarray:
    """
    Ensure that the array is monotonously nonincreasing or nondecreasing along axis 1.
    Args:
        arr (np.ndarray): Input array.
        nonincreasing (bool): If True, ensure nonincreasing; if False, ensure nondecreasing.
    Returns:
        np.ndarray: Monotonically nonincreasing array.
    """
    if nonincreasing:
        return np.minimum.accumulate(arr, axis=1)
    else:
        return np.maximum.accumulate(arr, axis=1)


def get_hazards_from_cif(cif: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """
    Convert CIF to hazard function.
    Args:
        cif (np.ndarray): CIF predictions.
        surv (np.ndarray): Survival function predictions.
    Returns:
        np.ndarray: Hazard function.
    """
    # Ensure cif is a 3D array
    if cif.ndim == 2:
        cif = cif[:, np.newaxis, :]
    # lag survival function
    lagged_surv = np.column_stack([np.ones((surv.shape[0], 1)), surv[:, :-1]])
    hazards = np.diff(cif, prepend=0, axis=1) / lagged_surv[..., np.newaxis]
    return hazards
