from hazardous import metrics
import numpy as np

from conf.config import Experiment


def parse_path_for_experiment(parent_dir: str, experiment: Experiment):
    path = f"{parent_dir}/{experiment.endpoint}_"
    if experiment.wave is not None:
        path += f"{experiment.wave}_"
    path += f"{experiment.control_group_design}"
    if experiment.subset is not None:
        path += f"_{experiment.subset}"
    path += ".csv"
    return path


def get_metrics(
    y_test,
    y_pred,
    y_pred_cens,
    y_train,
    times,
):
    y_test_cens = y_test.copy()
    y_test_cens["event"] = (y_test_cens["event"] == 0).astype(int)
    y_train_cens = y_train.copy()
    y_train_cens["event"] = (y_train_cens["event"] == 0).astype(int)
    metrics_dict = {}
    metrics_dict["c_index_0"] = metrics.concordance_index_incidence(
        y_test_cens, 1 - y_pred_cens, y_train=y_train_cens, event_of_interest=1
    ).item()
    metrics_dict["c_index_1"] = metrics.concordance_index_incidence(
        y_test, y_pred[:, 1, :], y_train=y_train, event_of_interest=1
    ).item()
    metrics_dict["c_index_2"] = metrics.concordance_index_incidence(
        y_test, y_pred[:, 2, :], y_train=y_train, event_of_interest=2
    ).item()
    accuracy_in_time, tau = metrics.accuracy_in_time(
        y_test, y_pred[:, 1:, :], time_grid=times
    )
    metrics_dict["accuracy_in_time_any"] = accuracy_in_time.tolist()
    metrics_dict["mean_accuracy_in_time_any"] = np.mean(accuracy_in_time).item()
    metrics_dict["tau"] = tau.tolist()
    metrics_dict["ibs_0"] = metrics.integrated_brier_score_survival(
        y_test=y_test_cens, y_pred=y_pred_cens, y_train=y_train_cens, times=times
    ).item()
    metrics_dict["ibs_1"] = metrics.integrated_brier_score_incidence(
        y_test=y_test,
        y_pred=y_pred[:, 1, :],
        y_train=y_train,
        times=times,
        event_of_interest=1,  # type: ignore
    ).item()
    metrics_dict["ibs_2"] = metrics.integrated_brier_score_incidence(
        y_test=y_test,
        y_pred=y_pred[:, 2, :],
        y_train=y_train,
        times=times,
        event_of_interest=2,  # type: ignore
    ).item()
    metrics_dict["ibs_any"] = metrics.integrated_brier_score_incidence(
        y_test=y_test,
        y_pred=1 - y_pred[:, 0, :],
        y_train=y_train,
        times=times,
        event_of_interest="any",
    ).item()

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
