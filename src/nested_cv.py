from hazardous import SurvivalBoost
from joblib import Parallel, delayed, cpu_count
import logging
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
import warnings

from .utils.utils import ensure_monotonicity, get_metrics

# Set up logging
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6.*",
    category=FutureWarning,
    module="sklearn",
)


class SB(SurvivalBoost):
    """
    A wrapper class for SurvivalBoost to add a method for predicting the censoring survival function.
    """

    def predict_censoring_survival_function(self, X, times=None):
        if times is None:
            times = self.time_grid_

        predictions_at_all_times = []

        for t in times:
            t = np.full((X.shape[0], 1), fill_value=t)
            X_with_time = np.hstack([t, X])
            predictions_at_t = self.weighted_targets_.ipcw_estimator.censoring_estimator_.predict_proba(
                X_with_time
            )[
                :, 0
            ]
            predictions_at_all_times.append(predictions_at_t)

        predicted_curves = np.array(predictions_at_all_times).T
        return predicted_curves


def objective(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    times: np.ndarray,
    eval_metric: str = "c_index_1",
):
    param_grid = {
        "n_iter": trial.suggest_int("n_iter", 50, 250),
        "learning_rate": trial.suggest_float(
            "learning_rate", low=0.01, high=0.1, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
    }
    clf = SB(**param_grid, show_progressbar=False)

    clf.fit(X_train, y_train, times=times)

    metrics_dict = get_metrics(
        y_test=y_val,
        y_pred=clf.predict_cumulative_incidence(X_val, times=times),
        y_pred_cens=clf.predict_censoring_survival_function(X_val, times=times),
        y_train=y_train,
        times=times,
    )
    return metrics_dict[eval_metric]


def tune_and_predict(
    X_all: pd.DataFrame,
    y_all: pd.DataFrame,
    n_folds_outer: int,
    n_folds_inner: int,
    seed: int,
    out_path: str,
    eval_metric: str,
    n_trials: int,
    n_jobs: int,
):
    logger.info(
        f"Running nested cross validation for SurvivalBoost with hyperparameter tuning based on {eval_metric}"
    )
    times = np.quantile(y_all["duration"], np.linspace(0, 1, num=100))
    times = np.unique(times)
    cif_predictions_1 = np.empty(
        (X_all.shape[0], len(times), len(y_all["event"].unique()) - 1)
    )
    cif_predictions_0 = np.empty(
        (X_all.shape[0], len(times), len(y_all["event"].unique()) - 1)
    )
    surv_predictions_1 = np.empty((X_all.shape[0], len(times)))
    surv_predictions_0 = np.empty((X_all.shape[0], len(times)))
    cens_surv_predictions_1 = np.empty((X_all.shape[0], len(times)))
    cens_surv_predictions_0 = np.empty((X_all.shape[0], len(times)))
    skf_outer = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=seed)
    skf_inner = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=seed)
    indices = list(range(len(X_all)))
    eval_metrics_dict = {}

    # outer loop
    for i, (outer_train_idx, outer_test_idx) in enumerate(
        skf_outer.split(np.asarray(indices), y_all["event"])
    ):
        logger.info(f"Begin hyperparameter optimization for fold {i+1}/{n_folds_outer}")
        # set up timeout to circumvent deadlock
        # it may be better to switch to MySQL or PostgreSQL altogether, but this would make the code application more complicated
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{out_path}/optuna_fold{i+1}.db",
            engine_kwargs={"connect_args": {"timeout": 100}},
        )
        study = optuna.create_study(
            storage=storage,
            direction="minimize" if eval_metric.startswith("ibs") else "maximize",
            study_name=(f"SurvivalBoost fold {i+1}"),
            load_if_exists=True,
        )

        # split for CV
        X_train_outer = X_all.iloc[outer_train_idx, :]
        y_train_outer = y_all.iloc[outer_train_idx, :]
        X_test_outer = X_all.iloc[outer_test_idx, :]
        y_test_outer = y_all.iloc[outer_test_idx, :]

        # inner loop (trials can be optimized)
        def inner_loop(
            n_trials: int,
            sleep_for: float = 0.0,
        ):
            """A wrapper for parallelized hyperparameter optimization in the inner loop."""
            # sleep for a random time to avoid deadlock
            if sleep_for > 0:
                time.sleep(sleep_for)
            shared_study = optuna.load_study(
                study_name=study.study_name,
                storage=f"sqlite:///{out_path}/optuna_fold{i+1}.db",
            )
            indices_inner = list(range(len(X_train_outer)))

            shared_study.optimize(
                lambda trial: np.mean(
                    [
                        objective(
                            trial,
                            X_train_outer.iloc[inner_train_idx],
                            y_train_outer.iloc[inner_train_idx],
                            X_train_outer.iloc[inner_val_idx],
                            y_train_outer.iloc[inner_val_idx],
                            eval_metric=eval_metric,
                            times=times,
                        )
                        for inner_train_idx, inner_val_idx in skf_inner.split(
                            np.asarray(indices_inner),
                            [endpoint for endpoint in y_train_outer["event"]],  # type: ignore
                        )
                    ]
                ),
                n_trials,
                gc_after_trial=True,
            )

        # hyper parameter optimization
        if n_jobs == 1:
            inner_loop(
                n_trials=n_trials,
            )
        else:
            n_workers = n_jobs if n_jobs != -1 else cpu_count()
            n_workers = min(n_workers, n_trials)
            Parallel(n_jobs=n_workers)(
                delayed(inner_loop)(
                    n_trials=n_trials // n_workers,
                    sleep_for=sleep_for,
                )
                for sleep_for in np.random.choice(
                    np.linspace(0, 5, n_workers), n_workers
                )
            )

        # refit and evaluate on outer test set
        best_param = study.best_params
        clf = SB(**best_param, show_progressbar=False)
        clf.fit(X_train_outer, y_train_outer, times=times)

        # make factual predictions and compute metrics
        cens_surv_factual = clf.predict_censoring_survival_function(
            X_test_outer, times=times
        )
        cifs_factual = clf.predict_cumulative_incidence(X_test_outer, times=times)

        eval_metrics_dict[f"Fold {i+1}"] = get_metrics(
            y_test=y_test_outer,
            y_pred=cifs_factual,
            y_pred_cens=cens_surv_factual,
            y_train=y_train_outer,
            times=times,
        )

        # make counterfactual predictions and store
        X_test_outer_1 = X_test_outer.copy()
        X_test_outer_1["exposed"] = True
        X_test_outer_0 = X_test_outer.copy()
        X_test_outer_0["exposed"] = False
        cens_surv_1 = clf.predict_censoring_survival_function(
            X_test_outer_1, times=times
        )
        cens_surv_0 = clf.predict_censoring_survival_function(
            X_test_outer_0, times=times
        )
        cifs_1 = clf.predict_cumulative_incidence(X_test_outer_1, times=times)
        cifs_0 = clf.predict_cumulative_incidence(X_test_outer_1, times=times)

        cens_surv_predictions_1[outer_test_idx] = cens_surv_1
        cens_surv_predictions_0[outer_test_idx] = cens_surv_0
        surv_predictions_1[outer_test_idx] = cifs_1[:, 0, :]
        surv_predictions_0[outer_test_idx] = cifs_0[:, 0, :]
        cif_predictions_1[outer_test_idx] = np.swapaxes(cifs_1[:, 1:, :], 1, 2)
        cif_predictions_0[outer_test_idx] = np.swapaxes(cifs_0[:, 1:, :], 1, 2)

    # ensure monotoniciy (not automatically given in SurvivalBoost)
    return (
        ensure_monotonicity(cif_predictions_1, nonincreasing=False),
        ensure_monotonicity(cif_predictions_0, nonincreasing=False),
        ensure_monotonicity(surv_predictions_1, nonincreasing=True),
        ensure_monotonicity(surv_predictions_0, nonincreasing=True),
        ensure_monotonicity(cens_surv_predictions_1, nonincreasing=True),
        ensure_monotonicity(cens_surv_predictions_0, nonincreasing=True),
        times,
        eval_metrics_dict,
    )
