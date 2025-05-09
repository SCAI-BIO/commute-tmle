import hydra
from hydra.core.config_store import ConfigStore
import json
import logging
import numpy as np
import os
import pandas as pd
from pytmle import PyTMLE, InitialEstimates
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .nested_cv import tune_and_predict
from .utils.utils import parse_path_for_experiment, get_hazards_from_cif
from conf.config import RunConfig

# Set up logging
logger = logging.getLogger(__name__)

# Set up the config store
cs = ConfigStore.instance()
cs.store(name="pipeline_config", node=RunConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: RunConfig):
    # parse file path according to given experiment
    csv_path = parse_path_for_experiment(
        cfg.general.covariates_merged_path, cfg.experiment
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # load data, drop unnecessary columns and undersample exposure groups if requested
    logger.info("Loading data...")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=cfg.fit.exclude_columns)
    if cfg.fit.subset_condition is not None:
        # for stratification according to given subset condition
        df = df.query(cfg.fit.subset_condition)
        logger.info(
            f"Using subset of {len(df)} patients with condition {cfg.fit.subset_condition}"
        )
    if cfg.fit.undersample_exposure_groups:
        lowest_freq = df["exposed"].value_counts().min()
        df = (
            df.groupby("exposed")
            .apply(lambda x: x.sample(lowest_freq), include_groups=False)
            .reset_index()
            .drop(columns=["level_1"])
        )
        logger.info(
            f"Using subset of {len(df)} patients with undersampled exposure groups"
        )

    # cross fitting of SurvivalBoost model with hyperparameter tuning
    logger.info("Cross-fitting SurvivalBoost model with hyperparameter tuning...")
    cif_1, cif_0, surv_1, surv_0, cens_surv_1, cens_surv_0, times, eval_metrics_dict = (
        tune_and_predict(
            X_all=df.drop(columns=["event_indicator", "event_time"]),
            y_all=df[["event_indicator", "event_time"]].rename(
                columns={"event_indicator": "event", "event_time": "duration"}
            ),
            n_folds_outer=cfg.fit.n_folds_outer,
            n_folds_inner=cfg.fit.n_folds_inner,
            seed=cfg.general.seed,
            out_path=cfg.general.output_path,
            eval_metric=cfg.fit.eval_metric,
            n_trials=cfg.fit.n_trials,
            n_jobs=cfg.fit.n_jobs,
            optuna_storage=cfg.fit.optuna_storage,
            experiment_prefix=csv_path.split("/")[-1].split(".")[0],
        )
    )
    # Save eval_metrics_dict as JSON
    output_json_path = f"{cfg.general.output_path}/nested_cv_eval_metrics.json"
    with open(output_json_path, "w") as json_file:
        json.dump(eval_metrics_dict, json_file, indent=4)

    # transform CIF to hazards
    haz_1 = get_hazards_from_cif(cif_1, surv_1)
    haz_0 = get_hazards_from_cif(cif_0, surv_0)

    # set initial estimates
    initial_estimates = {
        0: InitialEstimates(
            times=times,
            g_star_obs=1 - df["exposed"].astype(int).values,
            propensity_scores=None,  # will be estimated by PyTMLE
            hazards=haz_0,
            event_free_survival_function=surv_0,
            censoring_survival_function=cens_surv_0,
        ),
        1: InitialEstimates(
            times=times,
            g_star_obs=df["exposed"].astype(int).values,
            propensity_scores=None,  # will be estimated by PyTMLE
            hazards=haz_1,
            event_free_survival_function=surv_1,
            censoring_survival_function=cens_surv_1,
        ),
    }

    # map the event times to the time grid used by SurvivalBoost
    time_grid_indices = (
        np.searchsorted(
            np.append(times, [np.max(times) + 1]), df["event_time"], side="right"
        )
        - 1
    )
    df["event_time"] = times[time_grid_indices]

    # initialize PyTMLE
    target_times = [t for t in cfg.fit.target_times if t <= df["event_time"].max()]
    tmle = PyTMLE(
        df,
        col_event_times="event_time",
        col_event_indicator="event_indicator",
        col_group="exposed",
        target_times=target_times,
        g_comp=True,
        verbose=3,
        initial_estimates=initial_estimates,  # pass initial estimates directly to the second TMLE stage
    )

    # fit TMLE
    logger.info(
        "Fitting TMLE with pre-computed initial estimates from SurvivalBoost..."
    )
    # use only classifiers that natively support missing values (https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)
    propensity_score_models = [
        RandomForestClassifier(),
        HistGradientBoostingClassifier(),
    ]
    tmle.fit(
        max_updates=cfg.fit.max_updates, propensity_score_models=propensity_score_models
    )

    # save TMLE results
    logger.info("Saving TMLE results and diagnostics...")
    for pred_type in ["risks", "rr", "rd"]:
        est_df = tmle.predict(type=pred_type)
        est_df.to_csv(
            f"{cfg.general.output_path}/{pred_type}_estimates.csv", index=False
        )
        est_g_comp_df = tmle.predict(type=pred_type, g_comp=True)
        est_g_comp_df.to_csv(
            f"{cfg.general.output_path}/{pred_type}_estimates_g_computation.csv",
            index=False,
        )
        tmle.plot(
            save_path=f"{cfg.general.output_path}/{pred_type}_estimates.svg",
            g_comp=False,
            type=pred_type,
            color_1="#c00000",
            color_0="#699aaf",
        )
        if pred_type != "risks":
            tmle.plot_evalue_contours(
                save_dir_path=f"{cfg.general.output_path}/plots_{pred_type}_evalue_contours",
                type=pred_type,
            )

    tmle.plot_nuisance_weights(
        save_dir_path=f"{cfg.general.output_path}/plots_nuisance_weights",
        color_1="#c00000",
        color_0="#699aaf",
    )


if __name__ == "__main__":
    main()
