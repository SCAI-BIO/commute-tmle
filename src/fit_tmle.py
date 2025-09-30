from hazardous import SurvivalBoost
import hydra
from hydra.core.config_store import ConfigStore
import json
import logging
import numpy as np
from omegaconf import OmegaConf
import os
import pandas as pd
import pickle
from pytmle import PyTMLE, InitialEstimates
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .nested_cv import tune_and_predict
from .utils.propensity_score_matching import perform_propensity_score_matching
from .utils.plotting import plot_eval_metrics
from .utils.utils import parse_path_for_experiment, get_hazards_from_cif
from .utils.plotting import plot_eval_metrics
from .utils.propensity_score_matching import perform_propensity_score_matching
from conf.config import RunConfig

# Set up logging
logger = logging.getLogger(__name__)

# Set up the config store
cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)


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
    if cfg.fit.subset_condition is not None:
        # for stratification according to given subset condition
        df = df.query(cfg.fit.subset_condition)
        logger.info(
            f"Using subset of {len(df)} patients with condition {cfg.fit.subset_condition}"
        )
    if cfg.fit.control_pool_subsample_factor is not None and (
        sum(df["exposed"] == False)
        > cfg.fit.control_pool_subsample_factor * sum(df["exposed"] == True)
    ):
        # in some settings it may be necessary to subsample the control pool because k-NN matching can be very memory intensive
        exposed_group = df[df["exposed"] == True]
        control_group = df[df["exposed"] == False]
        n_controls = int(cfg.fit.control_pool_subsample_factor * len(exposed_group))
        control_group_subsampled = control_group.sample(
            n=n_controls, random_state=cfg.general.seed
        )
        df = pd.concat([exposed_group, control_group_subsampled], ignore_index=True)
        logger.info(
            f"Subsampled control pool to {n_controls} rows; total dataset size is now {len(df)}"
        )
    if cfg.fit.perform_propensity_score_matching:
        matched_ids = perform_propensity_score_matching(
            df,
            treatment="exposed",
            indx="patient_id",
            caliper=cfg.fit.propensity_score_matching_caliper,
            grid_search=cfg.fit.propensity_score_matching_grid_search,
            exclude=cfg.fit.exclude_columns + ["event_time", "event_indicator"],
            save_plots_to=f"{cfg.general.output_path}/plots_propensity_score_matching",
        )
        df = df[df["patient_id"].isin(matched_ids)]
        logger.info(
            f"Using subset of {len(df)} patients after propensity score matching"
        )
    df = df.drop(columns=cfg.fit.exclude_columns, errors="ignore")

    if not cfg.fit.run_evalues_benchmark:
        # cross fitting of SurvivalBoost model with hyperparameter tuning
        logger.info("Cross-fitting SurvivalBoost model...")
        (
            cif_1,
            cif_0,
            surv_1,
            surv_0,
            cens_surv_1,
            cens_surv_0,
            times,
            eval_metrics_dict,
        ) = tune_and_predict(
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
            skip_tuning=not cfg.fit.tune_hyperparameters,
            given_hyperparameters=OmegaConf.to_container(
                cfg.survivalboost_params, resolve=True
            ),
        )
        # Save eval_metrics_dict as JSON and plots
        os.makedirs(f"{cfg.general.output_path}/cv_eval_metrics", exist_ok=True)
        output_json_path = (
            f"{cfg.general.output_path}/cv_eval_metrics/nested_cv_eval_metrics.json"
        )
        plot_eval_metrics(
            save_path=f"{cfg.general.output_path}/cv_eval_metrics",
            metrics_dict=eval_metrics_dict,
            endpoint_name=cfg.experiment.endpoint,
        )
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
        models = None
    else:
        # perform e-value benchmark; in this special case, SurvivalBoost is fitted within PyTMLE on discretized event times
        assert (
            not cfg.fit.tune_hyperparameters
        ), "Hyperparameter tuning is not supported for e-value computation. Set fit.tune_hyperparameters to False."
        logger.warning(
            "E-value benchmark does not work with the current release version of PyTMLE, but only with the exp/survivalboost branch."
        )
        initial_estimates = None
        quantile_bins = np.quantile(
            df["event_time"], np.linspace(0, 1, num=22), method="lower"
        )
        # discretize event_time into bins like this is also done in the cross-fitting outside of PyTMLE
        df["event_time"] = np.digitize(df["event_time"], quantile_bins, right=True) - 1
        df["event_time"] = df["event_time"].map(
            lambda x: (
                quantile_bins[int(x)]
                if 0 <= x < len(quantile_bins)
                else quantile_bins[-1]
            )
        )
        models = [SurvivalBoost(**cfg.survivalboost_params, show_progressbar=False)]

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
        evalues_benchmark=cfg.fit.run_evalues_benchmark,
    )

    # fit TMLE
    logger.info("Fitting TMLE...")
    # use only classifiers that natively support missing values (https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)
    propensity_score_models = [
        RandomForestClassifier(),
        HistGradientBoostingClassifier(),
    ]
    tmle.fit(
        max_updates=cfg.fit.max_updates,
        propensity_score_models=propensity_score_models,
        models=models,
        cv_folds=cfg.fit.n_folds_outer,
    )

    if cfg.general.pickle_tmle:
        # pickle TMLE object
        logger.info("Saving TMLE object...")
        with open(f"{cfg.general.output_path}/tmle.pkl", "wb") as f:
            pickle.dump(tmle, f)

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
