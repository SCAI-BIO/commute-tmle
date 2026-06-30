import hydra
from hydra.core.config_store import ConfigStore
import logging
import os
import pandas as pd

from .utils.propensity_score_matching import perform_propensity_score_matching
from .utils.utils import parse_path_for_experiment
from .utils.plotting import plot_aalen_johansen
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

    share_matched = []
    aj_rr_estimates = []
    aj_rd_estimates = []
    psm = None
    for caliper in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        logger.info(f"Performing propensity score matching with caliper: {caliper}")
        matched_ids, psm = perform_propensity_score_matching(
            df,
            treatment="exposed",
            indx="patient_id",
            caliper=caliper,
            grid_search=cfg.fit.propensity_score_matching_grid_search,
            calibrate_propensities=cfg.fit.propensity_score_matching_calibrate,
            exclude=cfg.fit.exclude_columns + ["event_time", "event_indicator"],
            save_plots_to=f"{cfg.general.output_path}/plots_propensity_score_matching_caliper_{caliper}",
            psm=psm,
        )
        df_matched = df[df["patient_id"].isin(matched_ids)]

        share_matched.append(
            {
                "caliper": caliper,
                "share_matched": sum(df_matched["exposed"]) / sum(df["exposed"]),
            }
        )

        max_target_time = max(cfg.fit.target_times)

        # plot Aalen-Johansen curves
        est_aj_dict = plot_aalen_johansen(
            save_path=cfg.general.output_path,
            T=df_matched["event_time"],
            E=df_matched["event_indicator"],
            exposed=df_matched["exposed"],
            target_times=[max_target_time],
        )
        aj_rr_estimates.append(
            {
                "caliper": caliper,
                "Pt Est": est_aj_dict["rr"]["Pt Est"].item(),
                "CI_lower": est_aj_dict["rr"]["CI_lower"].item(),
                "CI_upper": est_aj_dict["rr"]["CI_upper"].item(),
            }
        )
        aj_rd_estimates.append(
            {
                "caliper": caliper,
                "Pt Est": est_aj_dict["rd"]["Pt Est"].item(),
                "CI_lower": est_aj_dict["rd"]["CI_lower"].item(),
                "CI_upper": est_aj_dict["rd"]["CI_upper"].item(),
            }
        )
    share_matched_df = pd.DataFrame(share_matched)
    aj_rr_estimates_df = pd.DataFrame(aj_rr_estimates)
    aj_rd_estimates_df = pd.DataFrame(aj_rd_estimates)

    share_matched_df.to_csv(
        os.path.join(cfg.general.output_path, "share_matched.csv"), index=False
    )
    aj_rr_estimates_df.to_csv(
        os.path.join(cfg.general.output_path, "aj_rr_estimates.csv"), index=False
    )
    aj_rd_estimates_df.to_csv(
        os.path.join(cfg.general.output_path, "aj_rd_estimates.csv"), index=False
    )


if __name__ == "__main__":
    main()
