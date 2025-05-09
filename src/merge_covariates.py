import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .utils.utils import parse_path_for_experiment
from .cohort_specific import COHORT_SPECIFIC_COV_MERGE_FUNCTIONS
from conf.config import RunConfig

# Set up the config store
cs = ConfigStore.instance()
cs.store(name="pipeline_config", node=RunConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: RunConfig):
    input_path = parse_path_for_experiment(cfg.general.dates_set_path, cfg.experiment)
    df = pd.read_csv(
        input_path,
        parse_dates=["index_date"],
    )
    # direct to the cohort-specific merge function implemented in the cohort_specific folder
    merge_covariates_fn = COHORT_SPECIFIC_COV_MERGE_FUNCTIONS[cfg.cohort.name]
    df_merged = merge_covariates_fn(
        df, **OmegaConf.to_container(cfg.cohort, resolve=True)
    )
    df_merged = df_merged.set_index("patient_id")

    # save df_merged
    save_path = parse_path_for_experiment(
        cfg.general.covariates_merged_path, cfg.experiment
    )
    df_merged.to_csv(save_path, float_format="%.2f")


if __name__ == "__main__":
    main()
