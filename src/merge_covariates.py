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
    df_merged.drop(
        columns=["diagnoses", "diagnosis_dates", "drugs", "prescription_dates"],
        errors="ignore",
    ).to_csv(save_path, float_format="%.2f")

    if cfg.general.store_json:
        """This whole code block is not needed for the appplication of PyTMLE.
        However, it can be useful to prepare the generated data for the
        individualized risk models."""
        df_merged["index_date"] = df_merged["index_date"].dt.strftime("%Y%m%d")
        if cfg.cohort.female_coded_as_one:
            df_merged["sex"] = df_merged["sex"].replace({0: "Male", 1: "Female"})
        else:
            df_merged["sex"] = df_merged["sex"].replace({1: "Male", 0: "Female"})
        df_merged["state"] = "UNK"
        endpoint_dict = {0: "", 1: cfg.experiment.endpoint, 2: "death"}
        df_merged["endpoint_labels"] = df_merged["event_indicator"].apply(
            lambda x: [endpoint_dict[x]]
        )

        df_merged["additional_features"] = df_merged.drop(
            columns=[
                "index_date",
                "age_at_index",
                "endpoint_labels",
                "event_indicator",
                "event_time",
                "sex",
                "index_date",
                "diagnoses",
                "diagnosis_dates",
                "drugs",
                "prescription_dates",
                "exposed",
            ],
            errors="ignore",
        ).apply(lambda x: x.to_dict(), axis=1)
        json_output = (
            df_merged.reset_index()[
                [
                    "patient_id",
                    "age_at_index",
                    "sex",
                    "index_date",
                    "diagnoses",
                    "diagnosis_dates",
                    "drugs",
                    "prescription_dates",
                    "event_time",
                    "endpoint_labels",
                    "exposed",
                    "additional_features",
                ]
            ]
            .rename(columns={"event_time": "time_to_endpoint"})
            .to_json(orient="records")
        )
        save_path = parse_path_for_experiment(
            cfg.general.covariates_merged_path, cfg.experiment, file_extension="json"
        )
        with open(save_path, "w") as file:
            file.write(json_output)


if __name__ == "__main__":
    main()
