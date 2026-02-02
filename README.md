# Is COVID-19 increasing the risk for neurodegeneration? A causal inference study on three real-world population cohorts
Code accompanying a research article by Jannis Guski, Sofie Theisen Honoré, Guillaume Azarias, Steven Sison, Søren Brunak, and Holger Fröhlich. If you have any questions regarding the code or paper, please feel free to get in touch (jannis.guski@scai.fraunhofer.de).

## Objective
This is a pipeline to estimate the average exposure effects of SARS-CoV-2 infections (operationalized by either a positive test result or a documented diagnosis U07.1) on the risk of receiving a diagnosis of Alzheimer’s Disease (AD, G30.*), Parkinson’s Disease (PD, G20), or unspecified dementia (F03.*) in the years after in a whole cohort or strata of a cohort. Our own implementation of Targeted Maximum Likelihood Estimation (TMLE) from the `PyTMLE` package is used to get doubly robust estimates with a flexible integration of models for nuisance functions. 

## Instructions for mamba / conda
1. Clone the repository and move into the project directory.
2. Create with `mamba env create -f environment.yml`
3. Activate environment with `mamba activate commute-tmle`

## Experiments
The configuration is set up in `hydra` and provides experiments that are a combination of some design choices:

- wave (first wave, second wave, third wave, Omicron wave, all)

- subset (tested positive, hospitalized, all)

- control group design (pre-pandemic control follow-up, equal control follow-up, tested positive vs. tested negative)

## Bash scripts
The bash scripts call `hydra` multiruns for multiple experiments. Change the environment variables like `INPUT_CSV` or `EXPERIMENTS` for custom calls.

`00_create_mock_input.sh`: Creates some random data; only for test or development purposes.

`01_set_dates.sh`: Selects subsets and sets index / censoring dates for `INPUT_CSV` (typically from `.data/a_inputs`) based on the experiments in `./conf/experiment`. Results are stored in `./data/b_dates_set`.

`02_merge_covariates.sh`: Maps the covariates to the outputs of `01_set_dates.sh` and stores the results in `./data/c_covariates_merged`. Either static covariates or the last available value before the index for longitudinal covariates. Needs to be customized to each cohort.

`03_fit_tmle.sh`: Performs a nested cross validation of an initial hazards estimator, then updates the predictons using TMLE from the `PyTMLE` package for the datasets with merged covariates from `./data/c_covariates_merged`.

The experiment-specific outputs of each script (result tables, plots) will be saved in the `multirun` folder.

## Inputs
The input to `01_set_dates.sh` is a cohort-specific csv file with the following fields:
| patient_id | birth_date | date_first_tested_positive | date_first_covid_diagnosis | hospitalized_due_to_covid | date_first_tested | date_first_ad_diagnosis | date_first_pd_diagnosis | date_first_unspecified_dementia_diagnosis | death_date | censoring_global |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| identifier of the patient | date of birth | date, only if applicable |  date, only if applicable | boolean, whether the patient was hospitalized due to her / his first COVID infection; only if applicable | date (positive or negative) only if applicable | date, AD (G30) only if applicable | date, PD (G20.*) only if applicable | date, unspecified dementia (F03.*) only if applicable | date, only if applicable | global cohort censoring date; may be the same for the whole cohort |
