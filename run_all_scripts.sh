#!/usr/bin/env bash

# all the other scripts should be adapted for a run with a specific cohort and initial csv file should be in data/a_inputs

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# if some experiments should be ignored (e.g., all involing hospitalized patients) - remember to 
#mv conf/experiment/*hospital* conf/experiment/ignore

./00_create_mock_data.sh # remove when running on real data
./01_set_dates.sh
./02_merge_covariates.sh
./03_fit_tmle.sh

#mv  conf/experiment/ignore/* conf/experiment