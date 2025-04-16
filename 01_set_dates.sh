#!/usr/bin/env bash
# change path to actual input csv
INPUT_CSV="./data/a_inputs/mock_input.csv"
# this reads all experiments from ./conf/experiment and transforms it to a comma-separated list
EXPERIMENTS=$(ls ./conf/experiment/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# run the script to set dates for each experiment
python3 -m src.set_dates --multirun +experiment=${EXPERIMENTS} general.input_csv=${INPUT_CSV}