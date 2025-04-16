#!/usr/bin/env bash
N_ROWS=10000

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# run the script to set dates for each experiment
python3 -m src.create_mock_input mock_data.n=${N_ROWS}