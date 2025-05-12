#!/usr/bin/env bash
# change to the name of your cohort
COHORT="mock_data"
# this reads all experiments from ./conf/experiment and transforms it to a comma-separated list
EXPERIMENTS=$(ls ./conf/experiment/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# hardware requirements
N_JOBS=1
MEM_GB=32
TIMEOUT_MIN=1440
# when running on HPC cluster, you may want to add variable for partition

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# run the script to set dates for each experiment
python3 -m src.merge_covariates \
    --multirun \
    +experiment=${EXPERIMENTS} \
    +cohort=${COHORT} \
    general.input_csv=${INPUT_CSV} \
    hydra.launcher.cpus_per_task=${N_JOBS} \
    hydra.launcher.timeout_min=${TIMEOUT_MIN} \
    hydra.launcher.mem_gb=${MEM_GB} \
    hydra/launcher=submitit_local
    #hydra/launcher=submitit_slurm # change for run on HPC cluster

