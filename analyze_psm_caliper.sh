#!/usr/bin/env bash
# this reads all experiments from ./conf/experiment and transforms it to a comma-separated list
EXPERIMENTS=$(ls ./conf/experiment/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# target times
TARGET_TIMES="[700]"

# for stratification
SUBSET_CONDITION="\"(event_time>30)\""

CONTROL_POOL_SUBSAMPLE_FACTOR=3

# hardware requirements
N_JOBS=20
MEM_GB=64
TIMEOUT_MIN=2880
# when running on HPC cluster, you may want to add variable for partition

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# run the script to fit PyTMLE for each given csv input
python3 -m src.analyze_psm_caliper --multirun \
 +experiment=${EXPERIMENTS} \
 fit.target_times=${TARGET_TIMES} \
 fit.subset_condition=${SUBSET_CONDITION} \
 fit.control_pool_subsample_factor=${CONTROL_POOL_SUBSAMPLE_FACTOR} \
 hydra.launcher.cpus_per_task=${N_JOBS} \
 hydra.launcher.timeout_min=${TIMEOUT_MIN} \
 hydra.launcher.mem_gb=${MEM_GB} \
 hydra/launcher=submitit_slurm # change for run on HPC cluster
