#!/usr/bin/env bash
# this reads all experiments from ./conf/experiment and transforms it to a comma-separated list
EXPERIMENTS=$(ls ./conf/experiment/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# for stratification
SUBSET_CONDITION=null # e.g., "age_at_index<80"

# hardware requirements
N_JOBS=4
MEM_GB=32
TIMEOUT_MIN=1440

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# run the script to fit PyTMLE for each given csv input
python3 -m src.fit_tmle --multirun \
 +experiment=${EXPERIMENTS} \
 fit.n_jobs=${N_JOBS} \
 fit.subset_condition=${SUBSET_CONDITION} \
 hydra.launcher.cpus_per_task=${N_JOBS} \
 hydra.launcher.timeout_min=${TIMEOUT_MIN} \
 hydra.launcher.mem_gb=${MEM_GB} \
 hydra/launcher=submitit_local
# hydra/launcher=submitit_slurm # change for run on HPC cluster

