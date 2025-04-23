#!/usr/bin/env bash
# this reads all experiments from ./conf/experiment and transforms it to a comma-separated list
EXPERIMENTS=$(ls ./conf/experiment/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# target times
TARGET_TIMES="[100,200,300,400,500,600,700]"

# for stratification
SUBSET_CONDITION=null # e.g., "age_at_index<80"

# for optuna storage
# this is the database connection string for PostgreSQL
OPTUNA_STORAGE_PORT = 5433
OPTUNA_STORAGE_HOST = "localhost" # change for execution on a cluster
OPTUNA_STORAGE = "postgresql://commute@${OPTUNA_STORAGE_HOST}:${OPTUNA_STORAGE_HOST}/optuna"


# hardware requirements
N_JOBS=6
MEM_GB=32
TIMEOUT_MIN=1440
# when running on HPC cluster, you may want to add variable for partition

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# create database
pg_ctl init -D commute_tmle_db
pg_ctl -o "-F -p ${OPTUNA_STORAGE_PORT}" -D commute_tmle_db -l logfile start
psql -d postgres -p ${OPTUNA_STORAGE_PORT} -tAc "SELECT 1 FROM pg_database WHERE datname = 'optuna';" | grep -q 1
if [ $? -eq 0 ]; then
    echo "Database 'optuna' already exists. Skipping creation."
else
    createuser -p ${OPTUNA_STORAGE_PORT} --superuser commute
    createdb -p ${OPTUNA_STORAGE_PORT} -O commute optuna
fi


# run the script to fit PyTMLE for each given csv input
python3 -m src.fit_tmle --multirun \
 +experiment=${EXPERIMENTS} \
 fit.n_jobs=${N_JOBS} \
 fit.target_times=${TARGET_TIMES} \
 fit.subset_condition=${SUBSET_CONDITION} \
 fit.optuna_storage=${OPTUNA_STORAGE} \
 hydra.launcher.cpus_per_task=${N_JOBS} \
 hydra.launcher.timeout_min=${TIMEOUT_MIN} \
 hydra.launcher.mem_gb=${MEM_GB} \
 hydra/launcher=submitit_local
# hydra/launcher=submitit_slurm # change for run on HPC cluster

