#!/usr/bin/env bash
# this reads all experiments from ./conf/experiment and transforms it to a comma-separated list
EXPERIMENTS=$(ls ./conf/experiment/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# target times
TARGET_TIMES="[100,200,300,400,500,600,700]"

# for stratification
SUBSET_CONDITION="\"(event_time>30)\""

CONTROL_POOL_SUBSAMPLE_FACTOR=2
PERFORM_PROPENSITY_SCORE_MATCHING=false

# for cross-fitting
N_FOLDS_OUTER=5
N_FOLDS_INNER=3

# for optuna storage
# this is the database connection string for PostgreSQL
REMOVE_EXISTING_DB=false # set to false if you want to keep the existing database
OPTUNA_STORAGE_PORT=5433
OPTUNA_STORAGE_HOST="localhost" # on a cluster, should be the hostname or IP of the node that this script is run on
OPTUNA_STORAGE_PASSWORD="QkSv1SG4sN0FP8d3R3Ju"
OPTUNA_STORAGE="postgresql://commute:${OPTUNA_STORAGE_PASSWORD}@${OPTUNA_STORAGE_HOST}:${OPTUNA_STORAGE_PORT}/optuna"

# hardware requirements
N_JOBS=20
MEM_GB=64
TIMEOUT_MIN=2880
# when running on HPC cluster, you may want to add variable for partition

# activate mamba environment
source ~/.bashrc
mamba activate commute-tmle

# remove existing database if specified
if [ "$REMOVE_EXISTING_DB" = true ]; then
    echo "Removing existing database..."
    pg_ctl -o "-F -p ${OPTUNA_STORAGE_PORT}" -D commute_tmle_db stop
    rm -rf commute_tmle_db
fi
# create database
pg_ctl init -D commute_tmle_db
pg_ctl -o "-F -p ${OPTUNA_STORAGE_PORT}" -D commute_tmle_db -l logfile start
psql -d postgres -p ${OPTUNA_STORAGE_PORT} -tAc "SELECT 1 FROM pg_database WHERE datname = 'optuna';" | grep -q 1
if [ $? -eq 0 ]; then
    echo "Database 'optuna' already exists. Skipping creation."
else
    # change configuration such that connections from other nodes are supported (https://stackoverflow.com/questions/38466190/cant-connect-to-postgresql-on-port-5432)
    sed -i 's/#listen_addresses/listen_addresses/g' commute_tmle_db/postgresql.conf
    sed -i 's/localhost/*/g' commute_tmle_db/postgresql.conf
    echo -e "host\tall\tall\t0.0.0.0/0\tmd5" >> commute_tmle_db/pg_hba.conf
    sed -i "s/max_connections = 100/max_connections = 1000/g" commute_tmle_db/postgresql.conf
    pg_ctl -o "-F -p ${OPTUNA_STORAGE_PORT}" -D commute_tmle_db restart
    psql -d postgres -p ${OPTUNA_STORAGE_PORT} -c "CREATE USER commute WITH PASSWORD '${OPTUNA_STORAGE_PASSWORD}';"
    createdb -p ${OPTUNA_STORAGE_PORT} -O commute optuna

fi

# create studies (must be done before running the script)
for experiment in $(echo ${EXPERIMENTS} | tr ',' ' '); do
    for i in $(seq 1 ${N_FOLDS_OUTER}); do
        study_name="${experiment}_fold_${i}"
        optuna create-study --study-name ${study_name} \
        --storage ${OPTUNA_STORAGE} \
        --skip-if-exists \
        --direction maximize >/dev/null 2>&1
    done
done

# run the script to fit PyTMLE for each given csv input
python3 -m src.fit_tmle --multirun \
 +experiment=${EXPERIMENTS} \
 fit.n_jobs=${N_JOBS} \
 fit.target_times=${TARGET_TIMES} \
 fit.subset_condition=${SUBSET_CONDITION} \
 fit.optuna_storage=${OPTUNA_STORAGE} \
 fit.n_folds_outer=${N_FOLDS_OUTER} \
 fit.n_folds_inner=${N_FOLDS_INNER} \
 fit.perform_propensity_score_matching=${PERFORM_PROPENSITY_SCORE_MATCHING} \
 fit.control_pool_subsample_factor=${CONTROL_POOL_SUBSAMPLE_FACTOR} \
 hydra.launcher.cpus_per_task=${N_JOBS} \
 hydra.launcher.timeout_min=${TIMEOUT_MIN} \
 hydra.launcher.mem_gb=${MEM_GB} \
 hydra/launcher=submitit_local
 #hydra/launcher=submitit_slurm # change for run on HPC cluster

