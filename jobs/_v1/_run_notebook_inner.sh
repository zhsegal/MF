#!/usr/bin/env bash
set -eu
NB="$1"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

N="${LSB_DJOB_NUMPROC:-4}"
export OMP_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" MKL_NUM_THREADS="$N" NUMBA_NUM_THREADS="$N"

cd "$(dirname "$NB")"
echo "[$(date)] host=$(hostname)  notebook=$NB  threads=$N"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$NB"
echo "[$(date)] done"
