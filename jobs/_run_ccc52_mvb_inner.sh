#!/usr/bin/env bash
# Inner driver for run_ccc52_mvb.py. The `lr` stage needs liana, which lives in mrvi_env; every
# other stage needs pydeseq2 + scanpy + polars, which live in neural_nmf_env. Both envs are
# activated the same way, so the only thing that differs is the name.
set -eu
STAGE="$1"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
case "$STAGE" in
    lr) mamba activate mrvi_env ;;
    *)  mamba activate neural_nmf_env ;;
esac

N="${LSB_DJOB_NUMPROC:-4}"
export OMP_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" MKL_NUM_THREADS="$N" NUMBA_NUM_THREADS="$N"

cd "$(dirname "${BASH_SOURCE[0]}")"
echo "[$(date)] host=$(hostname) stage=$STAGE threads=$N env=$CONDA_DEFAULT_ENV"
exec python -u run_ccc52_mvb.py "$@"
