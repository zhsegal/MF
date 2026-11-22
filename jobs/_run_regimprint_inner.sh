#!/usr/bin/env bash
# Inner script for jobs/run_regimprint.sh. $1 = stage; rest = stage arguments.
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MF_DIR="$(cd "$JOB_DIR/.." && pwd)"
PKG="${RI_PKG:-$MF_DIR/docs/regimprint_v0.2}"
STAGE="$1"; shift

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
N="${LSB_DJOB_NUMPROC:-4}"
export OMP_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" MKL_NUM_THREADS="$N" NUMBA_NUM_THREADS="$N"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1

cd "$PKG"
echo "[$(date)] host=$(hostname) stage=$STAGE index=${LSB_JOBINDEX:-0} args=$*"
case "$STAGE" in
  prep)
    exec python -u "$JOB_DIR/run_regimprint_prep.py" "$@" ;;
  verify)                                   # $1 = output dir
    V="$1"; mkdir -p "$V"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -c "import jax; print('jax devices:', jax.devices())"
    pytest -q tests -m "not slow" 2>&1 | tee "$V/pytest_fast.txt"
    pytest -q tests -m slow 2>&1 | tee "$V/pytest_slow.txt"
    echo "VERIFY_OK" ;;
  time)                                     # $1 = out file; rest = time_one_fit.py arguments
    O="$1"; shift
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -u scripts/time_one_fit.py "$@" 2>&1 | grep --line-buffered -v -E "ConvergenceWarning|cd_fast" | tee "$O" ;;
  sim)                                      # $1 = planted|mixture, $2 = reps, $3 = out file
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -u scripts/sim_interaction_checks.py "$1" --reps "$2" 2>&1 | tee "$3" ;;
  tune|final|report)
    [[ "$STAGE" == report ]] || nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    exec python -u -m regimprint.run "$STAGE" "$@" ;;
  b1|b2)                                    # array: fold = LSB_JOBINDEX - 1 (or FOLD env for a single job)
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    F="${FOLD:-$(( ${LSB_JOBINDEX:-1} - 1 ))}"
    exec python -u -m regimprint.run "$STAGE" --fold "$F" "$@" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
