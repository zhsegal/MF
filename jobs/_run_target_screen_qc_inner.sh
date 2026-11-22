#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

export TARGET_SCREEN_THREADS="${LSB_DJOB_NUMPROC:-4}"
cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"

exec python -u run_target_screen_qc.py "$@"
