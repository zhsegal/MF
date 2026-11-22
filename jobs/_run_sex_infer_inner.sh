#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
exec python -u run_sex_infer.py "$@"
