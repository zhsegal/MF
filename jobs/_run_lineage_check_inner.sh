#!/usr/bin/env bash
set -eu
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "[$(date)] host=$(hostname) env=$CONDA_DEFAULT_ENV"
exec python -u run_lineage_check.py "$@"
