#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
# mrvi_env, not neural_nmf_env: neural_nmf_env has no liana.
mamba activate mrvi_env

cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"
python -c "import liana, anndata, scanpy; print('liana', liana.__version__, '| anndata', anndata.__version__, '| scanpy', scanpy.__version__)"

exec python -u run_ccc_build.py "$@"
