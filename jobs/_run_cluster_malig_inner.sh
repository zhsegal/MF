#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

export CLUSTER_MALIG_THREADS="${LSB_DJOB_NUMPROC:-4}"
cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"
python -c "import anndata, scanpy, igraph, leidenalg; print('anndata', anndata.__version__, '| scanpy', scanpy.__version__)"

exec python -u run_cluster_malig.py "$@"
