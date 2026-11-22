#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
# neural_nmf_env is enough here: external_ref imports only degradome_data/degradome_cells,
# neither of which reaches ccc_helpers (and therefore liana).
mamba activate neural_nmf_env

cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"
python -c "import anndata, scanpy, h5py; print('anndata', anndata.__version__, '| scanpy', scanpy.__version__, '| h5py', h5py.__version__)"

exec python -u run_external_degradome_build.py "$@"
