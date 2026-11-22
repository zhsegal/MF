#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
set +u; mamba activate pxlgn; set -u   # pxlgn activate.d scripts read unset MKL_* vars

export COSMX_HARREMAN_THREADS="${LSB_DJOB_NUMPROC:-4}"
cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"
python -c "import harreman, anndata; print('harreman', getattr(harreman, '__version__', '?'), '| anndata', anndata.__version__)"

exec python -u run_cosmx_harreman.py "$@"
