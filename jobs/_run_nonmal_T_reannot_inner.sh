#!/usr/bin/env bash
set -eu
STAGE="$1"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
N="${LSB_DJOB_NUMPROC:-4}"
export OMP_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" MKL_NUM_THREADS="$N" NUMBA_NUM_THREADS="$N"
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "[$(date)] host=$(hostname) stage=$STAGE threads=$N env=$CONDA_DEFAULT_ENV"
if [[ "$STAGE" == "embed" ]]; then
  python -c "import torch;print('cuda:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
fi
exec python -u run_nonmal_T_reannot.py "$@"
