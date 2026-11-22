#!/usr/bin/env bash
# Inner script: run jobs/run_pseq30.py on the bsub-allocated node.
set -eu
MF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
N="${LSB_DJOB_NUMPROC:-4}"
export OMP_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" MKL_NUM_THREADS="$N" NUMBA_NUM_THREADS="$N"
cd "$MF_DIR"
echo "[$(date)] host=$(hostname)  python=$(which python)  threads=$N"
for f in data/atlas_joint/pseq29_cd4_cells_v1.h5ad data/atlas_joint/pseq29_cells_v1.parquet \
         data/external_ref/perturbseq_gwcd4i/GWCD4i_DE_Stim48hr.h5ad; do
    test -s "$MF_DIR/$f" || { echo "missing $f -- run nb29 / jobs/run_perturbseq_prep.sh first" >&2; exit 2; }
done
exec python "$MF_DIR/jobs/run_pseq30.py" --n-cpus "$N"
