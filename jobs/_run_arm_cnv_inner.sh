#!/usr/bin/env bash
# Inner script: execute 20_skin/23_malignancy_tcr_cnv headless on the bsub-allocated node.
# CPU-only (infercnvpy + OLGA); the arm-CNV step uses all allocated cores via os.sched_getaffinity.
set -eu
MF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB="$MF_DIR/notebooks/20_skin/23_malignancy_tcr_cnv.ipynb"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

cd "$MF_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"
python -c "import os; print('allocated cores (sched_getaffinity):', len(os.sched_getaffinity(0)))"
python -c "import infercnvpy, olga; print('infercnvpy/olga OK')"

# v5 draws the external healthy-skin diploid reference from the Integrated CTCL atlas; fail early
# with a clear message if it is absent (mirrors _run_blood_cnv_inner.sh's OneK1K guard).
test -s "$MF_DIR/data/Integrated_CTCL_skincellatlas_final_portal_tags.h5ad" \
    || { echo "missing data/Integrated_CTCL_skincellatlas_final_portal_tags.h5ad — needed for the v5 external healthy reference" >&2; exit 2; }

exec jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"
