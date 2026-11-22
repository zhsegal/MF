#!/usr/bin/env bash
# Inner script: execute nb34 (blood malignancy TCR+CNV) headless on the bsub-allocated node.
# CPU-only (infercnvpy + OLGA); the inferCNV / arm-CNV steps use all allocated cores via
# os.sched_getaffinity.
set -eu
MF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB="$MF_DIR/notebooks/30_blood/32_malignancy_tcr_cnv.ipynb"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

cd "$MF_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"
python -c "import os; print('allocated cores (sched_getaffinity):', len(os.sched_getaffinity(0)))"
python -c "import infercnvpy, olga; print('infercnvpy/olga OK')"

# The pooled diploid reference is built from OneK1K; fail early with a clear message if it is absent.
test -s "$MF_DIR/data/external_ref/onek1k.h5ad" \
    || { echo "missing data/external_ref/onek1k.h5ad — run jobs/download_onek1k.sh first" >&2; exit 2; }
# v3 re-expresses the blood T cells in the full 40,821-gene space, read from the joint object.
test -s "$MF_DIR/data/atlas_joint/joint_annotated.h5ad" \
    || { echo "missing data/atlas_joint/joint_annotated.h5ad — needed for the v3 full-gene CNV input" >&2; exit 2; }

exec jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"
