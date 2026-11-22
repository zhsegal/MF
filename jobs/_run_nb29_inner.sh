#!/usr/bin/env bash
# Inner script: execute notebooks/20_skin/29_perturbseq_reverse_engineering.ipynb headlessly on
# the bsub-allocated node (the heavy cells -- §2 cell 8/9 and §4C -- read the 12 GB skin-T object).
set -eu
MF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
NB="$MF_DIR/notebooks/20_skin/29_perturbseq_reverse_engineering.ipynb"
cd "$MF_DIR/notebooks/20_skin"
echo "[$(date)] host=$(hostname) python=$(which python)"
test -s "$MF_DIR/data/external_ref/perturbseq_gwcd4i/GWCD4i_DE_Stim48hr.h5ad" \
    || { echo "missing GWCD4i_DE_Stim48hr.h5ad -- run jobs/run_perturbseq_prep.sh first" >&2; exit 2; }
export MPLBACKEND=Agg
exec python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=neural_nmf_env \
    --log-level=INFO "$NB"
