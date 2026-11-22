#!/usr/bin/env bash
# Inner script: run jobs/run_perturbseq_prep.py on the bsub-allocated node.
set -eu
MF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
cd "$MF_DIR"
echo "[$(date)] host=$(hostname)  python=$(which python)"
test -s "$MF_DIR/data/external_ref/perturbseq_gwcd4i/GWCD4i.DE_stats.h5ad" \
    || { echo "missing GWCD4i.DE_stats.h5ad -- run jobs/download_perturbseq_gwcd4i.sh first" >&2; exit 2; }
exec python "$MF_DIR/jobs/run_perturbseq_prep.py" --condition "${CONDITION:-Stim48hr}"
