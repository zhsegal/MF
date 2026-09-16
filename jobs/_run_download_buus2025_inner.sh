#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="$(cd "$JOB_DIR/.." && pwd)"
COHORT="$NB_DIR/data/B5_buus2025_blood_skin"

echo "[$(date)] host=$(hostname)  cohort=$COHORT"

# download.sh is pure wget/curl/tar (no python). build_samples_tsv.py needs pandas.
cd "$COHORT"
bash download.sh

echo "[$(date)] download done -> building sample sheet"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
python build_samples_tsv.py

echo "[$(date)] DONE. Inspect meta/samples.tsv + the kept/dropped audit above."
