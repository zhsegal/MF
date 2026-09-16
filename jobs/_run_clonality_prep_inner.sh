#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cd "$JOB_DIR"
echo "[$(date)] host=$(hostname)  pwd=$(pwd)"
echo "python: $(which python)"

python -u run_clonality_prep.py "$@"

# Chain the GPU fold array from inside the prep job: the fold count is data-driven and only
# known once the config exists, so it cannot be resolved at prep-submit time.
if [[ "${SUBMIT_FOLDS:-0}" == "1" ]]; then
    echo "[$(date)] prep ok — submitting the fold array"
    ARRAY=1 "$JOB_DIR/run_clonality_folds.sh"
fi
