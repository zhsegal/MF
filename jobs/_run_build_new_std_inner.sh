#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="$(cd "$JOB_DIR/.." && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

# override with STD_LABELS="D13 D15" so a targeted rebuild does not need an edit
read -r -a LABELS <<< "${STD_LABELS:-B6 B7 B8 D7 D8 D9 D10 D11 D13 D14 D15}"
IDX="${LSB_JOBINDEX:-${1:-0}}"
LABEL="${LABELS[$((IDX-1))]}"
echo "[$(date)] host=$(hostname) idx=$IDX label=$LABEL"

cd "$NB_DIR"
python - "$LABEL" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
import atlas_join_helpers as H
label = sys.argv[1]
p = H.build_standardized(label, Path("."), force=True, run_scrublet=True)
import scanpy as sc
a = sc.read_h5ad(p)
print(f"\n[{label}] shape={a.shape} donors={a.obs['donor'].nunique()} "
      f"samples={a.obs['sample_id'].nunique()}")
print(a.obs[["dataset","donor","sample_id","disease","compartment","tissue"]]
      .drop_duplicates().to_string(index=False))
print("qc:", a.uns["qc"])
PY
echo "[$(date)] DONE $LABEL"
