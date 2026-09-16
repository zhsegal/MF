#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="$(cd "$JOB_DIR/.." && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
cd "$NB_DIR"
echo "[$(date)] host=$(hostname) building tcr_clones.parquet (force)"
python - <<'PY'
import sys; sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
import atlas_join_helpers as H
p = H.build_tcr_table(Path("."), force=True)
t = pd.read_parquet(p)
print(f"\n== tcr_clones.parquet: {len(t):,} cells, {t['clone_id'].nunique():,} clones")
print(t.groupby("dataset").agg(cells=("cell_id","size"), donors=("donor","nunique"),
                               samples=("sample_id","nunique"),
                               malignant=("is_malignant","sum")).to_string())
PY
echo "[$(date)] DONE"
