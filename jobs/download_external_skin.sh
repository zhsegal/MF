#!/usr/bin/env bash
# Download the Reynolds 2021 healthy adult skin h5ad (CELLxGENE) — the external healthy
# reference nb28 §11 compares every mappable level against.
#
#   Reynolds et al. Science 2021;371:eaba6500 — "Developmental cell programs are co-opted in
#   inflammatory skin disease". 195,739 healthy adult skin cells (dermis + epidermis),
#   5 donors, all female, all 10x 3' v2. Disease = normal only.
#   collection 73f82ac8-15cc-4fcd-87f8-5683723fce7f / dataset f0f0d7c4-3bec-428e-9539-c99d36548d96
#   ArrayExpress E-MTAB-8142.
#
# Why this one. It is the reference Li 2024 used to annotate the CTCL atlas, so its taxonomy
# lines up with our cell_type_orig, and 288 of the panel's 290 genes are present (missing:
# MMP26 and TMEM2, so ha_catabolic drops to 4 genes and must abstain externally).
# What it does not give us: this deposit's full_clustering has 34 labels and collapses
# monocytes to Mono_mac, so it cannot anchor nb22's Mono_CD14/Mono_CD16 split.
# It is NOT part of the joint
# atlas (docs/DATA_PROVENANCE.md §3, "Li's own reference inputs (ledger only, not cohorts)"),
# so comparing our HC against it is not circular.
#
# What it is NOT: an independent replicate of our HC arm. One lab, one chemistry (3' v2 against
# our mostly 5'), one sex, 5 donors. Every contrast built on it is donor-level and unblocked and
# carries the control-family calibration — see helpers/external_ref.py.
#
# ~1.03 GB. Resumable (curl -C -): re-run after an interruption and it continues.
#
# Usage:
#   ./download_external_skin.sh
set -euo pipefail

URL="https://datasets.cellxgene.cziscience.com/f8b5c4f4-7ece-4314-9f49-fc254bbd5c71.h5ad"
EXPECTED_BYTES=1032810712

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$JOB_DIR")"
OUT_DIR="$PROJ_DIR/data/external_ref"
OUT="$OUT_DIR/reynolds2021_healthy_skin.h5ad"
PROV="$OUT_DIR/reynolds2021_provenance.json"
mkdir -p "$OUT_DIR"

AVAIL_KB="$(df -Pk "$OUT_DIR" | awk 'NR==2 {print $4}')"
if (( AVAIL_KB < 3000000 )); then
    echo "REFUSING: only ${AVAIL_KB} KB free on $OUT_DIR; need ~3 GB headroom" >&2
    exit 1
fi

if [[ -f "$OUT" ]] && [[ "$(stat -c %s "$OUT")" == "$EXPECTED_BYTES" ]]; then
    echo "already complete: $OUT ($EXPECTED_BYTES bytes)"
else
    echo "[$(date)] downloading -> $OUT"
    curl -L -C - --retry 5 --retry-delay 10 -o "$OUT" "$URL"
    GOT="$(stat -c %s "$OUT")"
    if [[ "$GOT" != "$EXPECTED_BYTES" ]]; then
        echo "SIZE MISMATCH: got $GOT, expected $EXPECTED_BYTES — re-run to resume" >&2
        exit 1
    fi
    echo "[$(date)] ok: $OUT ($GOT bytes)"
fi

cat > "$PROV" <<JSON
{
  "name": "reynolds2021",
  "title": "Human healthy adult skin scRNA-seq data",
  "paper": "Reynolds et al., Science 2021;371:eaba6500",
  "doi": "10.1126/science.aba6500",
  "accession": "E-MTAB-8142",
  "source": "CELLxGENE",
  "collection_id": "73f82ac8-15cc-4fcd-87f8-5683723fce7f",
  "dataset_id": "f0f0d7c4-3bec-428e-9539-c99d36548d96",
  "asset_url": "$URL",
  "path": "data/external_ref/reynolds2021_healthy_skin.h5ad",
  "bytes": $EXPECTED_BYTES,
  "n_obs": 195739,
  "n_donors": 5,
  "donors": ["S1", "S2", "S3", "S4", "S5"],
  "assay": "10x 3' v2",
  "sex": ["female"],
  "disease": ["normal"],
  "tissue": ["dermis", "skin epidermis"],
  "fetched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "in_joint_atlas": false,
  "relation_to_atlas": "Li 2024 used this as its healthy-skin annotation reference (DATA_PROVENANCE.md §3, ledger only, not a cohort). Label taxonomy is therefore shared with our cell_type_orig; expression is independent.",
  "known_confounds": ["single cohort/lab", "10x 3' v2 vs our mostly 5'", "all female", "5 donors"],
  "used_by": ["helpers/external_ref.py", "jobs/run_external_degradome_build.py", "notebooks/_archive/20_skin/28_tme_degradome.ipynb §11"]
}
JSON
echo "[$(date)] wrote $PROV"
