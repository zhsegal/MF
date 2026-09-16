#!/usr/bin/env bash
# Download the OneK1K healthy-PBMC h5ad (CELLxGENE) used as the external diploid reference for
# the blood inferCNV in nb34.
#
#   Yazar et al. Science 2022 — "Single-cell eQTL mapping identifies cell type specific genetic
#   control of autoimmune disease". 1,248,980 PBMC from 981 healthy donors, 10x 3' v2.
#   collection dde06e0f-ab3b-46be-96a2-a8082383c4a1 / dataset 3faad104-2ab8-4434-816d-474d8d2641db
#   CELLxGENE schema 7.1.0 -> raw counts live in X (raw_data_location="X"),
#   var index = Ensembl IDs with symbols in var["feature_name"].
#
# ~4.43 GB. Resumable (curl -C -): re-run after an interruption and it continues.
# Only the download is done here; C.build_healthy_pbmc_ref() subsets it to the cached
# ~12k-cell CD4 reference without ever loading the full object.
#
# Usage:
#   ./download_onek1k.sh
set -euo pipefail

URL="https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad"
EXPECTED_BYTES=4434273970

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(dirname "$JOB_DIR")/data/external_ref"
OUT="$OUT_DIR/onek1k.h5ad"
mkdir -p "$OUT_DIR"

if [[ -f "$OUT" ]] && [[ "$(stat -c %s "$OUT")" == "$EXPECTED_BYTES" ]]; then
    echo "already complete: $OUT ($EXPECTED_BYTES bytes)"
    exit 0
fi

echo "[$(date)] downloading -> $OUT"
curl -L -C - --retry 5 --retry-delay 10 -o "$OUT" "$URL"

GOT="$(stat -c %s "$OUT")"
if [[ "$GOT" != "$EXPECTED_BYTES" ]]; then
    echo "SIZE MISMATCH: got $GOT, expected $EXPECTED_BYTES — re-run to resume" >&2
    exit 1
fi
echo "[$(date)] ok: $OUT ($GOT bytes)"
