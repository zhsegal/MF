#!/usr/bin/env bash
set -eu
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="$(cd "$JOB_DIR/.." && pwd)"
MANIFEST="$NB_DIR/data/_new_cohorts.tsv"

echo "[$(date)] host=$(hostname)"

# LSF job-array index -> manifest row (1-based over data rows)
IDX="${LSB_JOBINDEX:-${1:-0}}"
if [ "$IDX" -lt 1 ]; then echo "need LSB_JOBINDEX or an index arg"; exit 2; fi

ROW=$(sed -n "$((IDX + 1))p" "$MANIFEST")
LABEL=$(echo "$ROW" | cut -f1)
GSE=$(echo "$ROW"   | cut -f2)
[ -n "$LABEL" ] || { echo "no manifest row $IDX"; exit 2; }

echo "[$(date)] idx=$IDX label=$LABEL gse=$GSE"
bash "$NB_DIR/data/download_geo_cohort.sh" "$LABEL" "$GSE"
echo "[$(date)] DONE $LABEL"
