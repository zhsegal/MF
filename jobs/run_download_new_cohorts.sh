#!/usr/bin/env bash
# Download every cohort in data/_new_cohorts.tsv as an LSF job array (one task per accession).
# Transfer queue has outbound egress; download_geo_cohort.sh self-tests and fails fast.
# ~8.6 GB total across 14 accessions, all resumable (wget -c) -- a re-submit continues.
# Usage:
#   ./run_download_new_cohorts.sh              # all rows
#   ./run_download_new_cohorts.sh 4-6          # manifest rows 4..6 only
#   QUEUE=gsla-cpu ./run_download_new_cohorts.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="$(cd "$JOB_DIR/.." && pwd)"
INNER="$JOB_DIR/_run_download_new_cohorts_inner.sh"
N=$(( $(wc -l < "$NB_DIR/data/_new_cohorts.tsv") - 1 ))
RANGE="${1:-1-$N}"
QUEUE="${QUEUE:-long}"          # transfer/gsla-cpu are not usable by this account; `long` is (see run_arm_cnv.sh)
MEM_MB="${MEM_MB:-8000}"
WALL_H="${WALL_H:-12}"
LOG="$JOB_DIR/run_download_new_cohorts.%I.bsub.log"

bsub \
    -q "$QUEUE" \
    -n 1 \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "geo_dl[${RANGE}]%4" \
    -o "$LOG" -e "$LOG" \
    "$INNER"
