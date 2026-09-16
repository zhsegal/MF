#!/usr/bin/env bash
# Build data/<LABEL>/processed/standardized.h5ad for the 11 new v2 expression cohorts.
# One array task per cohort. D13/D14 read UNFILTERED CellRanger h5 (the full ~737k 10x
# whitelist) and D13/D15 run hashsolo on their multiplexed lanes -> the biggest memory
# users here. Held cohorts are NOT rebuilt: QC is per-cohort and Scrublet is seeded
# (random_state=0), so their v1 caches are already bit-identical to what a rebuild gives.
# Usage:  ./run_build_new_std.sh            |   ./run_build_new_std.sh 9-11
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=$(wc -w <<< "${STD_LABELS:-B6 B7 B8 D7 D8 D9 D10 D11 D13 D14 D15}")
RANGE="${1:-1-$N}"
QUEUE="${QUEUE:-long}"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-12}"
LOG="$JOB_DIR/run_build_new_std${LOG_TAG:+_$LOG_TAG}.%I.bsub.log"
bsub -env "all" -q "$QUEUE" -n 1 -R "rusage[mem=${MEM_MB}]" -W "${WALL_H}:00" \
     -J "newstd${LOG_TAG:+_$LOG_TAG}[${RANGE}]%4" -o "$LOG" -e "$LOG" \
     "$JOB_DIR/_run_build_new_std_inner.sh"
