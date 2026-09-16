#!/usr/bin/env bash
# Submit the GSE284075 (Buus 2025 / cohort B5) download + sample-sheet build.
# Uses the data-transfer queue (egress); download.sh self-tests egress and fails fast.
# Usage:
#   ./run_download_buus2025.sh
#   QUEUE=gsla-cpu MEM_MB=16000 WALL_H=12 ./run_download_buus2025.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_download_buus2025_inner.sh"
QUEUE="${QUEUE:-transfer}"        # data-transfer queue has outbound egress
MEM_MB="${MEM_MB:-8000}"
WALL_H="${WALL_H:-12}"
LOG="$JOB_DIR/run_download_buus2025.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -n 2 \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J buus2025_dl \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
