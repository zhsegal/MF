#!/usr/bin/env bash
# resolVI on the CosMx early-MF layer (65,717 cells × 1000 genes) for nb28e, on a GPU queue.
#
# MEMORY: counts are a 65k × 1000 int CSR (~50 MB); the job peaks well under 8 GB. 16 GB is
# headroom and keeps the reservation small on gsla_high_gpu (lab-wide 700 GB cap).
# WALL: 100 epochs over ~130 minibatches is minutes on one GPU; 2 h is headroom.
#
# Usage:
#   ./run_cosmx_resolvi.sh
#   ./run_cosmx_resolvi.sh --force --epochs 200
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_cosmx_resolvi_inner.sh"
QUEUE="${QUEUE:-gsla_high_gpu}"
MEM_MB="${MEM_MB:-16000}"
WALL_H="${WALL_H:-2}"
LOG="$JOB_DIR/run_cosmx_resolvi.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J cosmx_resolvi \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted cosmx_resolvi -> $QUEUE  (gpu=1, ${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
