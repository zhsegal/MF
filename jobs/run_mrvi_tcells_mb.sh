#!/usr/bin/env bash
# Submit MrVI T-cell pseudo-sample train + DE to gsla_high_gpu.
# Usage:
#   ./run_mrvi_tcells_mb.sh
#   ./run_mrvi_tcells_mb.sh --max-epochs 50 --force
#   ./run_mrvi_tcells_mb.sh --skip-train     # reuse models/mrvi_tcells_mb
#   MEM_MB=128000 WALL_H=12 ./run_mrvi_tcells_mb.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_tcells_mb_inner.sh"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-24}"
LOG="$JOB_DIR/run_mrvi_tcells_mb.bsub.log"
rm -f "$LOG"

bsub \
    -q gsla_high_gpu \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J mrvi_tcells_mb \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
