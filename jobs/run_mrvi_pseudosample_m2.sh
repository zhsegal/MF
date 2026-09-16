#!/usr/bin/env bash
# Submit the Method-2 pseudo-sample MrVI train (+ optional DE) to gsla_high_gpu.
# Usage:
#   ./run_mrvi_pseudosample_m2.sh
#   ./run_mrvi_pseudosample_m2.sh --max-epochs 50 --run-de --force
#   ./run_mrvi_pseudosample_m2.sh --skip-train     # reuse models/mrvi_pseudosample_m2
#   MEM_MB=128000 WALL_H=12 ./run_mrvi_pseudosample_m2.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_pseudosample_m2_inner.sh"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-24}"
LOG="$JOB_DIR/run_mrvi_pseudosample_m2.bsub.log"
rm -f "$LOG"

bsub \
    -q gsla_high_gpu \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J mrvi_psm2 \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
