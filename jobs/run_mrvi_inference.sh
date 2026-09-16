#!/usr/bin/env bash
# Submit MrVI inference (donor distances + DA) to gsla_high_gpu.
# Usage:
#   ./run_mrvi_inference.sh                 # defaults
#   ./run_mrvi_inference.sh --force
#   MEM_MB=128000 WALL_H=12 ./run_mrvi_inference.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_inference_inner.sh"
MEM_MB="${MEM_MB:-200000}"
WALL_H="${WALL_H:-24}"
LOG="$JOB_DIR/run_mrvi_inference.bsub.log"
rm -f "$LOG"

bsub \
    -q gsla_high_gpu \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J mrvi_inference \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
