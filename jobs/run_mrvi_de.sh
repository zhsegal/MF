#!/usr/bin/env bash
# Submit MrVI DE job to gsla_high_gpu LSF queue.
# Usage:
#   ./run_mrvi_de.sh                       # defaults
#   ./run_mrvi_de.sh --mc-samples 100      # extra args -> run_mrvi_de.py
#   MEM_MB=128000 WALL_H=12 ./run_mrvi_de.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_de_inner.sh"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-24}"
LOG="$JOB_DIR/run_mrvi_de.bsub.log"
rm -f "$LOG"

bsub \
    -q gsla_high_gpu \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J mrvi_de \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
