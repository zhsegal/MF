#!/usr/bin/env bash
# Submit the CD4 CTCL atlas SemanticSCVI sweep (train all variants + projection
# benchmark + report) to gsla_high_gpu. Reads jobs/cd4_sweep_config.json + the
# slim input written by 19_semantic_cd4_atlas_sweep.ipynb.
# Usage:
#   ./run_cd4_sweep.sh
#   MEM_MB=192000 WALL_H=96 ./run_cd4_sweep.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_cd4_sweep_inner.sh"
MEM_MB="${MEM_MB:-128000}"
WALL_H="${WALL_H:-72}"
LOG="$JOB_DIR/run_cd4_sweep.bsub.log"
rm -f "$LOG"

bsub \
    -q gsla_high_gpu \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J cd4_sweep \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
