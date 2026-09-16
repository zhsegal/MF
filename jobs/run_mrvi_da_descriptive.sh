#!/usr/bin/env bash
# Submit MrVI descriptive DA + donor distances (nb32) to a GPU queue.
# Queue is overridable; `bqueues` on this cluster lists only gsla_high_gpu (GPU), gsla-cpu,
# gsla-mem — there is no gsla-long-gpu.
#
# MEMORY: the group limit NYOSEF_gsla_high_gpu caps the whole lab at 700 GB of *reserved*
# memory on gsla_high_gpu and it sits at ~698 GB most of the time, so an oversized reservation
# pends forever ("Resource (mem) limit defined on queue has been reached"). Right-sizing:
# the MrVI input is two identical float64 CSRs of 345M nnz (~11 GB; the .py now aliases X to
# the layer instead of holding both), DA output arrays are ~1 GB, training peaked at 82 GB
# only because of the training dataloaders -> 64 GB is ample and schedules far sooner.
# Usage:
#   ./run_mrvi_da_descriptive.sh --force
#   MEM_MB=100000 WALL_H=12 ./run_mrvi_da_descriptive.sh --force
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_da_descriptive_inner.sh"
QUEUE="${QUEUE:-gsla_high_gpu}"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-12}"
LOG="$JOB_DIR/run_mrvi_da_descriptive.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J mrvi_da_descriptive \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted mrvi_da_descriptive -> $QUEUE  (gpu=1, ${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
