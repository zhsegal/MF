#!/usr/bin/env bash
# Submit nb37 Parts 0-3 (the CPU prep the GPU fold jobs depend on) to a CPU queue.
#
# Produces: benchmark_results/clonality_transfer/_job_input/{clonality_input.h5ad,
# clonality_semantic_map.pt}, donor_table.csv, and jobs/clonality_folds_config.json.
#
# CPU queue, not gsla_high_gpu: this is a 20 GB backed read + HVG, no GPU work, and the GPU
# queue is congested. Memory is per-slot, so the defaults reserve ~256 GB on one host.
#
# Usage:
#   ./run_clonality_prep.sh
#   SUBMIT_FOLDS=1 ./run_clonality_prep.sh    # also submit the GPU fold array on success
#   NSLOTS=8 MEM_MB=32000 QUEUE=long ./run_clonality_prep.sh
#
# SUBMIT_FOLDS chains rather than using `bsub -w`: the fold count is data-driven (it depends on
# how many donors clear the eligibility gate) and is only known once the config is written.
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_clonality_prep_inner.sh"
QUEUE="${QUEUE:-long}"
NSLOTS="${NSLOTS:-8}"
MEM_MB="${MEM_MB:-32000}"     # per-slot -> ~256 GB total at defaults
WALL_H="${WALL_H:-24}"
LOG="$JOB_DIR/run_clonality_prep.bsub.log"
rm -f "$LOG"
chmod +x "$INNER" "$JOB_DIR/run_clonality_folds.sh"
export SUBMIT_FOLDS="${SUBMIT_FOLDS:-0}"

bsub \
    -q "$QUEUE" \
    -n "$NSLOTS" \
    -R "span[hosts=1] rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J clon_prep \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
echo "submitted clon_prep -> $QUEUE  (-n $NSLOTS, ${MEM_MB}MB/slot)"
echo "log: $LOG"
