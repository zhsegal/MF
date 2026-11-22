#!/usr/bin/env bash
# (arm, donor, gene) tables for the surface-target screen, on a CPU queue.
#
# NO GPU. One pass over the count matrix per 256-gene block, ~2,300 genes, so ~9 passes over a
# 540k-cell CSR. MEMORY: one block is n_cells x 256 float32 (~0.6 GB for skin, ~0.7 GB for blood)
# plus the same again for the log1p copy; 32 GB is ample. WALL: dominated by HDF5 reads, expect
# 20-60 min per compartment.
#
# Usage:
#   ./run_target_screen.sh --tissue skin
#   ./run_target_screen.sh --tissue blood
#   MEM_MB=64000 WALL_H=8 ./run_target_screen.sh --tissue skin
#   AFTER=<jobid> ./run_target_screen.sh --tissue blood    # start when that job is done
set -euo pipefail
prev=""

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_target_screen_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-32000}"
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-6}"

TISSUE="skin"
for i in "$@"; do
    if [[ "$prev" == "--tissue" ]]; then TISSUE="$i"; fi
    prev="$i"
done
JOBNAME="target_screen_${TISSUE}"
LOG="$JOB_DIR/run_target_screen.${TISSUE}.bsub.log"
DEP=()
if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub \
    "${DEP[@]}" \
    -q "$QUEUE" \
    -n "$NCPU" -R "span[hosts=1]" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "$JOBNAME" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted $JOBNAME -> $QUEUE  (${NCPU} cpu, ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
