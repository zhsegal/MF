#!/usr/bin/env bash
# Per-cell CITE-seq table for nb45, on a CPU queue.
#
# NO GPU. Row-subset read of the 61,215 buus2025 cells (~9% of the matrix) plus the 128-marker
# obsm. Seconds of compute; the memory is the obsm read (667k x 128 float32) plus the subset.
#
# Usage:
#   ./run_adt_cells.sh
#   ./run_adt_cells.sh --max-cells 5000     # smoke test
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_adt_cells_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-32000}"
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-2}"
LOG="$JOB_DIR/run_adt_cells.bsub.log"
DEP=()
if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub \
    "${DEP[@]}" \
    -q "$QUEUE" \
    -n "$NCPU" -R "span[hosts=1]" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "adt_cells_blood" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted adt_cells_blood -> $QUEUE  (${NCPU} cpu, ${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
