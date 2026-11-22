#!/usr/bin/env bash
# Execute nb42 (TME degradome) on a CPU node.
#
# NO GPU: nothing in nb42 trains. The heavy steps are reading skin_degradome.h5ad (720,125 x
# 156) and the 95,233 x 40,821 equivalence test set, 20 sparse matmuls for the shuffle null,
# and six DESeq2 fits on a (level, donor) x 40,821 matrix. 64 GB covers the test set with
# room to spare.
#
# Usage:
#   ./run_nb42.sh
#   MEM_MB=96000 WALL_H=8 ./run_nb42.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_nb42_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-4}"
LOG="$JOB_DIR/run_nb42.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J nb42_degradome \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted nb42_degradome -> $QUEUE  (${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
