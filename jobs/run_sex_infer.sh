#!/usr/bin/env bash
# Per-sample expression sex call for the clinical metadata layer, on CPU.
#
# NO GPU: one streaming pass over layers/raw_counts of joint_mrvi_input.h5ad (34 GB).
#
# MEMORY: indptr (2.2M) plus one 50k-row block of the CSR; ~16 GB is generous.
#
# WALL: ~15-40 min, I/O bound.
#
# Usage:
#   ./run_sex_infer.sh
#   MEM_MB=32000 ./run_sex_infer.sh --chunk 100000
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_sex_infer_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-16000}"
WALL_H="${WALL_H:-2}"
LOG="$JOB_DIR/run_sex_infer.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J sex_infer \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted sex_infer -> $QUEUE  (${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
