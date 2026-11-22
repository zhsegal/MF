#!/usr/bin/env bash
# scRNA -> CosMx malignancy transfer model for nb28e, on a CPU queue. NO GPU.
#
# MEMORY: reads only the CD4 rows of skin_T_tcr_annotated_v4.h5ad (12 GB on disk) in 40k-row
# chunks, keeping ~1000 panel columns, so the full 42k-gene matrix is never resident. Peak is a
# chunk (~1-2 GB) plus the panel CSR and the dense CosMx gene-gate table (65k × 1000 float).
# 16 GB per slot × 4 is ample.
# WALL: dominated by the h5 read; well under 2 h. 6 h is headroom.
#
# Usage:
#   ./run_cosmx_transfer.sh
#   ./run_cosmx_transfer.sh --max-per-donor 8000
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_cosmx_transfer_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-16000}"   # per slot on this LSF setup -> NCPU x MEM_MB in total
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-6}"
LOG="$JOB_DIR/run_cosmx_transfer.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -n "$NCPU" -R "span[hosts=1]" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J cosmx_transfer \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted cosmx_transfer -> $QUEUE  (${NCPU} cpu, ${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
