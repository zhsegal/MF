#!/usr/bin/env bash
# Hotspot + Harreman on the CosMx early-MF layer for nb28e, on a CPU queue, in the `pxlgn` env
# (the only env with harreman 0.1.4 / hotspot; neural_nmf_env is left untouched).
#
# Needs data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_cells_28e.parquet: run nb28e through
# §R first.
# MEMORY: 65k cells × 1000 genes; the pooled local correlation is ~300 × 300 genes over a
# 10-NN graph. Harreman's cell-type-aware test holds per-pair count matrices. 16 GB × 4 slots
# is ample.
# WALL: CPU torch; expect < 2 h. 8 h is headroom.
#
# Usage:
#   ./run_cosmx_harreman.sh
#   AFTER=<jobid> ./run_cosmx_harreman.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_cosmx_harreman_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-16000}"   # per slot -> NCPU x MEM_MB in total
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-8}"
LOG="$JOB_DIR/run_cosmx_harreman.bsub.log"
DEP=()
if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -n "$NCPU" -R "span[hosts=1]" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    "${DEP[@]}" \
    -J cosmx_harreman \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted cosmx_harreman -> $QUEUE  (${NCPU} cpu, ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
