#!/usr/bin/env bash
# Re-annotate the non-malignant T/NK cells. One stage per bsub; chain with AFTER=<jobid>.
#
#   ./run_nonmal_T_reannot.sh prep
#   AFTER=<id> ./run_nonmal_T_reannot.sh passA
#   AFTER=<id> ./run_nonmal_T_reannot.sh passB
#   AFTER=<id> ./run_nonmal_T_reannot.sh embed      # GPU
#   AFTER=<id> ./run_nonmal_T_reannot.sh label
#   AFTER=<id> ./run_nonmal_T_reannot.sh tables
#
# Knobs: QUEUE, NCPU, MEM_MB (per slot), WALL_H, AFTER.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_nonmal_T_reannot_inner.sh"; chmod +x "$INNER"
if [[ $# -lt 1 ]]; then echo "usage: run_nonmal_T_reannot.sh prep|passA|passB|embed|label|tables" >&2; exit 1; fi
STAGE="$1"; shift

GPU=()
case "$STAGE" in
    prep)   NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}"; Q=short ;;
    passA)  NCPU="${NCPU:-8}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}"; Q=short ;;
    passB)  NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-32000}"; WALL_H="${WALL_H:-4}"; Q=short ;;
    embed)  NCPU="${NCPU:-8}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-8}"; Q=gsla_high_gpu
            GPU=(-gpu "num=1:j_exclusive=yes") ;;
    label)  NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-24000}"; WALL_H="${WALL_H:-4}"; Q=short ;;
    tables) NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-24000}"; WALL_H="${WALL_H:-2}"; Q=short ;;
    *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
QUEUE="${QUEUE:-$Q}"
LOG="$JOB_DIR/run_nonmal_T_reannot.${STAGE}.bsub.log"
DEP=(); if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub -q "$QUEUE" -n "$NCPU" -R "span[hosts=1]" -R "rusage[mem=${MEM_MB}]" \
     -W "${WALL_H}:00" "${DEP[@]}" "${GPU[@]}" \
     -J "nmT_${STAGE}" -o "$LOG" -e "$LOG" \
     "$INNER" "$STAGE" "$@"
echo "submitted nmT_${STAGE} -> $QUEUE (${NCPU} cpu x ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
