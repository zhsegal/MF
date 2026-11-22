#!/usr/bin/env bash
# Submit one stage of the benign-CD4-arm QC (follow-up to run_ccc52_mvb.sh).
#
#   ./run_ccc52_qc.sh all     # prep -> pass -> tab in one job (the usual call)
#   ./run_ccc52_qc.sh tab     # re-make the tables from qc_accumulators.npz only
#
# Knobs: QUEUE, NCPU, MEM_MB (per slot), WALL_H, AFTER.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_ccc52_qc_inner.sh"; chmod +x "$INNER"
if [[ $# -lt 1 ]]; then echo "usage: run_ccc52_qc.sh prep|pass|tab|all" >&2; exit 1; fi
STAGE="$1"; shift

case "$STAGE" in
    prep) NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}" ;;
    pass) NCPU="${NCPU:-8}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}" ;;
    tab)  NCPU="${NCPU:-2}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-1}" ;;
    all)  NCPU="${NCPU:-8}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}" ;;
    *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
QUEUE="${QUEUE:-short}"
LOG="$JOB_DIR/run_ccc52_qc.${STAGE}.bsub.log"
DEP=(); if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub -q "$QUEUE" \
     -n "$NCPU" -R "span[hosts=1]" -R "rusage[mem=${MEM_MB}]" \
     -W "${WALL_H}:00" "${DEP[@]}" \
     -J "ccc52qc_${STAGE}" -o "$LOG" -e "$LOG" \
     "$INNER" "$STAGE" "$@"

echo "submitted ccc52qc_${STAGE} -> $QUEUE (${NCPU} cpu x ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
