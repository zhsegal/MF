#!/usr/bin/env bash
# Submit one stage of the malignant-vs-benign CD4 differential-CCC analysis (nb52).
#
#   ./run_ccc52_mvb.sh lr                    # mrvi_env: rebuild the 1,422-interaction resource
#   ./run_ccc52_mvb.sh prep                  # cohort, arms, blood benign sub-clustering, row labels
#   ./run_ccc52_mvb.sh pass                  # one streamed pass over joint_annotated.h5ad
#   ./run_ccc52_mvb.sh fit                   # DESeq2, partner gate, tables, report, bundle
#   ./run_ccc52_mvb.sh bundle                # re-assemble reports/ccc52_mvb/for_analysis/ only
#   AFTER=<jobid> ./run_ccc52_mvb.sh prep    # chain
#
# Knobs: QUEUE, NCPU, MEM_MB (per slot), WALL_H, AFTER.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_ccc52_mvb_inner.sh"; chmod +x "$INNER"
if [[ $# -lt 1 ]]; then echo "usage: run_ccc52_mvb.sh lr|prep|pass|fit" >&2; exit 1; fi
STAGE="$1"; shift

case "$STAGE" in
    lr)   NCPU="${NCPU:-1}"; MEM_MB="${MEM_MB:-8000}";  WALL_H="${WALL_H:-1}" ;;
    prep) NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}" ;;
    pass) NCPU="${NCPU:-8}"; MEM_MB="${MEM_MB:-16000}"; WALL_H="${WALL_H:-4}" ;;
    fit)  NCPU="${NCPU:-8}"; MEM_MB="${MEM_MB:-8000}";  WALL_H="${WALL_H:-4}" ;;
    bundle) NCPU="${NCPU:-1}"; MEM_MB="${MEM_MB:-4000}"; WALL_H="${WALL_H:-1}" ;;
    *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
QUEUE="${QUEUE:-short}"
LOG="$JOB_DIR/run_ccc52_mvb.${STAGE}.bsub.log"
DEP=(); if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub -q "$QUEUE" \
     -n "$NCPU" -R "span[hosts=1]" -R "rusage[mem=${MEM_MB}]" \
     -W "${WALL_H}:00" "${DEP[@]}" \
     -J "ccc52_${STAGE}" -o "$LOG" -e "$LOG" \
     "$INNER" "$STAGE" "$@"

echo "submitted ccc52_${STAGE} -> $QUEUE (${NCPU} cpu x ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
