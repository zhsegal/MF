#!/usr/bin/env bash
# Execute a notebook in place on a CPU node, for notebooks too heavy for the login node.
#
# jupyter nbconvert --execute --inplace, with the kernel named in the notebook's metadata, from
# the notebook's own directory (the NB_DIR walk-up needs that). On success the outputs are saved
# into the notebook; on a cell error nothing is written and the traceback is in the log.
#
# THREADS: BLAS/numba are capped at the slot count; uncapped they spawn one thread per core on
# these 256-core nodes and small linear algebra slows ~50x.
#
# Usage:
#   ./run_notebook.sh notebooks/20_skin/28_tme_degradome_mc.ipynb
#   AFTER=<jobid> ./run_notebook.sh <notebook>          # start when that job is done
#   MEM_MB=32000 NCPU=4 WALL_H=8 ./run_notebook.sh <notebook>
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_notebook_inner.sh"
NB="$(realpath "$1")"
[[ -f "$NB" ]] || { echo "no such notebook: $NB"; exit 1; }
NAME="$(basename "$NB" .ipynb)"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-24000}"   # per slot on this LSF setup -> NCPU x MEM_MB in total
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-6}"
LOG="$JOB_DIR/run_notebook_${NAME}.bsub.log"
DEP=()
if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -n "$NCPU" -R "span[hosts=1]" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    "${DEP[@]}" \
    -J "nb_${NAME}" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$NB"

echo "submitted nb_${NAME} -> $QUEUE  (${NCPU} cpu x ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
