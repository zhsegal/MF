#!/usr/bin/env bash
# Per-(arm, donor, gene) histograms of log1p CP10K for the pooled screens (nb46, nb48), on a
# CPU queue.
#
# NO GPU. One pass over the count matrix per 128-gene block, CD4 arms only -- the same read as
# run_target_screen_qc.sh (~2 min, ~3 GB for blood; skin is 2,100 genes over 540k cells, so give
# it MEM_MB=32000 WALL_H=4).
#
# Usage:
#   ./run_target_pool_hist.sh --tissue blood
#   MEM_MB=32000 WALL_H=4 ./run_target_pool_hist.sh --tissue skin
#   AFTER=<jobid> ./run_target_pool_hist.sh --tissue blood
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_target_pool_hist_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-16000}"
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-2}"

TISSUE="blood"
prev=""
for i in "$@"; do
    if [[ "$prev" == "--tissue" ]]; then TISSUE="$i"; fi
    prev="$i"
done
JOBNAME="target_pool_hist_${TISSUE}"
LOG="$JOB_DIR/run_target_pool_hist.${TISSUE}.bsub.log"
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
