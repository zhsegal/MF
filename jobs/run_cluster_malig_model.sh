#!/usr/bin/env bash
# Step 1 + Step 2 of the cluster-level malignancy call (nb26b), on a CPU queue.
#
# Runs on the clustering job's outputs (run_cluster_malig.sh): cluster-level tables only
# (a few thousand clusters x 42k genes), plus one review sheet per sample with open clusters.
# MEMORY ~4 GB (the dense log-CPM matrix); WALL a few minutes plus ~1 s per review sheet.
#
# Usage:
#   ./run_cluster_malig_model.sh                                  # auto pick, MAX_ERR 2%
#   ./run_cluster_malig_model.sh --program program+vskew --max-err 0.01
#   AFTER=<jobid> ./run_cluster_malig_model.sh                    # start when that job is done
set -euo pipefail
prev=""

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_cluster_malig_model_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-8000}"    # per slot
NCPU="${NCPU:-2}"
WALL_H="${WALL_H:-2}"
# --tissue is passed straight through to the python job; it only changes the LSF job
# name and the log file here, so a blood run cannot overwrite the skin log.
TISSUE="skin"
for i in "$@"; do
    if [[ "$prev" == "--tissue" ]]; then TISSUE="$i"; fi
    prev="$i"
done
JOBNAME="cluster_malig_model_${TISSUE}"
LOG="$JOB_DIR/run_cluster_malig_model.${TISSUE}.bsub.log"
DEP=()
if [[ -n "${AFTER:-}" ]]; then DEP=(-w "done(${AFTER})"); fi
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -n "$NCPU" -R "span[hosts=1]" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    "${DEP[@]}" \
    -J "$JOBNAME" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted $JOBNAME -> $QUEUE  (${NCPU} cpu, ${MEM_MB}MB, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
