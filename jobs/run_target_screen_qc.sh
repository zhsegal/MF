#!/usr/bin/env bash
# (arm, donor, gene) tables WITH the contamination diagnostics, on a CPU queue.
#
# NO GPU. One pass over the count matrix per 128-gene block, ~1,300 genes. The v1 job
# (run_target_screen.sh) measured 77 s wall / 3.9 GB for blood; this one holds four arrays per
# block instead of two and adds a per-donor coupling pass, so budget a few minutes and 32 GB.
#
# Usage:
#   ./run_target_screen_qc.sh --tissue blood
#   MEM_MB=64000 WALL_H=8 ./run_target_screen_qc.sh --tissue skin   # 540k cells, ~2,100 genes
#   ./run_target_screen_qc.sh --tissue blood --max-cells 50000   # smoke test
#   AFTER=<jobid> ./run_target_screen_qc.sh --tissue blood
#
# The cluster-circularity control of 42/43 section 2d is the same job with the arms taken from
# the TCR clone:
#   TAG=mal_tcr ./run_target_screen_qc.sh --tissue skin --call mal_tcr --no-cluster \
#       --out-dir data/targets/by_call_mal_tcr
# TAG keeps the two runs of one tissue from sharing a job name and a log file.
set -euo pipefail
prev=""

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_target_screen_qc_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-32000}"
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-4}"

TISSUE="blood"
for i in "$@"; do
    if [[ "$prev" == "--tissue" ]]; then TISSUE="$i"; fi
    prev="$i"
done
TAG="${TAG:-}"            # disambiguates a second run of the same tissue (e.g. the mal_tcr control)
JOBNAME="target_screen_qc_${TISSUE}${TAG:+_$TAG}"
LOG="$JOB_DIR/run_target_screen_qc.${TISSUE}${TAG:+.$TAG}.bsub.log"
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
