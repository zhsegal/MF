#!/usr/bin/env bash
# Per-sample Leiden clustering of all skin T cells for nb26b, on a CPU queue.
#
# NO GPU: scanpy HVG / PCA / Leiden / UMAP on one sample at a time (largest ~56k cells).
#
# MEMORY: obs (540k rows incl. CDR3 strings) ~2 GB; one sample's raw-count CSR at full gene width
# is < 1 GB; the scaled 2k-HVG dense matrix of the largest sample ~0.5 GB. 16 GB is ample.
#
# THREADS: 4 slots, and run_cluster_malig.py caps BLAS/numba at 4 threads. Uncapped, they spawn one
# thread per core and a 250-cell sample takes ~100 s instead of ~2 s.
#
# WALL: measured 1-5 s per sample of <= 1k cells with the cap; the ~20 samples above 10k cells
# dominate. Expect well under 1 h; 6 h is headroom. Resumable: finished samples are cached in
# data/atlas_joint/skin_T_clusters_v1_parts/, so a re-submit only does what is missing.
#
# Usage:
#   ./run_cluster_malig.sh
#   ./run_cluster_malig.sh --force            # recompute every sample
#   MEM_MB=64000 WALL_H=8 ./run_cluster_malig.sh
#   AFTER=<jobid> ./run_cluster_malig.sh --tissue blood   # start when that job is done
set -euo pipefail
prev=""

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_cluster_malig_inner.sh"
QUEUE="${QUEUE:-short}"   # gsla-cpu is not open to this user; short = 24 h CPU
MEM_MB="${MEM_MB:-16000}"   # per slot on some LSF setups -> up to 4 x 16 GB
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-6}"
# --tissue is passed straight through to the python job; it only changes the LSF job
# name and the log file here, so a blood run cannot overwrite the skin log.
TISSUE="skin"
for i in "$@"; do
    if [[ "$prev" == "--tissue" ]]; then TISSUE="$i"; fi
    prev="$i"
done
JOBNAME="cluster_malig_${TISSUE}"
LOG="$JOB_DIR/run_cluster_malig.${TISSUE}.bsub.log"
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
