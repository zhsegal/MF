#!/usr/bin/env bash
# Submit jobs/run_pseq30.py: per-donor signatures + shared design + 30 fixed-hyperparameter fits for
# notebooks/_archive/20_skin/30_perturbseq_per_donor.ipynb. CPU queue; reads the 6.8 GB nb29 CD4 cache and the
# 5 GB Perturb-seq h5ad (one layer). Peak RAM ~40 GB, ~30-60 min.
# Usage:
#   ./run_pseq30.sh
#   QUEUE=long WALL_H=12 ./run_pseq30.sh
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_pseq30_inner.sh"
QUEUE="${QUEUE:-short}"
NCPU="${NCPU:-8}"
MEM_MB="${MEM_MB:-8000}"    # per slot; LSF reserves MEM_MB * NCPU = 64 GB total
WALL_H="${WALL_H:-6}"
LOG="$JOB_DIR/run_pseq30.bsub.log"
rm -f "$LOG"
chmod +x "$INNER"
bsub -q "$QUEUE" -n "$NCPU" -R "span[hosts=1] rusage[mem=${MEM_MB}]" -W "${WALL_H}:00" \
    -J pseq30 -o "$LOG" -e "$LOG" "$INNER"
echo "submitted pseq30 -> $QUEUE (${NCPU} cpu x ${MEM_MB} MB, ${WALL_H} h); log: $LOG"
