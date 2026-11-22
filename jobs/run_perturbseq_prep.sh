#!/usr/bin/env bash
# Submit jobs/run_perturbseq_prep.py: compact the 15.6 GB GWCD4i DE h5ad to one condition
# (default Stim48hr) for notebooks/_archive/20_skin/29_perturbseq_reverse_engineering.ipynb.
# Dense float64 layers are read in row blocks; peak RAM ~40 GB, ~15 min. CPU queue.
# Usage:
#   ./run_perturbseq_prep.sh
#   CONDITION=Rest MEM_MB=60000 ./run_perturbseq_prep.sh
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_perturbseq_prep_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-16000}"   # per slot; LSF reserves MEM_MB * 4 = 64 GB total
WALL_H="${WALL_H:-4}"
CONDITION="${CONDITION:-Stim48hr}"
LOG="$JOB_DIR/run_perturbseq_prep.${CONDITION}.bsub.log"
rm -f "$LOG"
chmod +x "$INNER"
bsub -q "$QUEUE" -n 4 -R "span[hosts=1] rusage[mem=${MEM_MB}]" -W "${WALL_H}:00" \
    -J "pseq_prep_${CONDITION}" -o "$LOG" -e "$LOG" \
    env CONDITION="$CONDITION" "$INNER"
echo "submitted pseq_prep_${CONDITION} -> $QUEUE (${MEM_MB} MB); log: $LOG"
