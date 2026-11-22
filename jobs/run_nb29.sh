#!/usr/bin/env bash
# Submit a headless run of notebooks/20_skin/29_perturbseq_reverse_engineering.ipynb.
# The notebook caches its heavy steps, so a re-run after a fix is cheap.
#   ./run_nb29.sh            # 150 GB, 24 h, queue long
#   MEM_MB=200000 ./run_nb29.sh
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_nb29_inner.sh"
QUEUE="${QUEUE:-long}"; WALL_H="${WALL_H:-24}"
NCPU="${NCPU:-8}"
MEM_MB="${MEM_MB:-20000}"   # per slot; LSF reserves MEM_MB * NCPU total (160 GB at defaults)
LOG="$JOB_DIR/run_nb29.bsub.log"
rm -f "$LOG"; chmod +x "$INNER"
bsub -q "$QUEUE" -n "$NCPU" -R "span[hosts=1] rusage[mem=${MEM_MB}]" -W "${WALL_H}:00" \
    -J nb29 -o "$LOG" -e "$LOG" "$INNER"
