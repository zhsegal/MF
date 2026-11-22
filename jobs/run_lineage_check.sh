#!/usr/bin/env bash
# Verify the lineage migration and report what it changed. obs-only, a couple of minutes.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_lineage_check_inner.sh"; chmod +x "$INNER"
QUEUE="${QUEUE:-short}"; NCPU="${NCPU:-4}"; MEM_MB="${MEM_MB:-24000}"; WALL_H="${WALL_H:-1}"
LOG="$JOB_DIR/run_lineage_check.bsub.log"; rm -f "$LOG"
bsub -q "$QUEUE" -n "$NCPU" -R "span[hosts=1]" -R "rusage[mem=${MEM_MB}]" -W "${WALL_H}:00" \
     -J lineage_check -o "$LOG" -e "$LOG" "$INNER" "$@"
echo "submitted lineage_check -> $QUEUE  log: $LOG"
