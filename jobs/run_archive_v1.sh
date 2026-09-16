#!/usr/bin/env bash
# Freeze the v1 atlas before the v2 rebuild. ~255 GB.
# MUST complete before the first concat_joint: it runs force=True and overwrites in place.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$JOB_DIR/run_archive_v1.bsub.log"; rm -f "$LOG"
bsub -q "${QUEUE:-long}" -n 1 -R "rusage[mem=${MEM_MB:-8000}]" -W "${WALL_H:-24}:00" \
     -J archive_v1 -o "$LOG" -e "$LOG" "$JOB_DIR/_run_archive_v1_inner.sh"
