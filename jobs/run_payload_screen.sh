#!/usr/bin/env bash
# Payload-selectivity screen (Gate 1-3) for a CD70 TRUCK, skin. CPU only.
#
# Two matrix passes, one 58-gene column block each: the skin T object (0.76B non-zeros) and
# joint_annotated.h5ad (3.46B non-zeros, ~55 GB streamed once). Budget ~1 h wall, ~12 GB.
#
# Usage:
#   ./run_payload_screen.sh                       # the real run
#   ./run_payload_screen.sh --skip-joint          # T/NK arms only, ~5 min (code check)
#   TAG=smoke ./run_payload_screen.sh --skip-joint --max-cells 50000
#   AFTER=<jobid> ./run_payload_screen.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_payload_screen_inner.sh"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-24000}"
NCPU="${NCPU:-4}"
WALL_H="${WALL_H:-6}"

TAG="${TAG:-}"
JOBNAME="payload_screen${TAG:+_$TAG}"
LOG="$JOB_DIR/run_payload_screen${TAG:+.$TAG}.bsub.log"
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

echo "submitted $JOBNAME -> $QUEUE  (${NCPU} cpu, ${MEM_MB}MB/slot, ${WALL_H}h${AFTER:+, after $AFTER})  log: $LOG"
