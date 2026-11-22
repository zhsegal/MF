#!/usr/bin/env bash
# Build the external healthy-skin (Reynolds 2021) degradome object for nb28 §11, on CPU.
#
# NO GPU: sparse I/O plus one scoring pass.
#
# MEMORY: the source is 195,739 x 27,867. Row blocks are streamed, but the symbol-collapse
# operator, both output CSRs over the ~288 panel columns and the per-cell score columns are
# resident; ~24 GB is comfortable and leaves room for the gzip write.
#
# WALL: ~10-20 min.
#
# Prerequisite: ./download_external_skin.sh
#
# Usage:
#   ./run_external_degradome_build.sh
#   MEM_MB=48000 ./run_external_degradome_build.sh --force
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_external_degradome_build_inner.sh"
SRC="$(dirname "$JOB_DIR")/data/external_ref/reynolds2021_healthy_skin.h5ad"
QUEUE="${QUEUE:-short}"
MEM_MB="${MEM_MB:-24000}"
WALL_H="${WALL_H:-2}"
LOG="$JOB_DIR/run_external_degradome_build.bsub.log"

if [[ ! -f "$SRC" ]]; then
    echo "missing $SRC -- run ./download_external_skin.sh first" >&2
    exit 1
fi
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J ext_degradome_build \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted ext_degradome_build -> $QUEUE  (${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
