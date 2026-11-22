#!/usr/bin/env bash
# Build the skin degradome object for nb28 on a CPU queue.
#
# NO GPU: pure sparse I/O over joint_annotated.h5ad. gsla_high_gpu would queue behind the
# group reservation cap for nothing.
#
# MEMORY: one 50k-row chunk of the source CSR is ~1.3 GB; the two output CSRs over the ~290
# panel genes are small (~0.4 GB); the full-gene pseudobulk cube is the real cost --
# (28 levels x 123 samples) x 42,347 float64, ~1.2 GB, and the 3-donor full-gene equivalence
# test set peaks around 5 GB while it is assembled.
#
# v2 is 945k cells against v1's 720k and ~290 panel columns against 156, so the v1 envelope of
# 64 GB / 4 h is raised to 96 GB / 6 h.
#
# WALL: ~25-35 min for the main pass, plus the test-set pass and two gzip writes.
#
# Usage:
#   ./run_degradome_build.sh --force
#   MEM_MB=128000 WALL_H=8 ./run_degradome_build.sh --force
#   ./run_degradome_build.sh --force --skip-testset   # nb28 §0 will then fail its gate
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_degradome_build_inner.sh"
QUEUE="${QUEUE:-short}"   # gsla-cpu is not open to this user; short = 24 h CPU
MEM_MB="${MEM_MB:-96000}"
WALL_H="${WALL_H:-6}"
LOG="$JOB_DIR/run_degradome_build.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J degradome_build \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted degradome_build -> $QUEUE  (${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
