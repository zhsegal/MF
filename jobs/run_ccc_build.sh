#!/usr/bin/env bash
# Build the skin CCC object for the LIANA analysis (nb34) on a CPU queue.
#
# NO GPU: this job is pure sparse I/O over joint_annotated.h5ad. Asking for
# gsla_high_gpu would queue behind the 700 GB group reservation cap for nothing.
#
# MEMORY: one 50k-row chunk of the source CSR is ~1.3 GB; the two output CSRs over
# the ~1,832 resource genes total ~1.6 GB; the full-gene pseudobulk cube is
# (levels x samples) x 40,821 float64 ~0.5 GB; the 3-donor full-gene equivalence
# test set peaks around 3 GB while it is assembled. 64 GB is generous.
#
# WALL: ~15 min of I/O for the main pass plus the test-set pass and two gzip
# writes -> 4 h is a wide margin.
#
# Usage:
#   ./run_ccc_build.sh --force
#   MEM_MB=96000 WALL_H=8 ./run_ccc_build.sh --force
#   ./run_ccc_build.sh --force --skip-testset     # skips the nb34 equivalence gate
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_ccc_build_inner.sh"
QUEUE="${QUEUE:-short}"   # gsla-cpu is not open to this user; short = 24 h CPU
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-4}"
LOG="$JOB_DIR/run_ccc_build.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J ccc_build \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"

echo "submitted ccc_build -> $QUEUE  (${MEM_MB}MB, ${WALL_H}h)  log: $LOG"
