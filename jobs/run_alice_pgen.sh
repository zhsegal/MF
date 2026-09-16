#!/usr/bin/env bash
# Submit the OLGA Pgen / ALICE sweep for nb21 (CPU-only; OLGA does not use the GPU).
# Writes data/atlas_joint/alice_cd4_per_donor.parquet and alice_cd8.parquet.
# Usage:
#   ./run_alice_pgen.sh                 # defaults
#   ./run_alice_pgen.sh --force         # recompute caches
#   QUEUE=new-medium MEM_MB=64000 WALL_H=48 ./run_alice_pgen.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_alice_pgen_inner.sh"
QUEUE="${QUEUE:-gsla_high_gpu}"   # OLGA is CPU-only; set QUEUE to a CPU queue if available
MEM_MB="${MEM_MB:-32000}"
WALL_H="${WALL_H:-48}"
LOG="$JOB_DIR/run_alice_pgen.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -n 4 \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J alice_pgen \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
