#!/usr/bin/env bash
# Build the joint atlas -> joint_annotated.h5ad in a BIG-MEMORY batch job (nb10 Steps 1-3b).
# The interactive kernel (63 GB) OOMs on the concat.
#
# v2 SIZING (2026-09-05): the atlas goes 1,173,694 -> ~2,198,834 cells (1.87x) and
# 40,821 -> 42,506 genes, so v1's measured 92 GB peak scales to roughly 180 GB and the old
# 200 GB default is no longer a margin. Default raised to 380 GB. If it still OOMs, the fix
# is anndata.experimental.concat_on_disk rather than a bigger reservation -- but that
# changes the code path, so try memory first.
# CPU-only work. Default queue is gsla_high_gpu (proven at 200 GB for this user; it will
# hold a GPU it doesn't use) -- override QUEUE to a CPU queue that permits 200 GB if available.
# Usage:
#   ./run_build_joint.sh
#   ./run_build_joint.sh --force-tcr
#   QUEUE=long MEM_MB=200000 WALL_H=24 ./run_build_joint.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_build_joint_inner.sh"
QUEUE="${QUEUE:-gsla_high_gpu}"
MEM_MB="${MEM_MB:-380000}"
WALL_H="${WALL_H:-24}"
LOG="$JOB_DIR/run_build_joint.bsub.log"
rm -f "$LOG"

# reserve a GPU only on the gpu queue; CPU queues just take -n/mem
GPU_ARG=(); case "$QUEUE" in *gpu*) GPU_ARG=(-gpu "num=1:j_exclusive=yes");; esac

# single slot: LSF multiplies rusage[mem] by -n, so -n 1 keeps the reservation = MEM_MB
# (the build is single-threaded pandas/anndata; extra slots only inflate the reservation).
bsub \
    -q "$QUEUE" \
    "${GPU_ARG[@]}" \
    -n 1 \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J build_joint \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
