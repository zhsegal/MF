#!/usr/bin/env bash
# Submit MrVI training on the full joint atlas to a GPU queue.
#
# MEMORY: the 200000 default is ~2.5x what the job actually uses and will PEND FOREVER on
# gsla_high_gpu ("Resource (mem) limit defined on queue has been reached") whenever the lab is
# near the NYOSEF_gsla_high_gpu group cap of 700 GB reserved memory. Measured peaks (LSF
# "Max Memory" in the bsub logs): full atlas 1.17M cells = 70 GB, skin 750k cells = 82 GB.
# The peak is dominated by the 48 GB joint_annotated.h5ad read in build_hvg_input, so it is
# roughly constant across compartment subsets. MEM_MB=120000 is ample for any of them.
#
# QUEUE: the 700 GB cap applies to gsla_high_gpu ONLY. short-gpu / long-gpu have no memory
# limit, just a per-user GPU count -- BUT they are priority 74 vs gsla_high_gpu's 240 and are
# therefore PREEMPTIBLE: a blood run there was suspended (SSUSP, "preempted by a higher
# priority job") ~10 min into training and could have stayed suspended indefinitely. Prefer
# gsla_high_gpu with a right-sized MEM_MB; treat the other GPU queues as a last resort, and
# only for a restart that can resume from the HVG cache. Pass WALL_H within the queue RUNLIMIT
# (short-gpu 6 h, long-gpu 96 h, gsla_high_gpu 72 h).
#
# v2 SIZING (2026-09-05): 1.87x the cells of v1, and the peak is dominated by reading
# joint_annotated.h5ad (48 GB in v1 -> ~90 GB in v2). v1 peaks were full 70 GB / skin 82 GB;
# budget ~160 GB for v2. Default raised 120000 -> 200000. Watch the gsla_high_gpu 700 GB
# group cap noted above -- at 200 GB a single run takes 29% of it.
#
# RESTARTS ARE CHEAP once the HVG cache exists: build_hvg_input() prints "reusing HVG input
# cache" and skips the 48 GB read entirely, so a training-only rerun fits in MEM_MB=48000.
#
# Usage:
#   ./run_mrvi_joint.sh
#   ./run_mrvi_joint.sh --max-epochs 30 --force
#   ./run_mrvi_joint.sh --skip-train          # reuse models/mrvi_joint
#   ./run_mrvi_joint.sh --subsample 300000    # fit on 300k cells, transform all
#   MEM_MB=256000 WALL_H=96 ./run_mrvi_joint.sh
#   JOB_TAG=skin WALL_H=48 ./run_mrvi_joint.sh --subset-compartment Skin --tag skin
#   QUEUE=short-gpu JOB_TAG=blood MEM_MB=120000 WALL_H=6 ./run_mrvi_joint.sh \
#       --subset-compartment Blood --tag blood --skip-diag
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_joint_inner.sh"
QUEUE="${QUEUE:-gsla_high_gpu}"
MEM_MB="${MEM_MB:-200000}"
WALL_H="${WALL_H:-72}"
# Optional GPU model restriction, OFF by default. An earlier read of the
# CUDA_ERROR_ILLEGAL_ADDRESS failures blamed the A40 nodes; that was wrong -- blood failed
# on an L40S too. The actual fix is XLA_PYTHON_CLIENT_PREALLOCATE=true in run_mrvi_joint.py.
# Kept only as an escape hatch: GMODEL=NVIDIAL40S ./run_mrvi_joint.sh
GMODEL="${GMODEL-}"
JOB_TAG="${JOB_TAG:-}"                       # distinct job name + log per subset run
SUFFIX="${JOB_TAG:+_$JOB_TAG}"
LOG="$JOB_DIR/run_mrvi_joint${SUFFIX}.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -gpu "num=1:j_exclusive=yes${GMODEL:+:gmodel=$GMODEL}" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "mrvi_joint${SUFFIX}" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
