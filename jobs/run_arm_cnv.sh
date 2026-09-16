#!/usr/bin/env bash
# Submit nb30 (malignancy TCR+CNV, incl. the per-arm inferCNV that feeds nb31) to a CPU queue.
# Arm-CNV is CPU-only and now parallel: n_jobs = allocated cores. Request many slots on ONE host.
# Use a general CPU queue (long/short), NOT gsla_high_gpu: the GPU queue is congested and a 32-slot
# job there pended ~6 h, whereas `long` satisfied a 64-slot span[hosts=1] reservation in ~40 s.
# The steps checkpoint per donor (skin_T_cnv_per_cell_v5_* / skin_T_arm_cnv_v5_parts/) so a
# re-submit resumes.
#
# v5 = the v4 caller on the v2 atlas: same two full inferCNV passes (per-cell score + per-arm) at
# window 100 over ~15.4k positioned genes with a 0.5 null holdout, but ~114 query donors / ~493k
# T cells instead of v4's 38 / 250k. Budget ~2x v4: ~6-10 h wall, ~120 GB peak. Defaults
# (NSLOTS=64 MEM_MB=4000) reserve 256 GB, which still covers it. The per-donor checkpointing means
# a WALL_H overrun costs only the donor in flight -- resubmit and it resumes.
# A pilot pass is worth it first: 7 of the 14 v2 cohorts have never been through this caller.
# There is no pilot knob any more -- add `CNV_DONORS = CNV_DONORS[:5]` after Step 5's prep cell and
# give CNV_CACHE / ARM_CACHE a `_pilot` suffix so the truncated run does not poison the real
# caches. Costs ~30 min and exercises every code path.
# Usage:
#   ./run_arm_cnv.sh
#   NSLOTS=32 MEM_MB=6000 QUEUE=short WALL_H=72 ./run_arm_cnv.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_arm_cnv_inner.sh"
QUEUE="${QUEUE:-long}"
NSLOTS="${NSLOTS:-64}"        # cores; process_map fans chunks across these
MEM_MB="${MEM_MB:-4000}"      # per-slot; LSF reserves MEM_MB * NSLOTS total (~256 GB at defaults)
WALL_H="${WALL_H:-72}"
LOG="$JOB_DIR/run_arm_cnv.bsub.log"
rm -f "$LOG"
chmod +x "$INNER"

bsub \
    -q "$QUEUE" \
    -n "$NSLOTS" \
    -R "span[hosts=1] rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J arm_cnv \
    -o "$LOG" -e "$LOG" \
    "$INNER"
echo "submitted arm_cnv -> $QUEUE  (-n $NSLOTS, span[hosts=1], ${MEM_MB}MB/slot)"
echo "log: $LOG"
