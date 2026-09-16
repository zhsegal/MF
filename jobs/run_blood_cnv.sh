#!/usr/bin/env bash
# Submit nb34 (blood malignancy TCR+CNV) to a CPU queue.
#
# v3 runs inferCNV on the FULL gene space (~20k positioned genes vs v2's 7,463), so the two
# per-sample passes are ~3x the v2 work on top of the one-off full-gene object build (a blocked
# h5py read of the 48 GB joint_annotated.h5ad, ~12 GB written). Half the baseline is additionally
# held out and scored as the diploid null. Do not run this interactively on the login node.
#
# Use a general CPU queue (long/short), NOT gsla_high_gpu: the GPU queue is congested and a 32-slot
# job there pended ~6 h, whereas `long` satisfied a 64-slot span[hosts=1] reservation in ~40 s.
# Step 12 checkpoints per sample (blood_T_arm_cnv_v3_parts/) so a re-submit resumes, and the
# full-gene object + Step 11 scores are cached, so a resubmitted job skips what already ran.
#
# Prerequisites: jobs/download_onek1k.sh (4.4 GB, one-off) and data/atlas_joint/joint_annotated.h5ad.
#
# Usage:
#   ./run_blood_cnv.sh
#   NSLOTS=32 MEM_MB=6000 QUEUE=short WALL_H=72 ./run_blood_cnv.sh
#   DEPEND='done(115683)' ./run_blood_cnv.sh     # hold until the download job finishes
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_blood_cnv_inner.sh"
QUEUE="${QUEUE:-long}"
NSLOTS="${NSLOTS:-64}"        # cores; process_map fans chunks across these
MEM_MB="${MEM_MB:-4000}"      # per-slot; LSF reserves MEM_MB * NSLOTS total (~256 GB at defaults)
WALL_H="${WALL_H:-72}"
LOG="$JOB_DIR/run_blood_cnv.bsub.log"
rm -f "$LOG"
chmod +x "$INNER"

DEPEND_ARGS=()
[[ -n "${DEPEND:-}" ]] && DEPEND_ARGS=(-w "$DEPEND")

bsub \
    "${DEPEND_ARGS[@]}" \
    -q "$QUEUE" \
    -n "$NSLOTS" \
    -R "span[hosts=1] rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J blood_cnv \
    -o "$LOG" -e "$LOG" \
    "$INNER"
echo "submitted blood_cnv -> $QUEUE  (-n $NSLOTS, span[hosts=1], ${MEM_MB}MB/slot)"
echo "log: $LOG"
