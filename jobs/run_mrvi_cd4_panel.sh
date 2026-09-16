#!/usr/bin/env bash
# Submit one nb43 MrVI training (CD4 cells, one gene list) to a GPU queue.
#
# The input `data/atlas_joint/cd4_mrvi_input.h5ad` is already gene-subset to the union of the
# two lists (~10.5k genes over ~443k CD4 cells, a few GB), so this job is far lighter than
# run_mrvi_joint.sh, which had to read the 48-90 GB annotated atlas. MEM_MB=64000 is ample.
#
# QUEUE: gsla_high_gpu is priority 240 and NOT preemptible; short-gpu / long-gpu are priority
# 74 and can be suspended indefinitely mid-training. Stay on gsla_high_gpu unless the group's
# 700 GB reserved-memory cap is hit -- at 64 GB this run takes 9% of it.
#
# Usage (both arms; run them one after the other or in parallel):
#   JOB_TAG=hvg10k  ./run_mrvi_cd4_panel.sh --gene-list tables/cd4_hvg_10000.csv --tag hvg10k
#   JOB_TAG=panel1k ./run_mrvi_cd4_panel.sh --gene-list tables/cd4_panel_1000.csv --tag panel1k
#   JOB_TAG=panel1k ./run_mrvi_cd4_panel.sh --gene-list tables/cd4_panel_1000.csv --tag panel1k \
#       --skip-train                     # re-emit latents from models/mrvi_cd4_panel1k
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_mrvi_cd4_panel_inner.sh"
QUEUE="${QUEUE:-gsla_high_gpu}"
MEM_MB="${MEM_MB:-64000}"
WALL_H="${WALL_H:-24}"
GMODEL="${GMODEL-}"
JOB_TAG="${JOB_TAG:-cd4}"                    # distinct job name + log per arm
LOG="$JOB_DIR/run_mrvi_cd4_${JOB_TAG}.bsub.log"
rm -f "$LOG"
chmod +x "$INNER"

bsub \
    -q "$QUEUE" \
    -gpu "num=1:j_exclusive=yes${GMODEL:+:gmodel=$GMODEL}" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "mrvi_cd4_${JOB_TAG}" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
echo "submitted mrvi_cd4_${JOB_TAG} -> $QUEUE  (1 GPU, ${MEM_MB}MB)"
echo "log: $LOG"
