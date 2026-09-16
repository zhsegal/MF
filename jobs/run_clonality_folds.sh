#!/usr/bin/env bash
# Submit the leave-3-samples-out fold training for nb38 (the malignancy axis delta-hat). Reads
# jobs/clonality_folds_config.json + the slim input written by notebooks/20_skin/25_delta_gene_axis.ipynb Part A;
# writes z_<run>.npy per run into cfg["out_dir"]. Which arms run is whatever the config lists
# (nb38: structural + structural_donor; the v1 nb37 config also had semantic / ldvae / scvi).
#
# Usage:
#   ./run_clonality_folds.sh                      # one job, every run sequentially
#   ./run_clonality_folds.sh --folds scvi_fold5   # a single run (smoke test / resume)
#   ./run_clonality_folds.sh --folds 0 1 2        # fold 0-2 of every arm
#   ./run_clonality_folds.sh --folds structural_donor_all   # donor-batch arm (all-donor only)
#   ./run_clonality_folds.sh --force              # retrain, ignore caches
#   ARRAY=1 ./run_clonality_folds.sh              # one bsub job per run, in parallel
#   ARRAY=1 MEM_MB=192000 ./run_clonality_folds.sh
#
# Queue / GPU knobs — `bqueues` before choosing; gsla_high_gpu is often backed up:
#   QUEUE=rhel96-gpu WALL_H=6 GPU_EXCL=0 ARRAY=1 ./run_clonality_folds.sh
#   RUNS_FILTER='^(ldvae|scvi)_' ARRAY=1 ./run_clonality_folds.sh   # submit a subset (ARRAY only)
# GPU_EXCL=0 shares the GPU with other jobs. This model needs a few GB (2500 genes, n_latent=10),
# so exclusivity buys nothing and is the main reason jobs sit in PEND.
#   HOSTS=dgn02 ARRAY=1 ./run_clonality_folds.sh   # pin to specific host(s)
# HOSTS matters: on 2026-08-03, dgn06/hgn57/hgn58 accepted these jobs, allocated a GPU, reported
# RUN, and never exec'd the task (bpeek "Not yet started", CPU peak 0.00 for 3h+, MAX_MEM ~5 MB).
# All 15 runs that completed had landed on dgn02. If jobs sit in RUN at single-digit MB with 0% CPU,
# check `bjobs -l` for the exec host and resubmit with HOSTS pinned to a host that works.
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_clonality_folds_inner.sh"
# Measured peak across all three arms is 2.7-3.2 GB (680 MB input + two adata copies + torch), so
# 6 GB is ~2x headroom. Do NOT raise this "just in case": WEXAC bills reserved-minus-used against
# fair share, and the original 128 GB reservation both wasted ~29 GB/job and left every job PENDing.
MEM_MB="${MEM_MB:-6000}"
WALL_H="${WALL_H:-72}"
ARRAY="${ARRAY:-0}"
QUEUE="${QUEUE:-gsla_high_gpu}"
GPU_EXCL="${GPU_EXCL:-1}"
RUNS_FILTER="${RUNS_FILTER:-.}"
HOSTS="${HOSTS:-}"
if [[ "$GPU_EXCL" == "1" ]]; then GPU_SPEC="num=1:j_exclusive=yes"; else GPU_SPEC="num=1"; fi
HOST_ARGS=()
[[ -n "$HOSTS" ]] && HOST_ARGS=(-m "$HOSTS")

if [[ "$ARRAY" == "1" ]]; then
    # the run list is data-driven (arms x folds, and the fold count depends on how many donors
    # clear the eligibility gate) -> read the names the notebook wrote rather than assuming any
    RUNS="$(python - "$JOB_DIR/clonality_folds_config.json" <<'EOF'
import json, sys
try:
    print(" ".join(r["name"] for r in json.load(open(sys.argv[1]))["runs"]))
except Exception:
    print("")
EOF
)"
    if [[ -z "$RUNS" ]]; then
        echo "cannot read the run list from clonality_folds_config.json — run the notebook's" >&2
        echo "config cell (Part 3) first, or submit without ARRAY=1 for a single sequential job." >&2
        exit 1
    fi
    RUNS="$(tr ' ' '\n' <<<"$RUNS" | grep -E "$RUNS_FILTER" | tr '\n' ' ')"
    if [[ -z "${RUNS// /}" ]]; then
        echo "RUNS_FILTER='$RUNS_FILTER' matched no run in the config" >&2
        exit 1
    fi
    # one job per run; each caches independently and skips itself if its latent already exists
    for r in $RUNS; do
        LOG="$JOB_DIR/run_clonality_${r}.bsub.log"
        rm -f "$LOG"
        bsub \
            -q "$QUEUE" \
            -gpu "$GPU_SPEC" \
            "${HOST_ARGS[@]}" \
            -R "rusage[mem=${MEM_MB}]" \
            -W "${WALL_H}:00" \
            -J "clon_${r}" \
            -o "$LOG" -e "$LOG" \
            "$INNER" --folds "$r" "$@" > /dev/null
    done
    echo "submitted $(wc -w <<<"$RUNS") runs to $QUEUE (gpu: $GPU_SPEC, mem ${MEM_MB}MB${HOSTS:+, hosts $HOSTS})"
    exit 0
fi

# name the job/log after the selected runs, so two concurrent single-run submissions (e.g. smoke
# tests) do not write to the same file
TAG="folds"
if [[ "${1:-}" == "--folds" ]]; then
    TAG="$(sed 's/[^A-Za-z0-9_]\+/_/g' <<<"${*:2}" | sed 's/^_//; s/_$//')"
    TAG="${TAG:-folds}"
fi
LOG="$JOB_DIR/run_clonality_${TAG}.bsub.log"
rm -f "$LOG"

bsub \
    -q "$QUEUE" \
    -gpu "$GPU_SPEC" \
    "${HOST_ARGS[@]}" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "clon_${TAG}" \
    -o "$LOG" -e "$LOG" \
    "$INNER" "$@"
echo "submitted clon_${TAG} -> $LOG"
