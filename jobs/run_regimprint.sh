#!/usr/bin/env bash
# Submit one regimprint (nb31) stage. Package: docs/regimprint_v0.2 (override: RI_PKG); outputs: data/regimprint_nb31/.
#
#   ./run_regimprint.sh prep                                   # CPU: tables + inputs.npz
#   ./run_regimprint.sh verify  <outdir>                       # GPU: pytest fast+slow
#   ./run_regimprint.sh time <outfile> --input IN [--set ...]  # GPU: scripts/time_one_fit.py (step C0)
#   ./run_regimprint.sh sim planted 20 <outfile>               # GPU
#   ./run_regimprint.sh tune  --input IN --out RUN [--tune-params ...]
#   FOLD=0 ./run_regimprint.sh b1 --input IN --out RUN         # one fold
#   RANGE=2-5 ./run_regimprint.sh b1 --input IN --out RUN      # LSF array, fold = index - 1
#   ./run_regimprint.sh final --input IN --out RUN
#   ./run_regimprint.sh report --out RUN                       # CPU
# Knobs: QUEUE, WALL_H, MEM_MB (per slot), NCPU, AFTER="done(123) && done(124)", HOSTS, GMODEL, TAG
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$JOB_DIR/_run_regimprint_inner.sh"; chmod +x "$INNER"
STAGE="$1"; shift
LOGD="$JOB_DIR/../data/regimprint_nb31/logs"; mkdir -p "$LOGD"

GPU=1
case "$STAGE" in prep|report) GPU=0 ;; esac
if [[ "$GPU" == 1 ]]; then
    QUEUE="${QUEUE:-long-gpu}"; NCPU="${NCPU:-2}"; MEM_MB="${MEM_MB:-12000}"; WALL_H="${WALL_H:-48}"
else
    QUEUE="${QUEUE:-short}"; NCPU="${NCPU:-2}"; MEM_MB="${MEM_MB:-48000}"; WALL_H="${WALL_H:-6}"
fi
TAG="${TAG:-$STAGE}"
NAME="ri_${TAG}"; LOG="$LOGD/${TAG}.log"
ARGS=(-q "$QUEUE" -n "$NCPU" -R "span[hosts=1]" -R "rusage[mem=${MEM_MB}]" -W "${WALL_H}:00")
if [[ "$GPU" == 1 ]]; then ARGS+=(-gpu "num=1${GMODEL:+:gmodel=$GMODEL}"); fi
if [[ -n "${HOSTS:-}" ]]; then ARGS+=(-m "$HOSTS"); fi
if [[ -n "${AFTER:-}" ]]; then ARGS+=(-w "$AFTER"); fi
ENVV=()
if [[ -n "${RANGE:-}" ]]; then NAME="${NAME}[${RANGE}]"; LOG="$LOGD/${TAG}_%I.log"; fi
if [[ -n "${FOLD:-}" ]]; then ARGS+=(-env "all, FOLD=${FOLD}"); LOG="$LOGD/${TAG}_fold${FOLD}.log"; fi

bsub "${ARGS[@]}" -J "$NAME" -o "$LOG" -e "$LOG" "$INNER" "$STAGE" "$@"
echo "log: $LOG"
