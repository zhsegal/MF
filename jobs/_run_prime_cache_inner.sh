#!/usr/bin/env bash
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
echo "[prime] $(date) host=$(hostname)"
exec python -u prime_degradome_cache.py
