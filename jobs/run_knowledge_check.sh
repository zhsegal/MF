#!/usr/bin/env bash
# Check that KNOWLEDGE/ still describes this repository.
#
#   ./run_knowledge_check.sh              # report
#   ./run_knowledge_check.sh --strict     # exit 1 on any ERROR
#
# This one does NOT go to LSF. It reads text and lists directories -- seconds, a few MB, no data
# object is opened -- so submitting it would cost more than running it. It is the exception to
# "jobs go to the cluster", and it is the exception on purpose.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env
exec python -u "$JOB_DIR/run_knowledge_check.py" "$@"
