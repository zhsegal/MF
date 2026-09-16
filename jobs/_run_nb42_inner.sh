#!/usr/bin/env bash
# Execute 42_tme_degradome.ipynb end-to-end. CPU only -- nothing here trains.
set -eo pipefail

NB_DIR=/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
PYBIN=/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/bin/python
NB="$NB_DIR/42_tme_degradome.ipynb"

cd "$NB_DIR"
echo "[nb42] $(date) host=$(hostname)"
echo "[nb42] python: $PYBIN"
"$PYBIN" -c "import pydeseq2, statsmodels, scanpy; print('pydeseq2 ok | scanpy', scanpy.__version__)"

"$PYBIN" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$NB"
echo "[nb42] $(date) DONE"
