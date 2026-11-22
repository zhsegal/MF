# Retired with the v1 degradome notebook (2026-09-17)

`prime_degradome_cache.py`, `run_nb42.sh`, `_run_nb42_inner.sh`, `_run_prime_cache_inner.sh`
drove `notebooks/_archive/old/28_tme_degradome_v1.ipynb`. They call the source-attribution cube, the
shuffle null and the four pseudobulk DESeq2 contrasts, none of which exist in
`helpers/degradome_helpers.py` any more — the v2 analysis is single-cell and study-blocked.

`jobs/run_degradome_build.{sh,py}` is NOT retired: the streaming build is still how the object
is made, and it is driven entirely by `helpers/degradome_data.py`.

## Retired 2026-11-22: the notebook-executing drivers

Notebooks are no longer an execution surface (`KNOWLEDGE/DECISIONS.md`), so every wrapper whose
job was to run one moved here:

| Driver | Ran | Replacement |
|---|---|---|
| `run_notebook.sh`, `_run_notebook_inner.sh` | any notebook, via `nbconvert --execute --inplace` | none — port the analysis to `jobs/` |
| `run_arm_cnv.sh`, `_run_arm_cnv_inner.sh` | `20_skin/23_malignancy_tcr_cnv.ipynb` (skin inferCNV) | **port pending** — `KNOWLEDGE/ANALYSES.md` |
| `run_blood_cnv.sh`, `_run_blood_cnv_inner.sh` | `30_blood/32_malignancy_tcr_cnv.ipynb` (blood inferCNV) | **port pending** |
| `run_nb29.sh`, `_run_nb29_inner.sh` | `20_skin/29_perturbseq_reverse_engineering.ipynb` | `run_perturbseq_prep.py` covers the data prep only |

Their paths were updated to `notebooks/_archive/...` so they are not silently broken, but running
them re-executes an archived notebook, which the project rule forbids. They are kept as the
record of how the inferCNV and Perturb-seq steps were driven.
