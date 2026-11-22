# Retired with the v1 degradome notebook (2026-09-17)

`prime_degradome_cache.py`, `run_nb42.sh`, `_run_nb42_inner.sh`, `_run_prime_cache_inner.sh`
drove `notebooks/old/28_tme_degradome_v1.ipynb`. They call the source-attribution cube, the
shuffle null and the four pseudobulk DESeq2 contrasts, none of which exist in
`helpers/degradome_helpers.py` any more — the v2 analysis is single-cell and study-blocked.

`jobs/run_degradome_build.{sh,py}` is NOT retired: the streaming build is still how the object
is made, and it is driven entirely by `helpers/degradome_data.py`.
