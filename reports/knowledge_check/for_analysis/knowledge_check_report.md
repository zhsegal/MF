# Is the knowledge base still true? (2026-11-22 18:14)

`jobs/run_knowledge_check.py` compared `KNOWLEDGE/` against the repository.

**0 ERROR, 5 WARN.**

| check | severity | n |
|---|---|---|
| job_wrapper | WARN | 2 |
| path_ref | WARN | 3 |

## Findings

| severity | check | subject | detail | suggested fix |
|---|---|---|---|---|
| WARN | job_wrapper | `jobs/run_regimprint_prep.py` | no run_*.sh wrapper, so it has no documented route onto LSF | add the bsub wrapper, or say in ANALYSES.md that it runs inline |
| WARN | job_wrapper | `jobs/run_subclone_review_pdf.py` | no run_*.sh wrapper, so it has no documented route onto LSF | add the bsub wrapper, or say in ANALYSES.md that it runs inline |
| WARN | path_ref | `KNOWLEDGE/HISTORY.md` | cites `jobs/run_notebook.sh`, which does not exist | update the citation, or say in the text that the file is gone |
| WARN | path_ref | `KNOWLEDGE/NOTEBOOK_MAP.md` | cites `data/Li2024_atlas/visium/`, which does not exist | update the citation, or say in the text that the file is gone |
| WARN | path_ref | `KNOWLEDGE/NOTEBOOK_MAP.md` | cites `data/cache/mrvi_ctcl_cache.h5ad`, which does not exist | update the citation, or say in the text that the file is gone |

## What this does not check

Only that the map matches the territory. It cannot tell you whether a number in `RESULTS.md` is *correct*, whether a caveat is still true, or whether an analysis was any good. A registered bundle with a wrong headline passes every check here.
