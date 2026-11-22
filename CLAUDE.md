# MF — CTCL atlas

**Read `KNOWLEDGE/README.md` first.** It is the source of truth for what this project is, where
the data lives, which script answers which question, and what is known to be broken. This file
is only the operating rules.

Atlas v2 = 2,157,693 cells, 14 studies (1,345,527 skin / 811,024 blood). Integration is **MRVI**
(`scvi.external.MRVI`), not SemanticSCVI.

## The shape of the project

```
data/        ~612 GB of objects. An analysis READS it and never writes to it.
jobs/        the only execution surface: run_<x>.py + its run_<x>.sh bsub wrapper
helpers/     library code the jobs import (flat modules, no sub-packages)
reports/     output, one directory per analysis, each ending in for_analysis/
KNOWLEDGE/   the knowledge base — start here
docs/        source material and superseded specs; linked from KNOWLEDGE, not browsed directly
notebooks/_archive/   retired. Never run.
```

## Rules

**Never run a notebook.** They are archived under `notebooks/_archive/` as the record of how
frozen artifacts were made. If an archived analysis is needed again, port it to `jobs/` first —
`KNOWLEDGE/HISTORY.md` says what each one did and which are already scripted.

**Lineage comes from `helpers/lineage.py`.** Never test `cell_type_final == "CD4"`. That label is
an else-branch and its benign CD4 arm is only ~46% conventional CD4 in skin
(`KNOWLEDGE/CAVEATS.md`). `cluster_malig_helpers.read_obs()` applies the overlay by default; the
one place that opts out is the frozen `mal_cluster` pipeline, which the labels were derived from.

**Every analysis ends in a bundle.** `reports/<name>/for_analysis/` = generated `INDEX.md` +
plain CSVs + `report.md`, written by `helpers/bundle.py`. No figures, no binaries — it is meant
to be pasted whole into another model, so it must orient its own reader. Figures are opt-in.

**Heavy work goes to LSF, never the login node.** Claude submits it (`./jobs/run_<x>.sh`),
watches it to the end, reads the log, fixes and resubmits on failure, and tells the user only
when it has finished with outputs in place. Pick the queue from the `wexac-queue-guide` memory
and right-size cpu/mem. `short` is **preemptable**, so jobs must be idempotent and write at the
end; a job that exits instantly with an empty log was preempted — just resubmit. After each job,
update that memory with queue, size, pend, runtime and max memory.

⚠️ **The login node is fragile.** Heavy jobs AND expensive Bash (recursive `find`/`rg` over the
tree, reads of the 112 GB atlas) can crash it and lock the user out. Keep commands cheap and
scoped. Kernel/env is `neural_nmf_env`; `liana` lives in `mrvi_env`.

**`data/` is near capacity** — check `df` before writing anything large. `data/_archive_v1/` is
read-only.

**Check the annotation vintage before quoting a number.** `KNOWLEDGE/RESULTS.md` records which
annotation each existing result used. v1 and v2 artifacts share filenames and `_v2`/`_v3`
suffixes are not recency markers.

**Artifact filenames encode pre-reorg notebook numbers** (`tables/deg28mc_*`, `tables/tgt49_*`,
`figures/nb38_*`). They are read back by code — do not rename them. `KNOWLEDGE/HISTORY.md`
decodes them.

## Path contract

`helpers/` modules resolve the project root as `Path(__file__).resolve().parent.parent`, and jobs
add `helpers/` to `sys.path`. Keep helper modules flat in `helpers/` and keep `jobs/` and
`helpers/` in the same parent. Never hardcode an absolute path.
`helpers/delta_axis_helpers.py` reaching the SemanticSCVI fork via `SCVI_NB_DIR` is the only
cross-repo dependency; do not add more.

## Progress

Log significant changes to the Notion MF / CTCL atlas page per the protocol in the root
`../.claude/CLAUDE.md`.
