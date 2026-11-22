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

**`data/` is near capacity** — check `df` before writing anything large. The v1 freeze that
used to sit in data/_archive_v1/ has been deleted; v1 objects are not recoverable
(`KNOWLEDGE/DATA_MAP.md`).

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

## Closing the loop — do this before you report a result

An analysis is not finished when the job exits. It is finished when the next agent can find it.
Five steps, in this order:

1. **The bundle.** `reports/<name>/for_analysis/` via `helpers/bundle.py`.
2. **`KNOWLEDGE/`.** Update every file the work touched — this is the step that gets skipped:
   | You did this | Update |
   |---|---|
   | produced numbers someone will quote | `RESULTS.md` — one row: where, **which annotation**, the headline |
   | wrote or changed a `jobs/run_*.py` | `ANALYSES.md` — the registry row, and strike it from the pending table if it closed a gap |
   | found a data defect, or a trap in your own method | `CAVEATS.md` — with its blast radius |
   | chose something a later agent might undo | `DECISIONS.md` |
   | superseded an artifact | `RESULTS.md` (mark the old row) + `HISTORY.md` |
   Carry the real number, not a rounded memory of it, and cite the file it came from.
   **Verify each number against the file on disk before writing it down** — a table you generated
   earlier in the session may have been regenerated since.
3. **Claude memory is pointers only.** `KNOWLEDGE/` is the source of truth and the thing that
   survives; memory is a per-machine cache that a new checkout will not have. Do **not** write a
   project fact into memory — write it into `KNOWLEDGE/` and, if it changes how an agent should
   work, add one line to the matching memory pointing there. The exceptions, which are about
   *how to work* rather than about the project, are `wexac-queue-guide` (update after **every**
   job: queue, slots, pend, runtime, max mem) and `submit-and-watch-jobs`.
4. **Notion.** One bullet, per the protocol in the root `../.claude/CLAUDE.md`.
5. **Check your own work**: `./jobs/run_knowledge_check.sh`. It runs inline in a second and
   tells you which of the four steps you actually completed — an unregistered bundle, a job
   nothing documents, a path you moved out from under a citation. Fix what it finds now, while
   you still remember why. A clean run is not proof the knowledge base is *correct*, only that
   it is consistent with the file tree: it cannot see a wrong number or a missing caveat.

If you ran out of room to do all five, say so in your reply rather than leaving it silent.
