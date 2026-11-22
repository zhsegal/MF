# Start here

This is the knowledge base for the MF / CTCL atlas project. It is the **source of truth** for
everything that is not derivable by reading the code: where the data is, what has been decided,
what is known to be broken, and which script answers which question.

If you are an agent picking up an analysis question, read this file, then `ANALYSES.md`, then the
`CAVEATS.md` entries that touch your data. That is usually enough to start.

## The project in four sentences

Multi-tissue single-cell atlas of cutaneous T-cell lymphoma (mycosis fungoides / Sézary
syndrome): **2,157,693 cells, 14 studies, 1,345,527 skin and 811,024 blood**. The malignant cells
are a CD4 T-cell clone, so almost every question is some form of *malignant CD4 versus the
patient's own benign CD4*. The current objective is surface targets for CAR-T and bispecifics,
plus the cell–cell communication the clone rewires. Integration is **MRVI** (`scvi.external.MRVI`),
not SemanticSCVI.

## How this project is organised

| Directory | What it is |
|---|---|
| `data/` | ~612 GB of objects. Never modified by an analysis. See `DATA_MAP.md`. |
| `jobs/` | **The only execution surface.** Every analysis is a `jobs/run_*.py` submitted by its `jobs/run_*.sh` bsub wrapper. See `ANALYSES.md`. |
| `helpers/` | Library code the jobs import. Flat modules, no sub-packages. |
| `reports/` | Analysis output. Each one ends in `reports/<name>/for_analysis/` — see "The output contract". |
| `KNOWLEDGE/` | This directory. |
| `docs/` | Source material, design specs and superseded documents. Linked from here; never the first place to look. |
| `notebooks/_archive/` | Retired. **Notebooks are never run.** Kept as the record of how frozen artifacts were made. |
| `tables/`, `figures/`, `models/` | Artifacts. Gitignored — they exist on disk only. |

## The rules that matter

1. **Never run a notebook.** They are archived. If a notebook's analysis is needed again, port it
   to `jobs/` first; `HISTORY.md` records what each one did.
2. **Heavy work goes to LSF**, never the login node — the login node is fragile and can lock the
   user out. Submit with the job's `.sh` wrapper, watch it, report only when it has finished.
   Big Bash commands (recursive `find`/`rg` over the tree, reads of the 112 GB atlas) can crash it
   too.
3. **Lineage comes from `helpers/lineage.py`**, never from a literal `cell_type_final == "CD4"`.
   That old label is an else-branch and is roughly 20% CD8 and 20% Treg in skin. See
   `CAVEATS.md#the-cd4-label-was-a-default`.
4. **`data/` is read-only to an analysis.** Outputs go to `reports/` and `tables/`. The one
   exception is a job whose declared purpose is to build a data object.
5. **Check the annotation vintage before quoting a number.** `RESULTS.md` says which annotation
   each existing report was built on. v1 and v2 artifacts share filenames, and `_v2`/`_v3`
   suffixes are *not* recency markers.

## The output contract

Every analysis ends by writing `reports/<name>/for_analysis/`, containing:

- `INDEX.md` — orientation, a file table with row counts, how to read the columns, and a
  "things that will mislead you if you do not know them" section;
- the numerical results as plain `.csv`;
- `report.md` — what was done, what was decided, what the numbers say.

No figures, no `.npz`, no `.parquet` in the bundle. It is meant to be pasted whole into another
model, so everything in it must be readable as text and self-explaining. `helpers/bundle.py`
writes it; `reports/ccc52_mvb/for_analysis/` is the worked example.

## The files here

| File | What it answers |
|---|---|
| `PROJECT.md` | What are we studying, what is the biology, what is the current objective |
| `DATA_MAP.md` | Where is X, what is in it, what will bite me |
| `ANALYSES.md` | Which script answers which question, what it reads and writes, is it current |
| `DECISIONS.md` | Why is it done this way, what is frozen and must not be recomputed |
| `CAVEATS.md` | What is known to be wrong or misleading in the data |
| `RESULTS.md` | What has been found, where the numbers live, which annotation they used |
| `HISTORY.md` | What changed when and why; the retirement ledger |

## Keeping it true

A claim here carries its source: a `file:line`, a report path, or the run that produced it.
When an analysis changes a fact, the same change updates this directory. Knowledge that is
derivable from code belongs in a docstring, not here.
