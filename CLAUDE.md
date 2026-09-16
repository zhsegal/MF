# MF — CTCL atlas

CTCL / Mycosis Fungoides multi-tissue atlas. **Atlas v2** = 2,157,693 cells, 14 studies
(1,345,527 skin / 811,024 blood). Uses **MRVI** (`scvi.external.MRVI`), *not* SemanticSCVI.
Layout and reading order: `README.md`. Pre-reorg notebook numbers: `NOTEBOOK_MAP.md`.

## Path contract — do not break it

- Notebooks resolve `NB_DIR` by walking up from `cwd` for a directory named **`MF`** containing
  `data/`. Renaming this directory breaks all 18 of them.
- Notebooks then `sys.path.insert(0, str(NB_DIR / "helpers"))` and import the modules flat
  (`import ccc_utils as U`). Keep helper modules directly in `helpers/`, not in sub-packages.
- Modules in `helpers/` that need the project root use `Path(__file__).resolve().parent.parent`.
  Never hardcode an absolute path — two modules did, and the move broke them.
- `helpers/delta_axis_helpers.py` imports `_pert_utils`, `benchmark_modalities` and
  `factor_evidence` from the SemanticSCVI fork via `SCVI_NB` (env: `SCVI_NB_DIR`). That is the
  only cross-repo dependency; do not add more.

## v1 vs v2 — the failure mode to avoid

v1 and v2 artifacts share filenames. `11_atlas_v2_qc` exists because a notebook once reported v1
numbers on a v2 input. Before quoting anything: confirm the input is a v2 object
(`atlas_obs_full_v2.parquet`, `joint_*`, `skin_T_annotated.h5ad`, `blood_T_annotated.h5ad`), and
note that `_v2`/`_v3` suffixes are **not** reliable recency markers — `skin_T_annotated_v2.h5ad`
was older than the unsuffixed file. The v1 freeze is `data/_archive_v1/`; it is read-only.

Malignancy call revisions are versioned in the filename (`skin_T_malignancy_v5.parquet`,
`blood_T_malignancy_v3.parquet`). Check which revision a notebook pins before comparing results.

## Compute

- Kernel `neural_nmf_env`. **Never run heavy jobs on the login node** — training,
  `get_normalized_expression`, inferCNV, full-notebook execution, large fits. They go to
  `jobs/*.sh` (bsub) or to the user's GPU kernel. Big/expensive Bash (recursive `find`/`rg` over
  the tree, reads of the 112 GB atlas) can crash the login node too — keep commands scoped.
- `data/` is ~815 GB (375 GB `atlas_joint`, 251 GB `_archive_v1`, the rest raw cohorts) on a
  volume that runs near capacity. Check `df` before writing anything large.

## Notebooks

- Edit `.ipynb` with the `NotebookEdit` tool. Never use the Jupyter MCP.
- Don't execute notebooks — make the edit and hand the run back.
- Outputs: prefer `.shape`/`.head()`; print `adata`, not `.to_df()`.
- `polars` > `pandas` for big frames; `pathlib` > `os.path`.
- Artifact filenames encode the notebook that wrote them using **pre-reorg** numbers
  (`figures/nb38_*`, `tables/ccc40_*`, `tables/deg42_*`). They are read back by code — don't
  rename them.

## Progress

Log significant changes to the Notion page per the protocol in root `../.claude/CLAUDE.md`
(this project → the MF / CTCL atlas page).
