# What changed, when, and why

The full per-notebook retirement ledger — every notebook, its old number, when it was retired and the stated reason — is kept verbatim in `NOTEBOOK_MAP.md` beside this file. This page is the summary and the decoder.

Also the decoder for artifact filenames, and the record of what each archived notebook did.

## Timeline

| When | What |
|---|---|
| 2026-05/06 | v1 atlas: Li 2024 replication, SemanticSCVI geometric experiments, early malignancy calls. Was frozen in data/_archive_v1/, **deleted 2026-11-22**; only `docs/archive/ATLAS_V1_PARAMETERS.md` survives. Parameters: `docs/archive/ATLAS_V1_PARAMETERS.md`. |
| 2026-09-05 | Dedup gate: 45 identity groups, GSE266862 ≡ GSE269981 byte-identical. `docs/ATLAS_V2_DEDUP_GATE.md`. Note it records that `detect_duplicate_cells` does not do what its docstring says. |
| 2026-09-16 | Repo reorganisation. Notebooks renumbered; spatial arm retired (`docs/SPATIAL_data_map.md`). This is the mtime on most of `docs/` and does not reflect when the content was written. |
| 2026-10-22 | **`mal_cluster` becomes the headline malignancy call**, replacing `mal_knn` (demoted to a sensitivity arm). Several notebooks retired the same day. Blood `Ambient_soup` rescue decided: **no**. |
| 2026-11-05 | Xenium Prime 5K add-on panel finalised (`docs/XENIUM_5K_ADDON_SPEC.md`); 10x rejected 8 genes on 11-04, backfilled. |
| 2026-11-19 | CD70 payload selectivity screen (`reports/51_payload_selectivity.md`). |
| 2026-11-22 | Malignant-vs-benign CCC analysis (`reports/ccc52_mvb/`); its QC showed the CD4 label is a default; **non-malignant T/NK re-annotation** (`nonmal_T_labels_v1`); project reorganised to data + scripts + knowledge, notebooks retired. |

## The malignancy call, by revision

| Call | File | Status |
|---|---|---|
| `mal_tcr` | `skin_T_malignancy_v5.parquet`, `blood_T_malignancy_v4.parquet` | current, a sequence fact |
| `mal_cnv` | same files (`cnv_callable` gates it) | current, sensitivity |
| `mal_knn` | retired notebook's output | **demoted 2026-10-22**; ~23% healthy false-positive floor |
| `mal_cluster` | `*_T_malignancy_cluster_v1.parquet` | **headline since 2026-10-22**; frozen |
| `combined_malignant` | — | **retired**: a disjunction hides which measurement fired |

Skin is at malignancy v5, blood at v4. (`CLAUDE.md` used to say blood v3; that was stale.)

## Artifact filename prefixes

Filenames encode the **pre-reorganisation notebook number**. They are read back by code — never
rename them.

| Prefix | Produced by | Now |
|---|---|---|
| `ccc_*`, `ccc40_*` | v1 CCC notebooks | archived; `helpers/ccc_helpers`, `ccc_data` |
| `deg28_*`, `deg28mc_*`, `deg28b*`, `amb28c*` | degradome family, `_mc` = cluster-call variant | `28_tme_degradome_mc` archived, **port pending** |
| `deg28d_*`, `deg28e_*`, `cmx28f_*` | CosMx spatial | `run_cosmx_*.py` |
| `tgt41_*` … `tgt49_*` | surface-target screens | `run_target_screen*.py`, `tgt49_export_donor_long.py` |
| `pay51_*` | payload screen | `run_payload_screen.py`, `pay51_export_donor_long.py` |
| `pseq29_*`, `pseq30_*` | Perturb-seq | `run_perturbseq_prep.py`, `run_pseq30.py` |
| `nb31b_*`, `nb31c_*`, `nb26b_*`, `nb32b_*` | blood subtypes, ambient rescue, cluster malignancy | partly scripted |
| `nb38_*` | delta gene axis | archived, legacy |

`tables/*_llm_export.md` and `*_run_log.md` are the **only prose copy** of the tgt41–49, pseq29,
ccc40 and deg28 results. `tables/` is gitignored, so they exist on disk only — do not delete them.

## Archived notebooks

All of `notebooks/` is retired as of 2026-11-22 and lives under `notebooks/_archive/`. Nothing is
run from there. Outputs are preserved in the working tree (git stores notebooks stripped), and
`notebooks/_archive/_rendered/` holds a text rendering of each so the printed numbers are
greppable without opening a 20 MB JSON.

**Produced a frozen artifact — not ported, will not be regenerated:**

| Notebook | Produced | Note |
|---|---|---|
| `20_skin/21_reannotation` | skin `cell_type_final`, `skin_T_annotated.h5ad` | The hand map that made `CD4` an else-branch. **Superseded for T cells** by `nonmal_T_labels_v1`. |
| `30_blood/31_reannotation` | blood `cell_type_final` | Same; ~20% of blood CD4 came from a second pass over the `Unk` residue. |
| `20_skin/22`, `22b`; `30_blood/31b`, `31c` | myeloid/fibroblast, unassigned, blood non-T subtypes, ambient rescue | Hand annotation via `helpers/annot_utils.py`. 22b is where "every NK and ILC cell was called CD8" is recorded. |
| `10_atlas/11_atlas_v2_qc` | the v2 release QC verdict | One-off release gate. |
| `20_skin/26b`, `30_blood/32b` cluster malignancy | the manual `mal_cluster` decisions | The automated part is `run_cluster_malig*.py`; the **manual calls and the choice of score** were made here. |

**Analysis that exists only in an archived notebook — port before re-running:**

| Notebook | Why it matters | Size |
|---|---|---|
| `20_skin/28_tme_degradome_mc` | The degradome/TME analysis behind `tables/deg28mc_*`. | ~4,000 code lines, 92 inline functions |
| `20_skin/24_subclone_tcr_signaling` | Produces `subclones_v4.parquet`, consumed by `run_regimprint_prep.py` and `run_subclone_review_pdf.py`. | 29 cells, 11 functions |
| `20_skin/23`, `30_blood/32` malignancy_tcr_cnv | The ALICE + inferCNV pipeline order and parameters (the heavy steps are `run_arm_cnv.sh` / `run_blood_cnv.sh`). | thin, but the order matters |
| `10_atlas/12`, `13`, `14`; `00_preprocess/01` | Descriptive, skin-vs-blood subclone comparison, calling malignancy on new data. `14` is the only route for new data. | small |
| `40_targets/42`–`49`, `28f` | Screen statistics and figures on top of `target_*` helpers and the scripted long tables. | mostly helper calls |

`notebooks/_archive/old/` holds the ~46 pre-2026-09 notebooks (v1 CCC, SemanticSCVI, early
malignancy). `notebooks/_archive/old/ccc_v1_implementation_dossier.md` documents the v1 CCC
generation including its malignancy call.

## Retired on purpose

- **Spatial arm** (Visium/resolVI deconvolution) — retired 2026-09-16, recipe kept in
  `docs/SPATIAL_data_map.md` in case it reopens.
- **`jobs/_v1/`** — the v1 degradome drivers. They depend on functions that no longer exist in
  `helpers/degradome_helpers.py` (`jobs/_v1/README.md`).
- **`jobs/run_notebook.sh` (deleted)** and `_run_notebook_inner.sh` — the generic notebook executor.
  Retired with the notebooks; a `run_notebook_<nb>.bsub.log` in `jobs/` is the fossil record of
  which notebooks were ever executed on the cluster.
- **SemanticSCVI** — this project uses MRVI. The fork is a separate repo, reached only by
  `helpers/delta_axis_helpers.py` via `SCVI_NB_DIR`, and that helper is itself legacy.

## Legacy job drivers, kept but not current

These are headless drivers written for notebooks that are now archived. They still run, but they
feed a retired analysis, so they are **not** in `ANALYSES.md` — do not reach for one to answer a
new question without reading it first. Listed here so nobody rewrites one from scratch.

| Script | Drove | Note |
|---|---|---|
| `jobs/run_alice_pgen.py` | `old/21_tcr_alice_neighborhood` | OLGA Pgen / ALICE sweep; writes the two caches that notebook read. Still the only ALICE driver. |
| `jobs/run_ccc_build.py` | nb34–36 (v1 LIANA CCC) | Builds the skin CCC object by streaming `joint_annotated.h5ad`. Superseded for the malignant-vs-benign question by `run_ccc52_mvb.py`. |
| `jobs/run_clonality_prep.py` | nb37 Parts 0–3 | Heavy CPU prep producing the fold job's input. |
| `jobs/run_clonality_folds.py` | nb38 Part A | Leave-3-samples-out training of the malignancy-axis arms. |
| `jobs/run_mrvi_cd4_panel.py` | nb43 | Trains MrVI on a CD4 subset restricted to one gene list; two runs give the matched latents. |
| `jobs/run_cd4_sweep.py` | `old/19_semantic_cd4_atlas_sweep` | SemanticSCVI hyperparameter sweep. Legacy twice over — the project uses MRVI. |
| `jobs/regimprint_timing.py` | — | Timing diagnostic for regimprint fits (ms per gradient, NUTS tree size). No model change; a profiler, not an analysis. |
