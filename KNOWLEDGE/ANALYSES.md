# What to run to answer what

Every analysis is a `jobs/run_*.py` submitted by its `jobs/run_*.sh` wrapper. **Never run a
notebook.** If the question below is not covered, check `HISTORY.md` — it may have been a
notebook that is now archived, in which case it needs porting first.

How to run one: `./jobs/run_<x>.sh [args]` from the repo root. The wrapper picks the queue and
size; knobs are `QUEUE`, `NCPU`, `MEM_MB`, `WALL_H`, `AFTER=<jobid>`. Watch the log it prints,
and only report once it has finished. Sizes and runtimes that have actually been observed are in
the `wexac-queue-guide` memory.

**Annotation**: every job below uses `helpers/lineage.py` unless the Notes say otherwise.

---

## Cell identity and malignancy

| Question | Script | Reads | Writes | Status |
|---|---|---|---|---|
| What lineage is each non-malignant T/NK cell? | `run_nonmal_T_reannot.py` (`prep passA passB embed label tables`) | `joint_annotated.h5ad` | `data/atlas_joint/nonmal_T_labels_v1.parquet`, `reports/nonmal_T_reannot/` | **current** — the source of the lineage call. `embed` needs a GPU (`gsla_high_gpu`). |
| Did the lineage switch land, and what did it change? | `run_lineage_check.py` | obs parquet, label parquet | `reports/lineage_check/` | **current**. Run it after anything touches `helpers/lineage.py`. |
| Is a cell malignant? | `run_cluster_malig.py` → `run_cluster_malig_model.py` | `skin_T_tcr_annotated_v4.h5ad` | `*_T_malignancy_cluster_v1.parquet` | **frozen** — do not re-run without reading `DECISIONS.md#mal_cluster-is-frozen`. |
| What is in the benign CD4 arm? (diagnostic) | `run_ccc52_qc.py` | `joint_annotated.h5ad` | `reports/ccc52_mvb/qc_benign_arm/` | **v1-pinned by design** — it characterises the *old* arm. |

## Surface targets (CAR-T / bispecific)

| Question | Script | Reads | Writes | Status |
|---|---|---|---|---|
| Per (arm, donor, gene) expression for the screen | `run_target_screen.py --tissue {skin,blood}` | `{skin,blood}_T_annotated.h5ad` | `data/targets/*_target_long.parquet` | **current** |
| …with contamination / artifact QC | `run_target_screen_qc.py --tissue …` | same | `data/targets/` | **current** |
| Per-donor histograms for pooled stratification | `run_target_pool_hist.py --tissue …` | `data/targets/` | `data/targets/` | **current** |
| Measured surface protein (CITE-seq) per cell | `run_adt_cells.py` | buus25 ADT in `joint_annotated.h5ad` | `data/targets/blood_adt_*` | **current**, blood only (8 of 38 donors) |
| Donor-level export of the skin CD4 surfaceome screen | `tgt49_export_donor_long.py` | `data/targets/skin_target_long.parquet` | `tables/tgt49_*` | **current**. Contains `repair_d14()` — the only fix for the gaydosik2023 empty rows. |
| Is a payload target selective? (CD70 TRUCK) | `run_payload_screen.py` | `skin_T_tcr_annotated_v4.h5ad`, `joint_annotated.h5ad` | `tables/pay51_*`, `reports/51_payload_selectivity.md` | **current** |
| Donor-level export of the payload screen | `pay51_export_donor_long.py` | reuses `run_payload_screen` internals | `tables/pay51_*` | **current** |
| Build the surfaceome reference | `build_surfaceome_ref.py` | `data/external_ref/surfaceome/raw/` | `surfaceome.parquet` | **current**, rarely re-run |

## Cell–cell communication

| Question | Script | Reads | Writes | Status |
|---|---|---|---|---|
| Which LR genes does the malignant clone change vs the patient's own benign CD4? | `run_ccc52_mvb.py --annot v2` (`prep pass fit`; `lr` only to rebuild the resource) | `joint_annotated.h5ad`, `nonmal_T_labels_v1.parquet` | `reports/ccc52_mvb_v2/` | **current.** Benign arm from `helpers/lineage.py`. `--annot v1` regenerates the superseded `reports/ccc52_mvb/` instead. |
| Degradome / protease axis in the TME | `run_degradome_build.py` builds the object; the analysis is **not yet ported** | `joint_annotated.h5ad` | `data/degradome/skin_degradome.h5ad` | build **current**, analysis **pending port** (`HISTORY.md`) |
| Healthy-skin degradome reference | `run_external_degradome_build.py` | Reynolds 2021 | external degradome object | **current** |

## Keeping the knowledge base true

| Question | Script | Reads | Writes | Status |
|---|---|---|---|---|
| Does `KNOWLEDGE/` still describe the repo? | `run_knowledge_check.py` (`--strict` to fail on ERROR) | `KNOWLEDGE/*.md`, the file tree | `reports/knowledge_check/` | **current**. Runs inline in seconds — no LSF. Checks registration, bundle shape, and every path and `file:line` citation. It cannot tell you whether a number is *right*. |
| Did the lineage overlay land correctly? | `run_lineage_check.py` | compartment objects + `nonmal_T_labels_v1.parquet` | `reports/lineage_check/` | **current** |

## Atlas, cohort and metadata

| Question | Script | Reads | Writes | Status |
|---|---|---|---|---|
| Build the joint atlas | `run_build_joint.py` | raw cohorts | `atlas_joint/` | **current**, rarely re-run |
| Cohort composition (Table 1) | `build_table1.py` | obs parquet | `tables/table1_atlas_v2.csv` | **current** |
| Curated clinical layer | `build_clinical_metadata.py` → `apply_clinical_meta.py` | `data/clinical/raw/` | `clinical_metadata_v1.csv`, `clin_*` obs columns | **current** |
| Propagate obs corrections into every object | `apply_atlas_v2_fixups.py --apply` | built objects | obs columns | **current**, dry-run by default |
| Per-sample sex from expression | `run_sex_infer.py` | `joint_mrvi_input.h5ad` | `tables/sample_sex_inferred.csv` | **current** |

## Perturb-seq and modelling

| Question | Script | Reads | Writes | Status |
|---|---|---|---|---|
| Compact the GWCD4i Perturb-seq DE | `run_perturbseq_prep.py` | 15.6 GB DE object | compact `.h5ad` | **current** |
| Per-donor pert2state | `run_pseq30.py` | `pseq29_cd4_cells_v1.h5ad` | `pseq30_*` | **current** |
| Regulator-imprint model inputs | `run_regimprint_prep.py` | pseq29 cells, subclones, CNV, GWCD4i DE | `data/regimprint_nb31/` | **current** |
| MRVI: integration, donor distances, DE/DA | `run_mrvi_joint.py`, `run_mrvi_inference.py`, `run_mrvi_de.py`, `run_mrvi_da_descriptive.py`, `run_mrvi_tcells_mb.py`, `run_mrvi_pseudosample_m2.py` | atlas objects | `models/`, `figures/*.nc` | **current**, expensive |

## Spatial (CosMx)

| Question | Script | Status |
|---|---|---|
| Denoise / correct segmentation | `run_cosmx_resolvi.py` | **current**, GPU |
| Hotspot + Harreman modules | `run_cosmx_harreman.py` | **current** (`pxlgn` env) |
| Transfer the malignancy call to CosMx | `run_cosmx_transfer.py` | **current** |

---

## If you want malignant vs benign CD4 expression *today*

This is the most common question and it now has an answer on disk.

1. **Read `reports/ccc52_mvb_v2/for_analysis/`** — built 2026-11-22 on the clean arm. Per tissue,
   `de_all_genes_{skin,blood}.csv` is a full-transcriptome paired DE (11,887 skin / 12,577 blood
   genes), `de_lr_genes_*` the LR subset. Three columns decide whether a number is usable:
   - `significant` already encodes padj < 0.05 **and** |log2FC| ≥ 1 **and** detection ≥ 10% in the
     higher arm. `padj` is BH over the 1,422-gene LR family only, so a gene outside it has a
     p-value and a blank `padj` — never compare the two.
   - `donors_concordant` / `donors_eligible` is the per-donor sign check. Quote it.
   - `label_defining_gene == True` means the arm was *defined* using that gene (CD4, CD40LG,
     CD8A/B, FOXP3, CCR7, SELL, NKG7, GZMK, KLRF1, TRDC, …). Circular; do not report as biology.
2. **Before reusing any older number**, check `annotation_shift.csv` in the same bundle: it puts
   the v1 and v2 effect side by side for every LR gene and gives a verdict. Skin has 23 genes
   that shift by ≥ 1.5 and 79 more by 0.5–1.5; blood has 5 and 40.
3. **To regenerate**: `./jobs/run_ccc52_mvb.sh prep --annot v2`, then `pass`, then `fit`, each
   chained with `AFTER=<jobid>`. ~13 min of cluster time in total (71 s + 312 s + 457 s). The
   `lr` stage only needs re-running if the LIANA resource itself changed.
4. **Surface genes only**: `run_target_screen.py --tissue skin` is already migrated, so a fresh
   run gives clean arms over ~2,355 surface genes.

## Pending — known gaps

| What | Why it matters | Size |
|---|---|---|
| **Re-run the surface-target screens on the clean arm** | `tables/tgt4*` and `reports/51_payload_selectivity.md` are v1-arm results. The code is already migrated, so this is a re-run, not a rewrite. | hours |
| **Port the degradome analysis** (`28_tme_degradome_mc`) | ~4,000 lines, 92 inline functions, currently only in an archived notebook. Its object builder is scripted; the analysis is not. | days |
| **Port the subclone analysis** (`24_subclone_tcr_signaling`) | Produces `subclones_v4.parquet`, which `run_regimprint_prep.py` and `run_subclone_review_pdf.py` consume. Currently un-regenerable. | days |
| **Rebuild `mal_cluster` on the clean lineage** | The call was derived from the old labels. Circular to fix casually; needs re-clustering, re-fitting and re-review. | a project |
| **Fix the gaydosik2023 empty rows in the atlas itself** | Only one export repairs them today. | medium |
| **Port the inferCNV drivers** | `run_arm_cnv.sh` / `run_blood_cnv.sh` executed notebooks 23 and 32 and are retired to `jobs/_v1/`. inferCNV itself is scripted inside `helpers/skin_T_cnv_helpers.py`; what is missing is a headless driver. | medium |
