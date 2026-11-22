# helpers/

Flat modules, imported by notebooks after `sys.path.insert(0, str(NB_DIR / "helpers"))`.
Anything here that needs the project root uses `Path(__file__).resolve().parent.parent`.

## Atlas build & dedup — a 4-stage pipeline, run in this order

| module | role |
|---|---|
| `check_overlap.py` | CLI. Compares raw deposits barcode-by-barcode → `tables/atlas_dedup_v2.csv` |
| `make_dedup_decisions.py` | CLI. Turns that into the keep/drop ledger → `tables/atlas_dedup_v2_decisions.csv` |
| `atlas_join_helpers.py` | The build itself (`concat_joint`, …); consumes the ledger |
| `preflight_v2.py` | CLI. Validates the inputs before a `concat_joint` run |
| `atlas_v2_fixups.py` | Post-build metadata patches; driven by `jobs/apply_atlas_v2_fixups.py` |

The three CLIs have no importers — that is expected, they are entry points, not libraries.

## Analysis modules

| module | used by |
|---|---|
| `alice_helpers.py` | ALICE TCR-neighbourhood statistics — skin/blood malignancy + subclones |
| `skin_T_cnv_helpers.py` | inferCNV pipeline for T cells (both compartments, despite the name) |
| `subclone_helpers.py` | subclone calling and programs |
| `tcr_signaling_helpers.py` | TCR / co-stimulation signaling contrasts |
| `semantic_malig_helpers.py` | malignancy-plane fitting and plots |
| `semantic_clonality_helpers.py` | clonality transfer in a shared latent |
| `delta_axis_helpers.py` | the δ̂ malignancy axis; **the only module that reaches into the SemanticSCVI fork** (`SCVI_NB`, env `SCVI_NB_DIR`) |
| `perturbseq_helpers.py` | `29_perturbseq_reverse_engineering`: the glue around the authors' pert2state model (gene/regulator filters, cis mask, scramble, replication metric, pathway tests, control panel) plus the inferCNV extension (gene→arm map from the CNV GTF, arm-class tests, per-donor dosage-predicted program, per-cell knockdown-mimicry scores vs `arm_z`). Flat; imports only `skin_T_cnv_helpers` and `pert2state_model` |
| `pert2state_model.py` | **vendored verbatim** (MIT) from `github.com/emdann/pert2state_model` — Zhu, Dann et al. Cell 2026 — the SVD + elastic-net cell-state-from-perturbations model; only a licence header was added |
| `malignancy.py` | the **four** malignancy calls (ALICE TCR, inferCNV, kNN propagation, 26b/32b per-sample cluster), each with a definedness mask; never OR-ed. `HEADLINE = mal_cluster` since 2026-10-22. Tissue-keyed: `use_tissue("blood")` before `load_malignancy` or blood silently assembles the skin parquets |
| `cluster_malig_helpers.py` | the per-sample cluster malignancy pipeline (26b skin, 32b blood). `CONFIG[tissue]` + `use_tissue()` hold everything that differs: paths, counts layer, grouping key (skin `sample_id`, blood `donor`), obs columns, CTCL vs Sézary markers, `MIN_CD4`. `read_obs` overlays `CELLTYPE_FIX_CSV` on `cell_type_T` (blood only) and keeps the original as `cell_type_T_nb31`; `nk_vs_t_scores` is the CD8-vs-NK separation the 32b review runs on |
| `annot_utils.py` | shared hand-annotation machinery for `22`, `22b` and `31b` (coverage, size-locked cluster maps, the numeric dump, sidecar write + reconcile) |
| `surface_targets.py` | frozen literal for `40_targets`: the 143-gene surface panel in 12 classes, the 248-gene trafficking machinery in 11 phases, verified CTCL clinical precedent, the blood CITE-seq ADT↔gene map, and `load_surfaceome()` / `screen_genes()`. No imports beyond pandas |
| `target_screen.py` | the five scoring axes of the target screen — prevalence, paired within-donor selectivity, on-target off-tumour, surface plausibility, escape risk — plus `rank_table`, `control_check` and `strat_selectivity` (the same paired test inside each stratum, with an interaction). Donor-level throughout; no composite score by design |
| `target_cohort.py` | donor truth for both screens. `donor` is not a patient in either compartment. **Blood** (`43`/`44`/`45`/`46`): 54 of 131 keys are geskin2026 lane×HTO wells of ex vivo drug cultures, five are healthy donors carried under `disease == "SS"`, two patients appear twice. **Skin** (`42`/`47`/`48`): 19 of 146 keys carry a clone that is not CD4 (CD8, γδ, DN — their CD4 cells land in the *reactive* arm), 15 carry a diagnosis outside the MF/SS spectrum, and ten patients were biopsied more than once. `blood_donor_table()` / `skin_donor_table()` / `donor_table(tissue)` read the curated `clin_*` obs columns and return the whitelist with its per-key `drop_reason`, the curated diagnosis, the treatment booleans and `burden`; `filter_long` applies it |
| `target_qc.py` | the artifact layer for `42` §2 and `43` §2: lineage-control genes as an **empirical** contamination null, six per-gene tests (trim collapse, concentration, lineage coupling, UMI coupling, clean-cell recompute, burden scaling), prevalence-matched residuals so the statistics are not just a function of detection rate, and `band_is_calibrated` — the gate that refuses to exclude anything when the controls do not behave like contamination. `use_tissue()` switches the contaminant modules, the null and the coupling columns between compartments; call it in every notebook, because the module keeps whatever the last one set |
| `target_het.py` | heterogeneity for `44` and `47`: DerSimonian-Laird random effects with prediction intervals, COPA and the outlier sum with a sign-flip null, GMM-BIC bimodality, donor clustering, and `eligibility` (the fraction of patients with at least X% of the tumour positive) |
| `target_figs.py` | the `28_tme_degradome_mc` drawing layer, rewritten against the target screen's long table: paired per-donor slope plots with BH stars and donor votes, dumbbells, donor×gene heatmaps, captions, and PNG+PDF+source-CSV saving, and `llm_export()` — the last cell of every `40_targets` notebook, which prints its complete numerical result as one paste-able block and writes it to `tables/<prefix>llm_export.md` |
| `degradome_data.py` | frozen config for `28_tme_degradome`: paths, roster, gates, the ~290-gene / 23-family panel |
| `degradome_helpers.py` | build + gate for the degradome object (the streaming pass, the asserts, coverage, abstentions) |
| `degradome_cells.py` | the single-cell analysis layer for `28`: family scores, donor medians, study-blocked van Elteren tests, the within-sample paired contrast (`28` §4d/§9c), the plots |
| `ccc_panel.py` | frozen literal: the curated ligand-receptor panel for `28` §10 (`PRIORITY_PAIRS`, `CCC_PANEL`, `NEGATIVES`). Moved out of `docs/` because `docs/` is never on `sys.path`. No imports of its own |
| `ccc_axis.py` | `28` §10: parses `ccc_panel.PRIORITY_PAIRS` into concrete edges, scores them per biopsy off the full-gene pseudobulk cube, tests them under the §4d paired design, and draws the chord / circle-network / hierarchy / delta-heatmap figures. **No liana** -- it runs in `neural_nmf_env` |
| `final_figures_helpers.py` | plotting for the retired `old/22_final_figures.ipynb` |

## The two CCC generations — deliberately not merged

`ccc_helpers.py` (v1) and `ccc_utils.py` (v2) share 19 function names: `ccc_utils.py` is a
rewrite that resolves its defaults through a config module (`ccc_data_sub.py`) instead of module
constants (`ccc_data.py`). Both are live:

- `27_ccc_subtype` uses `ccc_utils` + `ccc_data_sub`.
- `ccc_axis.py` is a **third** thing and deliberately not a generation of these two: it does not use
  liana, does not rank by specificity, and reads the v2 degradome pseudobulk cube rather than the
  v1-era `data/ccc/ccc_skin.h5ad` that `ccc_utils` loads. Its numbers are not comparable to
  `tables/ccc40_*`, and `28` §10 says so in the run log.
- `degradome_helpers.py` uses `ccc_helpers` for six functions that have **no** `ccc_utils`
  equivalent: `stream_lognorm_subset`, `_write_testset`, `load_ccc_obs`, `read_source_index`,
  `read_source_var`, `resolve_rows` and `sanitize_obs`. (It no longer needs
  `build_ccc_celltype`: the v2 rewrite builds its CD4 split from `malignancy.py` instead.)

Retiring `ccc_helpers.py` + `ccc_data.py` therefore means porting those six into `ccc_utils.py`
and re-validating the degradome outputs. Open follow-up; not a rename.

`ccc_helpers`/`ccc_data` are also what the archived `old/ccc_v1_*` notebooks import.

## Added since this README was first written

Flagged by `jobs/run_knowledge_check.py`, which fails on a helper nobody documented — an
undocumented helper gets reimplemented by the next agent instead of reused.

| Module | What it is |
|---|---|
| `lineage.py` | **The** lineage call. Reads `nonmal_T_labels_v1.parquet`; `CD4_BENIGN`, `CD4_ANY`, `apply_lineage`, `arm_mask`. Never test `cell_type_final == "CD4"` anywhere else. |
| `bundle.py` | `write_bundle()` — the `reports/<name>/for_analysis/` output contract: generated INDEX.md + CSVs + report.md, binaries refused. |
| `clinical_meta.py` | The curated clinical layer: `clin_*` obs columns, identity merges, `clin_person_key`. |
| `cosmx_spatial.py` | CosMx spatial loading and transfer helpers. |
| `external_ref.py` | External reference datasets (surfaceome, healthy-skin degradome). |
| `pseq30_helpers.py` | Perturb-seq / pseudobulk DE plumbing, including the `DefaultInference(n_cpus=N)` wiring pydeseq2 needs on both `DeseqDataSet` and `DeseqStats`. |
| `target_pool.py` | Pooled stratification for the surface-target screens. |
