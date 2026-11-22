# MF — CTCL / Mycosis Fungoides multi-tissue atlas

> **This is no longer the entry point — read `KNOWLEDGE/README.md` first.**
>
> What is still accurate here: the atlas numbers, the malignancy-call policy, the cohort and
> donor-truth facts, and the label caveats. All of it is carried, with its blast radius, into
> `KNOWLEDGE/` (`PROJECT.md`, `DECISIONS.md`, `CAVEATS.md`, `DATA_MAP.md`).
>
> What is stale: the "reading order" sections describe notebooks that were retired on
> 2026-11-22 and are now under `notebooks/_archive/`. **Notebooks are never run.** The
> executable equivalent of each track is in `KNOWLEDGE/ANALYSES.md`. Also note this file says
> blood malignancy is v4 while the old `CLAUDE.md` said v3 — v4 is correct.

An integrated single-cell atlas of cutaneous T-cell lymphoma (MF / Sézary syndrome) built from
the Li–Haniffa 2024 skin atlas plus the published GEO/ArrayExpress cohorts, with paired TCR where
the deposit allows it. **Atlas v2** (rebuilt 2026-09-05) is 2,157,693 cells over 14 studies —
1,345,527 skin (174 samples) and 811,024 blood (89 samples). Modelling is **MRVI**
(`scvi.external.MRVI`); malignancy is called from TCR clonality and inferCNV, not from a marker
score.

Current objective: **surface targets for CAR-T / bispecific antibodies** that separate malignant
from reactive T cells in skin and blood (`notebooks/40_targets/`).

`notebooks/10_atlas/11_atlas_v2_qc.ipynb` is the release gate: it re-derives every number quoted
about the atlas and writes a PASS/WARN/FAIL ledger to `tables/atlas_v2_qc_report.csv`. Run it
after any rebuild.

## Layout

```
notebooks/00_preprocess/   download, standardize, build the joint atlas
notebooks/10_atlas/        whole-atlas QC, descriptives, skin↔blood comparison, external tool
notebooks/20_skin/         skin: annotation → malignancy → subclones → degradome
notebooks/30_blood/        blood: annotation → malignancy
notebooks/40_targets/      CAR-T / bispecific surface-target discovery
  41_surface_panel.ipynb   the shared roster: panel ∪ surfaceome, machinery, precedent
  skin/                    42 screen → 47 stratification → 48 pooled
  blood/                   43 screen → 44 stratification → 45 CITE-seq → 46 pooled
notebooks/old/             superseded notebooks, kept for provenance (see NOTEBOOK_MAP.md)
helpers/                   the analysis library imported by the notebooks
jobs/                      bsub drivers for anything too big for a kernel
docs/                      provenance, dedup gate, method notes; docs/archive/ = superseded
data/                      raw cohorts, built objects, caches  (git-ignored)
tables/  figures/  models/  benchmark_results/                 (git-ignored)
```

## Reading order

**00_preprocess** — `01_download_and_prelim` (Li portal objects, QC) →
`02_build_joint_atlas` (concat + dedup + TCR clone table).
The v2 cohort-onboarding path is `data/build_samples_tsv_generic.py` →
`data/finalize_samples.py` → `data/append_v2_metadata.py` → `jobs/run_build_joint.py`, with the
overlap gate in `helpers/check_overlap.py` → `helpers/make_dedup_decisions.py`
(see `docs/ATLAS_V2_DEDUP_GATE.md`).

**10_atlas** — `11_atlas_v2_qc` (gate) → `12_atlas_descriptive` (counts, MrVI UMAPs) →
`13_skin_vs_blood_subclone_signaling` → `14_annotate_new_data` (self-contained notebook a
collaborator runs on *their* data).

**20_skin** — `21_reannotation` → `22_myeloid_fibro_reannotation` →
`22b_unassigned_reannotation` → `23_malignancy_tcr_cnv` (ALICE TCR + inferCNV, revision v5) →
`26b_cluster_malignancy` (the headline call) → `24_subclone_tcr_signaling` →
`28_tme_degradome_mc` (build + figures + motility, merged) → `28f_cosmx_figures`
(CosMx in situ, post-resolVI) → `29_perturbseq_reverse_engineering` (the MF malignant-vs-reactive CD4
signature modelled as a linear combination of genome-scale CD4 Perturb-seq knockdown signatures — Zhu,
Dann et al. Cell 2026, Stim48hr, their `pert2state` model vendored in `helpers/pert2state_model.py` —
then crossed with the v5 inferCNV arm calls; inputs via `jobs/download_perturbseq_gwcd4i.sh` →
`jobs/run_perturbseq_prep.sh`).

Malignancy is exactly four calls, assembled by `helpers/malignancy.py` and never OR-ed:
`mal_tcr` (ALICE clonality), `mal_cnv` (inferCNV), `mal_knn` (label propagation on the MrVI
latent) and `mal_cluster` (26b per-sample Leiden). **`mal_cluster` is the headline since
2026-10-22** — it is defined on every T type rather than CD4 only, and it is anchored per sample
on a sequence fact. `mal_knn` is now a sensitivity arm; the notebook that produced it
(`26_transcriptome_malignancy_holdout`) is in `notebooks/old/` and its parquet stays on disk.

The `_mc` notebooks take their malignant arms from `mal_cluster`; the `mal_tcr` originals
(28, 28b, 28c) and the pre-resolVI CosMx notebooks (28d, 28e) were moved to `notebooks/old/` on
2026-09-30 (`NOTEBOOK_MAP.md`). `28_tme_degradome_mc` is scoped to **MF + HC skin** — malignancy is
annotated for MF and not for SS or CTCL_other — and works at **single-cell** level on log1p CP10K,
with study-blocked tests and per-donor medians overlaid on every violin. On 2026-10-22 it absorbed
`28b` (figures) and `28c` (amoeboid motility) as Parts II and III; each part's §0 is self-contained,
so a figure can be redrawn without running the build.

Retired 2026-10-22 (see `NOTEBOOK_MAP.md`): `25_delta_gene_axis`,
`26_transcriptome_malignancy_holdout`, `27_ccc_subtype`. The numbers 25–27 are deliberately left
free rather than reused — artifact filenames encode the notebook that wrote them.

**30_blood** — `31_reannotation` → `31b_subtype_reannotation` (non-T subtypes in two rounds, the
off-tumour denominator; all 143,440 non-T cells labelled) → `31c_ambient_rescue` (whether the
25,791 `Ambient_soup` cells can be rescued — verdict 2026-10-22: no, they stay `Ambient_soup`) →
`32_malignancy_tcr_cnv` (ALICE + inferCNV, **v4 = rebuilt on the v2 annotation**) →
`32b_cluster_malignancy` (`mal_cluster` for blood, plus the cell-type QC below).
`33_subclone_tcr_signaling` was retired on 2026-10-22.

`32b` also writes `data/atlas_joint/blood_T_celltype_corrections_v1.csv`: `nb31` labelled CD8 on
`CD8A`, which NK cells express as a homodimer, so cytotoxic NK clusters arrived as CD8 (the object
holds only 606 NK in 667,584 cells). The file corrects the 12 clusters the malignancy review
examined and `cluster_malig_helpers.read_obs` overlays it on `cell_type_T` for every consumer;
~100 further CD8-labelled clusters carry the same signature and are **not** corrected, so the
blood CD8 and NK arms are mixtures until `nb31` is redone.

The two cluster-malignancy jobs are tissue-parametrized — `run_cluster_malig{,_model}.sh
--tissue blood` — so skin and blood share one implementation. Blood clusters per **donor**, not
per `sample_id`: the geskin `sample_id` is an HTO lane pooling 5–6 donors.

**40_targets** — surface targets for CAR-T / bispecific antibodies, the project's current
objective. `41_surface_panel` is the shared roster (curated panel ∪ surfaceome, trafficking
machinery, clinical precedent); from there the two compartments run as parallel series in
`skin/` and `blood/`, built on the same template so a result in one can be read against the
other:

| | skin | blood |
|---|---|---|
| screen | `42_target_screen_skin` | `43_target_screen_blood` |
| stratification + heterogeneity | `47_skin_stratification_heterogeneity` | `44_blood_stratification_heterogeneity` |
| pooled + stratified | `48_skin_pooled_stratification` | `46_blood_pooled_stratification` |
| measured protein | — (no CITE-seq in skin) | `45_blood_protein_trafficking` |

Both screens do the same five axes (prevalence, selectivity vs same-donor reactive T,
on-target off-tumour, surface plausibility, escape risk) behind the same two gates: a **cohort
fix** that turns donor keys into independent patients, and **six artifact tests** against an
empirical contamination null that is allowed to refuse to fire. The nulls differ because the
contaminants do — monocyte, platelet, erythrocyte and B-cell transcripts in blood; keratin,
collagen, endothelial, melanocyte, mast and myeloid transcripts in skin. `43 §8` compares the
compartments. Each notebook's last cell prints its complete numerical result as one block to
paste into another model, and writes it to `tables/tgt4N_llm_export.md`.

The compartments are not symmetric, in both directions: the blood object is a 10,000-gene HVG
subset and the skin object carries all 42,347, so `CD4` and `TNFRSF8` are testable only in skin;
but the skin screen can see the whole tissue in its off-tumour axis, where blood has the T and
NK arms only.

Inputs are built by `jobs/download_surfaceome.sh` → `jobs/build_surfaceome_ref.py` and
**`jobs/run_target_screen_qc.sh --tissue {skin,blood}`**, which supersedes the v1
`run_target_screen.sh` in both compartments: true-library CP10K, robust location statistics, a
contamination-filtered recompute, depth matching, per-cell couplings to that tissue's
contaminant modules, and the lineage-control genes. `--call mal_tcr` rebuilds the same tables
with the arms taken from the TCR clone, the control for the malignancy call being a cluster
label while the comparison is between those same clusters. `jobs/run_target_pool_hist.sh
--tissue {skin,blood}` writes the per-group histograms behind the pooled AUROC in 46/48, and
`jobs/run_adt_cells.sh` writes the per-cell CITE-seq table 45 reads.

**The screen ranks transcripts; it cannot call a target.** CCR4, TNFRSF8, KIR3DL2 and IL2RA are
approved or registrational in CTCL and serve as positive controls — a ranking that does not
recover them is miscalibrated, which is exactly what the v1 skin pass did (worst control 1,968
of 2,115) and why 42 was rebuilt.

The cohort facts are handled by `helpers/target_cohort.py`, not by hand. **Blood:** 54 of the
131 donor keys are geskin2026 lane×HTO wells (ex vivo 2-day drug cultures of the same few
patients), and five more are healthy donors carried under `disease == "SS"` whose CD4 cells
therefore sat in the *reactive* arm — 52 independent paired donors remain. **Skin:** of 146
keys, 19 carry a clone that is not CD4 (CD8, γδ or DN by `clin_phenotype`, plus the pcAECTCL and
CD8-aggressive cases) and so have no malignant CD4 arm at all, 15 carry a diagnosis outside the
MF/SS spectrum (LyP, SPP, LPP), and ten patients were biopsied more than once — 109 independent
keys, 107 people remain.

Blood alone can be checked against protein, but only for 17 buus2025 donors (15 SS, 2 MF by
`clin_dx`), `NaN` everywhere else — and a third of the 128 markers sit at isotype-control level,
so the panel is gated before use. Skin has no protein layer at all.

## Where the atlas lives

| object | path |
|---|---|
| full v2 atlas (2.16 M × 42,347) | `data/atlas_joint/joint_annotated.h5ad` (112 GB — never `sc.read_h5ad` it) |
| obs only | `data/atlas_joint/atlas_obs_full_v2.parquet` (77 MB — use this) |
| MrVI inputs | `data/atlas_joint/joint_mrvi_input{,_skin,_blood}.h5ad` |
| annotated compartments | `data/atlas_joint/{skin,blood}_T_annotated.h5ad` |
| v1 freeze | `data/_archive_v1/` — read-only, do not write |

## Running things

- Kernel: `neural_nmf_env`. Notebooks resolve the project root themselves (`NB_DIR`) and add
  `helpers/` to `sys.path`, so they run from wherever Jupyter was started.
- Anything heavy (atlas rebuild, MrVI training, inferCNV, DE) goes through `jobs/*.sh` (bsub,
  up to ~200 GB RAM). **The login node is fragile — do not run these on it.**
- `helpers/delta_axis_helpers.py` reaches into the SemanticSCVI fork for three modules; point
  `SCVI_NB_DIR` at that checkout if it moves.

## See also

- `NOTEBOOK_MAP.md` — pre-reorg notebook numbers → current paths.
- `docs/DATA_PROVENANCE.md` — which cohort came from where, and what was excluded and why.
- `helpers/README.md` — what each module is for, and the two CCC generations.
