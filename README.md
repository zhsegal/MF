# MF — CTCL / Mycosis Fungoides multi-tissue atlas

An integrated single-cell atlas of cutaneous T-cell lymphoma (MF / Sézary syndrome) built from
the Li–Haniffa 2024 skin atlas plus the published GEO/ArrayExpress cohorts, with paired TCR where
the deposit allows it. **Atlas v2** (rebuilt 2026-09-05) is 2,157,693 cells over 14 studies —
1,345,527 skin (174 samples) and 811,024 blood (89 samples). Modelling is **MRVI**
(`scvi.external.MRVI`); malignancy is called from TCR clonality and inferCNV, not from a marker
score.

`notebooks/10_atlas/11_atlas_v2_qc.ipynb` is the release gate: it re-derives every number quoted
about the atlas and writes a PASS/WARN/FAIL ledger to `tables/atlas_v2_qc_report.csv`. Run it
after any rebuild.

## Layout

```
notebooks/00_preprocess/   download, standardize, build the joint atlas, ingest Visium
notebooks/10_atlas/        whole-atlas QC, descriptives, skin↔blood comparison, external tool
notebooks/20_skin/         skin: annotation → malignancy → subclones → CCC → degradome
notebooks/30_blood/        blood: annotation → malignancy → subclones
notebooks/old/             superseded notebooks, kept for provenance (see NOTEBOOK_MAP.md)
helpers/                   the analysis library imported by the notebooks
jobs/                      bsub drivers for anything too big for a kernel
docs/                      provenance, dedup gate, method notes; docs/archive/ = superseded
data/                      raw cohorts, built objects, caches  (git-ignored)
tables/  figures/  models/  benchmark_results/                 (git-ignored)
```

## Reading order

**00_preprocess** — `01_download_and_prelim` (Li portal objects, QC) →
`02_build_joint_atlas` (concat + dedup + TCR clone table) → `03_spatial_visium_download`.
The v2 cohort-onboarding path is `data/build_samples_tsv_generic.py` →
`data/finalize_samples.py` → `data/append_v2_metadata.py` → `jobs/run_build_joint.py`, with the
overlap gate in `helpers/check_overlap.py` → `helpers/make_dedup_decisions.py`
(see `docs/ATLAS_V2_DEDUP_GATE.md`).

**10_atlas** — `11_atlas_v2_qc` (gate) → `12_atlas_descriptive` (counts, MrVI UMAPs) →
`13_skin_vs_blood_subclone_signaling` → `14_annotate_new_data` (self-contained notebook a
collaborator runs on *their* data) → `15_spatial_resolvi_deconvolution`.

**20_skin** — `21_reannotation` → `22_myeloid_fibro_reannotation` → `23_malignancy_tcr_cnv`
(ALICE TCR + inferCNV, currently at revision v5) → `24_subclone_tcr_signaling` →
`25_delta_gene_axis` → `26_transcriptome_malignancy_holdout` → `27_ccc_subtype` →
`28_tme_degradome`.

**30_blood** — `31_reannotation` → `32_malignancy_tcr_cnv` → `33_subclone_tcr_signaling`.

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
