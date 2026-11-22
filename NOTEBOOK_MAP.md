# Notebook map — pre-reorg id → current path

The project was reorganized on 2026-09-16: it moved out of
`scvi-tools-neural-nmf/notebooks/MF/` to its own top-level directory and repo, and the notebooks
were regrouped by analysis track and renumbered. Older prose — in `notebooks/old/`, in `docs/`,
in bsub logs, in lab notes — refers to notebooks by their pre-reorg number (`nb30`, `nb10b`, …).
This table decodes those.

## Live notebooks

| pre-reorg | now |
|---|---|
| `01_download_and_prelim` | `notebooks/00_preprocess/01_download_and_prelim.ipynb` |
| `10_build_joint_atlas` | `notebooks/00_preprocess/02_build_joint_atlas.ipynb` |
| `45_atlas_v2_qc` | `notebooks/10_atlas/11_atlas_v2_qc.ipynb` |
| `32_atlas_descriptive` | `notebooks/10_atlas/12_atlas_descriptive.ipynb` |
| `41_skin_vs_blood_subclone_signaling` | `notebooks/10_atlas/13_skin_vs_blood_subclone_signaling.ipynb` |
| `46_annotate_new_data` | `notebooks/10_atlas/14_annotate_new_data.ipynb` |
| `10b_skin_reannotation` | `notebooks/20_skin/21_reannotation.ipynb` |
| `10c_skin_myeloid_fibro_reannotation` | `notebooks/20_skin/22_myeloid_fibro_reannotation.ipynb` |
| `30_malignant_annotation_tcr_cnv` | `notebooks/20_skin/23_malignancy_tcr_cnv.ipynb` |
| `31_subclone_tcr_signaling` | `notebooks/20_skin/24_subclone_tcr_signaling.ipynb` |
| `38_delta_gene_axis` | `notebooks/old/25_delta_gene_axis.ipynb` (retired 2026-10-22) |
| `43_transcriptome_malignancy_study_holdout` | `notebooks/old/26_transcriptome_malignancy_holdout.ipynb` (retired 2026-10-22) |
| `40_ccc_subtype` | `notebooks/old/27_ccc_subtype.ipynb` (retired 2026-10-22) |
| `42_tme_degradome` | `notebooks/old/28_tme_degradome_v1.ipynb` (retired 2026-09-17; replaced by a v2 rewrite at the same live path) |
| `33_blood_reannotation` | `notebooks/30_blood/31_reannotation.ipynb` |
| `34_blood_malignant_annotation_tcr_cnv` | `notebooks/30_blood/32_malignancy_tcr_cnv.ipynb` |
| `35_blood_subclone_tcr_signaling` | `notebooks/old/33_subclone_tcr_signaling.ipynb` (retired 2026-10-22) |

## Retired at the reorg

| pre-reorg | now | why |
|---|---|---|
| `03_mrvi_replication` | `notebooks/old/03_mrvi_replication.ipynb` | Li-only v1 MrVI replication; its `data/cache/mrvi_ctcl_cache.h5ad` was deleted with it |
| `22_final_figures` | `notebooks/old/22_final_figures.ipynb` | re-rendered panels from the pre-v2 malignancy chain (old nb14/16/17/18) |
| `24_spatial_visium_download` | `notebooks/old/03_spatial_visium_download.ipynb` | Visium ingest; retired 2026-09-16 with the rest of the spatial arm |
| `42_tme_degradome` | `notebooks/old/28_tme_degradome_v1.ipynb` | v1 degradome: 749,510 skin cells, 7 studies, a 156-gene protease-only panel and a `skin_myeloid_fibro_subtypes.csv` that no longer exists. Retired 2026-09-17; `20_skin/28_tme_degradome.ipynb` is a v2 rewrite, not an edit. Artifacts in `data/degradome/_v1/`, `tables/_v1_deg42/`, `figures/degradome/_v1/`; its bsub drivers in `jobs/_v1/` |

## Removed after the reorg

| pre-reorg | why |
|---|---|
| `25_resolvi_train` | `10_atlas/15_spatial_resolvi_deconvolution` — CondSCVI→DestVI on the Li-2024 Visium sections. The spatial arm was dropped: the low-resolution Visium slides went unused and nothing downstream read `ctcl_visium_destvi.h5ad`. Deleted 2026-09-16; last version is in commit `8f660e4`. The ingest notebook `00_preprocess/03_spatial_visium_download` and `data/Li2024_atlas/visium/` remain. |

## Retired after the reorg (2026-09-30)

Moved to `notebooks/old/` under their live names. Their artifacts keep their prefixes.

| was | superseded by | why |
|---|---|---|
| `20_skin/28_tme_degradome` | `20_skin/28_tme_degradome_mc` | `mal_tcr` arms; the `_mc` copy uses `mal_cluster` (26b). Tables `deg28_*` |
| `20_skin/28b_tme_degradome_figures` | `20_skin/28b_tme_degradome_figures_mc` | same swap. Tables/figures `deg28b_*` |
| `20_skin/28c_amoeboid_motility` | `20_skin/28c_amoeboid_motility_mc` | same swap. Artifacts `amb28c_*` |
| `20_skin/28d_cosmx_degradome_spatial` | `20_skin/28f_cosmx_figures` | raw-count CosMx degradome; 28f is post-resolVI. Its lineage calls (§0–§1, writing `cosmx_s1_cells_28d.parquet` for resolVI) are now 28f §P. Artifacts `deg28d_*` |
| `20_skin/28e_cosmx_tcell_compartments` | `20_skin/28f_cosmx_figures` | T-cell compartments / transfer score / Harreman; 28f reads none of it. It remains the only reader of `cosmx_s1_cells_28e.parquet`, `run_cosmx_transfer.sh` and `run_cosmx_harreman.sh` without `--resolvi`. Artifacts `deg28e_*` (the `_rvi` Hotspot tables are 28f's) |

## Retired 2026-10-22 (the `mal_cluster` consolidation)

`mal_cluster` (26b) became the project headline malignancy call, and the skin track was cut back to
one annotation → malignancy chain. Moved to `notebooks/old/` under their live names unless noted.
Artifacts keep their prefixes and stay on disk.

| was | why |
|---|---|
| `20_skin/25_delta_gene_axis` | δ̂ clonality axis on the MrVI latent; the only cross-repo dependency (SemanticSCVI fork via `SCVI_NB_DIR`). Artifacts `nb38_*` |
| `20_skin/26_transcriptome_malignancy_holdout` | produced `mal_knn`, the MrVI label-propagation call. Superseded as headline by 26b; `cd4_transcriptome_malignancy_v1.parquet` stays on disk and `mal_knn` is still assembled by `helpers/malignancy.py` as a sensitivity arm. Artifacts `nb43_*` |
| `20_skin/27_ccc_subtype` | second-generation CCC. Built on `data/ccc/ccc_skin.h5ad`, a v1-sized object (727k cells), split on `mal_tcr_alice`. Artifacts `ccc40_*` |
| `20_skin/28b_tme_degradome_figures_mc` → `old/28b_tme_degradome_figures_mc_premerge` | merged into `28_tme_degradome_mc` as **Part II**. Artifacts `deg28bmc_*` unchanged |
| `20_skin/28c_amoeboid_motility_mc` → `old/28c_amoeboid_motility_mc_premerge` | merged into `28_tme_degradome_mc` as **Part III**. Artifacts `amb28cmc_*` unchanged |
| `30_blood/33_subclone_tcr_signaling` | blood subclone / TCR-signalling; outputs are from the v1 annotation and the blood track is scoped to annotation + malignancy for now. Artifacts `*_blood_v1` |

The pre-merge `28_tme_degradome_mc` has no separate copy: its cells **are** Part I of the merged
notebook, unchanged.

Numbers 25, 26, 27 and 33 are left free. They are not reused: artifact filenames encode the
notebook that wrote them, and reusing a number would desynchronize the two.

## Renamed inside `notebooks/old/`

The first-generation CCC pipeline used numbers 34–39, which were later reassigned to unrelated
live notebooks. It now carries an unambiguous prefix:

| was | now |
|---|---|
| `34_ccc_build_object` | `old/ccc_v1_01_build_object.ipynb` |
| `35_ccc_descriptive` | `old/ccc_v1_02_descriptive.ipynb` |
| `36_ccc_headline_figures` | `old/ccc_v1_03_headline_figures.ipynb` |
| `38_ccc_subtype_descriptive` | `old/ccc_v1_04_subtype_descriptive.ipynb` |
| `39_ccc_subtype_headline_figures` | `old/ccc_v1_05_subtype_headline_figures.ipynb` |
| `ccc_nb38_39_implementation_dossier.md` | `old/ccc_v1_implementation_dossier.md` |

All other `notebooks/old/*.ipynb` keep their original names and numbers.

## What was deliberately *not* renamed

Artifact filenames are data keys read back by code, so they still carry the old numbers:
`figures/nb38_*.png`, `figures/nb43_*.png`, `tables/ccc40_*.csv`, `tables/deg42_*.csv`,
`jobs/run_nb42.sh`. Renaming them would mean editing every reader. Decode them with the table
above.
