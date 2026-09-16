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
| `24_spatial_visium_download` | `notebooks/00_preprocess/03_spatial_visium_download.ipynb` |
| `45_atlas_v2_qc` | `notebooks/10_atlas/11_atlas_v2_qc.ipynb` |
| `32_atlas_descriptive` | `notebooks/10_atlas/12_atlas_descriptive.ipynb` |
| `41_skin_vs_blood_subclone_signaling` | `notebooks/10_atlas/13_skin_vs_blood_subclone_signaling.ipynb` |
| `46_annotate_new_data` | `notebooks/10_atlas/14_annotate_new_data.ipynb` |
| `25_resolvi_train` | `notebooks/10_atlas/15_spatial_resolvi_deconvolution.ipynb` |
| `10b_skin_reannotation` | `notebooks/20_skin/21_reannotation.ipynb` |
| `10c_skin_myeloid_fibro_reannotation` | `notebooks/20_skin/22_myeloid_fibro_reannotation.ipynb` |
| `30_malignant_annotation_tcr_cnv` | `notebooks/20_skin/23_malignancy_tcr_cnv.ipynb` |
| `31_subclone_tcr_signaling` | `notebooks/20_skin/24_subclone_tcr_signaling.ipynb` |
| `38_delta_gene_axis` | `notebooks/20_skin/25_delta_gene_axis.ipynb` |
| `43_transcriptome_malignancy_study_holdout` | `notebooks/20_skin/26_transcriptome_malignancy_holdout.ipynb` |
| `40_ccc_subtype` | `notebooks/20_skin/27_ccc_subtype.ipynb` |
| `42_tme_degradome` | `notebooks/20_skin/28_tme_degradome.ipynb` |
| `33_blood_reannotation` | `notebooks/30_blood/31_reannotation.ipynb` |
| `34_blood_malignant_annotation_tcr_cnv` | `notebooks/30_blood/32_malignancy_tcr_cnv.ipynb` |
| `35_blood_subclone_tcr_signaling` | `notebooks/30_blood/33_subclone_tcr_signaling.ipynb` |

## Retired at the reorg

| pre-reorg | now | why |
|---|---|---|
| `03_mrvi_replication` | `notebooks/old/03_mrvi_replication.ipynb` | Li-only v1 MrVI replication; its `data/cache/mrvi_ctcl_cache.h5ad` was deleted with it |
| `22_final_figures` | `notebooks/old/22_final_figures.ipynb` | re-rendered panels from the pre-v2 malignancy chain (old nb14/16/17/18) |

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
