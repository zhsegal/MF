# Where the data is

Everything lives under `data/` (~612 GB, on a volume that runs near capacity — check `df` before
writing anything large). **An analysis never modifies `data/`**; it reads from it and writes to
`reports/` and `tables/`. The exception is a job whose declared purpose is to build a data object.

Sizes are rounded. Read the caveat column before using an object.

## The atlas — `data/atlas_joint/`

| Object | Shape / size | What it is | Watch out |
|---|---|---|---|
| `joint_annotated.h5ad` | 2,157,693 × 42,347, **112 GB** | The whole atlas, both compartments, all cell types. Counts in `layers/raw_counts` (CSR, float64, integer-valued). | The **only full-gene source for blood** and the only object with **non-T cells**. Its `obs` has no `cell_type_final` — join by `cell_id` from the obs parquet. A full streamed pass costs ~5–7 min at ~2 GB; never load it whole. |
| `atlas_obs_full_v2.parquet` | 2,157,693 × 116 cols, 77 MB | Every obs column: identity, QC, TCR, malignancy, annotation, and 53 curated `clin_*` columns. | The fast way to answer anything that does not need expression. Carries a stale `has_tcr` for li2024 and an older atlas-level `is_malignant` — neither is the headline. |
| `skin_T_annotated.h5ad` | 539,916 × 42,347, 12 GB | Skin T compartment, **full gene space**. | — |
| `blood_T_annotated.h5ad` | 667,584 × **10,000**, 16 GB | Blood T compartment, **HVG subset**. | **Asymmetric with skin.** No `CD4` gene. 34% of blood UMIs fall outside it. See `CAVEATS.md`. |
| `skin_T_tcr_annotated_v4.h5ad` | 538,808 cells, 12 GB | Skin T with TCR; the input to the cluster-malignancy pipeline. | 1,083 gaydosik2023 cells have **empty count rows** — see `CAVEATS.md`. |
| `nonmal_T_labels_v1.parquet` | 576,188 rows | **The current lineage call** for every non-malignant T/NK cell: `cell_id, tissue, donor, sample_id, cluster_new, label_new`. | Read it only through `helpers/lineage.py`. Malignant cells are absent by construction. |
| `skin_T_malignancy_cluster_v1.parquet` / `blood_T_…` | per cell | **`mal_cluster` — the headline malignancy call.** Columns `cell_id, sample_id, cluster, call_source, mal_cluster, mal_cluster_prob`. | Frozen; see `DECISIONS.md`. Skin has 45,496 cells with no call. |
| `skin_T_clusters_v1.parquet` / `blood_T_…` | 538,808 / 667,584 | The per-sample Leiden clusters `mal_cluster` is called on, plus `top_trbv`, `top_trav`, `tcr_ok`, `family`. | Skin groups by `sample_id`, **blood groups by `donor`** — the blood column is still *named* `sample_id` but holds donor ids. |
| `skin_T_malignancy_v5.parquet` / `blood_T_malignancy_v4.parquet` | per cell | inferCNV + ALICE calls (`mal_cnv`, `mal_tcr`). | Revision is in the filename; check which one a result pinned. |
| `blood_T_celltype_corrections_v1.csv` | 1,716 cells | nb32b's 12 CD8→NK cluster corrections. | Applied **on read only**; the h5ad is not rewritten. Its "from" label is wrong for 66 rows. |
| `blood_subtypes.csv`, `skin_myeloid_fibro_subtypes.csv`, `skin_unassigned_subtypes.csv` | per cell | Non-T fine labels from the hand-annotation notebooks. | `Ambient_soup` / `LowQC` / `Doublet` are labels in here, not removals. |
| `joint_mrvi_input*.h5ad`, `cd4_mrvi_input.h5ad` | 1.3–34 GB | MRVI training inputs (HVG subsets) with `obsm/X_mrvi_u`, `X_mrvi_z`. | HVG subsets — never use for marker-level work. |
| `subclones_v4.parquet` | per cell | Subclone assignment. | Produced by an archived notebook; see `HISTORY.md`. |
| `tcr_clones.parquet` | per cell | TCR clonotypes. | — |

`obsm/X_adt` inside `joint_annotated.h5ad` is the only protein layer: 2,157,693 × 128 float32,
**NaN outside buus25**, names in `uns/adt_panel`. See `CAVEATS.md#adt-exists-for-exactly-one-dataset`.

## Raw cohorts — `data/<CODE>_<author><year>_<tissue>/`

One directory per study, `processed/standardized.h5ad` inside each. The codes are the donor-id
prefixes used throughout (`B5__PT09` is buus2025 patient PT09).

**Blood:** B1 borcherding2019, B2 borcherding2023, B3 borcherding2024, B4 geskin2026 (ex vivo —
see `CAVEATS.md`), B5 buus2025 (blood **and** skin; the only ADT), B6 ren2023, B7 harro2023,
B8 dorando2026.
**Skin:** D1 chennareddy2025, D2 johnson_timp2025, D3 brunner2024, D5 rindler2021frontimm,
D6 gaydosik2019, D7 rindler2021mc, D8 alkon2024, D9 lyp_vs_ctcl, D10 jonak2021, D11 brentuximab,
D12 pacritinib, D13 gaydosik2022, D14 gaydosik2023 (**empty count rows**), D15 il4ra_blockade,
D16 song2024, H herrera2021, `Li2024_atlas` (the largest skin cohort; partly 10x Flex).

Provenance, accessions, exclusions and the duplicate-library findings are in
`docs/DATA_PROVENANCE.md` and `docs/ATLAS_V2_DEDUP_GATE.md` (45 identity groups; GSE266862 and
GSE269981 are byte-identical libraries). `docs/DATA_PROVENANCE.md` still quotes **v1** cohort
counts in places — trust the atlas, not that document, for numbers.

## Other data trees

| Path | What |
|---|---|
| `data/targets/` | Long `(arm, donor, gene)` tables from the surface-target screens; `*_target_long.parquet` + manifest + meta. The screens' real output. |
| `data/degradome/skin_degradome.h5ad` | The 945k-cell skin degradome object. |
| `data/external_ref/` | Reynolds 2021 healthy skin, the surfaceome reference, the GWCD4i Perturb-seq DE (15.6 GB). |
| `data/clinical/` + `data/clinical_metadata_v1.csv` | The curated clinical layer: 265 samples, 194 persons. Decisions in `docs/CLINICAL_METADATA.md`. |
| `data/cache/` | Derived intermediates for MrVI jobs. Regenerable. |
| `data/_archive_v1/` | The v1 freeze, **read-only**. 251 GB. |
| `data/regimprint_nb31/` | Inputs for the regulator-imprint models (`docs/regimprint_v0.2/`). |

## Conventions

- **Donor identity**: `clin_person_key` → `patient_key` → `donor`. `donor` is dataset-prefixed and
  never collides; **`real_donor` does** — 15 values span studies.
- **Cell ids** are `<sample>|<barcode>`; in both T-compartment objects the obs index equals the
  `cell_id` column, and they match `nonmal_T_labels_v1.parquet` exactly (257,297 skin /
  318,891 blood, 100%).
- **Counts** live in `layers/raw_counts` (and `layers/counts` in the compartment objects);
  `X` is normalised in some objects and raw in others — always name the layer.
- **Artifact filenames encode the pre-reorg notebook number** (`tables/ccc40_*`, `figures/nb38_*`,
  `tables/deg28mc_*`, `tables/tgt49_*`, `tables/pay51_*`). They are read back by code — do not
  rename them. `HISTORY.md` decodes the numbers.
