# Spatial-transcriptomics data map — CTCL/MF atlas

Where spatial data lives across the atlas and the catalogued MF/SS studies, what is actually
usable, and how to fetch it. Written to open the spatial arm of the project (companion to
`00_preprocess/03_spatial_visium_download.ipynb`).

## TL;DR

**Only the Li 2024 Visium (10x, 23 skin sections) is genuine spot-level spatial
transcriptomics that pools with the scRNA atlas.** Everything else catalogued as "spatial" is
either ROI-level (GeoMx DSP) or protein-only imaging (MELC) — kept here for completeness, not
acted on. There is **no processed Visium h5ad** published; the data is per-section spaceranger
outputs on EMBL-EBI BioStudies (verified 2026-07-12), so we ingest with `sc.read_visium` +
concatenate rather than downloading one object.

## The map

| Source | Modality | Poolable with scRNA atlas? | Accession / repo |
|---|---|---|---|
| **Li/Strobl/Poyner 2024 (Haniffa), Nat Immunol** | **10x Visium**, 23 sections (8 CTCL + 15 healthy, 15 donors) | ✅ spot-level, atlas gene space | **E-MTAB-13614** (CTCL) + **E-MTAB-14559** (healthy) |
| Li 2024 companion | 10x 5′ scRNA + VDJ (**not spatial**) | n/a — already the scRNA atlas | E-MTAB-12303 |
| Danielsen 2024 (Front Oncol) | GeoMx DSP-WTA, ROI-level, 15 pts | ❌ ROI counts (epidermis/basal/CD4-T/Pautrier regions), not spots | "see paper" (GEO/Suppl.) |
| Choi/Park 2025 (Blood Adv) | GeoMx DSP-WTA, ROI-level, 27 pts (largest MF spatial cohort) | ❌ ROI counts | "see paper" (GEO/Suppl.) |
| Folliculotropic vs classic MF 2025 (JID) | GeoMx DSP-WTA, ROI-level | ❌ ROI counts | Supplement |
| Amechi 2025 (JID) | GeoMx DSP-WTA + spatial proteomics, ROI-level | ❌ ROI counts | Supplement |
| Sarkar 2024 (npj Syst Biol Appl) | MELC multiplex protein imaging (≥35 ch, **no RNA**) | ❌ protein only | Zenodo `10.5281/zenodo.11125482` |

The GeoMx/MELC rows are in `CTCL_MF_SS_dataset_availability.csv` (rows 12–16). They are **not
poolable**: GeoMx gives region-of-interest pseudobulk (not spots/cells), and MELC is antibody
imaging with no transcriptome. If a DSP arm is wanted later it must be its own
platform-homogeneous analysis, not merged with Visium/scRNA.

## Li 2024 Visium — what's actually on the repository

Confirmed against the EMBL-EBI BioStudies file manifests on 2026-07-12 (there is **no** processed
h5ad on the WebAtlas/Sanger CDN — all `cellatlas-ctcl.cog.sanger.ac.uk/*.h5ad` probes returned 404).

**E-MTAB-13614 — CTCL skin Visium** (ENA raw `ERP155847`). 8 sections, one spaceranger tarball each:

```
CTCL1_spaceranger_output.tar … CTCL8_spaceranger_output.tar   (~214–376 MB each, uncompressed .tar)
E-MTAB-13614.sdrf.txt / .idf.txt                              (sample metadata)
```

SDRF metadata per section: `Characteristics[disease]=Cutaneous T-cell lymphoma`, sex, age,
`organism part=skin of body`, ENA sample/run IDs, FASTQ URIs.

**E-MTAB-14559 — healthy skin Visium** (ENA raw `ERP165340`). 15 sections, one gzipped tarball each:

```
WS_D_SKNsp<id>.tar.gz   (15 files, ~273–557 MB each)   e.g. WS_D_SKNsp10264994.tar.gz
E-MTAB-14559.sdrf.txt / .idf.txt
```

`WS_D_SKN` = whole-skin dermis; `sp<id>` is the Sanger sample id. These are the 15 healthy
references (control tissue for the CTCL niche comparison).

Each tarball unpacks to a standard **spaceranger `outs/`** layout: `filtered_feature_bc_matrix/`
(or `.h5`), `spatial/` (`tissue_positions*.csv`, `scalefactors_json.json`, hires/lores PNGs) →
directly loadable with `scanpy.read_visium`.

## How to fetch (verified URLs)

Download pattern (BioStudies files endpoint, all return HTTP 200):

```
https://www.ebi.ac.uk/biostudies/files/E-MTAB-13614/<file>
https://www.ebi.ac.uk/biostudies/files/E-MTAB-14559/<file>
```

File lists come from the study JSON:

```bash
curl -sL "https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-13614" | \
  python -c "import json,sys; d=json.load(sys.stdin); [print(f['path'],f.get('size')) for f in d['section']['files']]"
```

Use `wget -c` / `aria2c -c` (resume-safe) into `data/Li2024_atlas/visium/`. Total ~2.4 GB (CTCL) +
~6 GB (healthy) ≈ **8.5 GB** across 23 tarballs — run in the background, not a blocking login-node
call. FTP mirror also available: `ftp://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/614/E-MTAB-13614/Files/`.

**Fallback (not needed):** raw FASTQs under ENA `ERP155847` / `ERP165340` would require spaceranger
re-alignment — only if a spaceranger `outs/` tarball is ever missing.

## Downstream modeling target

**resolVI** (`scvi.external.RESOLVI`, present in the fork at `src/scvi/external/resolvi/`).
Ingest is structured resolVI-ready (raw counts layer, `X_spatial` obsm, per-section batch key).

⚠️ **Resolution caveat:** resolVI is designed for *single-cell-resolved* spatial data; Visium
spots are multi-cell, so its neighbor-diffusion denoising is a partial fit. Ingest documents this;
the modeling decision is a later step. The atlas's RareCyte IF is protein imaging, not a
single-cell-transcriptomic substrate either.

## Sample provenance note

No published per-donor spatial↔scRNA crosswalk exists — only aggregate counts (23 sections =
8 CTCL + 15 healthy from 15 donors). Visium libraries have **no VDJ** and were not crosswalked to
atlas donors (`data/Li2024_atlas/tcr/vdj_accession_resolution.txt`). Section-level metadata
(disease/sex/age) comes from the SDRF files above.
