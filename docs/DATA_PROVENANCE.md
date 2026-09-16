# MF / CTCL atlas — dataset & paper provenance

**Purpose:** handoff for an independent reviewer checking whether any CTCL dataset was missed,
mis-sourced, double-counted, or silently dropped. **Scope:** CTCL/MF/SS patient cohorts only
(external reference DBs, gene sets, embeddings deliberately out of scope). **Date:** 2026-09-05; **revised 2026-09-05** after an accession-level re-audit against the GEO
FTP `filelist.txt` endpoints and the EBI BioStudies API (see `ATLAS_HANDOFF_FOR_CLAUDE_CODE.md`
§0 for what changed and why).
Numbers are **as realized in the built objects**, not as printed in the source papers.

Source-of-truth files: `data/atlas_joint/README.md`, `data/atlas_joint/metadata_schema.md`,
`docs/archive/DOWNLOAD_REPORT_CTCL_datasets.md`, `docs/archive/DOWNLOAD_remaining_CTCL_datasets.md`,
`CTCL_MF_SS_dataset_availability.csv` (27-row candidate landscape, swept 2026-06-03),
`data/INVENTORY.tsv`, `data/<LABEL>/meta/meta.json`, `tables/atlas_descriptive_*`,
`data/Li2024_atlas/tcr{,_pt}/RUN_LOG.md`, `docs/SPATIAL_data_map.md`.

**Per-sample manifest:** `tables/atlas_samples_manifest.csv` — all 149 samples, one row each,
generated from `data/atlas_joint/atlas_obs_full.parquet` (obs export of `joint_annotated.h5ad`,
the newest built atlas, 2026-07-21/26).

## 0 · Two-layer structure (read first)

The corpus is **nested**. The Li 2024 atlas is itself an integration of 4 sub-cohorts; we then
joined 9 further standalone GEO cohorts on top. Double-counting risk lives at both layers.

- Layer 1 = `CTCL_all_final_portal_tags.h5ad` (Li 2024 published post-QC object, 419,579 cells).
- Layer 2 = `data/atlas_joint/joint_annotated.h5ad` — Layer 1 + 9 GEO cohorts = **1,173,694 cells**.
- Built by `00_preprocess/02_build_joint_atlas.ipynb`; logic in `atlas_join_helpers.py`.

---

## 1 · Cohorts IN the joint atlas (10 members)

Realized counts from `data/atlas_joint/atlas_obs_full.parquet` (1,173,694 rows). Totals:
**149 samples · 170 donor-units · 267,764 TCR+ cells.** Compartments: Skin 749,510 ·
Blood 423,042 · LN 1,142. Disease: MF 549,879 · SS 445,211 · CTCL_other 146,589 · HC 32,015.

| code | `dataset` | paper | accession | repo | tissue | donors | samples | cells (post-QC) | TCR+ cells | modality | access |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Li2024_atlas | li24 | Li/Strobl/Poyner (Haniffa) 2024, Nat Immunol 25:2320 | E-MTAB-12303¹ | ArrayExpress | Skin (Derm/Epi) | 36 | 36 | 419,579 | 0¹ | 10x 5′ + 10x Flex | open |
| B4 | geskin26 | Geskin/Gaydosik/Fuschiotti 2026 (dupilumab), Cancer Immunol Res | GSE290850 | GEO | Blood | 65² | 11 lanes | 260,455 | 69,386 | 10x 5′ + TCR + HTO + 19-ADT | open |
| D1 | chennareddy25 | Chennareddy 2025, Br J Dermatol | GSE266862 | GEO | Skin | 22 | 22 | 191,147 | 61,955 | 10x 5′ + TCR | open |
| B5 | buus25 | Buus 2025, Cancer Discov (PMC12498100) | GSE284075 (PRJNA1196956) | GEO | Blood + Skin | 17³ | 40 | 127,861 | 70,169 | 10x 5′ + CITE 127-plex ADT + TCR | open (processed only; FASTQ withheld) |
| B2 | borcherding23 | Borcherding 2023, Blood Adv | GSE146586 | GEO | Blood | 5 | 5 | 42,642 | 36,062 | 10x 5′ + TCR | open |
| D6 | gaydosik19 | Gaydosik 2019, Clin Cancer Res | GSE128531 | GEO | Skin | 9 | 9 | 39,260 | 0 | 10x 3′ (no TCR) | open |
| D3 | brunner24 | Brunner 2024, JACI | GSE269981 | GEO | Skin | 5⁴ | 5⁴ | 37,883⁴ | 9,633 | 10x 5′ + TCR | open |
| H | herrera21 | Herrera 2021, Blood | GSE171811 | GEO | Blood + Skin | 8 | 14 | 32,002 | 17,742 | ECCITE-seq (RNA+49-ADT+TCRαβ/γδ+HTO) | open |
| B1 | borcherding19 | Borcherding 2019, Clin Cancer Res | GSE124899 | GEO | Blood | 2 | 4 | 14,253 | 0 | 10x 3′ | open |
| D5 | rindler21 | Rindler 2021, Front Immunol | GSE165623 | GEO | Skin+Blood+LN | 1 | 3 | 8,612 | 2,817 | 10x 5′ + TCR | open |

Sum = 1,173,694 ✓ (exact parquet row count).

¹ **Accession corrected 2026-09-05.** BioStudies titles resolve the three E-MTABs
unambiguously: `E-MTAB-12303` = *"Single cell RNA-seq of skin biopsies from patients with
CTCL"* (this cohort), `E-MTAB-13614` = 8 CTCL Visium sections, `E-MTAB-14559` = 15 healthy
Visium sections (§5). The doc previously credited `E-MTAB-14559` here. Note the object we
actually load (`CTCL_all_final_portal_tags.h5ad`) is Li's **processed portal file**, not the
ArrayExpress deposit — `E-MTAB-12303` is the accession of record, not the download source.
li24 `has_tcr=0` in atlas obs **by construction** — Li TCR was recovered separately (§4) and is
joined downstream by barcode in `23_malignancy_tcr_cnv` from `li2024_tcr_malignancy.parquet` (118,082 TCR+ cells).
**Known cross-table disagreement; not a data loss.**
² geskin26 "donors" are `lane_HTO` pseudo-donors after HashSolo demux — the **HTO→patient key was
never deposited on GEO**, so real patient identity is unrecoverable. Chain: 373,197 → 270,992
singlets → 260,455 post-QC.
³ B5 accession has 23 patients; 6 (PT03, PT11, PT12, PT17, PT19, PT20) are culture-only
(S. aureus enterotoxin / romidepsin timecourses, 4 h–294 d) and were excluded → 17 primary ex-vivo.
⁴ post-dedup vs D1 (see §2).

### 1a · Skin-only working atlas (`joint_mrvi_input_skin.h5ad`, `12_atlas_descriptive`)
749,510 cells · 7 studies · 82 donors · 99 samples · 12.76 % TCR+ · 6.43 % malignant.
Per study (cells / donors / samples): li2024 419,579/36/36 · chennareddy2025 191,147/22/22 ·
buus2025 55,483/3/20 · gaydosik2019 39,260/9/9 · brunner2024 37,883/5/5 ·
rindler2021_fi 4,288/1/1 · herrera2021 1,870/6/6.

### 1b · Blood working cohort (`31_reannotation`–35)
**Scope rule (was undocumented):** the blood arm is **T-cell restricted** — `31_reannotation` subsets the
423,042-cell Blood compartment to `cell_type_broad == "T"` (plus `INCLUDE_LN=False`, so the
1,142 LN cells are out). The 120,680-cell difference is that subset, not a loss: per-study
drops are geskin26 93,305 · buus25 10,787 · herrera21 9,023 · borcherding23 5,707 ·
rindler21 1,781 · borcherding19 77, which sum to 120,680 exactly. The **skin** working atlas
(§1a) is all cell types, which is why the two arms are not comparable head-to-head.

302,362 cells · 167,489 TCR+ · 98 donor-units, **53 pass the eligibility gate**.
Per study (units / cells / TCR+): geskin2026 65/167,150/67,022 · buus2025 17/61,591/47,599 ·
borcherding2023 5/36,935/35,014 · herrera2021 8/21,109/16,598 · borcherding2019 2/14,176/0 ·
rindler2021_fi 1/1,401/1,256.

### 1c · Li 2024 internal composition (36 donors — the nested layer)
| sub-cohort | origin paper | donors | cells | TCR+ |
|---|---|---|---|---|
| Sanger_Ncl_Fresh (CTCL1–8) | newly generated, Li 2024 | 8 | 277,158 | 105,057 |
| PKU (MF14–MF30) | Liu 2022, Nat Commun | 10 | 60,164 | 0 |
| Sanger_Ncl_FFPE (CTCL9–18) | newly generated, Li 2024 (10x Flex) | 10 | 49,634 | 0 |
| MDA (PT11–PT56) | Song 2022, Cancer Discov | 8 | 32,623 | 13,025 |

**20 of 36 Li donors have no TCR** — reasons in §4.

---

## 2 · Fetched but excluded / deduplicated

| item | accession | reason (verbatim) |
|---|---|---|
| **B3 Borcherding 2024** (moga + IFN) | GSE192836 | "453-gene targeted panel (not whole-transcriptome) AND its `*_tcr.csv` holds only a `TCR_Paired_Chains` boolean — **no CDR3 sequences** → no clones." Downloaded (18 GSM, 4.9 MB), kept on disk for record only. |
| **D2 Johnson/Timp 2025** (JHU CAF preprint, medRxiv) | none | "expression profile data … pending submission to GEO." No GSE/PRJNA/Zenodo anywhere in the preprint. Dir scaffolded, `raw/` + `processed/` empty; nb08_d2 cannot execute past load. |
| **D1↔D3 duplicate libraries** | GSE266862 / GSE269981 | Same Brunner libraries reused in both deposits for `P112, P115, P116, P121, P303, PGS`; P112 confirmed **byte-identical** (n=6,400, 100 % barcode overlap). D1's copy kept, 25,553 D3 cells dropped → brunner24 63,436→37,883, 11→5 samples. |
| **B5 cultured arms** | within GSE284075 | Perturbation timecourses excluded; 6 culture-only patients dropped (see ³ above). |
| **B5 WES/WGS layer** | — | "GSE284075 is scRNA-seq/CITE-seq only. The paper's WES/WGS (5 WES, 3 WGS) is not in this accession." |
| Li's 3 absorbed public studies | — | Liu 2022, Rindler *Mol Cancer* GSE173205, Song 2022 **not downloaded separately** → no expression double-count. |
| ~~D4 Rindler 2021 *Mol Cancer*~~ | ~~GSE173205~~ | **RETRACTED 2026-09-05 — the exclusion rationale was false.** `GSE173205` was cited by Li as a comparison dataset but never integrated: §1c accounts for all 36 Li donors as 8+10+10+8 = 419,579 cells with no Rindler sub-cohort. Moved to ingest as cohort **D7** — but it is **not** +10 patients: 4 of its 27 GSMs are the healthy controls `P112/P115/P116/P121` we already hold in D1, and `MF311/MF312/MF318` carry the same thin/thick design as D1's `P311/P312/P318_thin,_thick` (Vienna renames `MF###`↔`P###` between deposits). Net ≈7 new patients, pending the barcode gate. **No V(D)J in the deposit.** |
| Li 2024 ArrayExpress deposits | E-MTAB-12303, E-MTAB-13614 | **Corrected 2026-09-05.** `E-MTAB-12303` *is* the scRNA accession of record (§1) — its GEX FASTQ/CellRanger layer was not re-processed because the published processed object was used instead, but its **31 CellRanger vdj libraries were mined for V(D)J** (§4). `E-MTAB-13614` = the 8 CTCL Visium sections (§5). |

Donor namespacing (`Li__CTCL1` vs `B2__CTCL2`) resolves the `CTCL1–18` ↔ `CTCL2–6` ID collision.
B1 ⊂ B2 overlap handled: GSE124899 downloaded once under B1; B2 holds only the new GSE146586.

---

## 3 · Not obtained — with reason

**Controlled access (application required, cannot script)**
| study | accession | repo | note |
|---|---|---|---|
| Liu 2022, Nat Commun (MF skin scRNA+scTCR+WES) | HRA000166 | NGDC GSA-Human | scRNA already inside the Li atlas as sub-cohort PKU; **only its V(D)J and raw WES are lost** |
| Xue 2022, Cell Death Dis (SS, only scATAC in the corpus) | HRA000847 / 000826 / 000145 | GSA-Human | GSA-Human request, Section 7 |
| Su T, Duran GE, … Khodadoust MS 2022, OncoImmunology 11:2115197 (**Stanford**; pembrolizumab, 118,961 T cells) | phs002933.v1.p1 | dbGaP | "dbGaP authorized-access application (eRA Commons + DAC)" |
| Srinivas 2024, Front Oncol (MF skin, 18,630 T cells) | not stated | EGA (presumed) | "EGA Data Access Committee request" |
| Peiffer 2023, Front Oncol (SS blood+skin, n=1) | not stated | EGA (presumed) | Becker group; confirm code |
| Zhao 2025, Front Immunol (MF CAF) | HRA007111 | GSA-Human | tier not verified — Open vs Controlled unchecked |

**No deposit at all** — Du 2022, Cancer Letters: "deposited nothing — data only in article/supplement."

**Deposit pending** — Dorando/Payton 2025 (bioRxiv 2025.02.11.637715; 35 pts, 114 serial samples,
most multi-omic CTCL set to date); Childs 2026 JEADV in-house fraction (14/22 patients are reused
public data → overlap risk); D2 Johnson/Timp (§2).

**Wrong data type (deliberate)**

*Spatial exclusions are two different rules — do not conflate them (corrected 2026-09-05):*
- **ROI-level pseudobulk → exclude.** GeoMx DSP-WTA measures whole-transcriptome over hand-drawn ROIs, so a "spot" is a pseudobulk of hundreds of cells. Out of scope.
- **Single-cell-resolution imaging spatial → IN SCOPE, tracked separately.** CosMx SMI / MERFISH resolve individual cells and are compatible with the atlas's cell-level schema; they are kept on the spatial track (alongside Visium) rather than joined into the expression object. `GSE337336` (early-MF fibroblast microenvironment, 16 CosMx samples, `exprMat`/`fov_positions`/`metadata` per patient) is the first member. Jung 2025 CosMx belongs here, **not** under the GeoMx rule where it was previously filed.

- GeoMx DSP-WTA, ROI-level pseudobulk not spots/cells — Danielsen 2024 (15 pts), Choi/Park 2025
  (27 pts, largest MF spatial cohort), folliculotropic-vs-classic JID 2025, Amechi 2025
  (skin-of-color). None publish an accession ("see paper / Supplement").
- Sarkar 2024 npj Syst Biol Appl — MELC multiplex **protein** imaging, no RNA. Zenodo
  10.5281/zenodo.11125482, open, not fetched.
- Buus 2018 Blood Adv — targeted pre-droplet panel, not whole-transcriptome.
- Licht & Mailaender 2024, Cancers — bulk RNA + microbiome, not single-cell.

**Li's own reference inputs (ledger only, not cohorts)** — `E-MTAB-8142` (8 healthy skin, used by Li as the healthy-skin reference). Listed here so §2's inventory of Li inputs is complete.

**Fetched-adjacent but never used** — GSE168508 (Liu 2022 CTCL bulk, n=196) and GSE121212
(Tsoi 2019 AD/psoriasis bulk), listed in the Li 2024 replication doc for B-cell deconvolution
validation; **never downloaded, never used**.

---

## 4 · TCR / V(D)J provenance

Joint-atlas clone table `data/atlas_joint/tcr_clones.parquet`: **360,794 TCR+ cells, 84,464 clones**
(pre-dedup). Malignant call = per-sample dominant clone, ≥5 % of VDJ cells **and** ≥2× the 2nd clone.
Validated: D5 skin 888/1,476 (60 %); H SS1-blood Sézary 1,624/3,161 (51 %).

| source | accession | how obtained | samples | TCR+ cells | donors |
|---|---|---|---|---|---|
| D1 chennareddy25 | GSE266862 | GEO `filtered_contig_annotations.csv.gz` | 22 | 96,578 | 22 |
| B4 geskin26 | GSE290850 | contig CSVs (SZ* libraries only) | 6 | 102,890 | 6 |
| B5 buus25 | GSE284075 | contig CSVs | 40 | 71,648 | 17 |
| B2 borcherding23 | GSE146586 | contig CSVs | 5 | 41,526 | 5 |
| D3 brunner24 | GSE269981 | contig CSVs (pre-dedup) | 11 | 27,070 | 11 |
| H herrera21 | GSE171811 | ECCITE `*_clonotypeAB.tsv.gz` membership matrix | 14 | 17,899 | 14 |
| D5 rindler21 | GSE165623 | contig CSVs | 3 | 3,183 | 1 |
| **Li CTCL1–8** | **E-MTAB-12303** | 31 CellRanger vdj libraries (13 `CTCL{N}_VDJ` + 18 small `WSSS_SKN*`); barcode-overlap donor mapping, purity 0.92–0.97 | 31 libs | **105,057** | 8 |
| **Li PT11–PT56** | **PRJNA754592** (SRS9777814–830) | **re-derived: TRUST4 v1.1.5-r573 mining 5′ GEX BAMs.** The deposit's own VDJ BAM lacks `@RG` → unusable (this is why Li 2024 itself excluded this cohort's TCR). TCR-locus reads streamed remotely via ENA `.bai` (~15–20 MB/channel) instead of downloading ~850 GB of BAM | 13 of 15 channels | **13,025** | 8 |

Li per-cell result: `li2024_tcr_malignancy.parquet`, 419,579 rows, `has_tcr` = 118,082
(`10x_vdj` 105,057 + `TRUST4_5p_GEX_mined` 13,025). Dominant-clone vs the paper's `tumor_cell`:
**precision 0.981, recall 0.834, Jaccard 0.821** (per-donor 0.647–0.924).

**No TCR, and why:**
- Li PKU / MF14–MF30 (10 donors, 60,164 cells) — V(D)J is in controlled-access GSA **HRA000166**.
- Li Sanger_Ncl_FFPE / CTCL9–CTCL18 (10 donors, 49,634 cells) — archival FFPE run on **10x Flex**;
  E-MTAB-12303 holds only 4 FFPE cellranger_multi GEX libraries, **no VDJ exists**.
- Li PT47 (SRS9777819) / PT50 (SRS9777820) — VDJ-only channels with no GEX BAM to mine; donor-level
  coverage still complete via their other channels.
- D6 gaydosik19 — 10x **3′**; negligible TRUST4 yield, explicitly not attempted.
- B1 borcherding19 (all 4 samples) and B4 `PF46–PF50` lanes — no VDJ libraries in the deposit.
- Li's own processed TCR table (`tcr_meta_CTCL.csv`, covers all 14 TCR patients) is **not publicly
  resolvable**: WebAtlas is a JS SPA, cellgeni bucket 404s, GitHub exposes only Sanger lustre paths,
  absent from CZ CELLxGENE. Moot — raw V(D)J sufficed.

---

## 5 · Spatial

| item | value |
|---|---|
| Accessions | **E-MTAB-13614** (8 CTCL sections, ENA `ERP155847`) + **E-MTAB-14559** (15 healthy sections, ENA `ERP165340`) |
| Platform | 10x **Visium**, per-section spaceranger `outs/`. **No processed h5ad exists upstream** (all Sanger CDN h5ad probes 404) |
| Obtained | BioStudies files endpoint, `wget -c`, 23 tarballs ≈ 8.5 GB → `sc.read_visium` + inner-join concat |
| Realized | **23 sections, 79,911 spots × 18,085 genes** (`ctcl_visium.h5ad`); CTCL 5,031 spots (305–1,040/section) + healthy 74,880 (15 × full 4,992 grid, background not removed upstream) |
| Modeling | **CondSCVI → DestVI** (`15_spatial_resolvi_deconvolution`): spot QC ≥1,000 counts → 17,466 spots (CTCL 4,687 / healthy 12,779); reference = Li scRNA 415,742 cells → 49 labels collapsed to 14, balanced-subsampled to 61,263 cells, 15,790 shared genes; 2,000 epochs |
| ResolVI | setup dry-check only (`03_spatial_visium_download`). `visium/resolvi_model/` + `ctcl_visium_resolvi.h5ad` exist on disk but **no notebook in the tree trains or reads them** — dead artifacts |
| Gap | **No published per-donor spatial↔scRNA crosswalk.** Only aggregate counts (23 = 8 CTCL + 15 healthy from 15 donors). Visium libraries have no VDJ and were never crosswalked to atlas donors. Section metadata (disease/sex/age) from SDRF only |
| Caveat | 55 µm spots are multi-cell; DestVI/resolVI assume single-cell resolution → partial fit, logged |

Also in the Li paper but **not used**: RareCyte immunofluorescence (protein imaging, no transcriptome).

---

## 6 · Reviewer checklist — known gaps, provisional labels, soft spots

**Provisional clinical labels (never verified against the papers)**
- D6 `SC157 / SC158 / SC205` — MF-vs-SS subtype not in the deposit → **defaulted MF**.
- B1 `P4 / P5` — disease (SS) and sample→subject map **provisional** (study = 1 SS + 4 HC).
- B4 `PF46–PF50` — group identity unclear (no TCR files) → **defaulted SS**.
- geskin26 patient identity unrecoverable (HTO key not deposited) — 65 pseudo-donors.
- 27 samples had provisional lineage (9 Chennareddy γδ-vs-CD8, 18 Li "possible CD8 MF"), resolved by
  marker scoring of the malignant clone (CD4 / CD8A·CD8B / TRGC·TRDC vs TRAC·TRBC); samples with
  <20 malignant cells stay `provisional_unresolved`. Audit cols `lineage_orig/_method/_score_*`.

**QC used same-lab proxies where the paper's supplement is gated**
Per-paper thresholds in `QC_CONFIG`. Retention: li24 100 % (kept as published) · geskin26 96.1 % ·
borcherding19 94.1 % · borcherding23 91.4 %ᵖ · gaydosik19 87.6 %ᵖ · herrera21 84.6 %ᵖ ·
rindler21 67.5 % · chennareddy25 64.9 %ᵖ · brunner24 46.2 % · buus25 97.2 %ᵖ.
ᵖ = threshold is a proxy or default, **not** the paper's own: chennareddy25←brunner24,
borcherding23←borcherding19, buus25←herrera21, gaydosik19/herrera21 = generic defaults.

**Other soft spots**
- `data/atlas_joint/README.md` is **stale**: member table lists 9 cohorts and omits B5/buus25
  (added 2026-07-21); its "Realized counts" section is empty.
- The saved `02_build_joint_atlas` QC-audit cell **failed on B5** (`shape mismatch (638,102) vs (638,127)` at PT13,
  ADT panel width) — B5 is in the built atlas, but its 131,598→127,861 numbers come from the
  per-sample GEX shapes, not the audit table.
- D1, D2, D3, H have **no `meta/meta.json`** (only `patients.tsv` / `samples.tsv`); D5, D6, B1–B5 do.
- D6 malignant labels are the weakest in the corpus (patient-private clustering, no TCR or CNV).
- D5 is n=1; its skin fraction may overlap an atlas Rindler donor (donors are namespaced, so it
  cannot collide, but the same patient may be represented twice).
- The candidate landscape (`CTCL_MF_SS_dataset_availability.csv`) was swept **2026-06-03** — anything
  published since is unreviewed.

**Questions for the reviewer**
1. Any CTCL/MF/SS scRNA, scTCR, or spatial cohort published after 2026-06-03 that we missed?
2. Any accession above that is wrong, superseded, or has since moved to open access
   (esp. HRA007111 tier, Srinivas 2024 / Peiffer 2023 EGA codes, Dorando 2025, Childs 2026)?
3. Any patient represented twice across cohorts that donor namespacing hides
   (Childs 2026 reuses 14 public patients; Buus 2025 reuses GSE171811 = our cohort H — the deposit
   itself does not contain them, but a real-patient overlap check was never run)?
4. Do our per-paper QC proxies materially change retention vs the published thresholds?
