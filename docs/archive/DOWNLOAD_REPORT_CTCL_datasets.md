# CTCL dataset download report

**Date:** 2026-06-03
**Source instructions:** `DOWNLOAD_remaining_CTCL_datasets.md` §3 (OPEN, fetch now)
**Location:** all under `notebooks/MF/data/<LABEL>/` (raw/ processed/ meta/), same scheme as D1–D3/H.
**Method:** GEO `suppl/` `_RAW.tar` via `wget` (recipe §2), extracted into `raw/`. Per-dataset `meta/meta.json` + `data/INVENTORY.tsv` written.

## Downloaded (6 datasets, all OPEN, all verified to open)

| LABEL | Study | Accession | Compartment | GSMs | Modality / matrix | Size |
|---|---|---|---|---|---|---|
| `D5_rindler2021frontimm_multitissue` | Rindler 2021 (Front Immunol) | GSE165623 | Skin+Blood+LN | 3 | 10x 5'+scTCR, mtx (33 538 g) | 148M |
| `D6_gaydosik2019_skin` | Gaydosik 2019 | GSE128531 | Skin | 9 | 10x 3', per-sample CSV (33 694 g) | 228M |
| `B1_borcherding2019_blood` | Borcherding 2019 | GSE124899 | Blood (PBMC) | 4 | 10x, mtx (33 694 g) | 188M |
| `B2_borcherding2023_blood` | Borcherding 2023 | GSE146586 | Blood (PBMC) | 5 | 10x 5'+scTCR, mtx (33 694 g) | 357M |
| `B3_borcherding2024_blood` | Borcherding 2024 (moga+IFN) | GSE192836 | Blood (PBMC) | 18 | **BD Rhapsody targeted ~453-gene panel** + AbSeq + TCR (CSV) | 4.9M |
| `B4_geskin2026_dupilumab_blood` | Geskin 2026 (dupilumab) | GSE290850 | Blood (PBMC) | 11 | 10x 5'+scTCR+HTO, CellRanger `.h5` | 2.7G |

All matrices verified: dimensions read, HDF5 magic confirmed for B4. Whole-transcriptome 10x for D5/D6/B1/B2/B4.

### Flags
- **B3 (GSE192836) is NOT whole-transcriptome** — BD Rhapsody targeted ~453-gene panel (rna/abseq/tcr CSVs, cells as columns). Limited value for the transcriptome-wide semantic-NMF / DE meta-analysis (same caveat class as Buus 2018). Kept for completeness; use with caution.
- **B1 ⊂ B2 overlap (do-not-double-count):** GSE124899 is re-listed inside Borcherding 2023. Downloaded GSE124899 once under **B1** and only the new **GSE146586** under B2. Do not pool B1 with the GSE124899 portion of B2.
- **D5 is n=1** (single stage-IVB patient, 3 compartments) — good for compartment logic, not for cohort DE power.
- **D6 malignant labels are weak** (patient-private clustering, no TCR/CNV).

## NOT downloaded (with reason)

**From the OPEN list — deliberately skipped (tidiness / not relevant to scRNA NMF):**
- `D4_rindler2021molcancer_skin` (GSE173205) — **already integrated in the atlas** (`E-MTAB-14559`, notebooks 02–06). Redundant; fetch standalone only if a non-atlas copy is later needed.
- atlas top-up `E-MTAB-12303`, `E-MTAB-13614` — optional Li-2024 companion libraries (FFPE-Flex / spatial); not needed for current per-dataset scRNA work.
- `S1_sarkar2024_melc` (Zenodo 11125482) — MELC protein imaging, **no RNA**; unusable for transcriptomic NMF.

**Rest of the corpus — out of scope per your instruction ("the rest are not relevant"):**
- **Controlled access** (need an application, cannot script): Liu 2022 `HRA000166`, Xue 2022 `HRA000847/000826/000145` (GSA-Human); Gaydosik 2022 `phs002933` (dbGaP); Srinivas 2024, Peiffer 2023 (EGA).
- **Verify/conditional:** Zhao 2025 `HRA007111` (check Open vs Controlled tier first); Du 2022 (no repository deposit).
- **Deposit pending:** D2 Johnson/Timp (already scaffolded, awaiting release), Dorando 2025, Childs 2026 in-house fraction.
- **Wrong data type:** GeoMx DSP ROI-level (Danielsen/Park/FMF-JID/Amechi 2025); Buus 2018 targeted panel; Licht 2024 bulk.

## Provenance written
- `data/<LABEL>/meta/meta.json` for each of the 6 (label, study, accession, repo, disease, compartment, malignant_call_method, access, source_url, download_date, overlaps, notes).
- `data/INVENTORY.tsv` — label / accession / repo / status / n_gsm / n_files / size / date.

## Next step (not done here)
Harmonize each `raw/` into `processed/<label>.h5ad` (concat GSMs, attach `cell_type`/`donor_id`/malignant labels, HVG subset) per your `04–09` pipeline, then build geneformer/genept maps as in D1/D3.
