# Agent guide: download the remaining open CTCL MF/SS datasets

**Goal.** Fetch the public datasets in `CTCL_MF_SS_dataset_availability.csv` that are **open** and **not yet downloaded**, into the existing `data/` tree, with consistent labels and a provenance file per dataset. Prefer **processed count matrices** (`.h5` / `.h5ad` / `.mtx` / `.rds`) over raw FASTQ; only fetch FASTQ if re-alignment is explicitly required.

**Access status has been resolved** (June 2026). Each dataset below is now classified Open / Controlled / Verify / Already-held. Do **not** attempt anonymous download of Controlled data (GSA-Human HRA / dbGaP / EGA) — see Section 6.

> **Do-not-double-count note.** Your atlas object (`E-MTAB-14559` / cellatlas.io h5ad) already integrates **Liu 2022 (HRA000166)** and **Rindler 2021 Mol Cancer (GSE173205)**. Re-download those as standalone datasets only if you specifically want them outside the atlas; otherwise they are already in `02-06`.

---

## 0. Prerequisites

```bash
conda activate ctcl
pip install ffq GEOparse requests
conda install -y -c bioconda sra-tools entrez-direct aria2 pigz jq
```

## 1. Folder + label convention (extends what you already use)

```
data/<LABEL>/
  raw/         # exactly as downloaded
  processed/   # your harmonized .h5ad (created later)
  meta/        # meta.json + manifest + md5
```
`<LABEL>` continues your scheme: `D#_<author><year>_<compartment>` (skin scRNA), `B#_<...>` (blood/SS). Existing: atlas (`E-MTAB-14559`), `D1_chennareddy2025`, `D2_johnson_timp2025`, `D3_brunner2024` (GSE269981), `H_herrera2021`. **Start new skin datasets at D4.** Write `meta/meta.json` per dataset (label, study, accession, repo, disease, compartment, malignant_call_method, access, source_url, download_date, overlaps, notes).

## 2. Download recipes

**GEO (processed matrices — default):**
```bash
GSE=GSE173205
STEM=$(echo "$GSE" | sed -E 's/[0-9]{3}$/nnn/')   # -> GSE173nnn
wget -r -np -nH --cut-dirs=6 -e robots=off \
  -A '*.h5,*.h5ad,*.rds,*.mtx.gz,*barcodes*,*features*,*genes*,*matrix*,*.tar,*metadata*,*annot*,*contig*' \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/${STEM}/${GSE}/suppl/" -P raw/
```
GEO raw FASTQ (only if needed): `ffq --ftp "$GSE" | jq -r '.[].url' > raw/urls.txt && aria2c -x8 -s8 -d raw -i raw/urls.txt`

**ArrayExpress / BioStudies (E-MTAB):**
```bash
ACC=E-MTAB-12303
curl -s "https://www.ebi.ac.uk/biostudies/files/${ACC}/Files.json" -o meta/${ACC}_files.json
jq -r '..|.path? // empty' meta/${ACC}_files.json | while read f; do
  aria2c -x4 -d raw "https://ftp.ebi.ac.uk/biostudies/fire/${ACC}/Files/${f}" ; done
```

**Zenodo:** `REC=11125482; curl -s "https://zenodo.org/api/records/${REC}" | jq -r '.files[].links.self' | xargs -n1 -I{} aria2c -x4 -d raw {}`

## 3. Worklist — OPEN, fetch now

| LABEL | Study | Accession | Repo | Recipe |
|---|---|---|---|---|
| `D4_rindler2021molcancer_skin`* | Rindler 2021 Mol Cancer (MF skin progression) | `GSE173205` | GEO | §2 GEO |
| `D5_rindler2021frontimm_multitissue` | Rindler 2021 Front Immunol (MF skin+blood+LN) | `GSE165623` | GEO | §2 GEO |
| `D6_gaydosik2019_skin` | Gaydosik 2019 (advanced CTCL skin; clustering-only labels) | `GSE128531` | GEO | §2 GEO |
| `B1_borcherding2019_blood` | Borcherding 2019 (SS PBMC) | `GSE124899` | GEO | §2 GEO |
| `B2_borcherding2023_blood` | Borcherding 2023 (SS PBMC, treatment) | `GSE146586` (+`GSE124899`) | GEO | §2 GEO |
| `B3_borcherding2024_blood` | Borcherding 2024 (SS PBMC, moga+IFN) — **RESOLVED open** | `GSE192836` | GEO | §2 GEO |
| `B4_geskin2026_dupilumab_blood` | Geskin 2026 (SS PBMC, dupilumab) | `GSE290850` | GEO | §2 GEO |
| `atlas` (top up, optional) | Li 2024 companion libraries | `E-MTAB-12303`, `E-MTAB-13614` | ArrayExpress | §2 EBI |
| `S1_sarkar2024_melc` (optional) | Sarkar 2024 MELC protein imaging — no RNA | `zenodo 11125482` | Zenodo | §2 Zenodo |

*`D4` (GSE173205) and Liu 2022 are already in the atlas — fetch standalone only if needed (see double-count note).

## 4. Worklist — VERIFY: now RESOLVED

- **Zhao 2025** → **GSA-Human `HRA007111`** (2 MF + adjacent skin + 1 HC). GSA-Human sets access per record: read the "Access" field at https://ngdc.cncb.ac.cn/gsa-human/browse/HRA007111 — if **Open Access**, download directly (HTTPS/Aspera links on the page); if **Controlled**, file a DAC request (Section 7). Label `D7_zhao2025_skin`. Small cohort → low DE power; nice-to-have, not core.
- **Du 2022** → **no repository deposit** (statement: "all data and code are mentioned in the article and supplementary materials"). No public raw count matrices → cannot be re-analyzed at single-cell level; see Section 8.

## 5. After each download — verify + label

1. Count files vs. expected samples; verify md5 if provided. 2. Load one matrix, print shape, confirm it opens. 3. Write `meta/meta.json` (source_url, download_date). 4. Record **overlaps** — these are NOT independent: `D1_chennareddy2025` and `D3_brunner2024` (GSE269981) share Brunner/Vienna healthy controls; `B1_borcherding2019` is re-listed inside `B2_borcherding2023` (GSE124899); Liu 2022 + Rindler 2021 are inside the atlas; Childs 2026 reuses 14 public datasets. 5. Append to `data/INVENTORY.tsv`: `label  accession  repo  status  n_files  date`.

## 6. Controlled access — DO NOT script (submit an application). RESOLVED codes below.

| Study | Accession | Repository | How to request |
|---|---|---|---|
| **Liu 2022** (MF skin scRNA + WES) | `HRA000166` | NGDC **GSA-Human** | GSA-Human request (Section 7). *scRNA already in atlas; request only for raw FASTQ/WES.* |
| **Xue 2022** (SS scRNA + scATAC) | `HRA000847`, `HRA000826`, `HRA000145` | NGDC **GSA-Human** | GSA-Human request (Section 7) |
| Gaydosik 2022 pembrolizumab (SS) | `phs002933.v1.p1` | dbGaP | dbGaP authorized-access application (eRA Commons + DAC) |
| Srinivas 2024 (MF skin) | EGA (confirm code) | EGA | EGA Data Access Committee request |
| Peiffer 2023 (SS blood+skin) | EGA (confirm code) | EGA | EGA Data Access Committee request |
| Zhao 2025 (MF skin) — *only if its record is Controlled* | `HRA007111` | NGDC GSA-Human | check tier first; if Controlled, Section 7 |

> **GSA-Human caveat:** the `HRA` prefix does NOT by itself mean controlled — each record is tagged Open or Controlled. Confirm the "Access" field on each record page (`HRA000166`, `HRA000847`/`000826`/`000145`, `HRA007111`). Open records download directly; only Controlled ones need Section 7.

## 7. GSA-Human request — what you (the human) need to do

For `HRA000166` (Liu) and `HRA000847/000826/000145` (Xue) at <https://ngdc.cncb.ac.cn/gsa-human/>:
1. Register an NGDC account (institutional email).
2. Open the HRA record → "Request data".
3. Submit a Data Access Request: research purpose, your PI/institution, and a signed Data Access Agreement (PI signature; some require institutional/ethics confirmation).
4. Approval is granted by the data submitter/DAC (typically days–weeks); download link is then enabled.

## 8. Skip for the scRNA re-analysis (wrong data type or not released)

- **GeoMx DSP, ROI-level only** (Danielsen 2024, Park 2025, FMF-JID 2025, Amechi 2025): region-level counts in paper supplements, not single-cell — separate DSP arm.
- **Deposit pending**: `D2_johnson_timp2025` (medRxiv 2025.09.07.25335167, awaiting release), Dorando 2025 (bioRxiv 2025.02.11.637715), Childs 2026 in-house fraction (fetch its 14 reused public originals instead).
- **No public raw data**: Du 2022 (deposited nothing — data only in article/supplement); email the authors if you need it.
- **Not single-cell**: Buus 2018 (targeted panel), Licht 2024 (bulk).
