# CTCL scRNA-seq Datasets — Download & Structure Reference

**Audience:** Claude Code agent.
**Scope:** How to download two MF skin scRNA-seq datasets, what files are in each deposit, and what the metadata columns mean. No integration, no malignant-cell calling, no analysis pipeline — those are downstream concerns.
**Last updated:** 29 May 2026.

---

## 0. Overview

| # | Dataset | Repository | Cohort | Cells |
|---|---|---|---|---|
| **D1** | Chennareddy et al., *Br J Dermatol* 2025. DOI `10.1093/bjd/ljae313` | GEO `GSE266862` | 18 CTCL skin biopsies + 4 healthy controls (22 samples, 14 patients) | ~145,817 |
| **D2** | Johnson, Li, Solhjoo, Madan, Ali, Nash, Hicks, Timp. *medRxiv* 2025.09.07.25335167, posted 10 Sept 2025. DOI `10.1101/2025.09.07.25335167` | medRxiv preprint — verify deposit (see §3.1) | 1 healthy + 4 MF (5 skin samples) | ~28,902 |

> Note on attribution: the project's methods review calls D2 the "Geskin/Fuschiotti CAF preprint" — that is incorrect. The correct authors are the **Johnson / Timp / Hicks** group at Johns Hopkins. Cohort size and topic still match.

---

## 1. Environment setup

```bash
conda create -y -n ctcl python=3.11
conda activate ctcl
pip install --upgrade pip scanpy anndata scirpy pandas requests tqdm GEOparse ffq
conda install -y -c bioconda sra-tools aria2 jq
```

Directory layout:

```bash
mkdir -p data/{D1_chennareddy2025,D2_johnson_timp2025}/{raw,processed,meta}
mkdir -p logs
```

---

## 2. D1 — Chennareddy et al. 2025 *BJD*

### 2.1 Download

```bash
cd data/D1_chennareddy2025/raw

# Processed matrices (recommended) — GEO supplementary files
ACCESSION=GSE266862
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE266nnn/${ACCESSION}/suppl"
wget -nc -r -np -nH --cut-dirs=5 \
  -A "*.h5,*.tar,*_metadata*,*.tsv*,*matrix*,*barcodes*,*features*,*contig*" \
  "${BASE}/"

# Raw FASTQ via SRA (only if re-alignment is needed)
ffq -o ../meta/GSE266862.json GSE266862
jq -r '.[].files.ftp[].url' ../meta/GSE266862.json > sra_urls.txt
aria2c -x 8 -s 8 -i sra_urls.txt
```

The paper also notes overlap with prior Brunner-lab deposits `GSE173205` and `GSE222840` — these are *separate* GEO records (different healthy-control samples and earlier CTCL biopsies that are reused in this paper). Pull them only if you need the full overlap set.

### 2.2 File structure expected in GSE266862

Per sample (22 samples total), 10x Cell Ranger output:

```
<sample_id>_filtered_feature_bc_matrix.h5         # filtered gene-cell matrix, HDF5
<sample_id>_filtered_contig_annotations.csv.gz    # 10x V(D)J — TCRαβ or γδ
<sample_id>_clonotypes.csv.gz                     # V(D)J clonotype table
<sample_id>_metadata.tsv.gz                       # per-cell metadata if provided
```

Plus series-level files (one per GSE):

```
GSE266862_RAW.tar                                 # bundle of all per-sample folders
GSE266862_series_matrix.txt.gz                    # sample-level metadata table
GSE266862_family.soft.gz                          # SOFT metadata
```

The 22 samples are 18 CTCL + 4 HC. The four HC sample IDs (`P112, P115, P116, P121`) overlap with `GSE173205` / `GSE222840`.

### 2.3 Sample / patient metadata (from paper Table 1)

Build `data/D1_chennareddy2025/meta/patients.tsv` manually from Table 1 with these columns:

| Column | Type | Meaning |
|---|---|---|
| `participant_id` | str | Patient ID. Format `P<number>`. Matches the GEO sample ID prefix. |
| `diagnosis` | str | One of `HC`, `Tumour-stage MF`, `Erythrodermic MF`, `Plaque-stage MF`, `BL` (Berti lymphoma). |
| `histopath_diagnosis` | str | Refined histopathological label: `CD4+ MF`, `CD4+ CD30+ MF`, `GD MF`, `GD CD30+ MF`, `CD8+ CTCL`. |
| `subset` | str | One of `CD4_MF`, `gd_MF`, `Berti`, `HC` — the grouping used in the paper. |
| `age` | int | Years. |
| `sex` | str | `M` / `F`. |
| `disease_duration_years` | int | Years since diagnosis. |
| `current_treatment` | str | Treatment at sampling (e.g. `ECP`, `IFN-α`, `None`). |
| `previous_treatments` | str | Free-text list. |
| `stage` | str | ISCL/EORTC stage (e.g. `IIB`, `IVA1`, `IVB`, `IB`) or `T3N0M0` for non-MF entities. |
| `disease_course` | str | `Indolent` or `Aggressive`. |
| `lesion_type` | str | `Tumour`, `Plaque`, `Patch`, `Erythroderma`, `Ulcerated tumour`. |
| `blood_involvement` | str | `B0`, `B1`, `B2`, `B2b`. |
| `is_multi_site` | bool | True for `P195`/`P218` (Berti, 2 sites) and `P318` (γδ MF, 4 sites: P318A, P318B, P97, P124). |

The 7 classic CD4⁺ MF patient IDs are: **P76, P171, P204, P312, P311, P303, PGS** (all advanced-stage IIB–IVB, all aggressive).
The 4 HC IDs are: **P112, P115, P116, P121**.
Berti / γδ multi-site cases: **P195=P218** (same patient, 2 biopsies), **P318=P318A=P318B=P97=P124** (one patient, 4 biopsies).

### 2.4 Loading

```python
import scanpy as sc, pandas as pd
from pathlib import Path

raw_dir = Path("data/D1_chennareddy2025/raw")
meta    = pd.read_csv("data/D1_chennareddy2025/meta/patients.tsv", sep="\t")

adatas = {}
for h5 in sorted(raw_dir.glob("*_filtered_feature_bc_matrix.h5")):
    sample_id = h5.stem.replace("_filtered_feature_bc_matrix", "")
    a = sc.read_10x_h5(h5)
    a.var_names_make_unique()
    a.obs["sample_id"] = sample_id
    adatas[sample_id] = a

adata = sc.concat(adatas, label="sample_id", index_unique="-")
adata.obs = adata.obs.merge(
    meta, left_on="sample_id", right_on="participant_id", how="left"
)
adata.write_h5ad("data/D1_chennareddy2025/processed/chennareddy_concat.h5ad")
```

V(D)J:

```python
import scirpy as ir
# Per sample, read filtered_contig_annotations.csv with ir.io.read_10x_vdj
# then merge by cell barcode into adata.obs / adata.obsm["airr"]
```

### 2.5 Expected `adata` shape after loading

- `adata.shape ≈ (145_817, ~30_000)`
- `adata.obs["sample_id"].nunique() == 22`
- `adata.obs["subset"].value_counts()` → `CD4_MF: 7, gd_MF: 7, Berti: 4, HC: 4`

### 2.6 Sequencing technology

10x Genomics Chromium 5′ v1.1 + V(D)J. TCRαβ for CD4⁺ / Berti samples; γδ TCR for γδ MF samples (amplified per Mimitou ECCITE primers). Illumina NovaSeq, 150 bp paired-end. No CITE-seq, no spatial transcriptomics, no scATAC.

---

## 3. D2 — Johnson, Timp et al. 2025 *medRxiv*

### 3.1 Resolve the accession first

This is a preprint. Do not assume data is on GEO — verify before downloading.

```bash
# Fetch and inspect the Data Availability section
curl -sL https://www.medrxiv.org/content/10.1101/2025.09.07.25335167v1.full.pdf \
  -o data/D2_johnson_timp2025/meta/preprint.pdf
pdftotext data/D2_johnson_timp2025/meta/preprint.pdf - \
  | grep -A 20 -i "data availability\|data sharing\|accession"
```

Record whatever you find in `data/D2_johnson_timp2025/meta/accessions.txt`:
- `GSE______` — GEO, if deposited.
- SRA / BioProject `PRJNA______` — for raw FASTQ.
- Zenodo DOI `10.5281/zenodo.______` — Hicks lab commonly uses this for processed objects.
- GitHub repo with releases — check `github.com/timplab`, `github.com/stephaniehicks`.

If the statement says "upon publication" or "upon reasonable request," email the corresponding author **Winston Timp (`wtimp@jhu.edu`)** to request access. Log the request in `data/D2_johnson_timp2025/meta/access_log.md`. Mark the dataset as `pending` and stop — do not invent an accession.

### 3.2 Download (once accession is confirmed)

GEO path:

```bash
cd data/D2_johnson_timp2025/raw
GSE=GSE______   # from §3.1
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/${GSE:0:6}nnn/${GSE}/suppl"
wget -nc -r -np -nH --cut-dirs=5 \
  -A "*.h5,*.tar,*.tsv*,*matrix*,*barcodes*,*features*,*meta*,*.rds" \
  "${BASE}/"
```

Zenodo path:

```bash
cd data/D2_johnson_timp2025/raw
ZENODO_DOI=10.5281/zenodo.________
ZENODO_ID="${ZENODO_DOI##*.}"
curl -s "https://zenodo.org/api/records/${ZENODO_ID}" \
  | jq -r '.files[].links.self' > zenodo_urls.txt
aria2c -x 4 -s 4 -i zenodo_urls.txt
```

SRA-only path:

```bash
ffq -o ../meta/${SRA_PROJECT}.json ${SRA_PROJECT}
jq -r '.[].files.ftp[].url' ../meta/${SRA_PROJECT}.json > urls.txt
aria2c -x 8 -s 8 -i urls.txt
```

### 3.3 File structure expected

This is a Hicks-lab Bioconductor-style deposit. Most likely:

```
johnson_timp_2025_ctcl_caf.rds                # Seurat v4 object (original analysis)
johnson_timp_2025_ctcl_caf.h5ad               # AnnData export (if provided)
metadata.tsv                                   # per-cell metadata
sample_metadata.tsv                           # per-sample (patient-level) metadata
README.md                                      # data dictionary
```

If GEO instead, expect per-sample 10x Cell Ranger output:

```
<sample_id>_filtered_feature_bc_matrix.h5     # per-sample matrix
GSE______ _series_matrix.txt.gz               # sample-level metadata
```

### 3.4 Sample / patient metadata (from preprint, build manually)

Build `data/D2_johnson_timp2025/meta/patients.tsv`:

| Column | Type | Meaning |
|---|---|---|
| `sample_id` | str | Sample identifier as deposited. |
| `patient_id` | str | Patient identifier (one per sample in this cohort). |
| `disease` | str | `HC` or `MF`. |
| `disease_stage` | str | `HC`, `early` (IA–IIA), or `late` (IIB–IVB). |
| `stage_detail` | str | Specific TNM stage where given. |
| `age` | int | Years. |
| `sex` | str | `M` / `F`. |
| `self_reported_race` | str | The cohort explicitly samples racial diversity — preserve verbatim from the preprint. |
| `lesion_type` | str | Lesion type biopsied (patch / plaque / tumour / healthy). |

Cohort: 5 samples total — 1 HC, 2 early-stage MF, 2 late-stage MF.
Cell counts: 1,906 HC + 26,996 MF = 28,902.

### 3.5 Loading

```python
import scanpy as sc, pandas as pd
from pathlib import Path

raw = Path("data/D2_johnson_timp2025/raw")

# Case A: integrated object provided (.h5ad or .rds)
if (raw / "johnson_timp_2025_ctcl_caf.h5ad").exists():
    adata = sc.read_h5ad(raw / "johnson_timp_2025_ctcl_caf.h5ad")
elif (raw / "johnson_timp_2025_ctcl_caf.rds").exists():
    # Convert in R first using sceasy or anndata2ri
    raise RuntimeError("Convert .rds → .h5ad with sceasy in R before loading.")

# Case B: per-sample .h5 files from GEO
else:
    adatas = {}
    for h5 in sorted(raw.glob("*_filtered_feature_bc_matrix.h5")):
        sample_id = h5.stem.split("_")[0]
        a = sc.read_10x_h5(h5)
        a.var_names_make_unique()
        a.obs["sample_id"] = sample_id
        adatas[sample_id] = a
    adata = sc.concat(adatas, label="sample_id", index_unique="-")

meta = pd.read_csv("data/D2_johnson_timp2025/meta/patients.tsv", sep="\t")
adata.obs = adata.obs.merge(meta, on="sample_id", how="left")
adata.write_h5ad("data/D2_johnson_timp2025/processed/johnson_timp_concat.h5ad")
```

### 3.6 Expected `adata` shape

- `adata.shape ≈ (28_902, ~30_000)`
- `adata.obs["sample_id"].nunique() == 5`
- `adata.obs["disease_stage"].value_counts()` → `HC: 1, early: 2, late: 2` (sample counts)
- HC cells: 1,906; MF cells: 26,996.

### 3.7 Sequencing technology

10x Genomics Chromium scRNA-seq. **No V(D)J / TCR sequencing.** No CITE-seq. No spatial transcriptomics. Original analysis was Seurat v4 in R; processed object will likely be a Seurat `.rds` requiring conversion to `.h5ad` for Python use.

---

## 4. Quick sanity checks after loading both datasets

```python
import scanpy as sc

d1 = sc.read_h5ad("data/D1_chennareddy2025/processed/chennareddy_concat.h5ad")
d2 = sc.read_h5ad("data/D2_johnson_timp2025/processed/johnson_timp_concat.h5ad")

# D1 expectations
assert d1.obs["sample_id"].nunique() == 22
assert d1.n_obs > 100_000   # ~145,817
print("D1 subset counts:", d1.obs["subset"].value_counts().to_dict())

# D2 expectations
assert d2.obs["sample_id"].nunique() == 5
assert 25_000 < d2.n_obs < 35_000   # ~28,902
print("D2 stage counts:", d2.obs["disease_stage"].value_counts().to_dict())
```

---

## 5. Citations

- Chennareddy S, Rindler K, Ruggiero JR, *et al.* Single-cell RNA sequencing comparison of CD4+, CD8+ and T-cell receptor γδ+ cutaneous T-cell lymphomas reveals subset-specific molecular phenotypes. *Br J Dermatol* 2025; 192(2):269–282. doi: 10.1093/bjd/ljae313.
- Johnson CM, Li W, Solhjoo S, Madan V, Ali I, Nash K, Hicks S, Timp W. Single-cell RNA sequencing identifies subtypes of cancer-associated fibroblasts in early and late stages of mycosis fungoides. *medRxiv* 2025.09.07.25335167; doi: 10.1101/2025.09.07.25335167.

---

*If an accession or URL fails, do not invent a substitute — log the failure to `logs/download_failures.txt` and surface it to the user.*
