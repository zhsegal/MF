# GSE284075 — download and integrate into the atlas

**Accession:** GSE284075 (GEO) — Buus et al., *Cancer Discov* 2025, PMC12498100
**Content:** scTCR + CITE-seq, 23 L-CTCL patients (PT01–PT23), PBMC + matched skin (some dermis/epidermis-split). RNA + ADT + TCR VDJ.
**Target:** existing six-cohort AnnData atlas (~34 donors). Add as a seventh cohort keyed `buus2025`.
**Scope caveat:** GSE284075 is **scRNA-seq/CITE-seq only**. The paper's WES/WGS (5 WES, 3 WGS, PT03/PT13 subclone-sorted) is **not** in this accession — do not expect a DNA layer here.

Run in an environment with NCBI/SRA egress (not the chat sandbox). Prefer processed matrices; only touch FASTQ if processed data lacks ADT or VDJ.

---

## 0. Layout

```bash
mkdir -p data/buus2025/{suppl,processed,logs}
cd data/buus2025
```

## 1. Pull metadata + supplementary manifest (inspect before downloading blobs)

GEO FTP nests series as `GSE284nnn/GSE284075/`.

```bash
# sample-level metadata (PTxx, tissue, chemistry, GSM↔SRR)
wget -c "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE284nnn/GSE284075/matrix/GSE284075_series_matrix.txt.gz" -P suppl/

# list supplementary files WITHOUT downloading — decide path from what's actually there
curl -s "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE284nnn/GSE284075/suppl/" | grep -oE 'GSE284075[^"<]*'
```

**Decision from the listing:**
- If a **processed annotated object** is present (`*.h5ad`, `*.rds`, `*_metadata*.csv.gz` with per-cell subclone / malignant labels) → use it directly (Step 2A). Fastest path and it carries the authors' subclone/major-minor calls.
- Else assemble from per-sample matrices + VDJ (Step 2B).

Download whichever set applies:

```bash
# whole-series bundle (if a *_RAW.tar exists)
curl -L "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE284075&format=file" -o suppl/GSE284075_RAW.tar
tar -xvf suppl/GSE284075_RAW.tar -C suppl/
```

## 2A. Load — annotated object present (preferred)

```python
import scanpy as sc, anndata as ad
a = sc.read_h5ad("suppl/<annotated>.h5ad")     # or read the rds via anndata2ri / rpy2
a.obs["cohort"] = "buus2025"
# namespace donors to atlas convention: buus2025__PT01 ...
a.obs["donor"] = "buus2025__" + a.obs["<patient_col>"].astype(str)
# PRESERVE author annotations — rename to avoid clobbering atlas columns
a.obs = a.obs.rename(columns={
    "<their_malignant_col>": "buus_malignant",
    "<their_subclone_col>":  "buus_subclone",     # A/B/C major, .1/.2 minor
    "<their_celltype_col>":  "buus_celltype",
})
a.write("processed/buus2025.h5ad")
```

## 2B. Load — assemble from per-sample matrices

Write `scripts/load_buus2025.py`. Per GSM/PTxx sample:

```python
import scanpy as sc, scirpy as ir, anndata as ad, pandas as pd, glob, re

samples = []
for mtx in glob.glob("suppl/*filtered*matrix*"):   # adapt glob to actual names
    s = sc.read_10x_h5(mtx) if mtx.endswith(".h5") else sc.read_10x_mtx(mtx, gex_only=False)
    s.var_names_make_unique()
    # CITE-seq: split RNA vs ADT by feature_types
    is_adt = s.var["feature_types"] == "Antibody Capture"
    rna, adt = s[:, ~is_adt].copy(), s[:, is_adt].copy()
    rna.obsm["X_adt"] = adt.to_df()               # keep ADT on obsm (buus-only modality)
    rna.uns["adt_panel"] = list(adt.var_names)
    pid = re.search(r"(PT\d+)", mtx).group(1)
    tis = ...                                      # blood/skin/dermis/epidermis from series_matrix
    rna.obs["cohort"], rna.obs["donor"], rna.obs["tissue"] = "buus2025", f"buus2025__{pid}", tis
    # attach TCR
    vdj = glob.glob(f"suppl/*{pid}*filtered_contig*")
    if vdj:
        ir.io.read_10x_vdj(vdj[0])                 # then ir.pp.merge / index by barcode
    samples.append(rna)

a = ad.concat(samples, join="outer", index_unique="-")
a.write("processed/buus2025.h5ad")
```

If ADT or VDJ is missing from processed files, fetch FASTQ via the SRA project linked in `series_matrix` (SRP/PRJNA) and reprocess with `cellranger multi`:

```bash
# nf-core/fetchngs -> SRR list from series_matrix, then:
prefetch --option-file srr_list.txt && fasterq-dump --split-files <SRR>
```

## 3. QC — match atlas thresholds (do not invent numbers)

Read the atlas's own QC config (min genes/counts, max mito%, doublet handling) from the existing pipeline / atlas `.uns`, and apply the identical gates to `buus2025.h5ad`. Log pre/post cell counts per donor to `logs/`.

## 4. Integrate

```python
atlas = sc.read_h5ad("<ATLAS_PATH>.h5ad")
buus  = sc.read_h5ad("processed/buus2025.h5ad")

# gene-space harmonization: subset buus to atlas var / shared HVG before concat
merged = ad.concat([atlas, buus], join="inner", label="batch_add", keys=["atlas","buus2025"], merge="first")

# MrVI is donor-aware: the model must SEE the new donors -> re-run to include buus2025,
# recompute X_mrvi_u over the combined donor set (do NOT reuse the old embedding as-is).
# Re-run per the existing MrVI training cell in nb; sample_key='donor', batch as configured.

# TCR: merge scirpy AIRR across cohorts (other fresh cohorts already carry TCR).
```

Then re-run the downstream `old/23_subclonal_evolution` steps (malignancy call, subclone detection, MrVI AUC major/minor) on the extended object so `buus2025` donors flow through the same pipeline.

## 5. Sanity / dedup (must pass before use)

- Exactly **23** `buus2025__PTxx` donors; no ID collision with existing atlas donors (double-underscore namespacing enforced).
- **Herrera overlap:** GSE171811 (Herrera 2021) is your existing `herrera2021` arm and is a *separate* reused accession in this paper — GSE284075 should contain only the Copenhagen 23. Confirm no shared samples by matching TCRβ CDR3 / donor metadata against `herrera2021` before merging.
- ADT lives only on `buus2025` cells (`obsm['X_adt']`) — confirm it is not broadcast into the shared `var`.
- Malignant-fraction and cells/donor within expected range; write a short `logs/buus2025_integration_report.md`.

## 6. Save

Write the extended atlas and a standalone `processed/buus2025.h5ad` (with `buus_*` author labels retained) to the atlas outputs directory. Keep the ADT panel and VDJ intact.

---

### Reused accessions in this paper (NOT downloaded here — reference only)
GSE124899, GSE146586, GSE207679, GSE182861, **GSE171811 (= your herrera2021)**, GSE197619; dbGaP **phs002933.v1.p1** (controlled access). Only GSE284075 is new.
