/c# Instructions: Download and Preliminary Analysis of the Li/Strobl/Haniffa CTCL Atlas

**Target paper:** Li, R., Strobl, J., Poyner, E. F. M., Balbaa, A., et al. (2024). *Cutaneous T cell lymphoma atlas reveals malignant TH2 cells supported by a B cell-rich tumor microenvironment.* **Nature Immunology** 25:2320–2330. DOI: [10.1038/s41590-024-02018-1](https://doi.org/10.1038/s41590-024-02018-1).

**Audience:** A Claude Code agent operating in a Linux/macOS environment with ≥ 64 GB RAM, ≥ 200 GB free disk, and outbound internet access to `nature.com`, `ebi.ac.uk`, `cellatlas.io`, `cog.sanger.ac.uk`, `ncbi.nlm.nih.gov`, `github.com`, and `conda-forge`/`pypi`.

**Goal:** Pull down the processed single-cell + Visium spatial transcriptomics data, sanity-check it, and run a preliminary analysis that reproduces the paper's headline findings (TH2 skewing in advanced disease, B-cell enrichment, MHC-II⁺ fibroblast/DC niche).

---

## 0. Orientation: what data exists and where

The atlas integrates five logical pieces. The agent should plan around these:

| # | Data | Origin | Hosting | Use |
|---|---|---|---|---|
| 1 | **Processed scRNA-seq atlas** (~420 k cells, 45 CTCL patients integrated with healthy/AD/psoriasis skin) | Newly generated (n=18) + 3 prior CTCL studies + skin cell atlas | WebAtlas portal at `https://collections.cellatlas.io/ctcl` (h5ad downloads + interactive viewer) | Primary analysis object |
| 2 | **Raw scRNA-seq** for the 18 newly profiled CTCL patients (8 fresh FACS-sorted + 10 archival FFPE 10x Flex) | Newly generated | **ArrayExpress** (per paper's Data Availability and Nature Reporting Summary). Search EBI BioStudies for **"CTCL Haniffa"** to confirm the exact `E-MTAB-XXXXX` accession at runtime — see §1.4 | Re-alignment, re-QC, custom analyses |
| 3 | **Visium spatial transcriptomics** (23 sections: 8 CTCL + 15 healthy from 15 donors) | Newly generated | ArrayExpress (same submission as #2) + processed h5ad on the WebAtlas portal | Spatial deconvolution (cell2location), niche analysis |
| 4 | **Integrated public CTCL scRNA-seq** | Liu et al. 2022 *Nat Commun*; Rindler et al. 2021 *Mol Cancer*; Song et al. 2022 *Cancer Discov* | GEO (each paper's own accession — see §1.5) | Only needed if reproducing the integration from scratch |
| 5 | **Bulk RNA-seq for B-cell deconvolution validation** (n=196) | Liu et al. 2022 *Blood* CTCL bulk, plus Tsoi et al. 2019 AD/psoriasis bulk | GEO **GSE168508** (CTCL) and **GSE121212** (AD/psoriasis) | Bulk deconvolution + survival analysis |

**Strong recommendation:** the agent should *start with #1* — the processed h5ad on the WebAtlas portal. Re-running CellRanger on 45 patients of 10x data is many days and many TB; the published h5ad contains the QC'd, integrated, cell-typed matrix used to generate every figure in the paper.

---

## 1. Download the data

### 1.1 Create a working directory and skeleton

```bash
export PROJECT=/path/to/ctcl_atlas        # CHANGE ME
mkdir -p "$PROJECT"/{raw,processed,visium,bulk,scripts,results,logs,env}
cd "$PROJECT"
```

### 1.2 Set up a reproducible Python environment

Use **mamba** (much faster than conda for scanpy stack). If only conda is available, swap `mamba` → `conda`.

```bash
# install miniforge if not present
if ! command -v mamba >/dev/null 2>&1; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$PROJECT/env/miniforge"
  source "$PROJECT/env/miniforge/etc/profile.d/conda.sh"
  conda activate base
fi

mamba create -y -n ctcl python=3.11
mamba activate ctcl

# Core single-cell stack
mamba install -y -c conda-forge \
  numpy "pandas<2.3" scipy matplotlib seaborn \
  scanpy anndata h5py "leidenalg>=0.10" python-igraph \
  scikit-learn statsmodels \
  jupyterlab ipykernel ipywidgets tqdm rich

# Specialised tools used in the paper
pip install --quiet \
  harmonypy \
  celltypist \
  milopy \
  scrublet \
  scvi-tools \
  pertpy \
  cell2location \
  squidpy \
  decoupler-py \
  pydeseq2 \
  scirpy

# infercnvpy (Python port of inferCNV) for malignant-cell identification
pip install --quiet infercnvpy

python -c "import scanpy as sc; sc.logging.print_header()"
```

If GPU is available, install JAX/torch with CUDA for cell2location and scvi:
```bash
pip install --quiet "jax[cuda12]"  # only if NVIDIA GPU; otherwise skip
```

### 1.3 Pull the processed atlas h5ad from the WebAtlas portal

The portal at `https://collections.cellatlas.io/ctcl` is a JavaScript single-page app. The actual h5ad blobs are hosted on the Sanger object store (`*.cog.sanger.ac.uk`). The agent should discover the exact URLs at runtime — they have changed format across other cellatlas studies. Two reliable strategies:

**Strategy A — open the portal and read the network panel.** Use Claude in Chrome (or another headless browser) to load `https://collections.cellatlas.io/ctcl`, click any "Download" button, and capture the resulting URL.

**Strategy B — probe the conventional Sanger CDN paths.** The Haniffa lab has a stable pattern; try these and pick the ones that return HTTP 200:

```bash
# Probe candidate URLs (the agent should update this list at run time)
CANDIDATES=(
  "https://cellatlas-ctcl.cog.sanger.ac.uk/Submission_CTCL_revised_final.h5ad"
  "https://cellatlas-ctcl.cog.sanger.ac.uk/CTCL_scRNA.h5ad"
  "https://cellatlas-ctcl.cog.sanger.ac.uk/CTCL_atlas.h5ad"
  "https://cellatlas-ctcl.cog.sanger.ac.uk/CTCL_integrated.h5ad"
  "https://cellatlas-ctcl.cog.sanger.ac.uk/CTCL_visium.h5ad"
  "https://ctcl.cellgeni.sanger.ac.uk/CTCL.h5ad"
)
for url in "${CANDIDATES[@]}"; do
  echo -n "$url … "
  curl -sIL -o /dev/null -w "%{http_code}\n" --max-time 15 "$url"
done
```

For each URL that returns 200, download with `wget -c` (resume-safe):

```bash
cd "$PROJECT/processed"
wget -c "https://cellatlas-ctcl.cog.sanger.ac.uk/<file>.h5ad"
```

> **Note on file naming:** the WebAtlas pipeline produces files like `<study>_<modality>.h5ad` and `<study>_<sample>.zarr`. If the agent finds neither, fall back to the WebAtlas pipeline source on GitHub (`https://github.com/haniffalab/webatlas-pipeline`) — the `examples/` and release notes contain the canonical download manifests.

### 1.4 Look up the ArrayExpress accession (for raw FASTQs / re-alignment only)

The paper's Data Availability section deposits all newly-generated sequencing in ArrayExpress under a study-level accession. To find it programmatically:

```bash
# Search EBI BioStudies API for the project
curl -sL --max-time 30 \
  'https://www.ebi.ac.uk/biostudies/api/v1/search?query=%22cutaneous+T+cell+lymphoma+atlas%22+Haniffa&pageSize=20' \
  | python -c "import json,sys; d=json.load(sys.stdin); [print(h['accession'],h.get('title','')) for h in d.get('hits',[])]"
```

Expected result is one or two `E-MTAB-XXXXX` IDs covering (i) scRNA-seq from fresh skin, (ii) 10x Flex from FFPE, and (iii) 10x Visium. Once the accession is known, raw FASTQs / matrices can be pulled via the ENA mirror:

```bash
EMTAB="E-MTAB-XXXXX"   # fill in from the search above
curl -sL "https://www.ebi.ac.uk/biostudies/files/${EMTAB}/Files.json" -o "$PROJECT/raw/${EMTAB}_files.json"
# parse and aria2c -c -x 8 -i <list> to bulk-download
```

If the BioStudies search returns no hits, the data may have been deposited under a partner identifier (HCA/GEO). Search the same query at `https://www.ncbi.nlm.nih.gov/geo/` and check the paper's *Source data* link in the Nature SI: `https://www.nature.com/articles/s41590-024-02018-1#data-availability`.

> **The agent should NOT download raw FASTQs for preliminary analysis.** Use the processed h5ad. Raw downloads are only needed for re-alignment or extending the dataset.

### 1.5 (Optional) The three integrated public CTCL studies

If reproducing the integration is in scope:

```bash
# Liu et al. 2022 Nat Commun — CTCL skin scRNA-seq (GSE165623 covers Rindler 2021 ; Liu 2022 = GSE173205 or PRJNA773770)
# Rindler 2021 Mol Cancer — GSE173205 (advanced MF skin) plus GSE165623 (skin + blood + LN)
# Song 2022 Cancer Discov — transformed CTCL WGS + scRNA
# Pull with sra-tools / GEOparse only if needed; cf. each paper's Data Availability.
```

For preliminary analysis these are *already inside* the integrated h5ad — no separate download required.

### 1.6 Bulk RNA-seq for the B-cell deconvolution validation

```bash
cd "$PROJECT/bulk"

# CTCL bulk RNA-seq (Liu et al. 2022 Blood — PEG10 study) — GSE168508
# AD/psoriasis bulk RNA-seq (Tsoi et al. 2019 JID) — GSE121212

for GSE in GSE168508 GSE121212; do
  curl -sL "https://ftp.ncbi.nlm.nih.gov/geo/series/${GSE:0:6}nnn/${GSE}/suppl/" \
    | grep -oE 'href="[^"]+\.txt\.gz"|href="[^"]+\.csv\.gz"|href="[^"]+\.tar"' \
    | sed -E 's/href="//; s/"$//' \
    | while read f; do
        wget -c "https://ftp.ncbi.nlm.nih.gov/geo/series/${GSE:0:6}nnn/${GSE}/suppl/${f}"
      done
done
```

---

## 2. Preliminary analysis

All Python below assumes the `ctcl` conda env from §1.2 is activated and that the integrated h5ad is at `$PROJECT/processed/ctcl_atlas.h5ad` (rename to match what you actually downloaded). Write each step as a standalone script in `$PROJECT/scripts/` so failures are recoverable.

### 2.1 First-look inspection

`scripts/01_inspect.py`:

```python
import scanpy as sc, anndata as ad, pandas as pd, numpy as np
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.settings.figdir = "results/figures/"

H5AD = "processed/ctcl_atlas.h5ad"
adata = sc.read_h5ad(H5AD)
print(adata)

# Show what's already there from the published object
print("\n=== obs columns ===")
print(adata.obs.columns.tolist())
print("\n=== sample metadata head ===")
print(adata.obs.head(3).T)
print("\n=== cells per disease ===")
print(adata.obs.get("disease", adata.obs.get("Disease")).value_counts(dropna=False))
print("\n=== cells per stage ===")
for c in ["stage", "Stage", "disease_stage"]:
    if c in adata.obs.columns:
        print(adata.obs[c].value_counts(dropna=False)); break
print("\n=== cells per major cell type ===")
for c in ["cell_type", "annotation", "broad_celltype", "celltype"]:
    if c in adata.obs.columns:
        print(adata.obs[c].value_counts().head(20)); break
print(f"\nshape: {adata.shape}  layers: {list(adata.layers.keys())}  obsm: {list(adata.obsm.keys())}")
```

**Expected shape:** roughly `(420_000, 25_000-35_000)`. If the integrated atlas comes split into compartmental h5ads (T cells, APCs, stroma, B cells), download each. If you get only a "minimal" object without raw counts, also fetch the raw-counts variant — many downstream analyses (Milo, infercnv, deconvolution) need integer counts.

### 2.2 Sanity-check QC distributions

`scripts/02_qc.py`:

```python
import scanpy as sc
adata = sc.read_h5ad("processed/ctcl_atlas.h5ad")

# Recompute QC if not present
if "n_counts" not in adata.obs.columns:
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# Per-sample QC
for metric in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
    if metric in adata.obs.columns:
        sc.pl.violin(adata, metric, groupby=adata.obs.columns[
            adata.obs.columns.str.contains("sample|donor|patient", case=False)
        ][0], rotation=90, show=False, save=f"_{metric}.pdf")
```

The paper used a fixed pipeline: ambient RNA correction → doublet removal → `min_genes=200`, `max_pct_mt=15` (cell-type-dependent). The published object is already QC'd; you should observe homogeneous distributions across donors. Sharp outliers indicate sample drop-out or a doublet-rich cluster that survived.

### 2.3 Recompute UMAP / clustering and confirm the published cell-type labels

`scripts/03_umap.py`:

```python
import scanpy as sc
adata = sc.read_h5ad("processed/ctcl_atlas.h5ad")

# If integration is already present (X_scVI / X_harmony / X_pca_harmony), use it
emb_keys = [k for k in adata.obsm if k.startswith("X_") and any(
    s in k.lower() for s in ["scvi","harmony","mnn","scanorama","integrated"])]
if emb_keys:
    rep = emb_keys[0]
else:
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000,
                                batch_key="donor" if "donor" in adata.obs else None)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    sc.external.pp.harmony_integrate(adata, key="donor")
    rep = "X_pca_harmony"

sc.pp.neighbors(adata, use_rep=rep, n_neighbors=30)
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, resolution=1.0, key_added="leiden_r1")

sc.pl.umap(adata, color=["disease", "stage", "leiden_r1"],
           wspace=0.4, save="_overview.png")

# Cross-check the published cell-type column
ct_col = next(c for c in ["cell_type","annotation","broad_celltype","celltype"]
              if c in adata.obs.columns)
sc.pl.umap(adata, color=ct_col, save="_celltypes.png", legend_loc="on data")
adata.write_h5ad("processed/ctcl_atlas.umap.h5ad")
```

Sanity check: the four major lineages from Fig. 1b — T cells, B cells, APCs, fibroblasts/endothelial/keratinocytes — should be visible as distinct islands. B-cell cluster should be small but present in CTCL donors.

### 2.4 Identify malignant T cells (TCR + inferCNV + transcriptomic state)

The paper's combined criterion is the field standard. Reproduce it on the T-cell compartment.

`scripts/04_malignant_t.py`:

```python
import scanpy as sc, infercnvpy as cnv, numpy as np, pandas as pd

adata = sc.read_h5ad("processed/ctcl_atlas.umap.h5ad")

# Subset to T/NK
ct_col = next(c for c in ["cell_type","annotation","broad_celltype","celltype"]
              if c in adata.obs.columns)
tnk_mask = adata.obs[ct_col].astype(str).str.contains(
    "T cell|T_cell|Tc|Th|Treg|NK|ILC", case=False, regex=True)
adata_t = adata[tnk_mask].copy()
print(f"T/NK/ILC subset: {adata_t.shape}")

# inferCNV needs ordered gene positions
cnv.io.genomic_position_from_gtf(
    "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/"
    "Homo_sapiens.GRCh38.110.basic.annotation.gtf.gz",
    adata=adata_t)

# Reference = benign T cells from healthy skin (if present)
ref_cats = [c for c in adata_t.obs["disease"].unique()
            if "healthy" in c.lower() or "normal" in c.lower()]
cnv.tl.infercnv(adata_t,
                reference_key="disease",
                reference_cat=ref_cats,
                window_size=250)
cnv.tl.cnv_score(adata_t)

# Per-donor: cells with cnv_score > donor-specific threshold AND in dominant TCR clone = malignant
# (If TCR data is in obs, build that mask too; otherwise rely on cnv_score + clustering.)
sc.pl.umap(adata_t, color=["cnv_score"], save="_cnv_score.png")
adata_t.write_h5ad("processed/ctcl_tcells_cnv.h5ad")
```

The paper's malignant population recovers extensive CNVs (chr7 gain, chr10/17p loss being common) — the `cnv_score` distribution should be bimodal in CTCL but unimodal in healthy/AD/psoriasis controls.

### 2.5 Reproduce the headline TH2 skewing in advanced disease

`scripts/05_th2_skewing.py`:

```python
import scanpy as sc, numpy as np, pandas as pd
import matplotlib.pyplot as plt

adata_t = sc.read_h5ad("processed/ctcl_tcells_cnv.h5ad")
# Use the paper's published 'is_malignant' column if available; otherwise threshold cnv_score
mal = adata_t.obs.get("is_malignant", adata_t.obs["cnv_score"] > 0.05).astype(bool)
adata_m = adata_t[mal].copy()

# TH classifier on TBX21 / GATA3 / RORC
for g in ["TBX21","GATA3","RORC"]:
    if g not in adata_m.var_names:
        print(f"WARN: {g} not in var_names")
sc.tl.score_genes(adata_m, ["TBX21"],          score_name="Th1_score")
sc.tl.score_genes(adata_m, ["GATA3","IL4","IL13","IL5"], score_name="Th2_score")
sc.tl.score_genes(adata_m, ["RORC","IL17A","IL17F"],     score_name="Th17_score")

# Assign the dominant subtype per cell, then aggregate per donor
scores = adata_m.obs[["Th1_score","Th2_score","Th17_score"]]
adata_m.obs["Th_subtype"] = scores.idxmax(axis=1).str.replace("_score","")
per_donor = (adata_m.obs
             .groupby(["donor","stage"])["Th_subtype"]
             .value_counts(normalize=True)
             .unstack(fill_value=0)
             .reset_index())
print(per_donor.head())

# Wilcoxon: TH2 fraction early vs advanced
from scipy.stats import mannwhitneyu
early = per_donor.loc[per_donor.stage.str.contains("early|<IIB|I-IIA", case=False),"Th2"]
adv   = per_donor.loc[per_donor.stage.str.contains("advanced|IIB|III|IV", case=False),"Th2"]
print("TH2 fraction Wilcoxon U:", mannwhitneyu(early, adv, alternative="less"))
```

Expected: significantly higher TH2 fraction in advanced (`p ≈ 3 × 10⁻⁴` in the paper, *p = 0.0003*).

### 2.6 Reproduce the B-cell enrichment in CTCL vs healthy / AD / psoriasis

`scripts/06_bcells.py`:

```python
import scanpy as sc, pandas as pd

adata = sc.read_h5ad("processed/ctcl_atlas.umap.h5ad")
ct_col = next(c for c in ["cell_type","annotation","broad_celltype","celltype"]
              if c in adata.obs.columns)

# Per-donor B-cell fraction
b_mask = adata.obs[ct_col].astype(str).str.contains("B cell|B_cell|Bcell|plasma", case=False)
adata.obs["is_B"] = b_mask
b_frac = (adata.obs.groupby(["donor","disease"])["is_B"]
          .mean().reset_index())
print(b_frac.groupby("disease")["is_B"].describe())

# Replicate Fig. 7a-c: Wilcoxon CTCL vs each other group
from scipy.stats import mannwhitneyu
ref_groups = ["healthy","AD","AD_nonlesion","AD_lesion","Psoriasis_lesion","Psoriasis_nonlesion"]
ctcl_b = b_frac.loc[b_frac.disease.str.contains("CTCL|MF", case=False),"is_B"]
for g in ref_groups:
    sub = b_frac.loc[b_frac.disease.astype(str).str.contains(g, case=False),"is_B"]
    if len(sub) > 0:
        u, p = mannwhitneyu(ctcl_b, sub, alternative="greater")
        print(f"CTCL vs {g}: U={u:.0f}  p={p:.2e}  n={len(sub)}")
```

Expected: every comparison `p < 10⁻⁴`, CTCL ≫ healthy ≈ AD ≈ psoriasis.

### 2.7 Reproduce the MHC-II⁺ fibroblast (F2) and DC2/moDC_3 enrichment

`scripts/07_stroma_apc.py`:

```python
import scanpy as sc, pandas as pd, milopy.core as milo

adata = sc.read_h5ad("processed/ctcl_atlas.umap.h5ad")
# Subset to stromal + APC compartments
ct_col = next(c for c in ["cell_type","annotation","broad_celltype","celltype"]
              if c in adata.obs.columns)

# F2 / VE3 fibroblast & vascular endothelium check
adata.obs["is_F2_like"] = (
    adata[:, "CD74"].X.toarray().squeeze() > 0
) & (adata[:, "HLA-DRA"].X.toarray().squeeze() > 0)
# (Use the published 'cell_type' fine-grained label if available.)

# Milo differential abundance: CTCL vs healthy/AD/psoriasis
import milopy
adata.obs["disease_simple"] = adata.obs["disease"].replace(
    {"CTCL_early":"CTCL", "CTCL_advanced":"CTCL"})
milopy.core.make_nhoods(adata, prop=0.1)
milopy.core.count_nhoods(adata, sample_col="donor")
milopy.core.DA_nhoods(adata, design="~disease_simple")
milopy.plot.plot_DA_beeswarm(adata, group_by=ct_col)
```

Expected: F2 fibroblast, VE3 vascular endothelial, DC2 and moDC_3 neighborhoods come out positively enriched in CTCL at FDR 10 % — Fig. 5b and Fig. 6b of the paper.

### 2.8 Visium spatial: cell2location deconvolution on one slide

`scripts/08_spatial_celllocation.py`:

```python
import scanpy as sc
import cell2location as c2l

vis = sc.read_h5ad("processed/ctcl_visium.h5ad")  # downloaded in §1.3
sc_ref = sc.read_h5ad("processed/ctcl_atlas.h5ad")
print(vis); print(sc_ref)

# 1) Build the reference signature
from cell2location.models import RegressionModel
RegressionModel.setup_anndata(sc_ref, batch_key="donor",
                              labels_key="cell_type")
mod = RegressionModel(sc_ref)
mod.train(max_epochs=250, batch_size=2500, lr=2e-3)
sc_ref = mod.export_posterior(sc_ref, sample_kwargs={"num_samples":1000,"batch_size":2500})

# 2) Map onto Visium
c2l.models.Cell2location.setup_anndata(vis, batch_key="sample")
c2l_mod = c2l.models.Cell2location(vis, cell_state_df=
    sc_ref.varm["means_per_cluster_mu_fg"].copy(), N_cells_per_location=8)
c2l_mod.train(max_epochs=20000, batch_size=None, train_size=1)
vis = c2l_mod.export_posterior(vis)

# 3) Plot a representative early-CTCL section: expect F2 + B cells co-localising with tumor cells
import matplotlib.pyplot as plt
sc.pl.spatial(vis, library_id=vis.uns["spatial"][0],
              color=["q05_F2","q05_DC2","q05_moDC_3","q05_B_cell","q05_Tumor_T"],
              save="_ctcl_niche.png", ncols=3)
```

The paper's NMF over cell2location abundances yielded a "microenvironment 5" = malignant T + fibroblast + DC + B (Fig. 2c). The colocation should be visible on the spatial map.

### 2.9 Cell–cell communication: CXCL13 → CXCR5 axis between malignant T and B cells

`scripts/09_cellphonedb.py` — uses `liana-py` (Python port of CellPhoneDB):

```python
import scanpy as sc, liana as li

adata = sc.read_h5ad("processed/ctcl_atlas.umap.h5ad")
# Restrict to CTCL samples
adata = adata[adata.obs["disease"].str.contains("CTCL|MF", case=False, na=False)].copy()

# LIANA "rank_aggregate" runs multiple methods (CellPhoneDB, CellChat, NATMI, …) and aggregates
li.mt.rank_aggregate(adata, groupby="cell_type", resource_name="consensus",
                     expr_prop=0.1, verbose=True, use_raw=False)
res = li.pl.tileplot(adata,
                     fill="lr_means",
                     label="cellphone_pvals",
                     label_fun=lambda x: "*" if x < 0.05 else "",
                     top_n=15,
                     source_labels=["Malignant_T"],
                     target_labels=["B_cell","F2","DC2","moDC_3"])
res.savefig("results/figures/lr_malignant_to_TLS_niche.pdf")
```

Confirm: CXCL13–CXCR5, CD40LG–CD40, CD70–CD27, CD58–CD2, CD28–CD86 should rank highly between malignant T and B cells (Fig. 8a).

### 2.10 Drug-target nomination (drug2cell)

If GPU is available and the `drug2cell` package installs cleanly:

```python
import drug2cell as d2c, scanpy as sc
adata = sc.read_h5ad("processed/ctcl_atlas.umap.h5ad")
d2c.score(adata, use_raw=False)
d2c.gex.celltype(adata, groupby="cell_type")
```

Expect rituximab and obinutuzumab (CD20 binders) to score highest on the B-cell cluster, and 15-hydroxyprostaglandin dehydrogenase (HPGD inhibitors) on malignant T cells (Fig. 8d).

---

## 3. Suggested deliverables

After the agent completes §1–§2 it should commit to `$PROJECT/results/`:

1. `00_data_manifest.json` — every file downloaded, with size, sha256, source URL, and date.
2. `01_qc_report.html` — `scanpy.external.pl.report` or hand-rolled HTML showing the per-sample QC distributions vs the paper's Fig. 1b/Extended Data Fig. 1.
3. `02_umap_overview.png` — UMAP coloured by disease, stage, and cell type.
4. `03_th2_skewing.csv` and figure — per-donor TH1/TH2/TH17 fractions with the Wilcoxon p-value.
5. `04_bcell_enrichment.csv` and box-plot — per-donor B-cell fractions across disease groups.
6. `05_milo_da_results.csv` — Milo differential abundance with FDR < 10 % subset (should recover F2, VE3, DC2, moDC_3, B cells).
7. `06_visium_<sample>_celllocation.h5ad` + spatial plots for each section.
8. `07_lr_pairs.csv` — top LIANA ligand–receptor predictions between malignant T and the B-cell / fibroblast / DC niche.
9. `08_drug2cell_topN.csv` — predicted drug targets, with CD20-targeting antibodies expected at the top.
10. `analysis_log.md` — a short narrative noting which paper findings reproduced cleanly and which did not, with diagnostic hypotheses for any divergence (cell-type label drift, normalisation differences, etc.).

---

## 4. Failure modes and recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| `cellatlas.io/ctcl` returns blank page | Single-page app needs JS | Use headless Chrome / Claude in Chrome to load it; capture h5ad URLs from the network panel. Fall back to §1.4 ArrayExpress search. |
| `wget` 403 from Sanger CDN | URL changed; cached path stale | Probe the candidate list in §1.3; if all fail, open a GitHub issue at `haniffalab/webatlas-pipeline` or email `cellgeni@sanger.ac.uk`. |
| `h5py.OSError: object 'X' doesn't exist` when reading h5ad | File is a fragment (e.g. only obs/obsm) | Download the "full" rather than "minimal" h5ad. Re-check the WebAtlas portal for an "all data" link. |
| Out-of-memory on 420 k × 30 k matrix | RAM < 64 GB | (a) Load with `sc.read_h5ad(..., backed="r")` for inspection; (b) subset by compartment before clustering; (c) use `scanpy.experimental.pp.normalize_pearson_residuals` on chunks. |
| `infercnvpy` crashes on chromosome ordering | GTF mismatch | Use the same Ensembl release as the published object (likely v100–v110, GRCh38); pin via `cnv.io.genomic_position_from_gtf(..., gtf_version=110)`. |
| `cell2location` MPS / CPU is too slow | No CUDA GPU | Run on a single Visium section first (one library_id), set `max_epochs=5000`. Full 23-section run requires GPU. |
| Cell-type labels are unfamiliar (e.g. `T_CD4_TH2_TOX` not in the paper) | Atlas uses the Reynolds 2021 skin atlas taxonomy, refined for CTCL | Cross-reference with the WebAtlas viewer's legend, or with Supplementary Table 1 of the paper. |
| TH2-skewing test is not significant | Used `score_genes` on the wrong layer | Re-score on log-normalised counts (`adata.X` after `sc.pp.normalize_total + log1p`), not on raw counts or scVI latents. |

---

## 5. Useful references the agent should keep open

- Paper HTML: <https://www.nature.com/articles/s41590-024-02018-1>
- Paper PDF: <https://www.nature.com/articles/s41590-024-02018-1.pdf>
- WebAtlas portal: <https://collections.cellatlas.io/ctcl>
- WebAtlas pipeline source + examples: <https://github.com/haniffalab/webatlas-pipeline>
- Scanpy docs: <https://scanpy.readthedocs.io>
- cell2location tutorial: <https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html>
- LIANA (CellPhoneDB Python): <https://liana-py.readthedocs.io>
- inferCNVpy: <https://infercnvpy.readthedocs.io>
- Milo / milopy: <https://github.com/MarioniLab/milopy>
- CellTypist (skin atlas reference): <https://www.celltypist.org>
- Reynolds et al. 2021 skin cell atlas (the integration partner, E-MTAB-8142): <https://developmentcellatlas.ncl.ac.uk/datasets/hca_skin/>

---

## 6. Ethics, citation and data-use note

The atlas is open-access under CC-BY 4.0 (Springer Nature Open Access). Any publication or downstream resource must cite:

> Li, R., Strobl, J., Poyner, E. F. M. *et al.* Cutaneous T cell lymphoma atlas reveals malignant TH2 cells supported by a B cell-rich tumor microenvironment. *Nat. Immunol.* **25**, 2320–2330 (2024). https://doi.org/10.1038/s41590-024-02018-1

The newly-generated patient data carry Newcastle and North Tyneside NHS Health Authority Joint Ethics approval (08/H0906/95+5) and Newcastle CEPA Biobank approval (17/NE/0070); raw-level data may be controlled-access in places (EGA-style) — check the BioStudies entry's `released` flag and access conditions before redistribution.

---

*End of instructions. Total expected runtime for §1–§2 on a 32-core / 1× A100 / 128 GB RAM workstation: ~6–10 h (most of it cell2location and Milo). On CPU-only: ~24–48 h, with cell2location restricted to 1–2 representative slides.*
