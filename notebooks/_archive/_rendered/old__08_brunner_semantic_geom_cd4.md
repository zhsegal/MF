# old/08_brunner_semantic_geom_cd4.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/08_brunner_semantic_geom_cd4.ipynb` and is never run._

# SemanticSCVI (geometric) on D3 (Brunner/Vienna 2024, GSE269981) — **CD4** cells

Parallel to `07_d1_semantic_geom_cd4.ipynb` (Chennareddy GSE266862) but
applied to the Brunner et al. 2024 chronic-idiopathic-erythroderma cohort
(68 samples spanning HC, AD, CIE, and Erythrodermic CTCL). Unlike what
`download_GSE269981.md` implied, **GEO `!Sample_characteristics_ch1`
already labels condition directly** — eCTCL doesn't need TCR-based
discovery. We still use TCR clonality to flag malignant cells **within**
eCTCL patients (same dominance rule as 07).

**Cohort filter (this notebook):**
- **Skin only.** Blood samples dropped.
- HC skin (labeled).
- Erythrodermic CTCL skin (labeled).
- AD + CIE samples **dropped** (HC-only benign per chosen design).

**Pipeline (same as 07):**
1. Parse `GSE269981_series_matrix.txt` → per-sample manifest
   (`patient_id`, `tissue`, `assay`, `condition`).
2. Concat skin scRNA samples (HC + eCTCL) from `D3_brunner2024/raw/`.
3. QC, identify CD4 helper T cells (CD3⁺ CD4≥CD8 FOXP3-low).
4. TCR-clonality per eCTCL patient → per-cell malignant flag (cross-check
   that CIE patients are polyclonal is reported but they are dropped).
5. scVI clustering sanity check + dot plot.
6. SemanticSCVI (geometric, Geneformer prior, 200 epochs, coherence_weight=2000).
7. Per-factor separation, top-3 diagonal score, hierarchical clustering.

**Download** — see `download_GSE269981.md`. The notebook expects:
- `data/D3_brunner2024/raw/GSE269981_series_matrix.txt`
- `data/D3_brunner2024/raw/GSM<id>_*_matrix.mtx.gz` (+ barcodes/features)
- `data/D3_brunner2024/raw/GSM<id>_*_filtered_contig_annotations.csv.gz`
- `data/D3_brunner2024/raw/GSM<id>_*_clonotypes.csv.gz`


```python  # cell 1
# ============================================================
# Parameters — knobs match 07_d1_semantic_geom_cd4.ipynb
# ============================================================
import hashlib
import json
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")


def _semantic_cache_slug(kwargs, max_epochs, warmup_epochs, n_epochs_kl_warmup, hvg_top_n, n=10):
    blob = json.dumps(
        {
            "kwargs": dict(sorted(kwargs.items())),
            "max_epochs": max_epochs,
            "warmup_epochs": warmup_epochs,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "hvg_top_n": hvg_top_n,
        },
        default=str, sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


NB_DIR = _resolve_nb_dir()
print(f"NB_DIR = {NB_DIR}")

# ---- D3 (Brunner) paths ----
D3_DIR   = NB_DIR / "data" / "D3_brunner2024"
RAW_DIR  = D3_DIR / "raw"
META_DIR = D3_DIR / "meta"; META_DIR.mkdir(parents=True, exist_ok=True)
SERIES_MATRIX = RAW_DIR / "GSE269981_series_matrix.txt"
CONCAT_H5 = D3_DIR / "processed" / "concat.h5ad"
CD4_H5    = D3_DIR / "processed" / "d3_brunner_cd4_combined.h5ad"
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "d3_brunner_cd4_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_d3_brunner_semantic_geom_cd4"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR = NB_DIR / "tables"; TABLE_DIR.mkdir(exist_ok=True)

# ---- CD4 selection ----
TUMOR_LEIDEN_RESOLUTION = 0.5
DOMINANT_FRAC_THRESH    = 0.05  # dominant clone ≥ 5 % of V(D)J cells per sample
DOMINANT_RATIO_THRESH   = 2.0   # dominant clone ≥ 2× second clone

# ---- Preprocessing (ref values) ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = None

# ---- Shared model size ----
N_LATENT = 10
BATCH_KEY = "donor"

# ---- SemanticSCVI (Geneformer prior) — same as 07 ----
SEMANTIC_GENEFORMER_MAX_EPOCHS = 200
SEMANTIC_GENEFORMER_WARMUP_EPOCHS = 20
SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP = 100
SEMANTIC_GENEFORMER_KWARGS = dict(
    loss_mode="geometric",
    coherence_weight=2000.0,
    n_gene_sample=1024,
    n_latent=N_LATENT,
    n_layers=2,
    n_hidden=128,
    dropout_rate=0.1,
    gene_likelihood="nb",
    weights_positive=True,
    use_batch_norm=False,
)
SEMANTIC_GENEFORMER_CACHE_DIR = MODEL_CACHE_DIR / "semantic_scvi" / _semantic_cache_slug(
    {**SEMANTIC_GENEFORMER_KWARGS, "batch_key": BATCH_KEY},
    SEMANTIC_GENEFORMER_MAX_EPOCHS,
    SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP,
    HVG_TOP_N,
)
print(f"semantic_geom cache_dir: {SEMANTIC_GENEFORMER_CACHE_DIR}")
FORCE_TRAIN_SEMANTIC_GENEFORMER = False

# ---- interpretation (ref values) ----
PER_PROJECTION_N_TOP = 30
CLUSTER_N_TOP        = 500
CLUSTER_MAX_K        = 20
TOP_PER_CLUSTER      = 25
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))   # notebooks/ holds benchmark_helpers.py

import gc, gzip, re
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.io as sio
import torch
import scvi

from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
sc.settings.verbosity = 1
print("scvi", scvi.__version__, "| scanpy", sc.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Step 1 — Parse GSE269981 series matrix → per-sample manifest

Brunner deposits scRNA-seq and scTCR-seq as **separate GSMs** for the same
patient/tissue. GEO fields used:

- `!Sample_title` → `P<id>_<X> (scRNA-seq)` or `(scTCR-seq)` — patient + assay.
- `!Sample_source_name_ch1` → tissue (`Skin` / `Blood`). HC/AD titles use
  `P74_AD1` / `P112_HC1` instead of a tissue word, so we always read tissue
  from `source_name_ch1`.
- `!Sample_characteristics_ch1 "condition: …"` → `HC`, `AD`, `CIE`,
  `Erythrodermic CTCL`.

Mapping: `Erythrodermic CTCL → eCTCL`, others passthrough. Pivot to one row
per `(patient, tissue)` holding both `gsm_gex` and `gsm_vdj`. Keep
`Skin × {HC, eCTCL}` only — drop Blood, AD, CIE.

```python  # cell 4
if not SERIES_MATRIX.exists():
    raise FileNotFoundError(
        f"Missing {SERIES_MATRIX}.  See download_GSE269981.md "
        f"(wget GSE269981_series_matrix.txt.gz; gunzip)."
    )

def _split_tab(line: str) -> list[str]:
    return [t.strip().strip('"') for t in line.rstrip("\n").split("\t")[1:]]

titles = accs = sources = None
char_rows: list[list[str]] = []
with open(SERIES_MATRIX, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            titles = _split_tab(line)
        elif line.startswith("!Sample_geo_accession"):
            accs = _split_tab(line)
        elif line.startswith("!Sample_source_name_ch1"):
            sources = _split_tab(line)
        elif line.startswith("!Sample_characteristics_ch1"):
            char_rows.append(_split_tab(line))

if titles is None or accs is None or sources is None or len(titles) != len(accs):
    raise RuntimeError("series matrix missing !Sample_title / !Sample_geo_accession / !Sample_source_name_ch1")
print(f"parsed {len(accs)} GSMs from series matrix")

# Per-GSM condition: find the !Sample_characteristics_ch1 row whose values
# start with "condition: ". Each row is one characteristic; GSMs share the
# same row layout, so we pick the row covering condition.
def _row_kind(row: list[str]) -> str | None:
    vals = [v for v in row if v]
    if not vals:
        return None
    first = vals[0].lower()
    if first.startswith("condition"):
        return "condition"
    if first.startswith("tissue"):
        return "tissue"
    if first.startswith("cell type"):
        return "cell_type"
    return None

cond_row = next((r for r in char_rows if _row_kind(r) == "condition"), None)
if cond_row is None or len(cond_row) != len(accs):
    raise RuntimeError("no 'condition:' characteristics row matched GSM count")

def _strip_prefix(v: str) -> str:
    return v.split(":", 1)[1].strip() if ":" in v else v.strip()

cond_map = {"HC": "HC", "AD": "AD", "CIE": "CIE",
            "Erythrodermic CTCL": "eCTCL", "eCTCL": "eCTCL"}

title_re = re.compile(
    r"^(?P<pid>P[A-Za-z0-9]+)_[A-Za-z0-9]+\s*\((?P<assay>scRNA-seq|scTCR-seq)\)",
    re.IGNORECASE,
)

rows = []
for gsm, t, src, raw_cond in zip(accs, titles, sources, cond_row):
    m = title_re.search(t)
    pid    = m.group("pid").upper() if m else None
    assay  = m.group("assay") if m else None
    tissue = src.strip().capitalize() if src else None
    cond   = _strip_prefix(raw_cond)
    group  = cond_map.get(cond, cond)
    rows.append({"gsm": gsm, "title": t, "patient_id": pid,
                 "tissue": tissue, "assay": assay,
                 "condition_raw": cond, "group": group})

manifest = pd.DataFrame(rows)
print("\nparse coverage:")
print(f"  matched: {manifest['patient_id'].notna().sum()}/{len(manifest)}")
print("\nby group × tissue × assay:")
print(manifest.groupby(["group","tissue","assay"], dropna=False).size().to_string())

# Pivot: one row per (patient, tissue) with both GSMs
gex = (manifest.query("assay == 'scRNA-seq'")
              [["patient_id","tissue","group","gsm"]]
              .rename(columns={"gsm":"gsm_gex"}))
vdj = (manifest.query("assay == 'scTCR-seq'")
              [["patient_id","tissue","gsm"]]
              .rename(columns={"gsm":"gsm_vdj"}))
samples = gex.merge(vdj, on=["patient_id","tissue"], how="left")

# Keep skin + HC/eCTCL only (drop blood, drop AD, drop CIE per HC-only benign choice)
samples = samples[samples["tissue"] == "Skin"].copy()
samples = samples[samples["group"].isin(["HC", "eCTCL"])].copy()
samples = samples.sort_values(["group","patient_id"]).reset_index(drop=True)
samples.to_csv(META_DIR / "samples.tsv", sep="\t", index=False)

print(f"\nskin samples kept: {len(samples)}")
print(samples.groupby("group").size().to_string())
print(f"\npatients (group → list):")
for g, sub in samples.groupby("group"):
    print(f"  {g}: {sorted(sub['patient_id'].tolist())}")
print(f"\nsaved {META_DIR / 'samples.tsv'}")
```

## Step 2 — Concatenate skin scRNA samples (one-off; cached)

```python  # cell 6
CONCAT_H5.parent.mkdir(parents=True, exist_ok=True)
samples = pd.read_csv(META_DIR / "samples.tsv", sep="\t")
print(f"manifest: {len(samples)} skin samples (HC + eCTCL)")

def _find_one(gsm: str, suffix_keys: tuple[str, ...]) -> Path | None:
    """Look up RAW_DIR for a file whose name starts with GSM and contains any of
    the suffix tokens (matrix.mtx / barcodes.tsv / features.tsv / genes.tsv)."""
    cands = list(RAW_DIR.glob(f"{gsm}_*"))
    for p in cands:
        n = p.name.lower()
        if any(k in n for k in suffix_keys):
            return p
    return None

if CONCAT_H5.exists():
    print(f"using cached {CONCAT_H5}")
    adata = sc.read_h5ad(CONCAT_H5)
else:
    adatas = {}
    for _, row in samples.iterrows():
        pid = row["patient_id"]
        gsm = row["gsm_gex"]
        mtx = _find_one(gsm, ("matrix.mtx",))
        bc  = _find_one(gsm, ("barcodes.tsv",))
        ft  = _find_one(gsm, ("features.tsv", "genes.tsv"))
        if not (mtx and bc and ft):
            print(f"  MISSING gex files for {pid} (gsm={gsm}) — skipping")
            continue
        opener = gzip.open if mtx.suffix == ".gz" else open
        with opener(mtx, "rb") as fh:
            X = sio.mmread(fh).tocsr().T.tocsr()        # cells x genes
        barcodes = pd.read_csv(bc, header=None, sep="\t", compression="infer")[0].astype(str).tolist()
        ft_df = pd.read_csv(ft, header=None, sep="\t", compression="infer")
        if ft_df.shape[1] >= 3:
            ft_df = ft_df.iloc[:, :3]
            ft_df.columns = ["gene_id", "gene_name", "feature_type"]
        elif ft_df.shape[1] == 2:
            ft_df.columns = ["gene_id", "gene_name"]
        else:
            ft_df.columns = ["gene_name"]
            ft_df["gene_id"] = ft_df["gene_name"]
        a = sc.AnnData(X=X.astype(np.float32))
        a.obs_names = [f"{pid}_Skin_{b}" for b in barcodes]
        a.var_names = ft_df["gene_name"].astype(str).tolist()
        a.var["gene_id"]   = ft_df["gene_id"].values
        a.var["gene_name"] = ft_df["gene_name"].values
        a.var_names_make_unique()
        a.obs["patient_id"]       = pid
        a.obs["tissue"]           = row["tissue"]
        a.obs["group"]            = row["group"]
        a.obs["gsm_gex"]          = gsm
        a.obs["gsm_vdj"]          = row["gsm_vdj"] if pd.notna(row["gsm_vdj"]) else ""
        a.obs["sample_id_in_raw"] = f"{pid}_Skin"
        adatas[f"{pid}_Skin"] = a
        print(f"  {pid}_Skin: {a.shape}  group={row['group']}")

    if not adatas:
        raise RuntimeError(
            f"No samples loaded. Check {RAW_DIR} has unpacked GSM_*_matrix.mtx.gz files."
        )

    print(f"\nconcatenating {len(adatas)} samples …")
    adata = sc.concat(adatas, axis=0, join="outer", index_unique=None, fill_value=0)
    adata.obs["donor"] = adata.obs["patient_id"].astype(str)
    adata.layers["raw_counts"] = adata.X.copy()
    adata.write_h5ad(CONCAT_H5)
    print(f"wrote {CONCAT_H5}  | shape {adata.shape}")

print("\nsample breakdown:")
print(adata.obs["group"].value_counts())
print("per-patient cell counts:")
print(adata.obs.groupby(["group","patient_id"]).size().to_string())
```

## Step 4 — Manual annotation: scVI → Leiden → dotplot

Integrate the QC-passed concat with a quick scVI run, cluster the latent
with Leiden, then inspect a T-cell-marker dot plot and hand-pick the
T-cell clusters (mirrors `07_d1_…` Step 2).

```python  # cell 8
# Quick scVI integration for clustering. Uses raw_counts via layer; adata.X untouched.
SCVI_INT_DIR = MODEL_CACHE_DIR / "scvi_int_step4"
SCVI_INT_DIR.parent.mkdir(parents=True, exist_ok=True)

if (SCVI_INT_DIR / "model.pt").exists():
    print(f"loading cached scVI from {SCVI_INT_DIR}")
    scvi_int = scvi.model.SCVI.load(str(SCVI_INT_DIR), adata=adata)
else:
    scvi.model.SCVI.setup_anndata(adata, batch_key="donor", layer="raw_counts")
    scvi_int = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    scvi_int.train(max_epochs=150, early_stopping=True, accelerator="auto")
    scvi_int.save(str(SCVI_INT_DIR), overwrite=True)
    print(f"saved scVI to {SCVI_INT_DIR}")
adata.obsm["X_scVI"] = scvi_int.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI", random_state=0)
sc.tl.leiden(adata, resolution=1.0, random_state=0, key_added="leiden_full")
sc.tl.umap(adata, random_state=0)
print("leiden cluster sizes:", adata.obs["leiden_full"].value_counts().to_dict())

sc.pl.umap(
    adata, color=["leiden_full", "donor", "group", "tissue"],
    ncols=2, wspace=0.3, frameon=False,
)
```

```python  # cell 9
# Log-normalize a side copy for the full-cluster dotplot. Keep adata.X = raw counts.
ad_dp = adata.copy()
ad_dp.X = ad_dp.layers["raw_counts"].copy()
sc.pp.normalize_total(ad_dp, target_sum=1e4)
sc.pp.log1p(ad_dp)

t_markers = [g for g in
    ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "FOXP3", "TRAC", "TRBC1", "TRBC2"]
    if g in ad_dp.var_names]
dp = sc.pl.dotplot(
    ad_dp, t_markers, groupby="leiden_full",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "d3_brunner_full_t_cell_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "d3_brunner_full_t_cell_dotplot.png")
del ad_dp; gc.collect()
```

```python  # cell 10
# ✏️ FILL IN — leiden_full clusters that are T cells (CD3+). All others dropped.
T_CLUSTERS: list[str] = ["0", "1", "2", "3", "8", "9", "12", "37"]
if not T_CLUSTERS:
    print("⚠️ T_CLUSTERS empty — adata unchanged. Fill in the list above and re-run.")
else:
    adata = adata[adata.obs["leiden_full"].astype(str).isin(T_CLUSTERS)].copy()
    print("T-cell subset:", adata.shape)
    print(adata.obs["leiden_full"].value_counts())
```

## Step 4b — Recluster T cells, annotate subtypes, subset CD4 + cache

Re-cluster the T-cell subset on the existing `X_scVI` latent for finer
resolution, then map each `t_leiden` cluster to a cell-type label via the
subtype dot plot. Subset to CD4 and cache the result for downstream steps.


```python  # cell 12
# Re-cluster T-cell subset on the existing scVI latent for finer subtype resolution.
sc.pp.neighbors(adata, use_rep="X_scVI", random_state=0)
sc.tl.leiden(adata, resolution=0.8, random_state=0, key_added="t_leiden")
sc.tl.umap(adata, random_state=0)
print("t_leiden cluster sizes:", adata.obs["t_leiden"].value_counts().to_dict())

# Dotplot of T-subtype markers.
ad_dp = adata.copy()
ad_dp.X = ad_dp.layers["raw_counts"].copy()
sc.pp.normalize_total(ad_dp, target_sum=1e4)
sc.pp.log1p(ad_dp)

subtype_markers = {
    "CD4":       ["CD4"],
    "CD8":       ["CD8A", "CD8B"],
    "Treg":      ["FOXP3", "IL2RA", "CTLA4"],
    "naive":     ["CCR7", "SELL", "LEF1", "TCF7"],
    "memory":    ["IL7R", "CD27", "CD28"],
    "effector":  ["GZMA", "GZMB", "GZMK", "PRF1", "GNLY", "NKG7"],
    "exhausted": ["PDCD1", "LAG3", "TIGIT", "HAVCR2"],
    "prolif":    ["MKI67", "TOP2A", "STMN1"],
}
subtype_markers = {k: [g for g in v if g in ad_dp.var_names] for k, v in subtype_markers.items()}
dp = sc.pl.dotplot(
    ad_dp, subtype_markers, groupby="t_leiden",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "d3_brunner_t_subtype_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "d3_brunner_t_subtype_dotplot.png")
del ad_dp; gc.collect()

# ✏️ EDIT THIS — map t_leiden cluster → cell type. Unmapped → "unknown".
```

```python  # cell 13
t_cluster_to_celltype: dict[str, str] = {
    "0":  "UNK",
    "1":  "UNK",
    "2":  "CD4",
    "3":  "CD4",
    "4":  "CD4",
    "5":  "CD4",
    "6":  "CD4",
    "7":  "CD8",
    "8":  "CD4",
    "9":  "UNK",
    "10": "CD4",
}
adata.obs["t_celltype"] = (
    adata.obs["t_leiden"].astype(str).map(t_cluster_to_celltype).fillna("unknown")
)
print(adata.obs["t_celltype"].value_counts())

sc.pl.umap(
    adata, color=["t_leiden", "t_celltype", "group", "donor"],
    ncols=2, wspace=0.3, frameon=False,
)
```

```python  # cell 14
adata_cd4 = adata[adata.obs["t_celltype"] == "CD4"].copy()
print("CD4 subset:", adata_cd4.shape)
```

```python  # cell 15
# Cache annotated CD4 adata (pre-TCR-rule).
CD4_ANNOT_H5 = D3_DIR / "processed" / "d3_brunner_cd4_annotated.h5ad"
CD4_ANNOT_H5.parent.mkdir(parents=True, exist_ok=True)
adata_cd4.write_h5ad(CD4_ANNOT_H5)
print("wrote", CD4_ANNOT_H5)
```

## Step 5 — TCR clonality: malignant clone within eCTCL patients

eCTCL identity already comes from GEO `condition:` (Step 1) — TCR isn't
needed to *find* eCTCL patients. Within each eCTCL skin sample, the
dominant clone (≥5 % of V(D)J cells, ≥2× second) tags malignant cells
(`is_malignant`); the rest of the CD4 cells in the same patient are
benign. HC patients are not tested (cannot be malignant). The cohort no
longer contains CIE patients (dropped in Step 1).

```python  # cell 17
# Reload cached annotated CD4 adata from end of Step 4.
CD4_ANNOT_H5 = D3_DIR / "processed" / "d3_brunner_cd4_annotated.h5ad"
adata_cd4 = sc.read_h5ad(CD4_ANNOT_H5)
print("loaded", CD4_ANNOT_H5, "|", adata_cd4.shape)
```

```
loaded /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D3_brunner2024/processed/d3_brunner_cd4_annotated.h5ad | (36198, 38224)
```

```python  # cell 18
def _load_vdj_for_sample(gsm_vdj: str):
    """Return (barcode -> clonotype_id, clonotype freq table) for a V(D)J GSM."""
    if not gsm_vdj or (isinstance(gsm_vdj, float) and pd.isna(gsm_vdj)):
        return None, None
    contig_cands = [p for p in RAW_DIR.glob(f"{gsm_vdj}_*")
                    if "filtered_contig_annotations" in p.name.lower()]
    clono_cands  = [p for p in RAW_DIR.glob(f"{gsm_vdj}_*")
                    if "clonotypes" in p.name.lower() and "filtered" not in p.name.lower()]
    if not contig_cands or not clono_cands:
        return None, None
    contig = pd.read_csv(contig_cands[0], usecols=["barcode","raw_clonotype_id"], compression="infer")
    contig = contig.dropna(subset=["raw_clonotype_id"])
    contig = contig[contig["raw_clonotype_id"] != "None"]
    clean_vdj_barcodes = contig["barcode"].astype(str).str.replace(r"-1$", "", regex=True)
    bc2clone = dict(zip(clean_vdj_barcodes, contig["raw_clonotype_id"].astype(str)))
    clonos = pd.read_csv(clono_cands[0], compression="infer")
    clonos = clonos.sort_values("frequency", ascending=False).reset_index(drop=True)
    return bc2clone, clonos


samples_df = pd.read_csv(META_DIR / "samples.tsv", sep="\t")
samples_df = samples_df[samples_df["patient_id"].isin(adata_cd4.obs["patient_id"].astype(str).unique())]

mal_decisions = []
adata_cd4.obs["clonotype_id"]    = ""
adata_cd4.obs["is_malignant"]    = False
adata_cd4.obs["dom_clone_id"]    = ""
adata_cd4.obs["dom_clone_prop"]  = np.nan

for _, row in samples_df.iterrows():
    pid = row["patient_id"]
    group = row["group"]
    bc2clone, clonos = _load_vdj_for_sample(row["gsm_vdj"])
    if bc2clone is None:
        print(f"  {pid} ({group}): NO V(D)J files — clonality skipped")
        mal_decisions.append({
            "patient_id": pid, "group": group,
            "dom_clone_id": "", "dom_clone_prop": 0.0, "dom_ratio": 0.0,
            "is_dominant": False, "n_clones_total": 0,
        })
        continue

    if len(clonos) == 0:
        dom_id, dom_prop, ratio = "", 0.0, 0.0
    else:
        dom_id   = str(clonos.loc[0, "clonotype_id"])
        dom_prop = float(clonos.loc[0, "proportion"]) if "proportion" in clonos.columns else \
                   float(clonos.loc[0, "frequency"]) / float(clonos["frequency"].sum())
        dom_freq = float(clonos.loc[0, "frequency"])
        snd_freq = float(clonos.loc[1, "frequency"]) if len(clonos) > 1 else 0.0
        ratio    = dom_freq / max(snd_freq, 1.0)

    # Apply dominance test only to eCTCL (HC patients can't be malignant)
    is_dominant = (
        (group == "eCTCL")
        and (dom_prop >= DOMINANT_FRAC_THRESH)
        and (ratio   >= DOMINANT_RATIO_THRESH)
    )
    mal_decisions.append({
        "patient_id": pid, "group": group,
        "dom_clone_id": dom_id, "dom_clone_prop": dom_prop, "dom_ratio": ratio,
        "is_dominant": is_dominant, "n_clones_total": len(clonos),
    })

    cell_mask = adata_cd4.obs["patient_id"] == pid
    # obs_names = "<patient>_Skin_<10x_barcode>" — take the trailing barcode, strip "-1"
    bc_short = [n.rsplit("_", 1)[-1].replace("-1", "") for n in adata_cd4.obs_names[cell_mask]]
    clone_ids = [bc2clone.get(bc, "") for bc in bc_short]
    adata_cd4.obs.loc[cell_mask, "clonotype_id"]   = clone_ids
    adata_cd4.obs.loc[cell_mask, "dom_clone_id"]   = dom_id
    adata_cd4.obs.loc[cell_mask, "dom_clone_prop"] = dom_prop
    if is_dominant:
        adata_cd4.obs.loc[cell_mask, "is_malignant"] = pd.Series(
            [c == dom_id and c != "" for c in clone_ids], index=adata_cd4.obs_names[cell_mask]
        )

# Disease label = group (HC / eCTCL); add stage_class for colour mapping
adata_cd4.obs["disease"]     = adata_cd4.obs["group"].astype(str)
adata_cd4.obs["stage_class"] = adata_cd4.obs["disease"].map({
    "HC": "HC", "eCTCL": "advanced"
}).astype(str)

# Print Summary (mirrors 07_d1)
mal_df = pd.DataFrame(mal_decisions)
print("\nper-sample dominance decisions:")
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
    print(mal_df.to_string(index=False))

print("\nmalignant CD4 calls:")
print(adata_cd4.obs["is_malignant"].value_counts())

print("\nby patient:")
print(adata_cd4.obs.groupby("patient_id", observed=True)["is_malignant"].agg(["sum", "size"]).to_string())
```

```
per-sample dominance decisions:
patient_id group dom_clone_id  dom_clone_prop  dom_ratio  is_dominant  n_clones_total
      P112    HC   clonotype1           0.071      3.600        False             183
      P115    HC   clonotype1           0.013      1.000        False             510
      P116    HC   clonotype1           0.025      1.600        False             467
      P121    HC   clonotype1           0.068      1.438        False             238
      P105 eCTCL   clonotype1           0.174      4.245         True             770
      P192 eCTCL   clonotype1           0.765     28.253         True             659
      P196 eCTCL   clonotype1           0.695     12.206         True             227
      P303 eCTCL   clonotype1           0.579      5.206         True             226
       P87 eCTCL   clonotype1           0.994    835.667         True              14
       P94 eCTCL   clonotype1           0.034      1.664        False            3289
       PGS eCTCL   clonotype1           0.468      1.455        False             631

malignant CD4 calls:
is_malignant
False    30699
True      5499
Name: count, dtype: int64

by patient:
             sum  size
patient_id            
P87          152  3471
P94            0  6963
P105         121  1651
P112           0   551
P115           0  1003
P116           0   871
P121           0   561
P192        3782  8253
P196         605  2292
P303         839  4423
PGS            0  6159
```

```python  # cell 19
patients_list_to_remove=['PGS','P94']
adata_cd4=adata_cd4[~adata_cd4.obs['patient_id'].isin(patients_list_to_remove)].copy()
print("\nby patient:")
print(adata_cd4.obs.groupby("patient_id", observed=True)["is_malignant"].agg(["sum", "size"]).to_string())
```

```
by patient:
             sum  size
patient_id            
P87          152  3471
P105         121  1651
P112           0   551
P115           0  1003
P116           0   871
P121           0   561
P192        3782  8253
P196         605  2292
P303         839  4423
```

## Step 9 — Build Geneformer semantic map, HVG-subset

```python  # cell 21
# Attach Ensembl gene_id from any features.tsv / genes.tsv (sc.concat dropped var cols).
if "gene_id" not in adata_cd4.var.columns:
    ft = next(
        (p for p in RAW_DIR.glob("*")
         if any(k in p.name.lower() for k in ("features.tsv", "genes.tsv"))),
        None,
    )
    if ft is None:
        raise FileNotFoundError(f"no features.tsv / genes.tsv under {RAW_DIR}")
    feats = pd.read_csv(ft, header=None, sep="\t", compression="infer")
    if feats.shape[1] >= 2:
        feats = feats.iloc[:, :2]
        feats.columns = ["gene_id", "gene_name"]
    else:
        feats.columns = ["gene_name"]
        feats["gene_id"] = feats["gene_name"]
    sym2ens = dict(zip(feats["gene_name"].astype(str), feats["gene_id"].astype(str)))
    adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]

n_mapped = int(sum(str(g).startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")

# Rebuild map if cache exists with mismatched shape (stale from prior run).
if SEMANTIC_CACHE_GENEFORMER.exists():
    _cached = torch.load(SEMANTIC_CACHE_GENEFORMER, weights_only=False)
    if _cached.shape[0] != adata_cd4.n_vars:
        print(f"stale cache {tuple(_cached.shape)} != n_vars={adata_cd4.n_vars}; rebuilding")
        SEMANTIC_CACHE_GENEFORMER.unlink()
    del _cached

semantic_map = get_or_build_geneformer_map(
    adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
assert semantic_map.shape[0] == adata_cd4.n_vars, (
    f"semantic_map rows {semantic_map.shape[0]} != adata_cd4.n_vars {adata_cd4.n_vars}"
)
print("semantic_map (raw):", tuple(semantic_map.shape))

# Idempotent on re-run: only HVG-subset if not already at/below HVG_TOP_N.
if HVG_TOP_N is not None and adata_cd4.n_vars > HVG_TOP_N:
    sc.pp.highly_variable_genes(adata_cd4, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata_cd4.var["highly_variable"].values
    adata_cd4 = adata_cd4[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
print("after HVG:", adata_cd4.shape, "| semantic_map:", tuple(semantic_map.shape))

if SUBSAMPLE_N is not None and adata_cd4.n_obs > SUBSAMPLE_N:
    sc.pp.subsample(adata_cd4, n_obs=SUBSAMPLE_N, random_state=42)
    print(f"subsampled to {adata_cd4.n_obs} cells")
else:
    print(f"no subsampling (n_obs={adata_cd4.n_obs}, SUBSAMPLE_N={SUBSAMPLE_N})")
```

```
gene_id mapped to Ensembl: 36591/38224
Loading cached Geneformer map from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/d3_brunner_cd4_geneformer.pt
semantic_map (raw): (38224, 1152)
after HVG: (23076, 2500) | semantic_map: (2500, 1152)
no subsampling (n_obs=23076, SUBSAMPLE_N=None)
```

## Step 9b — Optionally drop healthy-control cells before training

```python  # cell 23
# Toggle: True → train on eCTCL only (drop HC); False → keep HC.
EXCLUDE_HC = True

if EXCLUDE_HC:
    n_before = adata_cd4.n_obs
    keep_obs = adata_cd4.obs["disease"].astype(str) != "HC"
    adata_cd4 = adata_cd4[keep_obs].copy()
    print(f"dropped HC: {n_before} -> {adata_cd4.n_obs} cells")
    print(adata_cd4.obs["disease"].value_counts())
else:
    print(f"keeping HC (EXCLUDE_HC=False); n_obs={adata_cd4.n_obs}")
```

```
dropped HC: 23076 -> 20090 cells
disease
eCTCL    20090
Name: count, dtype: int64
```

```python  # cell 24
adata_cd4
```

```
AnnData object with n_obs × n_vars = 20090 × 2500
    obs: 'patient_id', 'tissue', 'group', 'gsm_gex', 'gsm_vdj', 'sample_id_in_raw', 'donor', '_scvi_batch', '_scvi_labels', 'leiden_full', 't_leiden', 't_celltype', 'clonotype_id', 'is_malignant', 'dom_clone_id', 'dom_clone_prop', 'disease', 'stage_class'
    var: 'gene_id', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
    uns: '_scvi_manager_uuid', '_scvi_uuid', 'donor_colors', 'group_colors', 'leiden_full', 'leiden_full_colors', 'neighbors', 't_celltype_colors', 't_leiden', 't_leiden_colors', 'tissue_colors', 'umap', 'hvg'
    obsm: 'X_scVI', 'X_umap'
    layers: 'raw_counts'
    obsp: 'connectivities', 'distances'
```

## Step 10 — Train SemanticSCVI (geometric, Geneformer prior) — GPU

```python  # cell 26
# Local override: flip to True to ignore the on-disk model cache and re-train.
FORCE_TRAIN_SEMANTIC_GENEFORMER = True

model = train_or_load_semantic_scvi(
    adata_cd4,
    semantic_map,
    cache_dir=SEMANTIC_GENEFORMER_CACHE_DIR,
    force_train=FORCE_TRAIN_SEMANTIC_GENEFORMER,
    max_epochs=SEMANTIC_GENEFORMER_MAX_EPOCHS,
    warmup_epochs=SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    n_epochs_kl_warmup=SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP,
    batch_key=BATCH_KEY,
    labels_key=None,
    **SEMANTIC_GENEFORMER_KWARGS,
)
print("trained:", model.is_trained)
```

```
[12:52:02] Training SemanticSCVI (max_epochs=200, warmup_epochs=20, n_epochs_kl_warmup=100)...

Warmup Enabled: Semantic Loss (geometric) OFF for 20 epochs.

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Epoch 200/200: 100%|██████████| 200/200 [04:33<00:00,  1.36s/it, v_num=1, train_loss_step=853, train_loss_epoch=805]      
`Trainer.fit` stopped: `max_epochs=200` reached.

Epoch 200/200: 100%|██████████| 200/200 [04:33<00:00,  1.37s/it, v_num=1, train_loss_step=853, train_loss_epoch=805]
[12:56:36]   SemanticSCVI done in 274.1s
[12:56:36]   saved to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_d3_brunner_semantic_geom_cd4/semantic_scvi/91318ccb2b
trained: True
```

## Step 11 — UMAP of the SemanticSCVI embedding

```python  # cell 28
adata_cd4.obs.head()
```

```
patient_id tissue  group     gsm_gex     gsm_vdj  \
P105_Skin_AAACCTGAGTCAATAG-1       P105   Skin  eCTCL  GSM8331789  GSM8331823   
P105_Skin_AAACCTGGTACTCAAC-1       P105   Skin  eCTCL  GSM8331789  GSM8331823   
P105_Skin_AAACCTGGTCCGAACC-1       P105   Skin  eCTCL  GSM8331789  GSM8331823   
P105_Skin_AAACCTGGTTTAGGAA-1       P105   Skin  eCTCL  GSM8331789  GSM8331823   
P105_Skin_AAACCTGTCAGCAACT-1       P105   Skin  eCTCL  GSM8331789  GSM8331823   

                             sample_id_in_raw donor  _scvi_batch  \
P105_Skin_AAACCTGAGTCAATAG-1        P105_Skin  P105            1   
P105_Skin_AAACCTGGTACTCAAC-1        P105_Skin  P105            1   
P105_Skin_AAACCTGGTCCGAACC-1        P105_Skin  P105            1   
P105_Skin_AAACCTGGTTTAGGAA-1        P105_Skin  P105            1   
P105_Skin_AAACCTGTCAGCAACT-1        P105_Skin  P105            1   

                              _scvi_labels leiden_full t_leiden t_celltype  \
P105_Skin_AAACCTGAGTCAATAG-1             0           9        4        CD4   
P105_Skin_AAACCTGGTACTCAAC-1             0           9        4        CD4   
P105_Skin_AAACCTGGTCCGAACC-1             0           8        2        CD4   
P105_Skin_AAACCTGGTTTAGGAA-1             0           9        4        CD4   
P105_Skin_AAACCTGTCAGCAACT-1             0           9        4        CD4   

                             clonotype_id  is_malignant dom_clone_id  \
P105_Skin_AAACCTGAGTCAATAG-1  clonotype71         False   clonotype1   
P105_Skin_AAACCTGGTACTCAAC-1                      False   clonotype1   
P105_Skin_AAACCTGGTCCGAACC-1                      False   clonotype1   
P105_Skin_AAACCTGGTTTAGGAA-1  clonotype76         False   clonotype1   
P105_Skin_AAACCTGTCAGCAACT-1  clonotype60         False   clonotype1   

                              dom_clone_prop disease stage_class  
P105_Skin_AAACCTGAGTCAATAG-1        0.174497   eCTCL    advanced  
P105_Skin_AAACCTGGTACTCAAC-1        0.174497   eCTCL    advanced  
P105_Skin_AAACCTGGTCCGAACC-1        0.174497   eCTCL    advanced  
P105_Skin_AAACCTGGTTTAGGAA-1        0.174497   eCTCL    advanced  
P105_Skin_AAACCTGTCAGCAACT-1        0.174497   eCTCL    advanced
```

```python  # cell 29
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())
sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
sc.tl.umap(ad_emb, random_state=0)

color = [c for c in ["status","disease","patient_id","stage_class", "is_malignant"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```
<Figure size 2080x1000 with 4 Axes>
```

## Top loading genes per projection (quick look)

```python  # cell 31
W = model.get_loadings().reindex(adata_cd4.var_names)
top_per_factor = {
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist()
    for col in W.columns
}
top_df = pd.DataFrame(top_per_factor)
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
with pd.option_context("display.max_columns", None, "display.width", 250,
                       "display.max_colwidth", 30):
    print(top_df)
top_df
```

```
proj_Z_0  proj_Z_1  proj_Z_2 proj_Z_3 proj_Z_4  proj_Z_5  proj_Z_6    proj_Z_7 proj_Z_8   proj_Z_9
top_1       FCMR      DAB1      GZMA       HP   NKAIN1    FCER1A     COTL1     HMGCLL1  PHYHIPL     UNC13C
top_2       PASK      HBA1      CST7  PPP1R9A    GABRD      CD1C      NME1       CCL13    MYH7B      RAB3B
top_3        MAL     ERBB4      GNLY  COL26A1    UBE2C    LGALS2     PRDX2      GABRB2      CA1      SYCE1
top_4       TCF7     ALAS2     KLRB1    SSUH2     PLK1       LYZ     C1QBP       DUOX2   TCEAL7      LCNL1
top_5       CCR7      HBA2   ALOX5AP     PCK1     BEX1   CLEC10A    IMPDH2   KIAA1549L     SYT4     DNAAF1
top_6       IL7R      GPC6    CORO1B   ADGRB3    SPC25      CPVL      NHP2       TDRD6  CCDC168       EREG
top_7        CD7     NRXN1     DUSP4     ARSF      CCK  SERPINA1     HMGA1      MAMDC2     AARD     ZNF556
top_8        LTB     KANK4     ANXA6     DKK1     CGB7     CCL17      RALA       SCRG1      NPY    ALDH1A2
top_9        LBH  ADAMTSL1      CD69  HSD17B2    CDCA3      CD1E      TUBB       CCL18     FLG2      INHBA
top_10     ISG20      ARSE    IFITM2   CRISP3    CENPA      FCN1      AHCY        HBG1  ANKRD55    S100A12
top_11     NELL2     REEP1      CCL5     EBF2   DLGAP5   HLA-DRA       TXN      SYCP2L    SIRPG      CXCL5
top_12      HPGD     PRSS2     ITM2A  PLEKHD1    GTSE1    FCER1G     LITAF       CDKL4  SPATA17       PENK
top_13     ITM2C    SLC1A2    TENT5C     GJA5   KIF20A  C15orf48     NABP2        PIGR      CA9    SLC12A3
top_14     LIME1      ALPL   TNFRSF4    SFTPC     HMMR  HLA-DPA1     SNRPE      SLC1A7     MYCN       AQP9
top_15      MT1F      EDN3     ITGB7    NPTX2  ONECUT2    CLEC4A      TPI1  AC110995.1    LAIR2      GRIA4
top_16    TUBA4A     ABCB5  C12orf75   SLC6A3    KIF2C    CALHM6     ACOT7        OTOA    FXYD7    ANGPTL7
top_17     FOXP3       HBB    LGALS1    RSPO4    HJURP     IGSF6    MRPL15       FCRL1     AIRE       OLR1
top_18     PDE4D      LGR5    IFITM1     WIF1     NEK2  HLA-DPB1     GAPDH     CYP26B1     RGL4      SCN5A
top_19      AQP3     PCSK1      CD96     LDB3    KIF4A  HLA-DQB1       PHB      RBFOX1    IL17F    SELENOV
top_20       CLU    IGFBP2      GZMB    HIF3A    BIRC5  HLA-DRB1  MARCKSL1         AK8    HMHB1        GP9
top_21       ID3      GPD1    LGALS3     NEFL    CDCA8     CXCL2    RUVBL2       GRIK4   ZNF683      TRPM1
top_22     S1PR1   ZCCHC12      PLP2    CADM2     CD1A  HLA-DQB2     HPRT1      KIF26A      MT3      FOXJ1
top_23    CITED2      HAP1      CAPG     CD8B    CCNA2     CXCL3     YWHAH       RBM20    SARDH      WNT7A
top_24     HACD1     PLIN1    CORO1A    EPHA3     RRM2    TYROBP    RANBP1       FABP3  KIR2DL1      LYPD2
top_25    S100A8    MYBPC1      RHOH    ROPN1    ESCO2  HLA-DQA1      IDH2    KIAA1211     IL9R      PADI4
top_26   CCDC167     ABCA8      PRF1     HHIP    KIF15     MMP12     EIF5A       LYVE1     NCR3     CLEC4F
top_27  SLC9A3R1   PCDHB11      ADSL      PGR   DEPDC1      SPI1    SNRPD1     GALNT15  TRAV8-3  TMPRSS11E
top_28     IFI27      OLAH    SEC11C    CDH18   PIMREG  HLA-DRB5      CDK4       FGF10    PAGE5      MMP10
top_29      SELL     FABP6     HSPA8    DNAH3    FABP7      C1QB    MRPL12     COLEC12      IL4    PPP1R17
top_30    TTC39C      PRR4    CDC123      C4A    KIF14    FCGR2B      ENO1       LYNX1      BLK     IGLON5

        proj_Z_0  proj_Z_1  proj_Z_2 proj_Z_3 proj_Z_4  proj_Z_5  proj_Z_6  \
top_1       FCMR      DAB1      GZMA       HP   NKAIN1    FCER1A     COTL1   
top_2       PASK      HBA1      CST7  PPP1R9A    GABRD      CD1C      NME1   
top_3        MAL     ERBB4      GNLY  COL26A1    UBE2C    LGALS2     PRDX2   
top_4       TCF7     ALAS2     KLRB1    SSUH2     PLK1       LYZ     C1QBP   
top_5       CCR7      HBA2   ALOX5AP     PCK1     BEX1   CLEC10A    IMPDH2   
top_6       IL7R      GPC6    CORO1B   ADGRB3    SPC25      CPVL      NHP2   
top_7        CD7     NRXN1     DUSP4     ARSF      CCK  SERPINA1     HMGA1   
top_8        LTB     KANK4     ANXA6     DKK1     CGB7     CCL17  
... [2997 more chars truncated]
```

### Per-factor heatmap: fraction tumor per quantile bin of Z_k

```python  # cell 33
from sklearn.metrics import roc_auc_score

# Rebuild is_mal / plot_idx / aucs from current z + ad_emb (self-heals stale kernel state).
is_mal = ad_emb.obs["is_malignant"].astype(bool).astype(int).values
assert len(is_mal) == z.shape[0], f"is_mal {len(is_mal)} != z {z.shape[0]}"
n_factors = z.shape[1]
aucs = {}
for k in range(n_factors):
    a = roc_auc_score(is_mal, z[:, k])
    aucs[f"Z_{k}"] = max(a, 1 - a)

MAX_PER_CLASS = 15000
rng = np.random.default_rng(0)
idx_mal = np.where(is_mal == 1)[0]; idx_ben = np.where(is_mal == 0)[0]
if len(idx_mal) > MAX_PER_CLASS: idx_mal = rng.choice(idx_mal, MAX_PER_CLASS, replace=False)
if len(idx_ben) > MAX_PER_CLASS: idx_ben = rng.choice(idx_ben, MAX_PER_CLASS, replace=False)
plot_idx = np.concatenate([idx_mal, idx_ben])

N_BINS = 400
sub_z   = z[plot_idx]
sub_mal = is_mal[plot_idx]

heat = np.zeros((n_factors, N_BINS), dtype=float)
for k in range(n_factors):
    order = np.argsort(sub_z[:, k])
    mal_sorted = sub_mal[order].astype(float)
    heat[k] = np.array([chunk.mean() for chunk in np.array_split(mal_sorted, N_BINS)])

fig, ax = plt.subplots(figsize=(11, 0.45 * n_factors + 1.5))
im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
ax.set_yticks(range(n_factors))
ax.set_yticklabels([f"Z_{k}  (AUROC={aucs[f'Z_{k}']:.3f})" for k in range(n_factors)], fontsize=9)
ax.set_xticks([0, N_BINS // 2, N_BINS - 1])
ax.set_xticklabels(["low Z_k","mid","high Z_k"], fontsize=9)
ax.set_xlabel(f"cells sorted by Z_k (binned into {N_BINS} quantiles)", fontsize=9)
ax.set_title("Per-factor malignant/benign separation — class-balanced", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("fraction malignant in bin")
fig.tight_layout()
out = FIG_DIR / "d3_brunner_semantic_geom_cd4_factor_separation_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d3_brunner_semantic_geom_cd4_factor_separation_heatmap.png

<Figure size 1100x600 with 2 Axes>
```

### Pick top-3 most-separating factors automatically

```python  # cell 35
# Manual override: set MANUAL_FX3/FY3/FZ3 to ints to pick by hand; leave None for auto top-3 by AUROC.
MANUAL_FX3, MANUAL_FY3, MANUAL_FZ3 = 0, 4, 5

top_factors = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:3]
auto_idx = [int(k.split("_")[1]) for k, _ in top_factors]
print("auto top-3 factors (by AUROC):", top_factors)

FX3 = MANUAL_FX3 if MANUAL_FX3 is not None else auto_idx[0]
FY3 = MANUAL_FY3 if MANUAL_FY3 is not None else auto_idx[1]
FZ3 = MANUAL_FZ3 if MANUAL_FZ3 is not None else auto_idx[2]
print(f"using FX3={FX3} (AUROC={aucs[f'Z_{FX3}']:.3f}), "
      f"FY3={FY3} (AUROC={aucs[f'Z_{FY3}']:.3f}), "
      f"FZ3={FZ3} (AUROC={aucs[f'Z_{FZ3}']:.3f})")
```

```
auto top-3 factors (by AUROC): [('Z_8', 0.9115549124021267), ('Z_2', 0.8752785052886982), ('Z_0', 0.854720521705562)]
using FX3=0 (AUROC=0.855), FY3=4 (AUROC=0.682), FZ3=5 (AUROC=0.828)
```

### 2-factor scatter grid for the top-5 factors

```python  # cell 37
from itertools import combinations

FACTORS_GRID = [int(k.split("_")[1]) for k, _ in sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:5]]
pairs = list(combinations(FACTORS_GRID, 2))
print("FACTORS_GRID:", FACTORS_GRID)

ncols = 5
nrows = (len(pairs) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
axes = axes.flatten()

rng_draw = np.random.default_rng(0)
shuf = rng_draw.permutation(len(plot_idx))
sub_mal = is_mal[plot_idx][shuf]
colors = np.where(sub_mal == 1, "tab:red", "tab:blue")

for ax, (fx, fy) in zip(axes, pairs):
    sub_x = z[plot_idx, fx][shuf]; sub_y = z[plot_idx, fy][shuf]
    ax.scatter(sub_x, sub_y, c=colors, s=2, alpha=0.3, linewidths=0, rasterized=True)
    ax.set_xlabel(f"Z_{fx}", fontsize=9); ax.set_ylabel(f"Z_{fy}", fontsize=9)
    ax.set_title(f"Z_{fx} vs Z_{fy}  ({aucs[f'Z_{fx}']:.2f}/{aucs[f'Z_{fy}']:.2f})", fontsize=9)
    ax.tick_params(labelsize=7)
for j in range(len(pairs), len(axes)): axes[j].axis("off")
fig.tight_layout()
out = FIG_DIR / f"d3_brunner_semantic_geom_cd4_scatter_grid_{'_'.join(map(str, FACTORS_GRID))}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
FACTORS_GRID: [8, 2, 0, 5, 6]
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d3_brunner_semantic_geom_cd4_scatter_grid_8_2_0_5_6.png

<Figure size 1500x600 with 10 Axes>
```

### 3-D scatter on (FX3, FY3, FZ3) with diagonal score s3

```python  # cell 39
FX3
```

```
0
```

```python  # cell 40
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import roc_curve

x3 = z[:, FX3]; y3 = z[:, FY3]; zZ3 = z[:, FZ3]
mu_x3, sd_x3 = x3.mean(), x3.std()
mu_y3, sd_y3 = y3.mean(), y3.std()
mu_z3, sd_z3 = zZ3.mean(), zZ3.std()
xz3 = (x3 - mu_x3) / sd_x3
yz3 = (y3 - mu_y3) / sd_y3
zz3 = (zZ3 - mu_z3) / sd_z3
s3 = xz3 + yz3 + zz3
y_true = is_mal

rng3 = np.random.default_rng(0)
shuf3 = rng3.permutation(len(plot_idx))
xx3 = x3[plot_idx][shuf3]; yy3 = y3[plot_idx][shuf3]; zz3p = zZ3[plot_idx][shuf3]
ss3 = s3[plot_idx][shuf3]
cc3 = np.where(is_mal[plot_idx][shuf3] == 1, "tab:red", "tab:blue")

median_s3 = float(np.median(s3))
fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
auc_s3_raw = float(roc_auc_score(y_true, s3))
auc_s3 = max(auc_s3_raw, 1 - auc_s3_raw)

def s_plane(c, xg, yg):
    return sd_z3 * (c - (xg - mu_x3) / sd_x3 - (yg - mu_y3) / sd_y3) + mu_z3

xg = np.linspace(xx3.min(), xx3.max(), 20)
yg = np.linspace(yy3.min(), yy3.max(), 20)
XG, YG = np.meshgrid(xg, yg)
ZG_med  = s_plane(median_s3, XG, YG)
ZG_best = s_plane(best_thr3, XG, YG)
s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

fig = plt.figure(figsize=(14, 7))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axL.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by status   AUROC(s3) = {auc_s3:.3f}", fontsize=10)

axR = fig.add_subplot(1, 2, 2, projection="3d")
sc3 = axR.scatter(xx3, yy3, zz3p, c=ss3, cmap="RdBu_r", vmin=-s_vmax3, vmax=s_vmax3, s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axR.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3})", fontsize=10)
fig.colorbar(sc3, ax=axR, fraction=0.03, pad=0.10).set_label("s3 value")

fig.suptitle(f"D3 (Brunner) 3-D scatter Z_{FX3}, Z_{FY3}, Z_{FZ3} with diagonal score s3", fontsize=12, y=1.02)
fig.tight_layout()
out = FIG_DIR / f"d3_brunner_semantic_geom_cd4_scatter3d_Z{FX3}_Z{FY3}_Z{FZ3}_with_s3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d3_brunner_semantic_geom_cd4_scatter3d_Z0_Z4_Z5_with_s3.png

<Figure size 1400x700 with 3 Axes>
```

### Unsupervised thresholds on s3

```python  # cell 42
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import fisher_exact

def report(name, y_pred):
    acc_a = accuracy_score(y_true, y_pred); acc_b = accuracy_score(y_true, 1 - y_pred)
    if acc_b > acc_a: y_pred = 1 - y_pred; acc = acc_b
    else: acc = acc_a
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return {"rule": name, "accuracy": acc, "precision_tumor": p, "recall_tumor": r,
            "f1_tumor": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "odds_ratio": odds, "fisher_p": pval}

rows3 = []
rows3.append(report("s3 > median(s3)", (s3 > np.median(s3)).astype(int)))
prior3 = y_true.mean()
thr_prior3 = np.quantile(s3, 1 - prior3)
rows3.append(report(f"top {prior3:.0%} of s3 (prior-matched)", (s3 > thr_prior3).astype(int)))
gmm3 = GaussianMixture(n_components=2, random_state=0).fit(s3.reshape(-1, 1))
rows3.append(report("GMM(2) on s3", gmm3.predict(s3.reshape(-1, 1))))
km3 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.c_[xz3, yz3, zz3])
rows3.append(report(f"KMeans(2) on (Z_{FX3},Z_{FY3},Z_{FZ3}) standardized", km3.labels_))
rows3.append(report(f"s3 > {best_thr3:.3f}  (best threshold — supervised)", (s3 > best_thr3).astype(int)))

results3 = pd.DataFrame(rows3)
print(f"AUROC of s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3}):  {auc_s3:.4f}")
print(f"Prior P(tumor):                                   {prior3:.3f}")
print(f"Trivial-majority baseline accuracy:               {max(prior3, 1-prior3):.3f}\n")
with pd.option_context("display.float_format", "{:.4f}".format,
                       "display.max_columns", None, "display.width", 200):
    print(results3.to_string(index=False))
```

```
AUROC of s3 = z(Z_0)+z(Z_4)+z(Z_5):  0.6549
Prior P(tumor):                                   0.274
Trivial-majority baseline accuracy:               0.726

                                     rule  accuracy  precision_tumor  recall_tumor  f1_tumor   TP   FP   FN    TN  odds_ratio  fisher_p
                          s3 > median(s3)    0.5867           0.3604        0.6583    0.4658 3620 6425 1879  8166      2.4486    0.0000
            top 27% of s3 (prior-matched)    0.6888           0.4315        0.4315    0.4315 2373 3126 3126 11465      2.7842    0.0000
                             GMM(2) on s3    0.5105           0.1654        0.1949    0.1790 1072 5408 4427  9183      0.4112    1.0000
  KMeans(2) on (Z_0,Z_4,Z_5) standardized    0.7273           0.5010        0.9434    0.6545 5188 5167  311  9424     30.4254    0.0000
s3 > 0.557  (best threshold — supervised)    0.6490           0.3986        0.5546    0.4638 3050 4602 2449  9989      2.7033    0.0000
```

### Best-AUROC separating plane in (Z_FX3, Z_FY3, Z_FZ3) — gene-direction sweep

Instead of the diagonal `s3 = z(Z_FX3)+z(Z_FY3)+z(Z_FZ3)`, try each candidate **direction** in
the 3-D subspace and pick the one that best separates malignant vs benign CD4:

- **Per-gene** direction: OLS coefficients of log-normalized expression on standardized
  `(z_FX3, z_FY3, z_FZ3)`.
- **Pseudotime** direction: same OLS, with `sc.tl.dpt` pseudotime as the target.
- **Diagonal** `(1,1,1)` — the existing `s3` baseline.
- **Logistic-regression** coefficients on `(Z_FX3, Z_FY3, Z_FZ3)` — *optimal* linear separator
  in this 3-D subspace, an upper bound for any single-direction score.

For each direction we project cells onto it (standardized axes), find the best
Youden-J threshold, and report AUROC / accuracy / precision / recall / F1 plus the
plane normal `(vx, vy, vz)` and intercept `thr`.


```python  # cell 44
import scanpy as sc

# 1) Pseudotime via DPT on the latent kNN graph (ad_emb already has neighbors from UMAP cell).
ROOT_IS_MALIGNANT = False   # editable: root DPT at a benign cell
_root_mask = ad_emb.obs["is_malignant"].astype(bool).values == ROOT_IS_MALIGNANT
assert _root_mask.any(), f"no cell with is_malignant=={ROOT_IS_MALIGNANT}"
root_idx = int(np.where(_root_mask)[0][0])
ad_emb.uns["iroot"] = root_idx
sc.tl.diffmap(ad_emb, random_state=0)
sc.tl.dpt(ad_emb)
pt = ad_emb.obs["dpt_pseudotime"].to_numpy()
pt = np.where(np.isfinite(pt), pt, np.nan)
print(f"pseudotime: root_idx={root_idx}  min={np.nanmin(pt):.3f}  median={np.nanmedian(pt):.3f}  max={np.nanmax(pt):.3f}  n_nan={np.isnan(pt).sum()}")
```

```
pseudotime: root_idx=0  min=0.000  median=0.245  max=1.000  n_nan=0
```

```python  # cell 45
# 2) Editable gene list — default = top 4 from each of Z_FX3, Z_FY3, Z_FZ3 (from top_df).
GENES_FOR_PLANE = (
    top_df[f"proj_Z_{FX3}"].head(4).tolist()
    + top_df[f"proj_Z_{FY3}"].head(4).tolist()
    + top_df[f"proj_Z_{FZ3}"].head(4).tolist()
)
GENES_FOR_PLANE = list(dict.fromkeys(GENES_FOR_PLANE))
# GENES_FOR_PLANE = ["CCR4", "SELL", "CCL5", "NKG7", "GZMA", "GZMH", "CST7", "CXCL13"]
missing = [g for g in GENES_FOR_PLANE if g not in adata_cd4.var_names]
assert not missing, f"missing in adata_cd4: {missing}"
print(f"{len(GENES_FOR_PLANE)} genes:", GENES_FOR_PLANE)
```

```
12 genes: ['FCMR', 'PASK', 'MAL', 'TCF7', 'NKAIN1', 'GABRD', 'UBE2C', 'PLK1', 'FCER1A', 'CD1C', 'LGALS2', 'LYZ']
```

```python  # cell 46
# 3) Candidate direction vectors in (Z_FX3, Z_FY3, Z_FZ3) — OLS coeffs on standardized axes.
from numpy.linalg import lstsq
from sklearn.linear_model import LogisticRegression
from scipy.sparse import issparse

Z3  = np.column_stack([z[:, FX3], z[:, FY3], z[:, FZ3]])
Z3z = np.column_stack([xz3, yz3, zz3])
A   = np.column_stack([Z3z, np.ones(len(Z3z))])

X_raw = adata_cd4.X
size_factor = np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float64)
size_factor[size_factor == 0] = 1.0

def gene_lognorm(name: str) -> np.ndarray:
    j = adata_cd4.var_names.get_loc(name)
    col = X_raw[:, j]
    col = col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()
    return np.log1p(col / size_factor * 1e4).astype(np.float64)

def fit_direction(y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    coef, *_ = lstsq(A[mask], y[mask], rcond=None)
    return coef[:3]

dir_rows = [("diagonal_(1,1,1)", np.array([1.0, 1.0, 1.0]))]
dir_rows.append(("pseudotime", fit_direction(pt)))
for g in GENES_FOR_PLANE:
    dir_rows.append((g, fit_direction(gene_lognorm(g))))
lr = LogisticRegression(max_iter=1000, C=1.0).fit(Z3z, is_mal)
dir_rows.append((f"LR_optimal_(Z{FX3},Z{FY3},Z{FZ3})", lr.coef_.ravel().astype(np.float64)))

dirs = {name: v / max(np.linalg.norm(v), 1e-12) for name, v in dir_rows}
print(f"{len(dirs)} candidate directions:", list(dirs.keys()))
```

```
15 candidate directions: ['diagonal_(1,1,1)', 'pseudotime', 'FCMR', 'PASK', 'MAL', 'TCF7', 'NKAIN1', 'GABRD', 'UBE2C', 'PLK1', 'FCER1A', 'CD1C', 'LGALS2', 'LYZ', 'LR_optimal_(Z0,Z4,Z5)']
```

```python  # cell 47
# 4) Score each direction: AUROC + Youden-best threshold + accuracy stats.
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_fscore_support
from scipy.stats import fisher_exact

def eval_direction(name: str, v: np.ndarray) -> dict:
    s = Z3z @ v
    auc = roc_auc_score(y_true, s)
    flipped = auc < 0.5
    if flipped:
        s, auc = -s, 1 - auc
        v = -v
    fpr, tpr, thrs = roc_curve(y_true, s)
    j = np.argmax(tpr - fpr)
    best_thr = float(thrs[j])
    y_pred = (s > best_thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0,
    )
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return dict(
        name=name, AUROC=auc, acc=acc, prec=pr, rec=rc, f1=f1,
        TP=tp, FP=fp, FN=fn, TN=tn, OR=float(odds), p=float(pval),
        vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
        thr=best_thr, flipped=flipped,
    )

results_plane = (
    pd.DataFrame([eval_direction(n, v) for n, v in dirs.items()])
    .sort_values("AUROC", ascending=False)
    .reset_index(drop=True)
)
with pd.option_context("display.max_columns", None, "display.width", 200,
                       "display.float_format", "{:.4f}".format):
    print(results_plane.to_string(index=False))
out_csv = FIG_DIR / "d3_brunner_semantic_geom_cd4_gene_planes_stats.csv"
results_plane.to_csv(out_csv, index=False)
print("saved", out_csv)
results_plane
```

```
name  AUROC    acc   prec    rec     f1   TP   FP   FN    TN      OR      p      vx      vy      vz     thr  flipped
LR_optimal_(Z0,Z4,Z5) 0.9139 0.8209 0.6253 0.8623 0.7249 4742 2842  757 11749 25.8966 0.0000  0.6781  0.2199 -0.7013  0.5557    False
                 TCF7 0.9112 0.8220 0.6292 0.8511 0.7235 4680 2758  819 11833 24.5167 0.0000  0.7988  0.1528 -0.5818  0.5174    False
                 PASK 0.9056 0.8171 0.6224 0.8434 0.7162 4638 2814  861 11777 22.5444 0.0000  0.8740  0.0845 -0.4785  0.4529    False
                  MAL 0.8993 0.7987 0.5906 0.8622 0.7010 4741 3287  758 11304 21.5096 0.0000  0.8423 -0.0614 -0.5355  0.3737    False
                 FCMR 0.8890 0.7870 0.5747 0.8531 0.6867 4691 3472  808 11119 18.5926 0.0000  0.9150 -0.0937 -0.3925  0.2814    False
           pseudotime 0.8803 0.7342 0.5082 0.9007 0.6498 4953 4793  546  9798 18.5441 0.0000  0.4468 -0.2099 -0.8697  0.2355     True
                  LYZ 0.8497 0.7180 0.4920 0.9284 0.6431 5105 5271  394  9320 22.9099 0.0000  0.1449 -0.1037 -0.9840  0.0950     True
               LGALS2 0.8275 0.6986 0.4741 0.9287 0.6278 5107 5664  392  8927 20.5335 0.0000  0.1174 -0.2058 -0.9715 -0.0347     True
                 CD1C 0.7672 0.6616 0.4360 0.8040 0.5653 4421 5720 1078  8871  6.3603 0.0000  0.2027 -0.5236 -0.8275  0.0522     True
               FCER1A 0.7560 0.6597 0.4333 0.7896 0.5595 4342 5679 1157  8912  5.8892 0.0000  0.1666 -0.5304 -0.8313  0.0678     True
                GABRD 0.6908 0.6155 0.3877 0.6983 0.4986 3840 6065 1659  8526  3.2539 0.0000 -0.2078  0.9335 -0.2922 -0.0266    False
     diagonal_(1,1,1) 0.6549 0.6490 0.3986 0.5546 0.4638 3050 4602 2449  9989  2.7033 0.0000  0.5774  0.5774  0.5774  0.3214    False
                 PLK1 0.6473 0.5782 0.3557 0.6665 0.4638 3665 6640 1834  7951  2.3929 0.0000 -0.2938  0.9142 -0.2791 -0.0693    False
                UBE2C 0.6318 0.5494 0.3405 0.6898 0.4559 3793 7346 1706  7245  2.1928 0.0000 -0.3323  0.8970 -0.2916 -0.1457    False
               NKAIN1 0.5921 0.4853 0.3188 0.7741 0.4516 4257 9098 1242  5493  2.0694 0.0000 -0.6105  0.2409 -0.7545 -0.3879    False
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d3_brunner_semantic_geom_cd4_gene_planes_stats.csv

                     name     AUROC       acc      prec       rec        f1  \
0   LR_optimal_(Z0,Z4,Z5)  0.913882  0.820856  0.625264  0.862339  0.724910   
1                    TCF7  0.911248  0.821951  0.629201  0.851064  0.723506   
2                    PASK  0.905588  0.817073  0.622383  0.843426  0.716238   
3                     MAL  0.899288  0.798656  0.590558  0.862157  0.700968   
4                    FCMR  0.888959  0.786959  0.574666  0.853064  0.686722   
5              pseudotime  0.880288  0.734246  0.508208  0.900709  0.649787   
6                     LYZ  0.849688  0.718019  0.492001  0.928351  0.643150   
7                  LGALS2  0.827482  0.698556  0.474144  0.928714  0.627781   
8                    CD1C  0.767163  0.661623  0.435953  0.803964  0.565345   
9                  FCER1A  0.755973  0.659731  0.433290  0.789598  0.559536   
10                  GABRD  0.690848  0.615530  0.387683  0.698309  0.498572   
11       diagonal_(1,1,1)  0.654911  0.649029  0.398589  0.554646  0.463843   
12                   PLK1  0.647324  0.578198  0.355653  0.666485  0.463807   
13                  UBE2C  0.631809  0.549428  0.340515  0.689762  0.455944   
14                 NKAIN1  0.592075  0.485316  0.318757  0.774141  0.451575   

      TP    FP    FN     TN         OR              p        vx        vy  \
0   4742  2842   757  11749  25.896585   0.000000e+00  0.678074  0.219877   
1   4680  2758   819  11833  24.516731   0.000000e+00  0.798825  0.152775   
2   4638  2814   861  11777  22.544374   0.000000e+00  0.873996  0.084519   
3   4741  3287   758  11304  21.509643   0.000000e+00  0.842316 -0.061422   
4   4691  3472   808  11119  18.592598   0.000000e+00  0.914975 -0.093663   
5   4953  4793   
... [1295 more chars truncated]
```

```python  # cell 48
# 5) 3-D scatter: top-AUROC plane (left = is_malignant, right = top-gene expression).
top_row  = results_plane.iloc[0]
top_name = top_row["name"]
v_top    = np.array([top_row["vx"], top_row["vy"], top_row["vz"]], dtype=float)
v_top    = v_top / np.linalg.norm(v_top)
thr_top  = float(top_row["thr"])

def plane_z(v, thr, x_raw, y_raw):
    xz_ = (x_raw - mu_x3) / sd_x3
    yz_ = (y_raw - mu_y3) / sd_y3
    if abs(v[2]) < 1e-6:
        return np.full_like(x_raw, np.nan, dtype=float)
    zz_ = (thr - v[0] * xz_ - v[1] * yz_) / v[2]
    return sd_z3 * zz_ + mu_z3

xg2 = np.linspace(xx3.min(), xx3.max(), 20)
yg2 = np.linspace(yy3.min(), yy3.max(), 20)
XG2, YG2 = np.meshgrid(xg2, yg2)
ZG_top = plane_z(v_top, thr_top, XG2, YG2)

# Right-panel colour: pick a top gene of FX3 (or override).
RIGHT_GENE = top_df[f"proj_Z_{FX3}"].iloc[0]
assert RIGHT_GENE in adata_cd4.var_names
right_expr = gene_lognorm(RIGHT_GENE)
right_plot = right_expr[plot_idx][shuf3]
right_vmax = float(np.quantile(right_expr, 0.99))

fig = plt.figure(figsize=(14, 7))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG2, YG2, ZG_top, color="tab:green", alpha=0.25, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by is_malignant   top: {top_name}   AUROC={top_row['AUROC']:.3f}", fontsize=10)
status_handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",   label="malignant", markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue",  label="benign",    markersize=6),
    plt.Line2D([], [], marker="s", linestyle="", color="tab:green", alpha=0.25,
               label=f"{top_name} best (AUROC={top_row['AUROC']:.3f})", markersize=10),
]
axL.legend(handles=status_handles, loc="upper left", fontsize=8, frameon=False)

axR = fig.add_subplot(1, 2, 2, projection="3d")
scR = axR.scatter(xx3, yy3, zz3p, c=right_plot, cmap="viridis",
                  vmin=0, vmax=right_vmax, s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG2, YG2, ZG_top, color="tab:green", alpha=0.25, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by {RIGHT_GENE} log-norm expression", fontsize=10)
fig.colorbar(scR, ax=axR, fraction=0.03, pad=0.10).set_label(f"{RIGHT_GENE} log1p(cp10k)")

fig.suptitle(f"D3 Brunner — best-direction plane: {top_name}", fontsize=12, y=1.02)
fig.tight_layout()
safe = top_name.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "_").replace("/", "_")
out = FIG_DIR / f"d3_brunner_semantic_geom_cd4_scatter3d_best_plane_{safe}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d3_brunner_semantic_geom_cd4_scatter3d_best_plane_LR_optimal_Z0_Z4_Z5.png

<Figure size 1400x700 with 3 Axes>
```

```python  # cell 49
# 6) Interactive rotatable 3-D version of the top-AUROC plane.
# Writes a standalone .html (open in a browser, click+drag to rotate).
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import IFrame

mal_mask = is_mal[plot_idx][shuf3] == 1
ben_mask = ~mal_mask

fig3d = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        f"colored by is_malignant   top: {top_name}   AUROC={top_row['AUROC']:.3f}",
        f"colored by {RIGHT_GENE} log-norm expression",
    ),
    horizontal_spacing=0.05,
)

fig3d.add_trace(go.Scatter3d(
    x=xx3[mal_mask], y=yy3[mal_mask], z=zz3p[mal_mask],
    mode="markers", name="malignant",
    marker=dict(size=2, color="crimson", opacity=0.45),
), row=1, col=1)
fig3d.add_trace(go.Scatter3d(
    x=xx3[ben_mask], y=yy3[ben_mask], z=zz3p[ben_mask],
    mode="markers", name="benign",
    marker=dict(size=2, color="royalblue", opacity=0.45),
), row=1, col=1)
fig3d.add_trace(go.Surface(
    x=XG2, y=YG2, z=ZG_top, showscale=False, opacity=0.35,
    colorscale=[[0, "green"], [1, "green"]],
    name=f"{top_name} best (AUROC={top_row['AUROC']:.3f})", showlegend=True,
), row=1, col=1)

fig3d.add_trace(go.Scatter3d(
    x=xx3, y=yy3, z=zz3p, mode="markers",
    marker=dict(
        size=2, color=right_plot, colorscale="Viridis",
        cmin=0, cmax=right_vmax, opacity=0.55,
        colorbar=dict(title=f"{RIGHT_GENE} log1p(cp10k)", thickness=12, x=1.02),
    ),
    name=f"{RIGHT_GENE}", showlegend=False,
), row=1, col=2)
fig3d.add_trace(go.Surface(
    x=XG2, y=YG2, z=ZG_top, showscale=False, opacity=0.35,
    colorscale=[[0, "green"], [1, "green"]],
    name="top plane", showlegend=False,
), row=1, col=2)

axes_kw = dict(xaxis_title=f"Z_{FX3}", yaxis_title=f"Z_{FY3}", zaxis_title=f"Z_{FZ3}")
fig3d.update_layout(
    title=f"D3 Brunner — best-direction plane: {top_name}",
    width=1300, height=700, scene=axes_kw, scene2=axes_kw,
    legend=dict(itemsizing="constant"),
)

html_out = FIG_DIR / f"d3_brunner_semantic_geom_cd4_scatter3d_best_plane_{safe}.html"
fig3d.write_html(html_out, include_plotlyjs="cdn", full_html=True)
print("saved", html_out)
IFrame(html_out.as_posix(), width=1320, height=720)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d3_brunner_semantic_geom_cd4_scatter3d_best_plane_LR_optimal_Z0_Z4_Z5.html

<IPython.lib.display.IFrame at 0x145db8bb35c0>
```

## Step 13 — Gene loadings + hierarchical clustering

```python  # cell 51
W = model.get_loadings().reindex(adata_cd4.var_names)
adata_cd4.varm["W_d3_brunner_semantic_geom_cd4"] = W.values
adata_cd4.uns["W_d3_brunner_semantic_geom_cd4_columns"] = list(W.columns)
print("W (loadings):", W.shape, "| columns:", list(W.columns))

top_per_factor = {
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist()
    for col in W.columns
}
top_df = pd.DataFrame(top_per_factor)
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
top_df.to_csv(TABLE_DIR / "d3_brunner_semantic_geom_cd4_top_genes_per_factor.tsv", sep="\t")
print("\ntop genes per factor:")
with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 30):
    print(top_df.head(15))
```

```python  # cell 52
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns

max_vals = W.abs().max(axis=1)
top_idx = max_vals.sort_values(ascending=False).head(CLUSTER_N_TOP).index
subset = W.loc[top_idx]
keep = subset.values.std(axis=1) > 0
subset = subset.loc[keep]; top_idx = subset.index

dists = pdist(subset.values, metric="correlation")
dist_matrix = squareform(dists)
Z = linkage(dists, method="average")

best_k, best_score = 2, -1.0
for k in range(2, CLUSTER_MAX_K + 1):
    labels = fcluster(Z, t=k, criterion="maxclust")
    try:
        s = silhouette_score(dist_matrix, labels, metric="precomputed")
    except ValueError:
        continue
    if s > best_score: best_score, best_k = s, k
final_labels = fcluster(Z, t=best_k, criterion="maxclust")
print(f"hierarchical clusters: k={best_k} (silhouette={best_score:.3f}) over {len(top_idx)} genes")

clust = pd.DataFrame({"Cluster": final_labels, "max_loading": max_vals.loc[top_idx].values}, index=top_idx)
for c in sorted(clust["Cluster"].unique()):
    members = clust[clust["Cluster"] == c].sort_values("max_loading", ascending=False)
    print(f"\n=== cluster {c}  (n={len(members)}) ===")
    print(", ".join(members.head(TOP_PER_CLUSTER).index.tolist()))

corr = np.corrcoef(subset.values)
g = sns.clustermap(
    pd.DataFrame(corr, index=top_idx, columns=top_idx),
    row_linkage=Z, col_linkage=Z, cmap="vlag", center=0,
    xticklabels=False, yticklabels=False, figsize=(8, 8),
)
g.fig.suptitle(f"D3 Brunner semantic_geom CD4: top-{len(top_idx)} gene loading correlation (k={best_k})", y=1.01)
out = FIG_DIR / "d3_brunner_semantic_geom_cd4_gene_hierarchy.png"
g.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
```
