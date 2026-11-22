# old/08_d2_semantic_geom_cd4.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/08_d2_semantic_geom_cd4.ipynb` and is never run._

# SemanticSCVI (geometric) on D2 (Johnson/Timp 2025) — **CD4** cells

⚠️ **D2 data is PENDING.** Per the medRxiv preprint
(`10.1101/2025.09.07.25335167`, posted 2025-09-10):

> "The expression profile data analyzed in this study are pending submission
> to the Gene Expression Omnibus (GEO)."

Until data is received, this notebook halts at Step 1. To unblock:

1. Email **Courtney M. Johnson** (`cjohn263@jhmi.edu`) requesting the
   processed count matrices for the 5 skin samples (1 HC + 4 MF).
2. Place the files into `data/D2_johnson_timp2025/raw/` once received.
3. Update `data/D2_johnson_timp2025/meta/access_log.md` with the status.

**Cohort filter (advanced MF skin + HC):**
Per preprint Fig 1A, the 5 samples are:

| Sample | Stage | Race | In analysis? |
|---|---|---|---|
| Control 1 | HC | White | ✅ benign reference |
| MF 2 | IIB (late) | White | ✅ advanced MF |
| MF 3 | IB (early) | White | ❌ early-stage |
| MF 4 | IA (early) | Black | ❌ early-stage |
| MF 5 | IIB (late) | Black | ✅ advanced MF |

Three samples remain after the advanced-skin filter (1 HC + 2 late MF).

**Malignant CD4 calling — different from D1.**
D2 has **no V(D)J** (preprint §3.7: *"No V(D)J / TCR sequencing"*) and the
paper focuses on fibroblasts, not T-cell malignancy — so there is no
author-provided malignant T-cell label even when data arrives. We fall
back to the **ref's manual cluster-pick** workflow (`06_semantic_geom_cd4`
cells 4–8): cluster CD4 T cells on a standard scVI latent, dot plot
CD4 / CD8 / FOXP3 + composition by MF-vs-HC origin, hand-pick the clusters
dominated by MF samples as malignant.


```python  # cell 1
# ============================================================
# Parameters — knobs match 06_semantic_geom_cd4.ipynb (the ref)
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
        {"kwargs": dict(sorted(kwargs.items())),
         "max_epochs": max_epochs, "warmup_epochs": warmup_epochs,
         "n_epochs_kl_warmup": n_epochs_kl_warmup, "hvg_top_n": hvg_top_n},
        default=str, sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


NB_DIR = _resolve_nb_dir()
print(f"NB_DIR = {NB_DIR}")

# ---- D2 paths ----
D2_DIR    = NB_DIR / "data" / "D2_johnson_timp2025"
RAW_DIR   = D2_DIR / "raw"
META_TSV  = D2_DIR / "meta" / "patients.tsv"
CONCAT_H5 = D2_DIR / "processed" / "concat.h5ad"
CD4_H5    = D2_DIR / "processed" / "d2_cd4_combined.h5ad"
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "d2_cd4_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_d2_semantic_geom_cd4"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)

# ---- CD4 selection ----
TUMOR_LEIDEN_RESOLUTION = 0.5

# ---- Preprocessing (ref values) ----
HVG_TOP_N = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = None

# ---- Model size + batch ----
N_LATENT = 10
BATCH_KEY = "donor"

# ---- SemanticSCVI (Geneformer prior) — identical to ref ----
SEMANTIC_GENEFORMER_MAX_EPOCHS = 200
SEMANTIC_GENEFORMER_WARMUP_EPOCHS = 20
SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP = 100
SEMANTIC_GENEFORMER_KWARGS = dict(
    loss_mode="geometric",
    coherence_weight=2000.0,
    n_gene_sample=1024,
    n_latent=N_LATENT,
    n_layers=2, n_hidden=128, dropout_rate=0.1,
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

PER_PROJECTION_N_TOP = 30
CLUSTER_N_TOP        = 500
CLUSTER_MAX_K        = 20
TOP_PER_CLUSTER      = 25
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))

import gc, gzip
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import scvi

from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
sc.settings.verbosity = 1
print("scvi", scvi.__version__, "| scanpy", sc.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Step 1 — Load raw data (HALT if data not yet received)

D2 data is pending GEO submission. This cell halts if `raw/` is empty.

```python  # cell 4
patients = pd.read_csv(META_TSV, sep="\t")
print("planned cohort:")
print(patients[["sample_id", "disease_stage", "is_advanced_mf_skin"]].to_string(index=False))

raw_files = sorted(RAW_DIR.glob("*"))
if not raw_files:
    print("\n" + "=" * 60)
    print("D2 RAW DATA NOT FOUND — notebook halts here.")
    print("=" * 60)
    print(f"Expected files in: {RAW_DIR}")
    print("\nPer preprint Data Availability Statement:")
    print('  "The expression profile data analyzed in this study are')
    print('   pending submission to the Gene Expression Omnibus (GEO)."')
    print("\nTo unblock: email Courtney M. Johnson (cjohn263@jhmi.edu) to request data.")
    print("See data/D2_johnson_timp2025/meta/access_log.md for the contact log.")
    raise SystemExit(0)

print(f"\nfound {len(raw_files)} files in {RAW_DIR}")
for f in raw_files[:10]:
    print(" ", f.name)
```

## Step 2 — Concatenate samples

Once data arrives, this cell handles three likely deposit formats
(see `CTCL_datasets_instructions.md` §3.3):

1. **Integrated `.h5ad`** (most likely if Hicks lab follows their Bioconductor pattern).
2. **Integrated `.rds`** (Seurat v4) — must be converted to `.h5ad` via `sceasy` in R first.
3. **Per-sample 10x Cell Ranger output** — concat as we do for D1.

Edit the format selector below once you know what arrived.

```python  # cell 6
CONCAT_H5.parent.mkdir(parents=True, exist_ok=True)

# Detect format
h5ad_candidates = list(RAW_DIR.glob("*.h5ad"))
rds_candidates  = list(RAW_DIR.glob("*.rds"))
per_sample_h5   = list(RAW_DIR.glob("*_filtered_feature_bc_matrix.h5"))
per_sample_mtx  = list(RAW_DIR.glob("*matrix.mtx*"))

if CONCAT_H5.exists():
    print(f"using cached {CONCAT_H5}")
    adata = sc.read_h5ad(CONCAT_H5)
elif h5ad_candidates:
    src = h5ad_candidates[0]
    print(f"loading integrated h5ad: {src}")
    adata = sc.read_h5ad(src)
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
elif rds_candidates:
    raise RuntimeError(
        f"Found .rds in {RAW_DIR}. Convert to .h5ad in R first:\n"
        "  library(sceasy); sceasy::convertFormat(<rds>, from='seurat', to='anndata', outFile='<...>.h5ad')"
    )
elif per_sample_h5 or per_sample_mtx:
    print("per-sample format — concatenating 10x output")
    adatas = {}
    for h5 in per_sample_h5:
        sid = h5.stem.replace("_filtered_feature_bc_matrix", "")
        a = sc.read_10x_h5(h5)
        a.var_names_make_unique()
        a.obs["sample_id"] = sid
        adatas[sid] = a
        print(f"  {sid}: {a.shape}")
    adata = sc.concat(adatas, axis=0, join="outer", index_unique="-")
    adata.layers["raw_counts"] = adata.X.copy()
else:
    raise RuntimeError(f"no recognizable deposit format in {RAW_DIR}")

# Attach patient metadata
if "sample_id" not in adata.obs.columns:
    raise RuntimeError(
        "Need adata.obs['sample_id'] aligning with meta/patients.tsv sample_id column. "
        f"Current obs columns: {list(adata.obs.columns)}"
    )
adata.obs = adata.obs.merge(patients, on="sample_id", how="left")
adata.obs["donor"] = adata.obs["sample_id"].astype(str)
if "raw_counts" not in adata.layers:
    adata.layers["raw_counts"] = adata.X.copy()
adata.write_h5ad(CONCAT_H5)
print(f"\nwrote {CONCAT_H5}  | shape {adata.shape}")
print(adata.obs["disease_stage"].value_counts())
```

## Step 3 — Locate author annotations (T-cell labels, malignancy if any)

The preprint doesn't separately annotate malignant T cells — it focuses on
fibroblasts. But the integrated deposit *should* carry the authors' Seurat
cluster labels for the 29 clusters described in the Results (T cells, fibroblasts,
keratinocytes, etc.). Find and unify them here.

```python  # cell 8
print("obs columns in concatenated AnnData:")
for c in sorted(adata.obs.columns):
    n_unique = adata.obs[c].nunique() if adata.obs[c].dtype != float else "(float)"
    sample_vals = list(adata.obs[c].dropna().astype(str).unique()[:5])
    print(f"  {c:30s}  n_unique={n_unique}  sample={sample_vals}")

# CANDIDATES to look for and unify under "cell_type_author":
CANDIDATE_TYPE_COLS = [
    "cell_type", "celltype", "CellType", "Cell_Type",
    "seurat_clusters", "cluster", "Cluster", "annotation",
    "cell_id", "manual_annotation",
]
type_col = next((c for c in CANDIDATE_TYPE_COLS if c in adata.obs.columns), None)
if type_col is None:
    print("\n⚠️  no obvious cell-type column — will need to annotate from scratch via markers.")
else:
    print(f"\nusing '{type_col}' as author cell-type label.")
    adata.obs["cell_type_author"] = adata.obs[type_col].astype(str)
    print(adata.obs["cell_type_author"].value_counts())
```

## Step 4 — Filter to advanced MF skin + HC; basic QC

```python  # cell 10
keep_mask = adata.obs["is_advanced_mf_skin"] | (adata.obs["disease"] == "HC")
adata = adata[keep_mask].copy()
print("after sample filter:", adata.shape)
print(adata.obs["sample_id"].value_counts())

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
keep = (
    (adata.obs["pct_counts_mt"] < 15)
    & (adata.obs["n_genes_by_counts"] > 200)
    & (adata.obs["n_genes_by_counts"] < 6000)
)
adata = adata[keep].copy()
print("post-QC:", adata.shape)
adata.layers["raw_counts"] = adata.X.copy()
```

## Step 5 — Identify CD4⁺ T cells by marker expression

```python  # cell 12
adata.layers["counts"] = adata.layers["raw_counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

def _expr(a, gene):
    if gene not in a.var_names:
        return np.zeros(a.n_obs, dtype=np.float32)
    v = a[:, gene].X
    return (v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()).astype(np.float32)

adata.obs["score_CD3"]   = np.maximum.reduce([_expr(adata, g) for g in ["CD3D", "CD3E", "CD3G"]])
adata.obs["score_CD4"]   = _expr(adata, "CD4")
adata.obs["score_CD8"]   = np.maximum(_expr(adata, "CD8A"), _expr(adata, "CD8B"))
adata.obs["score_FOXP3"] = _expr(adata, "FOXP3")

is_tcell = adata.obs["score_CD3"] > 0.5
is_cd4   = (adata.obs["score_CD4"] > 0) & (adata.obs["score_CD4"] >= adata.obs["score_CD8"])
not_treg = adata.obs["score_FOXP3"] < 0.5
adata.obs["is_cd4_tcell"] = (is_tcell & is_cd4 & not_treg).astype(bool)

print("T cell count:", int(is_tcell.sum()))
print("CD4 helper T cells:", int(adata.obs["is_cd4_tcell"].sum()))
print("by sample:")
print(adata.obs.groupby("sample_id")["is_cd4_tcell"].sum().to_string())

adata_cd4 = adata[adata.obs["is_cd4_tcell"]].copy()
adata_cd4.X = adata_cd4.layers["raw_counts"].copy()
print("\nCD4 subset:", adata_cd4.shape)
del adata; gc.collect()
```

## Step 6 — Cluster CD4 T cells on standard scVI latent (manual malignant-pick)

D2 has no TCR. We cluster CD4 T cells on a standard scVI latent and use the
ref's **manual cluster-pick** approach (06 cells 4–8). The dot plot + per-cluster
composition (MF vs HC fraction) below guides the pick.

```python  # cell 14
scvi.model.SCVI.setup_anndata(adata_cd4, batch_key="donor", layer="raw_counts")
scvi_clust = scvi.model.SCVI(adata_cd4, n_layers=2, n_latent=10, gene_likelihood="nb")
scvi_clust.train(max_epochs=50, early_stopping=True, accelerator="auto")
adata_cd4.obsm["X_scVI"] = scvi_clust.get_latent_representation()

sc.pp.neighbors(adata_cd4, use_rep="X_scVI", random_state=0)
sc.tl.umap(adata_cd4, random_state=0)
sc.tl.leiden(adata_cd4, resolution=TUMOR_LEIDEN_RESOLUTION, random_state=0, key_added="cd4_leiden")
print("leiden cluster sizes:", adata_cd4.obs["cd4_leiden"].value_counts().to_dict())

sc.pl.umap(
    adata_cd4,
    color=["cd4_leiden", "disease", "disease_stage", "sample_id"],
    ncols=2, wspace=0.3, frameon=False, save=None,
)
```

### Dot plot CD4 / CD8 / FOXP3 + per-cluster MF/HC composition

```python  # cell 16
import matplotlib.pyplot as plt

markers = [g for g in ["CD4", "CD8A", "CD8B", "FOXP3", "CD3D"] if g in adata_cd4.var_names]
dp = sc.pl.dotplot(
    adata_cd4, var_names=markers, groupby="cd4_leiden",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "d2_cd4_tumor_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "d2_cd4_tumor_dotplot.png")

cluster_stats = (
    adata_cd4.obs.groupby("cd4_leiden")
    .agg(
        n_cells=("disease", "size"),
        frac_HC=("disease", lambda s: (s == "HC").mean()),
        frac_MF=("disease", lambda s: (s == "MF").mean()),
        n_samples=("sample_id", "nunique"),
    )
    .sort_values("frac_MF", ascending=False)
)
print("\nper-cluster MF/HC composition:")
print(cluster_stats.round(3).to_string())
```

## 🛑 Step 7 — Hand-pick malignant CD4 clusters

Set `CD4_MAL_CLUSTERS` based on the dot plot + composition table above.
Pick clusters that are (a) CD4+ / CD8- / FOXP3- by markers and (b) dominated
by MF samples (high `frac_MF`, low `frac_HC`). Clusters dominated by HC →
benign; mixed clusters → benign by default.

```python  # cell 18
# EDIT THIS LIST after inspecting the dot plot above.
CD4_MAL_CLUSTERS: list[str] = []   # e.g. ["0", "1", "3"]

assert CD4_MAL_CLUSTERS, "Inspect the dot plot, then fill CD4_MAL_CLUSTERS"

adata_cd4.obs["is_malignant"] = adata_cd4.obs["cd4_leiden"].isin(CD4_MAL_CLUSTERS).astype(bool)
adata_cd4.obs["status"]    = np.where(adata_cd4.obs["is_malignant"], "tumor_cd4", "benign_cd4")
adata_cd4.obs["cell_type"] = adata_cd4.obs["status"]

print(f"selected clusters: {CD4_MAL_CLUSTERS}")
print("status counts:", adata_cd4.obs["status"].value_counts().to_dict())
print("\nper-sample status:")
print(adata_cd4.obs.groupby("sample_id")["status"].value_counts().unstack(fill_value=0).to_string())
adata_cd4.write_h5ad(CD4_H5)
print("wrote", CD4_H5)
```

## Step 8 — Attach Ensembl gene_id + build Geneformer semantic map, HVG-subset

```python  # cell 20
# If the deposit didn't include ENSG IDs, attach them from any 10x features.tsv
# in the raw dir, or fall back to symbol-only matching (Geneformer accepts both
# but Ensembl is preferred).
if "gene_id" not in adata_cd4.var.columns:
    features_files = list(RAW_DIR.rglob("features.tsv*")) + list(RAW_DIR.rglob("*features.tsv*"))
    if features_files:
        ft = pd.read_csv(
            features_files[0], header=None, sep="\t",
            names=["gene_id", "gene_name", "feature_type"],
        )
        sym2ens = dict(zip(ft["gene_name"].astype(str), ft["gene_id"].astype(str)))
        adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
    else:
        # Fallback: use symbol as the key (Geneformer will treat as missing for non-ENSG)
        adata_cd4.var["gene_id"] = adata_cd4.var_names

n_mapped = int(sum(str(g).startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")

semantic_map = get_or_build_geneformer_map(adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id")
print("semantic_map (raw):", tuple(semantic_map.shape))

if HVG_TOP_N is not None:
    sc.pp.highly_variable_genes(adata_cd4, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata_cd4.var["highly_variable"].values
    adata_cd4 = adata_cd4[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
print("after HVG:", adata_cd4.shape, "| semantic_map:", tuple(semantic_map.shape))
```

## Step 9 — Train SemanticSCVI (geometric, Geneformer prior) — GPU

```python  # cell 22
model = train_or_load_semantic_scvi(
    adata_cd4,
    semantic_map,
    cache_dir=SEMANTIC_GENEFORMER_CACHE_DIR,
    force_train=FORCE_TRAIN_SEMANTIC_GENEFORMER,
    max_epochs=SEMANTIC_GENEFORMER_MAX_EPOCHS,
    warmup_epochs=SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    n_epochs_kl_warmup=SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP,
    batch_key=BATCH_KEY,
    **SEMANTIC_GENEFORMER_KWARGS,
)
print("trained:", model.is_trained)
```

## Step 10 — Projection analysis (mirrors 07 / 06)

```python  # cell 24
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())
sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
sc.tl.umap(ad_emb, random_state=0)

color = [c for c in ["status", "disease", "sample_id", "disease_stage"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```python  # cell 25
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import fisher_exact

is_mal = (ad_emb.obs["status"].astype(str).values == "tumor_cd4").astype(int)
n_factors = z.shape[1]
MAX_PER_CLASS = 15000
rng = np.random.default_rng(0)
idx_mal = np.where(is_mal == 1)[0]; idx_ben = np.where(is_mal == 0)[0]
if len(idx_mal) > MAX_PER_CLASS: idx_mal = rng.choice(idx_mal, MAX_PER_CLASS, replace=False)
if len(idx_ben) > MAX_PER_CLASS: idx_ben = rng.choice(idx_ben, MAX_PER_CLASS, replace=False)
plot_idx = np.concatenate([idx_mal, idx_ben])

aucs = {}
fig, axes = plt.subplots((n_factors + 1) // 2, 2, figsize=(11, 2.2 * ((n_factors + 1) // 2)), sharex=False)
axes = axes.flatten()
for k in range(n_factors):
    ax = axes[k]
    auc = roc_auc_score(is_mal, z[:, k]); aucs[f"Z_{k}"] = max(auc, 1 - auc)
    sub_vals = z[plot_idx, k]; sub_mal = is_mal[plot_idx]
    order = np.argsort(sub_vals); vals = sub_vals[order]; mal = sub_mal[order]
    rank = np.arange(len(vals))
    ax.scatter(rank[mal == 1], vals[mal == 1], s=2, c="tab:red", alpha=0.35, rasterized=True, linewidths=0)
    ax.scatter(rank[mal == 0], vals[mal == 0], s=2, c="tab:blue", alpha=0.35, rasterized=True, linewidths=0)
    ax.set_title(f"Z_{k}  |  AUROC = {aucs[f'Z_{k}']:.3f}", fontsize=10)
    ax.set_xlabel(f"cell rank (n={MAX_PER_CLASS}/class)", fontsize=8); ax.set_ylabel(f"Z_{k}", fontsize=8)
    ax.tick_params(labelsize=7)
for j in range(n_factors, len(axes)): axes[j].axis("off")
fig.tight_layout()
out = FIG_DIR / "d2_semantic_geom_cd4_factor_separation.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out); plt.show()

print("\nAUROC (symmetric) per factor:")
print(pd.Series(aucs).sort_values(ascending=False).to_string())

# Top-3 most-separating factors → 3-D scatter
top_factors = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:3]
FX3, FY3, FZ3 = [int(k.split("_")[1]) for k, _ in top_factors]
print(f"\ntop-3 factors → 3-D scatter: Z_{FX3}, Z_{FY3}, Z_{FZ3}")
```

```python  # cell 26
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

x3 = z[:, FX3]; y3 = z[:, FY3]; zZ3 = z[:, FZ3]
mu_x3, sd_x3 = x3.mean(), x3.std()
mu_y3, sd_y3 = y3.mean(), y3.std()
mu_z3, sd_z3 = zZ3.mean(), zZ3.std()
xz3 = (x3 - mu_x3) / sd_x3; yz3 = (y3 - mu_y3) / sd_y3; zz3 = (zZ3 - mu_z3) / sd_z3
s3 = xz3 + yz3 + zz3
y_true = is_mal
median_s3 = float(np.median(s3))
fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
auc_s3 = max(roc_auc_score(y_true, s3), 1 - roc_auc_score(y_true, s3))

rng3 = np.random.default_rng(0); shuf3 = rng3.permutation(len(plot_idx))
xx3 = x3[plot_idx][shuf3]; yy3 = y3[plot_idx][shuf3]; zz3p = zZ3[plot_idx][shuf3]
ss3 = s3[plot_idx][shuf3]
cc3 = np.where(is_mal[plot_idx][shuf3] == 1, "tab:red", "tab:blue")
s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

def s_plane(c, xg, yg):
    return sd_z3 * (c - (xg - mu_x3) / sd_x3 - (yg - mu_y3) / sd_y3) + mu_z3

xg = np.linspace(xx3.min(), xx3.max(), 20)
yg = np.linspace(yy3.min(), yy3.max(), 20)
XG, YG = np.meshgrid(xg, yg)
ZG_med  = s_plane(median_s3, XG, YG); ZG_best = s_plane(best_thr3, XG, YG)

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
fig.suptitle(f"D2 3-D scatter Z_{FX3}, Z_{FY3}, Z_{FZ3} with diagonal score s3", fontsize=12, y=1.02)
fig.tight_layout()
out = FIG_DIR / f"d2_semantic_geom_cd4_scatter3d_Z{FX3}_Z{FY3}_Z{FZ3}_with_s3.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out); plt.show()
```

## Step 11 — Gene loadings + hierarchical clustering

```python  # cell 28
W = model.get_loadings().reindex(adata_cd4.var_names)
adata_cd4.varm["W_d2_semantic_geom_cd4"] = W.values
print("W (loadings):", W.shape)

top_per_factor = {
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist() for col in W.columns
}
top_df = pd.DataFrame(top_per_factor)
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
top_df.to_csv(NB_DIR / "tables" / "d2_semantic_geom_cd4_top_genes_per_factor.tsv", sep="\t")
print("\ntop genes per factor (head):")
with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 30):
    print(top_df.head(15))
```

```python  # cell 29
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns

max_vals = W.abs().max(axis=1)
top_idx = max_vals.sort_values(ascending=False).head(CLUSTER_N_TOP).index
subset = W.loc[top_idx]; subset = subset.loc[subset.values.std(axis=1) > 0]
top_idx = subset.index
dists = pdist(subset.values, metric="correlation"); dist_matrix = squareform(dists)
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
g.fig.suptitle(f"D2 semantic_geom CD4: top-{len(top_idx)} gene loading correlation (k={best_k})", y=1.01)
out = FIG_DIR / "d2_semantic_geom_cd4_gene_hierarchy.png"
g.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
```
