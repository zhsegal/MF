# old/12_semantic_cd4_malignancy_calls.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/12_semantic_cd4_malignancy_calls.ipynb` and is never run._

# SemanticSCVI (geometric) CD4 factors vs. **cached malignancy calls**

Reloads the CD4 `SemanticSCVI` trained in `06_semantic_geom_cd4.ipynb` (same params →
same cache hit, **no retrain**) and re-runs the latent-UMAP + per-factor separation
heatmap of 06, but driven by the richer per-cell malignancy calls cached by
`11_li2024_tcr_malignancy.ipynb` (Step 8b/c) instead of the single `status` flag.

Calls (from `data/Li2024_atlas/li2024_malignancy_calls.parquet`, keyed by `obs_names`):
`cached_malignant`, `tcr_is_malignant`, `cnv_malignant`, `cnv_focal_score`,
`leiden_cnv_malignant` (scVI latent), `leiden_cnv_malignant_mrvi` (MRVI latent).

Run top-to-bottom on the `neural_nmf_env` GPU kernel.

```python  # cell 1
# ============================================================
# Parameters — same knobs/names as 05_semantic_geom_malignant.ipynb.
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
    """Stable 10-char hash of every param that affects the trained SemanticSCVI model.
    Change any of these and the cache_dir flips -> fresh train. Same params -> cache hit."""
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

# ---- paths ----
ATLAS_PATH     = NB_DIR / "data" / "cache" / "mrvi_ctcl_cache.h5ad"           # complete atlas (HVGs + X_mrvi)
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"     # gene_name -> gene_id only
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "mf_cd4_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_semantic_geom_cd4"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)

# ---- CD4 selection ----
BENIGN_CELL_TYPE = "Th"        # benign CD4 reference (= cell_type "Th")
TUMOR_LEIDEN_RESOLUTION = 0.5  # leiden resolution on MRVI latent within tumor cells

# ---- Preprocessing (ref values from 05) ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = None              # use ALL hand-picked tumor CD4 + Th

# ---- Shared model size (ref) ----
N_LATENT = 10

# ---- Batch effect ----
BATCH_KEY = "donor"

# ---- SemanticSCVI (Geneformer prior) — same as 05 ----
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

CALLS_PARQUET = NB_DIR / "data" / "Li2024_atlas" / "li2024_malignancy_calls.parquet"
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
semantic_geom cache_dir: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4/semantic_scvi/91318ccb2b
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))   # notebooks/ holds benchmark_helpers.py

import gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scvi

from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
print("scvi", scvi.__version__, "| scanpy", sc.__version__)
print("CUDA available:", torch.cuda.is_available())

sc.settings.figdir = FIG_DIR
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Seed set to 0

scvi 1.3.3 | scanpy 1.11.5
CUDA available: True

/tmp/ipykernel_1643393/2462664868.py:15: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scvi", scvi.__version__, "| scanpy", sc.__version__)
```

## Section A — rebuild `adata_cd4` and reload the trained model

Reconstructs the exact CD4 cell set from 06 (tumor-CD4 leiden clusters + benign Th),
then loads `SemanticSCVI` from cache. Seeds are fixed so the leiden labels — and thus
`CD4_TUMOR_CLUSTERS` — reproduce the trained model's cell set. If `train_or_load_…`
prints a training bar instead of a "loaded …" line, the cache slug drifted (diff the
param cell against 06).

```python  # cell 4
adata = sc.read_h5ad(ATLAS_PATH)
print("atlas:", adata.shape)

# 06 ref clustered tumor_cell ONLY (its cell 4) — NOT tumor+Th. The hand-picked
# CD4_TUMOR_CLUSTERS refer to this labeling. UMAP/plot dropped (not needed for cluster IDs).
ad_tumor = adata[adata.obs["cell_type"].astype(str) == "tumor_cell"].copy()
print("tumor:", ad_tumor.shape)
sc.pp.neighbors(ad_tumor, use_rep="X_mrvi", random_state=0)
sc.tl.leiden(ad_tumor, resolution=TUMOR_LEIDEN_RESOLUTION, random_state=0, key_added="tumor_leiden")
print("leiden clusters:", ad_tumor.obs["tumor_leiden"].value_counts().to_dict())
```

```
atlas: (419579, 10000)
tumor: (102367, 10000)

/tmp/ipykernel_1643393/190162227.py:9: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  sc.tl.leiden(ad_tumor, resolution=TUMOR_LEIDEN_RESOLUTION, random_state=0, key_added="tumor_leiden")

leiden clusters: {'0': 30462, '1': 19122, '2': 18984, '3': 13273, '4': 7547, '5': 6788, '6': 3661, '7': 2126, '8': 404}
```

```python  # cell 5
# EDIT THIS LIST after inspecting the dot plot above.
CD4_TUMOR_CLUSTERS: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"] 

assert CD4_TUMOR_CLUSTERS, "Inspect the dot plot, then fill CD4_TUMOR_CLUSTERS"

ad_tumor_cd4 = ad_tumor[ad_tumor.obs["tumor_leiden"].isin(CD4_TUMOR_CLUSTERS)].copy()
# Reproducibility guard: 06's reference selected exactly 101963 tumor-CD4 cells. A mismatch
# means leiden numbering drifted -> CD4_TUMOR_CLUSTERS invalid -> HVG/var_names won't match
# the trained model. Fail loud here, before the expensive model load.
assert ad_tumor_cd4.n_obs == 101963, (
    f"expected 101963 tumor-CD4 cells (06 reference); got {ad_tumor_cd4.n_obs} — "
    "leiden numbering drifted, CD4_TUMOR_CLUSTERS no longer valid")
print(f"selected {ad_tumor_cd4.n_obs} tumor CD4 cells "
      f"from {ad_tumor.n_obs} total tumor (clusters {CD4_TUMOR_CLUSTERS})")
print("by donor:")
print(ad_tumor_cd4.obs["donor"].value_counts().head(10))
```

```
selected 101963 tumor CD4 cells from 102367 total tumor (clusters ['0', '1', '2', '3', '4', '5', '6', '7'])
by donor:
donor
MF28      11072
CTCL3      9524
CTCL18     8223
CTCL1      8068
CTCL5      7722
CTCL8      7134
CTCL13     6100
MF14       4183
PT35       4091
CTCL6      3773
Name: count, dtype: int64
```

```python  # cell 6
ad_benign = adata[adata.obs["cell_type"].astype(str) == BENIGN_CELL_TYPE].copy()
print("benign Th:", ad_benign.shape)

adata_cd4 = sc.concat(
    [ad_tumor_cd4, ad_benign],
    axis=0, join="inner", label="status", keys=["tumor_cd4", "benign_cd4"],
    index_unique=None,
)
adata_cd4.X = adata_cd4.layers["raw_counts"].copy()  # NB likelihood needs raw counts
print("combined:", adata_cd4.shape,
      "| status:", adata_cd4.obs["status"].value_counts().to_dict())

# free memory before training
del adata, ad_tumor, ad_benign
gc.collect()
```

```
benign Th: (52824, 10000)
combined: (154787, 10000) | status: {'tumor_cd4': 101963, 'benign_cd4': 52824}

51915
```

```python  # cell 7
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
n_mapped = int(sum(g.startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")
```

```
gene_id mapped to Ensembl: 9766/10000
```

```python  # cell 8
# Build the FULL-gene Geneformer map first (cached). Then HVG-subset adata + map together.
# Delete mf_cd4_geneformer.pt to force a rebuild.
semantic_map = get_or_build_geneformer_map(
    adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
print("semantic_map (raw):", tuple(semantic_map.shape))

# Stale-cache guard: if the cached map was built on a previously HVG-subset adata,
# its rows won't match the current full-gene adata. Rebuild in that case.
if semantic_map.shape[0] != adata_cd4.n_vars:
    print(f"shape mismatch ({semantic_map.shape[0]} vs {adata_cd4.n_vars}) — rebuilding")
    SEMANTIC_CACHE_GENEFORMER.unlink()
    semantic_map = get_or_build_geneformer_map(
        adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
    )
    print("semantic_map (rebuilt):", tuple(semantic_map.shape))

if HVG_TOP_N is not None:
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
Loading cached Geneformer map from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_cd4_geneformer.pt
semantic_map (raw): (10000, 1152)
after HVG: (154787, 2500) | semantic_map: (2500, 1152)
no subsampling (n_obs=154787, SUBSAMPLE_N=None)
```

```python  # cell 9
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

```
INFO     File                                                                                                      
         /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4/seman
         tic_scvi/91318ccb2b/model.pt already downloaded                                                           
[15:49:34]   loaded SemanticSCVI from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4/semantic_scvi/91318ccb2b
trained: True
```

## Section A.2 — latent embedding + loadings

```python  # cell 11
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())

# Cache the latent UMAP (neighbors + umap are the slow part) -> .npy aligned to ad_emb
# row order (deterministic from the adata_cd4 rebuild). Re-runs reload. FORCE_UMAP to rebuild.
# Recompute on shape mismatch so a stale npy (wrong cell count) self-heals.
EMB_UMAP_NPY = NB_DIR / "data" / "Li2024_atlas" / "semantic_geom_cd4_emb_umap.npy"
FORCE_UMAP = False
if (not FORCE_UMAP) and EMB_UMAP_NPY.exists() and \
        np.load(EMB_UMAP_NPY).shape[0] == ad_emb.n_obs:
    ad_emb.obsm["X_umap"] = np.load(EMB_UMAP_NPY)
    print("loaded cached emb UMAP", ad_emb.obsm["X_umap"].shape)
else:
    sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
    sc.tl.umap(ad_emb, random_state=0)
    np.save(EMB_UMAP_NPY, ad_emb.obsm["X_umap"])
    print("computed + cached emb UMAP ->", EMB_UMAP_NPY)

color = [c for c in ["status", "stage", "study", "tissue", "donor"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```
computed + cached emb UMAP -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/Li2024_atlas/semantic_geom_cd4_emb_umap.npy

<Figure size 1664x1440 with 5 Axes>
```

```python  # cell 12
W = model.get_loadings().reindex(adata_cd4.var_names)   # genes x N_LATENT
n_factors = z.shape[1]
print("W (loadings):", W.shape, "| n_factors:", n_factors, "| columns:", list(W.columns))
W.head()
```

```
W (loadings): (2500, 10) | n_factors: 10 | columns: ['Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9']

               Z_0       Z_1       Z_2       Z_3       Z_4       Z_5  \
HES4      0.012427  0.006022  0.137283  0.004578  0.025926  0.005638   
ISG15     0.315019  0.033936  0.002390  0.103474  0.005595  0.109133   
RNF223    0.023841  0.025443  0.575253  0.021771  0.030152  1.019468   
TNFRSF18  0.056001  2.608525  0.005918  0.009065  0.386911  0.004751   
TNFRSF4   0.061097  0.404810  0.015318  1.610499  0.680414  0.001340   

               Z_6       Z_7       Z_8       Z_9  
HES4      3.258906  0.012049  0.071941  0.006463  
ISG15     1.420385  0.002057  0.006241  3.750064  
RNF223    0.037012  0.034518  4.401197  0.017732  
TNFRSF18  1.666576  0.028022  0.222135  0.080810  
TNFRSF4   0.077079  0.004370  0.421423  3.553838
```

## Section B — join the cached malignancy calls onto the model's cells

```python  # cell 14
calls = pd.read_parquet(CALLS_PARQUET)
print("calls:", calls.shape)

CALL_COLS = ["cached_malignant", "tcr_is_malignant", "cnv_malignant",
             "cnv_focal_score", "leiden_cnv_malignant", "leiden_cnv_malignant_mrvi"]
for c in CALL_COLS:
    ad_emb.obs[c] = calls[c].reindex(ad_emb.obs_names).values

# clean categorical labels for the boolean calls (continuous cnv_focal_score stays numeric)
BIN_CALLS = [c for c in CALL_COLS if c != "cnv_focal_score"]
for c in BIN_CALLS:
    ad_emb.obs[c + "_lbl"] = pd.Categorical(
        np.where(ad_emb.obs[c].astype(bool), "malignant", "benign"),
        categories=["benign", "malignant"],
    )

# NOTE: bool calls are False-filled (no NaN); cnv_focal_score is NaN for cells outside the
# inferCNV lymphoid query (rendered grey on the UMAP). A `*_malignant == False` therefore
# means "benign OR not CNV-evaluated", not strictly benign.
print("malignant counts:",
      {c: int(ad_emb.obs[c].astype(bool).sum()) for c in BIN_CALLS})
print("cnv_focal_score non-NaN:",
      int(ad_emb.obs["cnv_focal_score"].notna().sum()), "/", ad_emb.n_obs)
```

```
calls: (419579, 10)
malignant counts: {'cached_malignant': 101963, 'tcr_is_malignant': 40372, 'cnv_malignant': 67460, 'leiden_cnv_malignant': 49434, 'leiden_cnv_malignant_mrvi': 58325}
cnv_focal_score non-NaN: 101835 / 154787
```

## Section C — latent UMAP colored by each malignancy call

```python  # cell 16
color = (
    [c + "_lbl" for c in BIN_CALLS]
    + ["cnv_focal_score"]
    + [c for c in ["status", "cell_type"] if c in ad_emb.obs]
)
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False,
           save="_semantic_geom_cd4_malignancy_calls.png")
```

```
WARNING: saving figure to file /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/umap_semantic_geom_cd4_malignancy_calls.png

<Figure size 1664x1920 with 9 Axes>
```

## Section D — per-factor separation heatmaps (one per call)

Replicates 06's separation heatmap for each binary call: for factor `Z_k`, cells are
sorted by `Z_k` (low → high) and binned along x; bin color = fraction of malignant cells
(red = malignant, blue = benign). A clean red↔blue gradient = a factor that separates that
call. Class-balanced per call (`MAX_PER_CLASS` per class). Row labels carry the per-factor
AUROC. `cnv_focal_score` (continuous) is covered by the UMAP above, not this loop.

```python  # cell 18
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

N_BINS = 400
MAX_PER_CLASS = 20000   # class-balance cap per call


def call_aucs(zmat, y):
    """Per-factor AUROC of Z_k separating y, polarity-folded to >= 0.5."""
    return {f"Z_{k}": max(a, 1 - a)
            for k in range(zmat.shape[1])
            for a in [roc_auc_score(y, zmat[:, k])]}


def balanced_idx(y, cap, seed=0):
    """Indices of a class-balanced subsample: min(cap, n_pos, n_neg) per class."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = min(cap, len(pos), len(neg))
    return np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])
```

```python  # cell 19
for col in BIN_CALLS:
    y = ad_emb.obs[col].astype(bool).to_numpy().astype(int)
    if y.sum() in (0, len(y)):
        print(f"skip {col}: only one class present"); continue
    aucs = call_aucs(z, y)
    idx = balanced_idx(y, MAX_PER_CLASS)
    sub_z, sub_y = z[idx], y[idx]

    heat = np.zeros((n_factors, N_BINS), dtype=float)
    for k in range(n_factors):
        order = np.argsort(sub_z[:, k])
        y_sorted = sub_y[order].astype(float)
        heat[k] = np.array([chunk.mean() for chunk in np.array_split(y_sorted, N_BINS)])

    fig, ax = plt.subplots(figsize=(11, 0.45 * n_factors + 1.5))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels([f"Z_{k}  (AUROC={aucs[f'Z_{k}']:.3f})" for k in range(n_factors)],
                       fontsize=9)
    ax.set_xticks([0, N_BINS // 2, N_BINS - 1])
    ax.set_xticklabels(["low Z_k", "mid", "high Z_k"], fontsize=9)
    ax.set_xlabel(f"cells sorted by Z_k (binned into {N_BINS} quantiles)", fontsize=9)
    ax.set_title(f"Per-factor separation — {col}  "
                 f"(class-balanced, {int(sub_y.sum())} per class)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"fraction {col} in bin")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["all benign", "0.5", "all malignant"])

    fig.tight_layout()
    out = FIG_DIR / f"semantic_geom_cd4_sep_heatmap_{col}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_sep_heatmap_cached_malignant.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_sep_heatmap_tcr_is_malignant.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_sep_heatmap_cnv_malignant.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_sep_heatmap_leiden_cnv_malignant.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_sep_heatmap_leiden_cnv_malignant_mrvi.png

<Figure size 1100x600 with 2 Axes>
```

## Section E — 2-D / 3-D factor geometry vs. `leiden_cnv_malignant_mrvi`

Ports 06's 2-factor scatter grid, 3-D `(Z_1, Z_7, Z_6)` scatter, and the best-AUROC
gene-direction plane sweep — but with **`leiden_cnv_malignant_mrvi`** as the label instead of
`status`. Factor indices below (`FACTORS_GRID`, `FX3/FY3/FZ3`) are 06's status picks; re-pick
from the per-factor `aucs` printed in the setup cell if other factors separate this call
better. Figures are written with a `leiden_cnv_malignant_mrvi` suffix so they don't clobber
06's `status` outputs.

```python  # cell 21
# --- target annotation + derived vars for the 2-D / 3-D factor geometry ---
CALL = "leiden_cnv_malignant_mrvi"
POS_LABEL, NEG_LABEL = "mrvi_malignant", "benign"

is_mal = ad_emb.obs[CALL].astype(bool).to_numpy().astype(int)
y_true = is_mal
plot_idx = balanced_idx(is_mal, MAX_PER_CLASS)            # helpers from Section D
aucs = call_aucs(z, is_mal)
print(f"{CALL}: {int(is_mal.sum())} malignant / {len(is_mal)} cells | "
      f"balanced subsample {len(plot_idx)} ({len(plot_idx) // 2}/class)")
print("per-factor AUROC:", {k: round(v, 3) for k, v in aucs.items()})

# top loading genes per factor (consumed by the gene-direction sweep below)
top_df = pd.DataFrame({
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist()
    for col in W.columns
})
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
```

```
leiden_cnv_malignant_mrvi: 58325 malignant / 154787 cells | balanced subsample 40000 (20000/class)
per-factor AUROC: {'Z_0': 0.663, 'Z_1': 0.785, 'Z_2': 0.571, 'Z_3': 0.605, 'Z_4': 0.508, 'Z_5': 0.534, 'Z_6': 0.687, 'Z_7': 0.582, 'Z_8': 0.514, 'Z_9': 0.683}
```

```python  # cell 22
from itertools import combinations

FACTORS_GRID = [1, 6, 7, 3, 9]   # 06's status picks; re-pick from `aucs` if other factors win
pairs = list(combinations(FACTORS_GRID, 2))   # 10 pairs

ncols = 5
nrows = (len(pairs) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
axes = axes.flatten()

rng_draw = np.random.default_rng(0)
shuf = rng_draw.permutation(len(plot_idx))
sub_mal = is_mal[plot_idx][shuf]
colors = np.where(sub_mal == 1, "tab:red", "tab:blue")

for ax, (fx, fy) in zip(axes, pairs):
    sub_x = z[plot_idx, fx][shuf]
    sub_y = z[plot_idx, fy][shuf]
    ax.scatter(sub_x, sub_y, c=colors, s=2, alpha=0.3,
               linewidths=0, rasterized=True)
    ax.set_xlabel(f"Z_{fx}", fontsize=9)
    ax.set_ylabel(f"Z_{fy}", fontsize=9)
    ax.set_title(f"Z_{fx} vs Z_{fy}  (AUROC {aucs[f'Z_{fx}']:.2f}/{aucs[f'Z_{fy}']:.2f})",
                 fontsize=9)
    ax.tick_params(labelsize=7)

for j in range(len(pairs), len(axes)):
    axes[j].axis("off")

handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label=POS_LABEL, markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label=NEG_LABEL, markersize=6),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.suptitle(f"2-factor scatter grid — colored by {CALL}", y=1.01, fontsize=11)
fig.tight_layout()

out = FIG_DIR / f"semantic_geom_cd4_{CALL}_scatter_grid_{'_'.join(map(str, FACTORS_GRID))}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_leiden_cnv_malignant_mrvi_scatter_grid_1_6_7_3_9.png

<Figure size 1500x600 with 10 Axes>
```

### 3-D scatter — factors (Z_1, Z_7, Z_6)

Diagonal score `s3 = z(Z_1) + z(Z_7) + z(Z_6)`; left panel colored by the call, right by `s3`,
with median / Youden-best planes drawn at constant `s3`.

```python  # cell 24
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.metrics import roc_curve, roc_auc_score

FX3, FY3, FZ3 = 1, 7, 6

x3 = z[:, FX3]; y3 = z[:, FY3]; zZ3 = z[:, FZ3]
mu_x3, sd_x3 = x3.mean(), x3.std()
mu_y3, sd_y3 = y3.mean(), y3.std()
mu_z3, sd_z3 = zZ3.mean(), zZ3.std()

xz3 = (x3 - mu_x3) / sd_x3
yz3 = (y3 - mu_y3) / sd_y3
zz3 = (zZ3 - mu_z3) / sd_z3
s3 = xz3 + yz3 + zz3

# class-balanced + shuffled draw (reuse plot_idx)
rng3 = np.random.default_rng(0)
shuf3 = rng3.permutation(len(plot_idx))
xx3  = x3[plot_idx][shuf3]
yy3  = y3[plot_idx][shuf3]
zz3p = zZ3[plot_idx][shuf3]
ss3  = s3[plot_idx][shuf3]
cc3  = np.where(is_mal[plot_idx][shuf3] == 1, "tab:red", "tab:blue")

# thresholds on the diagonal score
median_s3 = float(np.median(s3))
fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
auc_s3 = float(roc_auc_score(y_true, s3)); auc_s3 = max(auc_s3, 1 - auc_s3)


def s_plane(c, xg, yg):
    return sd_z3 * (c - (xg - mu_x3) / sd_x3 - (yg - mu_y3) / sd_y3) + mu_z3


xg = np.linspace(xx3.min(), xx3.max(), 20)
yg = np.linspace(yy3.min(), yy3.max(), 20)
XG, YG = np.meshgrid(xg, yg)
ZG_med  = s_plane(median_s3, XG, YG)
ZG_best = s_plane(best_thr3, XG, YG)
s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

fig = plt.figure(figsize=(14, 7))

# --- Left: colored by the call ---
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axL.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by {CALL}   AUROC(s3) = {auc_s3:.3f}", fontsize=10)
status_handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",  label=POS_LABEL, markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label=NEG_LABEL, markersize=6),
    plt.Line2D([], [], marker="s", linestyle="", color="black", alpha=0.10, label=f"plane: s3 = median ({median_s3:+.2f})", markersize=10),
    plt.Line2D([], [], marker="s", linestyle="", color="black", alpha=0.20, label=f"plane: s3 = best   ({best_thr3:+.2f}, Youden)", markersize=10),
]
axL.legend(handles=status_handles, loc="upper left", fontsize=8, frameon=False)

# --- Right: colored by s3 value ---
axR = fig.add_subplot(1, 2, 2, projection="3d")
sc3 = axR.scatter(xx3, yy3, zz3p, c=ss3, cmap="RdBu_r", vmin=-s_vmax3, vmax=s_vmax3,
                  s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axR.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3})", fontsize=10)
cbar = fig.colorbar(sc3, ax=axR, fraction=0.03, pad=0.10); cbar.set_label("s3 value")

fig.suptitle(f"3-D scatter Z_{FX3}, Z_{FY3}, Z_{FZ3} — {CALL}", fontsize=12, y=1.02)
fig.tight_layout()
out = FIG_DIR / f"semantic_geom_cd4_{CALL}_scatter3d_Z{FX3}_Z{FY3}_Z{FZ3}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_leiden_cnv_malignant_mrvi_scatter3d_Z1_Z7_Z6.png

<Figure size 1400x700 with 3 Axes>
```

```python  # cell 25
# 1) Pseudotime via DPT on the latent kNN graph. ad_emb's neighbors may have been skipped
#    when the UMAP was loaded from cache — rebuild the graph if missing.
if "neighbors" not in ad_emb.uns:
    sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
_root_mask = ~ad_emb.obs[CALL].astype(bool).to_numpy()   # benign side as DPT root
assert _root_mask.any(), f"no benign cell for {CALL}"
root_idx = int(np.where(_root_mask)[0][0])
ad_emb.uns["iroot"] = root_idx
sc.tl.diffmap(ad_emb, random_state=0)
sc.tl.dpt(ad_emb)
pt = ad_emb.obs["dpt_pseudotime"].to_numpy()
pt = np.where(np.isfinite(pt), pt, np.nan)   # DPT can emit inf on disconnected components
print(f"pseudotime: root_idx={root_idx}  min={np.nanmin(pt):.3f}  "
      f"median={np.nanmedian(pt):.3f}  max={np.nanmax(pt):.3f}  n_nan={np.isnan(pt).sum()}")
```

```
pseudotime: root_idx=0  min=0.000  median=0.557  max=1.000  n_nan=0
```

```python  # cell 26
# 2) Editable gene list — default = top 4 from each of Z_1, Z_7, Z_6.
GENES_FOR_PLANE = (
    top_df["proj_Z_1"].head(4).tolist()
    + top_df["proj_Z_7"].head(4).tolist()
    + top_df["proj_Z_6"].head(4).tolist()
)
GENES_FOR_PLANE = list(dict.fromkeys(GENES_FOR_PLANE))  # de-dup, preserve order
missing = [g for g in GENES_FOR_PLANE if g not in adata_cd4.var_names]
assert not missing, f"missing in adata_cd4: {missing}"
print(f"{len(GENES_FOR_PLANE)} genes:", GENES_FOR_PLANE)
```

```
12 genes: ['CXCL13', 'EGLN3', 'ADGRG1', 'NTRK2', 'MYOD1', 'KIF20A', 'DLGAP5', 'UBE2C', 'GZMB', 'CCL4', 'XCL1', 'XCL2']
```

```python  # cell 27
# 3) Candidate direction vectors in (Z_1, Z_7, Z_6) — OLS coeffs on standardized axes.
from numpy.linalg import lstsq
from sklearn.linear_model import LogisticRegression
from scipy.sparse import issparse

Z3  = np.column_stack([z[:, FX3], z[:, FY3], z[:, FZ3]])
Z3z = np.column_stack([xz3, yz3, zz3])                     # from the 3-D scatter cell
A   = np.column_stack([Z3z, np.ones(len(Z3z))])            # intercept column

# log-normalized expression on the fly (adata_cd4.X is raw counts)
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
dir_rows.append(("LR_optimal_(Z1,Z7,Z6)", lr.coef_.ravel().astype(np.float64)))

dirs = {name: v / max(np.linalg.norm(v), 1e-12) for name, v in dir_rows}
print(f"{len(dirs)} candidate directions:", list(dirs.keys()))
```

```
15 candidate directions: ['diagonal_(1,1,1)', 'pseudotime', 'CXCL13', 'EGLN3', 'ADGRG1', 'NTRK2', 'MYOD1', 'KIF20A', 'DLGAP5', 'UBE2C', 'GZMB', 'CCL4', 'XCL1', 'XCL2', 'LR_optimal_(Z1,Z7,Z6)']
```

```python  # cell 28
# 4) Score each direction: AUROC + Youden-best threshold + accuracy stats.
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support,
)
from scipy.stats import fisher_exact

y_true = is_mal


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
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0,
    )
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return dict(
        name=name, AUROC=auc, acc=acc, prec=p, rec=r, f1=f1,
        TP=tp, FP=fp, FN=fn, TN=tn, OR=float(odds), p=float(pval),
        vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
        thr=best_thr, flipped=flipped,
    )


results = (
    pd.DataFrame([eval_direction(n, v) for n, v in dirs.items()])
    .sort_values("AUROC", ascending=False)
    .reset_index(drop=True)
)
with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4f}".format):
    print(results.to_string(index=False))
out_csv = FIG_DIR / f"semantic_geom_cd4_{CALL}_gene_planes_stats.csv"
results.to_csv(out_csv, index=False)
print("saved", out_csv)
results
```

```
name  AUROC    acc   prec    rec     f1    TP    FP    FN    TN     OR      p      vx      vy      vz     thr  flipped
LR_optimal_(Z1,Z7,Z6) 0.7853 0.7531 0.6803 0.6503 0.6649 37927 17826 20398 78636 8.2022 0.0000  0.9961 -0.0179 -0.0859  0.3505    False
                NTRK2 0.7787 0.7466 0.6720 0.6398 0.6555 37319 18212 21006 78250 7.6333 0.0000  0.9756  0.0933  0.1988  0.4658    False
                EGLN3 0.7640 0.7282 0.6265 0.6898 0.6566 40233 23984 18092 72478 6.7202 0.0000  0.7982 -0.0724  0.5980  0.3109    False
               CXCL13 0.7639 0.7231 0.6293 0.6453 0.6372 37640 22174 20685 74288 6.0963 0.0000  0.9131  0.0551 -0.4039  0.1989    False
               ADGRG1 0.7419 0.7084 0.5967 0.6981 0.6434 40717 27523 17608 68939 5.7921 0.0000  0.5201 -0.2205  0.8251  0.1812    False
     diagonal_(1,1,1) 0.7252 0.6759 0.5560 0.6951 0.6178 40540 32374 17785 64088 4.5124 0.0000  0.5774  0.5774  0.5774  0.1012    False
                 CCL4 0.6512 0.6207 0.4973 0.6274 0.5549 36596 36986 21729 59476 2.7083 0.0000 -0.1713 -0.1193  0.9780  0.0099    False
                 GZMB 0.6257 0.5968 0.4717 0.5849 0.5223 34116 38202 24209 58260 2.1491 0.0000 -0.2647 -0.0378  0.9636  0.0282    False
                 XCL1 0.6133 0.5762 0.4542 0.6178 0.5235 36036 43306 22289 53156 1.9845 0.0000 -0.2917 -0.1169  0.9493 -0.0579    False
                 XCL2 0.5987 0.5880 0.4575 0.5028 0.4791 29324 34771 29001 61691 1.7940 0.0000 -0.3317 -0.0955  0.9386  0.1105    False
           pseudotime 0.5497 0.5543 0.4284 0.5468 0.4804 31891 42555 26434 53907 1.5283 0.0000  0.5202  0.3877 -0.7610  0.0390    False
                MYOD1 0.5394 0.5814 0.4412 0.4160 0.4283 24266 30730 34059 65732 1.5240 0.0000  0.5741  0.1108 -0.8113  0.2301     True
               KIF20A 0.5339 0.5007 0.4136 0.7778 0.5400 45364 64330 12961 32132 1.7482 0.0000 -0.0408  0.9856 -0.1642 -0.5251    False
                UBE2C 0.5262 0.4956 0.4102 0.7730 0.5359 45085 64837 13240 31625 1.6609 0.0000 -0.1268  0.9899 -0.0629 -0.5389    False
               DLGAP5 0.5175 0.4909 0.4073 0.7715 0.5331 44995 65479 13330 30983 1.5972 0.0000 -0.1039  0.9854 -0.1350 -0.5368    False
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_leiden_cnv_malignant_mrvi_gene_planes_stats.csv

                     name     AUROC       acc      prec       rec        f1  \
0   LR_optimal_(Z1,Z7,Z6)  0.785334  0.753054  0.680268  0.650270  0.664931   
1                   NTRK2  0.778708  0.746632  0.672039  0.639846  0.655547   
2                   EGLN3  0.764033  0.728168  0.626516  0.689807  0.656640   
3                  CXCL13  0.763939  0.723110  0.629284  0.645349  0.637215   
4                  ADGRG1  0.741888  0.708432  0.596674  0.698105  0.643416   
5        diagonal_(1,1,1)  0.725172  0.675948  0.555997  0.695071  0.617804   
6                    CCL4  0.651247  0.620672  0.497350  0.627450  0.554876   
7                    GZMB  0.625691  0.596794  0.471750  0.584929  0.522278   
8                    XCL1  0.613285  0.576224  0.454186  0.617848  0.523524   
9                    XCL2  0.598652  0.588002  0.457508  0.502769  0.479072   
10             pseudotime  0.549675  0.554297  0.428378  0.546781  0.480391   
11                  MYOD1  0.539351  0.581431  0.441232  0.416048  0.428270   
12                 KIF20A  0.533938  0.500662  0.413550  0.777780  0.539987   
13                  UBE2C  0.526211  0.495584  0.410154  0.772996  0.535938   
14                 DLGAP5  0.517520  0.490855  0.407290  0.771453  0.533119   

       TP     FP     FN     TN        OR    p        vx        vy        vz  \
0   37927  17826  20398  78636  8.202163  0.0  0.996141 -0.017927 -0.085921   
1   37319  18212  21006  78250  7.633318  0.0  0.975586  0.093304  0.198814   
2   40233  23984  18092  72478  6.720173  0.0  0.798231 -0.072423  0.597982   
3   37640  22174  20685  74288  6.096333  0.0  0.913142  0.055113 -0.403898   
4   40717  27523  17608  68939  5.792
... [1214 more chars truncated]
```

```python  # cell 29
# 5) 3-D scatter: top-AUROC plane (left = label, right = KLF2 expression).
top_row  = results.iloc[0]
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
ZG_top  = plane_z(v_top, thr_top, XG2, YG2)

KLF2_GENE = "KLF2"
assert KLF2_GENE in adata_cd4.var_names, f"{KLF2_GENE} missing from adata_cd4.var_names"
klf2_expr = gene_lognorm(KLF2_GENE)
klf2_plot = klf2_expr[plot_idx][shuf3]
klf2_vmax = float(np.quantile(klf2_expr, 0.99))

fig = plt.figure(figsize=(14, 7))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG2, YG2, ZG_top, color="tab:green", alpha=0.25, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by {CALL}   top: {top_name}   AUROC={top_row['AUROC']:.3f}", fontsize=10)
status_handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",   label=POS_LABEL, markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue",  label=NEG_LABEL, markersize=6),
    plt.Line2D([], [], marker="s", linestyle="", color="tab:green", alpha=0.25, label=f"{top_name} best (AUROC={top_row['AUROC']:.3f})", markersize=10),
]
axL.legend(handles=status_handles, loc="upper left", fontsize=8, frameon=False)

axR = fig.add_subplot(1, 2, 2, projection="3d")
scR = axR.scatter(xx3, yy3, zz3p, c=klf2_plot, cmap="viridis",
                  vmin=0, vmax=klf2_vmax, s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG2, YG2, ZG_top, color="tab:green", alpha=0.25, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by {KLF2_GENE} log-norm expression", fontsize=10)
fig.colorbar(scR, ax=axR, fraction=0.03, pad=0.10).set_label(f"{KLF2_GENE} log1p(cp10k)")

fig.suptitle(f"Best-direction plane ({CALL}): {top_name}", fontsize=12, y=1.02)
fig.tight_layout()
safe = top_name.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "_").replace("/", "_")
out = FIG_DIR / f"semantic_geom_cd4_{CALL}_scatter3d_best_plane_{safe}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_leiden_cnv_malignant_mrvi_scatter3d_best_plane_LR_optimal_Z1_Z7_Z6.png

<Figure size 1400x700 with 3 Axes>
```

```python  # cell 30
# Interactive rotatable 3-D version of the top-AUROC plane (standalone .html).
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import IFrame

mal_mask = is_mal[plot_idx][shuf3] == 1
ben_mask = ~mal_mask

fig3d = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        f"colored by {CALL}   top: {top_name}   AUROC={top_row['AUROC']:.3f}",
        f"colored by {KLF2_GENE} log-norm expression",
    ),
    horizontal_spacing=0.05,
)

fig3d.add_trace(go.Scatter3d(
    x=xx3[mal_mask], y=yy3[mal_mask], z=zz3p[mal_mask],
    mode="markers", name=POS_LABEL,
    marker=dict(size=2, color="crimson", opacity=0.45),
), row=1, col=1)
fig3d.add_trace(go.Scatter3d(
    x=xx3[ben_mask], y=yy3[ben_mask], z=zz3p[ben_mask],
    mode="markers", name=NEG_LABEL,
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
        size=2, color=klf2_plot, colorscale="Viridis",
        cmin=0, cmax=klf2_vmax, opacity=0.55,
        colorbar=dict(title=f"{KLF2_GENE} log1p(cp10k)", thickness=12, x=1.02),
    ),
    name=f"{KLF2_GENE}", showlegend=False,
), row=1, col=2)
fig3d.add_trace(go.Surface(
    x=XG2, y=YG2, z=ZG_top, showscale=False, opacity=0.35,
    colorscale=[[0, "green"], [1, "green"]],
    name="top plane", showlegend=False,
), row=1, col=2)

axes_kw = dict(xaxis_title=f"Z_{FX3}", yaxis_title=f"Z_{FY3}", zaxis_title=f"Z_{FZ3}")
fig3d.update_layout(
    title=f"Best-direction plane ({CALL}): {top_name}",
    width=1300, height=700, scene=axes_kw, scene2=axes_kw,
    legend=dict(itemsizing="constant"),
)

html_out = FIG_DIR / f"semantic_geom_cd4_{CALL}_scatter3d_best_plane_{safe}.html"
fig3d.write_html(html_out, include_plotlyjs="cdn", full_html=True)
print("saved", html_out)
IFrame(html_out.as_posix(), width=1320, height=720)
```
