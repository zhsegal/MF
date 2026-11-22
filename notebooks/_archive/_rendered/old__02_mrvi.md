# old/02_mrvi.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/02_mrvi.ipynb` and is never run._

# MrVI on CTCL atlas (CTCL-only h5ad)

Following the [MrVI tutorial](https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/scrna/MrVI_tutorial.html) and MRVI package defaults (`DEFAULT_TRAIN_KWARGS` in `scvi/external/mrvi/_model.py`).

**Flow** (see `load_or_train_mrvi`):
1. If both `CACHE_H5AD` + `model.pt` exist → fast restore (no HVG, no train).
2. Else if `model.pt` exists → reload raw h5ad, redo HVG, attach trained model, write cache.
3. Else → load raw, HVG, train (~40m on A100), save model + cache.

Downstream: UMAP compare vs scVI; donor-level early-vs-advanced DE on a stratified T-cell subsample (full T-cell pool OOMs with `store_lfc=True`).

Flags: `FORCE_RETRAIN`, `FORCE_RECOMPUTE_DE`.

```python  # cell 1
import os
# CUDA/JAX preflight (must run BEFORE importing jax or scvi)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax, jaxlib
print(f"jax {jax.__version__} | jaxlib {jaxlib.__version__}")
print("jax devices:", jax.devices())
print("jax default backend:", jax.default_backend())  # MrVI needs 'gpu'

import torch
print(f"torch {torch.__version__} | cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("  ->", torch.cuda.get_device_name(0))

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import xarray as xr
import matplotlib.pyplot as plt
import scvi
from scvi.external import MRVI

sc.settings.set_figure_params(dpi=80, facecolor="white", figsize=(5, 3))
print("scvi", scvi.__version__)

def _resolve_nb_dir():
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")

NB_DIR = _resolve_nb_dir()
DATA_DIR = NB_DIR / "data"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = NB_DIR / "models" / "mrvi_ctcl"; MODEL_DIR.parent.mkdir(exist_ok=True)
CACHE_DIR = NB_DIR / "data" / "cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = str(FIG_DIR)

H5AD_PATH  = DATA_DIR / "CTCL_all_final_portal_tags.h5ad"
CACHE_H5AD = CACHE_DIR / "mrvi_ctcl_cache.h5ad"
DE_NC      = FIG_DIR / "mrvi_de_stage_tcells.nc"

# pipeline flags
FORCE_RETRAIN      = False
FORCE_RECOMPUTE_DE = False

print("H5AD_PATH  =", H5AD_PATH)
print("MODEL_DIR  =", MODEL_DIR, "(model.pt:", (MODEL_DIR/'model.pt').exists(), ")")
print("CACHE_H5AD =", CACHE_H5AD, "(exists:", CACHE_H5AD.exists(), ")")
print("DE_NC      =", DE_NC, "(exists:", DE_NC.exists(), ")")
```

## Load or train MrVI

```python  # cell 3
HVG_N         = 10_000
SAMPLE_KEY    = "donor"
BATCH_KEY     = "study"
COUNTS_LAYER  = "raw_counts"

def _prepare_hvg_adata():
    """Load raw h5ad + select HVGs (seurat_v3, batched on study) — deterministic."""
    a = sc.read_h5ad(H5AD_PATH)
    assert COUNTS_LAYER in a.layers and "X_scVI" in a.obsm
    a.X = a.layers[COUNTS_LAYER].copy()
    sc.pp.highly_variable_genes(
        a, n_top_genes=HVG_N, flavor="seurat_v3", batch_key=BATCH_KEY, subset=True,
    )
    return a

def _train_mrvi(a):
    scvi.settings.seed = 0
    MRVI.setup_anndata(a, layer=COUNTS_LAYER, sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)
    m = MRVI(a)
    m.train(
        max_epochs=100, batch_size=256,
        early_stopping=True, early_stopping_patience=15,
        check_val_every_n_epoch=1, train_size=0.9,
    )
    m.save(str(MODEL_DIR), overwrite=True, save_anndata=False)
    return m

def _write_cache(a):
    """Write slim adata cache. Counts live in .X; HVGs only; X_mrvi present."""
    a.obsm.pop("_scvi_extra_categorical_covs", None)
    a.uns["mrvi_artifacts"] = {
        "model_dir": str(MODEL_DIR.resolve()),
        "de_stage_tcells_nc": str(DE_NC.resolve()),
        "hvg_n_top_genes": HVG_N,
        "sample_key": SAMPLE_KEY,
        "batch_key": BATCH_KEY,
    }
    a.X = a.layers[COUNTS_LAYER]
    a.write_h5ad(CACHE_H5AD, compression="gzip")
    print(f"  cache -> {CACHE_H5AD}  ({CACHE_H5AD.stat().st_size/1e9:.2f} GB)")

def load_or_train_mrvi(force_retrain=False):
    """Returns (adata, model). Always populates adata.obsm['X_mrvi'] and writes cache."""
    model_pt = MODEL_DIR / "model.pt"

    if (not force_retrain) and CACHE_H5AD.exists() and model_pt.exists():
        print("[fast] full restore from cache")
        a = sc.read_h5ad(CACHE_H5AD)
        if COUNTS_LAYER not in a.layers:
            a.layers[COUNTS_LAYER] = a.X.copy()
        m = MRVI.load(str(MODEL_DIR), adata=a)
        return a, m

    if (not force_retrain) and model_pt.exists():
        print("[mid] model.pt found, no h5ad cache — rebuilding HVG adata then attaching model")
        a = _prepare_hvg_adata()
        m = MRVI.load(str(MODEL_DIR), adata=a)
        a.obsm["X_mrvi"] = m.get_latent_representation()
        _write_cache(a)
        return a, m

    print("[cold] no model — HVG + train from scratch")
    a = _prepare_hvg_adata()
    m = _train_mrvi(a)
    a.obsm["X_mrvi"] = m.get_latent_representation()
    _write_cache(a)
    return a, m
```

```python  # cell 4
adata, model = load_or_train_mrvi(force_retrain=FORCE_RETRAIN)
print(adata)
print("X_mrvi:", adata.obsm["X_mrvi"].shape)

# ELBO plot only if history is non-empty (it's empty when restored via MRVI.load)
hist = getattr(model, "history", {}) or {}
if isinstance(hist, dict) and len(hist.get("elbo_train", [])) > 0:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist["elbo_train"], label="train")
    if "elbo_validation" in hist:
        ax.plot(hist["elbo_validation"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("ELBO"); ax.legend(); ax.set_title("MrVI training")
    fig.tight_layout(); fig.savefig(FIG_DIR / "mrvi_elbo.png", dpi=150); plt.show()
else:
    print("(no training history — model was restored from disk)")
```

## UMAP: MrVI vs scVI

```python  # cell 6
# Cache per-rep UMAPs into obsm so re-runs are no-ops.
for rep, key in [("X_mrvi", "mrvi"), ("X_scVI", "scvi")]:
    out_key = f"X_umap_{key}"
    if out_key in adata.obsm:
        continue
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=30, key_added=key)
    sc.tl.umap(adata, min_dist=0.3, neighbors_key=key)
    adata.obsm[out_key] = adata.obsm["X_umap"].copy()
print("UMAPs:", [k for k in adata.obsm if k.startswith("X_umap")])
```

```python  # cell 7
color_keys = ["cell_type", "study", "donor", "stage"]

for col in color_keys:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rep_key, title in zip(axes, ["mrvi", "scvi"], ["MrVI", "scVI"]):
        adata.obsm["X_umap"] = adata.obsm[f"X_umap_{rep_key}"]
        sc.pl.umap(adata, color=col, ax=ax, show=False,
                   legend_loc="on data" if col == "cell_type" else "right margin",
                   legend_fontsize=6, size=2, title=f"{title}: {col}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"umap_mrvi_vs_scvi_{col}.png", dpi=150)
    plt.show()
```

```python  # cell 8
# Re-cache adata with X_mrvi + X_umap_mrvi + X_umap_scvi so the bsub DE job
# (notebooks/MF/jobs/run_mrvi_de.sh) reads a fully populated h5ad.
adata.obsm.pop("_scvi_extra_categorical_covs", None)
adata.uns["mrvi_artifacts"] = {
    "model_dir": str(MODEL_DIR.resolve()),
    "de_stage_tcells_nc": str(DE_NC.resolve()),
    "hvg_n_top_genes": HVG_N,
    "sample_key": SAMPLE_KEY,
    "batch_key": BATCH_KEY,
    "umap_keys": ["X_umap_mrvi", "X_umap_scvi"],
}
adata.X = adata.layers[COUNTS_LAYER]
adata.write_h5ad(CACHE_H5AD, compression="gzip")
print(f"re-cached -> {CACHE_H5AD}  ({CACHE_H5AD.stat().st_size/1e9:.2f} GB)")
```

## Plotting malignancy



```python  # cell 11
adata=sc.read_h5ad(CACHE_H5AD)
```

```python  # cell 12
tadata=adata[adata.obs.cell_type.isin(['tumor_cell','Th','Tc','Treg','Tc_IL13_IL22'])].copy()
tadata
```

```python  # cell 13
fig, axes = plt.subplots(1, 1, figsize=(20, 10))
sc.pl.umap(tadata, color='cell_type',   ax=axes)
```

## Differential expression — donor-level stage_group, T-cells

Binary covariate `stage_group ∈ {early, advanced}` (early = IA/IB/IIA; advanced = IIB/IIIB/IV/IVA2). Restrict to T-cells (paper's stage focus). The full T-cell pool with `store_lfc=True` OOMs (~127k × 10k × 3 float32 arrays ≈ 15 GB), so we **stratified-subsample per `cell_type`**.
`add_batch_specific_offsets=True` because data spans 4 studies.

```python  # cell 15
# Submit DE to gsla_high_gpu via bsub. Flip SUBMIT to True; cell below loads result.
# Re-run the next cell after `bjobs` shows DONE (or watch the log path printed).
import subprocess

SUBMIT       = True
JOBS_DIR     = NB_DIR / "jobs"
JOB_SCRIPT   = JOBS_DIR / "run_mrvi_de.sh"
JOB_LOG      = JOBS_DIR / "run_mrvi_de.bsub.log"

if SUBMIT:
    cmd = ["bash", str(JOB_SCRIPT)]
    if FORCE_RECOMPUTE_DE:
        cmd.append("--force")
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, end=""); print(r.stderr, end="")
    print(f"\nlog: {JOB_LOG}")
    print("status: !bjobs   tail: !tail -n 40 -f", JOB_LOG)
else:
    print("set SUBMIT=True to fire bsub; or run inline via the cell below.")
print("DE_NC exists:", DE_NC.exists())
```

```python  # cell 16
# Check DE job status. Re-run this cell to refresh.
import subprocess

print("=== bjobs -a (mrvi_de) ===")
print(subprocess.run(["bjobs", "-a", "-J", "mrvi_de"], capture_output=True, text=True).stdout or "(no jobs)")

print(f"DE_NC exists: {DE_NC.exists()}" + (f"  ({DE_NC.stat().st_size/1e6:.1f} MB)" if DE_NC.exists() else ""))

if JOB_LOG.exists():
    print(f"\n=== tail {JOB_LOG} ===")
    print(subprocess.run(["tail", "-n", "30", str(JOB_LOG)], capture_output=True, text=True).stdout)
```

```python  # cell 17
# Reconstruct the stratified T-cell subset used by the bsub DE job, then load the netcdf.
# Subsample params must match run_mrvi_de.py defaults (--seed 0 --max-per-ct 1500 --min-per-ct 50).
MAX_PER_CT = 1500
MIN_PER_CT = 50
RNG = np.random.default_rng(0)

t_mask = adata.obs["cell_type"].astype(str).str.contains(r"T[ _]?cell|^Tc|^Th|Treg", regex=True)
t_idx = np.where(t_mask.values)[0]
ct = adata.obs["cell_type"].astype(str).values

keep = []
for _, idx in pd.Series(t_idx).groupby(ct[t_idx]):
    idx = idx.values
    if len(idx) < MIN_PER_CT:
        continue
    keep.append(idx if len(idx) <= MAX_PER_CT else RNG.choice(idx, size=MAX_PER_CT, replace=False))
keep = np.sort(np.concatenate(keep))

adata_t = adata[keep].copy()
print("DE subset:", adata_t.shape)
print(adata_t.obs["cell_type"].value_counts())

assert DE_NC.exists(), f"missing {DE_NC} — submit the bsub job in the cell above first"
de_res = xr.open_dataset(DE_NC)
print(f"loaded DE <- {DE_NC}")
print(de_res)
```

```python  # cell 18
cov_name = [c for c in de_res.coords["covariate"].values if "stage_group" in c][0]
print("covariate:", cov_name)

es = de_res["effect_size"].sel(covariate=cov_name).values
adata.obs["mrvi_es_stage"] = np.nan
adata.obs.loc[adata_t.obs_names, "mrvi_es_stage"] = es

adata.obsm["X_umap"] = adata.obsm["X_umap_mrvi"]
vmax = float(np.nanpercentile(np.abs(es), 99))
sc.pl.umap(
    adata, color="mrvi_es_stage", cmap="RdBu_r",
    vmin=-vmax, vmax=vmax, size=2, na_color="lightgrey",
    title=f"MrVI effect size: {cov_name}",
    save="_mrvi_es_stage.png", show=True,
)
```

```python  # cell 19
# lfc lives on `covariate_sub` (batch offsets aren't stored for LFC), unlike
# effect_size which is on `covariate`. Select on whichever dim is present.
lfc_da = de_res["lfc"]
cov_dim = "covariate_sub" if "covariate_sub" in lfc_da.dims else "covariate"
lfc = lfc_da.sel({cov_dim: cov_name}).to_pandas()  # cells x genes
lfc.index = adata_t.obs_names
lfc.columns = adata_t.var_names

mean_lfc_by_ct = lfc.groupby(adata_t.obs["cell_type"].values).mean()
mean_lfc_by_ct.to_csv(FIG_DIR / "mrvi_mean_lfc_by_celltype.csv")

top_per_ct = {
    ct: mean_lfc_by_ct.loc[ct].reindex(mean_lfc_by_ct.loc[ct].abs().sort_values(ascending=False).index).head(10)
    for ct in mean_lfc_by_ct.index
}
for ct, s in top_per_ct.items():
    print(f"\n=== {ct} ===")
    print(s.to_string())
```

```python  # cell 20
union_top = sorted({g for s in top_per_ct.values() for g in s.index})
H = mean_lfc_by_ct[union_top]
vmax = float(np.nanpercentile(np.abs(H.values), 99))

fig, ax = plt.subplots(figsize=(min(0.25 * len(union_top) + 2, 18), 0.35 * len(H) + 2))
im = ax.imshow(H.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(union_top))); ax.set_xticklabels(union_top, rotation=90, fontsize=7)
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index)
ax.set_title(f"Mean LFC advanced vs early ({cov_name})")
fig.colorbar(im, ax=ax, shrink=0.6, label="LFC")
fig.tight_layout(); fig.savefig(FIG_DIR / "mrvi_lfc_heatmap.png", dpi=150); plt.show()
```
