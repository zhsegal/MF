# old/03_mrvi_replication.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/03_mrvi_replication.ipynb` and is never run._

# CTCL atlas replication — MrVI-centric (tumor_cell = malignant)

Reads the cached MrVI run from `02_mrvi.ipynb`. Malignant cells = `cell_type == "tumor_cell"` (paper label); no inferCNV.

- §1 MrVI sample-level diagnostics (donor distances)
- §2 Malignant vs nonmalignant (tumor_cell vs benign T): 2a pseudobulk DESeq2, 2b MrVI DE, 2c concordance
- §3 Early vs advanced disease (all cells): 3a pseudobulk DESeq2, 3b TH2 fraction (tumor cells), 3c MrVI stage DE, 3d concordance
- §4 MrVI differential abundance (early vs advanced): all-cells UMAP → per-group UMAPs (DA + cell_type) → per-cell-type DA ranking

Heavy MrVI fits/DE run on GPU via `jobs/run_mrvi_*.sh`; the notebook loads their `.nc` outputs.

```python  # cell 1
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax, jaxlib
print(f"jax {jax.__version__} | jaxlib {jaxlib.__version__} | devices {jax.devices()} | backend {jax.default_backend()}")

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import mannwhitneyu, spearmanr
import scvi
from scvi.external import MRVI

sc.settings.set_figure_params(dpi=100, facecolor="white", figsize=(8, 5))
print("scvi", scvi.__version__, "| scanpy", sc.__version__)

def _resolve_nb_dir():
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")

NB_DIR    = _resolve_nb_dir()
CACHE_DIR = NB_DIR / "data" / "cache"
FIG_DIR   = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR   = NB_DIR / "tables";  TAB_DIR.mkdir(exist_ok=True)
MODEL_MB_DIR = NB_DIR / "models" / "mrvi_tumor_vs_benignT"
sc.settings.figdir = str(FIG_DIR)

CACHE_H5AD = CACHE_DIR / "mrvi_ctcl_cache.h5ad"
print("CACHE_H5AD =", CACHE_H5AD)
RNG = 0
```

```
jax 0.8.2 | jaxlib 0.8.2 | devices [CudaDevice(id=0)] | backend gpu
scvi 1.3.3 | scanpy 1.11.5
CACHE_H5AD = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/cache/mrvi_ctcl_cache.h5ad
```

```python  # cell 2
# Extra deps for this notebook (pseudobulk DESeq2 + volcano labelling).
# Run once; %pip lands in the running kernel.
try:
    import pydeseq2  # noqa: F401
except ImportError:
    %pip install -q pydeseq2
try:
    import adjustText  # noqa: F401
except ImportError:
    %pip install -q adjustText

# ---- pseudobulk + DESeq2 + volcano helpers (used by §2a and §3a) ----
import scipy.sparse as _sp

def pseudobulk_counts(adata, sample_col, layer="raw_counts", min_cells=20):
    """Sum raw counts per pseudo-sample. Returns (samples x genes int DataFrame, n_cells Series)."""
    X = adata.layers[layer]
    samples = adata.obs[sample_col].astype(str).values
    rows, idx, ncells = [], [], {}
    for s in pd.unique(samples):
        mask = samples == s
        n = int(mask.sum())
        if n < min_cells:
            continue
        sub = X[mask]
        v = np.asarray(sub.sum(axis=0)).ravel() if _sp.issparse(sub) else np.asarray(sub).sum(0)
        rows.append(v); idx.append(s); ncells[s] = n
    counts = pd.DataFrame(np.vstack(rows), index=idx, columns=adata.var_names.astype(str))
    counts = counts.round().astype(int)
    return counts, pd.Series(ncells, name="n_cells")

def run_pydeseq2(counts, metadata, design, contrast, min_total=10, n_cpus=4):
    """Pseudobulk DESeq2 (NB GLM + Wald, BH-adjusted) — DESeq2-equivalent of the paper's QLF F-test.
    Returns results_df with extra `logfoldchanges`/`pvals_adj` aliases for downstream concordance."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts = counts.loc[:, counts.sum(axis=0) >= min_total]
    try:
        dds = DeseqDataSet(counts=counts, metadata=metadata, design=design,
                           quiet=True, n_cpus=n_cpus)
    except TypeError:  # older pydeseq2 API
        factors = [t.strip() for t in design.replace("~", "").split("+") if t.strip()]
        dds = DeseqDataSet(counts=counts, metadata=metadata, design_factors=factors, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    res = st.results_df.copy()
    res["logfoldchanges"] = res["log2FoldChange"]
    res["pvals_adj"] = res["padj"]
    return res

def plot_volcano(res, key_genes, title, fname, lfc_thr=1.0, padj_thr=0.05,
                 n_label=12, figsize=(8, 6)):
    """Volcano: colour significant genes (red up / blue down), label curated + top-|LFC| hits."""
    d = res.dropna(subset=["log2FoldChange", "padj"]).copy()
    d["nlog10p"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = d[(d["padj"] < padj_thr) & (d["log2FoldChange"].abs() > lfc_thr)]
    up, dn = sig[sig["log2FoldChange"] > 0], sig[sig["log2FoldChange"] < 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(d["log2FoldChange"], d["nlog10p"], s=5, alpha=0.25, color="lightgrey")
    ax.scatter(up["log2FoldChange"], up["nlog10p"], s=12, alpha=0.7, color="#d62728", label=f"up ({len(up)})")
    ax.scatter(dn["log2FoldChange"], dn["nlog10p"], s=12, alpha=0.7, color="#1f77b4", label=f"down ({len(dn)})")

    lab = [g for g in key_genes if g in d.index]
    lab += [g for g in sig.reindex(sig["log2FoldChange"].abs().sort_values(ascending=False).index).head(n_label).index
            if g not in lab]
    texts = []
    for g in lab:
        r = d.loc[g]
        ax.scatter(r["log2FoldChange"], r["nlog10p"], s=30, facecolor="none", edgecolor="k", lw=0.8)
        texts.append(ax.text(r["log2FoldChange"], r["nlog10p"], g, fontsize=9))
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    except Exception:
        pass
    ax.axhline(-np.log10(padj_thr), color="grey", lw=0.5, ls="--")
    ax.axvline(lfc_thr, color="grey", lw=0.5, ls="--"); ax.axvline(-lfc_thr, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("log2 fold change"); ax.set_ylabel("-log10 adjusted p")
    ax.set_title(title); ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.show()
    return sig

# ---- MrVI per-gene DE table + method-comparison barplot (used by §2b/c and §3c/d) ----
def mrvi_de_table(donor_lfc):
    """Per-gene MrVI DE from a (donors x genes) LFC matrix: mean log2FC + one-sample t-test
    vs 0 across donors, BH-adjusted. Donor is the replicate unit (avoids pseudoreplication).
    Output columns mirror plot_volcano (log2FoldChange/padj) + concordance aliases."""
    from scipy.stats import ttest_1samp
    X = np.asarray(donor_lfc.values, float)
    lfc = np.nanmean(X, axis=0)
    with np.errstate(all="ignore"):
        _, p = ttest_1samp(X, 0.0, axis=0, nan_policy="omit")
    p = np.asarray(p, float); p[~np.isfinite(p)] = 1.0
    m = len(p); order = np.argsort(p)
    adj = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    padj = np.empty(m); padj[order] = np.clip(adj, 0, 1)
    out = pd.DataFrame({"log2FoldChange": lfc, "padj": padj}, index=list(donor_lfc.columns))
    out["logfoldchanges"] = out["log2FoldChange"]; out["pvals_adj"] = out["padj"]
    return out

def plot_method_barh(pb_de, mrvi_de, key_genes, title, fname):
    """Grouped horizontal barplot comparing pseudobulk vs MrVI log2 fold change for curated genes."""
    genes = [g for g in key_genes if g in pb_de.index and g in mrvi_de.index]
    pb = pb_de.loc[genes, "logfoldchanges"].astype(float).values
    mr = mrvi_de.loc[genes, "logfoldchanges"].astype(float).values
    y = np.arange(len(genes)); h = 0.4
    fig, ax = plt.subplots(figsize=(6, max(3, 0.45 * len(genes))))
    ax.barh(y + h / 2, pb, height=h, color="#4c72b0", label="pseudobulk DESeq2")
    ax.barh(y - h / 2, mr, height=h, color="#dd8452", label="MrVI")
    ax.set_yticks(y); ax.set_yticklabels(genes); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("log2 fold change"); ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.show()
```

## Load cache + reload MrVI

```python  # cell 4
adata = sc.read_h5ad(CACHE_H5AD)
print(adata)
art = adata.uns["mrvi_artifacts"]
print("artifacts:", dict(art))

model = MRVI.load(str(art["model_dir"]), adata=adata, accelerator="gpu")
print("reloaded model. trained:", model.is_trained)

# Recreate stage_group + refresh sample_info on the loaded model
EARLY = {"IA", "IB", "IIA"}
adata.obs["stage_group"] = (
    adata.obs["stage"].astype(str).map(lambda s: "early" if s in EARLY else "advanced")
    .astype("category").cat.reorder_categories(["early", "advanced"])
)
assert (adata.obs.groupby("donor", observed=True)["stage_group"].nunique() == 1).all()
model.update_sample_info(adata)
print("donors per stage_group:")
print(adata.obs.drop_duplicates("donor").groupby("stage_group", observed=True)["donor"].nunique())
```

```
AnnData object with n_obs × n_vars = 419579 × 10000
    obs: 'tissue', 'sex', 'study', 'donor', 'tech', 'cell_type', 'stage', 'n_counts', '_indices', '_scvi_sample', '_scvi_batch', '_scvi_labels'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    uns: '_scvi_manager_uuid', '_scvi_uuid', 'broad_ct_colors', 'cell_type_colors', 'donor_colors', 'groups1_colors', 'hvg', 'leiden', 'log1p', 'mrvi', 'mrvi_artifacts', 'neighbors', 'scvi', 'stage_colors', 'study_colors', 'tech_colors', 'umap'
    obsm: 'X_mrvi', 'X_scVI', 'X_scVI_MDE', 'X_umap', 'X_umap_mrvi', 'X_umap_scvi'
    layers: 'raw_counts'
    obsp: 'connectivities', 'distances', 'mrvi_connectivities', 'mrvi_distances', 'scvi_connectivities', 'scvi_distances'
artifacts: {'batch_key': 'study', 'de_stage_tcells_nc': '/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/mrvi_de_stage_tcells.nc', 'hvg_n_top_genes': 10000, 'model_dir': '/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/mrvi_ctcl', 'sample_key': 'donor', 'umap_keys': array(['X_umap_mrvi', 'X_umap_scvi'], dtype=object)}
INFO     File /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/mrvi_ctcl/model.pt already     
         downloaded                                                                                                
INFO     Jax module moved to cuda:0.Note: Pytorch lightning will show GPU is not being used for the Trainer.       

/home/projects/nyosef/zvise/scvi-tools-neural-nmf/src/scvi/module/base/_base_module.py:723: UserWarning: Running mrVI with Jax version larger 0.4.35 can cause performance issues
  model.train(max_steps=1)
GPU available: True (cuda), used: False
TPU available: False, using: 0 TPU cores
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/lightning/pytorch/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.

Epoch 1/19:   5%|▌         | 1/19 [00:17<05:23, 18.00s/it, v_num=1, train_loss_step=6.04e+3, train_loss_epoch=6.04e+3]
`Trainer.fit` stopped: `max_steps=1` reached.

Epoch 1/19:   5%|▌         | 1/19 [00:18<05:24, 18.00s/it, v_num=1, train_loss_step=6.04e+3, train_loss_epoch=6.04e+3]
reloaded model. trained: True
donors per stage_group:
stage_group
early       17
advanced    19
Name: donor, dtype: int64
```

## §1 MrVI sample-level diagnostics

Donor-distance heatmap (averaged across cell types). Differential abundance (DA) is loaded and analysed in §4.

```python  # cell 6
# Submit MrVI inference (donor distances + DA) to gsla_high_gpu via bsub.
# Flip SUBMIT=True; set FORCE=True to overwrite existing .nc outputs.
import subprocess

SUBMIT       = False
FORCE        = False
JOBS_DIR     = NB_DIR / "jobs"
INF_SCRIPT   = JOBS_DIR / "run_mrvi_inference.sh"
INF_LOG      = JOBS_DIR / "run_mrvi_inference.bsub.log"
DIST_NC      = FIG_DIR / "mrvi_donor_distances.nc"
DA_NC        = FIG_DIR / "mrvi_da_stage.nc"

if SUBMIT:
    cmd = ["bash", str(INF_SCRIPT)] + (["--force"] if FORCE else [])
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, end=""); print(r.stderr, end="")
    print(f"\nlog: {INF_LOG}\nstatus: !bjobs   tail: !tail -n 40 -f {INF_LOG}")
else:
    print("set SUBMIT=True to fire bsub.")
print("DIST_NC exists:", DIST_NC.exists())
print("DA_NC   exists:", DA_NC.exists())
```

```
set SUBMIT=True to fire bsub.
DIST_NC exists: True
DA_NC   exists: True
```

```python  # cell 7
adata.obs.cell_type.value_counts()
```

```
cell_type
tumor_cell             102367
Th                      52824
Tc                      39889
Treg                    27614
Differentiated_KC       24682
F2                      19903
B_cell                  15211
VE2                     12528
Differentiated_KC*      10985
DC2                      9807
MigDC                    7062
F3                       6719
Mono_mac                 6384
Melanocyte               5640
Plasma                   5569
moDC_3                   5557
DC1                      5240
Pericyte_2               4760
Pericyte_1               4636
Tc17_Th17                4233
Macro_1                  3993
VE3                      3544
immune                   3466
Macro_2                  3058
VE1                      2712
pDC                      2654
NK                       2398
Undifferentiated_KC      2379
Tc_IL13_IL22             2336
moDC_2                   2267
F1                       2188
Proliferating_KC         1975
Inf_mac                  1966
LE1                      1939
LC_4                     1419
moDC_1                   1319
ILC1_3                   1275
ILC1_NK                  1090
LE2                       982
LC_2                      714
Schwann_1                 704
Mast_cell                 663
LC_3                      637
basal2                    541
LC_1                      470
Sebaceous                 418
channel                   371
Basal                     358
ILC2                      133
Name: count, dtype: int64
```

```python  # cell 8
# Load donor distances from bsub output and plot clustermap.
assert DIST_NC.exists(), f"missing {DIST_NC} — submit the inference job above first"
dist = xr.open_dataset(DIST_NC)
print(dist)

# Average across cell types -> donor x donor distance matrix
D = dist["cell_type"].mean(dim="cell_type_name").to_pandas()
D.index.name = "donor"; D.columns.name = "donor"

donor_stage = (adata.obs.drop_duplicates("donor")
               .set_index("donor")["stage_group"].astype(str))
row_colors = donor_stage.reindex(D.index).map({"early": "#4daf4a", "advanced": "#e41a1c"})

g = sns.clustermap(
    D, cmap="RdBu_r", center=0, figsize=(12, 12),
    row_colors=row_colors, col_colors=row_colors,
    xticklabels=True, yticklabels=True,
)
g.fig.suptitle("MrVI donor distance (mean over cell_type)\nred = advanced, green = early", y=1.02)
g.savefig(FIG_DIR / "mrvi_donor_distances.png", dpi=150)
plt.show()
```

```
<xarray.Dataset> Size: 259kB
Dimensions:         (cell_type_name: 49, sample_x: 36, sample_y: 36)
Coordinates:
  * cell_type_name  (cell_type_name) <U19 4kB 'tumor_cell' 'Th' ... 'ILC2'
  * sample_x        (sample_x) <U6 864B 'CTCL1' 'CTCL2' ... 'PT55' 'PT56'
  * sample_y        (sample_y) <U6 864B 'CTCL1' 'CTCL2' ... 'PT55' 'PT56'
Data variables:
    cell_type       (cell_type_name, sample_x, sample_y) float32 254kB ...

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/seaborn/matrix.py:530: ClusterWarning: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
  linkage = hierarchy.linkage(self.array, method=self.method,
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/seaborn/matrix.py:530: ClusterWarning: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
  linkage = hierarchy.linkage(self.array, method=self.method,

<Figure size 1200x1200 with 6 Axes>
```

## §2 Malignant vs nonmalignant — tumor_cell vs benign T cells

Malignant = `cell_type == "tumor_cell"` (paper label); no inferCNV. Nonmalignant = benign T cells (`Th, Tc, Treg, Tc17_Th17, Tc_IL13_IL22`). Subset to these cells, label `is_malignant`, report composition.

```python  # cell 10
TUMOR = "tumor_cell"
BENIGN_T = ["Th", "Tc", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]

ct = adata.obs["cell_type"].astype(str)
ad_t = adata[ct.isin([TUMOR, *BENIGN_T])].copy()   # keeps raw_counts layer
ad_t.obs["is_malignant"] = (ad_t.obs["cell_type"].astype(str) == TUMOR).values
print(ad_t)
print(f"malignant (tumor_cell): {int(ad_t.obs['is_malignant'].sum())} / {ad_t.n_obs} "
      f"({ad_t.obs['is_malignant'].mean():.1%})")

obs = ad_t.obs
print("\n== malignant by stage_group ==")
by_stage = (obs.groupby("stage_group", observed=True)["is_malignant"]
              .agg(n_cells="size", n_malignant="sum", frac_malignant="mean"))
print(by_stage.assign(frac_malignant=lambda d: d["frac_malignant"].round(3)))

print("\n== per-donor malignant fraction ==")
per_donor_stat = (obs.groupby("donor", observed=True)
                    .agg(n_cells=("is_malignant", "size"),
                         n_malignant=("is_malignant", "sum"),
                         frac_malignant=("is_malignant", "mean"),
                         stage_group=("stage_group", "first")))
print(f"donors: {len(per_donor_stat)}  |  with >=200 malignant: "
      f"{(per_donor_stat['n_malignant'] >= 200).sum()}")
print(per_donor_stat["frac_malignant"].describe().round(3))
per_donor_stat.to_csv(TAB_DIR / "Section2_malignant_stats.csv")

print("\n== cell_type composition ==")
print(obs["cell_type"].value_counts())
```

```
AnnData object with n_obs × n_vars = 229263 × 10000
    obs: 'tissue', 'sex', 'study', 'donor', 'tech', 'cell_type', 'stage', 'n_counts', '_indices', '_scvi_sample', '_scvi_batch', '_scvi_labels', 'stage_group', 'is_malignant'
    var: 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm', 'highly_variable_nbatches'
    uns: '_scvi_manager_uuid', '_scvi_uuid', 'broad_ct_colors', 'cell_type_colors', 'donor_colors', 'groups1_colors', 'hvg', 'leiden', 'log1p', 'mrvi', 'mrvi_artifacts', 'neighbors', 'scvi', 'stage_colors', 'study_colors', 'tech_colors', 'umap'
    obsm: 'X_mrvi', 'X_scVI', 'X_scVI_MDE', 'X_umap', 'X_umap_mrvi', 'X_umap_scvi'
    layers: 'raw_counts'
    obsp: 'connectivities', 'distances', 'mrvi_connectivities', 'mrvi_distances', 'scvi_connectivities', 'scvi_distances'
malignant (tumor_cell): 102367 / 229263 (44.7%)

== malignant by stage_group ==
             n_cells  n_malignant  frac_malignant
stage_group                                      
early         134444        38550           0.287
advanced       94819        63817           0.673

== per-donor malignant fraction ==
donors: 36  |  with >=200 malignant: 29
count    36.000
mean      0.496
std       0.339
min       0.000
25%       0.204
50%       0.534
75%       0.824
max       0.977
Name: frac_malignant, dtype: float64

== cell_type composition ==
cell_type
tumor_cell      102367
Th               52824
Tc               39889
Treg             27614
Tc17_Th17         4233
Tc_IL13_IL22      2336
Name: count, dtype: int64
```

## §2a Paper method — pseudobulk DESeq2 (malignant vs benign)

Paper Fig 3b method: quasi-likelihood F-test + BH on donor-aggregated pseudobulk (not cell-level Wilcoxon — that inflates significance via pseudoreplication; Squair 2021, Crowell 2022). Aggregate raw counts per **donor × status** and run **pydeseq2** (NB GLM, Wald + BH) with a paired design `~ donor + status`. (`ad_t_ln`, the log-norm subset, is reused by §3b.)

```python  # cell 12
# Log-normalize counts in a copy — reused by §3b (TH2 scoring on tumor cells).
ad_t_ln = ad_t.copy()
ad_t_ln.X = ad_t_ln.layers["raw_counts"].copy()
sc.pp.normalize_total(ad_t_ln, target_sum=1e4)
sc.pp.log1p(ad_t_ln)

# ---- pseudobulk DESeq2: malignant vs benign, paired on donor ----
ad_t.obs["status_mb"] = np.where(ad_t.obs["is_malignant"], "malignant", "benign")
ad_t.obs["pb_sample"] = ad_t.obs["donor"].astype(str) + "|" + ad_t.obs["status_mb"]

counts_mb, ncells_mb = pseudobulk_counts(ad_t, "pb_sample", min_cells=20)
meta_mb = pd.DataFrame(index=counts_mb.index)
meta_mb["donor"]  = [s.split("|")[0] for s in counts_mb.index]
meta_mb["status"] = [s.split("|")[1] for s in counts_mb.index]

# paired design needs both statuses present per donor
paired = meta_mb.groupby("donor")["status"].nunique()
keep_d = paired[paired == 2].index
mask = meta_mb["donor"].isin(keep_d).values
counts_mb, meta_mb = counts_mb.loc[mask], meta_mb.loc[mask]
print(f"paired pseudobulk: {mask.sum()} samples across {len(keep_d)} donors "
      f"(dropped {int((paired < 2).sum())} unpaired donors)")
meta_mb["status"] = pd.Categorical(meta_mb["status"], ["benign", "malignant"])

wil = run_pydeseq2(counts_mb, meta_mb, design="~donor + status",
                   contrast=["status", "malignant", "benign"])
wil.to_csv(TAB_DIR / "Fig3b_DEGs_malignant_vs_benign_pseudobulk.csv")

key_genes = ["TOX", "CD7", "CXCL13", "CD9", "GTSF1", "RUNX3", "GATA3", "TBX21", "CCR4"]
sig_mb = plot_volcano(wil, key_genes,
                      "Fig 3b — pseudobulk DESeq2 (malignant vs benign)",
                      FIG_DIR / "Fig3b_volcano_pseudobulk.png", figsize=(8, 6))
print(f"\nsignificant genes (padj<0.05, |log2FC|>1): {len(sig_mb)}")
print(sig_mb.reindex(sig_mb["log2FoldChange"].abs().sort_values(ascending=False).index).head(20)[["log2FoldChange", "padj", "baseMean"]].round(3))
```

```
WARNING: adata.X seems to be already log-transformed.
paired pseudobulk: 60 samples across 30 donors (dropped 6 unpaired donors)

<Figure size 800x600 with 1 Axes>

significant genes (padj<0.05, |log2FC|>1): 1266
          log2FoldChange  padj  baseMean
GZMK              -6.293   0.0  1824.787
FOXP3             -6.127   0.0   517.307
CD8A              -6.108   0.0  1152.229
KLRG1             -3.325   0.0   159.309
KLRD1             -5.351   0.0   126.610
CCL4L2            -5.757   0.0  1193.142
PTGDR             -5.294   0.0    37.526
KLF3              -4.235   0.0   162.894
RIC3              -5.684   0.0   112.466
EOMES             -4.589   0.0   183.425
LAG3              -5.033   0.0   575.774
ZNF80             -4.751   0.0    24.911
FCRL3             -5.248   0.0    99.222
CD7               -4.950   0.0  3314.742
ITGA4             -3.949   0.0   373.324
KLRC4             -5.250   0.0   119.591
CCL5              -4.111   0.0  4527.574
CD300A            -4.543   0.0    88.785
SHISAL2A          -4.673   0.0    74.607
PLEK              -4.105   0.0   143.215
```

## §2b MrVI DE — malignant vs benign (pseudo-sample fit)

MrVI needs sample-level covariates, but malignant status varies within donor → build per-donor × status pseudo-samples (`donor_mal`/`donor_ben`), dropping those <200 cells. The fit + DE run on GPU via `jobs/run_mrvi_tcells_mb.sh`, which subsets the main cache to tumor_cell + benign T exactly as in §2 above; this notebook loads the result (`mrvi_de_tumor_vs_benignT.nc`). The prep cell below mirrors the job's filtering so cell order matches the saved latent.

```python  # cell 14
ad_t.obs["sample_mb"] = (
    ad_t.obs["donor"].astype(str) + "_" +
    np.where(ad_t.obs["is_malignant"], "mal", "ben")
).astype("category")

sample_counts = ad_t.obs["sample_mb"].value_counts()
keep_samples = sample_counts[sample_counts >= 200].index
print(f"keeping {len(keep_samples)}/{len(sample_counts)} pseudo-samples (>=200 cells)")
ad_t_mb = ad_t[ad_t.obs["sample_mb"].isin(keep_samples)].copy()
ad_t_mb.obs["sample_mb"] = ad_t_mb.obs["sample_mb"].cat.remove_unused_categories()
ad_t_mb.obs["target_status"] = (
    np.where(ad_t_mb.obs["is_malignant"], "malignant", "benign")
)
ad_t_mb.obs["target_status"] = (
    ad_t_mb.obs["target_status"].astype("category")
    .cat.reorder_categories(["benign", "malignant"])
)
print("sample_mb -> target_status mapping (constant per pseudo-sample):")
print(ad_t_mb.obs.groupby("sample_mb", observed=True)["target_status"].nunique().value_counts())
```

```
keeping 58/66 pseudo-samples (>=200 cells)
sample_mb -> target_status mapping (constant per pseudo-sample):
target_status
1    58
Name: count, dtype: int64
```

```python  # cell 15
# Submit MrVI pseudo-sample training + DE (tumor_cell vs benign T) via bsub.
# Flip SUBMIT=True; set FORCE=True to overwrite existing DE netcdf.
import subprocess

SUBMIT       = False
FORCE        = False
MB_SCRIPT    = JOBS_DIR / "run_mrvi_tcells_mb.sh"
MB_LOG       = JOBS_DIR / "run_mrvi_tcells_mb.bsub.log"
MB_LATENT    = CACHE_DIR / "tumor_benignT_X_mrvi_mb.npy"
MB_HISTORY   = FIG_DIR / "mrvi_mb_history.json"
DE_MB_NC     = FIG_DIR / "mrvi_de_tumor_vs_benignT.nc"

if SUBMIT:
    cmd = ["bash", str(MB_SCRIPT)] + (["--force"] if FORCE else [])
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, end=""); print(r.stderr, end="")
    print(f"\nlog: {MB_LOG}\nstatus: !bjobs   tail: !tail -n 40 -f {MB_LOG}")
else:
    print("set SUBMIT=True to fire bsub.")
print("MB_LATENT exists:", MB_LATENT.exists())
print("DE_MB_NC  exists:", DE_MB_NC.exists())
```

```
set SUBMIT=True to fire bsub.
MB_LATENT exists: True
DE_MB_NC  exists: True
```

```python  # cell 16
# Load the trained MrVI pseudo-sample latent (from the bsub job).
# ad_t_mb must already exist from the prep cell above (same filtering as the bsub script).
assert MB_LATENT.exists(), f"missing {MB_LATENT} — submit the bsub job above first"
ad_t_mb.obsm["X_mrvi_mb"] = np.load(MB_LATENT)
print(f"X_mrvi_mb: {ad_t_mb.obsm['X_mrvi_mb'].shape}")
```

```
X_mrvi_mb: (228194, 10)
```

```python  # cell 17
# Load MrVI DE (tumor vs benign), aggregate per-cell LFC over malignant cells to per-donor,
# build a per-gene DE table (mean log2FC + donor-level t-test), and plot a volcano like §2a.
import h5netcdf  # noqa: F401  (force-loadable)
from xarray.backends import plugins as _xr_plugins
try:
    _xr_plugins.list_engines.cache_clear()
except AttributeError:
    pass

assert DE_MB_NC.exists(), f"missing {DE_MB_NC} — submit the bsub job above first"
de_mb = xr.open_dataset(DE_MB_NC, engine="h5netcdf")
cov_mb = [c for c in de_mb.coords["covariate"].values if "target_status" in c][0]
cov_dim_mb = "covariate_sub" if "covariate_sub" in de_mb["lfc"].dims else "covariate"
print("covariate:", cov_mb)

# per-cell × gene LFC for this covariate; restrict to malignant cells, group to per-donor mean
lfc_sel = de_mb["lfc"].sel({cov_dim_mb: cov_mb})                     # (cell_name, gene)
mal_cells = ad_t_mb.obs_names[ad_t_mb.obs["is_malignant"].values]
donor_mal = adata.obs["donor"].reindex(mal_cells).astype(str).values
donor_da = xr.DataArray(donor_mal, dims=["cell_name"],
                        coords={"cell_name": mal_cells.values})
donor_lfc_mb = (lfc_sel.sel(cell_name=mal_cells.values)
                .assign_coords(donor=donor_da).groupby("donor").mean().to_pandas())
print(f"MrVI LFC aggregated over {donor_lfc_mb.shape[0]} donors × {donor_lfc_mb.shape[1]} genes")

mrvi_mb = mrvi_de_table(donor_lfc_mb)
mrvi_mb.to_csv(TAB_DIR / "Fig3b_DEGs_malignant_vs_benign_MRVI.csv")

sig_mrvi_mb = plot_volcano(mrvi_mb, key_genes,
                           "Fig 3b — MrVI DE (malignant vs benign)",
                           FIG_DIR / "Fig3b_volcano_MRVI.png", figsize=(8, 6))
print(f"\nMrVI significant genes (padj<0.05, |log2FC|>1): {len(sig_mrvi_mb)}")
print(sig_mrvi_mb.reindex(sig_mrvi_mb["log2FoldChange"].abs().sort_values(ascending=False).index).head(20)[["log2FoldChange", "padj"]].round(3))
```

```
covariate: target_status_malignant
MrVI LFC aggregated over 29 donors × 10000 genes

<Figure size 800x600 with 1 Axes>

MrVI significant genes (padj<0.05, |log2FC|>1): 20
         log2FoldChange  padj
CD7              -2.038   0.0
SELL             -1.747   0.0
KLF2             -1.588   0.0
GNLY              1.454   0.0
CD8A             -1.330   0.0
FOXP3            -1.287   0.0
LGALS1            1.272   0.0
CD9               1.230   0.0
CD27             -1.160   0.0
FCMR             -1.152   0.0
SESN3            -1.101   0.0
S100A6            1.086   0.0
FTH1             -1.083   0.0
GIMAP4           -1.078   0.0
BHLHE40           1.068   0.0
TIMP1             1.042   0.0
SORL1            -1.035   0.0
CD5              -1.034   0.0
CXCL13            1.028   0.0
RIPOR2           -1.025   0.0
```

## §2c Concordance — pseudobulk DESeq2 ↔ MrVI (malignant vs benign)

```python  # cell 19
# Concordance: pseudobulk DESeq2 (wil) vs MrVI (mrvi_mb), malignant vs benign.
joined = (mrvi_mb[["logfoldchanges"]].rename(columns={"logfoldchanges": "lfc_mrvi"})
          .join(wil[["logfoldchanges"]].rename(columns={"logfoldchanges": "lfc_pb"}), how="inner")
          .dropna())
rho, _ = spearmanr(joined["lfc_pb"], joined["lfc_mrvi"])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(joined["lfc_pb"], joined["lfc_mrvi"], s=4, alpha=0.35, color="grey")
for g in key_genes:
    if g in joined.index:
        ax.scatter(joined.loc[g, "lfc_pb"], joined.loc[g, "lfc_mrvi"], color="red", s=30)
        ax.text(joined.loc[g, "lfc_pb"], joined.loc[g, "lfc_mrvi"], g, fontsize=9)
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("log2FC pseudobulk DESeq2"); ax.set_ylabel("mean log2FC MrVI")
ax.set_title(f"malignant vs benign concordance  Spearman ρ = {rho:.2f}")
fig.tight_layout(); fig.savefig(FIG_DIR / "Fig3b_concordance_pseudobulk_vs_MRVI.png", dpi=200); plt.show()

# Per-gene effect comparison between the two methods (curated genes).
plot_method_barh(wil, mrvi_mb, key_genes,
                 "Malignant vs benign — pseudobulk vs MrVI effect per gene",
                 FIG_DIR / "Fig3b_methods_barh.png")
```

```
<Figure size 600x600 with 1 Axes>
<Figure size 600x405 with 1 Axes>
```

## §3 Early vs advanced disease — all cells

Stage DE across the whole atlas. 3a paper-method pseudobulk DESeq2 (per-donor, all cells); 3b TH2 (GATA3) fraction in tumor cells by stage; 3c MrVI stage DE on all cells (per-CT capped); 3d concordance.

```python  # cell 21
# ---- 3a pseudobulk DESeq2: advanced vs early, ALL cells, per-donor (each donor = one stage) ----
counts_st, ncells_st = pseudobulk_counts(adata, "donor", min_cells=20)
d2s = adata.obs.drop_duplicates("donor").set_index("donor")["stage_group"].astype(str)
meta_st = pd.DataFrame(index=counts_st.index)
meta_st["stage_group"] = pd.Categorical(d2s.reindex(counts_st.index).values, ["early", "advanced"])
print(f"stage pseudobulk (all cells): {len(counts_st)} donors "
      f"({(meta_st['stage_group'] == 'early').sum()} early / "
      f"{(meta_st['stage_group'] == 'advanced').sum()} advanced)")

deg_adv = run_pydeseq2(counts_st, meta_st, design="~stage_group",
                       contrast=["stage_group", "advanced", "early"])
deg_adv.to_csv(TAB_DIR / "Fig4a_DEGs_advanced_vs_early_pseudobulk_allcells.csv")

stage_key = ["GATA3", "IL4", "IL13", "SELL", "CCR7", "LEF1", "TCF7",
             "IFNG", "GZMB", "GNLY", "NKG7", "TBX21", "RORC"]
sig_st = plot_volcano(deg_adv, stage_key,
                      "Fig 4a — pseudobulk DESeq2 (advanced vs early, all cells)",
                      FIG_DIR / "Fig4a_volcano_pseudobulk_allcells.png", figsize=(8, 6))
print(f"\nsignificant genes (padj<0.05, |log2FC|>1): {len(sig_st)}")
print(sig_st.reindex(sig_st["log2FoldChange"].abs().sort_values(ascending=False).index).head(20)[["log2FoldChange", "padj", "baseMean"]].round(3))
```

```
stage pseudobulk (all cells): 36 donors (17 early / 19 advanced)

<Figure size 800x600 with 1 Axes>

significant genes (padj<0.05, |log2FC|>1): 218
          log2FoldChange   padj  baseMean
HBB                6.906  0.000   357.935
LCE3E             -5.213  0.000   136.635
PRR9              -4.910  0.000    38.587
CHIT1              3.614  0.000    39.259
EHF               -3.454  0.000   992.566
FAM71B             5.423  0.000   155.265
CABCOCO1           4.203  0.000    32.294
SPRR2E            -4.896  0.000   951.100
CACNB4            -2.525  0.000   162.951
CLDN14            -3.803  0.000    38.260
HCN4               5.839  0.000     9.069
ATP6V1C2          -2.533  0.001    85.568
HILPDA            -3.320  0.001  5462.201
CXCR5              3.115  0.001   278.345
MEST               2.391  0.001   180.458
FA2H              -3.376  0.001    86.708
NPW                3.482  0.001   208.999
KRT2              -3.801  0.001   546.482
THEM5             -3.296  0.001   102.054
COX6B2            -2.461  0.001    23.978
```

```python  # cell 22
# 3b TH2 (GATA3) fraction in malignant (tumor_cell) cells by stage.
# Master-TF-only scoring: cytokines (IL4/IL13/IL5, IFNG, IL17) are sparse/noisy and dilute
# the lineage signal. Classify each tumor cell by argmax over the three master TFs.
mal_ln = ad_t_ln[ad_t_ln.obs["is_malignant"]].copy()  # log-norm tumor cells (from §2a)
print("tumor cells (log-norm):", mal_ln.shape)

sc.tl.score_genes(mal_ln, ["TBX21"], score_name="Th1_score")
sc.tl.score_genes(mal_ln, ["GATA3"], score_name="Th2_score")
sc.tl.score_genes(mal_ln, ["RORC"],  score_name="Th17_score")
mal_ln.obs["Th_subtype"] = (
    mal_ln.obs[["Th1_score", "Th2_score", "Th17_score"]].idxmax(axis=1).str.replace("_score", "", regex=False)
)

per_donor = (mal_ln.obs.groupby(["donor", "stage_group"], observed=True)["Th_subtype"]
             .value_counts(normalize=True).unstack(fill_value=0).reset_index())
for col in ["Th1", "Th2", "Th17"]:
    if col not in per_donor.columns:
        per_donor[col] = 0.0
per_donor.to_csv(TAB_DIR / "Fig4b_per_donor_TH_fractions.csv", index=False)

early = per_donor.loc[per_donor["stage_group"] == "early", "Th2"]
adv   = per_donor.loc[per_donor["stage_group"] == "advanced", "Th2"]
U, p = mannwhitneyu(early, adv, alternative="less")
print(f"TH2 (GATA3) fraction Mann-Whitney one-sided early<advanced: U={U:.0f}  p={p:.2e}  (paper: 3e-4)")

fig, ax = plt.subplots(figsize=(6, 5))
sns.violinplot(data=per_donor, x="stage_group", y="Th2", inner="box",
               order=["early", "advanced"], ax=ax)
sns.stripplot(data=per_donor, x="stage_group", y="Th2", color="k", size=4,
              order=["early", "advanced"], ax=ax)
ax.set_title(f"TH2 (GATA3) fraction in tumor cells\nMann-Whitney p = {p:.2e}")
fig.tight_layout(); fig.savefig(FIG_DIR / "Fig4b_TH2_violin.png", dpi=200); plt.show()
```

```
tumor cells (log-norm): (102367, 10000)
TH2 (GATA3) fraction Mann-Whitney one-sided early<advanced: U=37  p=9.77e-04  (paper: 3e-4)

<Figure size 600x500 with 1 Axes>
```

## §3c MrVI stage DE — all cells (early vs advanced)

Loads the all-cells stage DE from `jobs/run_mrvi_de.sh --all-cells` (per-cell-type capped subsample across all 49 types → `mrvi_de_stage_allcells.nc`). `stage_group` is constant per donor, so this uses the original full-atlas MrVI model (no new fit).

```python  # cell 24
# Load all-cells stage DE; aggregate per-cell LFC to per-donor, build a per-gene DE table
# (mean log2FC + donor-level t-test), and plot a volcano like §3a.
import h5netcdf  # noqa: F401
from xarray.backends import plugins as _xr_plugins
try:
    _xr_plugins.list_engines.cache_clear()
except AttributeError:
    pass

DE_STAGE_NC = FIG_DIR / "mrvi_de_stage_allcells.nc"
assert DE_STAGE_NC.exists(), f"missing {DE_STAGE_NC} — run jobs/run_mrvi_de.sh --all-cells first"
de_stage = xr.open_dataset(DE_STAGE_NC, engine="h5netcdf")
cov_stage = [c for c in de_stage.coords["covariate"].values if "stage_group" in c][0]
cov_dim_stage = "covariate_sub" if "covariate_sub" in de_stage["lfc"].dims else "covariate"
cell_dim = "cell_name" if "cell_name" in de_stage["lfc"].dims else "cell"
print("covariate:", cov_stage)

lfc_sel = de_stage["lfc"].sel({cov_dim_stage: cov_stage})            # (cell_name, gene)
cell_names = de_stage[cell_dim].values
donor_st = adata.obs["donor"].reindex(cell_names).astype(str)
print(f"DE cells (all-cells capped subsample): {len(cell_names)}  donors: {donor_st.nunique()}")

donor_da = xr.DataArray(donor_st.values, dims=[cell_dim], coords={cell_dim: cell_names})
donor_lfc_st = (lfc_sel.assign_coords(donor=donor_da).groupby("donor").mean().to_pandas())

mrvi_stage = mrvi_de_table(donor_lfc_st)
mrvi_stage.to_csv(TAB_DIR / "Fig4_DEGs_advanced_vs_early_MRVI_allcells.csv")

stage_key = ["GATA3", "IL4", "IL13", "SELL", "CCR7", "LEF1", "TCF7",
             "IFNG", "GZMB", "GNLY", "NKG7", "TBX21", "RORC"]
sig_mrvi_st = plot_volcano(mrvi_stage, stage_key,
                           "Fig 4 — MrVI DE (advanced vs early, all cells)",
                           FIG_DIR / "Fig4_volcano_MRVI_allcells.png", figsize=(8, 6))
print(f"\nMrVI significant genes (padj<0.05, |log2FC|>1): {len(sig_mrvi_st)}")
print(sig_mrvi_st.reindex(sig_mrvi_st["log2FoldChange"].abs().sort_values(ascending=False).index).head(20)[["log2FoldChange", "padj"]].round(3))
```

```
covariate: stage_group_advanced
DE cells (all-cells capped subsample): 62094  donors: 36

<Figure size 800x600 with 1 Axes>

MrVI significant genes (padj<0.05, |log2FC|>1): 0
Empty DataFrame
Columns: [log2FoldChange, padj]
Index: []
```

## §3d Concordance — pseudobulk DESeq2 ↔ MrVI for stage (all cells)

```python  # cell 26
# Concordance: pseudobulk DESeq2 (deg_adv) vs MrVI (mrvi_stage), advanced vs early.
joined_s = (mrvi_stage[["logfoldchanges"]].rename(columns={"logfoldchanges": "lfc_mrvi"})
            .join(deg_adv[["logfoldchanges"]].rename(columns={"logfoldchanges": "lfc_pb"}), how="inner")
            .dropna())
rho_s, _ = spearmanr(joined_s["lfc_pb"], joined_s["lfc_mrvi"])

stage_key = ["GATA3", "IL4", "IL13", "SELL", "CCR7", "LEF1", "TCF7",
             "IFNG", "GZMB", "GNLY", "NKG7"]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(joined_s["lfc_pb"], joined_s["lfc_mrvi"], s=4, alpha=0.35, color="grey")
for g in stage_key:
    if g in joined_s.index:
        ax.scatter(joined_s.loc[g, "lfc_pb"], joined_s.loc[g, "lfc_mrvi"], color="red", s=30)
        ax.text(joined_s.loc[g, "lfc_pb"], joined_s.loc[g, "lfc_mrvi"], g, fontsize=9)
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("log2FC pseudobulk DESeq2 (advanced vs early)")
ax.set_ylabel("mean log2FC MrVI (all cells)")
ax.set_title(f"stage concordance (all cells)  Spearman ρ = {rho_s:.2f}")
fig.tight_layout(); fig.savefig(FIG_DIR / "Fig4_concordance_pseudobulk_vs_MRVI_allcells.png", dpi=200); plt.show()

# Per-gene effect comparison between the two methods (curated stage genes).
plot_method_barh(deg_adv, mrvi_stage, stage_key,
                 "Advanced vs early — pseudobulk vs MrVI effect per gene",
                 FIG_DIR / "Fig4_methods_barh.png")
```

```
<Figure size 600x600 with 1 Axes>
<Figure size 600x495 with 1 Axes>
```

## §4 MrVI differential abundance (early vs advanced)

Per-cell DA log-enrichment from the §1 inference job (`mrvi_da_stage.nc`): for each cell, the log-enrichment of advanced vs early donors in its MrVI neighbourhood — **positive = shifted toward advanced**. Load it, map onto the MrVI UMAP for all cells, then break down by cell-type group and rank individual `cell_type`s by their DA shift.

```python  # cell 28
# Load MrVI differential abundance from the §1 bsub output (advanced − early per cell).
assert DA_NC.exists(), f"missing {DA_NC} — submit the §1 inference job first"
da = xr.open_dataset(DA_NC)
print(da)

le = da["stage_group_log_enrichs"].to_pandas()  # cells x {early, advanced}
adata.obs["mrvi_da_log_enrichment"] = (le["advanced"] - le["early"]).values
adata.obsm["X_umap"] = adata.obsm["X_umap_mrvi"]
vmax = float(np.nanpercentile(np.abs(adata.obs["mrvi_da_log_enrichment"]), 99))
print("\nDA log-enrichment (advanced − early) over all cells:")
print(adata.obs["mrvi_da_log_enrichment"].describe().round(3))
```

```
<xarray.Dataset> Size: 156MB
Dimensions:                  (cell_name: 419579, sample: 36, stage_group: 2)
Coordinates:
  * cell_name                (cell_name) <U49 82MB 'AAACCTGAGAAGCCCA-0_CTCL1_...
  * sample                   (sample) <U6 864B 'CTCL1' 'CTCL5' ... 'CTCL18'
  * stage_group              (stage_group) <U8 64B 'advanced' 'early'
Data variables:
    log_probs                (cell_name, sample) float32 60MB ...
    stage_group_log_probs    (cell_name, stage_group) float64 7MB ...
    stage_group_log_enrichs  (cell_name, stage_group) float64 7MB ...

DA log-enrichment (advanced − early) over all cells:
count    419579.000
mean         -4.603
std          15.095
min        -279.586
25%          -7.146
50%          -0.832
75%           1.949
max         228.645
Name: mrvi_da_log_enrichment, dtype: float64
```

```python  # cell 29
# All cells: MrVI UMAP coloured by DA log-enrichment and by cell_type.
sc.pl.umap(adata, color=["mrvi_da_log_enrichment", "cell_type"], cmap="RdBu_r",
           vmin=-vmax, vmax=vmax, size=2,
           title=["MrVI DA log-enrichment (advanced − early)", "cell_type"],
           show=True)
```

```
<Figure size 1782x500 with 3 Axes>
```

### Per-group breakdown

Groups: **B** (`B_cell, Plasma`), **keratinocyte** (`Differentiated_KC(*), Undifferentiated_KC, Proliferating_KC, Basal, basal2, Sebaceous`), **fibroblast** (`F1–F3`), **APC** (`DC1/2, MigDC, moDC_1–3, pDC, LC_1–4, Macro_1/2, Mono_mac, Inf_mac`). For each group: the MrVI UMAP of just those cells, coloured by DA log-enrichment and by pre-annotated `cell_type`. The table then ranks each fine `cell_type` by mean DA shift — top = the cluster most enriched in advanced samples.

```python  # cell 31
import anndata

CT_GROUPS = {
    "all": None,
    "B": ["B_cell", "Plasma"],
    "keratinocyte": ["Differentiated_KC", "Differentiated_KC*", "Undifferentiated_KC",
                     "Proliferating_KC", "Basal", "basal2", "Sebaceous"],
    "fibroblast": ["F1", "F2", "F3"],
    "APC": ["DC1", "DC2", "MigDC", "moDC_1", "moDC_2", "moDC_3", "pDC",
            "LC_1", "LC_2", "LC_3", "LC_4", "Macro_1", "Macro_2", "Mono_mac", "Inf_mac"],
}

ct = adata.obs["cell_type"].astype(str)
for grp, members in CT_GROUPS.items():
    if members is None:
        continue  # 'all' already shown above
    m = ct.isin(members).values
    if m.sum() == 0:
        print(f"{grp}: no cells in atlas — skipping"); continue
    # matrix-free AnnData (UMAP coords + obs only) keeps memory low
    sub = anndata.AnnData(
        obs=adata.obs.loc[m, ["cell_type", "mrvi_da_log_enrichment", "stage_group"]].copy(),
        obsm={"X_umap": adata.obsm["X_umap"][m]},
    )
    sub.obs["cell_type"] = sub.obs["cell_type"].astype("category").cat.remove_unused_categories()
    vmx = float(np.nanpercentile(np.abs(sub.obs["mrvi_da_log_enrichment"]), 99)) or 1.0
    print(f"\n=== {grp}: {int(m.sum())} cells across {sub.obs['cell_type'].nunique()} cell types ===")
    sc.pl.umap(sub, color=["mrvi_da_log_enrichment", "cell_type"], cmap="RdBu_r",
               vmin=-vmx, vmax=vmx, size=8,
               title=[f"{grp} — DA (advanced − early)", f"{grp} — cell_type"],
               show=True)
```

```
=== B: 20780 cells across 2 cell types ===

<Figure size 1782x500 with 3 Axes>

=== keratinocyte: 41338 cells across 7 cell types ===

<Figure size 1782x500 with 3 Axes>

=== fibroblast: 28810 cells across 3 cell types ===

<Figure size 1782x500 with 3 Axes>

=== APC: 52547 cells across 15 cell types ===

<Figure size 1782x500 with 3 Axes>
```

```python  # cell 32
# Which clusters moved most early -> advanced? Mean DA log-enrichment per cell_type (desc).
ct = adata.obs["cell_type"].astype(str)
tabs = []
for grp, members in CT_GROUPS.items():
    sel = np.ones(adata.n_obs, bool) if members is None else ct.isin(members).values
    t = (adata.obs.loc[sel].groupby("cell_type", observed=True)["mrvi_da_log_enrichment"]
         .agg(n_cells="size", mean_logE="mean", median_logE="median",
              frac_adv_enriched=lambda s: float((s > 0).mean())))
    t = t[t["n_cells"] > 0].sort_values("mean_logE", ascending=False)
    print(f"\n=== {grp}: DA log-enrichment by cell_type (top = most advanced-shifted) ===")
    print(t.round(3))
    tabs.append(t.reset_index().assign(group=grp))

da_by_ct = pd.concat(tabs, ignore_index=True)[
    ["group", "cell_type", "n_cells", "mean_logE", "median_logE", "frac_adv_enriched"]]
da_by_ct.round(4).to_csv(TAB_DIR / "Section4_DA_by_celltype.csv", index=False)
print("\nsaved -> tables/Section4_DA_by_celltype.csv")
```

```
=== all: DA log-enrichment by cell_type (top = most advanced-shifted) ===
                     n_cells  mean_logE  median_logE  frac_adv_enriched
cell_type                                                              
B_cell                 15211      4.831        4.115              0.872
Plasma                  5569      3.829        2.298              0.560
channel                  371      2.855        2.594              0.569
Tc_IL13_IL22            2336      1.142        0.991              0.626
Tc                     39889      0.626        0.701              0.610
tumor_cell            102367      0.033        0.050              0.508
ILC2                     133      0.015       -0.360              0.383
Sebaceous                418     -0.058       -2.120              0.447
NK                      2398     -0.084        0.389              0.535
VE3                     3544     -0.211        1.706              0.588
basal2                   541     -0.344       -1.032              0.466
Th                     52824     -0.489       -0.393              0.419
Treg                   27614     -1.221       -0.271              0.463
immune                  3466     -2.006       -1.059              0.400
LE1                     1939     -2.060       -1.045              0.471
Macro_2                 3058     -2.564       -1.305              0.449
ILC1_NK                 1090     -3.939       -2.744              0.344
Macro_1                 3993     -4.900       -3.186              0.362
pDC                     2654     -4.914       -1.994              0.396
DC2                     9807     -4.938       -3.471              0.384
Tc17_Th17               4233     -4.981       -3.406              0.306
DC1                     5240     -5.331       -3.655              0.391
Mono_mac                6384     -6.353       -1.399              0.443
Proliferating_KC        1975     -6.517       -4.809              0.378
Basal                    358     -7.375       -8.061              0.237
Mast_cell                663     -7.376       -6.269              0.317
ILC1_3                  1275     -8.378       -6.505              0.228
LC_1                     470     -9.074       -7.480              0.243
Schwann_1                704     -9.356       -8.323              0.261
Undifferentiated_KC     2379     -9.756       -8.287              0.304
Pericyte_2              4760     -9.816       -7.555              0.213
F2                     19903    -10.310       -9.077              0.188
Differentiated_KC*     10985    -10.594       -9.237              0.302
Melanocyte              5640    -11.492       -8.046              0.231
MigDC                   7062    -12.263       -9.579              0.222
LC_4                    1419    -14.264      -13.160              0.139
moDC_3                  5557    -15.482      -13.948              0.091
VE2                    12528    -15.746      -11.329              0.152
Differentiated_KC      24682    -16.078      -11.014              0.272
LC_2                     714    -16.901      -15.576              0.179
Pericyte_1              4636    -17.740      -14.493              0.102
F1                      2188    -19.778      -18.212              0.103
moDC_1                  1319    -21.183      -19.002              0.209
moDC_2                  2267    -21.789      -18.686              0.130
VE1                     2712    -22.153      -16.684              0.094
F3                      6719    -23.035      -19.745              0.090
Inf_mac                 1966    -23.639      -18.360              0.209
LE2                      982    -25.607      -22.931              0.095
LC_3                     637    -26.939      -20.716              0.116

=== B: DA log-enrichment by cell_type (top = most advanced-shifted) ===
           n_cells  mean_logE  median_logE  frac_adv_enriched
cell_type                                                    
B_cell       15211      4.831        4.115              0
... [2364 more chars truncated]
```
