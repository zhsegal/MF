# old/21b_tcr_clonality_descriptive.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/21b_tcr_clonality_descriptive.ipynb` and is never run._

# 21b — CD4 malignant burden (early vs advanced MF)

Loads nb21's saved T object (`skin_T_tcr_malig_v2.h5ad`), keeps **CD4 cells only** (MF + HC donors),
and summarises the ALICE malignancy call (`tcr_malignant_alice`) per sample. CPU-light.

Stages: **MF-early** {IA, IB, IIA} · **MF-advanced** {IIB, III, IV\*, advanced/late} · **HC**.

Figures: malignant vs total CD4 per sample · malignant fraction by stage · malignant count by stage ·
malignant fraction by disease.

```python  # cell 1
# ============================================================
# Parameters — load nb21's saved annotated T object + stage buckets.
# ============================================================
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)

# ---- input: nb21 output (cohort-filtered T cells + tcr_malignant_alice) ----
OUT_DIR = NB_DIR / "data" / "atlas_joint"
ADATA_MALIG_H5AD = OUT_DIR / "skin_T_tcr_malig_v2.h5ad"

CD4_T_TYPES = ["CD4"]
EARLY = {"IA", "IB", "IIA"}          # advanced = IIB|III|IV* + {advanced, late}

FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 2
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

np.random.seed(0)
plt.rcParams["figure.dpi"] = 120
```

```python  # cell 3
# Load nb21's saved annotated T object; keep CD4 cells only (incl. HC donors).
if not ADATA_MALIG_H5AD.exists():
    raise FileNotFoundError(f"{ADATA_MALIG_H5AD} missing — run 21_tcr_alice_neighborhood.ipynb first")
adata = sc.read_h5ad(ADATA_MALIG_H5AD)
adata = adata[adata.obs["cell_type_T2"].isin(CD4_T_TYPES)].copy()
print(adata)
print("tcr_malignant_alice:\n", adata.obs["tcr_malignant_alice"].value_counts())
```

```
AnnData object with n_obs × n_vars = 169014 × 49683
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagreement', 'leiden_broad', 'cell_type_broad', 'leiden_T', 'cell_type_T', 'cached_malignant', '_li_clone_key', 'tcr_clone_id', 'tcr_clone_size', 'tcr_is_expanded', 'tcr_is_dominant_clone', 'tcr_is_malignant', 'cell_type_T2', 'tcr_malignant_alice'
    uns: 'cell_type_T_colors', 'cell_type_broad_colors', 'cell_type_colors', 'dendrogram_leiden_T', 'dendrogram_leiden_broad', 'disease_colors', 'leiden_T', 'leiden_T_colors', 'leiden_broad', 'leiden_broad_colors', 'log1p', 'neighbors', 'pca', 'sample_id_colors', 'study_colors', 'umap'
    obsm: 'X_mrvi_u', 'X_mrvi_z', 'X_pca', 'X_scVI', 'X_scVI_MDE', 'X_umap', '_scvi_extra_categorical_covs'
    varm: 'PCs'
    layers: 'counts', 'raw_counts'
    obsp: 'connectivities', 'distances'
tcr_malignant_alice:
 tcr_malignant_alice
False    100394
True      68620
Name: count, dtype: int64
```

```python  # cell 4
# stage_group bucket: HC / MF-early / MF-advanced / other (case-insensitive on disease_stage).
import re
_ADV = re.compile(r"IIB|III|IV", re.I)


def bucket(disease, stage):
    if str(disease).strip() == "HC":
        return "HC"
    su = str(stage).strip().upper()
    if su in EARLY or su == "EARLY":
        return "MF-early"
    if _ADV.search(su) or su in {"ADVANCED", "LATE"}:
        return "MF-advanced"
    return "other"


adata.obs["stage_group"] = [bucket(d, s) for d, s in
                            zip(adata.obs["disease"].astype(str), adata.obs["disease_stage"].astype(str))]
print(adata.obs.drop_duplicates("donor")["stage_group"].value_counts())
```

```
stage_group
MF-advanced    14
other          10
MF-early        7
HC              3
Name: count, dtype: int64
```

```python  # cell 5
# Per-donor CD4 summary: malignant burden vs sample size, with stage + disease.
per_donor = (adata.obs.groupby("donor", observed=True)
             .agg(n_tcells=("tcr_malignant_alice", "size"),
                  n_malignant=("tcr_malignant_alice", "sum"),
                  disease=("disease", "first"),
                  stage_group=("stage_group", "first"))
             .reset_index())
per_donor["prop_malignant"] = per_donor["n_malignant"] / per_donor["n_tcells"]

PAL = {"MF-early": "#4daf4a", "MF-advanced": "#e41a1c", "HC": "#377eb8", "other": "#999999"}
ORDER = ["MF-early", "MF-advanced", "HC"]


def box_strip(ax, df, group, value, order, ylabel=None, palette=None):
    data = [df.loc[df[group].eq(g), value].dropna().to_numpy() for g in order]
    ax.boxplot(data, labels=[str(g).replace("MF-", "") for g in order], showfliers=False, widths=0.6)
    for i, (g, arr) in enumerate(zip(order, data), 1):
        ax.scatter(np.random.normal(i, 0.06, len(arr)), arr, s=18,
                   color=(palette or {}).get(g, "#377eb8"), alpha=0.85, zorder=3)
    ax.set_ylabel(ylabel or value)


print(per_donor)
```

```
donor  n_tcells  n_malignant     disease  stage_group  \
0               D1__P76      7966         5815          MF  MF-advanced   
1              D1__P115       490            0          HC           HC   
2              D1__P116       239            0          HC           HC   
3              D1__P171      6590         5559          MF  MF-advanced   
4              D1__P204      6998         1070          MF  MF-advanced   
5              D1__P311      5550         3776          MF  MF-advanced   
6              D1__P312      1926          848          MF  MF-advanced   
7               D1__PGS      3976         3267          MF  MF-advanced   
8               D3__P87      9245         1147  CTCL_other        other   
9               D3__P94      4520            0  CTCL_other        other   
10             D3__P105       879           66  CTCL_other        other   
11             D3__P192      4543         3363  CTCL_other        other   
12             D3__P196      1035          515  CTCL_other        other   
13            D5__MFIVB      1444         1081          MF  MF-advanced   
14               H__HC1         9            0          HC           HC   
15             H__MFIV1        48           20          MF        other   
16               H__SS1       105           62          MF        other   
17               H__SS2        94           22          MF        other   
18               H__SS3        42           15          MF        other   
19               H__SS4       245          176          MF        other   
20  Li2024_atlas__CTCL1      8202         6932          MF  MF-advanced   
21  Li2024_atlas__CTCL2      3589         1824          MF     MF-early   
22  Li2024_atlas__CTCL3     10631         8068          MF     MF-early   
23  Li2024_atlas__CTCL4      5968          718          MF     MF-early   
24  Li2024_atlas__CTCL5     11087         6379          MF  MF-advanced   
25  Li2024_atlas__CTCL6     13062         3077          MF     MF-early   
26  Li2024_atlas__CTCL7     37077         3062          MF     MF-early   
27  Li2024_atlas__CTCL8      9937         5250          MF     MF-early   
28   Li2024_atlas__PT11      1540         1137          MF  MF-advanced   
29   Li2024_atlas__PT35      3589         2474          MF  MF-advanced   
30   Li2024_atlas__PT47       415          227          MF  MF-advanced   
31   Li2024_atlas__PT50      2294          369          MF     MF-early   
32   Li2024_atlas__PT53      1925         1185          MF  MF-advanced   
33   Li2024_atlas__PT56      3754         1116          MF  MF-advanced   

    prop_malignant  
0         0.729977  
1         0.000000  
2         0.000000  
3         0.843551  
4         0.152901  
5         0.680360  
6         0.440291  
7         0.821680  
8         0.124067  
9         0.000000  
10        0.075085  
11        0.740260  
12        0.497585  
13        0.748615  
14        0.000000  
15        0.416667  
16        0.590476  
17        0.234043  
18        0.357143  
19        0.718367  
20        0.845160  
21        0.508220  
22        0.758913  
23        0.120308  
24        0.575359  
25        0.235569  
26        0.082585  
27        0.528328  
28        0.738312  
29        0.689329  
30        0.546988  
31        0.160854  
32        0.615584  
33        0.297283
```

```python  # cell 6
227/415
```

```
0.5469879518072289
```

## Figures

Per-sample CD4 malignant burden. Stage palette `MF-early=green`, `MF-advanced=red`, `HC=blue`. Saved to `figures/`.

```python  # cell 8
# Fig 1 — malignant cells vs total CD4 T cells per sample.
fig, ax = plt.subplots(figsize=(5, 3))
for g in [*ORDER, "other"]:
    s = per_donor[per_donor["stage_group"].eq(g)]
    if len(s):
        ax.scatter(s["n_tcells"], s["n_malignant"], s=30, color=PAL[g], label=g, alpha=0.85)
mx = per_donor["n_tcells"].max()
ax.plot([0, mx], [0, mx], "k--", lw=0.8)
ax.set_xlabel("CD4 T cells in sample"); ax.set_ylabel("malignant cells")
ax.set_title("Malignant cells vs sample size"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "cd4_malignant_vs_size.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

```python  # cell 9
# Fig 2 — proportion malignant: early vs advanced vs HC.
d = per_donor[per_donor["stage_group"].isin(ORDER)]
fig, ax = plt.subplots(figsize=(5, 3))
box_strip(ax, d, "stage_group", "prop_malignant", ORDER,
          ylabel="malignant fraction of CD4", palette=PAL)
ax.set_title("Malignant fraction by MF stage")
fig.tight_layout(); fig.savefig(FIG_DIR / "cd4_prop_malignant_by_stage.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
/tmp/ipykernel_3853867/1414790819.py:16: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  ax.boxplot(data, labels=[str(g).replace("MF-", "") for g in order], showfliers=False, widths=0.6)

<Figure size 600x360 with 1 Axes>
```

```python  # cell 10
# Fig 3 — number of malignant cells: early vs advanced vs HC.
fig, ax = plt.subplots(figsize=(5, 3))
box_strip(ax, d, "stage_group", "n_malignant", ORDER, ylabel="malignant cells", palette=PAL)
ax.set_title("Malignant cell count by MF stage")
fig.tight_layout(); fig.savefig(FIG_DIR / "cd4_n_malignant_by_stage.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
/tmp/ipykernel_3853867/1414790819.py:16: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  ax.boxplot(data, labels=[str(g).replace("MF-", "") for g in order], showfliers=False, widths=0.6)

<Figure size 600x360 with 1 Axes>
```

```python  # cell 11
# Fig 4 — malignant fraction by disease type.
dis_order = [x for x in ["MF", "SS", "CTCL_other", "HC"] if x in set(per_donor["disease"])]
fig, ax = plt.subplots(figsize=(5, 3))
box_strip(ax, per_donor, "disease", "prop_malignant", dis_order, ylabel="malignant fraction of CD4")
ax.set_title("Malignant fraction by disease")
fig.tight_layout(); fig.savefig(FIG_DIR / "cd4_prop_malignant_by_disease.png", dpi=150, bbox_inches="tight")
plt.show()
```
