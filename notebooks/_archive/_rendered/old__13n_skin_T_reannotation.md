# old/13n_skin_T_reannotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/13n_skin_T_reannotation.ipynb` and is never run._

# 13n · Skin T cells — re-annotation (per-cell CD4/CD8 quadrant, then per-quadrant clustering)

Re-dive into the ~402k skin **T** compartment from `13_skin_lowres_annotation.ipynb`. nb13's CD4/CD8
split was implicit in a wholesale cluster→subtype mapping and ignores per-cell dropout. Here we:

1. **Score lineage marker modules** — CD8 (CD8A/CD8B) and a broadened CD4 (CD4/CD40LG/IL7R), plus
   NK/γδ/ILC/CD3 modules.
2. **Per-cell quadrant call** on the two scores (one threshold each): `CD4`, `CD8`,
   `double_positive` (both on), `double_negative` (both off) — a full 2×2 partition.
3. **MRVI Leiden separately within each of the 4 quadrants** (`X_mrvi_u`): CD4 & CD8 → Li2024-aligned
   subtypes; DP & DN → profiled per cluster to resolve doublets / NK / γδ / low-quality.
4. **Validate** against Li2024 anchors + per-study balance + a malignant-CD4 guard.

**New obs columns** (old `cell_type_T` kept): `score_cd8`/`score_cd4`, `lineage_cd48` ∈ {CD4, CD8,
double_positive, double_negative}, `leiden_quad` (quadrant-local cluster id), and `cell_type_T2`
(resolved subtype/identity). Written to a **new** file `skin_T_annotated_v2.h5ad` — nbs 14–21
untouched until you switch.

**HEAVY cells (neighbors / Leiden / score_genes on 401k cells) run on the GPU kernel
(`neural_nmf_env`).** The user executes; this notebook only defines the steps.

```python  # cell 1
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def _resolve_nb_dir():
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(start)


NB_DIR = _resolve_nb_dir()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1
OUT = NB_DIR / "data" / "atlas_joint"

T_ANNOT        = OUT / "skin_T_annotated.h5ad"     # input (nb13)
CD4_CACHE_CSV  = OUT / "skin_T_cd4_quad.csv"       # per-quadrant Leiden + UMAP caches (by cell_id)
CD8_CACHE_CSV  = OUT / "skin_T_cd8_quad.csv"
DP_CACHE_CSV   = OUT / "skin_T_dp_quad.csv"
DN_CACHE_CSV   = OUT / "skin_T_dn_quad.csv"
T_ANNOT_V2     = OUT / "skin_T_annotated_v2.h5ad"  # output

FORCE_QUAD = True   # True -> recompute per-quadrant neighbors/Leiden/UMAP; False -> load from cache
print("NB_DIR:", NB_DIR)

# shared lineage panel — included in EVERY quadrant dot plot so off-lineage clusters are visible
# (e.g. a CD8 or NK cluster hiding inside the CD4 quadrant)
MARKERS_LINEAGE = {
    "T core": ["CD3D", "CD3E", "TRAC"],
    "CD4": ["CD4", "CD40LG", "IL7R"],
    "CD8": ["CD8A", "CD8B"],
    "NK": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "gd": ["TRDC", "TRGC1", "TRGC2"],
    "ILC": ["KIT", "GATA3"],
}
XUMAP = ["CD4", "CD8A", "CD8B", "NKG7", "TRDC"]   # cross-lineage UMAP overlays for every quadrant


def mrvi_subcluster(ad_full, mask, key, res, cache_csv, seed=SEED, force=False):
    """Full per-quadrant pipeline on X_mrvi_u: neighbors + Leiden + UMAP. Cached by cell_id. HEAVY."""
    sub = ad_full[mask].copy()
    if cache_csv.exists() and not force:
        df = pd.read_csv(cache_csv).set_index("cell_id")
        ids = sub.obs["cell_id"].astype(str)
        sub.obs[key] = df[key].astype(str).reindex(ids).to_numpy()
        sub.obsm["X_umap"] = df.loc[ids, ["umap1", "umap2"]].to_numpy()
        print(f"[{key}] loaded cached Leiden + UMAP ({sub.n_obs} cells) <- {cache_csv.name}")
    else:
        np.random.seed(seed)
        sc.pp.neighbors(sub, use_rep="X_mrvi_u", random_state=seed)
        sc.tl.leiden(sub, resolution=res, random_state=seed, key_added=key,
                     flavor="igraph", n_iterations=2, directed=False)
        sc.tl.umap(sub, random_state=seed)
        pd.DataFrame({"cell_id": sub.obs["cell_id"].astype(str).to_numpy(),
                      key: sub.obs[key].astype(str).to_numpy(),
                      "umap1": sub.obsm["X_umap"][:, 0],
                      "umap2": sub.obsm["X_umap"][:, 1]}).to_csv(cache_csv, index=False)
        print(f"[{key}] computed neighbors + Leiden (res={res}) + UMAP on {sub.n_obs} cells, "
              f"cached -> {cache_csv.name}")
    sub.obs[key] = sub.obs[key].astype("category")
    return sub


def _present(marker_dict):
    """Drop genes absent from adt.var_names and any now-empty groups."""
    out = {k: [g for g in v if g in adt.var_names] for k, v in marker_dict.items()}
    return {k: v for k, v in out.items() if v}
```

```
NB_DIR: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

## 1 — Load the nb13 T-cell object

`skin_T_annotated.h5ad` (~402k T cells): log1p `X`, raw `counts` layer, `X_mrvi_u`/`X_mrvi_z`
latents, cached T-`X_umap`, `leiden_T`, `cell_type_T`, and the Li2024 `cell_type` scaffold.

```python  # cell 3
adt = sc.read_h5ad(T_ANNOT)
assert "counts" in adt.layers, "counts layer missing"
assert float(adt.X.max()) < 20, "X is not log1p-normalized"
assert "X_mrvi_u" in adt.obsm and "X_umap" in adt.obsm, "need X_mrvi_u (call) + X_umap (overlays)"
print(adt)
print("\nold cell_type_T (nb13):")
print(adt.obs["cell_type_T"].value_counts().to_string())
```

```
AnnData object with n_obs × n_vars = 401566 × 49683
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagreement', 'leiden_broad', 'cell_type_broad', 'leiden_T', 'cell_type_T'
    uns: 'cell_type_T_colors', 'cell_type_broad_colors', 'cell_type_colors', 'dendrogram_leiden_T', 'dendrogram_leiden_broad', 'disease_colors', 'leiden_T', 'leiden_T_colors', 'leiden_broad', 'leiden_broad_colors', 'log1p', 'neighbors', 'pca', 'sample_id_colors', 'study_colors', 'umap'
    obsm: 'X_mrvi_u', 'X_mrvi_z', 'X_pca', 'X_scVI', 'X_scVI_MDE', 'X_umap', '_scvi_extra_categorical_covs'
    varm: 'PCs'
    layers: 'counts', 'raw_counts'
    obsp: 'connectivities', 'distances'

old cell_type_T (nb13):
cell_type_T
CD4_Th2          200504
CD4_Cm            61707
CD4_mem           50117
CD4_Treg          31273
CD8_effector      30020
CD8_em            26810
UNK                 885
CD4_cytotoxic       250
```

## 2 — Lineage marker module scores

`sc.tl.score_genes` on the log1p `X` for each module. CD4 mRNA is severely dropout-prone, so a
single-gene CD4 score is too weak — the **CD4 module is broadened to CD4 + CD40LG (CD154, highly
CD4-restricted) + IL7R**; CD8 stays CD8A/CD8B (CD8B = specificity anchor). `score_cd8` and
`score_cd4` are kept as **two independent axes** (the CD4/CD8 call in §5 is a 2-D quadrant on them,
not a single delta). Plus contaminant modules NK, γδ (TRGC/TRDC), ILC, and a CD3 T-core score.
Per-cell detection booleans (CD8A/CD8B/CD4/CD40LG > 0) quantify dropout per cluster. The histogram
below lets you eyeball provisional CD8 / CD4 cut points.

```python  # cell 5
from scipy.sparse import issparse

# CD4 mRNA is severely dropout-prone, so a single-gene CD4 score is too weak. Broaden the CD4 module
# with specific helper co-markers: CD40LG (CD154, highly CD4-restricted) and IL7R (lifts detection;
# mildly shared with CD8-mem so kept alongside CD4/CD40LG, not alone). CD8 stays CD8A/CD8B (CD8B is
# the specificity anchor). score_cd8 and score_cd4 are treated as TWO independent axes (no delta).
MARKER_SETS = {
    "score_cd3": ["CD3D", "CD3E", "TRAC"],
    "score_cd8": ["CD8A", "CD8B"],
    "score_cd4": ["CD4", "CD40LG", "IL7R"],
    "score_nk":  ["NKG7", "GNLY", "KLRD1", "NCAM1", "KLRF1"],
    "score_gd":  ["TRGC1", "TRGC2", "TRDC"],
    "score_ilc": ["KIT", "GATA3"],
}
present = set(adt.var_names)
for name, genes in MARKER_SETS.items():
    g = [x for x in genes if x in present]
    miss = sorted(set(genes) - present)
    if miss:
        print(f"{name}: missing {miss}")
    sc.tl.score_genes(adt, g, score_name=name, random_state=SEED)

# per-cell detection from raw counts (dropout accounting; mean over a cluster = fraction expressing)
for gx in ["CD8A", "CD8B", "CD4", "CD40LG"]:
    if gx in present:
        col = adt[:, gx].layers["counts"]
        col = col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()
        adt.obs[f"det_{gx}"] = (col > 0).astype(float)
    else:
        adt.obs[f"det_{gx}"] = 0.0

cols = ["score_cd3", "score_cd8", "score_cd4", "det_CD8A", "det_CD8B", "det_CD4", "det_CD40LG"]
print(adt.obs[cols].describe().round(3).to_string())
```

```
score_cd3   score_cd8   score_cd4    det_CD8A    det_CD8B     det_CD4  det_CD40LG
count  401566.000  401566.000  401566.000  401566.000  401566.000  401566.000  401566.000
mean        0.678      -0.124       0.286       0.152       0.031       0.129       0.162
std         0.747       0.531       0.644       0.359       0.172       0.335       0.369
min        -1.194      -0.706      -0.883       0.000       0.000       0.000       0.000
25%         0.184      -0.396      -0.298       0.000       0.000       0.000       0.000
50%         0.689      -0.303       0.245       0.000       0.000       0.000       0.000
75%         1.203      -0.191       0.744       0.000       0.000       0.000       0.000
max         3.488       3.922       3.542       1.000       1.000       1.000       1.000
```

```python  # cell 6
# per-cell distributions to eyeball the CD8 and CD4 thresholds (set them in §3)
fig, axes = plt.subplots(1, 3, figsize=(13, 3))

# (a) score_cd8 — the right shoulder is the CD8 programme; cut to its left = CD8-positive
axes[0].hist(adt.obs["score_cd8"], bins=120, color="tab:blue")
axes[0].set_yscale("log"); axes[0].set_title("(a) score_cd8"); axes[0].set_xlabel("score_cd8")

# (b) score_cd4 — broadened module; right shoulder = CD4 programme
axes[1].hist(adt.obs["score_cd4"], bins=120, color="tab:red")
axes[1].set_yscale("log"); axes[1].set_title("(b) score_cd4"); axes[1].set_xlabel("score_cd4")

# (c) joint density — CD8 top-left, CD4 bottom-right, double-neg bottom-left, double-pos top-right
hb = axes[2].hexbin(adt.obs["score_cd4"], adt.obs["score_cd8"], gridsize=60, bins="log", cmap="viridis")
axes[2].set_xlabel("score_cd4"); axes[2].set_ylabel("score_cd8")
axes[2].set_title("(c) joint density"); fig.colorbar(hb, ax=axes[2], label="log N")
plt.tight_layout(); plt.show()
```

```
<Figure size 1300x300 with 4 Axes>
```

## 3 — Per-cell CD4/CD8 quadrant call

Classify **every cell** on its two scores into four quadrants with one threshold each (`CD8_T`,
`CD4_T`), read off the §2 histograms:

- **CD8** — score_cd8 ≥ `CD8_T` and score_cd4 < `CD4_T`
- **CD4** — score_cd4 ≥ `CD4_T` and score_cd8 < `CD8_T`
- **double_positive** — both ≥ threshold (doublets / activated / ambient)
- **double_negative** — both < threshold (γδ / NKT / low-quality dropout)

The thresholds fully partition the cells. Each quadrant then gets its **own** MRVI clustering
(§4–§7); NK/γδ/ILC are resolved there, not as a global pre-pass.

```python  # cell 8
# one per-cell threshold per marker — read off the §2 histograms, then re-run. Fully partitions
# the cells into the four quadrants (no ambiguous bucket).
CD8_T = 0.050   # score_cd8 >= CD8_T  -> CD8 programme on
CD4_T = 0.050   # score_cd4 >= CD4_T  -> CD4 programme on

s8, s4 = adt.obs["score_cd8"].to_numpy(), adt.obs["score_cd4"].to_numpy()
cd8_on, cd4_on = s8 >= CD8_T, s4 >= CD4_T

quad = np.empty(adt.n_obs, dtype=object)
quad[cd8_on & ~cd4_on]  = "CD8"
quad[cd4_on & ~cd8_on]  = "CD4"
quad[cd8_on & cd4_on]   = "double_positive"
quad[~cd8_on & ~cd4_on] = "double_negative"
adt.obs["lineage_cd48"] = pd.Categorical(quad)

print(adt.obs["lineage_cd48"].value_counts(dropna=False).to_string())
print(f"\nassigned to CD4/CD8: {adt.obs['lineage_cd48'].isin(['CD4', 'CD8']).mean():.1%} of T cells")
```

```
lineage_cd48
CD4                197599
double_negative    140970
CD8                 32049
double_positive     30948

assigned to CD4/CD8: 57.2% of T cells
```

```python  # cell 9
# downsampled scatter in CD4/CD8 score space, coloured by the per-cell quadrant; lines = thresholds
rng = np.random.default_rng(SEED)
idx = rng.choice(adt.n_obs, size=min(25000, adt.n_obs), replace=False)
sub = adt.obs.iloc[idx]
PAL = {"CD4": "tab:red", "CD8": "tab:blue",
       "double_positive": "tab:purple", "double_negative": "lightgrey"}
plt.figure(figsize=(5.5, 5))
for lab, g in sub.groupby("lineage_cd48"):
    plt.scatter(g["score_cd4"], g["score_cd8"], s=3, c=PAL.get(lab, "k"),
                label=lab, alpha=0.4, edgecolors="none")
plt.axvline(CD4_T, ls="--", c="grey", lw=0.8); plt.axhline(CD8_T, ls="--", c="grey", lw=0.8)
plt.xlabel("score_cd4"); plt.ylabel("score_cd8")
plt.title(f"per-cell quadrant call (n={len(idx):,} shown)")
plt.legend(fontsize=7, markerscale=2); plt.tight_layout(); plt.show()
```

```
/tmp/ipykernel_3853867/2570388241.py:8: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for lab, g in sub.groupby("lineage_cd48"):

<Figure size 550x500 with 1 Axes>
```

## 4 — CD4 quadrant: MRVI clustering + Li2024-aligned subtypes

Cluster the high-confidence CD4 cells on their **own** `X_mrvi_u` neighbor graph, dot-plot CD4
helper markers, then map clusters → `CD4_Treg` (FOXP3/IL2RA/CTLA4), `CD4_Th2` (GATA3/CCR4),
`CD4_Th`/`CD4_Tcm` (CCR7/SELL/TCF7), `CD4_Tfh` (CXCL13/PDCD1), `CD4_cytotoxic` (GZMK/GZMB).
**HEAVY (GPU).**

```python  # cell 11
cd4 = mrvi_subcluster(adt, (adt.obs["lineage_cd48"] == "CD4").to_numpy(),
                      key="leiden_cd4", res=2.0, cache_csv=CD4_CACHE_CSV, force=FORCE_QUAD)
print("clusters:", cd4.obs["leiden_cd4"].value_counts().sort_index().to_dict())

# lineage panel (catch CD8/NK/gd contamination) + CD4 subtype markers
markers_cd4 = _present({**MARKERS_LINEAGE,
    "Treg": ["FOXP3", "IL2RA", "CTLA4"], "Th2": ["GATA3", "CCR4", "IL4R"],
    "Naive/cm": ["CCR7", "SELL", "TCF7"], "Th17": ["RORC", "CCR6", "IL17A"],
    "Tfh": ["CXCL13", "PDCD1", "ICOS"], "Cytotoxic": ["GZMK", "GZMB"], "Prolif": ["MKI67", "TOP2A"]})
sc.pl.dotplot(cd4, markers_cd4, groupby="leiden_cd4", standard_scale="var", dendrogram=True)
sc.pl.umap(cd4, color=["leiden_cd4", *XUMAP, "FOXP3", "GATA3"],
           ncols=4, frameon=False, size=4, legend_loc="on data")
```

```
[leiden_cd4] computed neighbors + Leiden (res=2.0) + UMAP on 197599 cells, cached -> skin_T_cd4_quad.csv
clusters: {'0': 10477, '1': 3517, '2': 6162, '3': 9426, '4': 13671, '5': 1711, '6': 10854, '7': 1166, '8': 9570, '9': 3071, '10': 2911, '11': 10140, '12': 12228, '13': 7075, '14': 6787, '15': 13204, '16': 12953, '17': 1483, '18': 9800, '19': 4096, '20': 10094, '21': 1122, '22': 8749, '23': 8265, '24': 7725, '25': 1507, '26': 9810, '27': 25}
WARNING: dendrogram data not found (using key=dendrogram_leiden_cd4). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 1562x1080 with 6 Axes>
<Figure size 2911.2x960 with 15 Axes>
```

```python  # cell 12
# template (re-run if clusters changed): print({c: "" for c in cd4.obs["leiden_cd4"].cat.categories})
# Li2024-aligned: CD4_Treg / CD4_Th2 / CD4_Th / CD4_Tcm / CD4_Tfh / CD4_cytotoxic
# fill each cluster from the §4 dot plot (all "CD4" for now)
cluster2ct_cd4 = {
    '0': 'CD4',
    '1': 'CD4',
    '2': 'CD4',
    '3': 'CD4',
    '4': 'CD4',
    '5': 'CD4',
    '6': 'CD4',
    '7': 'CD4',
    '8': 'CD4',
    '9': 'CD4',
    '10': 'NK',
    '11': 'CD4',
    '12': 'CD4',
    '13': 'CD4',
    '14': 'CD4',
    '15': 'CD4',
    '16': 'CD4',
    '17': 'CD4',
    '18': 'CD4',
    '19': 'CD4_Treg',
    '20': 'CD4',
    '21': 'CD4',
    '22': 'CD4',
    '23': 'CD4_Treg',
    '24': 'CD4_Treg',
    '25': 'CD4_Treg',
    '26': 'CD4',
    '27': 'CD4',
}

if (not cluster2ct_cd4 or any(v == "" for v in cluster2ct_cd4.values())
        or set(cluster2ct_cd4) != set(cd4.obs["leiden_cd4"].cat.categories)):
    print("cluster2ct_cd4 unfilled / keys mismatch — fill from the §4 dot plot.")
    print("template:", {c: "" for c in cd4.obs["leiden_cd4"].cat.categories})
else:
    cd4.obs["cell_type_T2"] = cd4.obs["leiden_cd4"].map(cluster2ct_cd4).astype("category")
    print(cd4.obs["cell_type_T2"].value_counts().to_string())
```

```
cell_type_T2
CD4         173095
CD4_Treg     21593
NK            2911
```

```python  # cell 13
sc.pl.umap(cd4, color=["leiden_cd4", "cell_type_T2"],
           ncols=4, frameon=False, size=4, legend_loc="on data")
```

```
<Figure size 1455.6x480 with 2 Axes>
```

## 5 — CD8 quadrant: MRVI clustering + Li2024-aligned subtypes

Cluster the high-confidence CD8 cells on their own `X_mrvi_u` graph, dot-plot CD8 effector/memory
markers, then map clusters → `CD8_effector` (GZMB/PRF1/GNLY), `CD8_em` (GZMK), `CD8_Tc17`
(RORC/CCR6/IL17A), `CD8_naive_cm` (CCR7/SELL/TCF7), `CD8_exhausted_Trm` (PDCD1/TOX/LAG3/ZNF683/ITGAE).
**HEAVY (GPU).**

```python  # cell 15
cd8 = mrvi_subcluster(adt, (adt.obs["lineage_cd48"] == "CD8").to_numpy(),
                      key="leiden_cd8", res=2.0, cache_csv=CD8_CACHE_CSV, force=FORCE_QUAD)
print("clusters:", cd8.obs["leiden_cd8"].value_counts().sort_index().to_dict())

# lineage panel (catch CD4/NK/gd contamination) + CD8 subtype markers
markers_cd8 = _present({**MARKERS_LINEAGE,
    "Naive/cm": ["CCR7", "SELL", "TCF7"], "Effector": ["GZMB", "PRF1", "GNLY"],
    "EM": ["GZMK", "CCL5"], "Tc17": ["RORC", "CCR6", "IL17A"],
    "Exhaust/Trm": ["PDCD1", "TOX", "LAG3", "ZNF683", "ITGAE"], "Prolif": ["MKI67", "TOP2A"]})
sc.pl.dotplot(cd8, markers_cd8, groupby="leiden_cd8", standard_scale="var", dendrogram=True)
sc.pl.umap(cd8, color=["leiden_cd8", *XUMAP, "GZMB", "GZMK"],
           ncols=4, frameon=False, size=4, legend_loc="on data")
```

```
[leiden_cd8] computed neighbors + Leiden (res=2.0) + UMAP on 32049 cells, cached -> skin_T_cd8_quad.csv
clusters: {'0': 1740, '1': 1350, '2': 550, '3': 1505, '4': 550, '5': 703, '6': 1338, '7': 59, '8': 589, '9': 894, '10': 1411, '11': 787, '12': 710, '13': 1023, '14': 747, '15': 451, '16': 1943, '17': 612, '18': 2220, '19': 490, '20': 1023, '21': 1037, '22': 487, '23': 1805, '24': 1264, '25': 382, '26': 949, '27': 1033, '28': 624, '29': 1714, '30': 783, '31': 1276}
WARNING: dendrogram data not found (using key=dendrogram_leiden_cd8). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 1525x1220 with 6 Axes>
<Figure size 2911.2x960 with 15 Axes>
```

```python  # cell 16
# template (re-run if clusters changed): print({c: "" for c in cd8.obs["leiden_cd8"].cat.categories})
# Li2024-aligned: CD8_naive_cm / CD8_effector / CD8_em / CD8_Tc17 / CD8_exhausted_Trm
cluster2ct_cd8 = {
    '0': 'CD8',
    '1': 'CD8',
    '2': 'CD8',
    '3': 'CD8',
    '4': 'CD8',
    '5': 'CD8',
    '6': 'CD8',
    '7': 'CD8',
    '8': 'CD8',
    '9': 'CD8',
    '10': 'CD8',
    '11': 'CD8',
    '12': 'CD8',
    '13': 'CD8',
    '14': 'CD8',
    '15': 'CD8',
    '16': 'CD8',
    '17': 'CD8',
    '18': 'CD8',
    '19': 'CD8',
    '20': 'CD8',
    '21': 'CD8',
    '22': 'CD8',
    '23': 'CD8',
    '24': 'CD8',
    '25': 'CD8',
    '26': 'CD8',
    '27': 'CD8',
    '28': 'CD8',
    '29': 'CD8',
    '30': 'CD8',
    '31': 'CD8',
   
    
}

if (not cluster2ct_cd8 or any(v == "" for v in cluster2ct_cd8.values())
        or set(cluster2ct_cd8) != set(cd8.obs["leiden_cd8"].cat.categories)):
    print("cluster2ct_cd8 unfilled / keys mismatch — fill from the §5 dot plot.")
    print("template:", {c: "" for c in cd8.obs["leiden_cd8"].cat.categories})
else:
    cd8.obs["cell_type_T2"] = cd8.obs["leiden_cd8"].map(cluster2ct_cd8).astype("category")
    print(cd8.obs["cell_type_T2"].value_counts().to_string())
```

```
cell_type_T2
CD8    32049
```

## 6 — Double-positive quadrant: rescue clear lineages, then cluster the rest

CD8 **and** CD4 programmes both on. First check whether DP still splits by lineage (histograms +
scatter) and **rescue** the clearly-leaning cells — strong CD4 → `CD4`, strong CD8 → `CD8` (written
into `lineage_cd48`; cells above both cuts stay as true doublets). The remaining double_positive
cells then get their own MRVI clustering, profiled per cluster (module scores + Li2024 labels +
dot plot) and mapped to a label (doublet / NK / gamma_delta / lowq). **HEAVY (GPU).**

```python  # cell 18
# can the DP cells still be split into CD4 vs CD8? look at the scores within this quadrant
dpm = (adt.obs["lineage_cd48"] == "double_positive").to_numpy()
dpo = adt.obs.loc[dpm]
print(f"double_positive cells: {int(dpm.sum()):,}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
# (a) CD8 and CD4 score distributions within DP — bimodal => a sub-population leans one way
axes[0].hist(dpo["score_cd8"], bins=80, color="tab:blue", alpha=0.6, label="score_cd8")
axes[0].hist(dpo["score_cd4"], bins=80, color="tab:red", alpha=0.6, label="score_cd4")
axes[0].axvline(CD8_T, ls="--", c="tab:blue", lw=0.8); axes[0].axvline(CD4_T, ls="--", c="tab:red", lw=0.8)
axes[0].legend(); axes[0].set_title("(a) DP: CD8 vs CD4 score"); axes[0].set_xlabel("score")

# (b) contrast axis — a valley would mean DP separates into a CD8-leaning and CD4-leaning part
delta = (dpo["score_cd8"] - dpo["score_cd4"]).to_numpy()
axes[1].hist(delta, bins=100, color="slategray")
axes[1].axvline(0, c="k", lw=0.6)
axes[1].set_title("(b) DP: score_cd8 − score_cd4"); axes[1].set_xlabel("Δ")

# (c) scatter coloured by which programme is stronger
col = np.where(delta >= 0, "tab:blue", "tab:red")
axes[2].scatter(dpo["score_cd4"], dpo["score_cd8"], s=4, c=col, alpha=0.3, edgecolors="none")
axes[2].axline((0, 0), slope=1, c="k", lw=0.5, ls="--")
axes[2].set_xlabel("score_cd4"); axes[2].set_ylabel("score_cd8")
axes[2].set_title("(c) DP: CD8>CD4 (blue) / CD4>CD8 (red)")
plt.tight_layout(); plt.show()
```

```
double_positive cells: 30,948

<Figure size 1300x320 with 3 Axes>
```

```python  # cell 19
# rescue clearly-leaning DP cells: strong CD4 (>DP_CD4_T) -> CD4, strong CD8 (>DP_CD8_T) -> CD8.
# cells above BOTH cuts stay double_positive (true doublets). Re-run the whole §6 if you re-run §3.
DP_CD4_T, DP_CD8_T = 1.5, 1.7

lin = adt.obs["lineage_cd48"].astype(str).to_numpy()
isdp = lin == "double_positive"
s8, s4 = adt.obs["score_cd8"].to_numpy(), adt.obs["score_cd4"].to_numpy()
resc_cd8 = isdp & (s8 > DP_CD8_T) & (s4 <= DP_CD4_T)
resc_cd4 = isdp & (s4 > DP_CD4_T) & (s8 <= DP_CD8_T)
lin[resc_cd8] = "CD8"
lin[resc_cd4] = "CD4"
adt.obs["lineage_cd48"] = pd.Categorical(lin)

print(f"rescued from DP  ->  CD8: {int(resc_cd8.sum()):,} | CD4: {int(resc_cd4.sum()):,}")
print(f"remaining double_positive: {int((lin == 'double_positive').sum()):,}")
print("\nlineage_cd48 now:")
print(adt.obs["lineage_cd48"].value_counts().to_string())
```

```
rescued from DP  ->  CD8: 1,695 | CD4: 638
remaining double_positive: 28,615

lineage_cd48 now:
lineage_cd48
CD4                198237
double_negative    140970
CD8                 33744
double_positive     28615
```

```python  # cell 20
PROFILE = ["score_cd3", "score_cd8", "score_cd4", "score_nk", "score_gd", "score_ilc",
           "det_CD8B", "det_CD4"]

dp = mrvi_subcluster(adt, (adt.obs["lineage_cd48"] == "double_positive").to_numpy(),
                     key="leiden_dp", res=2.0, cache_csv=DP_CACHE_CSV, force=FORCE_QUAD)
print("clusters:", dp.obs["leiden_dp"].value_counts().sort_index().to_dict())

gp = dp.obs.groupby("leiden_dp", observed=True)
prof = gp[PROFILE].mean().round(3)
prof.insert(0, "n", gp.size())
print("\nper-cluster profile (doublet = CD8 & CD4 both high; NK = high score_nk; gd = high score_gd):")
print(prof.to_string())
print("\nLi2024 labels per cluster:")
print(pd.crosstab(dp.obs["leiden_dp"], dp.obs["cell_type"]).T.to_string())

# full lineage panel to resolve what each DP cluster really is (doublet / NK / gd / lineage)
markers_dp = _present({**MARKERS_LINEAGE, "Myeloid": ["LYZ", "CD68"], "cycling": ["MKI67", "TOP2A"]})
sc.pl.dotplot(dp, markers_dp, groupby="leiden_dp", standard_scale="var")
sc.pl.umap(dp, color=["leiden_dp", *XUMAP, "MKI67"],
           ncols=4, frameon=False, size=6, legend_loc="on data")
```

```
[leiden_dp] computed neighbors + Leiden (res=2.0) + UMAP on 28615 cells, cached -> skin_T_dp_quad.csv
clusters: {'0': 1009, '1': 909, '2': 1003, '3': 1337, '4': 1033, '5': 664, '6': 809, '7': 1468, '8': 1300, '9': 585, '10': 1162, '11': 1268, '12': 725, '13': 1532, '14': 944, '15': 383, '16': 1539, '17': 406, '18': 1319, '19': 536, '20': 590, '21': 490, '22': 1282, '23': 964, '24': 637, '25': 371, '26': 941, '27': 864, '28': 383, '29': 641, '30': 953, '31': 568}

per-cluster profile (doublet = CD8 & CD4 both high; NK = high score_nk; gd = high score_gd):
              n  score_cd3  score_cd8  score_cd4  score_nk  score_gd  score_ilc  det_CD8B  det_CD4
leiden_dp                                                                                         
0          1009      0.893      0.810      0.552     0.098    -0.046      0.068     0.162    0.293
1           909      0.769      0.866      0.759    -0.050     0.015      0.153     0.084    0.067
2          1003      0.597      0.931      0.694     0.225    -0.007     -0.030     0.056    0.042
3          1337      1.159      1.159      0.577     0.474    -0.015     -0.141     0.060    0.031
4          1033      1.088      1.109      0.695     0.371    -0.009     -0.119     0.046    0.020
5           664      0.976      1.039      0.733     0.177    -0.028     -0.052     0.086    0.047
6           809      0.857      1.073      0.489     0.335    -0.027     -0.145     0.051    0.056
7          1468      0.892      0.965      0.724    -0.154    -0.054      0.215     0.110    0.036
8          1300      0.611      0.552      0.526     0.060     0.153      0.180     0.184    0.122
9           585      0.779      1.031      0.689     0.298    -0.005     -0.054     0.094    0.031
10         1162      0.543      0.906      0.636     0.239     0.007     -0.037     0.051    0.046
11         1268      0.367      0.890      0.702    -0.053    -0.018      0.124     0.167    0.028
12          725      0.490      0.758      0.638     0.224     0.119      0.119     0.159    0.039
13         1532      0.854      0.661      0.563     0.031     0.182      0.088     0.225    0.158
14          944      0.692      0.932      0.769    -0.009     0.078      0.078     0.086    0.038
15          383      0.596      0.867      0.549     1.053     0.146      0.005     0.039    0.107
16         1539      0.464      1.052      0.601     0.313     0.005     -0.143     0.060    0.019
17          406      0.754      0.726      0.638     0.312     0.166      0.126     0.167    0.099
18         1319      0.487      1.029      0.576     0.414    -0.003     -0.136     0.077    0.014
19          536      0.710      1.079      0.672     0.148    -0.028     -0.085     0.104    0.041
20          590      0.856      1.112      0.606     0.651     0.014     -0.144     0.034    0.022
21          490      0.905      0.824      0.568     0.537     0.098      0.061     0.194    0.102
22         1282      0.910      1.159      0.694     0.211    -0.035     -0.145     0.041    0.020
23          964      0.735      1.122      0.662     0.280    -0.022     -0.144     0.078    0.021
24          637      0.318      0.953      0.681     0.127     0.016      0.006     0.155    0.039
25          371      0.708      0.945      0.645     0.187    -0.025     -0.029     0.062    0.030
26          941      0.852      1.012      0.759    -0.140    -0.049      0.121     0.094    0.030
27          864      0.627      0.955      0.782    -0.097    -0.040      0.094     0.119    0.036
28          383      0.580      0.955      0.696     0.299     0.022     -0.093     0.078    0.023
29          641      0.409      0.678      0.575    -0.073     0.003      0.175     0.541    0.310
30          953      0.685      0.502      0.381     0.136     0.071      0.031     0.316    0.324
31          568      0.727      0.912      0.716    -0.095    -0.028      0.240     0.165    0.058

Li2024 labels per cluster:
leiden_dp       0    1    2     3    4    5  
... [3410 more chars truncated]
```

```python  # cell 21
# template (re-run if clusters changed): print({c: "" for c in dp.obs["leiden_dp"].cat.categories})
# label each DP cluster from the §6 profile/dotplot: doublet / NK / gamma_delta / CD4 / CD8 / lowq
cluster2lab_dp = {
    '0': 'CD8',
    '1': 'CD8',
    '2': 'CD8',
    '3': 'CD8',
    '4': 'CD8',
    '5': 'CD8',
    '6': 'CD4',
    '7': 'CD8',
    '8': 'CD4',
    '9': 'CD8',
    '10': 'CD8',
    '11': 'CD8',
    '12': 'CD8',
    '13': 'CD8',
    '14': 'CD8',
    '15': 'CD8',
    '16': 'CD8',
    '17': 'CD8',
    '18': 'CD8',
    '19': 'CD8',
    '20': 'CD8',
    '21': 'CD8',
    '22': 'CD8',
    '23': 'CD8',
    '24': 'CD8',
    '25': 'CD8',
    '26': 'CD8',
    '27': 'CD8',
    '28': 'CD8',
    '29': 'CD8',
    '30': 'CD4',
    '31': 'CD8',
   
   
   
    
}

if (not cluster2lab_dp or any(v == "" for v in cluster2lab_dp.values())
        or set(cluster2lab_dp) != set(dp.obs["leiden_dp"].cat.categories)):
    print("cluster2lab_dp unfilled / keys mismatch — fill from the §6 profile/dotplot.")
    print("template:", {c: "" for c in dp.obs["leiden_dp"].cat.categories})
else:
    dp.obs["cell_type_T2"] = dp.obs["leiden_dp"].map(cluster2lab_dp).astype("category")
    print(dp.obs["cell_type_T2"].value_counts().to_string())
```

```
cell_type_T2
CD8    25553
CD4     3062
```

## 7 — Double-negative quadrant: cluster + resolve identity

Neither CD8 nor CD4 programme on. Cluster on `X_mrvi_u`, then profile to separate **γδ T**
(high score_gd / TRDC), **NK / NKT** (high score_nk, low CD3), **ILC** (KIT/GATA3, no CD3),
**MAIT** (SLC4A10/KLRB1), and **low-quality dropout** (low score_cd3). The CD3⁺ cells here matter:
besides naive/memory and cytotoxic T states, **MF/Sézary malignant CD4 clones down-regulate CD4 and
CD7** and land in DN — flagged by MF markers (KIR3DL2, TOX, CCR4, PLS3, TWIST1) together with
aberrant loss of CD2/CD5/CD7. Map each cluster to a label. **HEAVY (GPU).**

```python  # cell 23
PROFILE = ["score_cd3", "score_cd8", "score_cd4", "score_nk", "score_gd", "score_ilc",
           "det_CD8B", "det_CD4"]

dn = mrvi_subcluster(adt, (adt.obs["lineage_cd48"] == "double_negative").to_numpy(),
                     key="leiden_dn", res=1.0, cache_csv=DN_CACHE_CSV, force=FORCE_QUAD)
print("clusters:", dn.obs["leiden_dn"].value_counts().sort_index().to_dict())

gp = dn.obs.groupby("leiden_dn", observed=True)
prof = gp[PROFILE].mean().round(3)
prof.insert(0, "n", gp.size())
print("\nper-cluster profile (gd = high score_gd; NK = high score_nk; low score_cd3 = low-quality):")
print(prof.to_string())
print("\nLi2024 labels per cluster:")
print(pd.crosstab(dn.obs["leiden_dn"], dn.obs["cell_type"]).T.to_string())

# lineage panel + markers to explain the CD3+ cells here: T states, MAIT, exhaustion, and
# MF/Sezary markers (malignant CD4 clones lose CD4/CD7 and fall into DN: KIR3DL2/TOX/CCR4/PLS3).
markers_dn = _present({**MARKERS_LINEAGE,
    "Naive/cm": ["CCR7", "SELL", "TCF7"],
    "Cytotoxic": ["GZMK", "GZMB", "PRF1"],
    "MAIT": ["SLC4A10", "KLRB1"],
    "Exhaustion": ["PDCD1", "LAG3", "TIGIT", "HAVCR2"],
    "MF/Sezary": ["KIR3DL2", "TOX", "CCR4", "PLS3", "TWIST1", "GATA3", "STAT5A"],
    "MF aberrant-loss": ["CD2", "CD5", "CD7"],
    "cycling": ["MKI67", "TOP2A"]})
sc.pl.dotplot(dn, markers_dn, groupby="leiden_dn", standard_scale="var")

dn_umap = ["leiden_dn", "score_cd3"] + [g for g in [*XUMAP, "CD3D", "CCR7", "GZMK",
           "KIR3DL2", "TOX", "CCR4", "CD7", "KLRB1"] if g in adt.var_names]
sc.pl.umap(dn, color=dn_umap, ncols=4, frameon=False, size=6, legend_loc="on data")
```

```
[leiden_dn] computed neighbors + Leiden (res=1.0) + UMAP on 140970 cells, cached -> skin_T_dn_quad.csv
clusters: {'0': 4882, '1': 5557, '2': 11158, '3': 10572, '4': 13055, '5': 3164, '6': 8597, '7': 11328, '8': 17588, '9': 10184, '10': 18144, '11': 3947, '12': 9252, '13': 2858, '14': 10684}

per-cluster profile (gd = high score_gd; NK = high score_nk; low score_cd3 = low-quality):
               n  score_cd3  score_cd8  score_cd4  score_nk  score_gd  score_ilc  det_CD8B  det_CD4
leiden_dn                                                                                          
0           4882      0.517     -0.276     -0.288     0.192    -0.042      0.086     0.000    0.011
1           5557      0.548     -0.284     -0.278     0.286     0.040     -0.038     0.001    0.035
2          11158      0.907     -0.331     -0.299     0.292     0.057     -0.025     0.005    0.022
3          10572      1.035     -0.330     -0.303     0.349     0.107      0.011     0.003    0.027
4          13055      0.413     -0.294     -0.334     0.920     0.137     -0.106     0.000    0.005
5           3164      0.253     -0.310     -0.328     0.238     0.055     -0.092     0.006    0.078
6           8597      0.762     -0.363     -0.313     0.303     0.081      0.027     0.002    0.048
7          11328      0.522     -0.347     -0.319     0.157     0.043      0.109     0.008    0.030
8          17588      0.185     -0.258     -0.285     0.168     0.051      0.013     0.000    0.008
9          10184      0.587     -0.388     -0.347     0.276     0.075     -0.075     0.005    0.057
10         18144      0.656     -0.377     -0.352     0.186     0.066     -0.022     0.010    0.070
11          3947      0.187     -0.310     -0.313     0.223     0.072      0.004     0.004    0.033
12          9252      0.782     -0.311     -0.298     0.085     0.052      0.088     0.001    0.027
13          2858      0.668     -0.268     -0.292    -0.087    -0.021     -0.056     0.003    0.016
14         10684      0.851     -0.289     -0.305    -0.096    -0.056      0.027     0.000    0.037

Li2024 labels per cluster:
leiden_dn        0     1     2     3     4     5     6     7      8     9     10    11    12    13    14
cell_type                                                                                               
B_cell           1     1     0     0     4     1     0     0      5     0      0     0     0     5     5
DC1              2    41     4     0     0    67     0     0      2     8     15     0     0     0     3
DC2              1     0     0     0     0     1     0     0      0     0      0     0     0     0     0
F2               0     0     0     0     0     0     0     0      1     0      0     0     0     0     0
ILC1_3          10    21     7     9   208     5     6    25     23     5      1     0    15     0     0
ILC1_NK          0     0     0     0   374     0     0     0      1     0      0     0     1     0     0
ILC2             0     0     0     4     0     0     0     1      2     0      0     0     1     0     2
Inf_mac          0     0     0     0     1     0     0     0      0     0      0     0     0     0     0
LC_2             2     0     0     0     0     0     0     0      0     0      0     0     0     0     0
Mast_cell        0     0     0     0     5     0     0     0      0     0      0     0     0     0     0
Mono_mac         0    39   256    78     2    26     5     4     12    81    109     8     8     0     7
NK              27     1     0     0  1204     6     0     0     11     0      0     1     0     0     0
Plasma           0     2     0     0    22    10     0     0     13     0      0     1     0     0    13
Tc             183    33    44    39  4360   170    15    49    135     4     21     5    37    34    96
Tc17_Th17     1351     6    78    22    12   244    14    11     16    50     72     6     3     0    26
Tc_IL13_IL22    29    34    61     8    12   181    53    46      9    67    133     6    16     0     
... [706 more chars truncated]
```

```python  # cell 24
# template (re-run if clusters changed): print({c: "" for c in dn.obs["leiden_dn"].cat.categories})
# label each DN cluster from the §7 profile/dotplot: gamma_delta / NK / ILC / lowq / CD4 / CD8
cluster2lab_dn = {
    '0': 'CD4',
    '1': 'CD4',
    '2': 'CD4',
    '3': 'CD4',
    '4': 'NK',
    '5': 'NK',
    '6': 'CD4',
    '7': 'CD4',
    '8': 'NK',
    '9': 'CD4',
    '10': 'CD4',
    '11': 'NK',
    '12': 'CD4',
    '13': 'CD4',
    '14': 'CD4',
}

if (not cluster2lab_dn or any(v == "" for v in cluster2lab_dn.values())
        or set(cluster2lab_dn) != set(dn.obs["leiden_dn"].cat.categories)):
    print("cluster2lab_dn unfilled / keys mismatch — fill from the §7 profile/dotplot.")
    print("template:", {c: "" for c in dn.obs["leiden_dn"].cat.categories})
else:
    dn.obs["cell_type_T2"] = dn.obs["leiden_dn"].map(cluster2lab_dn).astype("category")
    print(dn.obs["cell_type_T2"].value_counts().to_string())
```

```
cell_type_T2
CD4    103216
NK      37754
```

```python  # cell 25
sc.pl.umap(dn, color=['cell_type_T2'], ncols=4, frameon=False, size=6, legend_loc="on data")
```

```
<Figure size 640x480 with 1 Axes>
```

## 8 — Assemble `cell_type_T2` + compare to the old annotation

Merge the per-quadrant labels back onto the full object: CD4/CD8 subtypes from §4–§5, DP/DN
resolved labels from §6–§7 (unfilled buckets fall back to the `lineage_cd48` name). Also store the
quadrant-local cluster id in `leiden_quad`. Cross-tab against the old `cell_type_T` and the Li2024
`cell_type` purity table.

```python  # cell 27
# subtype: default to the quadrant name, overwrite with per-quadrant labels where they were filled
adt.obs["cell_type_T2"] = adt.obs["lineage_cd48"].astype(str)
adt.obs["leiden_quad"] = ""
for sub, key in [(cd4, "leiden_cd4"), (cd8, "leiden_cd8"), (dp, "leiden_dp"), (dn, "leiden_dn")]:
    tag = key.split("_")[1]
    adt.obs.loc[sub.obs_names, "leiden_quad"] = (tag + "_" + sub.obs[key].astype(str)).to_numpy()
    if "cell_type_T2" in sub.obs:
        adt.obs.loc[sub.obs_names, "cell_type_T2"] = sub.obs["cell_type_T2"].astype(str).to_numpy()
adt.obs["cell_type_T2"] = adt.obs["cell_type_T2"].astype("category")
adt.obs["leiden_quad"] = adt.obs["leiden_quad"].astype("category")
print(adt.obs["cell_type_T2"].value_counts(dropna=False).to_string())

print("\ncell_type_T2 (new, rows) x cell_type_T (old, cols):")
print(pd.crosstab(adt.obs["cell_type_T2"], adt.obs["cell_type_T"]).to_string())

known = adt.obs[adt.obs["cell_type"] != "Unknown"]
cmp = pd.crosstab(known["cell_type_T2"], known["cell_type"])
frac = cmp.div(cmp.sum(axis=1).clip(lower=1), axis=0)
summary = pd.DataFrame({"top_li2024": frac.idxmax(axis=1),
                        "purity": frac.max(axis=1).round(2),
                        "n_known": cmp.sum(axis=1)})
with pd.option_context("display.max_columns", None, "display.width", 200):
    print("\ndominant Li2024 label per cell_type_T2 (purity):")
    print(summary.to_string())
```

```
cell_type_T2
CD4         280011
CD8          59297
NK           40665
CD4_Treg     21593

cell_type_T2 (new, rows) x cell_type_T (old, cols):
cell_type_T   CD4_Cm  CD4_Th2  CD4_Treg  CD4_cytotoxic  CD4_mem  CD8_effector  CD8_em  UNK
cell_type_T2                                                                              
CD4            51229   175666     11009            245    31637          5617    4516   92
CD4_Treg         682      569     18722              0     1534            42      44    0
CD8             8964    13777      1399              5     3368         15871   15844   69
NK               832    10492       143              0    13578          8490    6406  724

dominant Li2024 label per cell_type_T2 (purity):
              top_li2024  purity  n_known
cell_type_T2                             
CD4           tumor_cell    0.55   164996
CD4_Treg            Treg    0.87    16322
CD8                   Tc    0.68    36326
NK                    Tc    0.35    14721
```

## 9 — Persist

Full T-cell object with the new lineage/subtype columns → `skin_T_annotated_v2.h5ad`. The original
`skin_T_annotated.h5ad` is left untouched.

```python  # cell 29
adt.write_h5ad(T_ANNOT_V2)
print("wrote ->", T_ANNOT_V2)
print(adt)
```

```
wrote -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_annotated_v2.h5ad
AnnData object with n_obs × n_vars = 401566 × 49683
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagreement', 'leiden_broad', 'cell_type_broad', 'leiden_T', 'cell_type_T', 'score_cd3', 'score_cd8', 'score_cd4', 'score_nk', 'score_gd', 'score_ilc', 'det_CD8A', 'det_CD8B', 'det_CD4', 'det_CD40LG', 'lineage_cd48', 'cell_type_T2', 'leiden_quad'
    uns: 'cell_type_T_colors', 'cell_type_broad_colors', 'cell_type_colors', 'dendrogram_leiden_T', 'dendrogram_leiden_broad', 'disease_colors', 'leiden_T', 'leiden_T_colors', 'leiden_broad', 'leiden_broad_colors', 'log1p', 'neighbors', 'pca', 'sample_id_colors', 'study_colors', 'umap'
    obsm: 'X_mrvi_u', 'X_mrvi_z', 'X_pca', 'X_scVI', 'X_scVI_MDE', 'X_umap', '_scvi_extra_categorical_covs'
    varm: 'PCs'
    layers: 'counts', 'raw_counts'
    obsp: 'connectivities', 'distances'
```

```python  # cell 30
sc.pl.umap(adt, color=["cell_type_T2"],
           ncols=4, frameon=False, size=4, legend_loc="on data")
```

```
<Figure size 640x480 with 1 Axes>
```

## Downstream

nbs 14–21 currently derive CD4/CD8 from `cell_type_T.str.split("_")[0]`. To adopt this corrected
split, point them at `skin_T_annotated_v2.h5ad` and use `lineage_cd48` for the lineage gate
(`CD4` / `CD8` / `double_positive` / `double_negative`) and `cell_type_T2` for the resolved
subtype/identity. nb15's CD8 inferCNV reference becomes `lineage_cd48 == "CD8"`, excluding the
double-positive / double-negative buckets from both query and reference.
