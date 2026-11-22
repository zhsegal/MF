# old/13_skin_lowres_annotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/13_skin_lowres_annotation.ipynb` and is never run._

# Skin atlas — low-resolution broad annotation

Loads the **skin-only** cells from the joint atlas (`compartment == "Skin"`, ~866k cells, 6 studies)
with their cached **MRVI** latents (`joint_X_mrvi_u_skin.npy` = sample-unaware integration view).
Computes a UMAP, **low-resolution Leiden** (broad lineages), and a marker **dot plot** — using the
Li2024 `cell_type` labels as the naming scaffold (the other skin cohorts are `Unknown`).

**HEAVY cells run on the GPU kernel (`neural_nmf_env`).** UMAP coords + Leiden labels are cached so
re-runs reload instead of recomputing.

```python  # cell 1
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


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

ANNOT = OUT / "joint_annotated.h5ad"
MRVI_U = OUT / "joint_X_mrvi_u_skin.npy"   # sample-unaware (integration view), 10-d
MRVI_Z = OUT / "joint_X_mrvi_skin.npy"     # sample-aware, 30-d
UMAP_NPY = OUT / "joint_umap_skin_u.npy"
LEIDEN_CSV = OUT / "skin_leiden_broad.csv"
print("NB_DIR:", NB_DIR)
```

```
NB_DIR: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

## 1 — Load the skin subset + MRVI latents

`compartment == "Skin"` row order matches the cached `_skin` latents (the MRVI job was run on this
exact slice, see nb 10 step 7b). **HEAVY** (reads the 1.34M-cell atlas) — GPU kernel.

```python  # cell 3
adata = sc.read_h5ad(ANNOT)
ad = adata[adata.obs["compartment"] == "Skin"].copy()
del adata

ad.obsm["X_mrvi_u"] = np.load(MRVI_U)   # integration view (drives neighbors/UMAP/Leiden)
ad.obsm["X_mrvi_z"] = np.load(MRVI_Z)   # sample-aware (kept for later substructure work)
assert ad.n_obs == ad.obsm["X_mrvi_u"].shape[0], "skin slice / latent row mismatch"
print(ad)
print("X_mrvi_u", ad.obsm["X_mrvi_u"].shape, "| X_mrvi_z", ad.obsm["X_mrvi_z"].shape)
```

```
AnnData object with n_obs × n_vars = 865826 × 49683
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagreement'
    obsm: 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs', 'X_mrvi_u', 'X_mrvi_z'
    layers: 'raw_counts'
X_mrvi_u (865826, 10) | X_mrvi_z (865826, 30)
```

## 2 — Existing labels (naming scaffold)

Which lineages already carry Li2024 `cell_type` names vs `Unknown` (the rows we annotate here).

```python  # cell 5
print("--- cell_type (skin) ---")
print(ad.obs["cell_type"].value_counts().head(25).to_string())
print("\n--- cell_type x study (Unknown = unlabelled cohorts) ---")
print(pd.crosstab(ad.obs["cell_type"], ad.obs["study"]).to_string())
```

```
--- cell_type (skin) ---
cell_type
Unknown               446247
tumor_cell            102367
Th                     52824
Tc                     39889
Treg                   27614
Differentiated_KC      24682
F2                     19903
B_cell                 15211
VE2                    12528
Differentiated_KC*     10985
DC2                     9807
MigDC                   7062
F3                      6719
Mono_mac                6384
Melanocyte              5640
Plasma                  5569
moDC_3                  5557
DC1                     5240
Pericyte_2              4760
Pericyte_1              4636
Tc17_Th17               4233
Macro_1                 3993
VE3                     3544
immune                  3466
Macro_2                 3058

--- cell_type x study (Unknown = unlabelled cohorts) ---
study                brunner2024  chennareddy2025  gaydosik2019  herrera2021  li2024  rindler2021_fi
cell_type                                                                                           
B_cell                         0                0             0            0   15211               0
Basal                          0                0             0            0     358               0
DC1                            0                0             0            0    5240               0
DC2                            0                0             0            0    9807               0
Differentiated_KC              0                0             0            0   24682               0
Differentiated_KC*             0                0             0            0   10985               0
F1                             0                0             0            0    2188               0
F2                             0                0             0            0   19903               0
F3                             0                0             0            0    6719               0
ILC1_3                         0                0             0            0    1275               0
ILC1_NK                        0                0             0            0    1090               0
ILC2                           0                0             0            0     133               0
Inf_mac                        0                0             0            0    1966               0
LC_1                           0                0             0            0     470               0
LC_2                           0                0             0            0     714               0
LC_3                           0                0             0            0     637               0
LC_4                           0                0             0            0    1419               0
LE1                            0                0             0            0    1939               0
LE2                            0                0             0            0     982               0
Macro_1                        0                0             0            0    3993               0
Macro_2                        0                0             0            0    3058               0
Mast_cell                      0                0             0            0     663               0
Melanocyte                     0                0             0            0    5640               0
MigDC                          0                0             0            0    7062               0
Mono_mac                       0                0             0            0    6384               0
NK                             0                0             0            0    2398               0
Pericyte_1                     0                0             0            0    4636               0
Pericyte_2                     0                0             0            0    4760               0
Plasma                         0                0             0            0    5569               0
Proliferating_KC               0                0  
... [2069 more chars truncated]
```

## 3 — Log-normalize for marker viz

Atlas `X` is raw counts. Keep counts in a layer; log-normalize `X` for the dot plot / marker overlays.

```python  # cell 7
ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
print("X is log1p-normalized:", float(ad.X.max()) < 20)
```

```
X is log1p-normalized: True
```

## 4 — Neighbors + UMAP on `X_mrvi_u` (HEAVY, cached)

Coords cached to `joint_umap_skin_u.npy`. The neighbor graph is still rebuilt on reload (needed for
Leiden) but UMAP layout — the slow part — is skipped.

```python  # cell 9
np.random.seed(SEED)
sc.pp.neighbors(ad, use_rep="X_mrvi_u", random_state=SEED)   # graph needed for both UMAP + Leiden

if UMAP_NPY.exists():
    ad.obsm["X_umap"] = np.load(UMAP_NPY)
    print("loaded cached UMAP", ad.obsm["X_umap"].shape)
else:
    sc.tl.umap(ad, random_state=SEED)
    np.save(UMAP_NPY, ad.obsm["X_umap"])
    print("computed + cached UMAP ->", UMAP_NPY)
```

```
computed + cached UMAP -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/joint_umap_skin_u.npy
```

## 5 — Low-resolution Leiden (broad lineages, cached)

`resolution=0.2` → a handful of clusters for the major compartments. Labels cached to
`skin_leiden_broad.csv` (`cell_id → leiden_broad`).

```python  # cell 11
LEIDEN_RES = 0.7
FORCE_LEIDEN = True   # True -> recompute (+ overwrite cache) even if skin_leiden_broad.csv exists

if LEIDEN_CSV.exists() and not FORCE_LEIDEN:
    lab = pd.read_csv(LEIDEN_CSV, index_col="cell_id")["leiden_broad"].astype(str)
    ad.obs["leiden_broad"] = lab.reindex(ad.obs["cell_id"].astype(str)).to_numpy()
    ad.obs["leiden_broad"] = ad.obs["leiden_broad"].astype("category")
    print("loaded cached Leiden labels (set FORCE_LEIDEN=True to recompute)")
else:
    sc.tl.leiden(ad, resolution=LEIDEN_RES, random_state=SEED,
                 key_added="leiden_broad", flavor="igraph", n_iterations=2, directed=False)
    pd.DataFrame({"cell_id": ad.obs["cell_id"].astype(str).to_numpy(),
                  "leiden_broad": ad.obs["leiden_broad"].astype(str).to_numpy()}) \
        .to_csv(LEIDEN_CSV, index=False)
    print(f"computed Leiden (res={LEIDEN_RES}) + cached ->", LEIDEN_CSV)

print("clusters:", ad.obs["leiden_broad"].value_counts().sort_index().to_dict())
```

```
computed Leiden (res=0.7) + cached -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_leiden_broad.csv
clusters: {'0': 88244, '1': 38073, '2': 7049, '3': 37863, '4': 56856, '5': 52491, '6': 16630, '7': 81245, '8': 137, '9': 8462, '10': 29262, '11': 91574, '12': 154753, '13': 24682, '14': 21528, '15': 9378, '16': 67, '17': 50476, '18': 67340, '19': 9683, '20': 19999, '21': 34}
```

## 6 — UMAP overview

```python  # cell 13
sc.pl.umap(
    ad,
    color=["leiden_broad", "cell_type", "study", "disease"],
    ncols=2, frameon=False, size=2, wspace=0.25,
)
```

```
<Figure size 1600x960 with 4 Axes>
```

## 7 — Marker dot plot (broad lineages)

Each Leiden cluster should light up one marker block. `standard_scale="var"` scales each gene 0–1
across clusters. Genes absent from the 49,683-gene outer-join are dropped first.

```python  # cell 15
markers = {
    "T": ["CD3D", "CD3E", "TRAC"],
    "CD4": ["CD4", "IL7R"],
    "CD8/NK": ["CD8A", "GZMB", "NKG7", "GNLY"],
    "Treg": ["FOXP3", "CTLA4"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "Plasma": ["MZB1", "JCHAIN", "IGHG1"],
    "Myeloid/DC": ["LYZ", "CD68", "ITGAX", "CD14"],
    "Mast": ["TPSAB1", "CPA3", "KIT"],
    "Keratinocyte": ["KRT14", "KRT5", "KRT1"],
    "Fibroblast": ["COL1A1", "DCN", "LUM"],
    "Vascular endo": ["PECAM1", "VWF", "CLDN5"],
    "Lymphatic endo": ["LYVE1", "PROX1"],
    "Melanocyte": ["MLANA", "PMEL", "TYR"],
}
genes_present = set(ad.var_names)
missing = sorted({g for v in markers.values() for g in v} - genes_present)
if missing:
    print("genes absent from var_names (dropped):", missing)
markers = {k: [g for g in v if g in genes_present] for k, v in markers.items()}
markers = {k: v for k, v in markers.items() if v}   # drop empty groups

sc.pl.dotplot(ad, markers, groupby="leiden_broad", standard_scale="var", dendrogram=True)
```

```
WARNING: dendrogram data not found (using key=dendrogram_leiden_broad). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/scanpy/tools/_dendrogram.py:138: UserWarning: You’re trying to run this on 49683 dimensions of `.X`, if you really want this, set `use_rep='X'`.
         Falling back to preprocessing with `sc.pp.pca` and default params.
  _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T, CD4, CD8/NK, etc.

<Figure size 1636x870 with 6 Axes>
```

## 8 — Map clusters → cell type

Fill `cluster2ct` from the dot plot (§7) + the Li2024 labels (§2). Run the helper print first if
clusters changed (new `LEIDEN_RES`) to get a fresh key template.

```python  # cell 17
# template (re-run if clusters changed): print({c: "" for c in ad.obs["leiden_broad"].cat.categories})

cluster2ct = {
    "0": "Ketratinocyte",
    "1": "Ketratinocyte",
    "2": "Ketratinocyte",
    "3": "Ketratinocyte",
    "4": "T",
    "5": "T",
    "6": "T",
    "7": "Myeloid",
    "8": "Myeloid",
    "9": "Ketratinocyte",
    "10": "T",
    "11": "T",
    "12": "T",
    "13": "B",
    "14": "Myeloid",
    "15": "Plasma",
    "16": "Plasma",
    "17": "Vascular",
    "18": "Fibroblast",
    "19": "Melanocyte",
    "20": "Fibroblast",
    "21": "Plasma",
    
}
assert set(cluster2ct) == set(ad.obs["leiden_broad"].cat.categories), \
    "keys must match leiden_broad clusters: " + str(sorted(ad.obs["leiden_broad"].cat.categories))

ad.obs["cell_type_broad"] = ad.obs["leiden_broad"].map(cluster2ct).astype("category")
print(ad.obs["cell_type_broad"].value_counts(dropna=False).to_string())
```

```
cell_type_broad
T                401566
Ketratinocyte    179691
Myeloid          102910
Fibroblast        87339
Vascular          50476
B                 24682
Melanocyte         9683
Plasma             9479
```

## 9 — Compare to the previous (Li2024) annotation

Where Li2024 `cell_type` exists (`!= Unknown`), cross-tab it against the new `cell_type_broad` to
check each broad label captures a coherent set of known types. Skipped if `cluster2ct` is unfilled.

```python  # cell 19
if "cell_type_broad" not in ad.obs or ad.obs["cell_type_broad"].astype(str).str.len().eq(0).all():
    print("cell_type_broad unfilled — complete cluster2ct in §8 first.")
else:
    known = ad.obs[ad.obs["cell_type"] != "Unknown"].copy()
    print(f"Li2024-labelled skin cells: {len(known):,} / {ad.n_obs:,}")

    cmp = pd.crosstab(known["cell_type_broad"], known["cell_type"])
    cmp = cmp.loc[:, cmp.sum() > 0]
    frac = cmp.div(cmp.sum(axis=1).clip(lower=1), axis=0).round(2)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nrow-normalized cell_type_broad x Li2024 cell_type:")
        print(frac.to_string())
        print("\ndominant Li2024 label per broad annotation (purity):")
        summary = pd.DataFrame({"top_li2024": frac.idxmax(axis=1),
                                "purity": frac.max(axis=1),
                                "n_known": cmp.sum(axis=1)})
        print(summary.to_string())
```

```
Li2024-labelled skin cells: 419,579 / 865,826

row-normalized cell_type_broad x Li2024 cell_type:
cell_type        B_cell  Basal   DC1  DC2  Differentiated_KC  Differentiated_KC*    F1    F2    F3  ILC1_3  ILC1_NK  ILC2  Inf_mac  LC_1  LC_2  LC_3  LC_4   LE1   LE2  Macro_1  Macro_2  Mast_cell  Melanocyte  MigDC  Mono_mac    NK  Pericyte_1  Pericyte_2  Plasma  Proliferating_KC  Schwann_1  Sebaceous    Tc  Tc17_Th17  Tc_IL13_IL22    Th  Treg  Undifferentiated_KC   VE1   VE2   VE3  basal2  channel  immune  moDC_1  moDC_2  moDC_3   pDC  tumor_cell
cell_type_broad                                                                                                                                                                                                                                                                                                                                                                                                                                                    
B                  0.84   0.00  0.00  0.0               0.00                0.00  0.00  0.00  0.00    0.00      0.0   0.0     0.00  0.00  0.00  0.00  0.00  0.00  0.00     0.00     0.00       0.00        0.00   0.00      0.00  0.00        0.00        0.00     0.0              0.00       0.00       0.00  0.00       0.00          0.00  0.00  0.01                 0.00  0.00  0.00  0.00    0.00     0.00    0.00    0.00    0.00    0.00  0.14        0.00
Fibroblast         0.00   0.00  0.00  0.0               0.00                0.00  0.06  0.51  0.17    0.00      0.0   0.0     0.00  0.00  0.00  0.00  0.00  0.00  0.00     0.00     0.00       0.00        0.00   0.00      0.00  0.00        0.12        0.13     0.0              0.00       0.01       0.00  0.00       0.00          0.00  0.00  0.00                 0.00  0.00  0.00  0.00    0.00     0.00    0.00    0.00    0.00    0.00  0.00        0.00
Ketratinocyte      0.00   0.01  0.01  0.0               0.52                0.23  0.00  0.01  0.00    0.00      0.0   0.0     0.00  0.00  0.00  0.00  0.00  0.00  0.00     0.00     0.00       0.01        0.00   0.00      0.00  0.00        0.00        0.00     0.0              0.04       0.00       0.01  0.01       0.00          0.00  0.00  0.01                 0.05  0.00  0.00  0.00    0.01     0.01    0.02    0.00    0.00    0.00  0.00        0.04
Melanocyte         0.00   0.00  0.00  0.0               0.00                0.00  0.00  0.00  0.00    0.00      0.0   0.0     0.00  0.00  0.00  0.00  0.00  0.00  0.00     0.00     0.00       0.00        0.95   0.00      0.00  0.00        0.00        0.00     0.0              0.00       0.03       0.00  0.01       0.00          0.00  0.00  0.00                 0.00  0.00  0.00  0.00    0.00     0.00    0.00    0.00    0.00    0.00  0.00        0.01
Myeloid            0.00   0.00  0.08  0.2               0.00                0.00  0.00  0.00  0.00    0.00      0.0   0.0     0.04  0.01  0.01  0.01  0.03  0.00  0.00     0.08     0.06       0.00        0.00   0.14      0.11  0.00        0.00        0.00     0.0              0.00       0.00       0.00  0.00       0.00          0.00  0.00  0.00                 0.00  0.00  0.00  0.00    0.00     0.00    0.03    0.03    0.05    0.11  0.00        0.00
Plasma             0.16   0.00  0.11  0.0               0.00                0.00  0.00  0.00  0.00    0.00      0.0   0.0     0.00  0.00  0.00  0.00  0.00  0.00  0.00     0.00     0.00       0.00        0.00   0.00      0.00  0.00        0.00        0.00     0.7              0.00       0.00       0.00  0.00       0.00          0.00  0.00  0.00                 0.00  0.00  0.00  0.00    0.00     0.00    0.00    0.00    0.00    0.00  0.00        0.02
T                  0.00   0.00  0.00  0.0               0.00                0.00  0.00  0.00  0.00    0.01      0.0   0.0     0.00  0.00  0.00  0.00  0.00  0.00  0.00     0.00     0.00       0.00        0.00   0.00      0.00  0.01        0.00        0.00     0.0              0.00      
... [1191 more chars truncated]
```

```python  # cell 20
sc.pl.umap(
    ad,
    color=["leiden_broad", "cell_type_broad", "study", "disease"],
    ncols=2, frameon=False, size=2, wspace=0.25,
)
```

```
<Figure size 1600x960 with 4 Axes>
```

# T-cell sub-clustering

Subset to the broad **T** compartment (`cell_type_broad == "T"`, ~402k cells), recompute
neighbors/UMAP/Leiden on `X_mrvi_u` for this slice only, dot-plot T/NK markers, annotate subtypes,
and cache the complete T-cell `AnnData`.

## 10 — Subset to T cells

```python  # cell 22
# T-cell sub-cluster cache paths
T_UMAP_NPY = OUT / "skin_T_umap_u.npy"
T_LEIDEN_CSV = OUT / "skin_T_leiden.csv"
T_ANNOT = OUT / "skin_T_annotated.h5ad"

adt = ad[ad.obs["cell_type_broad"] == "T"].copy()   # carries X (log1p), counts layer, X_mrvi_u/z
print(adt)
print("T cells:", adt.n_obs, "| X_mrvi_u", adt.obsm["X_mrvi_u"].shape)
```

```
AnnData object with n_obs × n_vars = 401566 × 49683
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagreement', 'leiden_broad', 'cell_type_broad'
    uns: 'log1p', 'neighbors', 'umap', 'leiden_broad', 'leiden_broad_colors', 'cell_type_colors', 'study_colors', 'disease_colors', 'pca', 'dendrogram_leiden_broad', 'cell_type_broad_colors'
    obsm: 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs', 'X_mrvi_u', 'X_mrvi_z', 'X_umap', 'X_pca'
    varm: 'PCs'
    layers: 'raw_counts', 'counts'
    obsp: 'distances', 'connectivities'
T cells: 401566 | X_mrvi_u (401566, 10)
```

## 11 — Neighbors + UMAP on the T subset (HEAVY, cached)

Re-run neighbors/UMAP on `X_mrvi_u` for T cells only so the embedding resolves T substructure
instead of the global-lineage layout. Coords cached to `skin_T_umap_u.npy`.

```python  # cell 24
np.random.seed(SEED)
sc.pp.neighbors(adt, use_rep="X_mrvi_u", random_state=SEED)   # rebuild graph on T-only subset

if T_UMAP_NPY.exists():
    adt.obsm["X_umap"] = np.load(T_UMAP_NPY)
    print("loaded cached T UMAP", adt.obsm["X_umap"].shape)
else:
    sc.tl.umap(adt, random_state=SEED)
    np.save(T_UMAP_NPY, adt.obsm["X_umap"])
    print("computed + cached T UMAP ->", T_UMAP_NPY)
```

```
computed + cached T UMAP -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_umap_u.npy
```

## 12 — Leiden on T cells (cached)

`resolution=1.0` → finer T/NK subtypes. Labels cached to `skin_T_leiden.csv` (`cell_id → leiden_T`).

```python  # cell 26
T_LEIDEN_RES = 1.0
FORCE_T_LEIDEN = True

if T_LEIDEN_CSV.exists() and not FORCE_T_LEIDEN:
    lab = pd.read_csv(T_LEIDEN_CSV, index_col="cell_id")["leiden_T"].astype(str)
    adt.obs["leiden_T"] = lab.reindex(adt.obs["cell_id"].astype(str)).to_numpy()
    adt.obs["leiden_T"] = adt.obs["leiden_T"].astype("category")
    print("loaded cached T Leiden labels (set FORCE_T_LEIDEN=True to recompute)")
else:
    sc.tl.leiden(adt, resolution=T_LEIDEN_RES, random_state=SEED,
                 key_added="leiden_T", flavor="igraph", n_iterations=2, directed=False)
    pd.DataFrame({"cell_id": adt.obs["cell_id"].astype(str).to_numpy(),
                  "leiden_T": adt.obs["leiden_T"].astype(str).to_numpy()}) \
        .to_csv(T_LEIDEN_CSV, index=False)
    print(f"computed T Leiden (res={T_LEIDEN_RES}) + cached ->", T_LEIDEN_CSV)

print("clusters:", adt.obs["leiden_T"].value_counts().sort_index().to_dict())
```

```
computed T Leiden (res=1.0) + cached -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_leiden.csv
clusters: {'0': 17041, '1': 33076, '2': 52068, '3': 4099, '4': 34410, '5': 59141, '6': 17333, '7': 33453, '8': 26810, '9': 28051, '10': 31273, '11': 46905, '12': 14802, '13': 885, '14': 1969, '15': 250}
```

## 13 — T-cell marker dot plot

Granular T/NK markers per `leiden_T` cluster. `standard_scale="var"` scales each gene 0–1 across
clusters; genes absent from the outer-join are dropped first.

```python  # cell 28
markers_T = {
    "T core": ["CD3D", "CD3E", "TRAC"],
    "CD4": ["CD4", "IL7R"],
    "CD8": ["CD8A", "CD8B"],
    "Naive/mem": ["CCR7", "SELL", "TCF7"],
    "Cytotoxic": ["GZMK", "GZMB", "PRF1", "NKG7", "GNLY"],
    "Treg": ["FOXP3", "CTLA4", "IL2RA"],
    "Th17/Tc17": ["RORC", "IL17A", "CCR6"],
    "Exhaustion": ["PDCD1", "TOX", "LAG3", "TIGIT", "HAVCR2"],
    "NK": ["NCAM1", "KLRD1", "KLRF1"],
    "ILC": ["KIT", "GATA3"],
    "Prolif": ["MKI67", "TOP2A"],
}
genes_present = set(adt.var_names)
missing = sorted({g for v in markers_T.values() for g in v} - genes_present)
if missing:
    print("genes absent from var_names (dropped):", missing)
markers_T = {k: [g for g in v if g in genes_present] for k, v in markers_T.items()}
markers_T = {k: v for k, v in markers_T.items() if v}

sc.pl.dotplot(adt, markers_T, groupby="leiden_T", standard_scale="var", dendrogram=True)
```

```
WARNING: dendrogram data not found (using key=dendrogram_leiden_T). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: T core, CD4, CD8, etc.

<Figure size 1451x660 with 6 Axes>
```

## 13.b — Map T clusters → subtype

Fill `cluster2ct_T` from the §13 dot plot + Li2024 `cell_type` (CD4/Th, CD8/Tc, Treg, NK/ILC,
cytotoxic, proliferating, malignant `tumor_cell`). Run with empty values first to print the template.

```python  # cell 30
# template (re-run if clusters changed): print({c: "" for c in adt.obs["leiden_T"].cat.categories})

cluster2ct_T = {
    "0": "CD4_mem",
    "1": "CD4_mem",
    "2": "CD4_Th2",
    "3": "CD4_Th2",
    "4": "CD4_Th2",
    "5": "CD4_Th2",
    "6": "CD4_Th2",
    "7": "CD4_Th2",
    "8": "CD8_em",
    "9": "CD8_effector",
    "10": "CD4_Treg",
    "11": "CD4_Cm",
    "12": "CD4_Cm",
    "13": "UNK",
    "14": "CD8_effector",
    "15": "CD4_cytotoxic",
 
}
assert set(cluster2ct_T) == set(adt.obs["leiden_T"].cat.categories), \
    "keys must match leiden_T clusters: " + str(sorted(adt.obs["leiden_T"].cat.categories))

if any(v == "" for v in cluster2ct_T.values()):
    print("cluster2ct_T unfilled — read the §13 dot plot, then map each leiden_T cluster.")
    print("clusters:", adt.obs["leiden_T"].value_counts().sort_index().to_dict())
else:
    adt.obs["cell_type_T"] = adt.obs["leiden_T"].map(cluster2ct_T).astype("category")
    print(adt.obs["cell_type_T"].value_counts(dropna=False).to_string())

    # sanity vs Li2024 labels
    known = adt.obs[adt.obs["cell_type"] != "Unknown"].copy()
    cmp = pd.crosstab(known["cell_type_T"], known["cell_type"])
    frac = cmp.div(cmp.sum(axis=1).clip(lower=1), axis=0).round(2)
    summary = pd.DataFrame({"top_li2024": frac.idxmax(axis=1),
                            "purity": frac.max(axis=1),
                            "n_known": cmp.sum(axis=1)})
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\ndominant Li2024 label per cell_type_T (purity):")
        print(summary.to_string())
```

```
cell_type_T
CD4_Th2          200504
CD4_Cm            61707
CD4_mem           50117
CD4_Treg          31273
CD8_effector      30020
CD8_em            26810
UNK                 885
CD4_cytotoxic       250

dominant Li2024 label per cell_type_T (purity):
               top_li2024  purity  n_known
cell_type_T                               
CD4_Cm                 Th    0.74    39537
CD4_Th2        tumor_cell    0.76   107502
CD4_Treg             Treg    0.87    21191
CD4_cytotoxic  tumor_cell    0.96      238
CD4_mem        tumor_cell    0.45    24900
CD8_effector           Tc    0.79    20418
CD8_em                 Tc    0.82    18007
UNK                ILC1_3    0.60      572
```

## 13.c — T-cell UMAP overview

T-only UMAP coloured by `leiden_T`, the new `cell_type_T`, `study` and `sample_id` — check the
subtypes are coherent and not driven by a single cohort/sample.

```python  # cell 32
colors = ["leiden_T"]
if "cell_type_T" in adt.obs:
    colors.append("cell_type_T")
colors += ["study", "sample_id"]

sc.pl.umap(
    adt,
    color=colors,
    ncols=2, frameon=False, size=2, wspace=0.25,
    legend_loc="on data" if "cell_type_T" in adt.obs else "right margin",
)
```

```
<Figure size 1600x960 with 4 Axes>
```

```python  # cell 33
sc.pl.umap(
    adt,
    color=["GATA3","CCR4","CXCL13","TCF7","SELL","GZMB","GNLY","KIR3DL2"],
    ncols=2, frameon=False, size=2, wspace=0.25,
    legend_loc="on data" if "cell_type_T" in adt.obs else "right margin",
)
```

```
<Figure size 1600x1920 with 16 Axes>
```

## 14 — Cache the complete T-cell adata

Full T-cell `AnnData` (log1p `X`, `counts` layer, MRVI latents, T-UMAP, `leiden_T`, `cell_type_T`)
written to `skin_T_annotated.h5ad` for downstream substructure work.

```python  # cell 35
adt.write_h5ad(T_ANNOT)
print("wrote complete T-cell adata ->", T_ANNOT)
print(adt)
```

```
wrote complete T-cell adata -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_T_annotated.h5ad
AnnData object with n_obs × n_vars = 401566 × 49683
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagreement', 'leiden_broad', 'cell_type_broad', 'leiden_T', 'cell_type_T'
    uns: 'log1p', 'neighbors', 'umap', 'leiden_broad', 'leiden_broad_colors', 'cell_type_colors', 'study_colors', 'disease_colors', 'pca', 'dendrogram_leiden_broad', 'cell_type_broad_colors', 'leiden_T', 'dendrogram_leiden_T', 'leiden_T_colors', 'cell_type_T_colors', 'sample_id_colors'
    obsm: 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs', 'X_mrvi_u', 'X_mrvi_z', 'X_umap', 'X_pca'
    varm: 'PCs'
    layers: 'raw_counts', 'counts'
    obsp: 'distances', 'connectivities'
```

```python  # cell 36
adt.obs.study.value_counts().index.tolist()
```

```
['li2024',
 'chennareddy2025',
 'brunner2024',
 'gaydosik2019',
 'rindler2021_fi',
 'herrera2021']
```
