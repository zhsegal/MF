# 30_blood/31b_subtype_reannotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/30_blood/31b_subtype_reannotation.ipynb` and is never run._

# 31b — Blood non-T subtypes

`31_reannotation` labels the blood compartment broadly and then follows the T/NK cells into
`blood_T_annotated.h5ad`. Everything else — monocytes, DC, B, plasma, platelets, HSPC — stays at
`cell_type_broad`. This notebook gives those cells a subtype, exactly as `22`/`22b` do for skin,
reusing `helpers/annot_utils.py` unchanged.

**Why it is needed now.** It is the off-tumour denominator of the target screen. In blood an
off-target hit is not a rash: a CAR that also binds monocytes, NK cells or HSPC causes cytopenia.
Without this sidecar, `43_target_screen_blood` can only check T/NK arms, so a myeloid or
progenitor hit is invisible — the exact failure mode that matters most in this compartment.

**Method**, matching the skin notebooks so the two compartments stay comparable:
scVI on the non-T subset → Leiden → a hand-written cluster→label map, size-locked by
`annot_utils.check_map` so a silent change in cluster numbering cannot renumber the labels.

**Writes** `data/atlas_joint/blood_subtypes.csv` (`cell_id`, `subtype_fine`, `subtype_coarse`,
`residual_source`) + `blood_subtypes_provenance.json`, the same sidecar shape as
`skin_myeloid_fibro_subtypes.csv`.

> **HEAVY** — scVI training. Run on the GPU kernel or hand it to `jobs/run_notebook.sh`.
> Hand-annotation notebook: the maps in §3 are filled in by **you**, not generated.

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))
import annot_utils as A

OUT_DIR = NB_DIR / "data" / "atlas_joint"
FIG_DIR = NB_DIR / "figures" / "blood_subtypes"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = NB_DIR / "tables"

MRVI_IN = OUT_DIR / "joint_mrvi_input_blood.h5ad"      # 811,024 blood cells, 10k HVG
BROAD_CSV = OUT_DIR / "blood_cell_type_final.csv"      # nb31 broad labels
T_OBJ = OUT_DIR / "blood_T_annotated.h5ad"             # the T/NK cells nb31 already followed

SUBTYPE_CSV = OUT_DIR / "blood_subtypes.csv"
PROV_JSON = OUT_DIR / "blood_subtypes_provenance.json"

# The clustered object, cached. This notebook is run TWICE: once to make the clusters (heavy --
# send it to jobs/run_notebook.sh) and once after you have written the map. Without the cache the
# second pass would retrain scVI from a fresh init, renumber the clusters, and your map would
# point at different cells. check_map's size lock would catch it, but only after the work was done
# twice.
CLUSTERED = OUT_DIR / "blood_nonT_clustered_v1.h5ad"
MODEL_DIR = NB_DIR / "models" / "blood_nonT_scvi_v1"

SEED = 0
LEIDEN_RES = 1.0
N_LATENT, N_HVG = 30, 3000
sc.settings.verbosity = 1
print("NB_DIR =", NB_DIR)
```

## §1 — The non-T subset

Everything `31` did **not** follow into the T object. The T cells are excluded by cell id rather
than by label, so a cell that `31` moved between broad classes cannot end up in both tracks.

```python  # cell 3
# ============================================================================
# §1  Subset
# ============================================================================
import h5py

broad = pd.read_csv(BROAD_CSV)
id_col = "cell_id" if "cell_id" in broad.columns else broad.columns[0]
lab_col = [c for c in broad.columns if c != id_col][0]
print(f"{BROAD_CSV.name}: {len(broad):,} rows | label column `{lab_col}`")
print(broad[lab_col].value_counts().to_string())

with h5py.File(T_OBJ, "r") as h5:
    t_ids = pd.Index(np.asarray(
        h5["obs"][h5["obs"].attrs["_index"]][:]).astype(str))
print(f"\nT/NK cells already annotated by 31: {len(t_ids):,}")

if CLUSTERED.exists():
    print(f"{CLUSTERED.name} exists -- \u00a71/\u00a72 are skipped, go to \u00a73")
adata = sc.read_h5ad(MRVI_IN)
print("blood MrVI input:", adata.shape)
is_T = adata.obs_names.isin(t_ids)
sub = adata[~is_T].copy()
del adata
sub.obs["cell_type_broad"] = (broad.set_index(id_col)[lab_col]
                              .reindex(sub.obs_names).fillna("unknown").to_numpy())
print(f"\nnon-T cells to subtype: {sub.n_obs:,}")
print(sub.obs["cell_type_broad"].value_counts().to_string())
```

## §2 — scVI + Leiden

Batch key is `study`, as in `31`. The latent is new rather than reusing the MrVI blood latent:
that one was fitted to separate donors across the whole compartment, and the structure needed
here is within the non-T cells.

```python  # cell 5
# ============================================================================
# §2  scVI + Leiden   (HEAVY once; everything after is a cache read)
# ============================================================================
# Two levels of cache, because the expensive step and the fragile step fail separately:
#   MODEL_DIR  the trained scVI weights, saved the moment training ends and BEFORE
#              neighbours / leiden / umap -- a crash or a kernel restart downstream no longer
#              costs the training.
#   CLUSTERED  the object with X_scVI + leiden_sub + UMAP, written at the end.
# Reload order: CLUSTERED (seconds) -> MODEL_DIR (re-derive latent, re-cluster) -> train.
#
# X is left as log1p CP10K and the raw counts live in layers["counts"], exactly as 22b does it.
# That split matters in both directions: scVI and seurat_v3 HVGs must see raw counts, and the
# marker tables in §3 must NOT -- a mean over raw counts is a function of sequencing depth, so
# comparing clusters on it ranks libraries rather than cell types.
if CLUSTERED.exists():
    sub = sc.read_h5ad(CLUSTERED)
    print(f"loaded {CLUSTERED.name}: {sub.shape} | "
          f"{sub.obs['leiden_sub'].nunique()} clusters -- scVI NOT retrained")
else:
    import scvi

    _src = next((L for L in ("raw_counts", "counts") if L in sub.layers), None)
    assert _src is not None, f"no counts layer on the subset; have {list(sub.layers)}"
    if _src != "counts":
        sub.layers["counts"] = sub.layers.pop(_src)
    print(f"counts layer: {_src} -> layers['counts']")

    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sub.uns["lognorm_applied"] = True

    sc.pp.highly_variable_genes(sub, n_top_genes=N_HVG, flavor="seurat_v3",
                                layer="counts", batch_key="study", subset=False)
    scvi.model.SCVI.setup_anndata(sub, layer="counts", batch_key="study")

    if MODEL_DIR.exists():
        model = scvi.model.SCVI.load(str(MODEL_DIR), adata=sub)
        print(f"loaded trained scVI from {MODEL_DIR.name} -- NOT retraining")
    else:
        model = scvi.model.SCVI(sub, n_latent=N_LATENT)
        model.train(early_stopping=True)
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_DIR), overwrite=True)
        print(f"trained and saved scVI -> {MODEL_DIR}")

    sub.obsm["X_scVI"] = model.get_latent_representation()

    sc.pp.neighbors(sub, use_rep="X_scVI", random_state=SEED)
    sc.tl.leiden(sub, resolution=LEIDEN_RES, key_added="leiden_sub",
                 flavor="igraph", n_iterations=2, random_state=SEED)
    sc.tl.umap(sub, random_state=SEED)
    sub.write_h5ad(CLUSTERED)
    print(f"wrote {CLUSTERED.name}: {sub.shape}")

# The gate on everything in §3 -- and a repair, because a cache written before the normalisation
# fix holds raw counts in X and the marker tables silently become depth rankings. Values in the
# tens or hundreds rather than single digits is exactly what that looks like. Repair rather than
# raise: the expensive part (scVI, clustering) is still valid, only the scale of X is wrong.
_xmax = float(sub.X.max())
if _xmax >= 20:
    print(f"X.max()={_xmax:.1f} -- raw counts, not log1p CP10K. Repairing from the counts layer.")
    _src = next((L for L in ("counts", "raw_counts") if L in sub.layers), None)
    assert _src is not None, f"cannot repair: no counts layer, have {list(sub.layers)}"
    if _src != "counts":
        sub.layers["counts"] = sub.layers[_src]
    sub.X = sub.layers["counts"].copy()
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sub.uns["lognorm_applied"] = True
    sub.write_h5ad(CLUSTERED)        # so the next reload is already correct
    print(f"repaired and rewrote {CLUSTERED.name}")
    _xmax = float(sub.X.max())
    print("\n*** Any round-1 map written before this repair was derived from RAW COUNT means, "
          "which rank sequencing depth rather than cell identity. Re-run §3a and re-check the "
          "map before accepting it. ***")
assert _xmax < 20, f"X.max()={_xmax:.1f} after repair -- stop and inspect"
print(f"X is log1p CP10K (max {_xmax:.2f}) | raw counts in layers['counts']")
print(sub.obs["leiden_sub"].value_counts().sort_index().to_string())
```

```
loaded blood_nonT_clustered_v1.h5ad: (143440, 10000) | 54 clusters -- scVI NOT retrained
X.max()=26791.0 -- raw counts, not log1p CP10K. Repairing from the counts layer.
repaired and rewrote blood_nonT_clustered_v1.h5ad

*** Any round-1 map written before this repair was derived from RAW COUNT means, which rank sequencing depth rather than cell identity. Re-run §3a and re-check the map before accepting it. ***
X is log1p CP10K (max 9.15) | raw counts in layers['counts']
leiden_sub
0      4800
1      1021
2      8175
3      3053
4       486
5      3607
6      4163
7      2081
8      1910
9        53
10      203
11      598
12     1023
13     1442
14     4858
15     1581
16     1055
17      657
18    22791
19        9
20     5509
21       30
22     2110
23     6169
24     2818
25     5963
26     3243
27      465
28     3670
29     7375
30       22
31     1021
32      466
33     1314
34     1781
35     5635
36     5940
37     1844
38      526
39     3443
40     4086
41      479
42      763
43     1757
44     2350
45     3941
46     3082
47     1268
48     2382
49      207
50       68
51       93
52       33
53       21
```

## §3 — The hand map

`paste_block` prints one block per cluster — marker means, detection fractions, donor/study
spread — and you write the label. `check_map` then locks the map to the cluster **sizes**: if the
clustering is ever re-run and the numbering shifts, the map fails loudly instead of relabelling
cells silently. That lock is the whole point of the pattern; do not remove it to "fix" an error.

```python  # cell 7
# ============================================================================
# §3a  Marker panels, the per-cluster blocks, and the numbers behind them
# ============================================================================
PANELS = {
    "mono_classical":   ["CD14", "S100A8", "S100A9", "VCAN", "LYZ", "FCN1"],
    "mono_nonclassical": ["FCGR3A", "MS4A7", "CDKN1C", "LST1", "TCF7L2"],
    "cDC1":   ["CLEC9A", "XCR1", "BATF3", "IDO1"],
    "cDC2":   ["CD1C", "FCER1A", "CLEC10A", "CD1D"],
    "pDC":    ["LILRA4", "IL3RA", "CLEC4C", "GZMB", "JCHAIN"],
    "B_naive": ["MS4A1", "CD79A", "CD79B", "TCL1A", "IGHD", "IGHM"],
    "B_memory": ["MS4A1", "CD27", "TNFRSF13B", "AIM2"],
    "plasma": ["JCHAIN", "MZB1", "XBP1", "DERL3", "PRDM1"],
    "NK":     ["NCAM1", "KLRD1", "NKG7", "GNLY", "FCGR3A", "KLRF1"],
    "platelet": ["PPBP", "PF4", "ITGA2B", "GP9"],
    "HSPC":   ["CD34", "SPINK2", "PRSS57", "MPO", "ELANE"],
    "erythroid": ["HBB", "HBA1", "HBD", "ALAS2", "GYPA"],
    "progenitor_mk": ["ITGA2B", "GATA1", "PLEK"],
    "doublet_T":  ["CD3E", "CD3D", "TRAC", "IL7R"],
    "stress_qc": ["MALAT1", "NEAT1", "HSPA1A", "FOS", "JUN"],
}
# present_panels returns (keep, dropped) -- the drop is the point, not a side effect. This
# object is a 10k-HVG matrix, so a panel can quietly lose the gene that defines it and leave a
# cluster that cannot be called rather than an error.
panels, DROPPED = A.present_panels(sub.var_names, PANELS, label="blood non-T")

_gone = [c for c in PANELS if c not in panels]
if _gone:
    print(f"\n!! panels with NO genes left on this object: {_gone}")
    print("   Any cluster that is really one of these CANNOT be called from markers here -- "
          "label it Unk rather than forcing it onto a neighbouring panel.")
_thin_panel = {c: g for c, g in panels.items() if len(g) < 2}
if _thin_panel:
    print(f"!! panels down to a single gene: {_thin_panel} -- one gene is not an identity call")

GENES = [g for gs in panels.values() for g in gs]
mean, frac = A.cluster_expression(sub, "leiden_sub", GENES)
qc = A.cluster_qc(sub, "leiden_sub", donor_key="donor", study_key="study")

VOCAB = ["Mono_classical", "Mono_nonclassical", "cDC1", "cDC2", "pDC", "B_naive", "B_memory",
         "Plasma", "NK_CD56dim", "NK_CD56bright", "Platelet", "HSPC", "Erythroid",
         "Doublet", "LowQC", "Unk"]

# paste_block RETURNS the block, it does not print it -- without print() the notebook shows a
# repr with escaped newlines and the block is unreadable.
print(A.paste_block(sub, "leiden_sub", panels, mean, frac, qc, VOCAB,
                    title="blood non-T, leiden res %.1f" % LEIDEN_RES))

# ---- the dotplot's numbers, printed and persisted --------------------------------
# A dotplot encodes exactly two quantities per cell of the grid: mean log1p CP10K (colour) and
# the fraction of cells detecting the gene (dot size). Reading them off a figure is guesswork at
# the margins, which is where the calls are actually hard -- so print both matrices and write
# them out. `mean` and `frac` here are the same objects the figure in §4 is drawn from.
pd.set_option("display.width", 250, "display.max_columns", 200)

print("\n" + "=" * 100)
print("DOTPLOT VALUES - mean log1p CP10K per cluster x gene  (the colour channel)")
print("=" * 100)
print(mean.to_string())

print("\n" + "=" * 100)
print("DOTPLOT VALUES - fraction of cells detecting the gene  (the dot-size channel)")
print("=" * 100)
print(frac.to_string())

print("\n" + "=" * 100)
print("CLUSTER QC  (n, donors, studies, top study share, depth, mito, doublet)")
print("=" * 100)
print(qc.to_string())

_long = (mean.stack().rename("mean_log1p_cp10k").to_frame()
         .join(frac.stack().rename("frac_detected"))
         .rename_axis(["leiden_sub", "gene"]).reset_index())
_long["panel"] = _long["gene"].map({g: cat for cat, gs in panels.items() for g in gs})
_long.to_csv(TAB_DIR / "nb31b_cluster_marker_values.csv", index=False)
qc.to_csv(TAB_DIR / "nb31b_cluster_qc.csv")
print(f"\nwrote {TAB_DIR / 'nb31b_cluster_marker_values.csv'} ({len(_long):,} rows)"
      f"\n      {TAB_DIR / 'nb31b_cluster_qc.csv'}")

# The one number that decides whether a block can be called at all: a cluster carried by a single
# donor or a single study is a batch, not a cell type, and labelling it costs you a real subtype.
_thin = qc[(qc["n_donors"] <= 2) | (qc["top_study_frac"] >= 0.9)]
if len(_thin):
    print(f"\n{len(_thin)} cluster(s) from <=2 donors or >=90% one study -- treat as batch "
          f"until proven otherwise: {list(_thin.index)}")
```

```
====================================================================================================
blood non-T, leiden res 1.0 — 143,440 cells, 54 clusters; values are `mean log1p CP10K / fraction detected`
allowed labels: ['Mono_classical', 'Mono_nonclassical', 'cDC1', 'cDC2', 'pDC', 'B_naive', 'B_memory', 'Plasma', 'NK_CD56dim', 'NK_CD56bright', 'Platelet', 'HSPC', 'Erythroid', 'Doublet', 'LowQC', 'Unk']
====================================================================================================

--- cluster 0  n=4,800  donors=37  studies=10  top_study=borcherding2023(0.35)  med_genes=1517  med_mito=4.06  doublet=0.041
    mono_classical   CD14 2.42/0.85  S100A8 5.57/0.99  S100A9 5.21/0.99  VCAN 3.00/0.91  LYZ 4.45/1.00  FCN1 3.59/0.96
    mono_nonclassical FCGR3A 0.22/0.11  MS4A7 0.91/0.47  CDKN1C 0.05/0.03  LST1 2.72/0.93  TCF7L2 0.17/0.11
    cDC1             CLEC9A 0.00/0.00  XCR1 0.00/0.00  BATF3 0.08/0.05  IDO1 0.00/0.00
    cDC2             CD1C 0.06/0.04  FCER1A 0.04/0.02  CLEC10A 0.14/0.08  CD1D 0.51/0.31
    pDC              LILRA4 0.02/0.01  IL3RA 0.03/0.02  CLEC4C 0.00/0.00  GZMB 0.01/0.01  JCHAIN 0.07/0.04
    B_naive          MS4A1 0.01/0.01  CD79A 0.02/0.01  CD79B 0.06/0.04  TCL1A 0.00/0.00  IGHD 0.00/0.00  IGHM 0.09/0.05
    B_memory         MS4A1 0.01/0.01  CD27 0.02/0.01  TNFRSF13B 0.00/0.00  AIM2 0.05/0.03
    plasma           JCHAIN 0.07/0.04  MZB1 0.01/0.01  XBP1 0.46/0.28  DERL3 0.01/0.01  PRDM1 0.02/0.01
    NK               NCAM1 0.00/0.00  KLRD1 0.01/0.00  NKG7 0.30/0.17  GNLY 0.07/0.04  FCGR3A 0.22/0.11  KLRF1 0.00/0.00
    platelet         PPBP 0.18/0.08  PF4 0.11/0.05  ITGA2B 0.03/0.02  GP9 0.04/0.02
    HSPC             CD34 0.00/0.00  SPINK2 0.00/0.00  PRSS57 0.01/0.01  MPO 0.05/0.03  ELANE 0.01/0.01
    erythroid        HBB 0.08/0.03  HBA1 0.02/0.01  HBD 0.00/0.00  ALAS2 0.01/0.00  GYPA 0.00/0.00
    progenitor_mk    ITGA2B 0.03/0.02  GATA1 0.00/0.00  PLEK 0.97/0.51
    doublet_T        CD3E 0.03/0.02  CD3D 0.02/0.02  TRAC 0.02/0.01  IL7R 0.04/0.02
    stress_qc        MALAT1 4.55/0.99  NEAT1 2.86/0.93  HSPA1A 0.50/0.29  FOS 3.13/0.91  JUN 0.72/0.33

--- cluster 1  n=1,021  donors=42  studies=9  top_study=dorando2026(0.26)  med_genes=2938  med_mito=4.43  doublet=0.067
    mono_classical   CD14 0.28/0.19  S100A8 1.26/0.46  S100A9 1.80/0.65  VCAN 0.85/0.45  LYZ 4.35/1.00  FCN1 1.49/0.61
    mono_nonclassical FCGR3A 0.03/0.02  MS4A7 0.43/0.38  CDKN1C 0.02/0.02  LST1 2.18/0.96  TCF7L2 0.03/0.04
    cDC1             CLEC9A 0.19/0.11  XCR1 0.04/0.04  BATF3 0.42/0.38  IDO1 0.18/0.13
    cDC2             CD1C 1.54/0.73  FCER1A 2.03/0.81  CLEC10A 1.67/0.75  CD1D 0.75/0.58
    pDC              LILRA4 0.13/0.09  IL3RA 0.11/0.10  CLEC4C 0.06/0.05  GZMB 0.04/0.02  JCHAIN 0.07/0.05
    B_naive          MS4A1 0.03/0.02  CD79A 0.04/0.04  CD79B 0.14/0.14  TCL1A 0.01/0.01  IGHD 0.01/0.01  IGHM 0.05/0.04
    B_memory         MS4A1 0.03/0.02  CD27 0.05/0.05  TNFRSF13B 0.00/0.00  AIM2 0.04/0.05
    plasma           JCHAIN 0.07/0.05  MZB1 0.02/0.02  XBP1 0.53/0.48  DERL3 0.01/0.01  PRDM1 0.07/0.07
    NK               NCAM1 0.01/0.02  KLRD1 0.01/0.01  NKG7 0.26/0.22  GNLY 0.08/0.05  FCGR3A 0.03/0.02  KLRF1 0.01/0.01
    platelet         PPBP 0.04/0.03  PF4 0.02/0.02  ITGA2B 0.01/0.01  GP9 0.00/0.00
    HSPC             CD34 0.00/0.00  SPINK2 0.01/0.02  PRSS57 0.02/0.02  MPO 0.01/0.01  ELANE 0.00/0.00
    erythroid        HBB 0.04/0.03  HBA1 0.00/0.00  HBD 0.00/0.00  ALAS2 0.00/0.00  GYPA 0.00/0.00
    progenitor_mk    ITGA2B 0.01/0.01  GATA1 0.00/0.00  PLEK 1.22/0.82
    doublet_T        CD3E 0.17/0.13  CD3D 0.08/0.07  TRAC 0.06/0.06  IL7R 0.14/0.10
    stress_qc        MALAT1 4.42/1.00  NEAT1 1.27/0.76  HSPA1A 0.33/0.29  FOS 3.18/0.97  JUN 0.98/0.49

--- cluster 2  n=8,175  donors=36  studies=10  top_study=dorando2026(0.51)  med_genes=1562  med_mito=4.51  doublet=0.066
    mono_classical   CD14 2.29/0.86  S100A8 5.57/1.00  S100A9 4.93/0.99  VCAN 2.77/0.91  LYZ 4.53/0.99  FCN1 3.41/0.98
    mono_nonclassic
... [140257 more chars truncated]
```

```python  # cell 8
# ============================================================================
# §3b  YOUR MAP -- fill this in from the blocks above, then run
# ============================================================================
# cluster id -> label from VOCAB. Leave a cluster out and check_map will say so.
SUB_MAP = {
    "0":  "Mono_classical",     # H  CD14/S100A8/9/VCAN/FCN1; 10 studies
    "1":  "cDC2",               # H  CD1C/FCER1A/CLEC10A ~0.75; a few CLEC9A+ (0.11) cDC1 embedded
    "2":  "Mono_classical",     # H  NEAT1/JUN-hi variant
    "3":  "Mono_classical",     # H  FCGR3A 0.36 -> intermediate admixture
    "4":  "Mono_classical",     # H  herrera-only, canonical profile (integration residual)
    "5":  "LowQC",              # 484 genes, mito 0.8, HBB/GNLY/LYZ ambient
    "6":  "Erythroid",          # RBC contamination: 303 genes, MALAT1-lo
    "7":  "Doublet",            # mono x T: CD3E/IL7R ~0.65, scrublet 0.10
    "8":  "Unk",                # low-magnitude CD3E/IL7R/GNLY/GZMB across 10 studies; residual T/NK leak?
    "9":  "Unk",                # geskin soup profile, n=53
    "10": "Doublet",            # neutrophil-like x T; mito 11.6, scrublet 0.12, ren-only
    "11": "Plasma",             # H  JCHAIN/MZB1/DERL3/XBP1/PRDM1
    "12": "Unk",                # HSPA1A 0.84 stress cluster pooling VCAN+/TCF7L2+/CD34+ cells; buus
    "13": "HSPC",               # H  CD34/SPINK2/PRSS57; GATA1/HBD/FCER1A -> MEP/baso-mast progenitors included
    "14": "B_naive",            # H  TCL1A 0.59, IGHD 0.54
    "15": "B_naive",            # H  buus batch
    "16": "Unk",                # B cell, subtype unresolved; every subtype marker below 24/29, mito 7.6
    "17": "B_naive",            # H  ren batch; TCL1A/IGHD >0.9
    "18": "Unk",                # geskin soup profile, n=22,791 (see note)
    "19": "Doublet",            # n=9, MS4A1 + LYZ + GNLY all ~1.0
    "20": "Doublet",            # mono x T/NK: 4,245 genes, CD3E 0.86, IL7R 0.89, GNLY 0.58
    "21": "Unk",                # geskin soup, n=30
    "22": "Unk",                # HBB 0.91 + LYZ/GNLY/CD3E soup
    "23": "LowQC",              # 627 genes, MALAT1-lo monocyte debris
    "24": "B_memory",           # M  TNFRSF13B 0.40, AIM2 0.38, CD27 0.25, IGHD 0.18
    "25": "Mono_classical",     # M  VCAN/FCN1/S100A8; FCGR3A 0.61 plausibly NK ambient; IDO1 0.72
    "26": "Mono_nonclassical",  # H  FCGR3A/CDKN1C/TCF7L2/LST1/MS4A7
    "27": "pDC",                # H  LILRA4/CLEC4C/IL3RA/GZMB/JCHAIN
    "28": "B_naive",            # H
    "29": "B_naive",            # M  TCL1A 0.53; CD27/TNFRSF13B ~0.28 -> memory admixture
    "30": "Unk",                # n=22
    "31": "LowQC",              # MALAT1 0.88 (anucleate), HBB 0.65, 579 genes
    "32": "Platelet",           # H  PPBP/PF4/GP9/ITGA2B
    "33": "Mono_classical",     # H  buus batch
    "34": "Unk",                # neutrophils (S100A8/9 1.0, LYZ/VCAN/FCN1 neg, 378 genes, NEAT1-hi); not in label set
    "35": "LowQC",              # 712 genes, MALAT1-lo monocytes
    "36": "Mono_classical",     # M  CD14 0.65/VCAN/FCN1; IDO1 0.61, IL7R 0.57
    "37": "Doublet",            # mono x NK/T: 4,323 genes, GZMB/GNLY/NKG7 ~0.6, CD3E 0.53
    "38": "Unk",                # geskin soup, T-leaning
    "39": "Erythroid",          # RBC
    "40": "Erythroid",          # RBC
    "41": "Mono_classical",     # H  gaydosik batch
    "42": "Mono_classical",     # H  ren batch, deep (3,325 genes)
    "43": "Unk",                # weak mono (S100A8/9 ~0.5, CD14 0.38) + IDO1/IL7R; candidate Mono_classical
    "44": "Unk",                # IDO1 0.89 / BATF3 0.40 / IL7R 0.85, CD14-lo: mregDC vs mono x T doublet
    "45": "LowQC",              # MALAT1 1.97, HBB 0.88, 808 genes
    "46": "Doublet",            # mono x RBC: HBB 1.00 at RBC-cluster magnitude; myeloid partner nonclassical-like
    "47": "Erythroid",          # RBC
    "48": "Mono_classical",     # M  CD14-hi/MS4A7-hi, LYZ/FCN1/LST1-lo: atypical activated state, buus-only
    "49": "Doublet",            # mono x T: CD3E/CD3D 0.95, 5,517 genes
    "50": "Unk",                # geskin soup; JCHAIN 1.0 without MZB1/DERL3 -> not plasma
    "51": "Unk",                # as 50
    "52": "Unk",                # n=33
    "53": "Unk",                # n=21
}
# The size lock: cluster id -> n cells, copied from the blocks. check_map refuses to apply the
# map if any count moved, which is what stops a re-clustering from relabelling cells silently.
SIZES = {'0': 4800, '1': 1021, '2': 8175, '3': 3053, '4': 486, '5': 3607, '6': 4163, '7': 2081, '8': 1910, '9': 53, '10': 203, '11': 598, '12': 1023, '13': 1442, '14': 4858, '15': 1581, '16': 1055, '17': 657, '18': 22791, '19': 9, '20': 5509, '21': 30, '22': 2110, '23': 6169, '24': 2818, '25': 5963, '26': 3243, '27': 465, '28': 3670, '29': 7375, '30': 22, '31': 1021, '32': 466, '33': 1314, '34': 1781, '35': 5635, '36': 5940, '37': 1844, '38': 526, '39': 3443, '40': 4086, '41': 479, '42': 763, '43': 1757, '44': 2350, '45': 3941, '46': 3082, '47': 1268, '48': 2382, '49': 207, '50': 68, '51': 93, '52': 33, '53': 21}

if not SUB_MAP:
    print("SUB_MAP is empty -- fill it from the blocks above. Nothing was written.")
else:
    A.check_map(sub, "leiden_sub", SUB_MAP, SIZES, VOCAB, label="blood non-T",
                out_col="subtype_fine")
    COARSE = {
        "Mono_classical": "Monocyte", "Mono_nonclassical": "Monocyte",
        "cDC1": "DC", "cDC2": "DC", "pDC": "DC",
        "B_naive": "B", "B_memory": "B", "Plasma": "Plasma",
        "NK_CD56dim": "NK", "NK_CD56bright": "NK",
        "Platelet": "Platelet", "HSPC": "HSPC", "Erythroid": "Erythroid",
        "Doublet": "Doublet", "LowQC": "LowQC", "Unk": "Unk",
    }
    sub.obs["subtype_coarse"] = sub.obs["subtype_fine"].map(COARSE).astype(str)
    print(pd.crosstab(sub.obs["subtype_fine"], sub.obs["subtype_coarse"]).to_string())
```

```
subtype_coarse         B    DC  Doublet  Erythroid  HSPC  LowQC  Monocyte  Plasma  Platelet    Unk
subtype_fine                                                                                      
B_memory            2818     0        0          0     0      0         0       0         0      0
B_naive            18141     0        0          0     0      0         0       0         0      0
Doublet                0     0    12935          0     0      0         0       0         0      0
Erythroid              0     0        0      12960     0      0         0       0         0      0
HSPC                   0     0        0          0  1442      0         0       0         0      0
LowQC                  0     0        0          0     0  20373         0       0         0      0
Mono_classical         0     0        0          0     0      0     33355       0         0      0
Mono_nonclassical      0     0        0          0     0      0      3243       0         0      0
Plasma                 0     0        0          0     0      0         0     598         0      0
Platelet               0     0        0          0     0      0         0       0       466      0
Unk                    0     0        0          0     0      0         0       0         0  35623
cDC2                   0  1021        0          0     0      0         0       0         0      0
pDC                    0   465        0          0     0      0         0       0         0      0
```

```python  # cell 9
# ============================================================================
# §4  Figures + sidecar
# ============================================================================
if "subtype_fine" in sub.obs:
    A.save_umap(sub, "subtype_fine", FIG_DIR / "nb31b_umap_subtype_fine.png",
                title="blood non-T subtypes")
    A.save_dotplot(sub, panels, "subtype_fine", FIG_DIR / "nb31b_dotplot.png",
                   title="blood non-T markers")

    # The same two channels as the figure, by label this time, printed and written out.
    mean_lab, frac_lab = A.cluster_expression(sub, "subtype_fine", GENES)
    print("\nDOTPLOT VALUES by subtype - mean log1p CP10K:")
    print(mean_lab.to_string())
    print("\nDOTPLOT VALUES by subtype - fraction detected:")
    print(frac_lab.to_string())
    (mean_lab.stack().rename("mean_log1p_cp10k").to_frame()
     .join(frac_lab.stack().rename("frac_detected"))
     .rename_axis(["subtype_fine", "gene"]).reset_index()
     .to_csv(TAB_DIR / "nb31b_subtype_marker_values.csv", index=False))
    print("\nwrote", TAB_DIR / "nb31b_subtype_marker_values.csv")

    frame = pd.DataFrame({
        "cell_id": sub.obs_names.astype(str),
        "subtype_fine": sub.obs["subtype_fine"].astype(str),
        "subtype_coarse": sub.obs["subtype_coarse"].astype(str),
        "residual_source": sub.obs["cell_type_broad"].astype(str),
    })
    prov = {
        "notebook": "30_blood/31b_subtype_reannotation.ipynb",
        "input": str(MRVI_IN), "broad_labels": str(BROAD_CSV),
        "excluded_T_object": str(T_OBJ),
        "method": f"scVI n_latent={N_LATENT} batch=study -> leiden res={LEIDEN_RES}",
        "n_cells": int(len(frame)), "seed": SEED,
        "vocab": VOCAB,
    }
    A.write_sidecar(frame, SUBTYPE_CSV, PROV_JSON, prov, expect_rows=len(frame))
    print("\nwrote", SUBTYPE_CSV)
    print(frame["subtype_fine"].value_counts().to_string())
else:
    print("no subtype_fine yet -- fill SUB_MAP in §3b first")
```

```
<Figure size 640x480 with 1 Axes>
WARNING: dendrogram data not found (using key=dendrogram_subtype_fine). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: B_memory, B_naive, Doublet, etc.
var_group_labels: mono_classical, mono_nonclassical, cDC1, etc.

<Figure size 1600x500 with 7 Axes>

DOTPLOT VALUES by subtype - mean log1p CP10K:
                   CD14  S100A8  S100A9  VCAN   LYZ  FCN1  FCGR3A  MS4A7  CDKN1C  LST1  TCF7L2  CLEC9A  XCR1  BATF3  IDO1  CD1C  FCER1A  CLEC10A  CD1D  LILRA4  IL3RA  CLEC4C  GZMB  JCHAIN  MS4A1  CD79A  CD79B  TCL1A  IGHD  IGHM  CD27  TNFRSF13B  AIM2  MZB1  XBP1  DERL3  PRDM1  NCAM1  KLRD1  NKG7  GNLY  KLRF1  PPBP   PF4  ITGA2B   GP9  CD34  SPINK2  PRSS57   MPO  ELANE   HBB  HBA1   HBD  ALAS2  GYPA  GATA1  PLEK  CD3E  CD3D  TRAC  IL7R  MALAT1  NEAT1  HSPA1A   FOS   JUN
B_memory           0.04    0.63    0.30  0.08  2.09  0.16    0.11   0.13    0.00  0.26    0.03    0.00  0.00   0.02  0.37  0.06    0.00     0.01  0.01    0.13   0.10    0.00  0.16    0.31   2.62   2.02   1.02   0.78  0.33  1.86  0.40       0.74  0.67  0.31  0.41   0.05   0.10   0.02   0.08  0.32  0.42   0.03  0.09  0.04    0.01  0.00  0.00    0.03    0.00  0.00   0.00  0.14  0.01  0.00   0.00  0.00   0.00  0.43  0.33  0.19  0.25  0.29    5.52   1.34    0.12  0.03  0.61
B_naive            0.01    0.16    0.14  0.02  0.38  0.03    0.05   0.17    0.00  0.23    0.01    0.00  0.00   0.03  0.11  0.14    0.00     0.00  0.04    0.06   0.10    0.01  0.22    0.57   2.86   2.81   1.88   1.61  0.80  2.02  0.31       0.36  0.28  0.38  0.33   0.17   0.06   0.01   0.05  0.22  0.29   0.02  0.03  0.01    0.00  0.00  0.00    0.03    0.00  0.00   0.00  0.42  0.08  0.02   0.02  0.00   0.00  0.45  0.25  0.18  0.21  0.18    5.52   0.75    0.10  0.55  1.17
Doublet            0.83    2.53    2.64  0.61  3.57  1.78    0.90   0.84    0.02  1.70    0.28    0.00  0.00   0.16  1.38  0.09    0.01     0.43  0.06    0.03   0.57    0.00  1.00    0.03   0.15   0.12   0.18   0.04  0.02  0.10  0.34       0.02  0.11  0.05  0.80   0.02   0.29   0.09   0.20  1.13  1.09   0.06  0.08  0.02    0.00  0.00  0.00    0.01    0.01  0.01   0.00  1.88  0.65  0.14   0.21  0.00   0.00  1.37  1.23  1.16  0.55  1.51    4.44   1.71    0.65  0.95  0.78
Erythroid          0.02    0.20    0.21  0.00  0.39  0.03    0.03   0.04    0.00  0.12    0.01    0.00  0.00   0.02  0.07  0.00    0.00     0.00  0.00    0.00   0.02    0.00  0.27    0.00   0.01   0.01   0.01   0.00  0.00  0.01  0.04       0.00  0.02  0.00  0.05   0.00   0.03   0.00   0.02  0.25  0.29   0.00  0.08  0.03    0.01  0.00  0.00    0.00    0.00  0.00   0.00  8.78  4.61  1.35   1.76  0.02   0.04  0.12  0.29  0.23  0.09  0.07    1.31   0.17    0.06  0.03  0.05
HSPC               0.02    0.11    0.07  0.04  0.38  0.02    0.05   0.05    0.16  0.55    0.15    0.07  0.00   0.01  0.04  0.00    0.72     0.00  0.00    0.00   0.27    0.00  0.13    0.21   0.04   0.10   0.15   0.01  0.10  0.52  0.15       0.00  0.02  0.27  0.97   0.05   0.04   0.00   0.03  0.44  0.24   0.02  0.07  0.02    0.26  0.03  1.27    1.25    1.75  0.06   0.02  1.04  0.27  0.47   0.08  0.00   0.46  0.87  0.51  0.34  0.17  0.40    5.30   1.39    0.58  1.34  0.98
LowQC              0.94    2.27    1.45  0.41  3.48  0.88    0.55   1.29    0.02  1.18    0.07    0.00  0.00   0.02  0.36  0.01    0.01     0.10  0.02    0.01   0.22    0.00  0.23    0.01   0.06   0.03   0.04   0.03  0.01  0.06  0.03       0.01  0.02  0.01  0.21   0.00   0.11   0.01   0.03  0.22  0.37   0.01  0.27  0.06    0.02  0.01  0.00    0.00    0.00  0.01   0.01  1.51  0.23  0.04   0.04  0.00   0.00  0.48  0.21  0.19  0.09  0.44    3.00   1.43    0.35  0.32  0.26
Mono_classical     1.94    4.38    3.94  2.13  4.36  2.73    0.64   1.07    0.07  2.31    0.35    0.00  0.00   0.10  0.70  0.05    0.02     0.26  0.
... [10539 more chars truncated]
```

## §5 — Second round: the `Unk` residue

Round 1 labels what has a clean marker signature. What is left as `Unk` is not a leftover to be
tidied away — in blood it is where the genuinely hard populations hide (basophils, ILC, HSPC,
pre-B, AS-DC) **and** where ambient RNA accumulates. Forcing those cells onto a neighbouring
label is worse than leaving them `Unk`, because it puts fabricated cells into the off-tumour
denominator that `43_target_screen_blood` reads.

### The geskin2026 soup problem

`geskin2026` lanes are HTO-pooled, 5–6 donors per lane, so their ambient (free-mRNA) fraction is
high and the leaked transcripts are the most abundant ones in the lane: haemoglobin, platelet
granule, lysozyme, immunoglobulin. A droplet carrying mostly soup looks like a **weighted average
of everything in the lane**, which is exactly why it clusters on its own and resists annotation —
it is not a cell type, and no marker panel will ever call it.

So this round scores each residual cluster on five ambient signatures before any label is written:

| diagnostic | what a soup cluster looks like |
|---|---|
| `soup_corr` | profile correlates with the lane's *global mean* — the signature of an average of everything |
| `amb_frac` | high share of counts in ubiquitous leaked genes (HB*, PPBP/PF4, LYZ, S100A8/9, IGKC) |
| `n_lineages` | positive for several *mutually exclusive* lineages at once |
| `med_genes` | low complexity — few genes detected |
| `top_study_frac` | dominated by one study, and specifically by geskin2026 |

A cluster that is high on `soup_corr` **and** `amb_frac` **and** `n_lineages` is ambient. Label it
`Ambient_soup` — that is a finding, not a failure, and it keeps those barcodes out of every
downstream denominator.

> Both the subset and its scVI model are cached, so the hand-map step can be revisited without
> retraining. Round 2 re-embeds rather than reusing the round-1 latent: that latent was fitted to
> separate the *whole* non-T compartment, and the structure needed here is inside the residue.

```python  # cell 11
# ============================================================================
# §5a  The residual pool: re-embed the Unk cells on their own
# ============================================================================
RESID_H5AD = OUT_DIR / "blood_nonT_resid_v1.h5ad"
RESID_MODEL = NB_DIR / "models" / "blood_nonT_resid_scvi_v1"
LEIDEN_RES2 = 1.5          # higher than round 1: a small pool needs finer splitting
MIN_RESID = 200            # below this a second round is not worth running
MIN_HVG_BATCH = 200        # seurat_v3 loess per study goes singular on tiny studies

assert "subtype_fine" in sub.obs, "run §3b first -- round 2 annotates what round 1 left as Unk"
_is_unk = sub.obs["subtype_fine"].astype(str).eq("Unk").to_numpy()
print(f"round-1 Unk: {int(_is_unk.sum()):,} of {sub.n_obs:,} cells "
      f"({_is_unk.mean():.1%})")

if int(_is_unk.sum()) < MIN_RESID:
    print(f"fewer than {MIN_RESID} Unk cells -- skip §5, leave them Unk")
    resid = None
else:
    if RESID_H5AD.exists():
        resid = sc.read_h5ad(RESID_H5AD)
        print(f"loaded {RESID_H5AD.name}: {resid.shape} | "
              f"{resid.obs['leiden_resid'].nunique()} clusters -- scVI NOT retrained")
    else:
        import scvi

        resid = sub[_is_unk].copy()
        resid.obs["round1_cluster"] = sub.obs["leiden_sub"].astype(str)[_is_unk].to_numpy()
        # X should already be log1p CP10K from §2. Repair here too: this cell is often reached
        # from a kernel whose `sub` predates the fix, and an un-normalised residue would make
        # every round-2 marker table a depth ranking.
        assert "counts" in resid.layers, "raw counts needed for seurat_v3 / scVI"
        if float(resid.X.max()) >= 20:
            print("residue X is raw counts -- repairing (re-run §2 to fix `sub` and its cache)")
            resid.X = resid.layers["counts"].copy()
            sc.pp.normalize_total(resid, target_sum=1e4)
            sc.pp.log1p(resid)
        assert float(resid.X.max()) < 20, "resid X is not log1p CP10K after repair"

        _bk = "study" if resid.obs["study"].nunique() > 1 else None

        # HVG batch: studies with < MIN_HVG_BATCH residue cells pooled into "_small"; if
        # "_small" is itself tiny, fold it into the largest study.
        _study = resid.obs["study"].astype(str)
        _n = _study.value_counts()
        _hb = _study.where(_study.map(_n).ge(MIN_HVG_BATCH), "_small")
        if 0 < int(_hb.eq("_small").sum()) < MIN_HVG_BATCH:
            _hb = _hb.replace("_small", _n.index[0])
        resid.obs["hvg_batch"] = _hb.astype("category")
        _hbk = "hvg_batch" if resid.obs["hvg_batch"].nunique() > 1 else None
        print("HVG batches:", resid.obs["hvg_batch"].value_counts().to_dict())

        for _span, _b in [(0.3, _hbk), (0.5, _hbk), (0.5, None)]:
            try:
                sc.pp.highly_variable_genes(resid, n_top_genes=min(N_HVG, resid.n_vars - 1),
                                            flavor="seurat_v3", layer="counts", batch_key=_b,
                                            span=_span, subset=False)
                print(f"HVG ok: span={_span}, batch_key={_b}")
                break
            except ValueError as e:
                print(f"seurat_v3 loess failed (span={_span}, batch_key={_b}): {e}")
        else:
            raise RuntimeError("seurat_v3 HVG failed on every fallback")
        del resid.obs["hvg_batch"]

        scvi.model.SCVI.setup_anndata(resid, layer="counts", batch_key=_bk)
        if RESID_MODEL.exists():
            model2 = scvi.model.SCVI.load(str(RESID_MODEL), adata=resid)
            print(f"loaded {RESID_MODEL.name} -- NOT retraining")
        else:
            model2 = scvi.model.SCVI(resid, n_latent=N_LATENT)
            model2.train(early_stopping=True)
            RESID_MODEL.parent.mkdir(parents=True, exist_ok=True)
            model2.save(str(RESID_MODEL), overwrite=True)
            print(f"trained and saved -> {RESID_MODEL}")

        resid.obsm["X_scVI"] = model2.get_latent_representation()
        sc.pp.neighbors(resid, use_rep="X_scVI", random_state=SEED)
        sc.tl.leiden(resid, resolution=LEIDEN_RES2, key_added="leiden_resid",
                     flavor="igraph", n_iterations=2, random_state=SEED)
        sc.tl.umap(resid, random_state=SEED)
        resid.write_h5ad(RESID_H5AD)
        print(f"wrote {RESID_H5AD.name}: {resid.shape}")

    print("\nresidual clusters:")
    print(resid.obs["leiden_resid"].value_counts().sort_index().to_string())
    print("\nby study (geskin2026 share is the first thing to look at):")
    print(pd.crosstab(resid.obs["leiden_resid"], resid.obs["study"].astype(str),
                      normalize="index").round(2).to_string())
```

```
round-1 Unk: 35,623 of 143,440 cells (24.8%)
HVG batches: {'geskin2026': 30917, 'ren2023': 1889, 'buus2025': 1712, 'herrera2021': 336, '_small': 277, 'gaydosik2022': 267, 'harro2023': 225}
seurat_v3 loess failed (span=0.3, batch_key=hvg_batch): b'There are other near singularities as well. 0.090619'
HVG ok: span=0.5, batch_key=hvg_batch

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
You are using a CUDA device ('NVIDIA A40') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Epoch 225/225: 100%|██████████| 225/225 [13:26<00:00,  3.56s/it, v_num=1, train_loss_step=2.03e+3, train_loss_epoch=2.17e+3]
`Trainer.fit` stopped: `max_epochs=225` reached.

Epoch 225/225: 100%|██████████| 225/225 [13:26<00:00,  3.58s/it, v_num=1, train_loss_step=2.03e+3, train_loss_epoch=2.17e+3]
trained and saved -> /home/projects/nyosef/zvise/MF/models/blood_nonT_resid_scvi_v1
wrote blood_nonT_resid_v1.h5ad: (35623, 10000)

residual clusters:
leiden_resid
0      708
1      693
2     1649
3     1000
4     2069
5      179
6      738
7      965
8      452
9      403
10     411
11     692
12    1659
13    2194
14    2750
15    1465
16    2771
17      12
18    1248
19     953
20     803
21     950
22     202
23     623
24     230
25    1011
26     294
27     276
28     471
29     203
30     666
31    1110
32    1289
33      98
34    1699
35      89
36     122
37      81
38    1686
39      93
40     355
41      99
42      93
43      69

by study (geskin2026 share is the first thing to look at):
study         borcherding2019  borcherding2023  buus2025  dorando2026  gaydosik2022  geskin2026  harro2023  herrera2021  ren2023  rindler2021_fi
leiden_resid                                                                                                                                    
0                        0.01             0.14      0.13         0.04          0.03        0.35       0.10         0.13     0.08            0.01
1                        0.00             0.01      0.40         0.00          0.03        0.10       0.20         0.12     0.14            0.00
2                        0.00             0.00      0.00         0.00          0.01        0.98       0.00         0.01     0.00            0.00
3                        0.00             0.00      0.05         0.01          0.00        0.94       0.00         0.00     0.00            0.00
4                        0.00             0.00      0.00         0.00          0.00        1.00       0.00         0.00     0.00            0.00
5                        0.00             0.00      0.47         0.00          0.00        0.11       0.01         0.41     0.01            0.00
6                        0.00             0.00      0.92         0.01          0.00        0.01       0.00         0.02     0.04            0.00
7                        0.00             0.00      0.01         0.00          0.03        0.93       0.01         0.01     0.01            0.00
8                        0.00             0.00      0.00         0.00          0.00        0.99       0.00         0.00     0.00            0.00
9                        0.00             0.00      0.01         0.00          0.00        0.99       0.00         0.00     0.00            0.00
10                       0.00             0.00      0.00         0.00          0.00        1.00       0.00         0.00     0.00            0.00
11                       0.00             0.00      0.00         0.00          0.00        1.00       0.00         0.00     0.00            0.00
12                       0.00             0.00      0.00         0.00          0.00        1.00       0.00         0.00     0.00            0.00
13 
... [4491 more chars truncated]
```

```python  # cell 12
# ============================================================================
# §5b  Ambient diagnostics -- is this cluster a cell type or is it soup?
# ============================================================================
# The genes that leak. Not a cell-type panel: these are simply the most abundant transcripts in a
# blood lane, so a droplet made of free mRNA is made of these.
SOUP_GENES = ["HBB", "HBA1", "HBA2", "HBD", "PPBP", "PF4", "LYZ", "S100A8", "S100A9",
              "IGKC", "IGHM", "JCHAIN", "MALAT1", "B2M", "TMSB4X", "FTL", "FTH1", "ACTB"]
# Mutually exclusive lineages. A real cell is one of these; a soup droplet is several at once.
EXCLUSIVE = {
    "T":        ["CD3E", "CD3D", "TRAC"],
    "B":        ["MS4A1", "CD79A", "CD79B"],
    "myeloid":  ["LYZ", "S100A8", "CD14", "FCN1"],
    "NK":       ["NKG7", "GNLY", "KLRD1"],
    "platelet": ["PPBP", "PF4", "GP9"],
    "erythro":  ["HBB", "HBD", "ALAS2"],
    "plasma":   ["MZB1", "JCHAIN", "DERL3"],
}
POS_FRAC = 0.25        # a lineage counts as present in a cluster above this detection rate
N_NULL_DRAWS = 25      # size-matched random subsets per cluster, for the soup_corr null
NULL_Q = 0.05          # a cluster must beat the 5th percentile of its own null

if resid is not None:
    _present = set(map(str, resid.var_names))
    _soup = [g for g in SOUP_GENES if g in _present]
    _key = "leiden_resid"
    _grp = resid.obs[_key].astype(str)

    # 1. soup_corr -- how much the cluster looks like the average of everything in the compartment
    _glob = np.asarray(sub.X.mean(axis=0)).ravel()
    _X = resid[:, :].X
    _X = np.asarray(_X.todense()) if hasattr(_X, "todense") else np.asarray(_X)
    _cm = pd.DataFrame(_X, index=_grp.to_numpy()).groupby(level=0).mean()
    from scipy.stats import spearmanr
    soup_corr = pd.Series({c: spearmanr(_cm.loc[c].to_numpy(), _glob).statistic
                           for c in _cm.index}, name="soup_corr")

    # 2. amb_frac -- share of each cell's RAW counts carried by the leaked genes
    _C = resid[:, _soup].layers["counts"]
    _C = np.asarray(_C.todense()) if hasattr(_C, "todense") else np.asarray(_C)
    _tot = np.asarray(resid.layers["counts"].sum(axis=1)).ravel()
    amb = pd.Series(_C.sum(1) / np.where(_tot == 0, 1, _tot), index=resid.obs_names)
    amb_frac = amb.groupby(_grp.to_numpy()).median().rename("amb_frac")

    # 3. n_lineages -- how many mutually exclusive lineages the cluster is positive for
    # Counted PER CELL, then averaged. The mean detection rate across a lineage's genes was the
    # wrong statistic: a soup droplet carries a little of everything, so it is positive for many
    # lineages at LOW level and that averages away to nothing.
    _lin, _percell = {}, []
    for lin, genes in EXCLUSIVE.items():
        gg = [g for g in genes if g in _present]
        if not gg:
            continue
        sx = resid[:, gg].X
        sx = np.asarray(sx.todense()) if hasattr(sx, "todense") else np.asarray(sx)
        pos = (sx > 0).any(1)                       # this cell shows this lineage at all
        _percell.append(pos)
        _lin[lin] = pd.Series((sx > 0).mean(1), index=resid.obs_names).groupby(
            _grp.to_numpy()).mean()
    lin_df = pd.DataFrame(_lin)
    n_lineages = (pd.Series(np.vstack(_percell).sum(0), index=resid.obs_names)
                  .groupby(_grp.to_numpy()).mean().rename("n_lineages"))

    qc2 = A.cluster_qc(resid, _key, donor_key="donor", study_key="study")
    gesk = (pd.crosstab(_grp, resid.obs["study"].astype(str), normalize="index")
            .get("geskin2026", pd.Series(0.0, index=qc2.index)).rename("geskin2026_frac"))

    # 4. The null. soup_corr has no meaningful absolute scale: a random sample of the pool IS
    # the pool average, so structureless clusters sit near the ceiling (~0.94 measured on this
    # object). A "high" cutoff would therefore flag everything, and a median cutoff flags half
    # the clusters whatever the truth is. So calibrate against size-matched random subsets: a
    # real cell type sits BELOW the null because it is distinct; soup sits at or above it.
    _rng = np.random.default_rng(SEED)
    _null = {}
    for c, n in resid.obs[_key].astype(str).value_counts().items():
        draws = []
        for _ in range(N_NULL_DRAWS):
            pick = _rng.choice(resid.n_obs, size=int(n), replace=False)
            draws.append(spearmanr(_X[pick].mean(0), _glob).statistic)
        _null[c] = float(np.quantile(draws, NULL_Q))
    null_corr = pd.Series(_null, name="soup_corr_null_p%d" % int(NULL_Q * 100))

    amb_tbl = (qc2[["n", "n_donors", "top_study", "top_study_frac", "med_genes"]]
               .join(gesk).join(soup_corr).join(null_corr).join(amb_frac).join(n_lineages)
               .join(lin_df.round(2)))
    amb_tbl["soup_vs_null"] = amb_tbl["soup_corr"] - amb_tbl[null_corr.name]
    amb_tbl = amb_tbl.sort_values("soup_vs_null", ascending=False)

    print("=" * 115)
    print("AMBIENT DIAGNOSTICS  (sorted by soup_vs_null)")
    print(f"  soup_corr vs a size-matched random null ({N_NULL_DRAWS} draws, p{int(NULL_Q*100)}).")
    print("  soup_vs_null >= 0  -> as unstructured as a random draw: ambient candidate")
    print("  soup_vs_null <  0  -> distinct from the pool average: a real population")
    print("=" * 115)
    print(amb_tbl.round(3).to_string())

    # soup_vs_null is the calibrated statistic and carries the call on its own. amb_frac and
    # n_lineages corroborate; they are NOT additional AND gates. Requiring all three fired on
    # nothing in testing even when a synthetic soup cluster was planted in the pool.
    _susp = amb_tbl[amb_tbl["soup_vs_null"] >= 0]
    _watch = amb_tbl[(amb_tbl["soup_vs_null"] < 0) & (amb_tbl["soup_vs_null"] >= -0.01)]
    _lowcx = amb_tbl[(amb_tbl["amb_frac"] >= 0.3) | (amb_tbl["med_genes"].astype(float) <= 600)]
    print(f"\nambient candidates (soup_vs_null >= 0): {list(_susp.index) or 'none'}")
    if len(_watch):
        print(f"borderline (within 0.01 of the null): {list(_watch.index)}")
    if len(_lowcx):
        print(f"high leaked-count share or very low complexity: {list(_lowcx.index)} "
              "-- check these even if they clear the null")
    print("These are a finding, not a failure. Labelling them Ambient_soup keeps fabricated "
          "cells out of the off-tumour denominator that 43_target_screen_blood reads.")
    amb_tbl.to_csv(TAB_DIR / "nb31b_residual_ambient.csv")
    print("\nwrote", TAB_DIR / "nb31b_residual_ambient.csv")
```

```
===================================================================================================================
AMBIENT DIAGNOSTICS  (sorted by soup_vs_null)
  soup_corr vs a size-matched random null (25 draws, p5).
  soup_vs_null >= 0  -> as unstructured as a random draw: ambient candidate
  soup_vs_null <  0  -> distinct from the pool average: a real population
===================================================================================================================
       n n_donors     top_study top_study_frac med_genes  geskin2026_frac  soup_corr  soup_corr_null_p5  amb_frac  n_lineages     T     B  myeloid    NK  platelet  erythro  plasma  soup_vs_null
17    12        1    geskin2026            1.0      2197            1.000      0.755              0.709     0.169       3.250  0.72  0.00     0.67  0.78      0.03     0.03    0.03         0.045
18  1248       15    geskin2026           0.98      1739            0.979      0.941              0.949     0.148       3.560  0.53  0.16     0.77  0.58      0.02     0.07    0.02        -0.008
9    403       11    geskin2026           0.99      2034            0.988      0.913              0.927     0.133       3.968  0.56  0.39     0.75  0.66      0.01     0.03    0.06        -0.014
41    99        6    geskin2026            1.0      1786            1.000      0.861              0.878     0.142       3.596  0.55  0.28     0.66  0.48      0.01     0.02    0.04        -0.017
26   294       11    geskin2026            1.0      1731            0.997      0.900              0.919     0.144       3.412  0.60  0.15     0.72  0.61      0.01     0.02    0.02        -0.019
36   122        8    geskin2026           0.99      1684            0.992      0.868              0.889     0.140       3.262  0.55  0.08     0.75  0.54      0.01     0.04    0.01        -0.022
28   471       10    geskin2026            1.0      2062            0.998      0.908              0.931     0.136       3.870  0.53  0.34     0.73  0.68      0.03     0.03    0.04        -0.022
14  2750       11    geskin2026            1.0      2025            0.998      0.937              0.960     0.150       3.614  0.61  0.18     0.80  0.69      0.02     0.04    0.03        -0.023
10   411        7    geskin2026            1.0      1952            0.998      0.905              0.929     0.146       3.657  0.60  0.21     0.75  0.68      0.02     0.03    0.03        -0.024
19   953       10    geskin2026            1.0      1796            0.996      0.919              0.945     0.143       3.632  0.60  0.23     0.73  0.66      0.01     0.03    0.03        -0.025
30   666        8    geskin2026            1.0      1963            0.998      0.912              0.938     0.139       3.616  0.60  0.21     0.75  0.70      0.02     0.02    0.03        -0.026
2   1649       52    geskin2026           0.98      1709            0.982      0.927              0.953     0.228       3.406  0.52  0.06     0.49  0.34      0.02     0.31    0.02        -0.026
21   950       14    geskin2026           0.99      1636            0.995      0.916              0.944     0.133       3.446  0.57  0.13     0.65  0.68      0.02     0.03    0.02        -0.027
43    69       11    geskin2026           0.97      1917            0.971      0.834              0.862     0.108       3.391  0.38  0.19     0.50  0.75      0.01     0.01    0.05        -0.028
27   276        6    geskin2026            1.0      1758            1.000      0.889              0.917     0.138       3.370  0.55  0.08     0.72  0.64      0.04     0.03    0.02        -0.028
4   2069       11    geskin2026            1.0      1784            0.996      0.928              0.957     0.150       3.592  0.53  0.20     0.74  0.65      0.02     0.03    0.02        -0.028
20   803       12    geskin2026            1.0      1818            0.995      0.913              0.942     0.133       3.407  0.57  0.12     0.68  0.72      0.02     0.02    0.02        -0.029
12  1659       10    g
... [5640 more chars truncated]
```

```python  # cell 13
# ============================================================================
# §5c  Round-2 markers: the populations round 1 has no panel for
# ============================================================================
# Round 1's panel is deliberately broad. These are the ones that are genuinely hard in blood --
# rare, low-complexity, or defined by a handful of genes that HVG selection tends to drop.
PANELS2 = {
    "neutrophil":   ["FCGR3B", "CSF3R", "CXCR2", "S100A12", "IFITM2", "PROK2"],
    "eosinophil":   ["CLC", "SIGLEC8", "PRG2", "EPX", "IL5RA"],
    "basophil":     ["CPA3", "MS4A2", "HDC", "GATA2", "CLC"],
    "mast":         ["TPSAB1", "TPSB2", "CPA3", "KIT"],
    "ILC":          ["KIT", "IL1RL1", "RORC", "AHR", "LTB", "IL7R"],
    "NK_bright":    ["NCAM1", "XCL1", "GZMK", "SELL", "KLRC1"],
    "NK_dim":       ["FCGR3A", "FGFBP2", "GZMB", "KLRF1", "PRF1"],
    "pDC":          ["LILRA4", "IRF7", "TCF4", "PLD4", "SERPINF1"],
    "ASDC":         ["AXL", "SIGLEC6", "PPP1R14A"],
    "cDC1":         ["CLEC9A", "XCR1", "CADM1", "BATF3"],
    "cDC2":         ["CD1C", "FCER1A", "CLEC10A", "FCGR2B"],
    "mono_interm":  ["CD14", "FCGR3A", "HLA-DRA", "CD74"],
    "megakaryo":    ["PF4", "PPBP", "TUBB1", "GP9", "ITGA2B", "NRGN", "CAVIN2"],
    "erythroid":    ["HBB", "HBD", "ALAS2", "GYPA", "AHSP", "SLC4A1"],
    "HSPC":         ["CD34", "SPINK2", "PRSS57", "CYTL1", "EGFL7", "NPR3"],
    "pre_B":        ["VPREB1", "DNTT", "MME", "IGLL1"],
    "plasma":       ["MZB1", "JCHAIN", "DERL3", "XBP1", "TNFRSF17"],
    "prolif":       ["MKI67", "TOP2A", "STMN1", "TUBA1B"],
    "doublet_T":    ["CD3E", "CD3D", "TRAC"],
    "soup":         SOUP_GENES,
}

VOCAB2 = ["Neutrophil", "Eosinophil", "Basophil", "Mast", "ILC", "NK_CD56bright", "NK_CD56dim",
          "pDC", "ASDC", "cDC1", "cDC2", "Mono_intermediate", "Megakaryocyte", "Erythroid",
          "HSPC", "pre_B", "Plasma", "Proliferating", "Doublet", "LowQC", "Ambient_soup", "Unk"]

if resid is not None:
    panels2, DROPPED2 = A.present_panels(resid.var_names, PANELS2, label="blood residual")
    _gone2 = [c for c in PANELS2 if c not in panels2]
    if _gone2:
        print(f"\n!! no genes left for: {_gone2} -- those populations are NOT callable here. "
              "Leave such a cluster Unk; do not promote it on a neighbouring panel.")

    GENES2 = list(dict.fromkeys(g for gs in panels2.values() for g in gs))
    mean2, frac2 = A.cluster_expression(resid, "leiden_resid", GENES2)

    print(A.paste_block(resid, "leiden_resid", panels2, mean2, frac2, qc2, VOCAB2,
                        title="blood residual (round-1 Unk), leiden res %.1f" % LEIDEN_RES2,
                        extra_cols=()))

    print("\n" + "=" * 100)
    print("DOTPLOT VALUES - mean log1p CP10K per residual cluster x gene")
    print("=" * 100)
    print(mean2.to_string())
    print("\n" + "=" * 100)
    print("DOTPLOT VALUES - fraction detected")
    print("=" * 100)
    print(frac2.to_string())

    _long2 = (mean2.stack().rename("mean_log1p_cp10k").to_frame()
              .join(frac2.stack().rename("frac_detected"))
              .rename_axis(["leiden_resid", "gene"]).reset_index())
    _long2["panel"] = _long2["gene"].map({g: c for c, gs in panels2.items() for g in gs})
    _long2.to_csv(TAB_DIR / "nb31b_residual_marker_values.csv", index=False)
    print("\nwrote", TAB_DIR / "nb31b_residual_marker_values.csv")
    print("\nRead each block against its row in the ambient table above. A cluster with no "
          "convincing panel AND ambient diagnostics -> Ambient_soup, not a forced label.")
```

```
blood residual: dropped from eosinophil: ['SIGLEC8']
====================================================================================================
blood residual (round-1 Unk), leiden res 1.5 — 35,623 cells, 44 clusters; values are `mean log1p CP10K / fraction detected`
allowed labels: ['Neutrophil', 'Eosinophil', 'Basophil', 'Mast', 'ILC', 'NK_CD56bright', 'NK_CD56dim', 'pDC', 'ASDC', 'cDC1', 'cDC2', 'Mono_intermediate', 'Megakaryocyte', 'Erythroid', 'HSPC', 'pre_B', 'Plasma', 'Proliferating', 'Doublet', 'LowQC', 'Ambient_soup', 'Unk']
====================================================================================================

--- cluster 0  n=708  donors=85  studies=10  top_study=geskin2026(0.35)  med_genes=1631  med_mito=3.68  doublet=0.077
    neutrophil       FCGR3B 0.00/0.00  CSF3R 0.05/0.03  CXCR2 0.00/0.00  S100A12 0.15/0.08  IFITM2 1.88/0.77  PROK2 0.01/0.00
    eosinophil       CLC 0.07/0.02  PRG2 0.00/0.00  EPX 0.00/0.00  IL5RA 0.02/0.01
    basophil         CPA3 0.02/0.01  MS4A2 0.03/0.01  HDC 0.05/0.02  GATA2 0.10/0.05  CLC 0.07/0.02
    mast             TPSAB1 0.04/0.02  TPSB2 0.01/0.01  CPA3 0.02/0.01  KIT 0.03/0.02
    ILC              KIT 0.03/0.02  IL1RL1 0.00/0.00  RORC 0.02/0.01  AHR 0.31/0.19  LTB 3.05/0.93  IL7R 1.87/0.74
    NK_bright        NCAM1 0.03/0.02  XCL1 0.10/0.05  GZMK 0.16/0.07  SELL 1.52/0.65  KLRC1 0.04/0.03
    NK_dim           FCGR3A 0.15/0.08  FGFBP2 0.06/0.03  GZMB 0.28/0.13  KLRF1 0.02/0.01  PRF1 0.24/0.13
    pDC              LILRA4 0.00/0.00  IRF7 0.52/0.29  TCF4 0.24/0.13  PLD4 0.01/0.00  SERPINF1 0.04/0.02
    ASDC             AXL 0.01/0.01  SIGLEC6 0.00/0.00  PPP1R14A 0.06/0.03
    cDC1             CLEC9A 0.00/0.00  XCR1 0.00/0.00  CADM1 0.05/0.03  BATF3 0.02/0.02
    cDC2             CD1C 0.01/0.01  FCER1A 0.05/0.03  CLEC10A 0.02/0.01  FCGR2B 0.08/0.05
    mono_interm      CD14 0.09/0.05  FCGR3A 0.15/0.08  HLA-DRA 1.01/0.42  CD74 2.55/0.83
    megakaryo        PF4 0.59/0.22  PPBP 1.14/0.30  TUBB1 0.58/0.21  GP9 0.37/0.16  ITGA2B 0.22/0.10  NRGN 0.82/0.25  CAVIN2 0.26/0.11
    erythroid        HBB 0.37/0.14  HBD 0.02/0.01  ALAS2 0.01/0.00  GYPA 0.01/0.00  AHSP 0.00/0.00  SLC4A1 0.00/0.00
    HSPC             CD34 0.04/0.02  SPINK2 0.08/0.05  PRSS57 0.07/0.03  CYTL1 0.02/0.01  EGFL7 0.06/0.03  NPR3 0.02/0.01
    pre_B            VPREB1 0.00/0.00  DNTT 0.00/0.00  MME 0.02/0.01  IGLL1 0.00/0.00
    plasma           MZB1 0.10/0.05  JCHAIN 0.20/0.07  DERL3 0.06/0.04  XBP1 0.63/0.34  TNFRSF17 0.01/0.01
    prolif           MKI67 0.04/0.02  TOP2A 0.06/0.04  STMN1 0.60/0.32  TUBA1B 1.59/0.68
    doublet_T        CD3E 2.37/0.88  CD3D 1.87/0.77  TRAC 0.82/0.44
    soup             HBB 0.37/0.14  HBA1 0.03/0.01  HBA2 0.07/0.03  HBD 0.02/0.01  PPBP 1.14/0.30  PF4 0.59/0.22  LYZ 1.05/0.36  S100A8 0.73/0.28  S100A9 0.70/0.28  IGKC 0.20/0.07  IGHM 0.14/0.07  JCHAIN 0.20/0.07  MALAT1 5.34/1.00  B2M 5.53/1.00  TMSB4X 4.92/1.00  FTL 3.96/0.98  FTH1 4.49/1.00  ACTB 4.86/1.00

--- cluster 1  n=693  donors=74  studies=8  top_study=buus2025(0.4)  med_genes=1545  med_mito=3.07  doublet=0.056
    neutrophil       FCGR3B 0.01/0.01  CSF3R 0.01/0.01  CXCR2 0.01/0.01  S100A12 0.01/0.01  IFITM2 2.93/0.92  PROK2 0.02/0.01
    eosinophil       CLC 0.00/0.00  PRG2 0.00/0.00  EPX 0.00/0.00  IL5RA 0.00/0.00
    basophil         CPA3 0.00/0.00  MS4A2 0.00/0.00  HDC 0.00/0.00  GATA2 0.00/0.00  CLC 0.00/0.00
    mast             TPSAB1 0.00/0.00  TPSB2 0.00/0.00  CPA3 0.00/0.00  KIT 0.54/0.26
    ILC              KIT 0.54/0.26  IL1RL1 0.00/0.00  RORC 0.05/0.03  AHR 0.75/0.37  LTB 2.32/0.75  IL7R 2.06/0.75
    NK_bright        NCAM1 0.08/0.05  XCL1 1.96/0.57  GZMK 0.24/0.11  SELL 1.18/0.47  KLRC1 0.56/0.26
    NK_dim           FCGR3A 0.16/0.08  FGFBP2 0.19/0.08  GZMB 0.86/0.30  KLRF1 0.15/0.08  PRF1 1.28/0.54
    pDC              LILRA4 0.00/0.00  IRF7 0.33/0.19  TCF4 0.10/0.06  PLD4 0.01/0.01  SERPINF1 0.01/0.01
    ASDC             AXL 0.00/0.00  SIGLEC6 0.00/0.00  PPP1R14A 0.00/0.00
    cDC1         
... [161064 more chars truncated]
```

```python  # cell 14
# ============================================================================
# §5d  YOUR ROUND-2 MAP -- then merge into subtype_fine and rewrite the sidecar
# ============================================================================
RESID_MAP = nb31c_residual_labels ={
    "0": "Doublet",
    "1": "NK_CD56bright",
    "2": "Ambient_soup",
    "3": "Unk",
    "4": "Ambient_soup",
    "5": "Unk",
    "6": "Unk",
    "7": "Unk",
    "8": "Ambient_soup",
    "9": "Ambient_soup",
    "10": "Ambient_soup",
    "11": "Ambient_soup",
    "12": "Ambient_soup",
    "13": "Ambient_soup",
    "14": "Ambient_soup",
    "15": "Ambient_soup",
    "16": "Ambient_soup",
    "17": "Unk",
    "18": "Ambient_soup",
    "19": "Ambient_soup",
    "20": "Ambient_soup",
    "21": "Ambient_soup",
    "22": "Ambient_soup",
    "23": "Ambient_soup",
    "24": "Unk",
    "25": "Ambient_soup",
    "26": "Ambient_soup",
    "27": "Ambient_soup",
    "28": "Ambient_soup",
    "29": "LowQC",
    "30": "Ambient_soup",
    "31": "Ambient_soup",
    "32": "Unk",
    "33": "Unk",
    "34": "Neutrophil",
    "35": "Neutrophil",
    "36": "Ambient_soup",
    "37": "ILC",
    "38": "LowQC",
    "39": "Doublet",
    "40": "Ambient_soup",
    "41": "Ambient_soup",
    "42": "Ambient_soup",
    "43": "Proliferating",
}
SIZES2 = {'0': 708, '1': 693, '2': 1649, '3': 1000, '4': 2069, '5': 179, '6': 738, '7': 965, '8': 452, '9': 403, '10': 411, '11': 692, '12': 1659, '13': 2194, '14': 2750, '15': 1465, '16': 2771, '17': 12, '18': 1248, '19': 953, '20': 803, '21': 950, '22': 202, '23': 623, '24': 230, '25': 1011, '26': 294, '27': 276, '28': 471, '29': 203, '30': 666, '31': 1110, '32': 1289, '33': 98, '34': 1699, '35': 89, '36': 122, '37': 81, '38': 1686, '39': 93, '40': 355, '41': 99, '42': 93, '43': 69}

if resid is None:
    print("no residual pool -- nothing to merge")
elif not RESID_MAP:
    print("RESID_MAP is empty -- fill it from §5b/§5c. Nothing was written.")
    print("Paste-ready template:")
    for c, n in resid.obs["leiden_resid"].value_counts().sort_index().items():
        print(f'    "{c}": "",   # n={n}')
    print("\nSIZES2:")
    print("    " + ", ".join(f'"{c}": {n}'
                             for c, n in resid.obs["leiden_resid"].value_counts()
                             .sort_index().items()))
else:
    A.check_map(resid, "leiden_resid", RESID_MAP, SIZES2, VOCAB2,
                label="blood residual", out_col="subtype_fine")

    COARSE2 = {
        "Neutrophil": "Granulocyte", "Eosinophil": "Granulocyte", "Basophil": "Granulocyte",
        "Mast": "Granulocyte", "ILC": "ILC", "NK_CD56bright": "NK", "NK_CD56dim": "NK",
        "pDC": "DC", "ASDC": "DC", "cDC1": "DC", "cDC2": "DC",
        "Mono_intermediate": "Monocyte", "Megakaryocyte": "Platelet", "Erythroid": "Erythroid",
        "HSPC": "HSPC", "pre_B": "B", "Plasma": "Plasma", "Proliferating": "Proliferating",
        "Doublet": "Doublet", "LowQC": "LowQC", "Ambient_soup": "Ambient_soup", "Unk": "Unk",
    }
    resid.obs["subtype_coarse"] = resid.obs["subtype_fine"].map(COARSE2).astype(str)

    # Merge: round-2 labels overwrite Unk only. Anything round 2 could not call stays Unk, which
    # is the honest outcome and is what keeps it out of the off-tumour denominator.
    lab2 = resid.obs["subtype_fine"].astype(str)
    co2 = resid.obs["subtype_coarse"].astype(str)
    fine = sub.obs["subtype_fine"].astype(str).copy()
    coarse = sub.obs["subtype_coarse"].astype(str).copy()
    rnd = pd.Series("round1", index=sub.obs_names)
    fine.loc[lab2.index] = lab2.to_numpy()
    coarse.loc[co2.index] = co2.to_numpy()
    rnd.loc[lab2.index] = "round2"
    sub.obs["subtype_fine"] = pd.Categorical(fine)
    sub.obs["subtype_coarse"] = pd.Categorical(coarse)
    sub.obs["annotation_round"] = pd.Categorical(rnd)

    still = int(fine.eq("Unk").sum())
    soup = int(fine.eq("Ambient_soup").sum())
    print(f"\nafter round 2: {still:,} still Unk | {soup:,} Ambient_soup | "
          f"{int(rnd.eq('round2').sum()):,} relabelled")
    print(sub.obs["subtype_fine"].value_counts().to_string())

    frame = pd.DataFrame({
        "cell_id": sub.obs_names.astype(str),
        "subtype_fine": sub.obs["subtype_fine"].astype(str),
        "subtype_coarse": sub.obs["subtype_coarse"].astype(str),
        "residual_source": sub.obs["cell_type_broad"].astype(str),
        "annotation_round": sub.obs["annotation_round"].astype(str),
    })
    prov = {
        "notebook": "30_blood/31b_subtype_reannotation.ipynb",
        "input": str(MRVI_IN), "broad_labels": str(BROAD_CSV),
        "method_round1": f"scVI n_latent={N_LATENT} batch=study -> leiden res={LEIDEN_RES}",
        "method_round2": f"re-embedded Unk residue -> leiden res={LEIDEN_RES2}",
        "n_cells": int(len(frame)), "seed": SEED,
        "vocab_round1": VOCAB, "vocab_round2": VOCAB2,
        "still_unk": still, "ambient_soup": soup,
        "ambient_table": str(TAB_DIR / "nb31b_residual_ambient.csv"),
    }
    A.write_sidecar(frame, SUBTYPE_CSV, PROV_JSON, prov, expect_rows=len(frame))
    print("\nrewrote", SUBTYPE_CSV)
```

```
after round 2: 4,511 still Unk | 25,791 Ambient_soup | 35,623 relabelled
subtype_fine
Mono_classical       33355
Ambient_soup         25791
LowQC                22262
B_naive              18141
Doublet              13736
Erythroid            12960
Unk                   4511
Mono_nonclassical     3243
B_memory              2818
Neutrophil            1788
HSPC                  1442
cDC2                  1021
NK_CD56bright          693
Plasma                 598
Platelet               466
pDC                    465
ILC                     81
Proliferating           69
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/blood_subtypes.csv  (143440, 5)
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/blood_subtypes_provenance.json

rewrote /home/projects/nyosef/zvise/MF/data/atlas_joint/blood_subtypes.csv
```

## Next

`43_target_screen_blood` §4 picks this sidecar up automatically and folds the non-T blood
compartments into the off-tumour axis.


