# 20_skin/22b_unassigned_reannotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/22b_unassigned_reannotation.ipynb` and is never run._

# 22b — The unassigned skin compartment: neutrophils, NK and ILC (atlas **v2**)

`21_reannotation` and `22_myeloid_fibro_reannotation` between them leave ~140k skin cells whose
label is either *nothing* or *demonstrably a placeholder*. This notebook re-embeds that residue
as one pool and annotates it, because the cell types hiding in it — neutrophils, NK, ILC — are
exactly the ones neither upstream notebook had markers for.

## Why this pool, specifically

| source | n | what is actually known about it |
|---|---|---|
| skin `Unk` | 38,280 | **one** `leiden_broad` cluster (`"5"`), labelled `Unk` in nb21 §7 and then never revisited: no dot plot, no sub-clustering, no markdown |
| myeloid `subtype_fine == "UNK"` | 15,399 | nb22 §7d residue after the UNK pool was re-clustered once |
| fibroblast `subtype_fine == "UNK"` | 30,874 | same |
| the NK-like block called `CD8` | 55,877 | `skin_T_unk_leiden` clusters 0,1,2,4,5,6 |

**≈140,430 cells.**

That last row is the uncomfortable one. nb21 §11 sent `leiden_T` cluster 10 (58,792 cells, Li2024
label `Tc`) to `UNK`, re-clustered it into 8 sub-clusters, and mapped six of them to `CD8` via
`cluster2ct_unk`. Its marker panel *did* carry an NK row (`NCAM1, KLRD1, KLRF1`) — the row was
plotted and never given a label, and no markdown in the notebook justifies or even mentions the
decision. So every NK and ILC cell in this atlas is currently called CD8. Including the block
here is the only way to find out how many; carrying `residual_source` on every cell is what makes
the relabelling visible and reversible.

## What is missing upstream, and why it matters here

`21_reannotation` contains **zero** granulocyte markers — no `FCGR3B`, `CSF3R`, `CXCR2`, `ELANE`,
`MPO`, `S100A8/A9` anywhere in the notebook — and no ILC-specific panel. So neither the `Unk`
cluster nor the 144,009 myeloid cells were ever *asked* whether they contained neutrophils.
`22_myeloid_fibro_reannotation` finds 306, in one Leiden cluster, across 2 donors.

Droplet 10x under-captures neutrophils (low RNA content, high RNase, they lyse in the protocol),
so "there are almost none" is a perfectly possible true answer. §6 reports the coverage either
way and does not manufacture a level: if neutrophils do not clear 25 cells in 5 MF donors,
`28_tme_degradome` keeps them as report-only and reads the granulocyte programme across myeloid
subsets instead.

## Run order

| § | what | cost | stop |
|---|---|---|---|
| 0–1 | setup, the unified label, the residual pool | CPU | |
| 2 | build the pool object off `joint_mrvi_input_skin.h5ad` | heavy I/O | |
| 3 | HVG → scVI → UMAP → Leiden | **GPU** — 1 training | |
| 4 | marker panels, incl. the three nb21 never had | CPU | |
| 5 | the numeric dump — paste into an LLM | CPU | |
| 6 | fill `CLUSTER2FINE` + `CLUSTER_SIZES` | by hand | **← stop** |
| 7 | UMAPs, coarse layer, cache, reconcile | instant | |
| 8 | the neutrophil verdict | instant | |

Shares `helpers/annot_utils.py` with nb22, so the two notebooks differ only in their panels and
their vocabulary.

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc, json, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from natsort import natsorted

warnings.filterwarnings("ignore")


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
import annot_utils as AU                      # shared with nb22

OUT = NB_DIR / "data" / "atlas_joint"
FIG = NB_DIR / "figures" / "unassigned_reannotation"; FIG.mkdir(parents=True, exist_ok=True)
TAB = NB_DIR / "tables"; TAB.mkdir(exist_ok=True)
MODELS = NB_DIR / "models"; MODELS.mkdir(exist_ok=True)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"

TAG = "resid"

# ---------------------------------------------------------------- inputs
SRC_H5AD     = OUT / "joint_mrvi_input_skin.h5ad"          # 1,345,527 x 10,000
LABEL_CSV    = OUT / "skin_cell_type_final.csv"            # nb21
SUBTYPE_CSV  = OUT / "skin_myeloid_fibro_subtypes.csv"     # nb22 §10
T_UNK_LEIDEN = OUT / "skin_T_unk_leiden.csv"               # nb21 §11.5
OBS_PARQUET  = OUT / "atlas_obs_full_v2.parquet"

# ---------------------------------------------------------------- caches
SUBSET_H5AD = OUT / f"skin_{TAG}_subset.h5ad"
PATHS = dict(hvg=OUT / f"skin_{TAG}_hvg.json",
             scvi=MODELS / f"skin_{TAG}_scvi",
             umap=OUT / f"skin_{TAG}_umap.npy",
             leiden=OUT / f"skin_{TAG}_leiden.csv",
             deg=TAB / f"skin_{TAG}_cluster_markers.csv",
             expr=TAB / f"skin_{TAG}_cluster_expression.csv",
             qc=TAB / f"skin_{TAG}_cluster_qc.csv")

# ---------------------------------------------------------------- outputs
UNASSIGNED_CSV = OUT / "skin_unassigned_subtypes.csv"      # read by degradome_data
PROV_JSON      = OUT / "skin_unassigned_provenance.json"

# ---------------------------------------------------------------- parameters
N_SKIN = 1_345_527
# nb21 §11.5 mapped these six sub-clusters of the UNK-T pool to CD8. That is the call this
# notebook re-opens; clusters 3 and 7 went to CD4 and are left alone.
NK_LIKE_CLUSTERS = ["0", "1", "2", "4", "5", "6"]
N_HVG, N_LATENT = 2000, 20
HVG_MIN_CELLS_PER_BATCH = 1000
LEIDEN_RES = 1.0
MIN_CELLS, MIN_SAMPLES = 25, 5             # the claim gate, copied from degradome_data
CTCL_DISEASES = ["MF"]
KEY = f"leiden_{TAG}"

print("NB_DIR", NB_DIR)
print("inputs:", "  ".join(f"{p.name}={'OK' if p.exists() else 'MISSING'}"
                           for p in (SRC_H5AD, LABEL_CSV, SUBTYPE_CSV, T_UNK_LEIDEN)))
print("caches:", "  ".join(f"{k}={'OK' if v.exists() else 'MISSING'}"
                           for k, v in PATHS.items() if k in ("hvg", "scvi", "umap", "leiden")))
```

```
NB_DIR /home/projects/nyosef/zvise/MF
inputs: joint_mrvi_input_skin.h5ad=OK  skin_cell_type_final.csv=OK  skin_myeloid_fibro_subtypes.csv=OK  skin_T_unk_leiden.csv=OK
caches: hvg=OK  scvi=MISSING  umap=MISSING  leiden=MISSING
```

## §1 · The unified label, and the residual pool

One `cell_type_v2` over all 1,345,527 skin cells: the nb21 broad/T label, overwritten by the
nb22 subtype wherever the sidecar has one. Then the pool is everything that label cannot
actually name, tagged with `residual_source` so no two sources are silently merged.

```python  # cell 3
# ============================================================================
# §1  cell_type_v2 + the residual pool
# ============================================================================
ct = pd.read_csv(LABEL_CSV, dtype=str).set_index("cell_id")          # nb21
assert len(ct) == N_SKIN, (len(ct), N_SKIN)

if not SUBTYPE_CSV.exists():
    raise FileNotFoundError(f"{SUBTYPE_CSV} missing — run 22_myeloid_fibro_reannotation §10")
side = pd.read_csv(SUBTYPE_CSV, dtype=str).set_index("cell_id")      # nb22
tunk = pd.read_csv(T_UNK_LEIDEN, dtype=str).set_index("cell_id")     # nb21 §11.5

lab = ct["cell_type_final"].copy()
lab.loc[side.index] = side["subtype_fine"]
assert lab.notna().all()

src = pd.Series(pd.NA, index=lab.index, dtype=object)
src[lab == "Unk"] = "broad_Unk"
src[lab.index.isin(side.index[side["subtype_fine"] == "UNK"]) &
    (side["lineage"].reindex(lab.index) == "Myeloid")] = "myeloid_UNK"
src[lab.index.isin(side.index[side["subtype_fine"] == "UNK"]) &
    (side["lineage"].reindex(lab.index) == "Fibroblast")] = "fibroblast_UNK"
nk_ids = tunk.index[tunk["leiden_unk"].isin(NK_LIKE_CLUSTERS)]
src[lab.index.isin(nk_ids)] = "CD8_nk_like"

pool_ids = src.dropna().index
print(f"cell_type_v2 over {len(lab):,} skin cells, {lab.nunique()} levels")
print(f"\nresidual pool: {len(pool_ids):,} cells")
print(src.dropna().value_counts().to_string())
assert src.dropna().index.is_unique
```

```
cell_type_v2 over 1,345,527 skin cells, 30 levels

residual pool: 140,430 cells
CD8_nk_like       55877
broad_Unk         38280
fibroblast_UNK    30874
myeloid_UNK       15399
```

```python  # cell 4
# ============================================================================
# §2  Build the pool object off joint_mrvi_input_skin.h5ad   (heavy I/O, first run only)
# ============================================================================
if SUBSET_H5AD.exists():
    sub = sc.read_h5ad(SUBSET_H5AD)
else:
    print(f"reading {SRC_H5AD.name} (16.4 GB) — first run only ...")
    _a = sc.read_h5ad(SRC_H5AD)
    assert _a.n_obs == N_SKIN and "raw_counts" in _a.layers
    assert _a.obs["cell_id"].is_unique
    m = _a.obs["cell_id"].astype(str).isin(set(pool_ids))
    sub = _a[m].copy()
    del _a; gc.collect()
    sub.layers["counts"] = sub.layers.pop("raw_counts")
    ids = sub.obs["cell_id"].astype(str)
    sub.obs["residual_source"] = pd.Categorical(src.reindex(ids).to_numpy())
    sub.obs["prior_label"] = pd.Categorical(lab.reindex(ids).to_numpy())
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sub.write_h5ad(SUBSET_H5AD, compression="gzip")
    print(f"wrote {SUBSET_H5AD.name}  {sub.shape}")

assert sub.n_obs == len(pool_ids), (sub.n_obs, len(pool_ids))
print(f"pool object: {sub.shape}")
print(sub.obs["residual_source"].value_counts().to_string())
```

```
pool object: (140430, 10000)
residual_source
CD8_nk_like       55877
broad_Unk         38280
fibroblast_UNK    30874
myeloid_UNK       15399
```

## §3 · Embed and cluster — **GPU**

Same recipe as nb22 §2b (HVG → scVI → neighbours → UMAP → Leiden, `batch_key="study"`,
`donor` as a categorical covariate, seed 0), so the two notebooks' clusterings are comparable
and a cell that moves between them moved for a reason.

Everything is frozen to disk on the first run; re-runs load and **verify** rather than trust the
row order — the UMAP is stored positionally, so §3 checks that one cluster really is tighter
than the cloud before it believes the `.npy` belongs to this clustering.

```python  # cell 6
# ============================================================================
# §3  HVG -> scVI -> UMAP -> Leiden   (first run TRAINS — GPU / bsub, not the login node)
# ============================================================================
def rebuild(a):
    """HVG -> scVI -> neighbors -> UMAP -> Leiden, and freeze all of it. GPU."""
    import scvi, torch
    scvi.settings.seed = SEED
    print(f"rebuilding — cuda={torch.cuda.is_available()}")

    if PATHS["hvg"].exists():
        hvg = json.loads(PATHS["hvg"].read_text())["genes"]
    else:
        n_by = a.obs["study"].value_counts()
        keep_studies = n_by[n_by >= HVG_MIN_CELLS_PER_BATCH].index.tolist()
        _fit = a[a.obs["study"].isin(keep_studies)].copy()
        sc.pp.highly_variable_genes(_fit, flavor="seurat_v3", n_top_genes=N_HVG,
                                    batch_key="study", layer="counts")
        hvg = _fit.var_names[_fit.var["highly_variable"].to_numpy()].tolist()
        del _fit; gc.collect()
        PATHS["hvg"].write_text(json.dumps(
            {"pool": "skin_residual", "n_top_genes": N_HVG, "flavor": "seurat_v3",
             "batch_key": "study", "hvg_studies": keep_studies, "seed": SEED,
             "genes": hvg}, indent=1))

    ahv = a[:, hvg].copy()
    scvi.model.SCVI.setup_anndata(ahv, layer="counts", batch_key="study",
                                  categorical_covariate_keys=["donor"])
    if PATHS["scvi"].exists():
        model = scvi.model.SCVI.load(str(PATHS["scvi"]), adata=ahv)
    else:
        model = scvi.model.SCVI(ahv, n_latent=N_LATENT, n_layers=2, gene_likelihood="nb")
        model.train(early_stopping=True, early_stopping_patience=15,
                    plan_kwargs={"lr": 1e-3}, check_val_every_n_epoch=1)
        model.save(str(PATHS["scvi"]), overwrite=True)
    a.obsm["X_scVI"] = model.get_latent_representation()
    del ahv, model; gc.collect()

    sc.pp.neighbors(a, use_rep="X_scVI", random_state=SEED)
    sc.tl.umap(a, random_state=SEED)
    np.save(PATHS["umap"], a.obsm["X_umap"])
    sc.tl.leiden(a, resolution=LEIDEN_RES, random_state=SEED, key_added=KEY,
                 flavor="igraph", n_iterations=2, directed=False)
    pd.DataFrame({"cell_id": a.obs["cell_id"].astype(str).to_numpy(),
                  KEY: a.obs[KEY].astype(str).to_numpy()}).to_csv(PATHS["leiden"], index=False)
    print(f"rebuilt: {a.obs[KEY].nunique()} clusters")


if not (PATHS["umap"].exists() and PATHS["leiden"].exists()):
    rebuild(sub)
else:
    s = pd.read_csv(PATHS["leiden"], dtype=str).set_index("cell_id")[KEY]
    ids = sub.obs["cell_id"].astype(str)
    assert len(s) == sub.n_obs and ids.isin(s.index).all(), (
        f"{PATHS['leiden'].name} does not cover the pool ({len(s):,} vs {sub.n_obs:,})")
    vals = s.reindex(ids).to_numpy()
    sub.obs[KEY] = pd.Categorical(vals, categories=natsorted(pd.unique(vals)))

    u = np.load(PATHS["umap"])
    assert u.shape[0] == sub.n_obs, (u.shape, sub.n_obs)
    sub.obsm["X_umap"] = u
    lab_ = sub.obs[KEY].astype(str).to_numpy()
    big = pd.Series(lab_).value_counts().idxmax()
    m = lab_ == big
    spread = np.median(np.linalg.norm(u[m] - u[m].mean(0), axis=1))
    overall = np.median(np.linalg.norm(u - u.mean(0), axis=1))
    assert spread < 0.6 * overall, (
        f"UMAP does not match the Leiden (cluster {big} spread {spread:.2f} vs "
        f"{overall:.2f}) — the .npy row order is stale, delete it and rebuild")
    print(f"loaded: {sub.obs[KEY].nunique()} clusters, UMAP OK "
          f"(spread {spread:.2f} vs {overall:.2f})")

print(sub.obs[KEY].value_counts().reindex(natsorted(sub.obs[KEY].cat.categories)).to_string())
```

```
Seed set to 0

rebuilding — cuda=True

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
You are using a CUDA device ('NVIDIA A10') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Epoch 57/57: 100%|██████████| 57/57 [08:48<00:00,  9.31s/it, v_num=1, train_loss_step=342, train_loss_epoch=337]
`Trainer.fit` stopped: `max_epochs=57` reached.

Epoch 57/57: 100%|██████████| 57/57 [08:48<00:00,  9.28s/it, v_num=1, train_loss_step=342, train_loss_epoch=337]
rebuilt: 53 clusters
leiden_resid
0      3380
1       111
2     11798
3       517
4     13986
5      8928
6      9862
7      7855
8      4395
9      9321
10      617
11      124
12       36
13        9
14     3098
15     6621
16     2967
17       30
18       23
19       26
20      121
21       51
22       20
23     1980
24       19
25      278
26       29
27       18
28     6714
29       50
30       33
31     1094
32       53
33      533
34       43
35       19
36       16
37     1637
38       28
39       26
40     1588
41     8631
42     7152
43    12907
44      508
45      543
46     8818
47      782
48     2734
49       22
50       25
51       76
52      178
```

## §4 · Marker panels — including the three nb21 never had

`Neutrophil`, `NK` and `ILC` are new to this project. The rest are carried from nb21's `markers`
and nb22's `OFF_MARKERS` so a cluster can be called a doublet, a pericyte or ambient rather than
being forced into an identity.

`joint_mrvi_input_skin.h5ad` is a **10,000-HVG** matrix, so a panel can lose its defining genes
silently. `AU.present_panels` prints every drop, and the cell below hard-fails if the neutrophil
panel loses too much: neutrophil identity rests on exactly those granule genes, and calling a
cluster `Neutrophils` on whatever survived an HVG filter is not an annotation.

Measured against this object before the notebook was written: **neutrophil 14/15** (only `PRTN3`
drops) and **NK 11/11** survive, so both are safely callable. **ILC keeps 7/10 but loses `RORC`,
`AHR` and `TBX21`** — its three lineage-defining transcription factors. What remains (`IL7R`,
`KIT`, `IL1RL1`, `KLRB1`, `GATA3`, `LTB`, `AREG`) separates ILC2 reasonably but cannot tell ILC1
from ILC3. So call `ILC` as one level here and do **not** subset it; a finer split needs the
full-gene source.

```python  # cell 8
# ============================================================================
# §4  Marker panels
# ============================================================================
MARKERS = {
    # --- the three that did not exist anywhere in this project before ---
    "Neutrophil":   ["FCGR3B", "CSF3R", "CXCR1", "CXCR2", "S100A8", "S100A9", "S100A12",
                     "ELANE", "PRTN3", "MPO", "CTSG", "LTF", "CAMP", "LCN2", "SELL"],
    "NK":           ["NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY", "PRF1", "GZMB", "FCGR3A",
                     "TYROBP", "KLRC1", "EOMES"],
    "ILC":          ["IL7R", "KIT", "RORC", "AHR", "IL1RL1", "KLRB1", "TBX21", "GATA3",
                     "LTB", "AREG"],
    # --- carried from nb21 / nb22 so a non-identity stays callable ---
    "T core":       ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC2"],
    "CD4/CD8":      ["CD4", "IL7R", "CD8A", "CD8B"],
    "Treg":         ["FOXP3", "CTLA4", "IL2RA"],
    "Cytotoxic":    ["GZMA", "GZMH", "GZMK", "PRF1"],
    "B/plasma":     ["MS4A1", "CD79A", "MZB1", "JCHAIN", "IGKC"],
    "Myeloid":      ["PTPRC", "LYZ", "AIF1", "HLA-DRA", "CD68", "ITGAX", "C1QA", "C1QB"],
    "DC/LC":        ["CD207", "CD1A", "CD1C", "CLEC9A", "LAMP3", "LILRA4", "IL3RA"],
    "Mast":         ["TPSAB1", "TPSB2", "CPA3", "KIT", "CMA1"],
    "Fibroblast":   ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Keratinocyte": ["KRT14", "KRT5", "KRT1", "KRT10", "DMKN", "KRTDAP"],
    "Endothelial":  ["PECAM1", "VWF", "CDH5", "PROX1", "CCL21", "ACKR1"],
    "Pericyte/SMC": ["RGS5", "NOTCH3", "PDGFRB", "KCNJ8", "MYH11", "ACTA2"],
    "Melanocyte":   ["MLANA", "PMEL", "TYRP1", "DCT"],
    "Neural":       ["PLP1", "S100B", "NRXN1"],
    "Erythroid":    ["HBB", "HBA1", "HBA2"],
    "Platelet":     ["PPBP", "PF4", "ITGA2B"],
    "Prolif":       ["MKI67", "TOP2A", "STMN1"],
}

PANEL, DROPPED = AU.present_panels(sub.var_names, MARKERS, label="pool")

_n_neu = len(PANEL.get("Neutrophil", []))
if _n_neu < 6:
    raise ValueError(
        f"only {_n_neu} of {len(MARKERS['Neutrophil'])} neutrophil markers survived the 10k-HVG "
        f"filter ({DROPPED.get('Neutrophil')}). Neutrophil identity rests on exactly those "
        f"genes — rebuild the pool object from the full-gene source "
        f"(data/atlas_joint/joint_annotated.h5ad) rather than calling a cluster on what is left.")
print(f"\nneutrophil panel: {_n_neu}/{len(MARKERS['Neutrophil'])} genes retained")

VOCAB = ["Neutrophils", "NK", "ILC",
         "LC", "cDC1", "cDC2", "DC_LAMP3", "pDC", "Mono_CD14", "Mono_CD16", "moDC",
         "Mac_FOLR2", "Mac_SPP1_TREM2", "Mac_infl",
         "F_papillary", "F_reticular", "F_mesenchymal", "F_inflammatory", "F_apCAF",
         "CD4", "CD4_Treg", "CD8", "B", "Plasma", "Mast", "Keratinocyte", "Vascular",
         "Melanocyte", "Pericyte", "Neural", "Erythroid", "Platelet",
         "Prolif", "Doublet", "LowQuality", "UNK"]
print(f"vocabulary: {len(VOCAB)} labels")
```

```
pool: dropped from Neutrophil: ['PRTN3']
  pool: dropped from ILC: ['RORC', 'AHR', 'TBX21']
  pool: dropped from Platelet: ['ITGA2B']

neutrophil panel: 14/15 genes retained
vocabulary: 36 labels
```

```python  # cell 9
# ============================================================================
# §4b  Leiden UMAP, and where each residual source landed
# ============================================================================
AU.save_umap(sub, [KEY, "residual_source", "study", "disease"],
             FIG / f"{TAG}_umap_leiden.png", ncols=4, wspace=0.25,
             title=[f"Leiden (res={LEIDEN_RES}, {sub.obs[KEY].nunique()} clusters)",
                    "residual_source", "study", "disease"])

print(pd.crosstab(sub.obs[KEY].astype(str), sub.obs["residual_source"].astype(str))
      .reindex(natsorted(sub.obs[KEY].cat.categories)).to_string())

AU.save_dotplot(sub, PANEL, KEY, FIG / f"{TAG}_dotplot_leiden.png",
                title="residual pool — marker panels by cluster", figsize=(20, 7))
```

```
<Figure size 3520x528 with 4 Axes>
residual_source  CD8_nk_like  broad_Unk  fibroblast_UNK  myeloid_UNK
leiden_resid                                                        
0                          6          3             147         3224
1                          0          2             103            6
2                         39          4             898        10857
3                          0          9             504            4
4                      13896          0              43           47
5                       8672          0             173           83
6                       9677          1             122           62
7                       7640          1             136           78
8                       4373          0              11           11
9                       9194          0              81           46
10                         1          0              16          600
11                         0          0              92           32
12                         0          0              36            0
13                         0          0               0            9
14                         2         32            3061            3
15                         0         19            6592           10
16                       719         24            2212           12
17                         0          0              30            0
18                         0          0              23            0
19                         0          0              26            0
20                         0          1             116            4
21                         0          0              51            0
22                         0          0              20            0
23                         0         19            1961            0
24                         0          0              19            0
25                         0          0             278            0
26                         0          0              29            0
27                         0          0              18            0
28                        79         17            6430          188
29                         0          0              50            0
30                         0          0              33            0
31                         2          5            1087            0
32                         0          0              53            0
33                         0        277             256            0
34                         0          0              43            0
35                         0          1              18            0
36                         0          0              16            0
37                      1574          1              18           44
38                         0          0              28            0
39                         0          0              26            0
40                         3         18            1566            1
41                         0       8334             297            0
42                         0       7084              68            0
43                         0      12596             311            0
44                         0        497              11            0
45                         0         12             531            0
46                         0       8536             280            2
47                         0        753              29            0
48                         0         34            2699            1
49                         0          0              22            0
50                         0          0              25            0
51                         0          0               1           75
52                         0          0             178            0
WARNING: dendrogram data not found (using key=dendrogram_leiden_resid). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.t
... [296 more chars truncated]
```

## §5 · The numbers — copy-paste this into an LLM

Per cluster: size, donor spread, study dominance, QC, which residual source it came from, mean
log1p expression and detected fraction for every panel gene, and the Wilcoxon top-15. Nothing is
thresholded away — a marker that is absent prints as `0.00/0.00`, which is evidence too.

The same numbers land in `tables/skin_resid_cluster_expression.csv` if you would rather work from
the file.

```python  # cell 11
# ============================================================================
# §5  Per-cluster expression / QC tables + the paste block
# ============================================================================
genes = list(dict.fromkeys(g for v in PANEL.values() for g in v))
mean, frac = AU.cluster_expression(sub, KEY, genes)
qc = AU.cluster_qc(sub, KEY)

# which residual source dominates each cluster — the column that says whether a relabelling is
# touching the never-examined Unk cluster or the CD8 block
srcmix = pd.crosstab(sub.obs[KEY].astype(str), sub.obs["residual_source"].astype(str),
                     normalize="index").round(2)
qc["top_source"] = srcmix.idxmax(axis=1).reindex(qc.index)
qc["top_source_frac"] = srcmix.max(axis=1).reindex(qc.index)

if not PATHS["deg"].exists():
    sc.tl.rank_genes_groups(sub, KEY, method="wilcoxon", n_genes=15)
    pd.DataFrame(sub.uns["rank_genes_groups"]["names"]).to_csv(PATHS["deg"], index=False)
top = pd.read_csv(PATHS["deg"], dtype=str)
TOPN = {c: ", ".join(top[c].dropna().tolist()) for c in top.columns}

mean.to_csv(PATHS["expr"]); qc.to_csv(PATHS["qc"])
print(AU.paste_block(sub, KEY, PANEL, mean, frac, qc, VOCAB,
                     "RESIDUAL SKIN POOL", extra_cols=("top_source", "top_source_frac"),
                     top_n=TOPN))
```

```
====================================================================================================
RESIDUAL SKIN POOL — 140,430 cells, 53 clusters; values are `mean log1p CP10K / fraction detected`
allowed labels: ['Neutrophils', 'NK', 'ILC', 'LC', 'cDC1', 'cDC2', 'DC_LAMP3', 'pDC', 'Mono_CD14', 'Mono_CD16', 'moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'F_papillary', 'F_reticular', 'F_mesenchymal', 'F_inflammatory', 'F_apCAF', 'CD4', 'CD4_Treg', 'CD8', 'B', 'Plasma', 'Mast', 'Keratinocyte', 'Vascular', 'Melanocyte', 'Pericyte', 'Neural', 'Erythroid', 'Platelet', 'Prolif', 'Doublet', 'LowQuality', 'UNK']
====================================================================================================

--- cluster 0  n=3,380  donors=132  studies=14  top_study=il4ra2026(0.24)  med_genes=1108  med_mito=4.25  doublet=nan
    top_source: myeloid_UNK
    top_source_frac: 0.95
    Neutrophil       FCGR3B 0.00/0.00  CSF3R 0.15/0.07  CXCR1 0.00/0.00  CXCR2 0.01/0.01  S100A8 1.11/0.32  S100A9 0.94/0.29  S100A12 0.03/0.01  ELANE 0.00/0.00  MPO 0.00/0.00  CTSG 0.01/0.00  LTF 0.16/0.06  CAMP 0.01/0.01  LCN2 0.00/0.00  SELL 0.14/0.07
    NK               NCAM1 0.07/0.04  KLRD1 0.09/0.04  KLRF1 0.01/0.01  NKG7 0.42/0.15  GNLY 0.48/0.14  PRF1 0.14/0.07  GZMB 0.35/0.12  FCGR3A 0.08/0.04  TYROBP 1.46/0.48  KLRC1 0.08/0.04  EOMES 0.03/0.01
    ILC              IL7R 0.70/0.27  KIT 0.03/0.01  IL1RL1 0.00/0.00  KLRB1 0.11/0.05  GATA3 0.31/0.16  LTB 0.83/0.31  AREG 0.70/0.25
    T core           CD3D 0.43/0.18  CD3E 0.47/0.20  CD3G 0.22/0.11  TRAC 0.16/0.08  TRBC2 0.26/0.12
    CD4/CD8          CD4 0.39/0.20  IL7R 0.70/0.27  CD8A 0.11/0.05  CD8B 0.04/0.02
    Treg             FOXP3 0.03/0.02  CTLA4 0.09/0.04  IL2RA 0.17/0.07
    Cytotoxic        GZMA 0.26/0.09  GZMH 0.07/0.03  GZMK 0.15/0.06  PRF1 0.14/0.07
    B/plasma         MS4A1 0.03/0.01  CD79A 0.02/0.01  MZB1 0.01/0.01  JCHAIN 0.02/0.01  IGKC 0.05/0.02
    Myeloid          PTPRC 1.46/0.53  LYZ 2.32/0.60  AIF1 1.01/0.39  HLA-DRA 3.73/0.76  CD68 0.59/0.27  ITGAX 0.56/0.21  C1QA 0.54/0.20  C1QB 0.55/0.21
    DC/LC            CD207 0.11/0.03  CD1A 0.30/0.12  CD1C 0.43/0.17  CLEC9A 0.33/0.16  LAMP3 0.33/0.13  LILRA4 0.03/0.02  IL3RA 0.36/0.13
    Mast             TPSAB1 0.02/0.01  TPSB2 0.01/0.01  CPA3 0.01/0.00  KIT 0.03/0.01  CMA1 0.00/0.00
    Fibroblast       COL1A1 0.13/0.06  COL1A2 0.10/0.04  DCN 0.19/0.09  LUM 0.03/0.01  PDGFRA 0.01/0.01
    Keratinocyte     KRT14 1.78/0.46  KRT5 1.21/0.35  KRT1 1.05/0.31  KRT10 1.35/0.43  DMKN 0.78/0.26  KRTDAP 0.68/0.24
    Endothelial      PECAM1 0.22/0.12  VWF 0.02/0.01  CDH5 0.01/0.01  PROX1 0.01/0.00  CCL21 0.06/0.03  ACKR1 0.03/0.01
    Pericyte/SMC     RGS5 0.04/0.02  NOTCH3 0.08/0.04  PDGFRB 0.03/0.01  KCNJ8 0.01/0.00  MYH11 0.05/0.02  ACTA2 0.11/0.04
    Melanocyte       MLANA 0.01/0.01  PMEL 0.06/0.02  TYRP1 0.03/0.01  DCT 0.02/0.01
    Neural           PLP1 0.01/0.00  S100B 0.92/0.29  NRXN1 0.01/0.00
    Erythroid        HBB 0.32/0.09  HBA1 0.02/0.01  HBA2 0.04/0.01
    Platelet         PPBP 0.01/0.00  PF4 0.00/0.00
    Prolif           MKI67 0.17/0.08  TOP2A 0.15/0.07  STMN1 0.62/0.25
    wilcoxon top     HLA-DRA, CD74, CD83, HLA-DRB1, HLA-DPB1, HLA-DQA1, HLA-DPA1, LYZ, SAT1, HLA-DQB1, NEAT1, SPI1, CPVL, IFI30, CST3

--- cluster 1  n=111  donors=8  studies=5  top_study=chennareddy2025(0.77)  med_genes=1135  med_mito=7.45  doublet=nan
    top_source: fibroblast_UNK
    top_source_frac: 0.93
    Neutrophil       FCGR3B 0.05/0.03  CSF3R 0.05/0.03  CXCR1 0.02/0.01  CXCR2 0.03/0.02  S100A8 0.79/0.28  S100A9 0.52/0.20  S100A12 0.09/0.05  ELANE 0.00/0.00  MPO 0.00/0.00  CTSG 0.00/0.00  LTF 0.83/0.32  CAMP 0.00/0.00  LCN2 0.00/0.00  SELL 0.53/0.23
    NK               NCAM1 0.00/0.00  KLRD1 0.06/0.04  KLRF1 0.00/0.00  NKG7 1.93/0.64  GNLY 0.77/0.27  PRF1 0.34/0.14  GZMB 0.32/0.14  FCGR3A 0.00/0.00  TYROBP 0.68/0.28  KLRC1 0.09/0.04  EOMES 0.08/0.03
    ILC              IL7R 1.70/0.59  KIT 0.08/0.03  IL1RL1 0.00/0.00  KLRB1 1.24/0.42  GATA3 0.
... [131373 more chars truncated]
```

## §6 · The annotation dict — **hand-filled, stop here**

Fill `CLUSTER2FINE` from §5, then paste the printed sizes into `CLUSTER_SIZES`. The size lock is
not ceremony: Leiden is not stable across a re-embedding, and a map whose keys still resolve but
whose clusters have moved relabels cells silently. `AU.check_map` turns that into an error.

`UNK` is a real answer. So are `Doublet` and `LowQuality` — a cluster that is 90% one donor with
a median of 400 genes and a doublet score of 0.4 is not a cell type, and forcing it to be one is
how the NK block ended up called CD8 in the first place.

```python  # cell 13
# ============================================================================
# §6  cluster -> subtype_fine   (HAND-FILLED from §5)
# ============================================================================
CLUSTER2FINE =  {
        "0": "UNK",         "1": "UNK",          "2": "UNK",          "3": "Vascular",
        "4": "CD8",         "5": "CD8",          "6": "CD8",          "7": "CD8",
        "8": "CD8",         "9": "CD8",          "10": "DC_LAMP3",    "11": "UNK",
        "12": "UNK",        "13": "UNK",         "14": "Vascular",    "15": "F_inflammatory",
        "16": "UNK",        "17": "UNK",         "18": "UNK",         "19": "UNK",
        "20": "Mac_SPP1_TREM2", "21": "UNK",     "22": "UNK",         "23": "UNK",
        "24": "UNK",        "25": "UNK",         "26": "UNK",         "27": "UNK",
        "28": "UNK",        "29": "UNK",         "30": "UNK",         "31": "UNK",
        "32": "Plasma",     "33": "UNK",         "34": "UNK",         "35": "UNK",
        "36": "Mast",       "37": "CD8",         "38": "UNK",         "39": "Vascular",
        "40": "Vascular",   "41": "Vascular",    "42": "Vascular",    "43": "Vascular",
        "44": "Vascular",   "45": "UNK",         "46": "Vascular",    "47": "Vascular",
        "48": "UNK",        "49": "UNK",         "50": "Vascular",    "51": "UNK",
        "52": "UNK",
    }          # e.g. {"0": "NK", "1": "Neutrophils", "2": "Doublet", ...}
CLUSTER_SIZES = {'0': 3380, '1': 111, '2': 11798, '3': 517, '4': 13986, '5': 8928, '6': 9862, '7': 7855, '8': 4395, '9': 9321, '10': 617, '11': 124, '12': 36, '13': 9, '14': 3098, '15': 6621, '16': 2967, '17': 30, '18': 23, '19': 26, '20': 121, '21': 51, '22': 20, '23': 1980, '24': 19, '25': 278, '26': 29, '27': 18, '28': 6714, '29': 50, '30': 33, '31': 1094, '32': 53, '33': 533, '34': 43, '35': 19, '36': 16, '37': 1637, '38': 28, '39': 26, '40': 1588, '41': 8631, '42': 7152, '43': 12907, '44': 508, '45': 543, '46': 8818, '47': 782, '48': 2734, '49': 22, '50': 25, '51': 76, '52': 178}         # paste the dict the UNFILLED branch prints

MAPPED, _sizes = AU.check_map(sub, KEY, CLUSTER2FINE, CLUSTER_SIZES, VOCAB,
                              "residual pool", out_col="subtype_fine")
if MAPPED:
    print(sub.obs["subtype_fine"].value_counts().to_string())
    print("\nwhere each label came from:")
    print(pd.crosstab(sub.obs["subtype_fine"].astype(str),
                      sub.obs["residual_source"].astype(str)).to_string())
else:
    print("\nFill the dicts above, then re-run from here — §7 onwards needs subtype_fine.")
```

```
subtype_fine
CD8               55984
Vascular          44052
UNK               32966
F_inflammatory     6621
DC_LAMP3            617
Mac_SPP1_TREM2      121
Plasma               53
Mast                 16

where each label came from:
residual_source  CD8_nk_like  broad_Unk  fibroblast_UNK  myeloid_UNK
subtype_fine                                                        
CD8                    55026          3             584          371
DC_LAMP3                   1          0              16          600
F_inflammatory             0         19            6592           10
Mac_SPP1_TREM2             0          1             116            4
Mast                       0          0              16            0
Plasma                     0          0              53            0
UNK                      845        398           17319        14404
Vascular                   5      37859            6178           10
```

## §7 · UMAPs, the coarse layer, cache, reconcile

Same four steps as nb22 §8–§11, so the two sidecars have the same shape and the same guarantees.

```python  # cell 15
# ============================================================================
# §7  Final UMAPs + the coarse layer
# ============================================================================
assert MAPPED, "fill CLUSTER2FINE in §6 first"

FINE_TO_COARSE = {
    "Neutrophils": "Granulocyte",
    "NK": "NK_ILC", "ILC": "NK_ILC",
    "LC": "DC", "cDC1": "DC", "cDC2": "DC", "DC_LAMP3": "DC", "pDC": "DC",
    "Mono_CD14": "Mono_Mac", "Mono_CD16": "Mono_Mac",
    "moDC": "Mono_Mac", "Mac_FOLR2": "Mono_Mac",
    "Mac_SPP1_TREM2": "Mono_Mac", "Mac_infl": "Mono_Mac",
    "F_papillary": "Fibroblast", "F_reticular": "Fibroblast", "F_mesenchymal": "Fibroblast",
    "F_inflammatory": "Fibroblast", "F_apCAF": "Fibroblast",
    "CD4": "T", "CD4_Treg": "T", "CD8": "T",
    "B": "B_Plasma", "Plasma": "B_Plasma",
    "Mast": "Mast", "Keratinocyte": "Keratinocyte", "Vascular": "Vascular",
    "Melanocyte": "Melanocyte", "Pericyte": "Mural", "Neural": "Neural",
    "Erythroid": "Other", "Platelet": "Other",
    "Prolif": "Prolif", "Doublet": "UNK", "LowQuality": "UNK", "UNK": "UNK",
}
seen = set(sub.obs["subtype_fine"].astype(str).unique())
missing = seen - set(FINE_TO_COARSE)
assert not missing, f"fine labels with no collapse rule: {sorted(missing)}"
sub.obs["subtype_coarse"] = pd.Categorical(
    sub.obs["subtype_fine"].astype(str).map(FINE_TO_COARSE))

AU.save_umap(sub, ["subtype_fine", "subtype_coarse", "residual_source"],
             FIG / f"{TAG}_umap_subtype.png", ncols=3, wspace=0.25,
             title=[f"subtype_fine ({sub.obs['subtype_fine'].nunique()})",
                    f"subtype_coarse ({sub.obs['subtype_coarse'].nunique()})",
                    "residual_source"])
AU.save_dotplot(sub, PANEL, "subtype_fine", FIG / f"{TAG}_dotplot_subtype.png",
                title="residual pool — panels by subtype_fine", figsize=(20, 6))
```

```
<Figure size 2640x528 with 3 Axes>
WARNING: dendrogram data not found (using key=dendrogram_subtype_fine). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: CD8, DC_LAMP3, F_inflammatory, etc.
var_group_labels: Neutrophil, NK, ILC, etc.

<Figure size 2200x660 with 7 Axes>
<scanpy.plotting._dotplot.DotPlot at 0x14efd9091700>
```

```python  # cell 16
# ============================================================================
# §8  Cache the sidecar + reconcile against the v2 atlas
# ============================================================================
resid = pd.DataFrame({
    "cell_id": sub.obs["cell_id"].astype(str).to_numpy(),
    "residual_source": sub.obs["residual_source"].astype(str).to_numpy(),
    "leiden_resid": sub.obs[KEY].astype(str).to_numpy(),
    "prior_label": sub.obs["prior_label"].astype(str).to_numpy(),
    "subtype_fine": sub.obs["subtype_fine"].astype(str).to_numpy(),
    "subtype_coarse": sub.obs["subtype_coarse"].astype(str).to_numpy(),
})

AU.write_sidecar(
    resid, UNASSIGNED_CSV, PROV_JSON,
    expect_rows=len(pool_ids),
    provenance={
        "notebook": "notebooks/20_skin/22b_unassigned_reannotation.ipynb",
        "atlas": "v2",
        "source_h5ad": str(SRC_H5AD), "subset_h5ad": str(SUBSET_H5AD),
        "label_source": str(LABEL_CSV), "subtype_source": str(SUBTYPE_CSV),
        "t_unk_leiden": str(T_UNK_LEIDEN), "nk_like_clusters": NK_LIKE_CLUSTERS,
        "seed": SEED, "n_hvg": N_HVG, "n_latent": N_LATENT,
        "hvg_min_cells_per_batch": HVG_MIN_CELLS_PER_BATCH,
        "scvi_batch_key": "study", "scvi_categorical_covariate_keys": ["donor"],
        "scvi_model": str(PATHS["scvi"]), "leiden_res": LEIDEN_RES,
        "cluster2fine": CLUSTER2FINE, "cluster_sizes": CLUSTER_SIZES,
        "fine_to_coarse": FINE_TO_COARSE,
        "pool_sources": src.dropna().value_counts().to_dict(),
        "fine_levels": sorted(resid["subtype_fine"].unique()),
    })

expect = resid["subtype_fine"].value_counts().to_dict()
AU.reconcile(resid, expect, obs_parquet=OBS_PARQUET, label_col="subtype_fine")

print()
print(pd.crosstab(resid["residual_source"], resid["subtype_fine"]).to_string())
```

```
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/skin_unassigned_subtypes.csv  (140430, 6)
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/skin_unassigned_provenance.json
reconciled 140,430 rows against atlas_obs_full_v2.parquet; 8 level counts match

subtype_fine       CD8  DC_LAMP3  F_inflammatory  Mac_SPP1_TREM2  Mast  Plasma    UNK  Vascular
residual_source                                                                                
CD8_nk_like      55026         1               0               0     0       0    845         5
broad_Unk            3         0              19               1     0       0    398     37859
fibroblast_UNK     584        16            6592             116    16      53  17319      6178
myeloid_UNK        371       600              10               4     0       0  14404        10
```

## §9 · The neutrophil verdict

The question this notebook exists to answer, stated against the gate `28_tme_degradome` actually
uses: **25 cells in 5 MF donors.**

Three outcomes, and the notebook is written to report whichever is true rather than to produce
the convenient one:

* *clears the gate* — `Neutrophils` becomes a claimable level and `ELANE`/`PRTN3`/`FCGR3B` are
  wired in as its positive control in nb28 §0;
* *found but below the gate* — reported with donor and study counts, carried as report-only, and
  nb28 reads the granulocyte programme across myeloid subsets instead;
* *not found* — droplet 10x under-captures granulocytes, and that is a true fact about this
  atlas rather than a failure of the search. Say so.

```python  # cell 18
# ============================================================================
# §9  Does Neutrophils clear the claim gate?
# ============================================================================
# reset_index is load-bearing: the parquet carries pandas index metadata whose index is NAMED
# cell_id, so pd.read_parquet returns a frame where cell_id is both an index level and a column
# and .merge(on="cell_id") raises "which is ambiguous". AU.reconcile above does not hit this
# because it reads through polars, which drops that metadata.
obs2 = (pd.read_parquet(OBS_PARQUET, columns=["cell_id", "donor", "study", "disease"])
        .reset_index(drop=True))
r = resid.merge(obs2, on="cell_id", how="left", validate="one_to_one")
assert r["donor"].notna().all()

report = []
for lvl in ["Neutrophils", "NK", "ILC"]:
    d = r[(r["subtype_fine"] == lvl) & r["disease"].astype(str).isin(CTCL_DISEASES)]
    if not len(d):
        report.append({"level": lvl, "n_cells": 0, "n_donors_ge25": 0, "n_donors": 0,
                       "n_studies": 0, "claimable": False, "from": ""})
        continue
    per_donor = d.groupby("donor", observed=True).size()
    report.append({
        "level": lvl, "n_cells": len(d),
        "n_donors": int(d["donor"].nunique()),
        "n_donors_ge25": int((per_donor >= MIN_CELLS).sum()),
        "n_studies": int(d["study"].nunique()),
        "claimable": bool((per_donor >= MIN_CELLS).sum() >= MIN_SAMPLES),
        "from": ", ".join(f"{k}:{v}" for k, v in
                          d["residual_source"].value_counts().items()),
    })
rep = pd.DataFrame(report)
rep.to_csv(TAB / f"skin_{TAG}_new_level_coverage.csv", index=False)
print(rep.to_string(index=False))

neu = rep[rep["level"] == "Neutrophils"].iloc[0]
print()
if neu["n_cells"] == 0:
    print("NEUTROPHILS: none recovered. Droplet 10x under-captures granulocytes (low RNA "
          "content, they lyse in the protocol); this is a fact about the atlas, not a failed "
          "search. nb28 reports the granulocyte programme across myeloid subsets instead.")
elif not neu["claimable"]:
    print(f"NEUTROPHILS: {int(neu['n_cells']):,} cells in {int(neu['n_donors'])} MF donors, "
          f"{int(neu['n_donors_ge25'])} of them with >= {MIN_CELLS} cells — below the "
          f"{MIN_CELLS}-cell / {MIN_SAMPLES}-donor gate. Carried as report-only: shown in nb28 "
          f"§9, never quoted as a fold change.")
else:
    print(f"NEUTROPHILS: {int(neu['n_cells']):,} cells clearing {MIN_CELLS} in "
          f"{int(neu['n_donors_ge25'])} MF donors across {int(neu['n_studies'])} studies — "
          f"CLEARS the gate. Add cfg.NEUTROPHIL_CONTROLS_CPM to the nb28 §0 positive controls "
          f"and rebuild the degradome object so the level enters the roster.")

print(f"\nnext: rebuild the degradome object so these levels enter the roster —")
print(f"  MEM_MB=96000 WALL_H=6 ./jobs/run_degradome_build.sh --force")
```

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_3066984/3231280438.py in ?()
      1 # ============================================================================
      2 # §9  Does Neutrophils clear the claim gate?
      3 # ============================================================================
      4 obs2 = pd.read_parquet(OBS_PARQUET, columns=["cell_id", "donor", "study", "disease"])
----> 5 r = resid.merge(obs2, on="cell_id", how="left", validate="one_to_one")
      6 assert r["donor"].notna().all()
      7 
      8 report = []

~/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/pandas/core/frame.py in ?(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10855         validate: MergeValidate | None = None,
  10856     ) -> DataFrame:
  10857         from pandas.core.reshape.merge import merge
  10858 
> 10
```
