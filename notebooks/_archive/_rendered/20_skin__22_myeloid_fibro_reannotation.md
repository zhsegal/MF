# 20_skin/22_myeloid_fibro_reannotation.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/22_myeloid_fibro_reannotation.ipynb` and is never run._

# 22 — Skin myeloid & fibroblast sub-annotation (atlas **v2**)

`21_reannotation` splits only the T compartment, so the CCC pipeline sees **Myeloid (144,009)** and
**Fibroblast (184,452)** as single pooled levels. This notebook splits them into states and writes
`skin_myeloid_fibro_subtypes.csv`, which `helpers/ccc_data_sub.py` and `helpers/degradome_data.py`
read on behalf of `27_ccc_subtype` and `28_tme_degradome`.

**This is a full rebuild on v2, not a re-run.** The v1 caches — a 160,195-cell pool, 48% of v2's
328,461 — were parked in `data/atlas_joint/_v1_myefib/` on 2026-09-17. Nothing is loaded from disk:
every embedding is retrained here, and both annotation dicts start empty and are filled by hand.

## Run order

| § | what | cost | stop |
|---|---|---|---|
| 0–1 | setup, marker panels, vocabulary | instant | |
| 2a | build the 328,461-cell subset off `joint_mrvi_input_skin.h5ad` (16.4 GB) | heavy I/O | |
| 2b | HVG → scVI → UMAP → Leiden, per lineage | **GPU** — 2 trainings | |
| 3–4 | Leiden UMAPs, marker dot plots | CPU | |
| 5 | the numeric dump — paste into an LLM | CPU | |
| 6 | fill `CLUSTER2FINE` + `CLUSTER_SIZES` | by hand | **← stop** |
| 7 | annotated UMAPs + labelled dot plot | CPU | |
| 7b–7c | re-cluster the UNK pool of both lineages, dump its numbers | **GPU** — 2 small trainings | |
| 7d | fill `UNK2FINE` + `UNK_CLUSTER_SIZES` | by hand | **← stop** |
| 8 | final UMAPs: `subtype_fine` vs the Leiden, study, disease | CPU | |
| 9 | `subtype_fine` → `subtype_coarse` + per-donor coverage of both | instant | |
| 10 | cache the sidecar + provenance | instant | |
| 11 | reconcile the sidecar against `atlas_obs_full_v2.parquet` | instant | |

§6 and §7d print a paste-ready template when their dict is empty and set `MAPPED = False`, which
§7 and §8 assert on — so the notebook halts at each hand-filled step instead of running past it.

**Monocytes are split into `Mono_CD14` / `Mono_CD16`** (clusters 15 and 22), matching the
classical/non-classical distinction the initial Reynolds-derived taxonomy makes and the single
pooled `Mono` level erased. `Mono_CD16` is 442 cells over 73 donors, so it is a **context level** —
plottable, not claimable — while `Mono_CD14` keeps the donor support `Mono` had. Both collapse to
`Mono_Mac` in §9, so nothing coarse changes.

**Labels are ontogeny/program states, not M1/M2.** M1/M2/TAM are scored as gradients in §5 and never
used as cluster names — they do not partition tissue macrophages. `UNK` is a real answer for a
doublet, pericyte or low-quality cluster.

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
OUT = NB_DIR / "data" / "atlas_joint"
FIG = NB_DIR / "figures" / "myefib_reannotation"; FIG.mkdir(parents=True, exist_ok=True)
TAB = NB_DIR / "tables"; TAB.mkdir(exist_ok=True)
MODELS = NB_DIR / "models"; MODELS.mkdir(exist_ok=True)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"

LINEAGES = ["Myeloid", "Fibroblast"]
TAGS = {"Myeloid": "mye", "Fibroblast": "fib"}

# ---------------------------------------------------------------- inputs
SRC_H5AD    = OUT / "joint_mrvi_input_skin.h5ad"     # 1,345,527 x 10,000 — read only if the subset is absent
LABEL_CSV   = OUT / "skin_cell_type_final.csv"       # 21_reannotation
SUBSET_H5AD = OUT / "skin_mye_fib_subset.h5ad"       # 328,461 x 10,000, layers['counts']

# ---------------------------------------------------------------- per-lineage caches
def paths(tag):
    return dict(hvg=OUT / f"skin_{tag}_hvg.json",
                scvi=MODELS / f"skin_{tag}_scvi",
                umap=OUT / f"skin_{tag}_umap.npy",
                leiden=OUT / f"skin_{tag}_leiden.csv",
                deg=TAB / f"skin_{tag}_cluster_markers.csv",
                expr=TAB / f"skin_{tag}_cluster_expression.csv",
                qc=TAB / f"skin_{tag}_cluster_qc.csv",
                cov=TAB / f"skin_{tag}_coverage_by_donor.csv")

# ---------------------------------------------------------------- outputs
SUBTYPE_CSV = OUT / "skin_myeloid_fibro_subtypes.csv"    # what nb38/39 read
PROV_JSON   = OUT / "skin_myeloid_fibro_provenance.json"

# ---------------------------------------------------------------- parameters
# --- v2 atlas contract ---------------------------------------------------------------------
# These are the v2 skin numbers (atlas rebuilt 2026-09-05). v1 was 749,510 / 93,435 / 66,760, and
# the v2 files replaced the v1 ones AT THE SAME PATHS — nothing in a filename tells you which you
# have, only the row count does. The v2 lineage pool is 328,461 cells, 2.05x the v1 one.
N_SKIN, N_MYE, N_FIB = 1_345_527, 144_009, 184_452
N_CELLS = {"Myeloid": N_MYE, "Fibroblast": N_FIB}
# Skin cohorts that exist only in the v2 build — the same content guard 21_reannotation uses, so a
# future in-place rebuild is caught by what is in the file and not by its size alone.
V2_SKIN_DATASETS = {"rindler21mc", "alkon24", "lyp26", "jonak21",
                    "brentuximab26", "gaydosik22", "gaydosik23", "il4ra26"}
N_HVG, N_LATENT = 2000, 20
HVG_MIN_CELLS_PER_BATCH = 1000          # seurat_v3's loess needs a usable batch
LEIDEN_RES = {"Myeloid": 1.0, "Fibroblast": 0.8}
CCC_MIN_CELLS, CCC_MIN_SAMPLES = 25, 5  # the claim gate, copied from ccc_data
CTCL_DISEASES = ["MF", "SS", "CTCL_other"]

print("caches:")
for lin in LINEAGES:
    p = paths(TAGS[lin])
    print(f"  {lin:11s} " + "  ".join(
        f"{k}={'OK' if v.exists() else 'MISSING'}" for k, v in p.items() if k in
        ("hvg", "scvi", "umap", "leiden")))
print(f"  subset      {'OK' if SUBSET_H5AD.exists() else 'MISSING'}")
```

```
caches:
  Myeloid     hvg=OK  scvi=OK  umap=OK  leiden=OK
  Fibroblast  hvg=OK  scvi=OK  umap=OK  leiden=OK
  subset      OK
```

```python  # cell 2
# ============================================================================
# §1  Marker panels, signatures, vocabulary
# ============================================================================
MARKERS = {
 "Myeloid": {
    "pan-myeloid":   ["PTPRC", "LYZ", "AIF1", "HLA-DRA", "CD68", "ITGAX"],
    "LC":            ["CD207", "CD1A", "EPCAM", "CLDN1", "PLEK2"],
    "cDC1":          ["CLEC9A", "XCR1", "WDFY4", "BATF3", "IRF8", "CADM1"],
    "cDC2":          ["CD1C", "CLEC10A", "FCER1A", "CD1B", "IRF4"],
    "DC_LAMP3":      ["LAMP3", "CCR7", "FSCN1", "CCL19", "CCL22", "IDO1", "BIRC3"],
    "pDC":           ["LILRA4", "IL3RA", "TCF4", "GZMB", "JCHAIN"],
    "Mono_CD14":     ["S100A8", "S100A9", "S100A12", "FCN1", "VCAN", "CD14"],
    "Mono_CD16":     ["FCGR3A", "CDKN1C", "LST1", "MS4A7", "CSF1R"],
    "moDC":          ["CD209", "IL1B", "EREG", "THBS1", "CD1C"],
    "Mac_FOLR2":     ["FOLR2", "LYVE1", "SELENOP", "C1QA", "C1QB", "C1QC", "MRC1", "CD163",
                      "F13A1", "STAB1", "MERTK"],
    "Mac_SPP1_TREM2": ["SPP1", "TREM2", "APOC1", "ACP5", "GPNMB", "MMP9", "MMP12", "CHI3L1"],
    "Mac_infl":      ["IL1B", "CXCL1", "CXCL2", "CXCL9", "CXCL10", "CXCL11", "TREM1", "CD300E",
                      "OLR1"],
    "Mac_ISG":       ["ISG15", "IFIT1", "IFI6"],
    "prolif":        ["MKI67", "TOP2A"],
    "!not-myeloid":  ["CD3D", "MS4A1", "KRT14", "COL1A1", "PECAM1", "TPSAB1"],
 },
 "Fibroblast": {
    "pan-fibro":      ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "F_papillary":    ["APCDD1", "ID1", "WIF1", "COL18A1", "PTGDS", "COL6A5", "COL23A1", "HSPB3",
                       "NTN1", "PDPN"],
    "F_reticular":    ["WISP2", "SLPI", "MFAP5", "TSPAN8", "MGP", "CD36", "FMO1", "MYOC", "THY1",
                       "PI16"],
    "F_mesenchymal":  ["ASPN", "POSTN", "GPC3", "SFRP1", "COMP", "COL11A1", "CTHRC1", "LRRC15",
                       "INHBA", "FAP", "TNC"],
    "F_inflammatory": ["CCL19", "APOE", "CXCL2", "CXCL3", "EFEMP1", "CXCL12", "CXCL13", "IL6",
                       "MMP1", "MMP3", "CXCL8", "IL24", "C3", "C7", "CFD"],
    "F_myofibro":     ["ACTA2", "TAGLN", "MYH11", "DES"],
    "F_apCAF":        ["HLA-DRA", "HLA-DQA1", "CD74"],
    "!pericyte/SMC":  ["RGS5", "NOTCH3", "PDGFRB", "KCNJ8"],
    "prolif":         ["MKI67", "TOP2A"],
    "!not-fibro":     ["PTPRC", "CD3D", "LYZ", "KRT14", "PECAM1", "MLANA"],
 },
}

# Graded signatures — reported as numbers in §5, never used as cluster names.
SIGNATURES = {
 "Myeloid": {
    "sig_M1":  ["CXCL9", "CXCL10", "CXCL11", "IL1B", "TNF", "NOS2", "CD86", "IDO1", "SOCS3"],
    "sig_M2":  ["CD163", "MRC1", "MSR1", "STAB1", "F13A1", "SELENOP", "CCL18", "TGFB1", "FOLR2"],
    "sig_TAM": ["SPP1", "TREM2", "GPNMB", "APOC1", "ACP5", "FABP5", "LGMN", "CTSB"],
    "sig_APC": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DQA1", "CD74", "CIITA"],
    "sig_angio": ["VEGFA", "THBS1", "ADM", "SLC2A1"],
 },
 "Fibroblast": {
    "sig_myoCAF": ["POSTN", "COMP", "COL11A1", "CTHRC1", "LRRC15", "FAP", "INHBA", "TNC", "ACTA2"],
    "sig_iCAF":   ["IL6", "CXCL1", "CXCL2", "CXCL8", "CXCL12", "CCL2", "MMP1", "MMP3", "PDGFRA"],
    "sig_apCAF":  ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "CD74", "CIITA"],
    "sig_ecm":    ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "FN1", "SPARC"],
 },
}

# The only labels §6 may use.
VOCAB = {
 "Myeloid": ["LC", "cDC1", "cDC2", "DC_LAMP3", "pDC", "Mono_CD14", "Mono_CD16", "moDC",
             "Mac_FOLR2",
             "Mac_SPP1_TREM2", "Mac_infl", "Mac_ISG", "Neutrophils", "Myeloid_prolif",
             "UNK"],
 "Fibroblast": ["F_papillary", "F_reticular", "F_mesenchymal", "F_inflammatory",
                "F_myofibroblast", "F_apCAF", "F_prolif", "UNK"],
}

for lin in LINEAGES:
    print(f"{lin}: {len(MARKERS[lin])} panels, {len(SIGNATURES[lin])} signatures, "
          f"{len(VOCAB[lin])} labels")
```

```
Myeloid: 15 panels, 5 signatures, 15 labels
Fibroblast: 10 panels, 4 signatures, 8 labels
```

## §2 · Load

`skin_mye_fib_subset.h5ad` (both lineages, 10k genes, raw counts in `layers['counts']`) plus the
cached per-lineage Leiden and UMAP. The Leiden is joined **on `cell_id`**; the UMAP is positional,
so §2b checks it actually agrees with the clustering rather than trusting row order.

On this v2 pass every cache is absent, so the `rebuild` branch (HVG → scVI → neighbors → UMAP →
Leiden) fires for both lineages. It is the heavy path — run it on a GPU kernel or via `bsub`,
never on the login node. Once it has run, the cells below re-read what it froze to disk.

```python  # cell 4
# ============================================================================
# §2a  Load the subset, log1p it, split by lineage
# ============================================================================
if SUBSET_H5AD.exists():
    sub_all = sc.read_h5ad(SUBSET_H5AD)
else:
    print(f"reading {SRC_H5AD.name} (16.4 GB) — first run only ...")
    _a = sc.read_h5ad(SRC_H5AD)
    assert _a.n_obs == N_SKIN and "raw_counts" in _a.layers
    _missing = V2_SKIN_DATASETS - set(_a.obs["dataset"].astype(str).unique())
    assert not _missing, f"v2 skin cohorts absent — this is the v1 build. missing: {sorted(_missing)}"
    assert _a.obs["cell_id"].is_unique
    lab = pd.read_csv(LABEL_CSV, dtype=str).set_index("cell_id")
    for c in ["cell_type_broad", "cell_type_final"]:
        _a.obs[c] = (lab[c].reindex(_a.obs["cell_id"].astype(str))
                     .astype(str).str.replace("Keartinocyte", "Keratinocyte").to_numpy())
    assert float((_a.obs["cell_type_final"] != "nan").mean()) > 0.999
    sub_all = _a[_a.obs["cell_type_final"].isin(LINEAGES).to_numpy()].copy()
    del _a; gc.collect()
    sub_all.layers["counts"] = sub_all.X.copy()
    del sub_all.layers["raw_counts"]
    for _k in list(sub_all.obsm):
        del sub_all.obsm[_k]
    sub_all.write_h5ad(SUBSET_H5AD, compression="gzip")
    print("wrote", SUBSET_H5AD, sub_all.shape)

_n = sub_all.obs["cell_type_final"].value_counts()
assert int(_n["Myeloid"]) == N_MYE and int(_n["Fibroblast"]) == N_FIB, _n.to_dict()

# X is raw counts in this file -> log1p CP10K for every plot and every number below
if float(sub_all.X[:2000].max()) > 30:
    sc.pp.normalize_total(sub_all, target_sum=1e4)
    sc.pp.log1p(sub_all)
assert float(sub_all.X.max()) < 20, "X does not look log1p-normalized"

A = {}
for lin in LINEAGES:
    a = sub_all[sub_all.obs["cell_type_final"] == lin].copy()
    for c in a.obs.columns:
        if str(a.obs[c].dtype) == "category":
            a.obs[c] = a.obs[c].cat.remove_unused_categories()
    assert a.n_obs == N_CELLS[lin], (a.n_obs, N_CELLS[lin])
    A[lin] = a
del sub_all; gc.collect()

for lin, a in A.items():
    print(f"{lin:11s} {a.n_obs:>7,} cells | {a.obs['donor'].nunique()} donors | "
          f"{a.obs['sample_id'].nunique()} samples | {a.obs['study'].nunique()} studies")
```

```
Myeloid     144,009 cells | 146 donors | 174 samples | 14 studies
Fibroblast  184,452 cells | 145 donors | 173 samples | 14 studies
```

```python  # cell 5
# ============================================================================
# §2b  Per-lineage embedding: rebuild when a cache is missing, else load and verify — GPU
# ============================================================================
def rebuild(a, lin):
    """HVG -> scVI -> neighbors -> UMAP -> Leiden, and freeze all of it. GPU."""
    import scvi, torch
    tag, p = TAGS[lin], paths(TAGS[lin])
    scvi.settings.seed = SEED
    print(f"[{lin}] rebuilding — cuda={torch.cuda.is_available()}")

    if p["hvg"].exists():
        hvg = json.loads(p["hvg"].read_text())["genes"]
    else:
        n_by = a.obs["study"].value_counts()
        keep_studies = n_by[n_by >= HVG_MIN_CELLS_PER_BATCH].index.tolist()
        _fit = a[a.obs["study"].isin(keep_studies)].copy()
        sc.pp.highly_variable_genes(_fit, flavor="seurat_v3", n_top_genes=N_HVG,
                                    batch_key="study", layer="counts")
        hvg = _fit.var_names[_fit.var["highly_variable"].to_numpy()].tolist()
        del _fit; gc.collect()
        p["hvg"].write_text(json.dumps(
            {"lineage": lin, "n_top_genes": N_HVG, "flavor": "seurat_v3", "batch_key": "study",
             "hvg_studies": keep_studies, "seed": SEED, "genes": hvg}, indent=1))

    ahv = a[:, hvg].copy()
    scvi.model.SCVI.setup_anndata(ahv, layer="counts", batch_key="study",
                                  categorical_covariate_keys=["donor"])
    if p["scvi"].exists():
        model = scvi.model.SCVI.load(str(p["scvi"]), adata=ahv)
    else:
        model = scvi.model.SCVI(ahv, n_latent=N_LATENT, n_layers=2, gene_likelihood="nb")
        model.train(early_stopping=True, early_stopping_patience=15,
                    plan_kwargs={"lr": 1e-3}, check_val_every_n_epoch=1)
        model.save(str(p["scvi"]), overwrite=True)
    a.obsm["X_scVI"] = model.get_latent_representation()
    del ahv, model; gc.collect()

    sc.pp.neighbors(a, use_rep="X_scVI", random_state=SEED)
    sc.tl.umap(a, random_state=SEED)
    np.save(p["umap"], a.obsm["X_umap"])
    sc.tl.leiden(a, resolution=LEIDEN_RES[lin], random_state=SEED, key_added=f"leiden_{tag}",
                 flavor="igraph", n_iterations=2, directed=False)
    pd.DataFrame({"cell_id": a.obs["cell_id"].astype(str).to_numpy(),
                  f"leiden_{tag}": a.obs[f"leiden_{tag}"].astype(str).to_numpy()}
                 ).to_csv(p["leiden"], index=False)
    print(f"[{lin}] rebuilt: {a.obs[f'leiden_{tag}'].nunique()} clusters")


KEY = {}
for lin in LINEAGES:
    tag, p = TAGS[lin], paths(TAGS[lin])
    a, KEY[lin] = A[lin], f"leiden_{tag}"

    if not (p["umap"].exists() and p["leiden"].exists()):
        rebuild(a, lin)
    else:
        # Leiden: joined on cell_id, so row order cannot corrupt it.
        s = pd.read_csv(p["leiden"], dtype=str).set_index("cell_id")[f"leiden_{tag}"]
        ids = a.obs["cell_id"].astype(str)
        assert len(s) == a.n_obs and ids.isin(s.index).all(), (
            f"{p['leiden'].name} does not cover {lin} ({len(s):,} vs {a.n_obs:,})")
        vals = s.reindex(ids).to_numpy()
        a.obs[KEY[lin]] = pd.Categorical(vals, categories=natsorted(pd.unique(vals)))

        # UMAP: stored positionally. Verify it belongs to THIS clustering rather than trusting
        # the row order — cells of one cluster must sit far tighter than the cloud as a whole.
        u = np.load(p["umap"])
        assert u.shape[0] == a.n_obs, (u.shape, a.n_obs)
        a.obsm["X_umap"] = u
        lab = a.obs[KEY[lin]].astype(str).to_numpy()
        big = pd.Series(lab).value_counts().idxmax()
        m = lab == big
        spread = np.median(np.linalg.norm(u[m] - u[m].mean(0), axis=1))
        overall = np.median(np.linalg.norm(u - u.mean(0), axis=1))
        assert spread < 0.6 * overall, (
            f"{lin}: UMAP does not match the Leiden (cluster {big} spread {spread:.2f} vs "
            f"{overall:.2f} overall) — the .npy row order is stale, delete it and rebuild")
        print(f"{lin:11s} loaded: {a.obs[KEY[lin]].nunique()} clusters, UMAP OK "
              f"(spread {spread:.2f} vs {overall:.2f})")

for lin in LINEAGES:
    print(f"\n{lin} cluster sizes:")
    print(A[lin].obs[KEY[lin]].value_counts()
          .reindex(natsorted(A[lin].obs[KEY[lin]].cat.categories)).to_string())
```

```
Myeloid     loaded: 25 clusters, UMAP OK (spread 1.68 vs 4.70)
Fibroblast  loaded: 36 clusters, UMAP OK (spread 1.55 vs 4.42)

Myeloid cluster sizes:
leiden_mye
0      4051
1     12468
2      8192
3      6981
4      3478
5     12447
6     14084
7      7646
8      1126
9      3063
10     1508
11     8332
12     2342
13     5866
14    12480
15     7187
16    13124
17      149
18     5110
19       19
20    10855
21     1411
22      442
23     1342
24      306

Fibroblast cluster sizes:
leiden_fib
0     15965
1      2003
2      6558
3     14329
4      1892
5      8510
6     15376
7     12775
8     20665
9     23744
10     1749
11    16399
12      183
13      109
14     9780
15      264
16     4358
17     6675
18     8466
19      897
20       31
21     3913
22       18
23      544
24     2858
25     3210
26       14
27       30
28      757
29       46
30     1308
31       14
32       30
33      928
34       25
35       29
```

## §3 · Leiden UMAPs

```python  # cell 7
# ============================================================================
# §3  UMAPs — myeloid and fibroblast
# ============================================================================
for lin in LINEAGES:
    a = A[lin]
    fig = sc.pl.umap(a, color=[KEY[lin], "study", "disease"], ncols=3, frameon=False, size=3,
                     wspace=0.25, legend_loc="on data", legend_fontsize=7, show=False,
                     return_fig=True)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.suptitle(f"{lin}: {a.n_obs:,} cells, {a.obs[KEY[lin]].nunique()} clusters "
                 f"(res={LEIDEN_RES[lin]})", y=1.02, fontsize=10)
    fig.savefig(FIG / f"{TAGS[lin]}_umap_leiden.png", dpi=200)
    plt.show(); plt.close(fig)
```

```
<Figure size 2640x528 with 3 Axes>
<Figure size 2640x528 with 3 Axes>
```

## §4 · Marker dot plots

```python  # cell 9
# ============================================================================
# §4  Dot plots (the panels of §1, standard-scaled per gene)
# ============================================================================
PANEL = {}
for lin in LINEAGES:
    present = set(A[lin].var_names)
    PANEL[lin] = {k: [g for g in v if g in present] for k, v in MARKERS[lin].items()}
    dropped = {k: [g for g in v if g not in present] for k, v in MARKERS[lin].items()}
    PANEL[lin] = {k: v for k, v in PANEL[lin].items() if v}
    for k, v in dropped.items():
        if v:
            print(f"{lin}: dropped from {k}: {v}")

for lin in LINEAGES:
    dp = sc.pl.dotplot(A[lin], PANEL[lin], groupby=KEY[lin], standard_scale="var",
                       dendrogram=True, figsize=(16, 5), return_fig=True,
                       title=f"{lin} — marker panels by cluster")
    dp.savefig(FIG / f"{TAGS[lin]}_dotplot_markers.png", dpi=200)
    dp.show()
```

```
Myeloid: dropped from Mac_FOLR2: ['MERTK']
WARNING: dendrogram data not found (using key=dendrogram_leiden_mye). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.

WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: pan-myeloid, LC, cDC1, etc.

<Figure size 1760x550 with 7 Axes>
<Figure size 1760x550 with 7 Axes>
WARNING: dendrogram data not found (using key=dendrogram_leiden_fib). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: 0, 1, 2, etc.
var_group_labels: pan-fibro, F_papillary, F_reticular, etc.

<Figure size 1760x550 with 7 Axes>
<Figure size 1760x550 with 7 Axes>
```

## §5 · The numbers — copy-paste this into an LLM

For every cluster: size / donor spread / QC, the five graded signature scores, **mean log1p
expression and detected fraction (`mean/frac`) for every panel gene**, the li2024 label it overlaps
most, and the Wilcoxon top-15. Nothing is thresholded away — a marker that is absent shows as
`0.00/0.00`, which is evidence too.

The same numbers go to `tables/skin_{mye,fib}_cluster_expression.csv` if you'd rather work from the
file.

```python  # cell 11
# ============================================================================
# §5a  Per-cluster expression / signature / QC tables
# ============================================================================
EXPR, QC, TOPN = {}, {}, {}

for lin in LINEAGES:
    a, key, p = A[lin], KEY[lin], paths(TAGS[lin])
    genes = [g for v in PANEL[lin].values() for g in v]
    genes = list(dict.fromkeys(genes))

    # mean log1p and detected fraction, per cluster x panel gene
    X = a[:, genes].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    grp = a.obs[key].astype(str).to_numpy()
    mean = pd.DataFrame(X, columns=genes).groupby(grp).mean()
    frac = pd.DataFrame((X > 0).astype(np.float32), columns=genes).groupby(grp).mean()
    order = natsorted(mean.index)
    EXPR[lin] = (mean.reindex(order).round(2), frac.reindex(order).round(2))
    del X; gc.collect()

    # graded signatures
    for name, gl in SIGNATURES[lin].items():
        gl = [g for g in gl if g in a.var_names]
        sc.tl.score_genes(a, gl, score_name=name, random_state=SEED)
    sigs = a.obs.groupby(key, observed=True)[list(SIGNATURES[lin])].mean().reindex(order).round(2)

    # QC + the li2024 external anchor
    g = a.obs.groupby(key, observed=True)
    qc = pd.DataFrame(index=pd.Index(order, name=key))
    qc["n"] = g.size()
    qc["n_donors"] = g["donor"].nunique()
    qc["n_studies"] = g["study"].nunique()
    qc["top_study"] = g["study"].apply(lambda s: s.value_counts().idxmax())
    qc["top_study_frac"] = g["study"].apply(lambda s: round(float(s.value_counts(normalize=True).iloc[0]), 2))
    qc["med_genes"] = g["n_genes"].median().astype(int)
    qc["med_mito"] = g["pct_mito"].median().round(1)
    qc["med_doublet"] = g["doublet_score"].median().round(2)
    known = a.obs[~a.obs["cell_type"].astype(str).isin(["Unknown", "nan", ""])]
    if len(known):
        ct = pd.crosstab(known[key].astype(str), known["cell_type"].astype(str))
        fr = ct.div(ct.sum(1).clip(lower=1), axis=0)
        qc["li2024_top"] = fr.idxmax(1).reindex(order)
        qc["li2024_frac"] = fr.max(1).round(2).reindex(order)
    QC[lin] = qc.join(sigs)

    # Wilcoxon top-15 (cached — it is the slow step here)
    if p["deg"].exists():
        deg = pd.read_csv(p["deg"], dtype={"group": str})
    else:
        sc.tl.rank_genes_groups(a, key, method="wilcoxon", pts=True)
        deg = sc.get.rank_genes_groups_df(a, group=None)
        deg = deg[(deg["pvals_adj"] < 0.01) & (deg["logfoldchanges"] > 0.5)]
        deg = deg.sort_values(["group", "scores"], ascending=[True, False])
        deg.to_csv(p["deg"], index=False)
    TOPN[lin] = (deg.groupby("group", observed=True).head(15)
                 .groupby("group", observed=True)["names"].apply(", ".join).reindex(order))

    long = (EXPR[lin][0].stack().rename("mean_log1p").to_frame()
            .join(EXPR[lin][1].stack().rename("frac_detected")))
    long.index.names = [key, "gene"]
    long.to_csv(p["expr"])
    QC[lin].to_csv(p["qc"])
    print(f"{lin}: {len(order)} clusters, {len(genes)} panel genes -> "
          f"{p['expr'].name}, {p['qc'].name}")
```

```
Myeloid: 25 clusters, 86 panel genes -> skin_mye_cluster_expression.csv, skin_mye_cluster_qc.csv
Fibroblast: 36 clusters, 71 panel genes -> skin_fib_cluster_expression.csv, skin_fib_cluster_qc.csv
```

```python  # cell 12
# ============================================================================
# §5b  The paste block
# ============================================================================
for lin in LINEAGES:
    a, key = A[lin], KEY[lin]
    mean, frac = EXPR[lin]
    qc = QC[lin]
    print("=" * 100)
    print(f"{lin.upper()} — {a.n_obs:,} skin cells, {len(mean)} Leiden clusters "
          f"(res={LEIDEN_RES[lin]}), values are `mean log1p CP10K / fraction detected`")
    print(f"allowed labels: {VOCAB[lin]}")
    print("=" * 100)
    for c in mean.index:
        r = qc.loc[c]
        head = (f"\n--- cluster {c}  n={int(r['n']):,}  donors={int(r['n_donors'])}  "
                f"studies={int(r['n_studies'])}  top_study={r['top_study']}({r['top_study_frac']})  "
                f"med_genes={int(r['med_genes'])}  med_mito={r['med_mito']}  "
                f"doublet={r['med_doublet']}")
        if "li2024_top" in qc.columns:
            head += f"  li2024={r['li2024_top']}({r['li2024_frac']})"
        print(head)
        print("  signatures: " + "  ".join(f"{s}={r[s]}" for s in SIGNATURES[lin]))
        for panel, genes in PANEL[lin].items():
            vals = "  ".join(f"{g} {mean.loc[c, g]:.2f}/{frac.loc[c, g]:.2f}" for g in genes)
            print(f"  {panel:15s} {vals}")
        print(f"  wilcoxon top15 : {TOPN[lin].get(c, '')}")
    print()
```

```
====================================================================================================
MYELOID — 144,009 skin cells, 25 Leiden clusters (res=1.0), values are `mean log1p CP10K / fraction detected`
allowed labels: ['LC', 'cDC1', 'cDC2', 'DC_LAMP3', 'pDC', 'Mono_CD14', 'Mono_CD16', 'moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mac_ISG', 'Neutrophils', 'Myeloid_prolif', 'UNK']
====================================================================================================

--- cluster 0  n=4,051  donors=125  studies=13  top_study=li2024(0.58)  med_genes=1787  med_mito=2.3  doublet=0.04  li2024=LC_3(0.23)
  signatures: sig_M1=-0.01  sig_M2=-0.73  sig_TAM=-0.3  sig_APC=1.51  sig_angio=-0.04
  pan-myeloid     PTPRC 1.45/0.63  LYZ 0.60/0.25  AIF1 1.88/0.68  HLA-DRA 2.41/0.42  CD68 1.12/0.53  ITGAX 0.27/0.11
  LC              CD207 2.76/0.75  CD1A 2.01/0.61  EPCAM 0.46/0.27  CLDN1 1.84/0.68  PLEK2 1.26/0.56
  cDC1            CLEC9A 0.01/0.00  XCR1 0.00/0.00  WDFY4 0.23/0.12  BATF3 0.06/0.04  IRF8 0.44/0.23  CADM1 0.01/0.01
  cDC2            CD1C 0.55/0.25  CLEC10A 0.57/0.27  FCER1A 2.20/0.70  CD1B 0.22/0.11  IRF4 0.61/0.29
  DC_LAMP3        LAMP3 0.69/0.28  CCR7 0.65/0.24  FSCN1 1.09/0.43  CCL19 0.03/0.02  CCL22 1.22/0.35  IDO1 0.98/0.33  BIRC3 1.16/0.46
  pDC             LILRA4 0.04/0.02  IL3RA 0.16/0.10  TCF4 0.20/0.13  GZMB 0.03/0.02  JCHAIN 0.02/0.01
  Mono_CD14       S100A8 0.28/0.13  S100A9 0.21/0.10  S100A12 0.00/0.00  FCN1 0.00/0.00  VCAN 0.01/0.01  CD14 0.10/0.06
  Mono_CD16       FCGR3A 0.02/0.01  CDKN1C 0.15/0.09  LST1 2.30/0.80  MS4A7 0.36/0.21  CSF1R 1.38/0.51
  moDC            CD209 0.01/0.00  IL1B 0.55/0.20  EREG 0.04/0.02  THBS1 0.03/0.01  CD1C 0.55/0.25
  Mac_FOLR2       FOLR2 0.01/0.00  LYVE1 0.00/0.00  SELENOP 0.03/0.01  C1QA 0.42/0.18  C1QB 0.43/0.17  C1QC 0.84/0.33  MRC1 0.02/0.01  CD163 0.01/0.01  F13A1 0.03/0.01  STAB1 0.03/0.02
  Mac_SPP1_TREM2  SPP1 0.05/0.02  TREM2 0.00/0.00  APOC1 0.06/0.04  ACP5 0.11/0.06  GPNMB 0.05/0.03  MMP9 1.68/0.58  MMP12 0.59/0.23  CHI3L1 0.01/0.01
  Mac_infl        IL1B 0.55/0.20  CXCL1 0.02/0.01  CXCL2 0.12/0.06  CXCL9 0.08/0.03  CXCL10 0.08/0.03  CXCL11 0.01/0.01  TREM1 0.00/0.00  CD300E 0.04/0.02  OLR1 0.02/0.01
  Mac_ISG         ISG15 0.50/0.27  IFIT1 0.02/0.01  IFI6 0.17/0.10
  prolif          MKI67 0.18/0.09  TOP2A 0.13/0.06
  !not-myeloid    CD3D 0.03/0.02  MS4A1 0.00/0.00  KRT14 0.37/0.18  COL1A1 0.06/0.03  PECAM1 0.05/0.03  TPSAB1 0.01/0.00
  wilcoxon top15 : FCGBP, CD207, S100B, CLDN1, PLEK2, CD1A, FCER1A, C15orf48, GNA15, PKIB, GSN, SERPINF1, CD74, ACOT7, RALA

--- cluster 1  n=12,468  donors=90  studies=12  top_study=li2024(0.93)  med_genes=1974  med_mito=2.4  doublet=0.04  li2024=moDC_3(0.46)
  signatures: sig_M1=0.33  sig_M2=-0.72  sig_TAM=0.03  sig_APC=0.28  sig_angio=0.08
  pan-myeloid     PTPRC 0.48/0.28  LYZ 2.26/0.75  AIF1 0.21/0.13  HLA-DRA 0.37/0.07  CD68 0.57/0.31  ITGAX 0.11/0.07
  LC              CD207 0.36/0.17  CD1A 0.15/0.08  EPCAM 0.02/0.02  CLDN1 0.22/0.12  PLEK2 0.06/0.04
  cDC1            CLEC9A 0.00/0.00  XCR1 0.00/0.00  WDFY4 0.13/0.10  BATF3 0.29/0.20  IRF8 0.07/0.04  CADM1 0.00/0.00
  cDC2            CD1C 0.12/0.06  CLEC10A 0.27/0.16  FCER1A 0.09/0.06  CD1B 0.37/0.18  IRF4 1.52/0.72
  DC_LAMP3        LAMP3 2.81/0.91  CCR7 3.74/0.95  FSCN1 2.87/0.81  CCL19 0.33/0.13  CCL22 5.08/0.95  IDO1 3.89/0.93  BIRC3 3.82/0.97
  pDC             LILRA4 0.00/0.00  IL3RA 0.34/0.21  TCF4 0.13/0.09  GZMB 0.05/0.02  JCHAIN 0.02/0.01
  Mono_CD14       S100A8 0.06/0.04  S100A9 0.05/0.03  S100A12 0.00/0.00  FCN1 0.00/0.00  VCAN 0.13/0.08  CD14 0.01/0.01
  Mono_CD16       FCGR3A 0.00/0.00  CDKN1C 0.04/0.03  LST1 0.36/0.24  MS4A7 0.82/0.44  CSF1R 0.02/0.01
  moDC            CD209 0.00/0.00  IL1B 0.32/0.10  EREG 0.32/0.13  THBS1 0.71/0.23  CD1C 0.12/0.06
  Mac_FOLR2       FOLR2 0.00/0.00  LYVE1 0.02/0.01  SELENOP 0.00/0.00  C1QA 0.10/0.06  C1QB 0.38/0.19  C1QC 0.33/0.17  MRC1 0.13/0.07  CD163 0.00/0.00  F13A1 0.01/0.00  STAB1 0.00/0.00
  Mac_SPP1_TR
... [110301 more chars truncated]
```

```python  # cell 13
# ============================================================================
# §6  cluster -> subtype_fine   (HAND-FILLED from §5)
# ============================================================================
# Keys 0-30 are laid out ready to fill. The Leiden decides the real count: the assert below
# requires set(CLUSTER2FINE[lin]) == the cluster IDs, so delete the keys past the last cluster
# (the UNFILLED branch prints exactly which IDs exist). Labels must come from VOCAB[lin];
# any "" left anywhere keeps MAPPED False, so §7 and §8 will not run on a half-filled map.
# Monocytes are split, not pooled: the initial (Reynolds-derived) skin taxonomy separates
# classical from non-classical, and this clustering already resolves both. Evidence, from
# tables/skin_mye_cluster_expression.csv (frac_detected) and _cluster_qc.csv:
#   15  n=7,187  121 donors  CD14 .48  FCGR3A .21  FCN1 .51  VCAN .31  S100A12 .22  -> Mono_CD14
#   22  n=  442   73 donors  CD14 .37  FCGR3A .81  CD68 .88  AIF1 .97  C1QA .29     -> Mono_CD16
#       (li2024 anchor for 22 is Mono_mac at 0.78 — the strongest anchor of any myeloid cluster)
# Cluster IDs and sizes do not move, so CLUSTER_SIZES below is unchanged and still locks.
CLUSTER2FINE = {
    "Myeloid": {
        "0": "LC", "1": "DC_LAMP3", "2": "DC_LAMP3", "3": "Mac_infl", "4": "UNK", "5": "moDC", "6": "Mac_FOLR2", "7": "Mac_infl",
        "8": "UNK", "9": "DC_LAMP3", "10": "UNK", "11": "Mac_FOLR2", "12": "UNK", "13": "UNK", "14": "cDC2", "15": "Mono_CD14",
        "16": "UNK", "17": "UNK", "18": "UNK", "19": "UNK", "20": "UNK", "21": "Mac_SPP1_TREM2", "22": "Mono_CD16", "23": "UNK",
        "24": "Neutrophils", 
    },
   "Fibroblast": {
        "0": "F_inflammatory", "1": "F_inflammatory", "2": "F_mesenchymal", "3": "F_inflammatory",
        "4": "UNK", "5": "F_reticular", "6": "F_papillary", "7": "F_inflammatory",
        "8": "F_reticular", "9": "F_apCAF", "10": "UNK", "11": "UNK",
        "12": "UNK", "13": "UNK", "14": "UNK", "15": "UNK",
        "16": "UNK", "17": "F_reticular", "18": "F_reticular", "19": "F_myofibroblast",
        "20": "UNK", "21": "UNK", "22": "UNK", "23": "UNK",
        "24": "UNK", "25": "F_mesenchymal", "26": "UNK", "27": "UNK",
        "28": "F_inflammatory", "29": "UNK", "30": "UNK",
        "31": "UNK", "32": "UNK", "33": "F_mesenchymal", "34": "UNK", "35": "UNK",
    },
}
# Paste from the sizes the UNFILLED branch prints — the lock that ties the map to this clustering.
CLUSTER_SIZES = {"Myeloid":     {'0': 4051, '1': 12468, '2': 8192, '3': 6981, '4': 3478, '5': 12447, '6': 14084, '7': 7646, '8': 1126, '9': 3063, '10': 1508, '11': 8332, '12': 2342, '13': 5866, '14': 12480, '15': 7187, '16': 13124, '17': 149, '18': 5110, '19': 19, '20': 10855, '21': 1411, '22': 442, '23': 1342, '24': 306}
, "Fibroblast": {'0': 15965, '1': 2003, '2': 6558, '3': 14329, '4': 1892, '5': 8510, '6': 15376, '7': 12775, '8': 20665, '9': 23744, '10': 1749, '11': 16399, '12': 183, '13': 109, '14': 9780, '15': 264, '16': 4358, '17': 6675, '18': 8466, '19': 897, '20': 31, '21': 3913, '22': 18, '23': 544, '24': 2858, '25': 3210, '26': 14, '27': 30, '28': 757, '29': 46, '30': 1308, '31': 14, '32': 30, '33': 928, '34': 25, '35': 29}}

MAPPED = True
for lin in LINEAGES:
    a, key = A[lin], KEY[lin]
    cats = [str(c) for c in a.obs[key].cat.categories]
    now = {c: int((a.obs[key].astype(str) == c).sum()) for c in cats}
    m = CLUSTER2FINE[lin]

    if (not m) or any(v == "" for v in m.values()):
        print(f"--- {lin}: UNFILLED. Paste into CLUSTER2FINE[{lin!r}]: ---")
        print("    {" + ", ".join(f'"{c}": ""' for c in cats) + "}")
        print(f"--- and into CLUSTER_SIZES[{lin!r}]: ---\n    {now!r}\n")
        MAPPED = False
        continue

    assert set(m) == set(cats), (
        f"{lin}: map keys != clusters\n  missing: {natsorted(set(cats) - set(m))}\n"
        f"  extra  : {natsorted(set(m) - set(cats))}")
    bad = set(m.values()) - set(VOCAB[lin])
    assert not bad, f"{lin}: labels outside the vocabulary: {sorted(bad)}"
    if CLUSTER_SIZES[lin]:
        assert now == {str(k): int(v) for k, v in CLUSTER_SIZES[lin].items()}, (
            f"{lin}: the Leiden changed since this map was authored — cluster IDs no longer mean "
            f"what the map says.\n  authored: {CLUSTER_SIZES[lin]}\n  now: {now}")
    else:
        print(f"!! {lin}: CLUSTER_SIZES is empty — paste {now!r} to lock the map")

    a.obs["subtype_fine"] = pd.Categorical(a.obs[key].astype(str).map(m))
    print(f"{lin}:")
    print(a.obs["subtype_fine"].value_counts().to_string())

if not MAPPED:
    print("\nFill the dicts above, then re-run from here — §7 onwards needs subtype_fine.")
```

```
Myeloid:
subtype_fine
UNK               44919
DC_LAMP3          23723
Mac_FOLR2         22416
Mac_infl          14627
cDC2              12480
moDC              12447
Mono_CD14          7187
LC                 4051
Mac_SPP1_TREM2     1411
Mono_CD16           442
Neutrophils         306
Fibroblast:
subtype_fine
F_inflammatory     45829
F_reticular        44316
UNK                43594
F_apCAF            23744
F_papillary        15376
F_mesenchymal      10696
F_myofibroblast      897
```

## §7 · Annotated UMAPs

```python  # cell 15
# ============================================================================
# §7  Annotated UMAPs + the labelled dot plot
# ============================================================================
assert MAPPED, "fill CLUSTER2FINE in §6 first"

for lin in LINEAGES:
    a = A[lin]
    fig = sc.pl.umap(a, color="subtype_fine", frameon=False, size=3, legend_loc="on data",
                     legend_fontsize=7, show=False, return_fig=True,
                     title=f"{lin} — subtype_fine ({a.obs['subtype_fine'].nunique()} states)")
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.savefig(FIG / f"{TAGS[lin]}_umap_subtype_fine.png", dpi=200)
    plt.show(); plt.close(fig)

    dp = sc.pl.dotplot(a, PANEL[lin], groupby="subtype_fine", standard_scale="var",
                       dendrogram=True, figsize=(16, 4.5), return_fig=True,
                       title=f"{lin} — panels by subtype_fine")
    dp.savefig(FIG / f"{TAGS[lin]}_dotplot_subtype_fine.png", dpi=200)
    dp.show()
```

```
<Figure size 704x528 with 1 Axes>
WARNING: dendrogram data not found (using key=dendrogram_subtype_fine). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: DC_LAMP3, LC, Mac_FOLR2, etc.
var_group_labels: pan-myeloid, LC, cDC1, etc.

<Figure size 1760x495 with 7 Axes>
<Figure size 1760x495 with 7 Axes>
<Figure size 704x528 with 1 Axes>
WARNING: dendrogram data not found (using key=dendrogram_subtype_fine). Running `sc.tl.dendrogram` with default parameters. For fine tuning it is recommended to run `sc.tl.dendrogram` independently.
WARNING: Groups are not reordered because the `groupby` categories and the `var_group_labels` are different.
categories: F_apCAF, F_inflammatory, F_mesenchymal, etc.
var_group_labels: pan-fibro, F_papillary, F_reticular, etc.

<Figure size 1760x495 with 7 Axes>
<Figure size 1760x495 with 7 Axes>
```

## §7b · Re-cluster the UNK

`UNK` in §6 is a cluster-level verdict, not a cell-level one: at res≈1.0 a doublet-looking cluster
still carries real LC / macrophage / fibroblast cells, and the genuinely off-lineage cells
(pericyte, keratinocyte, T, mast) sit mixed in with them. This section takes the UNK cells of
**both lineages**, embeds each pool on its own (UNK-only HVG → scVI → UMAP → Leiden, so the
variation that was previously swamped by the dominant states can separate), dumps the same numeric
block as §5 plus an off-lineage marker panel, and lets §7d relabel the sub-clusters.

- the UNK pool is defined from `CLUSTER2FINE` (the cluster map), not from `subtype_fine`, so this
  section is idempotent — re-running it after §7d does not shrink the pool;
- §7d rebuilds `subtype_fine` from `CLUSTER2FINE` first and then applies the refinement, so the
  labels never stack;
- labels still come from `VOCAB[lin]` only. A sub-cluster that really is pericyte / keratinocyte /
  T / doublet stays `UNK` — the point is to rescue the cells that are not.

**First run trains** (one small scVI per lineage, ~5k / ~10k cells) — run it on the GPU kernel.
After that everything is read from `skin_{mye,fib}_unk_embedding.csv`.

```python  # cell 17
# ============================================================================
# §7b  Re-cluster the UNK cells of both lineages   (first run TRAINS — GPU)
# ============================================================================
assert MAPPED, "fill CLUSTER2FINE in §6 first"

UNK_RES = {"Myeloid": 0.6, "Fibroblast": 0.6}
UNK_N_HVG, UNK_N_LATENT = 1500, 10
UNK_MIN_CELLS = 200          # below this the pool is too small to re-cluster honestly


def unk_paths(tag):
    return dict(hvg=OUT / f"skin_{tag}_unk_hvg.json",
                scvi=MODELS / f"skin_{tag}_unk_scvi",
                emb=OUT / f"skin_{tag}_unk_embedding.csv",   # cell_id, umap1/2, leiden — one file,
                deg=TAB / f"skin_{tag}_unk_cluster_markers.csv",   # joined on cell_id so the row
                expr=TAB / f"skin_{tag}_unk_cluster_expression.csv",  # order cannot corrupt it
                qc=TAB / f"skin_{tag}_unk_cluster_qc.csv")


def embed_unk(u, lin):
    """UNK-only HVG -> scVI -> UMAP -> Leiden, frozen into one cell_id-indexed csv. GPU."""
    import scvi, torch
    tag, p = TAGS[lin], unk_paths(TAGS[lin])
    scvi.settings.seed = SEED
    print(f"[{lin} UNK] embedding {u.n_obs:,} cells — cuda={torch.cuda.is_available()}")

    if p["hvg"].exists():
        hvg = json.loads(p["hvg"].read_text())["genes"]
    else:
        _fit = u.copy()
        sc.pp.highly_variable_genes(_fit, flavor="seurat_v3", n_top_genes=UNK_N_HVG,
                                    layer="counts")
        hvg = _fit.var_names[_fit.var["highly_variable"].to_numpy()].tolist()
        del _fit; gc.collect()
        p["hvg"].write_text(json.dumps(
            {"lineage": lin, "subset": "UNK", "n_top_genes": UNK_N_HVG, "flavor": "seurat_v3",
             "batch_key": None, "seed": SEED, "genes": hvg}, indent=1))

    uhv = u[:, hvg].copy()
    scvi.model.SCVI.setup_anndata(uhv, layer="counts", batch_key="study")
    if p["scvi"].exists():
        model = scvi.model.SCVI.load(str(p["scvi"]), adata=uhv)
    else:
        model = scvi.model.SCVI(uhv, n_latent=UNK_N_LATENT, n_layers=2, gene_likelihood="nb")
        model.train(early_stopping=True, early_stopping_patience=15,
                    plan_kwargs={"lr": 1e-3}, check_val_every_n_epoch=1)
        model.save(str(p["scvi"]), overwrite=True)
    u.obsm["X_scVI_unk"] = model.get_latent_representation()
    del uhv, model; gc.collect()

    sc.pp.neighbors(u, use_rep="X_scVI_unk", random_state=SEED, key_added="unk")
    sc.tl.umap(u, random_state=SEED, neighbors_key="unk")
    sc.tl.leiden(u, resolution=UNK_RES[lin], random_state=SEED, key_added=f"leiden_{tag}_unk",
                 flavor="igraph", n_iterations=2, directed=False, neighbors_key="unk")
    pd.DataFrame({"cell_id": u.obs["cell_id"].astype(str).to_numpy(),
                  "umap1": u.obsm["X_umap"][:, 0].astype(np.float32),
                  "umap2": u.obsm["X_umap"][:, 1].astype(np.float32),
                  f"leiden_{tag}_unk": u.obs[f"leiden_{tag}_unk"].astype(str).to_numpy()}
                 ).to_csv(p["emb"], index=False)
    print(f"[{lin} UNK] {u.obs[f'leiden_{tag}_unk'].nunique()} sub-clusters -> {p['emb'].name}")


U, UKEY = {}, {}
for lin in LINEAGES:
    a, tag, p = A[lin], TAGS[lin], unk_paths(TAGS[lin])
    UKEY[lin] = f"leiden_{tag}_unk"

    # from the cluster map, not from subtype_fine -> unchanged by §7d
    unk_clusters = [c for c, v in CLUSTER2FINE[lin].items() if v == "UNK"]
    m = a.obs[KEY[lin]].astype(str).isin(unk_clusters).to_numpy()
    if int(m.sum()) < UNK_MIN_CELLS:
        print(f"{lin}: {int(m.sum()):,} UNK cells (< {UNK_MIN_CELLS}) — not re-clustered")
        continue

    u = a[m].copy()
    u.obs["parent_cluster"] = a.obs[KEY[lin]].astype(str).to_numpy()[m]
    for c in u.obs.columns:
        if str(u.obs[c].dtype) == "category":
            u.obs[c] = u.obs[c].cat.remove_unused_categories()

    if p["emb"].exists():
        e = pd.read_csv(p["emb"], dtype={"cell_id": str}).set_index("cell_id")
        ids = u.obs["cell_id"].astype(str)
        assert len(e) == u.n_obs and ids.isin(e.index).all(), (
            f"{p['emb'].name} covers {len(e):,} cells but the UNK pool is {u.n_obs:,} — §6 changed "
            f"which clusters are UNK; delete the cache (and {p['scvi'].name}) and re-run")
        e = e.reindex(ids)
        u.obsm["X_umap"] = e[["umap1", "umap2"]].to_numpy()
        vals = e[UKEY[lin]].astype(str).to_numpy()
        u.obs[UKEY[lin]] = pd.Categorical(vals, categories=natsorted(pd.unique(vals)))
    else:
        embed_unk(u, lin)

    U[lin] = u
    print(f"{lin:11s} UNK {u.n_obs:>6,} cells ({u.n_obs / a.n_obs:.1%} of the lineage) from "
          f"parent clusters {natsorted(unk_clusters)} -> {u.obs[UKEY[lin]].nunique()} sub-clusters")

for lin, u in U.items():
    fig = sc.pl.umap(u, color=[UKEY[lin], "parent_cluster", "study", "disease"], ncols=4,
                     frameon=False, size=6, wspace=0.25, legend_loc="on data", legend_fontsize=7,
                     show=False, return_fig=True)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.suptitle(f"{lin} UNK: {u.n_obs:,} cells, {u.obs[UKEY[lin]].nunique()} sub-clusters "
                 f"(res={UNK_RES[lin]})", y=1.03, fontsize=10)
    fig.savefig(FIG / f"{TAGS[lin]}_unk_umap_leiden.png", dpi=200)
    plt.show(); plt.close(fig)
```

```
Myeloid     UNK 44,919 cells (31.2% of the lineage) from parent clusters ['4', '8', '10', '12', '13', '16', '17', '18', '19', '20', '23'] -> 21 sub-clusters

Fibroblast  UNK 43,594 cells (23.6% of the lineage) from parent clusters ['4', '10', '11', '12', '13', '14', '15', '16', '20', '21', '22', '23', '24', '26', '27', '29', '30', '31', '32', '34', '35'] -> 58 sub-clusters

<Figure size 3520x528 with 4 Axes>
<Figure size 3520x528 with 4 Axes>
```

```python  # cell 18
# ============================================================================
# §7c  The numbers for the UNK sub-clusters — copy-paste this into an LLM
# ============================================================================
# The lineage panels of §1 plus the identities an UNK cluster is usually made of.
OFF_MARKERS = {
 "T/NK":         ["CD3D", "CD3E", "TRBC2", "IL7R", "NKG7", "GNLY"],
 "B/plasma":     ["MS4A1", "CD79A", "MZB1", "JCHAIN"],
 "Mast":         ["TPSAB1", "TPSB2", "CPA3", "KIT"],
 "Keratinocyte": ["KRT14", "KRT5", "KRT1", "KRT10", "DMKN", "KRTDAP"],
 "Endothelial":  ["PECAM1", "VWF", "CDH5", "PROX1", "CCL21"],
 "Pericyte/SMC": ["RGS5", "NOTCH3", "PDGFRB", "KCNJ8", "MYH11"],
 "Melanocyte":   ["MLANA", "PMEL", "TYRP1", "DCT"],
 "Neural/Schwann": ["PLP1", "S100B", "NRXN1"],
 "Erythroid":    ["HBB", "HBA1"],
 "myeloid-core": ["PTPRC", "LYZ", "AIF1", "HLA-DRA", "CD68", "C1QA", "ITGAX"],
 "fibro-core":   ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
 "stress":       ["HSPA1A", "HSPA1B", "FOS", "JUN", "EGR1", "MT-CO1"],
}

UNK_PANEL, UEXPR, UQC, UTOPN = {}, {}, {}, {}

for lin, u in U.items():
    key, p = UKEY[lin], unk_paths(TAGS[lin])
    present = set(u.var_names)
    panel = {**PANEL[lin], **{k: [g for g in v if g in present] for k, v in OFF_MARKERS.items()}}
    UNK_PANEL[lin] = {k: v for k, v in panel.items() if v}
    genes = list(dict.fromkeys(g for v in UNK_PANEL[lin].values() for g in v))

    X = u[:, genes].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    grp = u.obs[key].astype(str).to_numpy()
    mean = pd.DataFrame(X, columns=genes).groupby(grp).mean()
    frac = pd.DataFrame((X > 0).astype(np.float32), columns=genes).groupby(grp).mean()
    order = natsorted(mean.index)
    UEXPR[lin] = (mean.reindex(order).round(2), frac.reindex(order).round(2))
    del X; gc.collect()

    for name, gl in SIGNATURES[lin].items():
        gl = [g for g in gl if g in u.var_names]
        sc.tl.score_genes(u, gl, score_name=name, random_state=SEED)
    sigs = u.obs.groupby(key, observed=True)[list(SIGNATURES[lin])].mean().reindex(order).round(2)

    g = u.obs.groupby(key, observed=True)
    qc = pd.DataFrame(index=pd.Index(order, name=key))
    qc["n"] = g.size()
    qc["n_donors"] = g["donor"].nunique()
    qc["n_studies"] = g["study"].nunique()
    qc["top_study"] = g["study"].apply(lambda s: s.value_counts().idxmax())
    qc["top_study_frac"] = g["study"].apply(lambda s: round(float(s.value_counts(normalize=True).iloc[0]), 2))
    qc["med_genes"] = g["n_genes"].median().astype(int)
    qc["med_mito"] = g["pct_mito"].median().round(1)
    qc["med_doublet"] = g["doublet_score"].median().round(2)
    qc["parent"] = g["parent_cluster"].apply(
        lambda s: "+".join(f"{k}:{v}" for k, v in s.value_counts(normalize=True).round(2)
                           .head(3).items()))
    known = u.obs[~u.obs["cell_type"].astype(str).isin(["Unknown", "nan", ""])]
    if len(known):
        ct = pd.crosstab(known[key].astype(str), known["cell_type"].astype(str))
        fr = ct.div(ct.sum(1).clip(lower=1), axis=0)
        qc["li2024_top"] = fr.idxmax(1).reindex(order)
        qc["li2024_frac"] = fr.max(1).round(2).reindex(order)
    UQC[lin] = qc.join(sigs)

    if p["deg"].exists():
        deg = pd.read_csv(p["deg"], dtype={"group": str})
    else:
        sc.tl.rank_genes_groups(u, key, method="wilcoxon", pts=True)
        deg = sc.get.rank_genes_groups_df(u, group=None)
        deg = deg[(deg["pvals_adj"] < 0.01) & (deg["logfoldchanges"] > 0.5)]
        deg = deg.sort_values(["group", "scores"], ascending=[True, False])
        deg.to_csv(p["deg"], index=False)
    UTOPN[lin] = (deg.groupby("group", observed=True).head(15)
                  .groupby("group", observed=True)["names"].apply(", ".join).reindex(order))

    long = (UEXPR[lin][0].stack().rename("mean_log1p").to_frame()
            .join(UEXPR[lin][1].stack().rename("frac_detected")))
    long.index.names = [key, "gene"]
    long.to_csv(p["expr"])
    UQC[lin].to_csv(p["qc"])
    print(f"{lin} UNK: {len(order)} sub-clusters, {len(genes)} panel genes -> "
          f"{p['expr'].name}, {p['qc'].name}")

for lin, u in U.items():
    key = UKEY[lin]
    mean, frac = UEXPR[lin]
    qc = UQC[lin]
    print("=" * 100)
    print(f"{lin.upper()} UNK POOL — {u.n_obs:,} cells that §6 could not call, "
          f"{len(mean)} sub-clusters (res={UNK_RES[lin]}), values are "
          f"`mean log1p CP10K / fraction detected`")
    print(f"allowed labels: {VOCAB[lin]}   (leave a sub-cluster UNK if it is off-lineage, "
          f"a doublet, or low-quality)")
    print("=" * 100)
    for c in mean.index:
        r = qc.loc[c]
        head = (f"\n--- UNK sub-cluster {c}  n={int(r['n']):,}  donors={int(r['n_donors'])}  "
                f"studies={int(r['n_studies'])}  top_study={r['top_study']}({r['top_study_frac']})  "
                f"med_genes={int(r['med_genes'])}  med_mito={r['med_mito']}  "
                f"doublet={r['med_doublet']}  parent={r['parent']}")
        if "li2024_top" in qc.columns:
            head += f"  li2024={r['li2024_top']}({r['li2024_frac']})"
        print(head)
        print("  signatures: " + "  ".join(f"{s}={r[s]}" for s in SIGNATURES[lin]))
        for panel, genes in UNK_PANEL[lin].items():
            vals = "  ".join(f"{g} {mean.loc[c, g]:.2f}/{frac.loc[c, g]:.2f}" for g in genes)
            print(f"  {panel:15s} {vals}")
        print(f"  wilcoxon top15 : {UTOPN[lin].get(c, '')}")
    print()

for lin, u in U.items():
    dp = sc.pl.dotplot(u, UNK_PANEL[lin], groupby=UKEY[lin], standard_scale="var",
                       dendrogram=True, figsize=(20, 4.5), return_fig=True,
                       title=f"{lin} UNK pool — lineage + off-lineage panels by sub-cluster")
    dp.savefig(FIG / f"{TAGS[lin]}_unk_dotplot_markers.png", dpi=200)
    dp.show()
```

```
Myeloid UNK: 21 sub-clusters, 128 panel genes -> skin_mye_unk_cluster_expression.csv, skin_mye_unk_cluster_qc.csv
Fibroblast UNK: 58 sub-clusters, 110 panel genes -> skin_fib_unk_cluster_expression.csv, skin_fib_unk_cluster_qc.csv
====================================================================================================
MYELOID UNK POOL — 44,919 cells that §6 could not call, 21 sub-clusters (res=0.6), values are `mean log1p CP10K / fraction detected`
allowed labels: ['LC', 'cDC1', 'cDC2', 'DC_LAMP3', 'pDC', 'Mono_CD14', 'Mono_CD16', 'moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mac_ISG', 'Neutrophils', 'Myeloid_prolif', 'UNK']   (leave a sub-cluster UNK if it is off-lineage, a doublet, or low-quality)
====================================================================================================

--- UNK sub-cluster 0  n=5,651  donors=125  studies=14  top_study=li2024(0.49)  med_genes=1256  med_mito=2.9  doublet=0.04  parent=13:0.87+4:0.04+16:0.03  li2024=DC2(0.79)
  signatures: sig_M1=-0.03  sig_M2=-0.3  sig_TAM=0.2  sig_APC=1.49  sig_angio=0.13
  pan-myeloid     PTPRC 1.36/0.60  LYZ 3.45/0.87  AIF1 2.14/0.75  HLA-DRA 3.04/0.51  CD68 1.15/0.51  ITGAX 0.69/0.34
  LC              CD207 0.70/0.26  CD1A 1.80/0.61  EPCAM 0.06/0.03  CLDN1 1.53/0.57  PLEK2 0.39/0.22
  cDC1            CLEC9A 0.03/0.02  XCR1 0.00/0.00  WDFY4 0.25/0.15  BATF3 0.20/0.12  IRF8 0.32/0.17  CADM1 0.01/0.01
  cDC2            CD1C 1.59/0.60  CLEC10A 2.05/0.74  FCER1A 1.46/0.55  CD1B 1.54/0.56  IRF4 0.77/0.35
  DC_LAMP3        LAMP3 0.51/0.24  CCR7 0.24/0.11  FSCN1 1.03/0.45  CCL19 0.05/0.03  CCL22 0.82/0.25  IDO1 0.20/0.08  BIRC3 0.83/0.35
  pDC             LILRA4 0.01/0.00  IL3RA 0.26/0.14  TCF4 0.21/0.13  GZMB 0.03/0.02  JCHAIN 0.01/0.01
  Mono_CD14       S100A8 0.57/0.24  S100A9 0.47/0.20  S100A12 0.01/0.01  FCN1 0.07/0.03  VCAN 0.11/0.06  CD14 0.25/0.11
  Mono_CD16       FCGR3A 0.16/0.08  CDKN1C 0.19/0.09  LST1 2.06/0.76  MS4A7 0.52/0.27  CSF1R 0.89/0.42
  moDC            CD209 0.08/0.05  IL1B 0.82/0.30  EREG 0.10/0.05  THBS1 0.08/0.04  CD1C 1.59/0.60
  Mac_FOLR2       FOLR2 0.03/0.02  LYVE1 0.01/0.01  SELENOP 0.05/0.02  C1QA 1.36/0.49  C1QB 1.64/0.56  C1QC 1.76/0.59  MRC1 0.76/0.34  CD163 0.05/0.02  F13A1 0.57/0.26  STAB1 0.17/0.08
  Mac_SPP1_TREM2  SPP1 0.18/0.08  TREM2 0.22/0.11  APOC1 0.33/0.17  ACP5 0.48/0.24  GPNMB 0.07/0.04  MMP9 0.76/0.32  MMP12 1.76/0.49  CHI3L1 0.01/0.01
  Mac_infl        IL1B 0.82/0.30  CXCL1 0.12/0.05  CXCL2 0.43/0.17  CXCL9 0.33/0.12  CXCL10 0.23/0.08  CXCL11 0.04/0.02  TREM1 0.11/0.06  CD300E 0.03/0.02  OLR1 0.16/0.09
  Mac_ISG         ISG15 0.70/0.33  IFIT1 0.02/0.01  IFI6 0.29/0.16
  prolif          MKI67 0.02/0.01  TOP2A 0.03/0.02
  !not-myeloid    CD3D 0.04/0.02  MS4A1 0.01/0.01  KRT14 0.58/0.24  COL1A1 0.10/0.05  PECAM1 0.16/0.10  TPSAB1 0.02/0.01
  T/NK            CD3D 0.04/0.02  CD3E 0.05/0.03  TRBC2 0.02/0.02  IL7R 0.75/0.30  NKG7 0.04/0.03  GNLY 0.08/0.04
  B/plasma        MS4A1 0.01/0.01  CD79A 0.01/0.00  MZB1 0.00/0.00  JCHAIN 0.01/0.01
  Mast            TPSAB1 0.02/0.01  TPSB2 0.01/0.01  CPA3 0.00/0.00  KIT 0.00/0.00
  Keratinocyte    KRT14 0.58/0.24  KRT5 0.32/0.16  KRT1 0.29/0.14  KRT10 0.74/0.37  DMKN 0.18/0.09  KRTDAP 0.21/0.10
  Endothelial     PECAM1 0.16/0.10  VWF 0.01/0.00  CDH5 0.00/0.00  PROX1 0.00/0.00  CCL21 0.02/0.01
  Pericyte/SMC    RGS5 0.01/0.00  NOTCH3 0.01/0.00  PDGFRB 0.01/0.00  KCNJ8 0.00/0.00  MYH11 0.01/0.00
  Melanocyte      MLANA 0.01/0.00  PMEL 0.01/0.01  TYRP1 0.02/0.01  DCT 0.01/0.00
  Neural/Schwann  PLP1 0.00/0.00  S100B 1.42/0.48  NRXN1 0.00/0.00
  Erythroid       HBB 0.12/0.04  HBA1 0.01/0.00
  myeloid-core    PTPRC 1.36/0.60  LYZ 3.45/0.87  AIF1 2.14/0.75  HLA-DRA 3.04/0.51  CD68 1.15/0.51  C1QA 1.36/0.49  ITGAX 0.69/0.34
  fibro-core      COL1A1 0.10/0.05  COL1A2 0.04/0.02  DCN 0.06/0.03  LUM 0.01/0.01  PDGFRA 0.01/0.01
  stress          HSPA1A 4.02/0.88  HSPA1B 3.31/0.81  FOS 2.45/0.71  JUN 1.96/0.67  EGR1 0.21/0.12
  wilcoxon top15 : TYROBP, CD1E, S100A11, 
... [232972 more chars truncated]
```

```python  # cell 19
# ============================================================================
# §7d  UNK sub-cluster -> subtype_fine   (HAND-FILLED from §7c)
# ============================================================================
UNK2FINE = {"Myeloid": {
        "0": "cDC2",            # CD1A .61, CD1B .56, CLDN1 .57 + CD1C .60, CLEC10A .74 -> epidermal CD1A+ cDC2; HSP-high
        "1": "cDC2",            # CLEC10A .74, FCER1A .63, CD1C .55; CCR7/IL1B activation + HSP/NF-kB stress; CD14/FCN1 low
        "2": "Myeloid_prolif",  # MKI67 .76, TOP2A .60; parents cDC2 + Mac
        "3": "UNK",             # 19 cells, 1 donor; IL22/GNLY/TRBV6-1 -> malignant-T doublet
        "4": "cDC1",            # CLEC9A .64, XCR1 .33, WDFY4 .76, IRF8 .87, CADM1 .51, DNASE1L3
        "5": "UNK",             # T/NK doublet on cDC1 background: CD3E .72, NKG7 .69, GZMB .62; doublet .11
        "6": "UNK",             # KC doublet / lowQ: KRT14/KRT5/KRT1/KRT10, NEAT1; 995 genes
        "7": "Mac_SPP1_TREM2",  # Flex LAM: MMP9 .88, GPNMB .69, APOE, ACP5 .52, APOC1 .53, CHI3L1 .38, NR1H3; FOLR2 .24
        "8": "UNK",             # 26 cells; macrophage + inflamed-KC transcripts (S100A7, KRTDAP, SERPINB4)
        "9": "Mac_FOLR2",       # Flex resident: CD163 .69, STAB1 .66, F13A1 .72, MRC1 .69, FOLR2 .46, CD209 .52, LGMN
        "10": "UNK",            # 20 cells; IL17F/CXCL13/CTLA4 -> malignant T
        "11": "UNK",            # T-myeloid doublet: CD3E .53, TRBC2, NKG7, GNLY; flat mixed-myeloid profile
        "12": "UNK",            # 30 cells, 1 donor; GZMB/CXCL10/KRT/Ig mixture
        "13": "cDC1",           # CLEC9A .48, WDFY4 .53, IRF8 .76, BATF3 .33; ~35% cycling -> flag per cell
        "14": "UNK",            # 18 cells; TRBV6-1/IL22/CCL20 -> malignant T
        "15": "UNK",            # mregDC-T doublet (Flex): CCL22/CCR7/IDO1 + CD3E .61, IL7R .65; doublet .11
        "16": "UNK",            # CXCL9 Mac + cytotoxic T/NK (GZMB .90, NKG7 .80); chennareddy 88% -> retest after CellBender
        "17": "pDC",            # pDC/AS-DC mix still unresolved: LILRA4 .27, TCF4 .63, SPIB, PPP1R14A, FCER1A .63
        "18": "UNK",            # 13 cells; mast (TPSAB1/CPA3/KIT)
        "19": "UNK",            # Mac-malignant T doublet (Flex): TRBC2 .95, JAK3, IKZF1; li2024 tumor_cell
        "20": "cDC1",           # Flex cDC1: WDFY4 .97, IRF8 .97, XCR1 .75, CLEC9A .68, CADM1 .68
    }, 
            "Fibroblast": {
        "0": "UNK",             # myeloid ambient/doublet: HLA-II .98, CCL17, LYZ .70; mito 7.1 (rindler)
        "1": "UNK",             # lowQ rindler fibroblasts (776 genes, mito 7.5); POSTN/TNC/COMP -> see Th2 note
        "2": "UNK",             # mast doublet: TPSAB1 .73, CPA3, KIT, HDC
        "3": "UNK",             # micro (78, 1 donor)
        "4": "F_apCAF",         # parent-9 program: CCL19 .65, C3 .74, C1S, TMEM176A/B, CD74 .92, PTGDS .81; T ambient -> gate
        "5": "UNK",             # CXCL13/Ig/CXCL9 contaminated; 641 genes
        "6": "UNK",             # mural: ACTA2 .88, MYH11 .63, RGS5 .55, PDGFRA .03; + myeloid ambient
        "7": "UNK",             # micro (mast)
        "8": "UNK",             # micro
        "9": "UNK",             # micro; POSTN/CCL26/CCL11 (Th2 motif)
        "10": "F_inflammatory", # secretory: CXCL12 .87, CFD .95, APOE .77, MGP .94 + CXCL2/IL6/IEG; multi-lineage ambient; rindler 94%
        "11": "UNK",            # micro
        "12": "UNK",            # micro; POSTN/CCL26/CCL13
        "13": "F_reticular",    # PI16 .55, MFAP5 .56, CCN5, CLEC3B, FBLN1, CXCL14, CFD .89; T ambient -> gate
        "14": "F_reticular",    # low conf: APOD/CFD/MGP/CXCL14, PI16 .41, MFAP5 .46; HBB/Ig ambient; 723 genes
        "15": "UNK",            # fibro-myeloid doublet: SPI1, TYROBP, FCER1G, AIF1 in DEGs; doublet .12
        "16": "UNK",            # micro
        "17": "UNK",            # intronic/long-gene profile (= parent 24; buus2025 50%) -> relabel after count harmonization
        "18": "UNK",            # KC doublet: KRT14 .97, KRT5 .88
        "19": "UNK",            # T doublet: IL32, GZMA, CD3D .52, TRBC2
        "20": "UNK",            # myeloid/NK + IFN: LYZ .83, CXCL9/10, GBP1, GNLY
        "21": "UNK",            # micro
        "22": "UNK",            # micro
        "23": "UNK",            # lowQ ambient: 621 genes, mito 7.0; CXCL13/GNLY/XCL1; chennareddy 86%
        "24": "UNK",            # fibro-T doublet (Flex): TRAC/ZAP70/IKZF1 DEGs on CCL19+ fibroblast
        "25": "UNK",            # micro
        "26": "UNK",            # micro (plasma)
        "27": "UNK",            # micro
        "28": "UNK",            # micro
        "29": "UNK",            # nerve-associated fibroblast: SFRP4, DKK3, CLDN1, ABCA8; li2024 Schwann_1 -> needs own label
        "30": "UNK",            # micro
        "31": "UNK",            # micro
        "32": "UNK",            # mural: ACTA2 .82, MYH11 .60, RGS5 .53, NDUFA4L2, PDGFRA .01
        "33": "UNK",            # micro; POSTN/CCL11/CCL26
        "34": "UNK",            # micro (endothelial: SELE, ACKR1)
        "35": "UNK",            # micro
        "36": "UNK",            # micro
        "37": "UNK",            # plasma doublet/ambient: IGKC, MZB1, JCHAIN, CD79A .34
        "38": "UNK",            # Th2-activated candidate: POSTN .92, CCL26, TNFAIP6; rindler 98%, 7 donors
        "39": "UNK",            # micro
        "40": "UNK",            # KC + HLA + HBB
        "41": "UNK",            # micro (myeloid)
        "42": "UNK",            # endothelial: ACKR1, PLVAP, CXCL9/11
        "43": "UNK",            # micro
        "44": "UNK",            # micro
        "45": "UNK",            # lowQ: CXCL13/Ig/mast/GNLY; 5 donors
        "46": "UNK", "47": "UNK", "48": "UNK", "49": "UNK", "50": "UNK", "51": "UNK",
        "52": "UNK", "53": "UNK", "54": "UNK", "55": "UNK", "56": "UNK", "57": "UNK",   # micro
    }}
UNK_CLUSTER_SIZES = {"Myeloid": {'0': 5651, '1': 3558, '2': 1584, '3': 19, '4': 7116, '5': 1404, '6': 2628, '7': 2623, '8': 26, '9': 4059, '10': 20, '11': 8396, '12': 30, '13': 2553, '14': 18, '15': 1123, '16': 1283, '17': 1361, '18': 13, '19': 439, '20': 1015}, 
                     
                     "Fibroblast": {'0': 645, '1': 3984, '2': 421, '3': 78, '4': 3270, '5': 289, '6': 982, '7': 17, '8': 53, '9': 60, '10': 4137, '11': 28, '12': 21, '13': 4930, '14': 383, '15': 3855, '16': 21, '17': 2840, '18': 1842, '19': 2086, '20': 1043, '21': 23, '22': 35, '23': 5036, '24': 864, '25': 26, '26': 196, '27': 26, '28': 10, '29': 534, '30': 34, '31': 13, '32': 2775, '33': 28, '34': 26, '35': 23, '36': 78, '37': 660, '38': 303, '39': 167, '40': 269, '41': 14, '42': 200, '43': 185, '44': 178, '45': 486, '46': 15, '47': 63, '48': 85, '49': 25, '50': 27, '51': 32, '52': 31, '53': 30, '54': 32, '55': 21, '56': 14, '57': 45}}

UNK_MAPPED = True
for lin, u in U.items():
    key = UKEY[lin]
    cats = [str(c) for c in u.obs[key].cat.categories]
    now = {c: int((u.obs[key].astype(str) == c).sum()) for c in cats}
    m = UNK2FINE.get(lin, {})

    if (not m) or any(v == "" for v in m.values()):
        print(f"--- {lin}: UNFILLED. Paste into UNK2FINE[{lin!r}]: ---")
        print("    {" + ", ".join(f'"{c}": ""' for c in cats) + "}")
        print(f"--- and into UNK_CLUSTER_SIZES[{lin!r}]: ---\n    {now!r}\n")
        UNK_MAPPED = False
        continue

    assert set(m) == set(cats), (
        f"{lin} UNK: map keys != sub-clusters\n  missing: {natsorted(set(cats) - set(m))}\n"
        f"  extra  : {natsorted(set(m) - set(cats))}")
    bad = set(m.values()) - set(VOCAB[lin])
    assert not bad, f"{lin} UNK: labels outside the vocabulary: {sorted(bad)}"
    if UNK_CLUSTER_SIZES.get(lin):
        assert now == {str(k): int(v) for k, v in UNK_CLUSTER_SIZES[lin].items()}, (
            f"{lin} UNK: the sub-clustering changed since this map was authored.\n"
            f"  authored: {UNK_CLUSTER_SIZES[lin]}\n  now: {now}")
    else:
        print(f"!! {lin} UNK: UNK_CLUSTER_SIZES is empty — paste {now!r} to lock the map")

if not UNK_MAPPED:
    print("\nFill the dicts above from §7c, then re-run this cell — §8 uses the refined labels.")
else:
    for lin in LINEAGES:
        a = A[lin]
        # stage 1 rebuilt from scratch -> this cell is idempotent, the levels never stack
        a.obs["subtype_fine"] = pd.Categorical(a.obs[KEY[lin]].astype(str).map(CLUSTER2FINE[lin]))
        if lin not in U:
            continue
        u, key = U[lin], UKEY[lin]
        lab = a.obs["subtype_fine"].astype(str).to_numpy()
        new = pd.Series(u.obs[key].astype(str).map(UNK2FINE[lin]).to_numpy(),
                        index=u.obs["cell_id"].astype(str).to_numpy())
        idx = pd.Index(a.obs["cell_id"].astype(str)).get_indexer(new.index)
        assert (idx >= 0).all() and (lab[idx] == "UNK").all(), (
            f"{lin}: the UNK pool no longer maps onto UNK cells — re-run §7b")
        lab[idx] = new.to_numpy()
        a.obs["subtype_fine"] = pd.Categorical(lab)
        u.obs["subtype_fine"] = pd.Categorical(new.to_numpy())

        rescued = int((new != "UNK").sum())
        print(f"\n{lin}: {rescued:,}/{len(new):,} UNK cells rescued "
              f"({rescued / a.n_obs:.1%} of the lineage)")
        print(pd.crosstab(u.obs["parent_cluster"].astype(str), u.obs["subtype_fine"].astype(str))
              .loc[natsorted(u.obs["parent_cluster"].astype(str).unique())].to_string())
        print(f"{lin} subtype_fine after §7d:")
        print(a.obs["subtype_fine"].value_counts().to_string())

    for lin, u in U.items():
        a = A[lin]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        sc.pl.umap(u, color="subtype_fine", frameon=False, size=6, legend_loc="on data",
                   legend_fontsize=7, ax=axes[0], show=False,
                   title=f"{lin} UNK pool — refined ({u.obs['subtype_fine'].nunique()} labels)")
        sc.pl.umap(a, color="subtype_fine", frameon=False, size=3, legend_loc="on data",
                   legend_fontsize=7, ax=axes[1], show=False,
                   title=f"{lin} — subtype_fine after §7d "
                         f"({a.obs['subtype_fine'].nunique()} states)")
        for ax in axes:
            for coll in ax.collections:
                coll.set_rasterized(True)
        fig.savefig(FIG / f"{TAGS[lin]}_unk_umap_subtype_fine.png", dpi=200)
        plt.show(); plt.close(fig)
```

```
Myeloid: 29,520/44,919 UNK cells rescued (20.5% of the lineage)
subtype_fine    Mac_FOLR2  Mac_SPP1_TREM2  Myeloid_prolif    UNK  cDC1  cDC2   pDC
parent_cluster                                                                    
4                      20               3              17    213    18  3197    10
8                      15               6               6   1018    23    57     1
10                     13              11            1329     57    34    62     2
12                    599              36              10   1436   127   122    12
13                     42               2              24    194   347  5245    12
16                    966              93             104  11274   320   256   111
17                      0               0               0     86     3    49    11
18                   2344            2433              60    175     6    92     0
19                      0               0               0     19     0     0     0
20                     41              31              34    862  9766    92    29
23                     19               8               0     65    40    37  1173
Myeloid subtype_fine after §7d:
subtype_fine
Mac_FOLR2         26475
DC_LAMP3          23723
cDC2              21689
UNK               15399
Mac_infl          14627
moDC              12447
cDC1              10684
Mono_CD14          7187
LC                 4051
Mac_SPP1_TREM2     4034
Myeloid_prolif     1584
pDC                1361
Mono_CD16           442
Neutrophils         306

Fibroblast: 12,720/43,594 UNK cells rescued (6.9% of the lineage)
subtype_fine    F_apCAF  F_inflammatory  F_reticular   UNK
parent_cluster                                            
4                    20               0          345  1527
10                   14               0           33  1702
11                 2567              79         4513  9240
12                    0               0            0   183
13                    0               0            0   109
14                  378             147          199  9056
15                    0               1            0   263
16                   75            3872           62   349
20                    0               0            0    31
21                   24              21           22  3846
22                    0               0            0    18
23                    2               4            6   532
24                  185               7          125  2541
26                    0               0            0    14
27                    0               0            0    30
29                    0               0            0    46
30                    5               6            8  1289
31                    0               0            0    14
32                    0               0            0    30
34                    0               0            0    25
35                    0               0            0    29
Fibroblast subtype_fine after §7d:
subtype_fine
F_inflammatory     49966
F_reticular        49629
UNK                30874
F_apCAF            27014
F_papillary        15376
F_mesenchymal      10696
F_myofibroblast      897

<Figure size 1430x605 with 2 Axes>
<Figure size 1430x605 with 2 Axes>
```

## §8 · Final UMAPs

`subtype_fine` after the §7d refinement, next to the Leiden it came from and the two covariates
that would explain it away if it were an artefact. Nothing is recomputed here -- this is the
picture of what §6 and §7d decided.

```python  # cell 21
# ============================================================================
# §8  Final UMAPs — subtype_fine against the Leiden it came from
# ============================================================================
import annot_utils as AU

assert UNK_MAPPED, "fill UNK2FINE in §7d first"

for lin in LINEAGES:
    a = A[lin]
    n_lab = a.obs["subtype_fine"].nunique()
    AU.save_umap(
        a, [KEY[lin], "subtype_fine", "study", "disease"],
        FIG / f"{TAGS[lin]}_umap_final_fine.png",
        ncols=4, wspace=0.25,
        title=[f"{lin} — Leiden (res={LEIDEN_RES[lin]})",
               f"{lin} — subtype_fine ({n_lab} states)", "study", "disease"])
    print(f"{lin}: {a.n_obs:,} cells, {n_lab} subtype_fine states")
```

```
<Figure size 3520x528 with 4 Axes>
Myeloid: 144,009 cells, 14 subtype_fine states

<Figure size 3520x528 with 4 Axes>
Fibroblast: 184,452 cells, 7 subtype_fine states
```

## §9 · The coarse layer

`subtype_fine` is the resolution the marker panels can defend; it is not always the resolution a
downstream contrast has donors for. `subtype_coarse` collapses it to the level that survives the
25-cell / 5-donor gate in most cohorts, and both are written to the sidecar so a consumer picks
its own granularity instead of re-deriving one.

The two monocyte levels are the clearest case of that split between what the markers defend and
what the donors support: `Mono_CD14` clears the gate with the ~52 donors the pooled `Mono` had,
`Mono_CD16` does not — 442 cells spread over 73 donors is ~6 cells each, so `n_donors_ge25` is
~0 and the coverage table below marks it unclaimable. That is the intended outcome, not a
failure: a real population the atlas can show and cannot yet make a claim about.

The collapse is by ontogeny, not by convenience: dendritic cells together, the monocyte-macrophage
continuum together, structural fibroblasts together. `Neutrophils` stays its own level even at 306
cells -- it is the level `22b` exists to test, and merging it here would erase the question.

```python  # cell 23
# ============================================================================
# §9  subtype_fine -> subtype_coarse
# ============================================================================
# "" is not used here: every fine label maps somewhere. Dropping happens downstream, against a
# measured coverage table, not silently inside a collapse.
FINE_TO_COARSE = {
    "Myeloid": {
        "LC": "LC",
        "cDC1": "DC", "cDC2": "DC", "DC_LAMP3": "DC", "pDC": "DC",
        "Mono_CD14": "Mono_Mac", "Mono_CD16": "Mono_Mac",
        "moDC": "Mono_Mac", "Mac_FOLR2": "Mono_Mac",
        "Mac_SPP1_TREM2": "Mono_Mac", "Mac_infl": "Mono_Mac", "Mac_ISG": "Mono_Mac",
        "Neutrophils": "Neutrophils",
        "Myeloid_prolif": "Prolif",
        "UNK": "UNK",
    },
    "Fibroblast": {
        "F_papillary": "F_structural", "F_reticular": "F_structural",
        "F_mesenchymal": "F_structural", "F_myofibroblast": "F_structural",
        "F_inflammatory": "F_inflammatory",
        "F_apCAF": "F_apCAF",
        "F_prolif": "Prolif",
        "UNK": "UNK",
    },
}

for lin in LINEAGES:
    a = A[lin]
    seen = set(a.obs["subtype_fine"].astype(str).unique())
    missing = seen - set(FINE_TO_COARSE[lin])
    assert not missing, f"{lin}: fine labels with no collapse rule: {sorted(missing)}"
    a.obs["subtype_coarse"] = pd.Categorical(
        a.obs["subtype_fine"].astype(str).map(FINE_TO_COARSE[lin]))

    print(f"\n{lin}:")
    print(pd.crosstab(a.obs["subtype_fine"].astype(str),
                      a.obs["subtype_coarse"].astype(str)).to_string())

    AU.save_umap(a, ["subtype_fine", "subtype_coarse"],
                 FIG / f"{TAGS[lin]}_umap_fine_vs_coarse.png", ncols=2, wspace=0.25,
                 title=[f"{lin} — subtype_fine ({a.obs['subtype_fine'].nunique()})",
                        f"{lin} — subtype_coarse ({a.obs['subtype_coarse'].nunique()})"])

# Per-donor coverage of both granularities inside the CTCL window — the number that decides
# what a downstream contrast may claim.
for lin in LINEAGES:
    ctcl = A[lin].obs[A[lin].obs["disease"].astype(str).isin(CTCL_DISEASES)]
    print(f"\n{lin} — CTCL window: {len(ctcl):,} cells, {ctcl['donor'].nunique()} donors")
    for col in ("subtype_fine", "subtype_coarse"):
        cov = AU.coverage(ctcl, col, min_cells=CCC_MIN_CELLS, min_samples=CCC_MIN_SAMPLES)
        cov.to_csv(TAB / f"skin_{TAGS[lin]}_coverage_{col}.csv")
        print(f"  --- {col} ---")
        print(cov.to_string())
```

```
Myeloid:
subtype_coarse     DC    LC  Mono_Mac  Neutrophils  Prolif    UNK
subtype_fine                                                     
DC_LAMP3        23723     0         0            0       0      0
LC                  0  4051         0            0       0      0
Mac_FOLR2           0     0     26475            0       0      0
Mac_SPP1_TREM2      0     0      4034            0       0      0
Mac_infl            0     0     14627            0       0      0
Mono_CD14           0     0      7187            0       0      0
Mono_CD16           0     0       442            0       0      0
Myeloid_prolif      0     0         0            0    1584      0
Neutrophils         0     0         0          306       0      0
UNK                 0     0         0            0       0  15399
cDC1            10684     0         0            0       0      0
cDC2            21689     0         0            0       0      0
moDC                0     0     12447            0       0      0
pDC              1361     0         0            0       0      0

<Figure size 1760x528 with 2 Axes>

Fibroblast:
subtype_coarse   F_apCAF  F_inflammatory  F_structural    UNK
subtype_fine                                                 
F_apCAF            27014               0             0      0
F_inflammatory         0           49966             0      0
F_mesenchymal          0               0         10696      0
F_myofibroblast        0               0           897      0
F_papillary            0               0         15376      0
F_reticular            0               0         49629      0
UNK                    0               0             0  30874

<Figure size 1760x528 with 2 Axes>

Myeloid — CTCL window: 139,764 cells, 124 donors
  --- subtype_fine ---
                n_cells  n_donors_present  n_donors_ge25  claimable
subtype_fine                                                       
Mac_FOLR2         25203               123             99       True
cDC2              20518               122             93       True
UNK               15123               123             85       True
cDC1              10512               118             77       True
moDC              12004               116             75       True
DC_LAMP3          23494               121             75       True
Mono_CD14          7180               114             50       True
Mac_infl          14485               103             44       True
LC                 3957               108             24       True
pDC                1349               108             18       True
Mac_SPP1_TREM2     3666               112             17       True
Myeloid_prolif     1534               106             12       True
Neutrophils         305                30              2      False
Mono_CD16           434                70              2      False
  --- subtype_coarse ---
                n_cells  n_donors_present  n_donors_ge25  claimable
subtype_coarse                                                     
Mono_Mac          62972               123            117       True
DC                55873               124            111       True
UNK               15123               123             85       True
LC                 3957               108             24       True
Prolif             1534               106             12       True
Neutrophils         305                30              2      False

Fibroblast — CTCL window: 160,413 cells, 124 donors
  --- subtype_fine ---
                 n_cells  n_donors_present  n_donors_ge25  claimable
subtype_fine                                                        
UNK                29864               119             87       True
F_apCAF            25921               119             83       True
F_inflammatory     40221               119             76       True
F_reticular        43403               110             73       True
F_mesenchymal       8330               111             59       True
F_p
... [567 more chars truncated]
```

## §10 · Cache the sidecar

`skin_myeloid_fibro_subtypes.csv` is what `helpers/ccc_data_sub.py` and the degradome config read
on behalf of `27_ccc_subtype` and `28_tme_degradome`. One row per myeloid/fibroblast skin cell,
keyed on `cell_id` -- never on row order, because the consumers join it onto objects built by a
different pass.

```python  # cell 25
# ============================================================================
# §10  skin_myeloid_fibro_subtypes.csv + provenance
# ============================================================================
merged = pd.concat([
    pd.DataFrame({
        "cell_id": A[lin].obs["cell_id"].astype(str).to_numpy(),
        "lineage": lin,
        "leiden_sub": A[lin].obs[KEY[lin]].astype(str).to_numpy(),
        "subtype_fine": A[lin].obs["subtype_fine"].astype(str).to_numpy(),
        "subtype_coarse": A[lin].obs["subtype_coarse"].astype(str).to_numpy(),
    }) for lin in LINEAGES], ignore_index=True)

AU.write_sidecar(
    merged, SUBTYPE_CSV, PROV_JSON,
    expect_rows=N_MYE + N_FIB,
    provenance={
        "notebook": "notebooks/20_skin/22_myeloid_fibro_reannotation.ipynb",
        "atlas": "v2",
        "source_h5ad": str(SRC_H5AD), "subset_h5ad": str(SUBSET_H5AD),
        "label_source": str(LABEL_CSV),
        "seed": SEED, "n_hvg": N_HVG, "n_latent": N_LATENT,
        "hvg_min_cells_per_batch": HVG_MIN_CELLS_PER_BATCH,
        "scvi_batch_key": "study", "scvi_categorical_covariate_keys": ["donor"],
        "scvi_models": {lin: str(paths(TAGS[lin])["scvi"]) for lin in LINEAGES},
        "leiden_res": LEIDEN_RES, "unk_leiden_res": globals().get("UNK_RES"),
        "ccc_min_cells": CCC_MIN_CELLS, "ccc_min_samples": CCC_MIN_SAMPLES,
        "ctcl_diseases": CTCL_DISEASES,
        "cluster2fine": CLUSTER2FINE, "cluster_sizes": CLUSTER_SIZES,
        "unk2fine": globals().get("UNK2FINE"),
        "unk_cluster_sizes": globals().get("UNK_CLUSTER_SIZES"),
        "fine_to_coarse": FINE_TO_COARSE,
        "n_cells": N_CELLS,
        "fine_levels": sorted(merged["subtype_fine"].unique()),
        "coarse_levels": sorted(merged["subtype_coarse"].unique()),
    })

print()
print(pd.crosstab(merged["lineage"], merged["subtype_coarse"]).to_string())
print("\nPaste into ccc_data_sub.py:")
for lin, name in (("Myeloid", "MYELOID_LEVELS"), ("Fibroblast", "FIBRO_LEVELS  ")):
    lv = sorted(set(merged.loc[merged.lineage == lin, "subtype_fine"]) - {"UNK"})
    print(f"  {name} = {lv}")
```

```
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/skin_myeloid_fibro_subtypes.csv  (328461, 5)
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/skin_myeloid_fibro_provenance.json

subtype_coarse     DC  F_apCAF  F_inflammatory  F_structural    LC  Mono_Mac  Neutrophils  Prolif    UNK
lineage                                                                                                 
Fibroblast          0    27014           49966         76598     0         0            0       0  30874
Myeloid         57457        0               0             0  4051     65212          306    1584  15399

Paste into ccc_data_sub.py:
  MYELOID_LEVELS = ['DC_LAMP3', 'LC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16', 'Myeloid_prolif', 'Neutrophils', 'cDC1', 'cDC2', 'moDC', 'pDC']
  FIBRO_LEVELS   = ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular']
```

## §11 · Reconcile against the v2 atlas

The sidecar claims to cover exactly the skin myeloid and fibroblast cells of atlas v2. This
asserts it: every `cell_id` present and unique in `atlas_obs_full_v2.parquet`, the covered set
equal to `cell_type_final in {Myeloid, Fibroblast}`, and every per-level count equal to what the
locked §6/§7d maps imply.

The expected counts are *derived from the maps*, not typed in -- so this is a real check that the
labels on the object match the dicts that were hand-filled, rather than a restatement of them.

```python  # cell 27
# ============================================================================
# §11  Reconcile the sidecar against atlas_obs_full_v2.parquet
# ============================================================================
from collections import Counter

# Expected per-level counts, derived from the two locked maps rather than transcribed.
expect = Counter()
for lin in LINEAGES:
    for cl, lab in CLUSTER2FINE[lin].items():
        if lab != "UNK":
            expect[lab] += CLUSTER_SIZES[lin][cl]
    for cl, lab in UNK2FINE.get(lin, {}).items():
        if lab != "UNK":
            expect[lab] += UNK_CLUSTER_SIZES[lin][cl]
    still_unk = sum(UNK_CLUSTER_SIZES[lin][cl]
                    for cl, lab in UNK2FINE.get(lin, {}).items() if lab == "UNK")
    expect["UNK"] += still_unk

assert sum(expect.values()) == N_MYE + N_FIB, (sum(expect.values()), N_MYE + N_FIB)

joined = AU.reconcile(
    merged, dict(expect),
    label_col="subtype_fine",
    where="compartment == 'Skin' and cell_type_final in ['Myeloid', 'Fibroblast']")

print()
print(pd.Series(expect).sort_values(ascending=False).to_string())
print(f"\ntotal {sum(expect.values()):,} = {N_MYE:,} myeloid + {N_FIB:,} fibroblast")
```

```
reconciled 328,461 rows against atlas_obs_full_v2.parquet; 20 level counts match

F_inflammatory     49966
F_reticular        49629
UNK                46273
F_apCAF            27014
Mac_FOLR2          26475
DC_LAMP3           23723
cDC2               21689
F_papillary        15376
Mac_infl           14627
moDC               12447
F_mesenchymal      10696
cDC1               10684
Mono_CD14           7187
LC                  4051
Mac_SPP1_TREM2      4034
Myeloid_prolif      1584
pDC                 1361
F_myofibroblast      897
Mono_CD16            442
Neutrophils          306

total 328,461 = 144,009 myeloid + 184,452 fibroblast
```
