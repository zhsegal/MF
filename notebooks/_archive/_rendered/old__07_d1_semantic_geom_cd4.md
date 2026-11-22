# old/07_d1_semantic_geom_cd4.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/07_d1_semantic_geom_cd4.ipynb` and is never run._

# SemanticSCVI (geometric) on D1 (Chennareddy 2025, GSE266862) — **CD4** cells

Replicates `06_semantic_geom_cd4.ipynb` on the Chennareddy *BJD* 2025 cohort.
Same SemanticSCVI config as the ref: Geneformer prior, geometric loss, 10
latents, 200 epochs, `coherence_weight=2000`, `batch_key=donor`.

**Cohort filter (advanced MF skin + HC):**
- 7 CD4⁺ MF patients (P76, P171, P204, P312, P311, P303, PGS) — all
  IIB–IVB, aggressive, skin biopsies.
- 4 healthy controls (P112, P115, P116, P121) — skin biopsies.
- Drop γδ MF and Berti lymphoma samples.

**Malignant CD4 calling — same method as the paper:**
The Chennareddy paper identifies malignant cells via **TCR clonality** —
*"the single top-expanded pathogenic T-cell clone for each sample"*. We apply
that method deterministically using the deposited 10x V(D)J files
(`*_clonotypes.csv.gz`, `*_filtered_contig_annotations.csv.gz`):

1. Per MF sample, find the dominant TCR clonotype (largest frequency).
2. Require dominance: ≥ 5 % of V(D)J-bearing cells **and** ≥ 2× the second
   clonotype (CTCL hallmark).
3. Cells in that clonotype → `is_malignant=True`.
4. All CD4⁺ T cells outside the dominant clone (HC + non-clonal MF) → benign.

After clonality calling we cluster CD4⁺ T cells on a standard scVI latent
and hand-pick clusters for sanity (mirrors ref cells 4–8 — but the
**clonotype call is primary**; clusters are confirmation).

**Workflow**
1. Run download / concat cells (one-off — skip if `concat.h5ad` exists).
2. Filter, QC, identify T cells, call malignant via clonotype.
3. Inspect cluster dot plot, optionally edit `CD4_MAL_CLUSTERS_OVERRIDE`.
4. Run the rest top-to-bottom on the `neural_nmf_env` GPU kernel.


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

# ---- D1 paths ----
D1_DIR   = NB_DIR / "data" / "D1_chennareddy2025"
RAW_DIR  = D1_DIR / "raw"
META_TSV = D1_DIR / "meta" / "patients.tsv"
CONCAT_H5 = D1_DIR / "processed" / "concat.h5ad"           # full 22-sample concat (post-QC)
CD4_H5    = D1_DIR / "processed" / "d1_cd4_combined.h5ad"  # advanced MF + HC, CD4+ T cells only
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "d1_cd4_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_d1_semantic_geom_cd4"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)

# ---- CD4 selection ----
TUMOR_LEIDEN_RESOLUTION = 0.5   # leiden on scVI latent of CD4 T cells
DOMINANT_FRAC_THRESH    = 0.05  # dominant clone ≥ 5 % of V(D)J cells in that sample
DOMINANT_RATIO_THRESH   = 2.0   # dominant clone ≥ 2× second clone

# ---- Preprocessing (ref values) ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = None              # use ALL malignant + benign CD4

# ---- Shared model size ----
N_LATENT = 10
BATCH_KEY = "donor"             # donor = participant_id

# ---- SemanticSCVI (Geneformer prior) — same as ref ----
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
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
semantic_geom cache_dir: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_d1_semantic_geom_cd4/semantic_scvi/91318ccb2b
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))   # notebooks/ holds benchmark_helpers.py

import gc, gzip, re
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.io as sio
import torch
import scvi

from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
sc.settings.verbosity = 1
print("scvi", scvi.__version__, "| scanpy", sc.__version__)
print("CUDA available:", torch.cuda.is_available())
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Seed set to 0

scvi 1.3.3 | scanpy 1.11.5
CUDA available: True

/tmp/ipykernel_2909126/2680766807.py:18: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scvi", scvi.__version__, "| scanpy", sc.__version__)
```

## Step 1 — Concatenate 22 samples from GSE266862 (one-off; cached)

```python  # cell 4
# Builds CONCAT_H5 from raw 10x mtx triples. Skips if cache exists.
CONCAT_H5.parent.mkdir(parents=True, exist_ok=True)

patients = pd.read_csv(META_TSV, sep="\t")
print(f"patients: {len(patients)} samples")
print(patients[["participant_id", "subset", "is_advanced_mf_skin"]].to_string(index=False))

if CONCAT_H5.exists():
    print(f"using cached {CONCAT_H5}")
    adata = sc.read_h5ad(CONCAT_H5)
else:
    adatas = {}
    for _, row in patients.iterrows():
        sid = row["sample_id_in_raw"]
        gsm = row["gsm_gex"]
        prefix = RAW_DIR / f"{gsm}_{sid}"
        mtx = Path(f"{prefix}_matrix.mtx.gz")
        bc  = Path(f"{prefix}_barcodes.tsv.gz")
        ft  = Path(f"{prefix}_features.tsv.gz")
        if not (mtx.exists() and bc.exists() and ft.exists()):
            print(f"  MISSING: {prefix}*  — skipping")
            continue
        # 10x mtx is genes x cells. Read manually for control over var.
        with gzip.open(mtx, "rb") as fh:
            X = sio.mmread(fh).tocsr().T.tocsr()        # cells x genes
        barcodes = pd.read_csv(bc, header=None, sep="\t")[0].astype(str).tolist()
        features = pd.read_csv(
            ft, header=None, sep="\t",
            names=["gene_id", "gene_name", "feature_type"],
        )
        a = sc.AnnData(X=X.astype(np.float32))
        a.obs_names = [f"{sid}_{b}" for b in barcodes]
        a.var_names = features["gene_name"].astype(str).tolist()
        a.var["gene_id"]   = features["gene_id"].values
        a.var["gene_name"] = features["gene_name"].values
        a.var_names_make_unique()
        a.obs["sample_id_in_raw"] = sid
        a.obs["gsm_gex"]          = gsm
        a.obs["participant_id"]   = row["participant_id"]
        a.obs["subset"]           = row["subset"]
        a.obs["disease"]          = row["disease"]
        a.obs["stage_class"]      = row["stage_class"]
        a.obs["is_advanced_mf_skin"] = bool(row["is_advanced_mf_skin"])
        adatas[sid] = a
        print(f"  {sid}: {a.shape}")

    print(f"\nconcatenating {len(adatas)} samples …")
    adata = sc.concat(adatas, axis=0, join="outer", index_unique=None, fill_value=0)
    # `donor` = participant_id (one donor per sample for most; P318 family handled below)
    adata.obs["donor"] = adata.obs["participant_id"].astype(str)
    # raw counts layer + housekeeping
    adata.layers["raw_counts"] = adata.X.copy()
    adata.write_h5ad(CONCAT_H5)
    print(f"wrote {CONCAT_H5}  | shape {adata.shape}")

print("\nsample breakdown:")
print(adata.obs["subset"].value_counts())
print(adata.obs["is_advanced_mf_skin"].value_counts())
```

```
patients: 22 samples
participant_id subset  is_advanced_mf_skin
          P112     HC                False
          P115     HC                False
          P116     HC                False
          P121     HC                False
           P76 CD4_MF                 True
          P171 CD4_MF                 True
          P204 CD4_MF                 True
          P312 CD4_MF                 True
          P311 CD4_MF                 True
          P303 CD4_MF                 True
           PGS CD4_MF                 True
          P218  Berti                False
          P195  Berti                False
          P230  gd_MF                False
           P95  gd_MF                False
          P127  gd_MF                False
           P71  gd_MF                False
     P318_thin  gd_MF                False
    P318_thick  gd_MF                False
           P97  gd_MF                False
          P124  gd_MF                False
          P202  Berti                False
using cached /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D1_chennareddy2025/processed/concat.h5ad

sample breakdown:
subset
gd_MF     114519
CD4_MF     96176
Berti      58529
HC         25094
Name: count, dtype: int64
is_advanced_mf_skin
False    198142
True      96176
Name: count, dtype: int64
```

## Step 2 — Manual annotation: scVI → Leiden → dotplot

Skip the sample-metadata pre-filter. Integrate the full 22-sample concat with a
quick scVI run, cluster the latent with Leiden, then inspect a canonical-marker
dotplot and edit `cluster_to_celltype` to label each cluster. Downstream steps
use `adata.obs["manual_celltype"]` (and/or `adata.obs["leiden_full"]`) to subset.

```python  # cell 6
# Quick scVI integration for clustering. Uses raw_counts via layer; adata.X untouched.
SCVI_INT_DIR = MODEL_CACHE_DIR / "scvi_int_step2"
SCVI_INT_DIR.parent.mkdir(parents=True, exist_ok=True)

if (SCVI_INT_DIR / "model.pt").exists():
    print(f"loading cached scVI from {SCVI_INT_DIR}")
    scvi_int = scvi.model.SCVI.load(str(SCVI_INT_DIR), adata=adata)
else:
    scvi.model.SCVI.setup_anndata(adata, batch_key="participant_id", layer="raw_counts")
    scvi_int = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    scvi_int.train(max_epochs=150, early_stopping=True, accelerator="auto")
    scvi_int.save(str(SCVI_INT_DIR), overwrite=True)
    print(f"saved scVI to {SCVI_INT_DIR}")
adata.obsm["X_scVI"] = scvi_int.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI", random_state=0)
sc.tl.leiden(adata, resolution=1.0, random_state=0, key_added="leiden_full")
sc.tl.umap(adata, random_state=0)
print("leiden cluster sizes:", adata.obs["leiden_full"].value_counts().to_dict())

sc.pl.umap(
    adata, color=["leiden_full", "donor", "subset", "disease"],
    ncols=2, wspace=0.3, frameon=False,
)
```

```
loading cached scVI from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_d1_semantic_geom_cd4/scvi_int_step2
INFO     File                                                                                                      
         /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_d1_semantic_geom_cd4/sc
         vi_int_step2/model.pt already downloaded                                                                  

/tmp/ipykernel_2909126/193181211.py:17: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  sc.tl.leiden(adata, resolution=1.0, random_state=0, key_added="leiden_full")

leiden cluster sizes: {'0': 35568, '1': 31379, '2': 26072, '3': 23037, '4': 18721, '5': 16397, '6': 15808, '7': 15668, '8': 15144, '9': 13392, '10': 13202, '11': 10333, '12': 8185, '13': 7997, '14': 7155, '15': 5579, '16': 5145, '17': 4441, '18': 4272, '19': 4262, '20': 3434, '21': 2853, '22': 1724, '23': 1220, '24': 1022, '25': 635, '26': 592, '27': 435, '28': 339, '29': 143, '30': 76, '31': 34, '32': 22, '33': 22, '34': 10}

<Figure size 1664x960 with 4 Axes>
```

```python  # cell 7
# Log-normalize a side copy for the full-cluster dotplot. Keep adata.X = raw counts.
ad_dp = adata.copy()
ad_dp.X = ad_dp.layers["raw_counts"].copy()
sc.pp.normalize_total(ad_dp, target_sum=1e4)
sc.pp.log1p(ad_dp)

t_markers = [g for g in
    ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "FOXP3", "TRAC", "TRBC1", "TRBC2"]
    if g in ad_dp.var_names]
dp = sc.pl.dotplot(
    ad_dp, t_markers, groupby="leiden_full",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "d1_full_t_cell_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "d1_full_t_cell_dotplot.png")
del ad_dp; gc.collect()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d1_full_t_cell_dotplot.png

90047
<Figure size 520x1325 with 4 Axes>
```

```python  # cell 8
# ✏️ FILL IN — leiden_full clusters that are T cells (CD3+). All other clusters are dropped.
T_CLUSTERS: list[str] = [
    "0", "1", "3", "4", "5", "9", "13", "20", "25", "26", "27", "28", "29", "31","34"
]
if not T_CLUSTERS:
    print("⚠️ T_CLUSTERS empty — adata unchanged. Fill in the list above and re-run.")
else:
    adata = adata[adata.obs["leiden_full"].astype(str).isin(T_CLUSTERS)].copy()
    print("T-cell subset:", adata.shape)
    print(adata.obs["leiden_full"].value_counts())
```

```
T-cell subset: (152113, 38224)
leiden_full
0     35568
1     31379
3     23037
4     18721
5     16397
9     13392
13     7997
20     3434
25      635
26      592
27      435
28      339
29      143
31       34
34       10
Name: count, dtype: int64
```

## Step 2b — Recluster T cells, annotate subtypes, subset CD4 + cache

Re-cluster the T-cell subset on the existing `X_scVI` latent for finer
resolution, then map each `t_leiden` cluster to a cell-type label via the
subtype dotplot. Subset to CD4 and cache the result for downstream steps.


```python  # cell 10
# Re-cluster T-cell subset on the existing scVI latent for finer subtype resolution.
sc.pp.neighbors(adata, use_rep="X_scVI", random_state=0)
sc.tl.leiden(adata, resolution=0.8, random_state=0, key_added="t_leiden")
sc.tl.umap(adata, random_state=0)
print("t_leiden cluster sizes:", adata.obs["t_leiden"].value_counts().to_dict())

# Dotplot of T-subtype markers.
ad_dp = adata.copy()
ad_dp.X = ad_dp.layers["raw_counts"].copy()
sc.pp.normalize_total(ad_dp, target_sum=1e4)
sc.pp.log1p(ad_dp)

subtype_markers = {
    "CD4":       ["CD4"],
    "CD8":       ["CD8A", "CD8B"],
    "Treg":      ["FOXP3", "IL2RA", "CTLA4"],
    "naive":     ["CCR7", "SELL", "LEF1", "TCF7"],
    "memory":    ["IL7R", "CD27", "CD28"],
    "effector":  ["GZMA", "GZMB", "GZMK", "PRF1", "GNLY", "NKG7"],
    "exhausted": ["PDCD1", "LAG3", "TIGIT", "HAVCR2"],
    "prolif":    ["MKI67", "TOP2A", "STMN1"],
}
subtype_markers = {k: [g for g in v if g in ad_dp.var_names] for k, v in subtype_markers.items()}
dp = sc.pl.dotplot(
    ad_dp, subtype_markers, groupby="t_leiden",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "d1_t_subtype_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "d1_t_subtype_dotplot.png")
del ad_dp; gc.collect()

# ✏️ EDIT THIS — map t_leiden cluster → cell type. Unmapped → "unknown".
```

```
t_leiden cluster sizes: {'0': 33389, '1': 22767, '2': 17569, '3': 15796, '4': 14171, '5': 12271, '6': 9874, '7': 8849, '8': 8309, '9': 3861, '10': 1655, '11': 1355, '12': 1118, '13': 653, '14': 433, '15': 33, '16': 10}
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d1_t_subtype_dotplot.png
t_celltype
CD4        117225
CD8         34878
unknown        10
Name: count, dtype: int64

<Figure size 1112x695 with 5 Axes>
<Figure size 1664x960 with 4 Axes>
```

```python  # cell 11
t_cluster_to_celltype: dict[str, str] = {
  "0": "CD4",
  "1": "CD4",
  "2": "CD4",
  "3": "CD8",
  "4": "CD4",
  "5": "CD4",
  "6": "CD4",
  "7": "CD4",
  "8": "CD4",
  "9": "CD4",
  "10": "CD4",
  "11": "CD4",
  "12": "CD4",
  "13": "CD4",
  "14": "UNK",
  "15": "CD4",
"16": "CD4",

}
adata.obs["t_celltype"] = (
    adata.obs["t_leiden"].astype(str).map(t_cluster_to_celltype).fillna("unknown")
)
print(adata.obs["t_celltype"].value_counts())

sc.pl.umap(
    adata, color=["t_leiden", "t_celltype", "disease", "donor"],
    ncols=2, wspace=0.3, frameon=False,
)
```

```
t_celltype
CD4    135884
CD8     15796
UNK       433
Name: count, dtype: int64

<Figure size 1664x960 with 4 Axes>
```

```python  # cell 12
adata_cd4 = adata[adata.obs["t_celltype"] == "CD4"].copy()
print("CD4 subset:", adata_cd4.shape)
```

```
CD4 subset: (135884, 38224)
```

```python  # cell 13
# Cache annotated CD4 adata (pre-TCR-rule).
CD4_ANNOT_H5 = D1_DIR / "processed" / "d1_cd4_annotated.h5ad"
CD4_ANNOT_H5.parent.mkdir(parents=True, exist_ok=True)
adata_cd4.write_h5ad(CD4_ANNOT_H5)
print("wrote", CD4_ANNOT_H5)
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D1_chennareddy2025/processed/d1_cd4_annotated.h5ad
```

## Step 4 — Apply paper's TCR-clonality rule: dominant clone per MF sample = malignant

For each MF sample we load `*_clonotypes.csv.gz` (clonotype frequencies)
and `*_filtered_contig_annotations.csv.gz` (per-barcode clonotype IDs).
The dominant clone — if it passes the dominance test — is tagged
`is_malignant=True`. HC samples have no malignant cells.

```python  # cell 15
# Reload cached annotated CD4 adata from end of Step 2b.
CD4_ANNOT_H5 = D1_DIR / "processed" / "d1_cd4_annotated.h5ad"
adata_cd4 = sc.read_h5ad(CD4_ANNOT_H5)
print("loaded", CD4_ANNOT_H5, "|", adata_cd4.shape)
```

```
loaded /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/D1_chennareddy2025/processed/d1_cd4_annotated.h5ad | (135884, 38224)
```

```python  # cell 16
import pandas as pd
import numpy as np
from pathlib import Path
DOMINANT_RATIO_THRESH=1.4
def _load_vdj_for_sample(sid, gsm_vdj):
    """Return (barcode -> clonotype_id), and the clonotype frequency table."""
    prefix = RAW_DIR / f"{gsm_vdj}_{sid}"
    contig_path = Path(f"{prefix}_filtered_contig_annotations.csv.gz")
    clono_path  = Path(f"{prefix}_clonotypes.csv.gz")
    
    if not contig_path.exists() or not clono_path.exists():
        return None, None
        
    contig = pd.read_csv(contig_path, usecols=["barcode", "raw_clonotype_id"])
    contig = contig.dropna(subset=["raw_clonotype_id"])
    contig = contig[contig["raw_clonotype_id"] != "None"]
    
    # FIX 1: Strip the common '-1' suffix from VDJ barcodes to ensure clean matching
    clean_vdj_barcodes = contig["barcode"].astype(str).str.replace(r"-1$", "", regex=True)
    bc2clone = dict(zip(clean_vdj_barcodes, contig["raw_clonotype_id"].astype(str)))
    
    clonos = pd.read_csv(clono_path)
    clonos = clonos.sort_values("frequency", ascending=False).reset_index(drop=True)
    
    return bc2clone, clonos

# Build per-sample malignant decision and per-cell call
mal_decisions = []
adata_cd4.obs["clonotype_id"]    = ""
adata_cd4.obs["is_malignant"]    = False
adata_cd4.obs["dom_clone_id"]    = ""
adata_cd4.obs["dom_clone_prop"]  = np.nan

# Assuming DOMINANT_FRAC_THRESH and DOMINANT_RATIO_THRESH are defined earlier in your script
# DOMINANT_FRAC_THRESH = 0.1  (example)
# DOMINANT_RATIO_THRESH = 2.0 (example)

for _, row in patients[
    patients["participant_id"].isin(adata_cd4.obs["participant_id"].astype(str).unique())
].iterrows():
    sid     = row["sample_id_in_raw"]
    pid     = row["participant_id"]
    disease = row["disease"]
    bc2clone, clonos = _load_vdj_for_sample(sid, row["gsm_vdj"])
    
    if bc2clone is None:
        print(f"  {pid}: NO V(D)J file — skipping clonality")
        continue

    # Decide dominance: top clonotype proportion + ratio
    if len(clonos) == 0:
        dom_id, dom_prop, ratio = "", 0.0, 0.0
    else:
        dom_id   = str(clonos.loc[0, "clonotype_id"])
        dom_prop = float(clonos.loc[0, "proportion"])
        dom_freq = float(clonos.loc[0, "frequency"])
        snd_freq = float(clonos.loc[1, "frequency"]) if len(clonos) > 1 else 0.0
        ratio    = dom_freq / max(snd_freq, 1.0)

    is_dominant = (
        (disease in ("MF",))
        and (dom_prop >= DOMINANT_FRAC_THRESH)
        and (ratio >= DOMINANT_RATIO_THRESH)
    )
    
    mal_decisions.append({
        "participant_id": pid, "sample_id": sid, "disease": disease,
        "dom_clone_id": dom_id, "dom_clone_prop": dom_prop, "dom_ratio": ratio,
        "is_dominant": is_dominant, "n_clones_total": len(clonos),
    })

    # Tag CD4 cells in this sample
    cell_mask = adata_cd4.obs["sample_id_in_raw"] == sid
    
    # FIX: Use rsplit("_", 1)[-1] to split at the LAST underscore. 
    # This correctly handles sample IDs that have underscores in them (like 'P303_skin').
    bc_short = [n.rsplit("_", 1)[-1].replace("-1", "") for n in adata_cd4.obs_names[cell_mask]]
    
    clone_ids = [bc2clone.get(bc, "") for bc in bc_short]
    
    adata_cd4.obs.loc[cell_mask, "clonotype_id"]   = clone_ids
    adata_cd4.obs.loc[cell_mask, "dom_clone_id"]   = dom_id
    adata_cd4.obs.loc[cell_mask, "dom_clone_prop"] = dom_prop
    if is_dominant:
        adata_cd4.obs.loc[cell_mask, "is_malignant"] = pd.Series(
            [c == dom_id and c != "" for c in clone_ids], index=adata_cd4.obs_names[cell_mask]
        )

# Print Summary
mal_df = pd.DataFrame(mal_decisions)
print("\nper-sample dominance decisions:")
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
    print(mal_df.to_string(index=False))

print("\nmalignant CD4 calls:")
print(adata_cd4.obs["is_malignant"].value_counts())

print("\nby sample:")
# FIX 3: Added `observed=True` (or False) to silence the pandas FutureWarning
print(adata_cd4.obs.groupby("participant_id", observed=True)["is_malignant"].agg(["sum", "size"]).to_string())
```

```
per-sample dominance decisions:
participant_id  sample_id disease dom_clone_id  dom_clone_prop  dom_ratio  is_dominant  n_clones_total
          P112       P112      HC   clonotype1           0.071      3.600        False             183
          P115       P115      HC   clonotype1           0.013      1.000        False             510
          P116       P116      HC   clonotype1           0.025      1.600        False             467
          P121       P121      HC   clonotype1           0.068      1.438        False             238
           P76        P76      MF   clonotype1           0.786   1253.857         True            2182
          P171       P171      MF   clonotype1           0.901     98.713         True             530
          P204       P204      MF   clonotype1           0.336      4.660         True            4421
          P312 P312_thick      MF   clonotype1           0.214      2.535         True            1136
          P311 P311_thick      MF   clonotype1           0.326      2.704         True            1566
          P303  P303_skin      MF   clonotype1           0.579      5.206         True             226
           PGS       P_GS      MF   clonotype1           0.468      1.455         True             631
          P218       P218      BL   clonotype1           0.605      2.040        False             492
          P195       P195      BL   clonotype1           0.433      4.420        False            4772
          P230       P230   gd_MF   clonotype1           0.170      4.050        False             317
           P95        P95   gd_MF   clonotype1           0.212      2.482        False            7716
          P127       P127   gd_MF   clonotype1           0.013      1.000        False            1851
           P71        P71   gd_MF   clonotype1           0.127      4.500        False             264
     P318_thin  P318_thin   gd_MF   clonotype1           0.008      1.053        False            1991
    P318_thick P318_thick   gd_MF   clonotype1           0.014      2.700        False            1652
           P97        P97   gd_MF   clonotype1           0.016      1.750        False            1098
          P124       P124   gd_MF   clonotype1           0.018      1.200        False            1173
          P202       P202      BL   clonotype1           0.002      1.167        False            4112

malignant CD4 calls:
is_malignant
False    113504
True      22380
Name: count, dtype: int64

by sample:
                 sum   size
participant_id             
P71                0    890
P76             6540  10671
P95                0  13065
P97                0   8574
P112               0    338
P115               0    694
P116               0    599
P121               0    314
P124               0   6853
P127               0   3460
P171            6184   8140
P195               0  12064
P202               0  19009
P204            2726  12098
P218               0   5447
P230               0    647
P303             827   1918
P311            2865   6515
P312             586   2642
P318_thick         0   7861
P318_thin          0   8730
PGS             2652   5355
```

## Step 8 — Build Geneformer semantic map, HVG-subset

```python  # cell 18
# Attach Ensembl gene_id from any features.tsv.gz (sc.concat dropped var cols).
if "gene_id" not in adata_cd4.var.columns:
    ft = next(RAW_DIR.glob("*_features.tsv.gz"))
    feats = pd.read_csv(ft, header=None, sep="\t", names=["gene_id", "gene_name", "feature_type"])
    sym2ens = dict(zip(feats["gene_name"].astype(str), feats["gene_id"].astype(str)))
    adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]

n_mapped = int(sum(str(g).startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")

# Rebuild map if cache exists with mismatched shape (stale from prior run).
if SEMANTIC_CACHE_GENEFORMER.exists():
    _cached = torch.load(SEMANTIC_CACHE_GENEFORMER, weights_only=False)
    if _cached.shape[0] != adata_cd4.n_vars:
        print(f"stale cache {tuple(_cached.shape)} != n_vars={adata_cd4.n_vars}; rebuilding")
        SEMANTIC_CACHE_GENEFORMER.unlink()
    del _cached

semantic_map = get_or_build_geneformer_map(
    adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
assert semantic_map.shape[0] == adata_cd4.n_vars, (
    f"semantic_map rows {semantic_map.shape[0]} != adata_cd4.n_vars {adata_cd4.n_vars}"
)
print("semantic_map (raw):", tuple(semantic_map.shape))

# Idempotent on re-run: only HVG-subset if not already at/below HVG_TOP_N.
if HVG_TOP_N is not None and adata_cd4.n_vars > HVG_TOP_N:
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
gene_id mapped to Ensembl: 36591/38224
Building Geneformer map (will cache to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/d1_cd4_geneformer.pt)

Fetching 3 files: 100%|██████████| 3/3 [00:00<00:00, 892.60it/s]

semantic_map (raw): (38224, 1152)
after HVG: (135884, 2500) | semantic_map: (2500, 1152)
no subsampling (n_obs=135884, SUBSAMPLE_N=None)
```

## Step 8b — Optionally drop healthy-control cells before training

```python  # cell 20
# Toggle: True → train on MF only (drop HC); False → keep HC.
EXCLUDE_HC = False

if EXCLUDE_HC:
    n_before = adata_cd4.n_obs
    keep_obs = adata_cd4.obs["disease"].astype(str) != "HC"
    adata_cd4 = adata_cd4[keep_obs].copy()
    print(f"dropped HC: {n_before} -> {adata_cd4.n_obs} cells")
    print(adata_cd4.obs["disease"].value_counts())
else:
    print(f"keeping HC (EXCLUDE_HC=False); n_obs={adata_cd4.n_obs}")
```

```python  # cell 21
adata_cd4=adata_cd4[adata_cd4.obs["participant_id"] != "PGS"].copy()
```

## Step 9 — Train SemanticSCVI (geometric, Geneformer prior) — GPU

```python  # cell 23
# Local override: flip to True to ignore the on-disk model cache and re-train.
FORCE_TRAIN_SEMANTIC_GENEFORMER = True

model = train_or_load_semantic_scvi(
    adata_cd4,
    semantic_map,
    cache_dir=SEMANTIC_GENEFORMER_CACHE_DIR,
    force_train=FORCE_TRAIN_SEMANTIC_GENEFORMER,
    max_epochs=SEMANTIC_GENEFORMER_MAX_EPOCHS,
    warmup_epochs=SEMANTIC_GENEFORMER_WARMUP_EPOCHS,
    n_epochs_kl_warmup=SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP,
    batch_key=BATCH_KEY,
    labels_key=None,
    **SEMANTIC_GENEFORMER_KWARGS,
)
print("trained:", model.is_trained)
```

```
[12:41:40] Training SemanticSCVI (max_epochs=200, warmup_epochs=20, n_epochs_kl_warmup=100)...

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
You are using a CUDA device ('NVIDIA A40') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Warmup Enabled: Semantic Loss (geometric) OFF for 20 epochs.
Epoch 200/200: 100%|██████████| 200/200 [28:19<00:00,  8.48s/it, v_num=1, train_loss_step=703, train_loss_epoch=711]
`Trainer.fit` stopped: `max_epochs=200` reached.

Epoch 200/200: 100%|██████████| 200/200 [28:19<00:00,  8.50s/it, v_num=1, train_loss_step=703, train_loss_epoch=711]
[13:09:59]   SemanticSCVI done in 1699.5s
[13:10:00]   saved to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_d1_semantic_geom_cd4/semantic_scvi/91318ccb2b
trained: True
```

## Step 10 — UMAP of the SemanticSCVI embedding

```python  # cell 25
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())
sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
sc.tl.umap(ad_emb, random_state=0)

color = [c for c in ["status", "disease", "participant_id", "stage_class"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```
<Figure size 1664x960 with 3 Axes>
```

## Step 11 — Per-factor separation: cells sorted by factor intensity, colored by status

```python  # cell 27
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

is_mal = ad_emb.obs["is_malignant"].astype(bool).values.astype(int)
n_factors = z.shape[1]

MAX_PER_CLASS = 15000
rng = np.random.default_rng(0)
idx_mal = np.where(is_mal == 1)[0]
idx_ben = np.where(is_mal == 0)[0]
if len(idx_mal) > MAX_PER_CLASS: idx_mal = rng.choice(idx_mal, MAX_PER_CLASS, replace=False)
if len(idx_ben) > MAX_PER_CLASS: idx_ben = rng.choice(idx_ben, MAX_PER_CLASS, replace=False)
plot_idx = np.concatenate([idx_mal, idx_ben])

ncols = 2
nrows = (n_factors + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.2 * nrows), sharex=False)
axes = axes.flatten()

aucs = {}
for k in range(n_factors):
    ax = axes[k]
    auc = roc_auc_score(is_mal, z[:, k])
    aucs[f"Z_{k}"] = max(auc, 1 - auc)
    sub_vals = z[plot_idx, k]; sub_mal = is_mal[plot_idx]
    order = np.argsort(sub_vals); vals = sub_vals[order]; mal = sub_mal[order]
    rank = np.arange(len(vals))
    ax.scatter(rank[mal == 1], vals[mal == 1], s=2, c="tab:red", alpha=0.35, rasterized=True, linewidths=0)
    ax.scatter(rank[mal == 0], vals[mal == 0], s=2, c="tab:blue", alpha=0.35, rasterized=True, linewidths=0)
    ax.set_title(f"Z_{k}  |  AUROC = {aucs[f'Z_{k}']:.3f}", fontsize=10)
    ax.set_xlabel(f"cell rank (n={MAX_PER_CLASS}/class)", fontsize=8)
    ax.set_ylabel(f"Z_{k}", fontsize=8)
    ax.tick_params(labelsize=7)

for j in range(n_factors, len(axes)): axes[j].axis("off")
fig.legend(handles=[
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",  label="tumor_cd4",  markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="benign_cd4", markersize=6),
], loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.tight_layout()
out = FIG_DIR / "d1_semantic_geom_cd4_factor_separation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()

print("\nAUROC (symmetric) per factor:")
print(pd.Series(aucs).sort_values(ascending=False).to_string())
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/d1_semantic_geom_cd4_factor_separation.png

<Figure size 1100x1100 with 10 Axes>

AUROC (symmetric) per factor:
Z_6    0.709224
Z_4    0.637810
Z_7    0.634753
Z_2    0.632723
Z_8    0.610419
Z_1    0.609926
Z_3    0.577282
Z_9    0.575211
Z_5    0.524113
Z_0    0.506951
```

### Per-factor heatmap: fraction tumor per quantile bin of Z_k

```python  # cell 29
N_BINS = 400
sub_z   = z[plot_idx]
sub_mal = is_mal[plot_idx]

heat = np.zeros((n_factors, N_BINS), dtype=float)
for k in range(n_factors):
    order = np.argsort(sub_z[:, k])
    mal_sorted = sub_mal[order].astype(float)
    heat[k] = np.array([chunk.mean() for chunk in np.array_split(mal_sorted, N_BINS)])

fig, ax = plt.subplots(figsize=(11, 0.45 * n_factors + 1.5))
im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
ax.set_yticks(range(n_factors))
ax.set_yticklabels([f"Z_{k}  (AUROC={aucs[f'Z_{k}']:.3f})" for k in range(n_factors)], fontsize=9)
ax.set_xticks([0, N_BINS // 2, N_BINS - 1])
ax.set_xticklabels(["low Z_k", "mid", "high Z_k"], fontsize=9)
ax.set_xlabel(f"cells sorted by Z_k (binned into {N_BINS} quantiles)", fontsize=9)
ax.set_title("Per-factor tumor/benign separation — class-balanced", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("fraction tumor_cd4 in bin")
fig.tight_layout()
out = FIG_DIR / "d1_semantic_geom_cd4_factor_separation_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

### Pick top-3 most-separating factors automatically

The ref used `Z_1, Z_3, Z_6` for its 3-D scatter. For D1 the ranking
likely differs — we let the script pick the top 3 by symmetric AUROC.
You can override `FX3, FY3, FZ3` below if you want fixed dims for
cross-dataset comparison.

```python  # cell 31
# Sort latent dims by symmetric AUROC, descending. Take top 3.
top_factors = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:3]
auto_idx = [int(k.split("_")[1]) for k, _ in top_factors]
print("auto top-3 factors:", top_factors)

# EDIT HERE to override for cross-dataset consistency (e.g. ref used [1,3,6]).
FX3, FY3, FZ3 = auto_idx[0], auto_idx[1], auto_idx[2]
print(f"using FX3={FX3}, FY3={FY3}, FZ3={FZ3}")
```

### 2-factor scatter grid for the top-5 factors

```python  # cell 33
from itertools import combinations

FACTORS_GRID = [int(k.split("_")[1]) for k, _ in sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:5]]
pairs = list(combinations(FACTORS_GRID, 2))
print("FACTORS_GRID:", FACTORS_GRID)

ncols = 5
nrows = (len(pairs) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
axes = axes.flatten()

rng_draw = np.random.default_rng(0)
shuf = rng_draw.permutation(len(plot_idx))
sub_mal = is_mal[plot_idx][shuf]
colors = np.where(sub_mal == 1, "tab:red", "tab:blue")

for ax, (fx, fy) in zip(axes, pairs):
    sub_x = z[plot_idx, fx][shuf]; sub_y = z[plot_idx, fy][shuf]
    ax.scatter(sub_x, sub_y, c=colors, s=2, alpha=0.3, linewidths=0, rasterized=True)
    ax.set_xlabel(f"Z_{fx}", fontsize=9); ax.set_ylabel(f"Z_{fy}", fontsize=9)
    ax.set_title(f"Z_{fx} vs Z_{fy}  ({aucs[f'Z_{fx}']:.2f}/{aucs[f'Z_{fy}']:.2f})", fontsize=9)
    ax.tick_params(labelsize=7)
for j in range(len(pairs), len(axes)): axes[j].axis("off")
fig.tight_layout()
out = FIG_DIR / f"d1_semantic_geom_cd4_scatter_grid_{'_'.join(map(str, FACTORS_GRID))}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

### 3-D scatter on (FX3, FY3, FZ3) with diagonal score s3

```python  # cell 35
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import roc_curve

x3 = z[:, FX3]; y3 = z[:, FY3]; zZ3 = z[:, FZ3]
mu_x3, sd_x3 = x3.mean(), x3.std()
mu_y3, sd_y3 = y3.mean(), y3.std()
mu_z3, sd_z3 = zZ3.mean(), zZ3.std()
xz3 = (x3 - mu_x3) / sd_x3
yz3 = (y3 - mu_y3) / sd_y3
zz3 = (zZ3 - mu_z3) / sd_z3
s3 = xz3 + yz3 + zz3
y_true = is_mal

# class-balanced + shuffled
rng3 = np.random.default_rng(0)
shuf3 = rng3.permutation(len(plot_idx))
xx3 = x3[plot_idx][shuf3]; yy3 = y3[plot_idx][shuf3]; zz3p = zZ3[plot_idx][shuf3]
ss3 = s3[plot_idx][shuf3]
cc3 = np.where(is_mal[plot_idx][shuf3] == 1, "tab:red", "tab:blue")

median_s3 = float(np.median(s3))
fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
auc_s3_raw = float(roc_auc_score(y_true, s3))
auc_s3 = max(auc_s3_raw, 1 - auc_s3_raw)

def s_plane(c, xg, yg):
    return sd_z3 * (c - (xg - mu_x3) / sd_x3 - (yg - mu_y3) / sd_y3) + mu_z3

xg = np.linspace(xx3.min(), xx3.max(), 20)
yg = np.linspace(yy3.min(), yy3.max(), 20)
XG, YG = np.meshgrid(xg, yg)
ZG_med  = s_plane(median_s3, XG, YG)
ZG_best = s_plane(best_thr3, XG, YG)
s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

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

fig.suptitle(f"D1 3-D scatter Z_{FX3}, Z_{FY3}, Z_{FZ3} with diagonal score s3", fontsize=12, y=1.02)
fig.tight_layout()
out = FIG_DIR / f"d1_semantic_geom_cd4_scatter3d_Z{FX3}_Z{FY3}_Z{FZ3}_with_s3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

### Unsupervised thresholds on s3

```python  # cell 37
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import fisher_exact

def report(name, y_pred):
    acc_a = accuracy_score(y_true, y_pred); acc_b = accuracy_score(y_true, 1 - y_pred)
    if acc_b > acc_a: y_pred = 1 - y_pred; acc = acc_b
    else: acc = acc_a
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return {"rule": name, "accuracy": acc, "precision_tumor": p, "recall_tumor": r,
            "f1_tumor": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "odds_ratio": odds, "fisher_p": pval}

rows3 = []
rows3.append(report("s3 > median(s3)", (s3 > np.median(s3)).astype(int)))
prior3 = y_true.mean()
thr_prior3 = np.quantile(s3, 1 - prior3)
rows3.append(report(f"top {prior3:.0%} of s3 (prior-matched)", (s3 > thr_prior3).astype(int)))
gmm3 = GaussianMixture(n_components=2, random_state=0).fit(s3.reshape(-1, 1))
rows3.append(report("GMM(2) on s3", gmm3.predict(s3.reshape(-1, 1))))
km3 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.c_[xz3, yz3, zz3])
rows3.append(report(f"KMeans(2) on (Z_{FX3},Z_{FY3},Z_{FZ3}) standardized", km3.labels_))
rows3.append(report(f"s3 > {best_thr3:.3f}  (best threshold — supervised)", (s3 > best_thr3).astype(int)))

results3 = pd.DataFrame(rows3)
print(f"AUROC of s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3}):  {auc_s3:.4f}")
print(f"Prior P(tumor):                                   {prior3:.3f}")
print(f"Trivial-majority baseline accuracy:               {max(prior3, 1-prior3):.3f}\n")
with pd.option_context("display.float_format", "{:.4f}".format,
                       "display.max_columns", None, "display.width", 200):
    print(results3.to_string(index=False))
```

## Step 12 — Gene loadings + hierarchical clustering

```python  # cell 39
W = model.get_loadings().reindex(adata_cd4.var_names)
adata_cd4.varm["W_d1_semantic_geom_cd4"] = W.values
adata_cd4.uns["W_d1_semantic_geom_cd4_columns"] = list(W.columns)
print("W (loadings):", W.shape, "| columns:", list(W.columns))

top_per_factor = {
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist()
    for col in W.columns
}
top_df = pd.DataFrame(top_per_factor)
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
top_df.to_csv(NB_DIR / "tables" / "d1_semantic_geom_cd4_top_genes_per_factor.tsv", sep="\t")
print("\ntop genes per factor:")
with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 30):
    print(top_df.head(15))
```

```python  # cell 40
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns

max_vals = W.abs().max(axis=1)
top_idx = max_vals.sort_values(ascending=False).head(CLUSTER_N_TOP).index
subset = W.loc[top_idx]
keep = subset.values.std(axis=1) > 0
subset = subset.loc[keep]; top_idx = subset.index

dists = pdist(subset.values, metric="correlation")
dist_matrix = squareform(dists)
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
g.fig.suptitle(f"D1 semantic_geom CD4: top-{len(top_idx)} gene loading correlation (k={best_k})", y=1.01)
out = FIG_DIR / "d1_semantic_geom_cd4_gene_hierarchy.png"
g.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
```
