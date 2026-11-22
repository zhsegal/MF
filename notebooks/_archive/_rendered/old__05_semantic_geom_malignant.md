# old/05_semantic_geom_malignant.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/05_semantic_geom_malignant.ipynb` and is never run._

# SemanticSCVI (geometric) on CTCL malignant cells

Trains the fork's `SemanticSCVI` geometric-loss factor model — the same **semantic_geom**
model used in `four_way_benchmark_haniffa_cd8.ipynb` (Geneformer prior, identical params) —
on the **tumor cells** of the CTCL atlas, then prints:

1. the **top loading genes per projection** (latent factor), and
2. a **hierarchical clustering** of the top-loaded genes.

Cells: `cell_type == "tumor_cell"` from the complete atlas. Batch key: `donor`.
Run on the `neural_nmf_env` GPU kernel.

```python  # cell 1
# ============================================================
# Parameters — edit here. Same knobs/names as four_way_benchmark_haniffa_cd8.ipynb.
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
ATLAS_PATH     = NB_DIR / "data" / "cache" / "mrvi_ctcl_cache.h5ad"           # complete atlas
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"     # gene_name -> gene_id only
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "mf_tumor_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_semantic_geom_tumor"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
CELL_TYPE = "tumor_cell"

# ---- Preprocessing (ref values) ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = 40000        # ref value; set None to use all ~102k tumor cells

# ---- Shared model size (ref) ----
N_LATENT = 10

# ---- Batch effect ----
BATCH_KEY = "donor"        # ref used "Site"; donor is this atlas's batch

# ---- SemanticSCVI (Geneformer prior) — EXACT ref values ----
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
PER_PROJECTION_N_TOP = 30   # top loading genes printed per projection
CLUSTER_N_TOP        = 500  # genes entering hierarchical clustering
CLUSTER_MAX_K        = 20   # silhouette search range for #clusters
TOP_PER_CLUSTER      = 25   # genes printed per hierarchical cluster
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
semantic_geom cache_dir: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_tumor/semantic_scvi/91318ccb2b
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))   # notebooks/ holds benchmark_helpers.py

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scvi

from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

scvi.settings.seed = 0
print("scvi", scvi.__version__, "| scanpy", sc.__version__)
print("CUDA available:", torch.cuda.is_available())
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Seed set to 0

scvi 1.3.3 | scanpy 1.11.5
CUDA available: True

/tmp/ipykernel_359356/1011679014.py:14: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scvi", scvi.__version__, "| scanpy", sc.__version__)
```

## Load tumor cells, attach Ensembl `gene_id`

```python  # cell 4
adata = sc.read_h5ad(ATLAS_PATH)
adata = adata[adata.obs["cell_type"].astype(str) == CELL_TYPE].copy()
adata.X = adata.layers["raw_counts"].copy()        # NB likelihood needs raw counts

# Geneformer tokens are keyed by Ensembl id; this atlas's var has only symbols.
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
adata.var["gene_id"] = [sym2ens.get(s, s) for s in adata.var_names.astype(str)]
n_mapped = int(sum(g.startswith("ENSG") for g in adata.var["gene_id"]))
print(adata.shape, "| cell_type:", adata.obs["cell_type"].unique().tolist())
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata.n_vars}")
```

```
(102367, 10000) | cell_type: ['tumor_cell']
gene_id mapped to Ensembl: 9766/10000
```

```python  # cell 5
adata = sc.read_h5ad(ATLAS_PATH)
adata.obs.value_counts()
```

```
tissue      sex  study             donor   tech  cell_type   stage  n_counts  _indices  _scvi_sample  _scvi_batch  _scvi_labels
Dermis      F    Sanger_Ncl_Fresh  CTCL1   10x   B_cell      IIB    743.0     13730     0             3            0               1
                                                                    838.0     15875     0             3            0               1
                                                                    940.0     14866     0             3            0               1
                                                                    1033.0    12099     0             3            0               1
                                                                    1207.0    12500     0             3            0               1
                                                                                                                                  ..
whole_skin  M    Sanger_Ncl_FFPE   CTCL18  flex  tumor_cell  IV     16676.0   418303    27            2            0               1
                                                                    16692.0   414271    27            2            0               1
                                                                    16916.0   419578    27            2            0               1
                                                                    16970.0   416806    27            2            0               1
                                                                    17182.0   413399    27            2            0               1
Name: count, Length: 419579, dtype: int64
```

```python  # cell 6
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

## Build Geneformer semantic map, then HVG-subset

```python  # cell 8
# Build the FULL-gene Geneformer map first (downloads Geneformer ~415 MB on first call),
# then HVG-subset adata + map together. Delete mf_tumor_geneformer.pt to force a rebuild.
semantic_map = get_or_build_geneformer_map(
    adata, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
print("semantic_map (raw):", tuple(semantic_map.shape))

if HVG_TOP_N is not None:
    sc.pp.highly_variable_genes(adata, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata.var["highly_variable"].values
    adata = adata[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
print("after HVG:", adata.shape, "| semantic_map:", tuple(semantic_map.shape))
```

```
Building Geneformer map (will cache to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_tumor_geneformer.pt)

Fetching 3 files: 100%|██████████| 3/3 [00:00<00:00,  5.54it/s]

semantic_map (raw): (10000, 1152)
after HVG: (102367, 2500) | semantic_map: (2500, 1152)
```

```python  # cell 9
if SUBSAMPLE_N is not None and adata.n_obs > SUBSAMPLE_N:
    sc.pp.subsample(adata, n_obs=SUBSAMPLE_N, random_state=42)
    print(f"subsampled to {adata.n_obs} cells")
else:
    print(f"no subsampling (n_obs={adata.n_obs}, SUBSAMPLE_N={SUBSAMPLE_N})")
```

```
subsampled to 40000 cells
```

## Train SemanticSCVI (geometric, Geneformer prior) — GPU

```python  # cell 11
model = train_or_load_semantic_scvi(
    adata,
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
[16:39:33] Training SemanticSCVI (max_epochs=200, warmup_epochs=20, n_epochs_kl_warmup=100)...
Warmup Enabled: Semantic Loss (geometric) OFF for 20 epochs.

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
You are using a CUDA device ('NVIDIA A40') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Epoch 200/200: 100%|██████████| 200/200 [10:52<00:00,  3.10s/it, v_num=1, train_loss_step=848, train_loss_epoch=863]      
`Trainer.fit` stopped: `max_epochs=200` reached.

Epoch 200/200: 100%|██████████| 200/200 [10:52<00:00,  3.26s/it, v_num=1, train_loss_step=848, train_loss_epoch=863]
[16:50:30]   SemanticSCVI done in 656.7s
[16:50:30]   saved to /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_tumor/semantic_scvi/91318ccb2b
trained: True
```

## UMAP of the SemanticSCVI embedding

```python  # cell 13
# UMAP of the SemanticSCVI latent embedding (batch_key=donor).
# Build a standalone AnnData from the model's registered adata so the latent and obs
# stay aligned regardless of the state of `adata` in the kernel.
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())
sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
sc.tl.umap(ad_emb, random_state=0)

color = [c for c in ["stage", "study", "tissue", "donor"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

## Gene loadings (projections)

```python  # cell 15
W = model.get_loadings().reindex(adata.var_names)   # genes x N_LATENT, indexed by gene symbol
adata.varm["W_semantic_geom"] = W.values
adata.uns["W_semantic_geom_columns"] = list(W.columns)
print("W (loadings):", W.shape, "| columns:", list(W.columns))
W.head()
```

```
W (loadings): (2500, 10) | columns: ['Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9']

               Z_0       Z_1       Z_2       Z_3       Z_4       Z_5  \
HES4      2.652264  0.039403  0.048648  0.011345  0.013453  0.013423   
ISG15     0.129656  0.006722  0.254838  0.037290  3.578124  0.095637   
RNF223    0.057485  0.166460  0.080636  0.035987  0.039556  0.084103   
TNFRSF18  3.059665  0.222652  0.171856  0.014217  2.975108  0.004890   
TNFRSF4   3.839775  0.322879  0.009722  0.036064  2.479763  0.003647   

               Z_6       Z_7       Z_8       Z_9  
HES4      0.138529  0.023366  0.017203  0.018721  
ISG15     0.102919  0.012847  1.638269  0.358292  
RNF223    4.481860  0.063819  0.043719  0.045792  
TNFRSF18  0.098528  0.016507  0.087677  0.054431  
TNFRSF4   0.357414  0.025916  0.027982  0.029705
```

### Top loading genes per projection

```python  # cell 17
top_per_factor = {
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist()
    for col in W.columns
}
top_df = pd.DataFrame(top_per_factor)
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
with pd.option_context("display.max_columns", None, "display.width", 250,
                       "display.max_colwidth", 30):
    print(top_df)
top_df
```

```
proj_Z_0 proj_Z_1  proj_Z_2 proj_Z_3 proj_Z_4 proj_Z_5   proj_Z_6 proj_Z_7  proj_Z_8  proj_Z_9
top_1      XCL1  ABHD12B    POU3F3    TXNIP     CCL5    SFRP2      CNTN3     PLK1      MT1X       FOS
top_2       TNF    GRIA1   COL10A1     RORA     NKG7      LUM       MYH7   DLGAP5      NQO1    DNAJB1
top_3      XCL2  GALNTL6     HMGA2   GPR183     CST7     MMP1      CASQ2    UBE2C      KLF2     DUSP1
top_4     RGS16    PCSK2     LBHD2    SESN3     GZMA      DCN      NETO1     ASPM    PYCARD     NR4A2
top_5      EGR1  ADAMTS8      FLG2   CDKN2D    GSTP1  COL15A1     PRKAA2   KIF20A    CORO1B      RGS2
top_6      SGK1    STOX2    ELOVL2   ZNF831     P4HB    P2RX2     PRSS35    CDC20      MT1E      FOSB
top_7      CD83  COL11A1     HSPA6     SMC4     CTSD     TDO2     GUCA2B    KIF14      UCP2      BAG3
top_8    NFKBID     FAT3   CLEC19A     CD28   CRELD2     SPIC      MYRIP    CDCA3       CLU       JUN
top_9    NFKBIZ    SYT13      PALM    NABP1   SDF2L1    CD163      CHRM1    CENPA   S100A10     NR4A1
top_10     IL13    ABCA4      MMP3     SOX4     HPGD      TNC       PCK1    GTSE1  C12orf75    ZNF331
top_11     CCL1     OCA2     CXCL1    SESN1   TXNRD1     OSMR     HAPLN1    BIRC5     KLRB1       GEM
top_12    KDM6B   LINGO2    NLGN4X   CHMP1B  SLC16A3   COL4A2      CADPS    TROAP      MT2A      AREG
top_13    ICAM1     RELN      GPX3     SELL   SEC11C   COL6A6       DLX5    CCNB2      PTMS   ZFAND2A
top_14     IER3    HTR2A  KIAA1755    TIAM1    PDIA4     VCAN       EBF2    HJURP    PLAAT4     DDIT3
top_15    KLF10    NPY1R     MAT1A   KIF21B  FAM162A  ADAMTS2  KIAA1549L     HBA2    LGALS1      RGS1
top_16    SOCS3   CHRDL1     FABP4    ITPKB     SSR3  SIGLEC1     SPINK1     HMMR    TALDO1      CCL4
top_17    NR4A3    TACR1       HRC     CCR4    PDIA6  SULT1C2      HPSE2    AURKB     STMN1      ATF3
top_18   TNFSF9    DCDC2   SULT1E1   MICAL1     IFI6   COL4A1    HMGCLL1    CCNA2      CAPG       HBB
top_19    DUSP5     GYG2     MUCL1   SLAMF1     MANF     C1QB      EDIL3     NEK2      KRT7      RHOB
top_20     IER5      CA3     HOXB5   HEXIM1    CXCR6     CPN2    ATP13A5    KIF23      MT1F   GADD45B
top_21     EGR2     PAK5     CLUL1    CCNG2     TYMP     MMP2       CHGA    CDCA8      AQP3     HMOX1
top_22   BCL2L1    EPHA5     AZGP1  TSC22D2     CTSB   CD300E     SHANK2    KIF2C  MARCKSL1      SAT1
top_23  TNFSF14   SLC6A7  SLC25A48     TMC8   BNIP3L    FSTL1       LCN2    CDCA2     PRDX2     CCL20
top_24    DUSP2   IGDCC3       VIT    PLCG1     GZMH  SELENOP      KCNK3   KIF18B     SLIRP     FOSL2
top_25   BCL2A1   SLC6A1      PDK4   KBTBD2    ITM2C     C1QC      SCN7A     ANLN     COTL1     PRMT9
top_26    IER5L    HTR1F     GSTA3     CHN1    SYTL3    MS4A2        BMX   CKAP2L       ID3     ARL4A
top_27     PLK3  FAM162B     NLGN1    ITGB1   UBALD2    APLNR      NRXN1     RRM2    GIMAP7  PPP1R15A
top_28   ADGRE5    OXGR1      IL18   OXNAD1   SLC1A4     KLK7      FGF23     CDK1     PRDX1      PER1
top_29    ARL5B    SALL1      SOX9     DGKA   FKBP11    CD209     GPR143   DEPDC1      NME1    CITED2
top_30    SOCS1     TAC1     CDH11    ABCC1    EFHD2   KCNK17       PLP1    TOP2A    IFITM3    TUBB2A

       proj_Z_0 proj_Z_1  proj_Z_2 proj_Z_3 proj_Z_4 proj_Z_5   proj_Z_6  \
top_1      XCL1  ABHD12B    POU3F3    TXNIP     CCL5    SFRP2      CNTN3   
top_2       TNF    GRIA1   COL10A1     RORA     NKG7      LUM       MYH7   
top_3      XCL2  GALNTL6     HMGA2   GPR183     CST7     MMP1      CASQ2   
top_4     RGS16    PCSK2     LBHD2    SESN3     GZMA      DCN      NETO1   
top_5      EGR1  ADAMTS8      FLG2   CDKN2D    GSTP1  COL15A1     PRKAA2   
top_6      SGK1    STOX2    ELOVL2   ZNF831     P4HB    P2RX2     PRSS35   
top_7      CD83  COL11A1     HSPA6     SMC4     CTSD     TDO2     GUCA2B   
top_8    NFKBID     FAT3   CLEC19A     CD28   CRELD2     SPIC      MYRIP   
top_9    NFKBIZ    SYT13      PALM    NABP1   SDF2L1    CD163      CHRM1   
top_10     IL13    ABCA4      MMP3     SOX4     HPGD      TNC       PCK1   
top_11  
... [2688 more chars truncated]
```

## Hierarchical clustering of top-loaded genes

Mirrors `benchmarking._get_hierarchical_clusters`: top genes by max |loading| →
correlation-distance → average linkage → #clusters chosen by silhouette.

```python  # cell 19
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

max_vals = W.abs().max(axis=1)
top_idx = max_vals.sort_values(ascending=False).head(CLUSTER_N_TOP).index
subset = W.loc[top_idx]

# correlation distance is undefined for zero-variance rows; drop them
keep = subset.values.std(axis=1) > 0
if not keep.all():
    print(f"dropping {(~keep).sum()} zero-variance genes before correlation pdist")
subset = subset.loc[keep]
top_idx = subset.index

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
    if s > best_score:
        best_score, best_k = s, k
final_labels = fcluster(Z, t=best_k, criterion="maxclust")
print(f"hierarchical clusters: k={best_k} (silhouette={best_score:.3f}) over {len(top_idx)} genes")

clust = pd.DataFrame(
    {"Cluster": final_labels, "max_loading": max_vals.loc[top_idx].values},
    index=top_idx,
)
for c in sorted(clust["Cluster"].unique()):
    members = clust[clust["Cluster"] == c].sort_values("max_loading", ascending=False)
    genes = members.head(TOP_PER_CLUSTER).index.tolist()
    print(f"\n=== cluster {c}  (n={len(members)}) ===")
    print(", ".join(genes))
```

```
hierarchical clusters: k=10 (silhouette=0.980) over 500 genes

=== cluster 1  (n=69) ===
FOS, DNAJB1, DUSP1, NR4A2, RGS2, FOSB, BAG3, JUN, NR4A1, ZNF331, GEM, AREG, ZFAND2A, DDIT3, RGS1, CCL4, ATF3, HBB, RHOB, GADD45B, HMOX1, SAT1, CCL20, FOSL2, PRMT9

=== cluster 2  (n=34) ===
XCL1, TNF, XCL2, RGS16, EGR1, SGK1, CD83, NFKBID, NFKBIZ, IL13, CCL1, KDM6B, ICAM1, IER3, KLF10, SOCS3, NR4A3, TNFSF9, DUSP5, IER5, EGR2, BCL2L1, TNFSF14, DUSP2, BCL2A1

=== cluster 3  (n=37) ===
CCL5, NKG7, CST7, GZMA, GSTP1, P4HB, CTSD, CRELD2, SDF2L1, HPGD, TXNRD1, SLC16A3, SEC11C, PDIA4, FAM162A, SSR3, PDIA6, IFI6, MANF, CXCR6, TYMP, CTSB, BNIP3L, GZMH, ITM2C

=== cluster 4  (n=47) ===
MT1X, NQO1, KLF2, PYCARD, CORO1B, MT1E, UCP2, CLU, S100A10, C12orf75, KLRB1, MT2A, PTMS, PLAAT4, LGALS1, TALDO1, STMN1, CAPG, KRT7, MT1F, AQP3, MARCKSL1, PRDX2, SLIRP, COTL1

=== cluster 5  (n=11) ===
TXNIP, RORA, GPR183, SESN3, CDKN2D, ZNF831, SMC4, CD28, NABP1, SOX4, SESN1

=== cluster 6  (n=3) ===
POU3F3, COL10A1, HMGA2

=== cluster 7  (n=3) ===
SFRP2, LUM, MMP1

=== cluster 8  (n=115) ===
ABHD12B, GRIA1, GALNTL6, PCSK2, ADAMTS8, STOX2, COL11A1, FAT3, SYT13, ABCA4, OCA2, LINGO2, RELN, HTR2A, NPY1R, CHRDL1, TACR1, DCDC2, GYG2, CA3, PAK5, EPHA5, SLC6A7, IGDCC3, SLC6A1

=== cluster 9  (n=45) ===
PLK1, DLGAP5, UBE2C, ASPM, KIF20A, CDC20, KIF14, CDCA3, CENPA, GTSE1, BIRC5, TROAP, CCNB2, HJURP, HBA2, HMMR, AURKB, CCNA2, NEK2, KIF23, CDCA8, KIF2C, CDCA2, KIF18B, ANLN

=== cluster 10  (n=136) ===
CNTN3, MYH7, CASQ2, NETO1, PRKAA2, PRSS35, GUCA2B, MYRIP, CHRM1, PCK1, HAPLN1, CADPS, DLX5, EBF2, KIAA1549L, SPINK1, HPSE2, HMGCLL1, EDIL3, ATP13A5, CHGA, SHANK2, LCN2, KCNK3, SCN7A
```

```python  # cell 20
import seaborn as sns

corr = np.corrcoef(subset.values)   # gene x gene correlation across projections
g = sns.clustermap(
    pd.DataFrame(corr, index=top_idx, columns=top_idx),
    row_linkage=Z, col_linkage=Z, cmap="vlag", center=0,
    xticklabels=False, yticklabels=False, figsize=(8, 8),
)
g.fig.suptitle(
    f"semantic_geom tumor: top-{len(top_idx)} gene loading correlation (k={best_k})", y=1.01
)
out = FIG_DIR / "semantic_geom_tumor_gene_hierarchy.png"
g.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_tumor_gene_hierarchy.png

<Figure size 800x800 with 4 Axes>
```
