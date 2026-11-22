# old/06_semantic_geom_cd4.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/06_semantic_geom_cd4.ipynb` and is never run._

# SemanticSCVI (geometric) on CTCL **CD4** cells — tumor CD4 + benign Th

Same model and knobs as `05_semantic_geom_malignant.ipynb` (Geneformer prior,
geometric loss, 10 latents, 200 epochs, batch=`donor`), but trained on a
**CD4-only** subset:

- **Tumor CD4**: hand-picked leiden clusters from the MRVI-embedded tumor cells
  (selected on a `CD4` / `CD8A` / `CD8B` dot plot below).
- **Benign CD4**: all `cell_type == "Th"` cells.

A `status` obs flag (`tumor_cd4` / `benign_cd4`) is added so the latent
embedding and the loadings can be inspected for malignancy-specific programs.

**Workflow**

1. Run cells 1–4 (load, leiden, dot plot).
2. Inspect the dot plot, then **edit `CD4_TUMOR_CLUSTERS`** in the hand-pick cell.
3. Run the rest top-to-bottom on the `neural_nmf_env` GPU kernel.

```python  # cell 1
# ============================================================
# Parameters — same knobs/names as 05_semantic_geom_malignant.ipynb.
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
ATLAS_PATH     = NB_DIR / "data" / "cache" / "mrvi_ctcl_cache.h5ad"           # complete atlas (HVGs + X_mrvi)
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"     # gene_name -> gene_id only
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "mf_cd4_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_semantic_geom_cd4"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)

# ---- CD4 selection ----
BENIGN_CELL_TYPE = "Th"        # benign CD4 reference (= cell_type "Th")
TUMOR_LEIDEN_RESOLUTION = 0.5  # leiden resolution on MRVI latent within tumor cells

# ---- Preprocessing (ref values from 05) ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = None              # use ALL hand-picked tumor CD4 + Th

# ---- Shared model size (ref) ----
N_LATENT = 10

# ---- Batch effect ----
BATCH_KEY = "donor"

# ---- SemanticSCVI (Geneformer prior) — same as 05 ----
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
semantic_geom cache_dir: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4/semantic_scvi/91318ccb2b
```

```python  # cell 2
import sys
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))   # notebooks/ holds benchmark_helpers.py

import gc
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
Seed set to 0
/tmp/ipykernel_359356/2050083769.py:15: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scvi", scvi.__version__, "| scanpy", sc.__version__)

scvi 1.3.3 | scanpy 1.11.5
CUDA available: True
```

## Load atlas, cluster tumor cells on the MRVI latent

Leiden + UMAP on `obsm["X_mrvi"]` within `cell_type == "tumor_cell"` only — gives
sharper resolution than the global tumor-vs-everything UMAP.

```python  # cell 4
adata = sc.read_h5ad(ATLAS_PATH)
print("atlas:", adata.shape)

ad_tumor = adata[adata.obs["cell_type"].astype(str) == "tumor_cell"].copy()
print("tumor:", ad_tumor.shape)

sc.pp.neighbors(ad_tumor, use_rep="X_mrvi", random_state=0)
sc.tl.umap(ad_tumor, random_state=0)
sc.tl.leiden(ad_tumor, resolution=TUMOR_LEIDEN_RESOLUTION, random_state=0, key_added="tumor_leiden")
print("leiden clusters:", ad_tumor.obs["tumor_leiden"].value_counts().to_dict())

sc.pl.umap(
    ad_tumor,
    color=["tumor_leiden", "donor", "stage"],
    ncols=2, wspace=0.3, frameon=False,
    save=None,
)
```

```
atlas: (419579, 10000)
tumor: (102367, 10000)
leiden clusters: {'0': 30556, '1': 19135, '2': 18779, '3': 13231, '4': 7538, '5': 6746, '6': 3859, '7': 2119, '8': 404}

<Figure size 1664x960 with 3 Axes>
```

```python  # cell 5
adata = sc.read_h5ad(ATLAS_PATH)
print("atlas:", adata.shape)

ad_tumor = adata[adata.obs["cell_type"].isin(["tumor_cell",'Th'])].copy()
print("tumor:", ad_tumor.shape)

sc.pp.neighbors(ad_tumor, use_rep="X_mrvi", random_state=0)
sc.tl.umap(ad_tumor, random_state=0)
sc.tl.leiden(ad_tumor, resolution=TUMOR_LEIDEN_RESOLUTION, random_state=0, key_added="tumor_leiden")
print("leiden clusters:", ad_tumor.obs["tumor_leiden"].value_counts().to_dict())

sc.pl.umap(
    ad_tumor,
    color=["tumor_leiden", "donor", "stage"],
    ncols=2, wspace=0.3, frameon=False,
    save=None,
)
```

```
atlas: (419579, 10000)
tumor: (155191, 10000)
leiden clusters: {'0': 35488, '1': 21743, '2': 21505, '3': 18365, '4': 18183, '5': 14481, '6': 9292, '7': 7383, '8': 6283, '9': 2108, '10': 360}

<Figure size 1664x960 with 3 Axes>
```

## Dot plot — CD4 / CD8A / CD8B per tumor leiden cluster

`mrvi_ctcl_cache.h5ad`'s `adata.X` is already log-normalized; the dot plot
uses it directly. Use this plot to decide which leiden clusters are CD4+.

```python  # cell 7
markers = [g for g in ["CD4", "CD8A", "CD8B"] if g in ad_tumor.var_names]
print("markers present:", markers)

dp = sc.pl.dotplot(
    ad_tumor, var_names=markers, groupby="tumor_leiden",
    standard_scale="var", return_fig=True,
)
dp.savefig(FIG_DIR / "semantic_geom_cd4_tumor_dotplot.png", dpi=200, bbox_inches="tight")
print("saved", FIG_DIR / "semantic_geom_cd4_tumor_dotplot.png")

# Mean log-norm expression per cluster, for reference
mean_expr = (
    pd.DataFrame(
        ad_tumor[:, markers].X.toarray() if hasattr(ad_tumor[:, markers].X, "toarray") else ad_tumor[:, markers].X,
        columns=markers, index=ad_tumor.obs_names,
    )
    .assign(tumor_leiden=ad_tumor.obs["tumor_leiden"].values)
    .groupby("tumor_leiden").mean()
    .sort_index()
)
print("mean log-norm expression per tumor leiden cluster:")
print(mean_expr.round(3))
```

```
markers present: ['CD4', 'CD8A']

saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_tumor_dotplot.png
mean log-norm expression per tumor leiden cluster:
                CD4   CD8A
tumor_leiden              
0             0.408  0.119
1             0.436  0.167
2             0.183  0.037
3             0.168  0.078
4             0.249  0.029
5             0.319  0.028
6             0.077  0.011
7             0.905  0.093
8             0.042  0.725

/tmp/ipykernel_359356/2249869138.py:18: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  .groupby("tumor_leiden").mean()

<Figure size 224x415 with 4 Axes>
```

## 🛑 Hand-pick CD4 tumor clusters

**Inspect the dot plot above**, then edit the list below to the leiden cluster
IDs (as strings) that look CD4+ (high `CD4`, low `CD8A`/`CD8B`). Re-run from this
cell down.

```python  # cell 9
# EDIT THIS LIST after inspecting the dot plot above.
CD4_TUMOR_CLUSTERS: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"] 

assert CD4_TUMOR_CLUSTERS, "Inspect the dot plot, then fill CD4_TUMOR_CLUSTERS"

ad_tumor_cd4 = ad_tumor[ad_tumor.obs["tumor_leiden"].isin(CD4_TUMOR_CLUSTERS)].copy()
print(f"selected {ad_tumor_cd4.n_obs} tumor CD4 cells "
      f"from {ad_tumor.n_obs} total tumor (clusters {CD4_TUMOR_CLUSTERS})")
print("by donor:")
print(ad_tumor_cd4.obs["donor"].value_counts().head(10))
```

```
selected 101963 tumor CD4 cells from 102367 total tumor (clusters ['0', '1', '2', '3', '4', '5', '6', '7'])
by donor:
donor
MF28      11072
CTCL3      9524
CTCL18     8223
CTCL1      8068
CTCL5      7722
CTCL8      7134
CTCL13     6100
MF14       4183
PT35       4091
CTCL6      3773
Name: count, dtype: int64
```

## Build the CD4 adata (tumor CD4 + benign Th)

Concat with a `status` obs flag, set `X = layers["raw_counts"]` for the NB
likelihood, then free the large parents.

```python  # cell 11
ad_benign = adata[adata.obs["cell_type"].astype(str) == BENIGN_CELL_TYPE].copy()
print("benign Th:", ad_benign.shape)

adata_cd4 = sc.concat(
    [ad_tumor_cd4, ad_benign],
    axis=0, join="inner", label="status", keys=["tumor_cd4", "benign_cd4"],
    index_unique=None,
)
adata_cd4.X = adata_cd4.layers["raw_counts"].copy()  # NB likelihood needs raw counts
print("combined:", adata_cd4.shape,
      "| status:", adata_cd4.obs["status"].value_counts().to_dict())

# free memory before training
del adata, ad_tumor, ad_benign
gc.collect()
```

```
benign Th: (52824, 10000)
combined: (154787, 10000) | status: {'tumor_cd4': 101963, 'benign_cd4': 52824}

24052
```

## Attach Ensembl `gene_id` (Geneformer keys by Ensembl)

```python  # cell 13
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
n_mapped = int(sum(g.startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")
```

```
gene_id mapped to Ensembl: 2438/2500
```

## Build Geneformer semantic map, then HVG-subset

HVGs are recomputed on the CD4 subset (tumor + benign), so a separate cache file
(`mf_cd4_geneformer.pt`) is used.

```python  # cell 15
# Build the FULL-gene Geneformer map first (cached). Then HVG-subset adata + map together.
# Delete mf_cd4_geneformer.pt to force a rebuild.
semantic_map = get_or_build_geneformer_map(
    adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
print("semantic_map (raw):", tuple(semantic_map.shape))

# Stale-cache guard: if the cached map was built on a previously HVG-subset adata,
# its rows won't match the current full-gene adata. Rebuild in that case.
if semantic_map.shape[0] != adata_cd4.n_vars:
    print(f"shape mismatch ({semantic_map.shape[0]} vs {adata_cd4.n_vars}) — rebuilding")
    SEMANTIC_CACHE_GENEFORMER.unlink()
    semantic_map = get_or_build_geneformer_map(
        adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
    )
    print("semantic_map (rebuilt):", tuple(semantic_map.shape))

if HVG_TOP_N is not None:
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
Loading cached Geneformer map from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_cd4_geneformer.pt
semantic_map (raw): (2500, 1152)
after HVG: (154787, 2500) | semantic_map: (2500, 1152)
no subsampling (n_obs=154787, SUBSAMPLE_N=None)
```

## Train SemanticSCVI (geometric, Geneformer prior) — GPU

```python  # cell 17
model = train_or_load_semantic_scvi(
    adata_cd4,
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
INFO     File                                                                                                      
         /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4/seman
         tic_scvi/91318ccb2b/model.pt already downloaded                                                           
[09:03:04]   loaded SemanticSCVI from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4/semantic_scvi/91318ccb2b
trained: True
```

## UMAP of the SemanticSCVI embedding

Colored by `status` (tumor vs benign), `stage`, `study`, `tissue`, `donor`.

```python  # cell 19
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())
sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
sc.tl.umap(ad_emb, random_state=0)

color = [c for c in ["status", "stage", "study", "tissue", "donor"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```
<Figure size 1664x1440 with 5 Axes>
```

```python  # cell 20


color = [c for c in ["cell_type", "stage", "study","donor"] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```
<Figure size 1664x960 with 4 Axes>
```

## Top loading genes per projection (quick look)

```python  # cell 22
W = model.get_loadings().reindex(adata_cd4.var_names)
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
proj_Z_0  proj_Z_1  proj_Z_2 proj_Z_3   proj_Z_4   proj_Z_5  proj_Z_6 proj_Z_7   proj_Z_8 proj_Z_9
top_1        MMP3    CXCL13    DEFB4A     KLF2     SLC1A2    MINDY4B      GZMB    MYOD1       PCP4   TUBA1C
top_2      CITED1     EGLN3     KRT19      FOS      DPEP1       WIF1      CCL4   KIF20A      TRPM1     NME1
top_3       TREM1    ADGRG1      IL31      CD7       IDO1    C5orf49      XCL1   DLGAP5     GUCA2B     ODC1
top_4        IL18     NTRK2      AQP5    DUSP1      IGHA2        HBB      XCL2    UBE2C      NETO1   TUBA1B
top_5     CEACAM4     RDH10     EDIL3     SELL   ARHGAP40     IL36RN      CCL3     PLK1      CNTN3    HSPD1
top_6      RNASE2      TNS3     CXCL5    NR4A2     COX4I2      FABP4     HMOX1     ASPM      CASQ2     CYCS
top_7        C1QB     CCL20    S100A1     CD69   SERPINB2        DCD      GNLY    CDC20       NNAT   TUBB2A
top_8         CR1       CLU  C22orf42   GIMAP4     CLSTN2      TPSD1      GZMH    CENPA     CLEC4F   SDF2L1
top_9        AQP9      ATF5    MLXIPL     JUNB       RND3       RBP1      NKG7      IL9    GALNTL6    DDIT3
top_10    EPB41L3       CD9      ALPG    TAGAP      KRT15     OR51E1      IL13    KIF14     SEL1L2      FTL
top_11     CD300E     P4HA2     CLDN8   GPR183   TNFRSF17       FLG2  C15orf48     NEK2      EPHA5  SLC25A5
top_12      C5AR1    BCL2L1    DIRAS1     FOSB       MMP7       LRP4      IFNG    HJURP     RNF223    FABP5
top_13        DPT     FURIN     OBP2B    KLRB1       VCAN       CDX1      CCL5    CDCA3      KCNK3     TUBB
top_14    COLEC12    SEMA4A    SLC4A1    S1PR1       GNG8       GJA4    CCL4L2    GTSE1      TKTL2   TXNRD1
top_15     PLA2G7      ZEB2     FIGLA      JUN       TCN1       HBA2       CGA    KIF23      GPR33     NQO1
top_16      KCNJ6   TSPAN13      TCIM   DNAJB1      MAT1A      CCL16      G0S2    BIRC5    COL11A1     MANF
top_17   TMEM176B  SERPINA1       IL5     CD55      CCBE1      CD207      IER3    CCNA2      MYOZ1     CKS2
top_18       C1QC       MSC       CCK     RGCC       APOD       PTH2      IL22   DEPDC1     PLCXD3     POMP
top_19       DAB2      ETV7     ADCY2     ICOS       IL24       CDH2      EGR2   KIF18B     GRIN2A    KPNA2
top_20        LYZ    TWIST1    TRIM63    ZFP36  KRTAP19-5   SERPINA5    TNFSF9    CCNB2       PCK1     MT2A
top_21     RNASE1     PTHLH       HRC     CCR7        CPE      LCE3E     HBEGF    KIF2C      STAB2    HSPB1
top_22  ARHGEF10L    GOLIM4       PYY    SOCS3       AHSP   SERPINA9      ATF3    CDCA2      LYPD2    CMSS1
top_23    SIGLEC1      IRF4      CPN2     FCMR       PMCH     GPR182      CSF1    TROAP     PLPPR4     MT1X
top_24       FPR2     MTSS1    SPINK1   ZNF331      FOXF1       CMA1     VCAM1    CDCA8      PCSK2    HMGA1
top_25       PODN    RCBTB2     HSPA6  GADD45B       MISP       LIM2     CEBPD     HMMR       LCN2   SEC61B
top_26       SOD3      NBAS     MUCL1    TXNIP       KERA      MS4A2     RASD1    AURKB  KIAA1549L  ZFAND2A
top_27     SLC8A1    GALNT2     GPR31     CD8A       NRP2    S100A7A      CD83   CKAP2L     SHANK2     NME2
top_28    SLCO2B1      SOX4      PRR9     PER1       RIN2  KRTAP4-11       GEM     ANLN      VEPH1    C1QBP
top_29     SIRPB2    ADAM19   TMEM210   SLC2A3      CCL11      IL36G     RAMP1     RRM2    SLITRK2   SNRPD1
top_30       C1QA      CTSH   MKRN2OS     BTG2       EBI3       RPTN      CCL1      PBK      P2RX2   DNAJA1

         proj_Z_0  proj_Z_1  proj_Z_2 proj_Z_3   proj_Z_4   proj_Z_5  \
top_1        MMP3    CXCL13    DEFB4A     KLF2     SLC1A2    MINDY4B   
top_2      CITED1     EGLN3     KRT19      FOS      DPEP1       WIF1   
top_3       TREM1    ADGRG1      IL31      CD7       IDO1    C5orf49   
top_4        IL18     NTRK2      AQP5    DUSP1      IGHA2        HBB   
top_5     CEACAM4     RDH10     EDIL3     SELL   ARHGAP40     IL36RN   
top_6      RNASE2      TNS3     CXCL5    NR4A2     COX4I2      FABP4   
top_7        C1QB     CCL20    S100A1     CD69   SERPINB2        DCD   
top_8         CR1       CLU  C22orf42   GIMAP4     CLSTN2      TPSD1   
top_9       
... [3058 more chars truncated]
```

### Heatmap view: per-factor sorted status bands

For each factor `Z_k`, cells are sorted by `Z_k` (low → high left → right) and
binned along the x-axis; the color of each bin is the fraction of tumor cells
(red = pure tumor, blue = pure benign, white = mixed). A factor that separates
well will show a clean red↔blue gradient along its row. Class-balanced
(`MAX_PER_CLASS` per class) so the fractions aren't biased by the 2:1 tumor/benign
imbalance.

```python  # cell 24
N_BINS = 400

# Class-balanced indices (reuses plot_idx from the scatter cell)
sub_z   = z[plot_idx]                # (2*MAX_PER_CLASS, n_factors)
sub_mal = is_mal[plot_idx]           # 0/1

heat = np.zeros((n_factors, N_BINS), dtype=float)
for k in range(n_factors):
    order = np.argsort(sub_z[:, k])
    mal_sorted = sub_mal[order].astype(float)
    # mean status per bin = fraction tumor in that quantile of Z_k
    heat[k] = np.array([chunk.mean() for chunk in np.array_split(mal_sorted, N_BINS)])

fig, ax = plt.subplots(figsize=(11, 0.45 * n_factors + 1.5))
im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
               interpolation="nearest")
ax.set_yticks(range(n_factors))
ax.set_yticklabels(
    [f"Z_{k}  (AUROC={aucs[f'Z_{k}']:.3f})" for k in range(n_factors)],
    fontsize=9,
)
ax.set_xticks([0, N_BINS // 2, N_BINS - 1])
ax.set_xticklabels(["low Z_k", "mid", "high Z_k"], fontsize=9)
ax.set_xlabel(f"cells sorted by Z_k (binned into {N_BINS} quantiles)", fontsize=9)
ax.set_title("Per-factor tumor/benign separation — class-balanced", fontsize=11)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("fraction tumor_cd4 in bin")
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["all benign", "0.5", "all tumor"])

fig.tight_layout()
out = FIG_DIR / "semantic_geom_cd4_factor_separation_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_factor_separation_heatmap.png

<Figure size 1100x600 with 2 Axes>
```

### 2-factor scatter — pick two `Z_k`, color cells by status

Edit `FACTOR_X` / `FACTOR_Y` below to any pair from `0..N_LATENT-1`. Same
class-balanced subsample as above (`plot_idx`) so red doesn't drown blue.

```python  # cell 26
# EDIT THESE.
FACTOR_X = 6
FACTOR_Y = 3

sub_x = z[plot_idx, FACTOR_X]
sub_y = z[plot_idx, FACTOR_Y]
sub_mal = is_mal[plot_idx]

# Shuffle draw order so neither class systematically covers the other.
rng_draw = np.random.default_rng(0)
shuf = rng_draw.permutation(len(sub_x))
sub_x, sub_y, sub_mal = sub_x[shuf], sub_y[shuf], sub_mal[shuf]

colors = np.where(sub_mal == 1, "tab:red", "tab:blue")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(sub_x, sub_y, c=colors, s=3, alpha=0.35,
           linewidths=0, rasterized=True)
ax.set_xlabel(f"Z_{FACTOR_X}")
ax.set_ylabel(f"Z_{FACTOR_Y}")
ax.set_title(
    f"Z_{FACTOR_X} vs Z_{FACTOR_Y}  "
    f"(AUROC_x={aucs[f'Z_{FACTOR_X}']:.3f}, AUROC_y={aucs[f'Z_{FACTOR_Y}']:.3f})",
    fontsize=10,
)
handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label="tumor_cd4", markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="benign_cd4", markersize=6),
]
ax.legend(handles=handles, frameon=False, loc="best")
fig.tight_layout()

out = FIG_DIR / f"semantic_geom_cd4_scatter_Z{FACTOR_X}_Z{FACTOR_Y}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_scatter_Z6_Z3.png

<Figure size 600x600 with 1 Axes>
```

```python  # cell 27
from itertools import combinations

FACTORS_GRID = [1, 6, 7, 3, 9]
pairs = list(combinations(FACTORS_GRID, 2))   # 10 pairs

ncols = 5
nrows = (len(pairs) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
axes = axes.flatten()

rng_draw = np.random.default_rng(0)
shuf = rng_draw.permutation(len(plot_idx))
sub_mal = is_mal[plot_idx][shuf]
colors = np.where(sub_mal == 1, "tab:red", "tab:blue")

for ax, (fx, fy) in zip(axes, pairs):
    sub_x = z[plot_idx, fx][shuf]
    sub_y = z[plot_idx, fy][shuf]
    ax.scatter(sub_x, sub_y, c=colors, s=2, alpha=0.3,
               linewidths=0, rasterized=True)
    ax.set_xlabel(f"Z_{fx}", fontsize=9)
    ax.set_ylabel(f"Z_{fy}", fontsize=9)
    ax.set_title(
        f"Z_{fx} vs Z_{fy}  "
        f"(AUROC {aucs[f'Z_{fx}']:.2f}/{aucs[f'Z_{fy}']:.2f})",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)

for j in range(len(pairs), len(axes)):
    axes[j].axis("off")

handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label="tumor_cd4", markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="benign_cd4", markersize=6),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.tight_layout()

out = FIG_DIR / f"semantic_geom_cd4_scatter_grid_{'_'.join(map(str, FACTORS_GRID))}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_scatter_grid_1_6_7_3_9.png

<Figure size 1500x600 with 10 Axes>
```

### How well does an *unsupervised* split of (Z_1, Z_6) recover malignancy?

Hypothesis: top-right of (Z_1, Z_6) is malignant, bottom-left is benign. Build a
1-D score `s = Z_1 + Z_6` and test several label-free thresholds, then compare
the predicted labels to the true `status`. Computed on **all** cells (not
subsampled). For reference we also report the best-threshold accuracy (peeks at
labels — upper bound for any monotone rule on `s`) and the AUROC of `s`.

```python  # cell 29
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_fscore_support
from scipy.stats import fisher_exact

FX, FY = 1, 6

x = z[:, FX]
y = z[:, FY]
# z-score each axis so neither dominates the diagonal score
xz = (x - x.mean()) / x.std()
yz = (y - y.mean()) / y.std()
s = xz + yz                     # diagonal projection: "top-right" score

# z-score each axis so the score is balanced. Higher s = upper-right corner.
y_true = is_mal                 # 1 = tumor_cd4, 0 = benign_cd4

def report(name, y_pred):
    # Allow either polarity (top-right=tumor OR top-right=benign); pick the
    # better one because the heuristic is label-free up to sign.
    acc_a = accuracy_score(y_true, y_pred)
    acc_b = accuracy_score(y_true, 1 - y_pred)
    if acc_b > acc_a:
        y_pred = 1 - y_pred
        acc = acc_b
    else:
        acc = acc_a
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0,
    )
    # 2x2 Fisher exact for independence (n is huge so p will be ~0 — keep as sanity)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return {
        "rule": name, "accuracy": acc, "precision_tumor": p, "recall_tumor": r,
        "f1_tumor": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "odds_ratio": odds, "fisher_p": pval,
    }

rows = []

# (1) median split on s (label-free, prior 50/50)
rows.append(report("s > median(s)", (s > np.median(s)).astype(int)))

# (2) prior-matched split: top fraction of s where fraction = empirical P(tumor)
prior = y_true.mean()
thr_prior = np.quantile(s, 1 - prior)
rows.append(report(f"top {prior:.0%} of s  (prior-matched)", (s > thr_prior).astype(int)))

# (3) GMM(2) on s — fully unsupervised threshold
gmm = GaussianMixture(n_components=2, random_state=0).fit(s.reshape(-1, 1))
gmm_pred = gmm.predict(s.reshape(-1, 1))
# polarity handled inside report()
rows.append(report("GMM(2) on s", gmm_pred))

# (4) KMeans(2) on the 2-D (xz, yz) — fully unsupervised, doesn't assume diagonal
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.c_[xz, yz])
rows.append(report("KMeans(2) on (Z_1,Z_6) standardized", km.labels_))

# (5) supervised upper bound: best threshold on s (Youden's J)
fpr, tpr, thrs = roc_curve(y_true, s)
j = tpr - fpr
best_thr = thrs[np.argmax(j)]
rows.append(report(f"s > {best_thr:.3f}  (best threshold — supervised)",
                   (s > best_thr).astype(int)))

results = pd.DataFrame(rows)
auc_s = roc_auc_score(y_true, s)
print(f"AUROC of s = Z_{FX}+Z_{FY} (standardized sum):  "
      f"{max(auc_s, 1-auc_s):.4f}")
print(f"Prior P(tumor) in this dataset:                  {prior:.3f}")
print(f"Trivial-majority baseline accuracy:              {max(prior, 1-prior):.3f}\n")
with pd.option_context("display.float_format", "{:.4f}".format,
                       "display.max_columns", None, "display.width", 200):
    print(results.to_string(index=False))
```

```
AUROC of s = Z_1+Z_6 (standardized sum):  0.8417
Prior P(tumor) in this dataset:                  0.659
Trivial-majority baseline accuracy:              0.659

                                     rule  accuracy  precision_tumor  recall_tumor  f1_tumor    TP    FP    FN    TN  odds_ratio  fisher_p
                            s > median(s)    0.7297           0.8885        0.6744    0.7667 68760  8633 33203 44191     10.6006    0.0000
            top 66% of s  (prior-matched)    0.7725           0.8273        0.8273    0.8273 84353 17610 17610 35214      9.5785    0.0000
                              GMM(2) on s    0.7193           0.8981        0.6474    0.7524 66013  7493 35950 45331     11.1089    0.0000
      KMeans(2) on (Z_1,Z_6) standardized    0.7244           0.8933        0.6605    0.7594 67345  8048 34618 44776     10.8233    0.0000
s > -0.285  (best threshold — supervised)    0.7470           0.8716        0.7224    0.7900 73658 10854 28305 41970     10.0625    0.0000
```

```python  # cell 30
# Lines of constant s = xz + yz in raw (Z_FX, Z_FY) coords:
#   (x - μx)/σx + (y - μy)/σy = c   →   y = σy * (c - (x - μx)/σx) + μy
mu_x, sd_x = x.mean(), x.std()
mu_y, sd_y = y.mean(), y.std()

def s_line(c, xs):
    return sd_y * (c - (xs - mu_x) / sd_x) + mu_y

# Class-balanced + shuffled draw (same plot_idx as earlier)
rng_draw = np.random.default_rng(0)
shuf = rng_draw.permutation(len(plot_idx))
xx = z[plot_idx, FX][shuf]
yy = z[plot_idx, FY][shuf]
ss = s[plot_idx][shuf]                              # per-cell s value
cc_status = np.where(is_mal[plot_idx][shuf] == 1, "tab:red", "tab:blue")

q33, q66 = np.quantile(s, [1/3, 2/3])
median_s = float(np.median(s))
fpr, tpr, thrs = roc_curve(y_true, s)
best_thr = float(thrs[np.argmax(tpr - fpr)])
s_vmax = float(np.quantile(np.abs(s), 0.99))         # symmetric ±99th pct

xs = np.linspace(xx.min(), xx.max(), 200)
arrow_len_x = 0.15 * (xx.max() - xx.min())
arrow_len_y = arrow_len_x * (sd_y / sd_x)

def draw_lines_and_arrow(ax):
    ax.plot(xs, s_line(median_s, xs), "k--", lw=1.5, label=f"s = median ({median_s:+.2f})")
    ax.plot(xs, s_line(q33, xs),      "k:",  lw=1.2, label=f"s = q33    ({q33:+.2f})")
    ax.plot(xs, s_line(q66, xs),      "k:",  lw=1.2, label=f"s = q66    ({q66:+.2f})")
    ax.plot(xs, s_line(best_thr, xs), "k-",  lw=1.5, label=f"s = best  ({best_thr:+.2f}, Youden)")
    ax.annotate("", xy=(mu_x + arrow_len_x, mu_y + arrow_len_y),
                xytext=(mu_x, mu_y),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(mu_x + arrow_len_x, mu_y + arrow_len_y, "  ↑ s", fontsize=10, va="center")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(f"Z_{FX}")
    ax.set_ylabel(f"Z_{FY}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.5))

# Left: colored by status
axL.scatter(xx, yy, c=cc_status, s=3, alpha=0.3, linewidths=0, rasterized=True)
draw_lines_and_arrow(axL)
axL.set_title(
    f"colored by status  (AUROC of s = "
    f"{max(roc_auc_score(y_true, s), 1-roc_auc_score(y_true, s)):.3f})",
    fontsize=10,
)
status_handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",  label="tumor_cd4",  markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="benign_cd4", markersize=6),
]
leg_status = axL.legend(handles=status_handles, loc="upper left", frameon=False)
axL.add_artist(leg_status)
axL.legend(loc="lower right", frameon=False, fontsize=8)

# Right: colored by s value (continuous)
sm = axR.scatter(xx, yy, c=ss, cmap="RdBu_r", vmin=-s_vmax, vmax=s_vmax,
                 s=3, alpha=0.6, linewidths=0, rasterized=True)
draw_lines_and_arrow(axR)
axR.set_title("colored by s = z(Z_{FX}) + z(Z_{FY})".format(FX=FX, FY=FY), fontsize=10)
axR.legend(loc="lower right", frameon=False, fontsize=8)
cbar = fig.colorbar(sm, ax=axR, fraction=0.04, pad=0.02)
cbar.set_label("s value")

fig.suptitle(f"Z_{FX} vs Z_{FY} with diagonal score s", fontsize=12, y=1.02)
fig.tight_layout()

out = FIG_DIR / f"semantic_geom_cd4_scatter_Z{FX}_Z{FY}_with_s.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_scatter_Z1_Z6_with_s.png

<Figure size 1300x650 with 3 Axes>
```

### 3-D version — factors (Z_1, Z_3, Z_6)

Same analysis in three dimensions: score `s3 = z(Z_1) + z(Z_3) + z(Z_6)`,
status-colored and s3-colored 3-D scatters side by side, and the median /
best-threshold **planes** drawn at constant `s3`. Stats below mirror the 2-D cell.

```python  # cell 32
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

FX3, FY3, FZ3 = 1, 3, 6

x3 = z[:, FX3]
y3 = z[:, FY3]
zZ3 = z[:, FZ3]
mu_x3, sd_x3 = x3.mean(), x3.std()
mu_y3, sd_y3 = y3.mean(), y3.std()
mu_z3, sd_z3 = zZ3.mean(), zZ3.std()

xz3 = (x3 - mu_x3) / sd_x3
yz3 = (y3 - mu_y3) / sd_y3
zz3 = (zZ3 - mu_z3) / sd_z3
s3 = xz3 + yz3 + zz3

# class-balanced + shuffled draw (reuse plot_idx)
rng3 = np.random.default_rng(0)
shuf3 = rng3.permutation(len(plot_idx))
xx3 = x3[plot_idx][shuf3]
yy3 = y3[plot_idx][shuf3]
zz3p = zZ3[plot_idx][shuf3]
ss3 = s3[plot_idx][shuf3]
cc3 = np.where(is_mal[plot_idx][shuf3] == 1, "tab:red", "tab:blue")

# thresholds
median_s3 = float(np.median(s3))
q33_3, q66_3 = np.quantile(s3, [1/3, 2/3])
fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
auc_s3 = float(roc_auc_score(y_true, s3))
auc_s3 = max(auc_s3, 1 - auc_s3)

# plane: (x-μx)/σx + (y-μy)/σy + (zZ-μz)/σz = c
#   →  zZ = σz * (c - (x-μx)/σx - (y-μy)/σy) + μz
def s_plane(c, xg, yg):
    return sd_z3 * (c - (xg - mu_x3) / sd_x3 - (yg - mu_y3) / sd_y3) + mu_z3

xg = np.linspace(xx3.min(), xx3.max(), 20)
yg = np.linspace(yy3.min(), yy3.max(), 20)
XG, YG = np.meshgrid(xg, yg)
ZG_med  = s_plane(median_s3, XG, YG)
ZG_best = s_plane(best_thr3, XG, YG)

# arrow tip in standardized direction (1,1,1) → unstandardize
ax_arrow_xyz = (mu_x3 + 0.5 * sd_x3, mu_y3 + 0.5 * sd_y3, mu_z3 + 0.5 * sd_z3)

s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

fig = plt.figure(figsize=(14, 7))

# --- Left: colored by status ---
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axL.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by status   AUROC(s3) = {auc_s3:.3f}", fontsize=10)
status_handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",  label="tumor_cd4",  markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="benign_cd4", markersize=6),
    plt.Line2D([], [], marker="s", linestyle="", color="black", alpha=0.10, label=f"plane: s3 = median ({median_s3:+.2f})", markersize=10),
    plt.Line2D([], [], marker="s", linestyle="", color="black", alpha=0.20, label=f"plane: s3 = best   ({best_thr3:+.2f}, Youden)", markersize=10),
]
axL.legend(handles=status_handles, loc="upper left", fontsize=8, frameon=False)

# --- Right: colored by s3 value ---
axR = fig.add_subplot(1, 2, 2, projection="3d")
sc3 = axR.scatter(xx3, yy3, zz3p, c=ss3, cmap="RdBu_r", vmin=-s_vmax3, vmax=s_vmax3,
                  s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG, YG, ZG_med,  color="black", alpha=0.10, edgecolor="none")
axR.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3})", fontsize=10)
cbar = fig.colorbar(sc3, ax=axR, fraction=0.03, pad=0.10)
cbar.set_label("s3 value")

fig.suptitle(f"3-D scatter Z_{FX3}, Z_{FY3}, Z_{FZ3} with diagonal score s3", fontsize=12, y=1.02)
fig.tight_layout()

out = FIG_DIR / f"semantic_geom_cd4_scatter3d_Z{FX3}_Z{FY3}_Z{FZ3}_with_s3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_scatter3d_Z1_Z3_Z6_with_s3.png

<Figure size 1400x700 with 3 Axes>
```

### Best-AUROC separating plane in (Z_1, Z_3, Z_6) — gene-direction sweep

Instead of the diagonal `s3 = z(Z_1)+z(Z_3)+z(Z_6)`, try each candidate **direction** in
the 3-D subspace and pick the one that best separates `tumor_cd4` vs `benign_cd4`:

- **Per-gene** direction: OLS coefficients of log-normalized expression on standardized
  `(z_1, z_3, z_6)` — the latent-space direction along which that gene varies most.
- **Pseudotime** direction: same OLS, with `sc.tl.dpt` pseudotime as the target.
- **Diagonal** `(1,1,1)` — the existing `s3` baseline.
- **Logistic-regression** coefficients on `(Z_1, Z_3, Z_6)` — *optimal* linear separator
  in this 3-D subspace, an upper bound for any single-direction score.

For each direction we project cells onto it (standardized axes), find the best
Youden-J threshold, and report AUROC / accuracy / precision / recall / F1 plus the
plane normal `(vx, vy, vz)` and intercept `thr`.

```python  # cell 34
import scanpy as sc
# 1) Pseudotime via DPT on the latent kNN graph (ad_emb already has neighbors from cell 18)
ROOT_CELL_OBS = "status"          # editable
ROOT_CELL_VAL = "benign_cd4"      # editable — pick benign side as DPT root
_root_mask = ad_emb.obs[ROOT_CELL_OBS].astype(str).values == ROOT_CELL_VAL
assert _root_mask.any(), f"no cell with {ROOT_CELL_OBS}=={ROOT_CELL_VAL!r}"
root_idx = int(np.where(_root_mask)[0][0])
ad_emb.uns["iroot"] = root_idx
sc.tl.diffmap(ad_emb, random_state=0)
sc.tl.dpt(ad_emb)
pt = ad_emb.obs["dpt_pseudotime"].to_numpy()
# DPT can produce inf at disconnected components; clip
pt = np.where(np.isfinite(pt), pt, np.nan)
print(f"pseudotime: root_idx={root_idx}  min={np.nanmin(pt):.3f}  median={np.nanmedian(pt):.3f}  max={np.nanmax(pt):.3f}  n_nan={np.isnan(pt).sum()}")
```

```
pseudotime: root_idx=101963  min=0.000  median=0.242  max=1.000  n_nan=0
```

```python  # cell 35
# 2) Editable gene list — default = top 4 from each of Z_1, Z_3, Z_6 (from top_df, cell 20).
GENES_FOR_PLANE = (
    top_df["proj_Z_1"].head(4).tolist()
    + top_df["proj_Z_3"].head(4).tolist()
    + top_df["proj_Z_6"].head(4).tolist()
)
# de-dup while preserving order, in case a gene appears in two axes
GENES_FOR_PLANE = list(dict.fromkeys(GENES_FOR_PLANE))
# GENES_FOR_PLANE=["CCR4", "SELL",  "CCL5", "NKG7", "GZMA", "GZMH", "CST7" ,"CXCL13"]
missing = [g for g in GENES_FOR_PLANE if g not in adata_cd4.var_names]
assert not missing, f"missing in adata_cd4: {missing}"
print(f"{len(GENES_FOR_PLANE)} genes:", GENES_FOR_PLANE)
```

```
12 genes: ['CXCL13', 'EGLN3', 'ADGRG1', 'NTRK2', 'KLF2', 'FOS', 'CD7', 'DUSP1', 'GZMB', 'CCL4', 'XCL1', 'XCL2']
```

```python  # cell 36
# 3) Candidate direction vectors in (Z_1, Z_3, Z_6) — OLS coeffs on standardized axes.
from numpy.linalg import lstsq
from sklearn.linear_model import LogisticRegression
from scipy.sparse import issparse

# standardize the 3 latent axes; reuse mu/sd from cell 32
Z3  = np.column_stack([z[:, FX3], z[:, FY3], z[:, FZ3]])
Z3z = np.column_stack([xz3, yz3, zz3])                     # already in cell 32
A   = np.column_stack([Z3z, np.ones(len(Z3z))])            # intercept column

# log-normalized expression on the fly (adata_cd4.X is raw counts per cell 10)
X_raw = adata_cd4.X
size_factor = np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float64)
size_factor[size_factor == 0] = 1.0

def gene_lognorm(name: str) -> np.ndarray:
    j = adata_cd4.var_names.get_loc(name)
    col = X_raw[:, j]
    col = col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()
    return np.log1p(col / size_factor * 1e4).astype(np.float64)

def fit_direction(y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    coef, *_ = lstsq(A[mask], y[mask], rcond=None)
    return coef[:3]

dir_rows = [("diagonal_(1,1,1)", np.array([1.0, 1.0, 1.0]))]
# pseudotime direction
dir_rows.append(("pseudotime", fit_direction(pt)))
# per-gene directions
for g in GENES_FOR_PLANE:
    dir_rows.append((g, fit_direction(gene_lognorm(g))))
# upper-bound: LR on (Z_1,Z_3,Z_6) directly
lr = LogisticRegression(max_iter=1000, C=1.0).fit(Z3z, is_mal)
dir_rows.append(("LR_optimal_(Z1,Z3,Z6)", lr.coef_.ravel().astype(np.float64)))

dirs = {name: v / max(np.linalg.norm(v), 1e-12) for name, v in dir_rows}
print(f"{len(dirs)} candidate directions:", list(dirs.keys()))
```

```
15 candidate directions: ['diagonal_(1,1,1)', 'pseudotime', 'CXCL13', 'EGLN3', 'ADGRG1', 'NTRK2', 'KLF2', 'FOS', 'CD7', 'DUSP1', 'GZMB', 'CCL4', 'XCL1', 'XCL2', 'LR_optimal_(Z1,Z3,Z6)']
```

```python  # cell 37
# 4) Score each direction: AUROC + Youden-best threshold + accuracy stats.
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support,
)
from scipy.stats import fisher_exact

y_true = is_mal  # from cell 22

def eval_direction(name: str, v: np.ndarray) -> dict:
    s = Z3z @ v
    auc = roc_auc_score(y_true, s)
    flipped = auc < 0.5
    if flipped:
        s, auc = -s, 1 - auc
        v = -v
    fpr, tpr, thrs = roc_curve(y_true, s)
    j = np.argmax(tpr - fpr)
    best_thr = float(thrs[j])
    y_pred = (s > best_thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0,
    )
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return dict(
        name=name, AUROC=auc, acc=acc, prec=p, rec=r, f1=f1,
        TP=tp, FP=fp, FN=fn, TN=tn, OR=float(odds), p=float(pval),
        vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
        thr=best_thr, flipped=flipped,
    )

results = (
    pd.DataFrame([eval_direction(n, v) for n, v in dirs.items()])
    .sort_values("AUROC", ascending=False)
    .reset_index(drop=True)
)
with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4f}".format):
    print(results.to_string(index=False))
out_csv = FIG_DIR / "semantic_geom_cd4_gene_planes_stats.csv"
results.to_csv(out_csv, index=False)
print("saved", out_csv)
results
```

```
name  AUROC    acc   prec    rec     f1    TP    FP    FN    TN      OR      p      vx      vy      vz     thr  flipped
LR_optimal_(Z1,Z3,Z6) 0.9597 0.8937 0.9414 0.8942 0.9172 91175  5672 10788 47152 70.2585 0.0000  0.5491 -0.7530  0.3625 -0.2957    False
                 KLF2 0.9594 0.8922 0.9425 0.8906 0.9158 90808  5535 11155 47289 69.5500 0.0000  0.5299 -0.7722  0.3506 -0.2852     True
                EGLN3 0.9507 0.8817 0.9322 0.8848 0.9078 90213  6566 11750 46258 54.0900 0.0000  0.6229 -0.5993  0.5029 -0.3299    False
                  CD7 0.9506 0.8839 0.9261 0.8952 0.9104 91274  7282 10689 45542 53.4037 0.0000  0.6913 -0.7197  0.0643 -0.2926     True
               ADGRG1 0.9474 0.8772 0.9342 0.8752 0.9037 89238  6288 12725 46536 51.9001 0.0000  0.4090 -0.6022  0.6856 -0.3125    False
           pseudotime 0.9255 0.8448 0.9313 0.8253 0.8751 84147  6212 17816 46612 35.4401 0.0000 -0.0352 -0.5796  0.8141 -0.1691    False
                NTRK2 0.9134 0.8331 0.8978 0.8426 0.8693 85910  9784 16053 43040 23.5420 0.0000  0.9175 -0.3282  0.2247 -0.2993    False
                 GZMB 0.8691 0.7889 0.8815 0.7850 0.8305 80041 10759 21922 42065 14.2752 0.0000 -0.2534 -0.4114  0.8755 -0.1539    False
               CXCL13 0.8621 0.8060 0.8564 0.8476 0.8520 86425 14489 15538 38335 14.7164 0.0000  0.8736 -0.3242 -0.3630 -0.2086    False
                  FOS 0.8291 0.7411 0.8988 0.6840 0.7768 69745  7856 32218 44968 12.3913 0.0000  0.2951 -0.9268 -0.2323  0.0383     True
                 XCL2 0.8054 0.7308 0.8388 0.7321 0.7818 74648 14348 27315 38476  7.3285 0.0000 -0.3362 -0.2477  0.9086 -0.1259    False
                 XCL1 0.8042 0.7356 0.8325 0.7494 0.7888 76410 15373 25553 37451  7.2847 0.0000 -0.3020 -0.2087  0.9302 -0.1642    False
                DUSP1 0.7520 0.6697 0.8697 0.5865 0.7005 59800  8962 42163 43862  6.9415 0.0000 -0.0067 -0.9824 -0.1864  0.1959     True
     diagonal_(1,1,1) 0.6599 0.6147 0.7866 0.5696 0.6607 58074 15753 43889 37071  3.1138 0.0000  0.5774  0.5774  0.5774  0.0128    False
                 CCL4 0.6072 0.5373 0.7728 0.4215 0.5455 42979 12634 58984 40190  2.3179 0.0000 -0.1562  0.3894  0.9077  0.3001    False
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_gene_planes_stats.csv

                     name     AUROC       acc      prec       rec        f1  \
0   LR_optimal_(Z1,Z3,Z6)  0.959662  0.893660  0.941433  0.894197  0.917207   
1                    KLF2  0.959420  0.892174  0.942549  0.890598  0.915837   
2                   EGLN3  0.950743  0.881670  0.932155  0.884762  0.907840   
3                     CD7  0.950578  0.883899  0.926113  0.895168  0.910378   
4                  ADGRG1  0.947363  0.877167  0.934175  0.875200  0.903726   
5              pseudotime  0.925510  0.844767  0.931252  0.825270  0.875064   
6                   NTRK2  0.913398  0.833080  0.897757  0.842561  0.869284   
7                    GZMB  0.869099  0.788865  0.881509  0.785000  0.830460   
8                  CXCL13  0.862060  0.806011  0.856422  0.847611  0.851994   
9                     FOS  0.829080  0.741102  0.898764  0.684023  0.776826   
10                   XCL2  0.805408  0.730837  0.838779  0.732109  0.781822   
11                   XCL1  0.804157  0.735598  0.832507  0.749389  0.788765   
12                  DUSP1  0.752022  0.669707  0.869666  0.586487  0.700542   
13       diagonal_(1,1,1)  0.659905  0.614683  0.786623  0.569560  0.660720   
14                   CCL4  0.607203  0.537313  0.772823  0.421516  0.545502   

       TP     FP     FN     TN         OR    p        vx        vy        vz  \
0   91175   5672  10788  47152  70.258477  0.0  0.549124 -0.753022  0.362520   
1   90808   5535  11155  47289  69.549985  0.0  0.529947 -0.772185  0.350551   
2   90213   6566  11750  46258  54.090031  0.0  0.622865 -0.599264  0.502914   
3   91274   7282  10689  45542  53.403710  0.0  0.691311 -0.719693  0.064276   
4   89238   6288  12725  46536  51.900143 
... [1220 more chars truncated]
```

```python  # cell 38
# 5) 3-D scatter: top-AUROC plane (left = status, right = KLF2 expression).
top_row  = results.iloc[0]
top_name = top_row["name"]
v_top    = np.array([top_row["vx"], top_row["vy"], top_row["vz"]], dtype=float)
v_top    = v_top / np.linalg.norm(v_top)
thr_top  = float(top_row["thr"])

# plane on standardized axes: vx*xz + vy*yz + vz*zz = thr  -->  z_raw via inverse-standardize
def plane_z(v, thr, x_raw, y_raw):
    xz_ = (x_raw - mu_x3) / sd_x3
    yz_ = (y_raw - mu_y3) / sd_y3
    if abs(v[2]) < 1e-6:
        return np.full_like(x_raw, np.nan, dtype=float)
    zz_ = (thr - v[0] * xz_ - v[1] * yz_) / v[2]
    return sd_z3 * zz_ + mu_z3

xg2 = np.linspace(xx3.min(), xx3.max(), 20)
yg2 = np.linspace(yy3.min(), yy3.max(), 20)
XG2, YG2 = np.meshgrid(xg2, yg2)
ZG_top  = plane_z(v_top, thr_top, XG2, YG2)

# KLF2 log-norm expression for right-panel coloring
KLF2_GENE = "KLF2"
assert KLF2_GENE in adata_cd4.var_names, f"{KLF2_GENE} missing from adata_cd4.var_names"
klf2_expr = gene_lognorm(KLF2_GENE)
klf2_plot = klf2_expr[plot_idx][shuf3]
klf2_vmax = float(np.quantile(klf2_expr, 0.99))

fig = plt.figure(figsize=(14, 7))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
axL.plot_surface(XG2, YG2, ZG_top,  color="tab:green", alpha=0.25, edgecolor="none")
axL.set_xlabel(f"Z_{FX3}"); axL.set_ylabel(f"Z_{FY3}"); axL.set_zlabel(f"Z_{FZ3}")
axL.set_title(f"colored by status   top: {top_name}   AUROC={top_row['AUROC']:.3f}", fontsize=10)
status_handles = [
    plt.Line2D([], [], marker="o", linestyle="", color="tab:red",   label="tumor_cd4",  markersize=6),
    plt.Line2D([], [], marker="o", linestyle="", color="tab:blue",  label="benign_cd4", markersize=6),
    plt.Line2D([], [], marker="s", linestyle="", color="tab:green", alpha=0.25, label=f"{top_name} best (AUROC={top_row['AUROC']:.3f})", markersize=10),
]
axL.legend(handles=status_handles, loc="upper left", fontsize=8, frameon=False)

axR = fig.add_subplot(1, 2, 2, projection="3d")
scR = axR.scatter(xx3, yy3, zz3p, c=klf2_plot, cmap="viridis",
                  vmin=0, vmax=klf2_vmax, s=3, alpha=0.55, depthshade=False)
axR.plot_surface(XG2, YG2, ZG_top, color="tab:green", alpha=0.25, edgecolor="none")
axR.set_xlabel(f"Z_{FX3}"); axR.set_ylabel(f"Z_{FY3}"); axR.set_zlabel(f"Z_{FZ3}")
axR.set_title(f"colored by {KLF2_GENE} log-norm expression", fontsize=10)
fig.colorbar(scR, ax=axR, fraction=0.03, pad=0.10).set_label(f"{KLF2_GENE} log1p(cp10k)")

fig.suptitle(f"Best-direction plane: {top_name}", fontsize=12, y=1.02)
fig.tight_layout()
safe = top_name.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "_").replace("/", "_")
out = FIG_DIR / f"semantic_geom_cd4_scatter3d_best_plane_{safe}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
plt.show()
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_scatter3d_best_plane_LR_optimal_Z1_Z3_Z6.png

<Figure size 1400x700 with 3 Axes>
```

```python  # cell 39
# Interactive rotatable 3-D version of the top-AUROC plane.
# Writes a standalone .html (open in a browser, click+drag to rotate).
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import IFrame

# tumor/benign masks (already class-balanced + shuffled via plot_idx/shuf3)
mal_mask = is_mal[plot_idx][shuf3] == 1
ben_mask = ~mal_mask

fig3d = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        f"colored by status   top: {top_name}   AUROC={top_row['AUROC']:.3f}",
        f"colored by {KLF2_GENE} log-norm expression",
    ),
    horizontal_spacing=0.05,
)

# --- Left: status-colored points ---
fig3d.add_trace(go.Scatter3d(
    x=xx3[mal_mask], y=yy3[mal_mask], z=zz3p[mal_mask],
    mode="markers", name="tumor_cd4",
    marker=dict(size=2, color="crimson", opacity=0.45),
), row=1, col=1)
fig3d.add_trace(go.Scatter3d(
    x=xx3[ben_mask], y=yy3[ben_mask], z=zz3p[ben_mask],
    mode="markers", name="benign_cd4",
    marker=dict(size=2, color="royalblue", opacity=0.45),
), row=1, col=1)

# --- Left plane: top (green) ---
fig3d.add_trace(go.Surface(
    x=XG2, y=YG2, z=ZG_top, showscale=False, opacity=0.35,
    colorscale=[[0, "green"], [1, "green"]],
    name=f"{top_name} best (AUROC={top_row['AUROC']:.3f})", showlegend=True,
), row=1, col=1)

# --- Right: KLF2-colored points ---
fig3d.add_trace(go.Scatter3d(
    x=xx3, y=yy3, z=zz3p, mode="markers",
    marker=dict(
        size=2, color=klf2_plot, colorscale="Viridis",
        cmin=0, cmax=klf2_vmax, opacity=0.55,
        colorbar=dict(title=f"{KLF2_GENE} log1p(cp10k)", thickness=12, x=1.02),
    ),
    name=f"{KLF2_GENE}",
    showlegend=False,
), row=1, col=2)
fig3d.add_trace(go.Surface(
    x=XG2, y=YG2, z=ZG_top, showscale=False, opacity=0.35,
    colorscale=[[0, "green"], [1, "green"]],
    name="top plane", showlegend=False,
), row=1, col=2)

axes_kw = dict(
    xaxis_title=f"Z_{FX3}", yaxis_title=f"Z_{FY3}", zaxis_title=f"Z_{FZ3}",
)
fig3d.update_layout(
    title=f"Best-direction plane: {top_name}",
    width=1300, height=700, scene=axes_kw, scene2=axes_kw,
    legend=dict(itemsizing="constant"),
)

html_out = FIG_DIR / f"semantic_geom_cd4_scatter3d_best_plane_{safe}.html"
fig3d.write_html(html_out, include_plotlyjs="cdn", full_html=True)
print("saved", html_out)
IFrame(html_out.as_posix(), width=1320, height=720)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_scatter3d_best_plane_LR_optimal_Z1_Z3_Z6.html

<IPython.lib.display.IFrame at 0x1552ae6a4350>
```

#### Stats — unsupervised heuristics on `s3` (all cells)

Same five rules as the 2-D version: median split, prior-matched threshold,
GMM(2) on `s3`, KMeans(2) on standardized 3-D coords, and best-threshold
(Youden) upper bound.

```python  # cell 41
rows3 = []

# (1) median split
rows3.append(report("s3 > median(s3)", (s3 > np.median(s3)).astype(int)))

# (2) prior-matched split
prior3 = y_true.mean()
thr_prior3 = np.quantile(s3, 1 - prior3)
rows3.append(report(f"top {prior3:.0%} of s3 (prior-matched)", (s3 > thr_prior3).astype(int)))

# (3) GMM(2) on s3
gmm3 = GaussianMixture(n_components=2, random_state=0).fit(s3.reshape(-1, 1))
rows3.append(report("GMM(2) on s3", gmm3.predict(s3.reshape(-1, 1))))

# (4) KMeans(2) on standardized 3-D
km3 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.c_[xz3, yz3, zz3])
rows3.append(report("KMeans(2) on (Z_1,Z_3,Z_6) standardized", km3.labels_))

# (5) best-threshold supervised upper bound
rows3.append(report(f"s3 > {best_thr3:.3f}  (best threshold — supervised)",
                    (s3 > best_thr3).astype(int)))

results3 = pd.DataFrame(rows3)
print(f"AUROC of s3 = z(Z_{FX3})+z(Z_{FY3})+z(Z_{FZ3}):  {auc_s3:.4f}")
print(f"Prior P(tumor):                                   {prior3:.3f}")
print(f"Trivial-majority baseline accuracy:               {max(prior3, 1-prior3):.3f}\n")
with pd.option_context("display.float_format", "{:.4f}".format,
                       "display.max_columns", None, "display.width", 200):
    print(results3.to_string(index=False))
```

```
AUROC of s3 = z(Z_8)+z(Z_4)+z(Z_6):  0.6093
Prior P(tumor):                                   0.452
Trivial-majority baseline accuracy:               0.548

                                     rule  accuracy  precision_tumor  recall_tumor  f1_tumor    TP   FP    FN    TN  odds_ratio  fisher_p
                          s3 > median(s3)    0.5769           0.5287        0.5852    0.5555  9379 8360  6649 11090      1.8712    0.0000
            top 45% of s3 (prior-matched)    0.5721           0.5217        0.6331    0.5721 10148 9302  5880 10148      1.8828    0.0000
                             GMM(2) on s3    0.5846           0.5528        0.4218    0.4785  6761 5469  9267 13981      1.8651    0.0000
  KMeans(2) on (Z_1,Z_3,Z_6) standardized    0.7284           0.6650        0.8036    0.7277 12880 6489  3148 12961      8.1723    0.0000
s3 > 4.864  (best threshold — supervised)    0.5482           0.0000        0.0000    0.0000     0    1 16028 19449      0.0000    1.0000
```

## Gene loadings (projections)

```python  # cell 43
W = model.get_loadings().reindex(adata_cd4.var_names)   # genes x N_LATENT
adata_cd4.varm["W_semantic_geom_cd4"] = W.values
adata_cd4.uns["W_semantic_geom_cd4_columns"] = list(W.columns)
print("W (loadings):", W.shape, "| columns:", list(W.columns))
W.head()
```

```
W (loadings): (2500, 10) | columns: ['Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9']

               Z_0       Z_1       Z_2       Z_3       Z_4       Z_5  \
HES4      0.012427  0.006022  0.137283  0.004578  0.025926  0.005638   
ISG15     0.315019  0.033936  0.002390  0.103474  0.005595  0.109133   
RNF223    0.023841  0.025443  0.575253  0.021771  0.030152  1.019468   
TNFRSF18  0.056001  2.608525  0.005918  0.009065  0.386911  0.004751   
TNFRSF4   0.061097  0.404810  0.015318  1.610499  0.680414  0.001340   

               Z_6       Z_7       Z_8       Z_9  
HES4      3.258906  0.012049  0.071941  0.006463  
ISG15     1.420385  0.002057  0.006241  3.750064  
RNF223    0.037012  0.034518  4.401197  0.017732  
TNFRSF18  1.666576  0.028022  0.222135  0.080810  
TNFRSF4   0.077079  0.004370  0.421423  3.553838
```

### Top loading genes per projection

```python  # cell 45
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
proj_Z_0  proj_Z_1  proj_Z_2 proj_Z_3   proj_Z_4   proj_Z_5  proj_Z_6 proj_Z_7   proj_Z_8 proj_Z_9
top_1        MMP3    CXCL13    DEFB4A     KLF2     SLC1A2    MINDY4B      GZMB    MYOD1       PCP4   TUBA1C
top_2      CITED1     EGLN3     KRT19      FOS      DPEP1       WIF1      CCL4   KIF20A      TRPM1     NME1
top_3       TREM1    ADGRG1      IL31      CD7       IDO1    C5orf49      XCL1   DLGAP5     GUCA2B     ODC1
top_4        IL18     NTRK2      AQP5    DUSP1      IGHA2        HBB      XCL2    UBE2C      NETO1   TUBA1B
top_5     CEACAM4     RDH10     EDIL3     SELL   ARHGAP40     IL36RN      CCL3     PLK1      CNTN3    HSPD1
top_6      RNASE2      TNS3     CXCL5    NR4A2     COX4I2      FABP4     HMOX1     ASPM      CASQ2     CYCS
top_7        C1QB     CCL20    S100A1     CD69   SERPINB2        DCD      GNLY    CDC20       NNAT   TUBB2A
top_8         CR1       CLU  C22orf42   GIMAP4     CLSTN2      TPSD1      GZMH    CENPA     CLEC4F   SDF2L1
top_9        AQP9      ATF5    MLXIPL     JUNB       RND3       RBP1      NKG7      IL9    GALNTL6    DDIT3
top_10    EPB41L3       CD9      ALPG    TAGAP      KRT15     OR51E1      IL13    KIF14     SEL1L2      FTL
top_11     CD300E     P4HA2     CLDN8   GPR183   TNFRSF17       FLG2  C15orf48     NEK2      EPHA5  SLC25A5
top_12      C5AR1    BCL2L1    DIRAS1     FOSB       MMP7       LRP4      IFNG    HJURP     RNF223    FABP5
top_13        DPT     FURIN     OBP2B    KLRB1       VCAN       CDX1      CCL5    CDCA3      KCNK3     TUBB
top_14    COLEC12    SEMA4A    SLC4A1    S1PR1       GNG8       GJA4    CCL4L2    GTSE1      TKTL2   TXNRD1
top_15     PLA2G7      ZEB2     FIGLA      JUN       TCN1       HBA2       CGA    KIF23      GPR33     NQO1
top_16      KCNJ6   TSPAN13      TCIM   DNAJB1      MAT1A      CCL16      G0S2    BIRC5    COL11A1     MANF
top_17   TMEM176B  SERPINA1       IL5     CD55      CCBE1      CD207      IER3    CCNA2      MYOZ1     CKS2
top_18       C1QC       MSC       CCK     RGCC       APOD       PTH2      IL22   DEPDC1     PLCXD3     POMP
top_19       DAB2      ETV7     ADCY2     ICOS       IL24       CDH2      EGR2   KIF18B     GRIN2A    KPNA2
top_20        LYZ    TWIST1    TRIM63    ZFP36  KRTAP19-5   SERPINA5    TNFSF9    CCNB2       PCK1     MT2A
top_21     RNASE1     PTHLH       HRC     CCR7        CPE      LCE3E     HBEGF    KIF2C      STAB2    HSPB1
top_22  ARHGEF10L    GOLIM4       PYY    SOCS3       AHSP   SERPINA9      ATF3    CDCA2      LYPD2    CMSS1
top_23    SIGLEC1      IRF4      CPN2     FCMR       PMCH     GPR182      CSF1    TROAP     PLPPR4     MT1X
top_24       FPR2     MTSS1    SPINK1   ZNF331      FOXF1       CMA1     VCAM1    CDCA8      PCSK2    HMGA1
top_25       PODN    RCBTB2     HSPA6  GADD45B       MISP       LIM2     CEBPD     HMMR       LCN2   SEC61B
top_26       SOD3      NBAS     MUCL1    TXNIP       KERA      MS4A2     RASD1    AURKB  KIAA1549L  ZFAND2A
top_27     SLC8A1    GALNT2     GPR31     CD8A       NRP2    S100A7A      CD83   CKAP2L     SHANK2     NME2
top_28    SLCO2B1      SOX4      PRR9     PER1       RIN2  KRTAP4-11       GEM     ANLN      VEPH1    C1QBP
top_29     SIRPB2    ADAM19   TMEM210   SLC2A3      CCL11      IL36G     RAMP1     RRM2    SLITRK2   SNRPD1
top_30       C1QA      CTSH   MKRN2OS     BTG2       EBI3       RPTN      CCL1      PBK      P2RX2   DNAJA1

         proj_Z_0  proj_Z_1  proj_Z_2 proj_Z_3   proj_Z_4   proj_Z_5  \
top_1        MMP3    CXCL13    DEFB4A     KLF2     SLC1A2    MINDY4B   
top_2      CITED1     EGLN3     KRT19      FOS      DPEP1       WIF1   
top_3       TREM1    ADGRG1      IL31      CD7       IDO1    C5orf49   
top_4        IL18     NTRK2      AQP5    DUSP1      IGHA2        HBB   
top_5     CEACAM4     RDH10     EDIL3     SELL   ARHGAP40     IL36RN   
top_6      RNASE2      TNS3     CXCL5    NR4A2     COX4I2      FABP4   
top_7        C1QB     CCL20    S100A1     CD69   SERPINB2        DCD   
top_8         CR1       CLU  C22orf42   GIMAP4     CLSTN2      TPSD1   
top_9       
... [3058 more chars truncated]
```

## Hierarchical clustering of top-loaded genes

Same recipe as 05: top genes by max |loading| → correlation-distance → average
linkage → #clusters chosen by silhouette.

```python  # cell 47
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

max_vals = W.abs().max(axis=1)
top_idx = max_vals.sort_values(ascending=False).head(CLUSTER_N_TOP).index
subset = W.loc[top_idx]

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
hierarchical clusters: k=7 (silhouette=0.969) over 500 genes

=== cluster 1  (n=28) ===
CXCL13, EGLN3, ADGRG1, NTRK2, RDH10, TNS3, CCL20, CLU, ATF5, CD9, P4HA2, BCL2L1, FURIN, SEMA4A, ZEB2, TSPAN13, SERPINA1, MSC, ETV7, TWIST1, PTHLH, GOLIM4, IRF4, MTSS1, RCBTB2

=== cluster 2  (n=201) ===
TUBA1C, NME1, ODC1, TUBA1B, HSPD1, CYCS, TUBB2A, SDF2L1, DDIT3, FTL, SLC25A5, FABP5, TUBB, TXNRD1, NQO1, MANF, CKS2, POMP, KPNA2, MT2A, HSPB1, CMSS1, MT1X, HMGA1, SEC61B

=== cluster 3  (n=6) ===
MINDY4B, WIF1, C5orf49, HBB, IL36RN, FABP4

=== cluster 4  (n=127) ===
PCP4, TRPM1, GUCA2B, NETO1, CNTN3, CASQ2, NNAT, CLEC4F, GALNTL6, SEL1L2, EPHA5, RNF223, KCNK3, TKTL2, GPR33, COL11A1, MYOZ1, PLCXD3, GRIN2A, PCK1, STAB2, LYPD2, PLPPR4, PCSK2, LCN2

=== cluster 5  (n=1) ===
MMP3

=== cluster 6  (n=98) ===
KLF2, FOS, CD7, DUSP1, SELL, NR4A2, CD69, GIMAP4, JUNB, TAGAP, GPR183, FOSB, KLRB1, S1PR1, JUN, DNAJB1, CD55, RGCC, ICOS, ZFP36, CCR7, SOCS3, FCMR, ZNF331, GADD45B

=== cluster 7  (n=39) ===
GZMB, CCL4, XCL1, XCL2, CCL3, HMOX1, GNLY, GZMH, NKG7, IL13, C15orf48, IFNG, CCL5, CCL4L2, CGA, G0S2, IER3, IL22, EGR2, TNFSF9, HBEGF, ATF3, CSF1, VCAM1, CEBPD
```

```python  # cell 48
import seaborn as sns

corr = np.corrcoef(subset.values)
g = sns.clustermap(
    pd.DataFrame(corr, index=top_idx, columns=top_idx),
    row_linkage=Z, col_linkage=Z, cmap="vlag", center=0,
    xticklabels=False, yticklabels=False, figsize=(8, 8),
)
g.fig.suptitle(
    f"semantic_geom CD4: top-{len(top_idx)} gene loading correlation (k={best_k})", y=1.01
)
out = FIG_DIR / "semantic_geom_cd4_gene_hierarchy.png"
g.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_gene_hierarchy.png

<Figure size 800x800 with 4 Axes>
```
