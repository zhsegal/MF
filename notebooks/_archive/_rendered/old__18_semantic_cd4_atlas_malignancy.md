# old/18_semantic_cd4_atlas_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/18_semantic_cd4_atlas_malignancy.ipynb` and is never run._

# SemanticSCVI (geometric) — atlas-wide CD4 (non-Treg) factors vs **malignancy**

Trains a geometric `SemanticSCVI` (Geneformer prior) on the curated combined skin-T atlas
(`skin_T_tcr_malig_v2.h5ad`, 242,959 cells) and inspects how its factors separate a chosen
malignancy call. Steps:

1. **Load** the atlas and **join all precomputed malignancy calls** — TCR ALICE (nb21),
   CNV all-T + CD4/CD8-ref (nb14/15), transcriptomic MrVI m1/m2 (nb16/17) — with a short summary.
2. **Pick the driving call** at the top via `MALIGNANCY_COL` (default `tcr_malignant_alice`);
   it becomes `is_malignant` and drives every panel below.
3. **Select samples** (de-dup / non-αβ-CD4 / size gate) → **subset to CD4 non-Treg**.
4. **Train / reload** the geometric `SemanticSCVI` (same knobs as nb12).
5. **Inspect separation**: latent UMAP, per-factor heatmaps, factor geometry + gene-direction planes.

Part 1 is CPU-light. Parts 2-3 (Geneformer map build, training, UMAP, DPT) are **HEAVY** — run on
the `neural_nmf_env` GPU kernel.

```python  # cell 1
# ============================================================
# Parameters — malignancy-call selector + SemanticSCVI knobs (geometric Geneformer, same as nb12).
# ============================================================
import hashlib
import json
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")


def _semantic_cache_slug(kwargs, max_epochs, warmup_epochs, n_epochs_kl_warmup, hvg_top_n, n=10):
    """Stable 10-char hash of every param that affects the trained SemanticSCVI model."""
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
V2             = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_malig_v2.h5ad"  # curated TCR-complete object
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"   # gene_name -> gene_id
SEMANTIC_CACHE_GENEFORMER = NB_DIR / "data" / "mf_cd4_atlas_geneformer.pt"
MODEL_CACHE_DIR = NB_DIR / "models" / ".model_cache_semantic_geom_cd4_atlas"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
EMB_UMAP_NPY = NB_DIR / "data" / "atlas_joint" / "semantic_geom_cd4_atlas_emb_umap.npy"

# ---- per-donor clone thresholds (drive the sample-selection rules in Part 1) ----
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2

# ---- CD4 non-Treg selection: cell_type_T2 == "CD4" (excludes CD4_Treg / CD8 / NK) ----

# ---- malignancy call that drives every downstream panel (key in M.MALIGNANCY_SOURCES) ----
#   tcr_malignant_alice (default, nb21 TCR ALICE)
#   cnvT_cell_gmm / cnvT_cell_mrvi / cnvT_cnvcluster / cnvT_cnvcluster_mrvi  (nb14 CNV, all-T)
#   cnv_cd4_cd8ref            (nb15 CNV, CD4 vs same-sample CD8 ref — CD4-only coverage)
#   mrvi_m1_labelspread       (nb16 transcriptomic LabelSpreading)
#   mrvi_m2_pseudosample      (nb17 transcriptomic pseudosample)
MALIGNANCY_COL = "tcr_malignant_alice"

# ---- Preprocessing ----
HVG_TOP_N  = 2500
HVG_FLAVOR = "seurat_v3"

# ---- Shared model size ----
N_LATENT  = 10
BATCH_KEY = "sample_id"

# ---- SemanticSCVI (Geneformer prior) — same as nb12 ----
SEMANTIC_GENEFORMER_MAX_EPOCHS = 200
SEMANTIC_GENEFORMER_WARMUP_EPOCHS = 20
SEMANTIC_GENEFORMER_N_EPOCHS_KL_WARMUP = 100
SEMANTIC_GENEFORMER_KWARGS = dict(
    loss_mode="geometric",
    coherence_weight=1000.0,
    n_gene_sample=1024,
    n_latent=N_LATENT,
    n_layers=1,
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

# ---- interpretation (ref values from nb12) ----
PER_PROJECTION_N_TOP = 30
```

```python  # cell 2
import sys, gc, importlib
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))            # MF/ holds atlas_join_helpers, skin_T_cnv_helpers, semantic_malig_helpers
if str(NB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(NB_DIR.parent))     # notebooks/ holds benchmark_helpers

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scvi

import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import semantic_malig_helpers as M
importlib.reload(H); importlib.reload(C); importlib.reload(M)
from benchmark_helpers import get_or_build_geneformer_map, train_or_load_semantic_scvi

assert MALIGNANCY_COL in M.MALIGNANCY_SOURCES, \
    f"MALIGNANCY_COL={MALIGNANCY_COL!r} not in {list(M.MALIGNANCY_SOURCES)}"

scvi.settings.seed = 0
np.random.seed(0)
sc.settings.verbosity = 1
sc.settings.figdir = FIG_DIR
print("scvi", scvi.__version__, "| scanpy", sc.__version__, "| CUDA:", torch.cuda.is_available())
```

## Part 1 — load curated atlas + all malignancy calls  · CPU-light

1. **Load** the curated TCR-complete combined skin-T object (`skin_T_tcr_malig_v2.h5ad`).
2. **Join all precomputed malignancy calls** (`M.load_malignancy_calls`): TCR ALICE (nb21),
   CNV all-T + CD4/CD8-ref (nb14/15), transcriptomic MrVI m1/m2 (nb16/17). Each lands as a
   bool obs column; print a short **summary** (prevalence + pairwise Jaccard).
3. **Select samples** (per-donor clone table): de-dup `D5__MFIVB`≡`D1__P303`, drop non-αβ-CD4
   entities (`MF_gamma_delta`, `CD8_aggressive_epidermotropic_CTCL`), size-gate
   `n_tcr_cells < 300` (keep all herrera2021).
4. **Subset** to CD4 non-Treg (`cell_type_T2 == "CD4"`), drop ribosomal genes, and set the
   driving label `is_malignant = adata.obs[MALIGNANCY_COL]` (HC forced benign).

```python  # cell 4
adata = sc.read_h5ad(V2)                       # curated TCR-complete T object (242,959 cells)

# Join every precomputed malignancy call (nb14-17 CNV/transcriptomic + nb21 TCR ALICE)
# onto obs by cell name; uncovered cells -> benign. Each becomes a bool obs column.
adata = M.load_malignancy_calls(adata, NB_DIR)
print("loaded:", adata.shape)
```

```
tcr_malignant_alice    <- alice_malignancy_v2.parquet                100.0% covered |  68,620 malignant
cnvT_cell_gmm          <- skin_T_malignancy_methods.parquet          100.0% covered | 140,371 malignant
cnvT_cell_mrvi         <- skin_T_malignancy_methods.parquet          100.0% covered |  94,391 malignant
cnvT_cnvcluster        <- skin_T_malignancy_methods.parquet          100.0% covered | 155,115 malignant
cnvT_cnvcluster_mrvi   <- skin_T_malignancy_methods.parquet          100.0% covered | 161,188 malignant
cnv_cd4_cd8ref         <- skin_cd4_cd8ref_malignancy.parquet          76.0% covered | 119,099 malignant
mrvi_m1_labelspread    <- calls.csv                                  100.0% covered | 126,240 malignant
mrvi_m2_pseudosample   <- skin_T_mrvi_m2_malignancy_methods.parquet  100.0% covered | 129,303 malignant
loaded: (242959, 49683)
```

```python  # cell 5
# Summary of all malignancy calls: prevalence + pairwise Jaccard (whole skin-T set).
counts, jaccard = M.malignancy_summary(adata)
print("driving call: MALIGNANCY_COL =", MALIGNANCY_COL, "\n")
print("prevalence per call:")
print(counts)
print("\npairwise Jaccard:")
with pd.option_context("display.width", 200, "display.max_columns", None):
    print(jaccard)
counts
```

```
driving call: MALIGNANCY_COL = tcr_malignant_alice 

prevalence per call:
                      n_malignant  pct_malignant
tcr_malignant_alice         68620          28.24
cnvT_cell_gmm              140371          57.78
cnvT_cell_mrvi              94391          38.85
cnvT_cnvcluster            155115          63.84
cnvT_cnvcluster_mrvi       161188          66.34
cnv_cd4_cd8ref             119099          49.02
mrvi_m1_labelspread        126240          51.96
mrvi_m2_pseudosample       129303          53.22

pairwise Jaccard:
                      tcr_malignant_alice  cnvT_cell_gmm  cnvT_cell_mrvi  cnvT_cnvcluster  cnvT_cnvcluster_mrvi  cnv_cd4_cd8ref  mrvi_m1_labelspread  mrvi_m2_pseudosample
tcr_malignant_alice                 1.000          0.279           0.400            0.302                 0.404           0.303                0.544                 0.481
cnvT_cell_gmm                       0.279          1.000           0.381            0.519                 0.507           0.416                0.442                 0.440
cnvT_cell_mrvi                      0.400          0.381           1.000            0.392                 0.523           0.374                0.574                 0.500
cnvT_cnvcluster                     0.302          0.519           0.392            1.000                 0.602           0.541                0.520                 0.504
cnvT_cnvcluster_mrvi                0.404          0.507           0.523            0.602                 1.000           0.487                0.717                 0.649
cnv_cd4_cd8ref                      0.303          0.416           0.374            0.541                 0.487           1.000                0.458                 0.441
mrvi_m1_labelspread                 0.544          0.442           0.574            0.520                 0.717           0.458                1.000                 0.718
mrvi_m2_pseudosample                0.481          0.440           0.500            0.504                 0.649           0.441                0.718                 1.000

                      n_malignant  pct_malignant
tcr_malignant_alice         68620          28.24
cnvT_cell_gmm              140371          57.78
cnvT_cell_mrvi              94391          38.85
cnvT_cnvcluster            155115          63.84
cnvT_cnvcluster_mrvi       161188          66.34
cnv_cd4_cd8ref             119099          49.02
mrvi_m1_labelspread        126240          51.96
mrvi_m2_pseudosample       129303          53.22
```

```python  # cell 6
# Sample-selection rules on the per-donor clone table (functional — sets the training cohort).
clone_summary = C.clone_summary_table(adata, FRAC_THRESH, RATIO_THRESH)
meta = (
    adata.obs[["donor", "study", "entity", "disease"]].astype(str)
    .groupby("donor", observed=True)
    .agg(lambda s: ", ".join(sorted(s[s != "nan"].unique())))
)
clone_summary = clone_summary.merge(meta, on="donor", how="left")

drop = {}
# 1. de-duplicate: D5__MFIVB == D1__P303 (identical clone profile) -> keep D5__MFIVB
if {"D5__MFIVB", "D1__P303"} <= set(clone_summary["donor"]):
    drop["D1__P303"] = "duplicate of D5__MFIVB"
# 2. drop non-alpha-beta-CD4 entities (TCRb caller is blind to these -> route to CNV)
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}
for d in clone_summary.loc[clone_summary["entity"].isin(DROP_ENTITIES), "donor"]:
    drop.setdefault(d, "non-ab-CD4 entity")
# 3. size gate: n_tcr_cells < 300, except study == herrera2021
small = (clone_summary["n_tcr_cells"] < 300) & (clone_summary["study"] != "herrera2021")
for d in clone_summary.loc[small, "donor"]:
    drop.setdefault(d, "n_tcr_cells < 300")

keep_donors = clone_summary.loc[~clone_summary["donor"].isin(drop), "donor"].tolist()
if drop:
    print("dropped:", drop)
adata = adata[adata.obs["donor"].isin(keep_donors)].copy()
print(f"samples kept: {len(keep_donors)} | cells: {adata.n_obs}")
```

```
samples kept: 34 | cells: 242959
```

```python  # cell 7
adata_cd4 = adata[adata.obs["cell_type_T2"].astype(str) == "CD4"].copy()   # CD4 non-Treg (excl. CD4_Treg/CD8/NK)

# drop ribosomal protein genes (RPS*/RPL*) — they dominate loadings and are uninformative here
ribo = adata_cd4.var_names.str.upper().str.startswith(("RPS", "RPL"))
print(f"removing {int(ribo.sum())} ribosomal protein genes (RPS*/RPL*)")
adata_cd4 = adata_cd4[:, ~ribo].copy()

adata_cd4.X = adata_cd4.layers["raw_counts"].copy()   # NB likelihood needs raw counts

# driving malignancy label (selected at top) — HC samples are benign by definition
is_mal = adata_cd4.obs[MALIGNANCY_COL].astype(bool).to_numpy()
hc = adata_cd4.obs["disease"].astype(str).eq("HC").to_numpy()
n_flip = int(is_mal[hc].sum())
is_mal[hc] = False
adata_cd4.obs["is_malignant"] = is_mal
print(f"CD4 non-Treg: {adata_cd4.shape}  | label = {MALIGNANCY_COL} (HC forced benign: {n_flip})")
print("is_malignant:", int(is_mal.sum()), "/", adata_cd4.n_obs)
print("by study:\n",
      adata_cd4.obs.groupby("study", observed=True)["is_malignant"].agg(["size", "sum"]))
del adata; gc.collect()
```

```
removing 105 ribosomal protein genes (RPS*/RPL*)
CD4 non-Treg: (169014, 49578)  | label = tcr_malignant_alice (HC forced benign: 0)
is_malignant: 68620 / 169014
by study:
                    size    sum
study                         
brunner2024       20222   5091
chennareddy2025   33735  20335
herrera2021         543    295
li2024           113070  41818
rindler2021_fi     1444   1081

2105
```

## Part 2 — SemanticSCVI (geometric, Geneformer prior)  · HEAVY (GPU kernel)

Map gene symbols → Ensembl (from `cnmf_malignant_counts.h5ad`), build/load the full-gene Geneformer
map, HVG-subset adata + map together, then train (or reload) the model. First run trains ~250k
cells on GPU; reruns hit the param-hashed cache.

```python  # cell 9
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
adata_cd4.var["gene_id"] = [sym2ens.get(s, s) for s in adata_cd4.var_names.astype(str)]
n_mapped = int(sum(g.startswith("ENSG") for g in adata_cd4.var["gene_id"]))
print(f"gene_id mapped to Ensembl: {n_mapped}/{adata_cd4.n_vars}")
del src; gc.collect()
```

```
gene_id mapped to Ensembl: 9766/49578

0
```

```python  # cell 10
# Build the FULL-gene Geneformer map first (cached). Then HVG-subset adata + map together.
# Delete the .pt to force a rebuild.
semantic_map = get_or_build_geneformer_map(
    adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
)
print("semantic_map (raw):", tuple(semantic_map.shape))

# Stale-cache guard: rebuild if the cached map rows don't match the current full-gene adata.
if semantic_map.shape[0] != adata_cd4.n_vars:
    print(f"shape mismatch ({semantic_map.shape[0]} vs {adata_cd4.n_vars}) — rebuilding")
    SEMANTIC_CACHE_GENEFORMER.unlink()
    semantic_map = get_or_build_geneformer_map(
        adata_cd4, SEMANTIC_CACHE_GENEFORMER, var_id_key="gene_id",
    )
    print("semantic_map (rebuilt):", tuple(semantic_map.shape))

# Keep only genes with a real Geneformer prior (non-zero embedding row). Out-of-vocab
# genes (~all non-coding RP11-*/AC0*/LINC*/MIR*) are zero-filled by the map builder and
# carry no semantic signal — drop them BEFORE HVG so HVG ranks variance only among
# semantically-grounded (effectively protein-coding) genes.
in_vocab = (semantic_map.norm(dim=1) > 0).cpu().numpy()
print(f"in-vocab (non-zero Geneformer row): {int(in_vocab.sum())}/{adata_cd4.n_vars}")
adata_cd4 = adata_cd4[:, in_vocab].copy()
semantic_map = semantic_map[torch.as_tensor(in_vocab)]

if HVG_TOP_N is not None:
    sc.pp.highly_variable_genes(adata_cd4, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
    keep = adata_cd4.var["highly_variable"].values
    adata_cd4 = adata_cd4[:, keep].copy()
    semantic_map = semantic_map[torch.as_tensor(keep)]
print("after HVG:", adata_cd4.shape, "| semantic_map:", tuple(semantic_map.shape))
```

```
Loading cached Geneformer map from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_cd4_atlas_geneformer.pt
semantic_map (raw): (49578, 1152)
in-vocab (non-zero Geneformer row): 9737/49578
after HVG: (169014, 2500) | semantic_map: (2500, 1152)
```

```python  # cell 11
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
         /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4_atlas
         /semantic_scvi/e451502ecb/model.pt already downloaded                                                     
[09:10:12]   loaded SemanticSCVI from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_geom_cd4_atlas/semantic_scvi/e451502ecb
trained: True

/home/projects/nyosef/zvise/scvi-tools-neural-nmf/src/scvi/data/fields/_dataframe_field.py:187: UserWarning: Category 0 in adata.obs['_scvi_labels'] has fewer than 3 cells. Models may not train properly.
  categorical_mapping = _make_column_categorical(
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/src/scvi/data/fields/_dataframe_field.py:227: UserWarning: Category 0 in adata.obs['_scvi_labels'] has fewer than 3 cells. Models may not train properly.
  new_mapping = _make_column_categorical(
```

## Part 3 — latent embedding, loadings, and malignancy separation

Mirrors nb12 Sections A.2–E with `is_malignant` (= the selected `MALIGNANCY_COL`) as the label.

```python  # cell 13
z = np.asarray(model.get_latent_representation())
ad_emb = sc.AnnData(z, obs=model.adata.obs.copy())

# Cache the latent UMAP (neighbors + umap are the slow part). Recompute on shape mismatch.
FORCE_UMAP = True
if (not FORCE_UMAP) and EMB_UMAP_NPY.exists() and \
        np.load(EMB_UMAP_NPY).shape[0] == ad_emb.n_obs:
    ad_emb.obsm["X_umap"] = np.load(EMB_UMAP_NPY)
    print("loaded cached emb UMAP", ad_emb.obsm["X_umap"].shape)
else:
    sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
    sc.tl.umap(ad_emb, random_state=0)
    np.save(EMB_UMAP_NPY, ad_emb.obsm["X_umap"])
    print("computed + cached emb UMAP ->", EMB_UMAP_NPY)

color = [c for c in ["study",  "cell_type_T", 'tcr_malignant_alice','cnvT_cnvcluster','mrvi_m2_pseudosample'] if c in ad_emb.obs]
sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False)
```

```
computed + cached emb UMAP -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/semantic_geom_cd4_atlas_emb_umap.npy

<Figure size 1664x1440 with 5 Axes>
```

```python  # cell 14
W = model.get_loadings().reindex(adata_cd4.var_names)   # genes x N_LATENT
n_factors = z.shape[1]
print("W (loadings):", W.shape, "| n_factors:", n_factors, "| columns:", list(W.columns))
W.head()
```

```
W (loadings): (2500, 10) | n_factors: 10 | columns: ['Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9']

             Z_0       Z_1       Z_2       Z_3       Z_4       Z_5       Z_6  \
A4GALT  0.017626  0.022922  0.278967  0.018663  1.472839  0.020350  0.016557   
AARD    0.026146  0.023540  0.025808  0.029813  0.030723  0.017964  0.023411   
ABCA13  0.021085  0.022515  0.091843  0.016488  0.049343  0.020907  0.030003   
ABCC9   0.003305  0.003304  0.012613  0.564566  0.004650  0.002899  0.003722   
ABCG2   0.007369  0.005337  0.018692  0.004499  0.010895  0.005034  0.006627   

             Z_7       Z_8       Z_9  
A4GALT  0.021802  0.018903  0.019010  
AARD    0.067237  2.781698  0.022375  
ABCA13  0.021519  2.547055  0.016297  
ABCC9   0.003382  0.013497  0.002103  
ABCG2   0.542078  0.019464  0.004855
```

```python  # cell 15
# top loading genes per factor (proj_Z_k) — same as the ref notebook
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
proj_Z_0 proj_Z_1  proj_Z_2  proj_Z_3   proj_Z_4 proj_Z_5 proj_Z_6 proj_Z_7  proj_Z_8  proj_Z_9
top_1      CRYAB     GZMA    SLC1A6     PSAT1       XCL2    NR4A3      HBB    UBE2C     APOC4      MT1E
top_2   SERPINH1     CCL5    SLC4A1     TRIB3      FCRL4    NR4A2     GZMK     PLK1    NKAIN1     FOXP3
top_3       DLL4    CRIP2     GAP43     PYCR1       CCL1     PER1   CCL4L2     RRM2   HSD17B2      MT1X
top_4       RRAD     IFI6     DCHS2    SDF2L1       CCL3      CTH    TCL1A    GTSE1       CA3      MT2A
top_5     DNAJB1     HPGD     KCNJ6      MANF       IL31   ZNF331    KLRC4     ASPM   COL10A1      MT1F
top_6        ARC    ITGB7      PSD2    CCDC86       XCL1    THBS1     IL24      IL9    GRIN2A    S100A4
top_7      HBEGF    MYADM    POU3F3       SRM       TGM5    CCL20   PTGER2      GFY      TTC6    LGALS1
top_8     COL2A1     PRF1     KCNJ9     NOP16       CCL4  TNFRSF9    FOXJ1   DLGAP5     L1TD1      KRT7
top_9      RASD1    CXCR6    HHIPL1     MYDGF     RFPL4A   HILPDA     SELL    CDC20     NRXN1     LAIR2
top_10      IFNG    FXYD7       PYY     IFRD2   C19orf84     IL13     HBA2    BIRC5     SNTG1  C12orf75
top_11    CRYBA4      F2R    NKAIN2     PAICS       LCN6    NR4A1     CCL4    HMGA2    IGDCC3    CORO1B
top_12      IL10   CDC25B    HCRTR2     PDIA4       TCN1    FOSL2   CD300A    CCNA2    KLHL30     CRIP1
top_13     HMOX1     KLF2       BTC  PPP1R14B        IL5   NFKBID    ITGA4    HJURP     GRIA4      UCP2
top_14      IL22      MX1      PCP4     ISOC2       GZMB      GEM    CXCR5    AURKB     SCN7A    PYCARD
top_15     NR4A1   UBXN11     PTH1R   CDK2AP2    CD300LD     IL21     KLF2    CENPA  TMEM132B      GLRX
top_16    DNAJB4    LIME1    SH3GL2      ODC1       AOC1   SEC24A    S1PR1   KIF20A    DIRAS1     DNPH1
top_17      IER3     ABI3    STK32A      LYAR      FOXC1    NR1D1   MS4A6A    CCNB2     ZFHX4    CXCL13
top_18  SERPINE1     AOAH      RARB  EBNA1BP2      XIRP1    DUSP5    CRTAM    CDCA3      MYH7     NINJ2
top_19     RGS16   GPR171  ATP6V0D2      CMC1       GZMH     FOSB   GIMAP4   PKMYT1     VEGFC      CD27
top_20    TNFSF9    SYTL2    GABRA4     MRTO4   KRTAP4-3      REL     NEFM    SPC25     NR0B1      CAPG
top_21   RASL11A  RASGRP2   GUCY1A2      NPM3  KRTAP12-3     ELL2     AIRE    INSM1   SLITRK4    TAGLN2
top_22      BAG3   GIMAP7      NTF3     CMSS1      MMP12     RGS1   DEFB4A  RIPPLY3     LIN7A  TNFRSF18
top_23     SNAI1     GBP5   COL11A1    SEC11C        CGA    ARL5B     HES1    KIF14   ABHD12B   S100A10
top_24     BAMBI     OAS1    ELAVL4    TOMM40      HOXB8      JMY    PLAC8   CKAP2L      CR1L       DBI
top_25   CCDC121    KLRB1    HS3ST6      P4HB      LCE3D   PMAIP1    KRT75    KIF2C     SYT13       VIM
top_26      EDN1     CD8A     PRDM6    CRELD2    KIR2DL4  BHLHE40     BEX3    TROAP      PALM   HTATIP2
top_27      CD83     TOB1      DTNA     NUCB2        IL3   SLC2A3   CHI3L2     CDK1    TRIM10     ANXA2
top_28     ARL4D     AQP3      UCP1      FASN      VCAM1   VPS37B     KRT2   RNASE2      GLI2     PTTG1
top_29       ID1     NKG7       RGN     BLVRB      MMP10    DUSP4   ABLIM1    CDCA8      PCK1   TMEM14A
top_30   ZFAND2A     PI16      SIX4      SSR3       NKG7    DDIT3    NUAK2    KIF15    RBFOX1     PRDX1

        proj_Z_0 proj_Z_1  proj_Z_2  proj_Z_3   proj_Z_4 proj_Z_5 proj_Z_6  \
top_1      CRYAB     GZMA    SLC1A6     PSAT1       XCL2    NR4A3      HBB   
top_2   SERPINH1     CCL5    SLC4A1     TRIB3      FCRL4    NR4A2     GZMK   
top_3       DLL4    CRIP2     GAP43     PYCR1       CCL1     PER1   CCL4L2   
top_4       RRAD     IFI6     DCHS2    SDF2L1       CCL3      CTH    TCL1A   
top_5     DNAJB1     HPGD     KCNJ6      MANF       IL31   ZNF331    KLRC4   
top_6        ARC    ITGB7      PSD2    CCDC86       XCL1    THBS1     IL24   
top_7      HBEGF    MYADM    POU3F3       SRM       TGM5    CCL20   PTGER2   
top_8     COL2A1     PRF1     KCNJ9     NOP16       CCL4  TNFRSF9    FOXJ1   
top_9      RASD1    CXCR6    HHIPL1     MYDGF     RFPL4A   HILPDA     SELL   
top
... [2811 more chars truncated]
```

### Malignancy calls on the latent

All calls were joined into `obs` in Part 1 and ride along on `ad_emb` (the latent AnnData).
The headline label is `is_malignant` (= the selected `MALIGNANCY_COL`, HC forced benign); the
other calls (`M.MALIGNANCY_SOURCES`) stay available as a method-comparison in the panels below.

```python  # cell 17
# Calls compared in the panels below: the driving label + every registry call present in obs.
CALLS = [c for c in M.MALIGNANCY_SOURCES if c in ad_emb.obs]
for c in CALLS:
    ad_emb.obs[c] = ad_emb.obs[c].astype(bool)
print("driving call:", MALIGNANCY_COL)
print("malignant counts:", {c: int(ad_emb.obs[c].sum()) for c in CALLS})
```

```
driving call: tcr_malignant_alice
malignant counts: {'tcr_malignant_alice': 68620, 'cnvT_cell_gmm': 98904, 'cnvT_cell_mrvi': 76263, 'cnvT_cnvcluster': 116027, 'cnvT_cnvcluster_mrvi': 131799, 'cnv_cd4_cd8ref': 111341, 'mrvi_m1_labelspread': 110990, 'mrvi_m2_pseudosample': 107760}
```

### Per-factor separation heatmaps (one per call)

For factor `Z_k`, cells are sorted by `Z_k` and binned along x; bin color = fraction of malignant
cells. A clean red↔blue gradient = a factor that separates that call. Class-balanced per call. Row
labels carry the per-factor AUROC.

```python  # cell 19
M.plot_factor_heatmaps(z, ad_emb, CALLS, FIG_DIR, prefix="semantic_geom_cd4_atlas")
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_tcr_malignant_alice.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_cnvT_cell_gmm.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_cnvT_cell_mrvi.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_cnvT_cnvcluster.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_cnvT_cnvcluster_mrvi.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_cnv_cd4_cd8ref.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_mrvi_m1_labelspread.png

<Figure size 1100x600 with 2 Axes>
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_sep_heatmap_mrvi_m2_pseudosample.png

<Figure size 1100x600 with 2 Axes>
```

### Factor geometry vs the driving call (`is_malignant` = `MALIGNANCY_COL`)

2-factor scatter grid, 3-D scatter, and a gene-direction plane sweep, all labeled by
`is_malignant`. **Re-pick `FACTORS_GRID` / `FX3,FY3,FZ3` from the per-factor `aucs` printed
below** — the defaults are nb12's picks and may not be the best separators for this call.

```python  # cell 21
MALIGNANCY_COL = "mrvi_m2_pseudosample"   # transcriptomic pseudosample (nb17)

CALL = MALIGNANCY_COL
POS_LABEL, NEG_LABEL = "malignant", "benign"
MAX_PER_CLASS = 20000   # class-balance cap per call

is_mal = ad_emb.obs["is_malignant"].astype(bool).to_numpy().astype(int)
y_true = is_mal
plot_idx = M.balanced_idx(is_mal, MAX_PER_CLASS)
aucs = M.call_aucs(z, is_mal)
print(f"{CALL}: {int(is_mal.sum())} malignant / {len(is_mal)} cells | "
      f"balanced subsample {len(plot_idx)} ({len(plot_idx) // 2}/class)")
print("per-factor AUROC:", {k: round(v, 3) for k, v in aucs.items()})

# top loading genes per factor (consumed by the gene-direction sweep below)
top_df = pd.DataFrame({
    f"proj_{col}": W[col].nlargest(PER_PROJECTION_N_TOP).index.tolist()
    for col in W.columns
})
top_df.index = [f"top_{i + 1}" for i in range(PER_PROJECTION_N_TOP)]
```

```
mrvi_m2_pseudosample: 68620 malignant / 169014 cells | balanced subsample 40000 (20000/class)
per-factor AUROC: {'Z_0': 0.549, 'Z_1': 0.52, 'Z_2': 0.618, 'Z_3': 0.556, 'Z_4': 0.615, 'Z_5': 0.638, 'Z_6': 0.78, 'Z_7': 0.655, 'Z_8': 0.708, 'Z_9': 0.773}
```

```python  # cell 22
FACTORS_GRID = [4, 5, 7, 9, 1]   # nb12 status picks; re-pick from `aucs` if other factors win
M.plot_factor_grid(z, is_mal, plot_idx, FACTORS_GRID, aucs, CALL, POS_LABEL, NEG_LABEL, FIG_DIR)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/mrvi_m2_pseudosample_scatter_grid_4_5_7_9_1.png

<Figure size 1500x600 with 10 Axes>
```

#### 3-D scatter — factors (Z_FX3, Z_FY3, Z_FZ3)

Diagonal score `s3 = z(Z_FX3) + z(Z_FY3) + z(Z_FZ3)`; left panel colored by the call, right by
`s3`, with median / Youden-best planes drawn at constant `s3`.

```python  # cell 24
FX3, FY3, FZ3 = 1, 9, 7   # nb12 picks; re-pick from `aucs`
geom = M.make_geom3d(z, FX3, FY3, FZ3, plot_idx)
M.plot_3d_threshold(geom, is_mal, CALL, POS_LABEL, NEG_LABEL, FIG_DIR)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/mrvi_m2_pseudosample_scatter3d_Z1_Z9_Z7.png

<Figure size 1400x700 with 3 Axes>
```

```python  # cell 25
# 1) Pseudotime via DPT on the latent kNN graph. ad_emb's neighbors may have been skipped
#    when the UMAP was loaded from cache — rebuild the graph if missing.
if "neighbors" not in ad_emb.uns:
    sc.pp.neighbors(ad_emb, use_rep="X", random_state=0)
_root_mask = ~ad_emb.obs["is_malignant"].astype(bool).to_numpy()   # benign side as DPT root
assert _root_mask.any(), f"no benign cell for {CALL}"
root_idx = int(np.where(_root_mask)[0][0])
ad_emb.uns["iroot"] = root_idx
sc.tl.diffmap(ad_emb, random_state=0)
sc.tl.dpt(ad_emb)
pt = ad_emb.obs["dpt_pseudotime"].to_numpy()
pt = np.where(np.isfinite(pt), pt, np.nan)   # DPT can emit inf on disconnected components
print(f"pseudotime: root_idx={root_idx}  min={np.nanmin(pt):.3f}  "
      f"median={np.nanmedian(pt):.3f}  max={np.nanmax(pt):.3f}  n_nan={np.isnan(pt).sum()}")
```

```
pseudotime: root_idx=0  min=0.000  median=0.152  max=1.000  n_nan=0
```

```python  # cell 26
# 2) Editable gene list — default = top 4 from each of Z_1, Z_7, Z_6.
GENES_FOR_PLANE = (
    top_df["proj_Z_1"].head(4).tolist()
    + top_df["proj_Z_9"].head(4).tolist()
    + top_df["proj_Z_7"].head(4).tolist()
)
GENES_FOR_PLANE = list(dict.fromkeys(GENES_FOR_PLANE))  # de-dup, preserve order
missing = [g for g in GENES_FOR_PLANE if g not in adata_cd4.var_names]
assert not missing, f"missing in adata_cd4: {missing}"
print(f"{len(GENES_FOR_PLANE)} genes:", GENES_FOR_PLANE)
```

```
12 genes: ['GZMA', 'CCL5', 'CRIP2', 'IFI6', 'MT1E', 'FOXP3', 'MT1X', 'MT2A', 'UBE2C', 'PLK1', 'RRM2', 'GTSE1']
```

```python  # cell 27
# 3) Candidate direction vectors in (Z_FX3, Z_FY3, Z_FZ3) — OLS coeffs on standardized axes.
from sklearn.linear_model import LogisticRegression

Z3z = geom["Z3z"]   # standardized 3-axis subspace from make_geom3d
size_factor = np.asarray(adata_cd4.X.sum(axis=1)).ravel().astype(np.float64)
size_factor[size_factor == 0] = 1.0

dir_rows = [("diagonal_(1,1,1)", np.array([1.0, 1.0, 1.0]))]
dir_rows.append(("pseudotime", M.fit_direction(Z3z, pt)))
for g in GENES_FOR_PLANE:
    dir_rows.append((g, M.fit_direction(Z3z, M.gene_lognorm(adata_cd4, g, size_factor))))
lr = LogisticRegression(max_iter=1000, C=1.0).fit(Z3z, is_mal)
dir_rows.append((f"LR_optimal_(Z{FX3},Z{FY3},Z{FZ3})", lr.coef_.ravel().astype(np.float64)))

dirs = {name: v / max(np.linalg.norm(v), 1e-12) for name, v in dir_rows}
print(f"{len(dirs)} candidate directions:", list(dirs.keys()))
```

```
15 candidate directions: ['diagonal_(1,1,1)', 'pseudotime', 'GZMA', 'CCL5', 'CRIP2', 'IFI6', 'MT1E', 'FOXP3', 'MT1X', 'MT2A', 'UBE2C', 'PLK1', 'RRM2', 'GTSE1', 'LR_optimal_(Z1,Z9,Z7)']
```

```python  # cell 28
# 4) Score each direction: AUROC + Youden-best threshold + accuracy stats.
results = (
    pd.DataFrame([M.eval_direction(n, v, Z3z, y_true) for n, v in dirs.items()])
    .sort_values("AUROC", ascending=False)
    .reset_index(drop=True)
)
with pd.option_context("display.max_columns", None, "display.width", 200,
                       "display.float_format", "{:.4f}".format):
    print(results.to_string(index=False))
out_csv = FIG_DIR / f"semantic_geom_cd4_atlas_{CALL}_gene_planes_stats.csv"
results.to_csv(out_csv, index=False)
print("saved", out_csv)
results
```

```
name  AUROC    acc   prec    rec     f1    TP    FP    FN    TN     OR      p      vx      vy      vz     thr  flipped
LR_optimal_(Z1,Z9,Z7) 0.7951 0.7385 0.6603 0.7332 0.6948 50311 25883 18309 74511 7.9105 0.0000 -0.2819  0.9071 -0.3125  0.1568    False
                FOXP3 0.7933 0.7295 0.6358 0.7814 0.7011 53623 30718 14997 69676 8.1103 0.0000 -0.3325  0.9190 -0.2118  0.0320    False
                 MT2A 0.7632 0.7079 0.6181 0.7342 0.6712 50383 31132 18237 69262 6.1464 0.0000 -0.2865  0.9455  0.1546  0.0396    False
                 MT1X 0.7561 0.7032 0.6146 0.7215 0.6638 49508 31043 19112 69351 5.7871 0.0000 -0.1585  0.9646  0.2108  0.0582    False
                 MT1E 0.7442 0.6907 0.5991 0.7197 0.6539 49385 33049 19235 67345 5.2318 0.0000 -0.2040  0.9360  0.2868  0.0132    False
           pseudotime 0.7227 0.6679 0.5729 0.7154 0.6363 49092 36594 19528 63800 4.3829 0.0000 -0.2313  0.8903  0.3924 -0.0498    False
                 IFI6 0.6766 0.6436 0.5563 0.6041 0.5792 41452 33061 27168 67333 3.1074 0.0000  0.6274  0.7782  0.0260  0.1968    False
                CRIP2 0.6590 0.6331 0.5451 0.5815 0.5627 39905 33303 28715 67091 2.7996 0.0000  0.5894  0.7618  0.2686  0.2062    False
                 CCL5 0.6582 0.6111 0.5153 0.7090 0.5968 48654 45765 19966 54629 2.9088 0.0000 -0.7146  0.2059 -0.6686 -0.1525     True
     diagonal_(1,1,1) 0.5845 0.5768 0.4802 0.5159 0.4974 35402 38317 33218 62077 1.7266 0.0000  0.5774  0.5774  0.5774  0.1574    False
                GTSE1 0.5750 0.5864 0.4904 0.4731 0.4816 32466 33742 36154 66652 1.7738 0.0000  0.1353 -0.2489 -0.9590  0.3223     True
                 PLK1 0.5571 0.5770 0.4771 0.4373 0.4564 30009 32885 38611 67509 1.5955 0.0000  0.1160 -0.2996 -0.9470  0.3501     True
                 RRM2 0.5545 0.5724 0.4724 0.4557 0.4639 31269 34921 37351 65473 1.5696 0.0000  0.2359 -0.2910 -0.9272  0.3298     True
                UBE2C 0.5517 0.5725 0.4715 0.4392 0.4548 30139 33778 38481 66616 1.5446 0.0000  0.1760 -0.3075 -0.9351  0.3452     True
                 GZMA 0.5469 0.5428 0.4470 0.5313 0.4855 36459 45110 32161 55284 1.3893 0.0000  0.9069  0.3386  0.2507  0.1026    False
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/semantic_geom_cd4_atlas_mrvi_m2_pseudosample_gene_planes_stats.csv

                     name     AUROC       acc      prec       rec        f1  \
0   LR_optimal_(Z1,Z9,Z7)  0.795128  0.738531  0.660301  0.733183  0.694836   
1                   FOXP3  0.793336  0.729519  0.635788  0.781449  0.701133   
2                    MT2A  0.763155  0.707900  0.618083  0.734232  0.671169   
3                    MT1X  0.756087  0.703249  0.614617  0.721481  0.663775   
4                    MT1E  0.744196  0.690653  0.599085  0.719688  0.653872   
5              pseudotime  0.722704  0.667945  0.572929  0.715418  0.636294   
6                    IFI6  0.676573  0.643645  0.556306  0.604080  0.579210   
7                   CRIP2  0.658984  0.633060  0.545091  0.581536  0.562724   
8                    CCL5  0.658191  0.611091  0.515299  0.709035  0.596839   
9        diagonal_(1,1,1)  0.584510  0.576751  0.480229  0.515914  0.497432   
10                  GTSE1  0.575030  0.586448  0.490364  0.473127  0.481591   
11                   PLK1  0.557063  0.576982  0.477136  0.437321  0.456362   
12                   RRM2  0.554492  0.572390  0.472413  0.455683  0.463897   
13                  UBE2C  0.551679  0.572467  0.471533  0.439216  0.454801   
14                   GZMA  0.546924  0.542813  0.446971  0.531317  0.485508   

       TP     FP     FN     TN        OR              p        vx        vy  \
0   50311  25883  18309  74511  7.910503   0.000000e+00 -0.281878  0.907121   
1   53623  30718  14997  69676  8.110301   0.000000e+00 -0.332488  0.919028   
2   50383  31132  18237  69262  6.146369   0.000000e+00 -0.286548  0.945510   
3   49508  31043  19112  69351  5.787064   0.000000e+00 -0.158521  0.964588   
4   49385  33049  19235  67345  5.23
... [1375 more chars truncated]
```

```python  # cell 29
# 5) 3-D scatter of the top-AUROC plane (left = label, right = marker-gene expression).
MARK_GENE = "KLF2" if "KLF2" in adata_cd4.var_names else GENES_FOR_PLANE[0]
mark_expr = M.gene_lognorm(adata_cd4, MARK_GENE, size_factor)
M.plot_best_plane(geom, results, is_mal, CALL, POS_LABEL, NEG_LABEL, mark_expr, MARK_GENE, FIG_DIR)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/mrvi_m2_pseudosample_scatter3d_best_plane_LR_optimal_Z1_Z9_Z7.png

<Figure size 1400x700 with 3 Axes>
```

```python  # cell 30
# Interactive rotatable 3-D version of the top-AUROC plane (standalone .html).
M.plot_best_plane_plotly(geom, results, is_mal, CALL, POS_LABEL, NEG_LABEL, mark_expr, MARK_GENE, FIG_DIR)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/mrvi_m2_pseudosample_scatter3d_best_plane_LR_optimal_Z1_Z9_Z7.html

<IPython.lib.display.IFrame at 0x150cb3f61e80>
```

## Part 4 — fair per-factor malignancy comparison vs CNV & M2

Each of the 10 factors is turned into a malignancy classifier and benchmarked head-to-head
against the two production calls (**CNV** `cnv_cd4_cd8ref`/nb15, **transcriptomic M2**
`mrvi_m2_pseudosample`/nb17) under one shared, leakage-free protocol.

**Ground truth = ALICE TCR** (`tcr_malignant_alice`): malignant = ALICE; benign anchor =
`has_tcr & ~ALICE & ~tcr_is_expanded` (the nb16/17 benign-anchor rule). Non-anchor cells are
**excluded** from evaluation; HC donors are benign.

**Fairness controls** — every predictor is scored on the *same* cells, with the *same* labels and
the *same* donor-grouped folds:
1. **Same cells**: intersection of the model's CD4 cells ∩ TCR anchors ∩ cells with a CNV score
   ∩ cells with an M2 score.
2. **Threshold from TCR, out-of-fold**: `StratifiedGroupKFold(by donor)`; per factor, per training
   fold pick the F1-optimal threshold *shape* — one-sided `>`/`<` **or** a two-sided band
   (`lo<x<hi` / outside) — then predict the held-out fold. Metrics are pooled over the OOF calls.
3. **All-factor logistic** (same folds, train-only fit) = the latent's combined ceiling.
4. **CNV & M2** are taken as their **published boolean calls** (not re-thresholded), but each also
   gets a threshold-free **AUC** from its continuous score (`cnv_cell_score`, `prob_m2`) — that AUC
   is the fair tiebreaker, since the published calls used in-sample thresholds.

CPU-light (reuses `z` already in memory).

```python  # cell 32
importlib.reload(M)

# --- continuous scores + TCR anchor fields from the two method outputs (indexed by barcode) ---
cnv = pd.read_parquet(NB_DIR / "data/atlas_joint/skin_cd4_cd8ref_malignancy.parquet")
m2  = pd.read_csv(NB_DIR / "data/atlas_joint/mrvi_malignancy/calls.csv").set_index("barcode")

bc = ad_emb.obs_names
factor_cols = [f"Z_{k}" for k in range(z.shape[1])]
E = pd.DataFrame(z, index=bc, columns=factor_cols)
E["donor"]   = ad_emb.obs["donor"].to_numpy()
E["study"]   = ad_emb.obs["study"].astype(str).to_numpy()
E["disease"] = ad_emb.obs["disease"].astype(str).to_numpy()

# ground truth (ALICE) + benign-anchor inputs + the two methods' score/call
E["alice"]        = ad_emb.obs["tcr_malignant_alice"].astype(bool).to_numpy()
E["has_tcr"]      = cnv["has_tcr"].reindex(bc).astype("boolean").fillna(False).to_numpy(bool)
E["tcr_expanded"] = cnv["tcr_is_expanded"].reindex(bc).astype("boolean").fillna(False).to_numpy(bool)
E["cnv_score"]    = cnv["cnv_cell_score"].reindex(bc).to_numpy(float)
E["cnv_call"]     = cnv["cnv_malig_cluster"].reindex(bc)        # published CD4/CD8-ref call (nb15 strat3)
E["m2_score"]     = m2["prob_m2"].reindex(bc).to_numpy(float)
E["m2_call"]      = m2["call_m2"].reindex(bc)                   # published M2 call (nb17, thr=0.429)

# --- anchors (truth = ALICE) and the common evaluable set ---
hc  = E["disease"].eq("HC").to_numpy()
mal = E["alice"].to_numpy() & ~hc
ben = E["has_tcr"].to_numpy() & ~E["alice"].to_numpy() & ~E["tcr_expanded"].to_numpy() & ~mal
have = (mal | ben) & np.isfinite(E["cnv_score"].to_numpy()) & np.isfinite(E["m2_score"].to_numpy()) \
       & E["cnv_call"].notna().to_numpy() & E["m2_call"].notna().to_numpy()

E = E.loc[have].copy()
E["y"] = mal[have].astype(int)
fold = M.make_folds(E["y"].to_numpy(), E["donor"].to_numpy(), n_splits=5, seed=0)

print(f"evaluable anchor cells: {len(E):,}  | malignant {int(E['y'].sum()):,} "
      f"({E['y'].mean():.1%})  | donors {E['donor'].nunique()}  | folds {fold.max() + 1}")
print("per-study (size / malignant):")
print(E.groupby("study", observed=True)["y"].agg(["size", "sum"]))
```

```
evaluable anchor cells: 94,030  | malignant 68,605 (73.0%)  | donors 32  | folds 5
per-study (size / malignant):
                  size    sum
study                        
brunner2024       6807   5091
chennareddy2025  23855  20335
herrera2021        397    280
li2024           61846  41818
rindler2021_fi    1125   1081
```

```python  # cell 33
from sklearn.linear_model import LogisticRegression

# factors to surface in the table + plots (the rest stay in the saved CSV)
SHOW_FACTORS = [4, 5, 7, 1]

y = E["y"].to_numpy()
rows = {}

# (1) each factor: OOF threshold classifier (shape learned per training fold from TCR)
oof_scores = {}   # polarity-folded score per predictor, for the ROC overlay
for c in factor_cols:
    s = E[c].to_numpy(float)
    oof, form, params = M.oof_eval(s, y, fold)
    r = M.binary_scores(y, oof, score=s)
    r["form"] = form
    r["threshold"] = ", ".join(f"{p:.3g}" for p in params)
    rows[c] = r
    oof_scores[c] = s if M.roc_auc_score(y, s) >= 0.5 else -s

# (2) all-factor logistic - per-fold fit + per-fold F1-optimal cut (no leakage)
X = E[factor_cols].to_numpy(float)
oof_p = np.full(len(y), np.nan); oof_c = np.zeros(len(y), bool)
for k in np.unique(fold):
    tr, te = fold != k, fold == k
    lr = LogisticRegression(max_iter=1000, random_state=0).fit(X[tr], y[tr])
    ptr, pte = lr.predict_proba(X[tr])[:, 1], lr.predict_proba(X[te])[:, 1]
    form_k, params_k, _ = M.fit_threshold_form(ptr, y[tr])
    oof_p[te] = pte; oof_c[te] = M.apply_threshold_form(pte, form_k, params_k)
r = M.binary_scores(y, oof_c, score=oof_p); r["form"] = "logistic"; r["threshold"] = "OOF F1-opt"
rows["semantic_all_factors"] = r
oof_scores["semantic_all_factors"] = oof_p

# (3) the two production methods - published boolean call as-is + threshold-free AUC
r = M.binary_scores(y, E["cnv_call"].astype(bool).to_numpy(), score=E["cnv_score"].to_numpy(float))
r["form"] = "published"; r["threshold"] = "nb15 per-donor GMM"; rows["cnv_cd4_cd8ref"] = r
oof_scores["cnv_cd4_cd8ref"] = (E["cnv_score"].to_numpy(float)
                                if M.roc_auc_score(y, E["cnv_score"]) >= 0.5 else -E["cnv_score"].to_numpy(float))
r = M.binary_scores(y, E["m2_call"].astype(bool).to_numpy(), score=E["m2_score"].to_numpy(float))
r["form"] = "published"; r["threshold"] = "0.429 (nb17 OOF)"; rows["mrvi_m2_pseudosample"] = r
oof_scores["mrvi_m2_pseudosample"] = E["m2_score"].to_numpy(float)

# --- full results table (all 10 factors), sorted by OOF F1 vs TCR ---
order_cols = ["form", "threshold", "f1", "auc", "balanced_acc", "precision",
              "recall", "specificity", "jaccard", "n"]
res_full = pd.DataFrame(rows).T[order_cols].sort_values("f1", ascending=False)
TABLES = NB_DIR / "tables"; TABLES.mkdir(exist_ok=True)
res_full.to_csv(TABLES / "factor_vs_cnv_m2_malignancy.csv")
print("saved (all factors):", TABLES / "factor_vs_cnv_m2_malignancy.csv")

# --- surfaced subset: selected factors + combined latent + the two methods ---
show = [f"Z_{k}" for k in SHOW_FACTORS] + ["semantic_all_factors", "cnv_cd4_cd8ref", "mrvi_m2_pseudosample"]
res = res_full.loc[[k for k in show if k in res_full.index]]
with pd.option_context("display.width", 200, "display.max_columns", None,
                       "display.float_format", "{:.3f}".format):
    print(res)
res
```

```
saved (all factors): /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/tables/factor_vs_cnv_m2_malignancy.csv
                           form           threshold    f1   auc balanced_acc precision recall specificity jaccard      n
Z_4                          gt               -1.41 0.848 0.711        0.559     0.755  0.967       0.151   0.735  94030
Z_5                          lt                 1.6 0.839 0.670        0.500     0.730  0.987       0.013   0.723  94030
Z_7                    band_out        0.615, 0.857 0.843 0.668        0.532     0.743  0.975       0.089   0.729  94030
Z_1                          lt               0.981 0.847 0.627        0.525     0.740  0.991       0.059   0.735  94030
semantic_all_factors   logistic          OOF F1-opt 0.928 0.948        0.851     0.913  0.944       0.757   0.866  94030
cnv_cd4_cd8ref        published  nb15 per-donor GMM 0.700 0.633        0.571     0.776  0.637       0.505   0.538  94030
mrvi_m2_pseudosample  published    0.429 (nb17 OOF) 0.934 0.956        0.874     0.930  0.937       0.810   0.875  94030

                           form           threshold        f1       auc  \
Z_4                          gt               -1.41  0.847564   0.71119   
Z_5                          lt                 1.6  0.839012  0.669883   
Z_7                    band_out        0.615, 0.857  0.843064   0.66833   
Z_1                          lt               0.981  0.846938  0.626526   
semantic_all_factors   logistic          OOF F1-opt  0.928092  0.947719   
cnv_cd4_cd8ref        published  nb15 per-donor GMM  0.699625  0.633396   
mrvi_m2_pseudosample  published    0.429 (nb17 OOF)  0.933613   0.95632   

                     balanced_acc precision    recall specificity   jaccard  \
Z_4                      0.559069  0.754539  0.966752    0.151386  0.735454   
Z_5                      0.499984  0.729601  0.987027     0.01294  0.722671   
Z_7                      0.531708  0.742671  0.974841    0.088574  0.728704   
Z_1                      0.524701  0.739579  0.990759    0.058643  0.734512   
semantic_all_factors     0.850553  0.913027  0.943663    0.757443  0.865832   
cnv_cd4_cd8ref           0.570943  0.776391  0.636674    0.505211  0.538018   
mrvi_m2_pseudosample     0.873712  0.930249  0.937002    0.810423  0.875492   

                          n  
Z_4                   94030  
Z_5                   94030  
Z_7                   94030  
Z_1                   94030  
semantic_all_factors  94030  
cnv_cd4_cd8ref        94030  
mrvi_m2_pseudosample  94030
```

```python  # cell 34
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

cnv_f1, m2_f1 = res.loc["cnv_cd4_cd8ref", "f1"], res.loc["mrvi_m2_pseudosample", "f1"]

# (a) OOF F1 per surfaced predictor, with CNV / M2 reference lines
order = res.index.tolist()
f1v = res["f1"].astype(float).to_numpy()
col = {"semantic_all_factors": "#2ca02c"}
bars = [col.get(i, "#c0392b" if i in ("cnv_cd4_cd8ref", "mrvi_m2_pseudosample") else "#888")
        for i in order]
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(x, f1v, width=0.6, color=bars)
ax.axhline(cnv_f1, ls="--", c="#c0392b", lw=1, label=f"CNV ({cnv_f1:.2f})")
ax.axhline(m2_f1, ls=":", c="#2471a3", lw=1, label=f"M2 ({m2_f1:.2f})")
for xi, v in zip(x, f1v):
    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
ax.set_ylabel("OOF F1 vs TCR")
ax.set_title("Per-factor vs combined latent vs CNV/M2 - OOF F1 vs TCR")
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(FIG_DIR / "factor_vs_cnv_m2_f1.png", dpi=150, bbox_inches="tight")
plt.show()

# (b) ROC overlay: surfaced factors + combined + CNV + M2 (scores already polarity-folded)
roc_factors = [f"Z_{k}" for k in SHOW_FACTORS]
fig, ax = plt.subplots(figsize=(5, 3))
for name in roc_factors + ["semantic_all_factors", "cnv_cd4_cd8ref", "mrvi_m2_pseudosample"]:
    s = oof_scores[name]
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, lw=1.2, label=f"{name} ({res.loc[name, 'auc']:.2f})")
ax.plot([0, 1], [0, 1], c="k", lw=0.5, ls=":")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("Malignancy ROC vs TCR: factors vs CNV / M2")
ax.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "factor_vs_cnv_m2_roc.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 500x300 with 1 Axes>
<Figure size 500x300 with 1 Axes>
```

**Reading the table** (`tables/factor_vs_cnv_m2_malignancy.csv`):

- **`f1` / `balanced_acc` / `jaccard`** — agreement of each predictor's binary call with the ALICE
  TCR truth, on the shared anchor set. Factors and the combined model are **out-of-fold** (threshold
  fit on other donors); CNV and M2 are their **published** calls.
- **`auc`** — threshold-free separation (polarity-folded). This is the fair head-to-head number,
  because the CNV/M2 *thresholds* were tuned in-sample whereas the factor thresholds are OOF.
- **`form` / `threshold`** — the shape chosen for each factor (`gt`/`lt` one-sided, or
  `band_in`/`band_out` two-sided) and its modal cut(s) across folds.

Sanity checks: `n` is identical for every row (same cells); M2's `f1`/`auc` here should land near its
reported ~0.88/0.92, confirming the harness and cell set are wired correctly. Rename the bar-plot
title to the actual finding once you've seen the numbers.
