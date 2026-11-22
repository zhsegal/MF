# 00_preprocess/01_download_and_prelim.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/00_preprocess/01_download_and_prelim.ipynb` and is never run._

# CTCL (MF) atlas — preliminary analysis

Loads the local Li/Strobl/Haniffa CTCL atlas h5ads from `./data/` and runs a light preliminary pass:
1. Locate + load local h5ads
2. Inspect obs/var/embeddings
3. QC distributions per donor
4. UMAP overview (uses pre-computed integrated embedding if present)
5. B-cell fraction CTCL vs healthy/AD/psoriasis (Fig. 7 of the paper)
6. TH1/TH2/TH17 scoring across stage in T cells (Fig. 3)

Heavier steps (inferCNV, cell2location, LIANA, drug2cell) are intentionally out of scope here — they live in follow-up notebooks. See `CTCL_atlas_download_and_preliminary_analysis.md` for the full plan.

**Citation:** Li R., Strobl J., Poyner E.F.M. *et al. Nat. Immunol.* **25**, 2320–2330 (2024). https://doi.org/10.1038/s41590-024-02018-1

```python  # cell 1
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

sc.settings.set_figure_params(dpi=80, facecolor="white", figsize=(5, 3))
sc.settings.figdir = str(FIG_DIR)
print("scanpy", sc.__version__)


def _resolve_nb_dir():
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")

NB_DIR = _resolve_nb_dir()
DATA_DIR = NB_DIR / "data"
FIG_DIR = NB_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

H5AD_INTEGRATED = DATA_DIR / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"
H5AD_CTCL_ONLY  = DATA_DIR / "CTCL_all_final_portal_tags.h5ad"
H5AD_PATH = H5AD_CTCL_ONLY  # primary for B-cell enrichment + UMAP; swap to CTCL_ONLY for stage analyses

print(f"NB_DIR    = {NB_DIR}")
print(f"DATA_DIR  = {DATA_DIR}")
print(f"H5AD_PATH = {H5AD_PATH}")
```

```
scanpy 1.11.5
NB_DIR    = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
DATA_DIR  = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data
H5AD_PATH = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/CTCL_all_final_portal_tags.h5ad

/tmp/ipykernel_1814137/481777811.py:9: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('scanpy')` instead
  print("scanpy", sc.__version__)
```

## 2. Load + first-look inspection

```python  # cell 3
adata = sc.read_h5ad(H5AD_PATH)
print(adata)
print("\nlayers:", list(adata.layers.keys()))
print("obsm:  ", list(adata.obsm.keys()))
print("uns:   ", list(adata.uns.keys())[:20])
```

```
AnnData object with n_obs × n_vars = 419579 × 15790
    obs: 'tissue', 'sex', 'study', 'donor', 'tech', 'cell_type', 'stage', 'n_counts'
    uns: '_scvi_manager_uuid', '_scvi_uuid', 'broad_ct_colors', 'cell_type_colors', 'donor_colors', 'groups1_colors', 'hvg', 'leiden', 'log1p', 'neighbors', 'tech_colors'
    obsm: 'X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs'
    layers: 'raw_counts'
    obsp: 'connectivities', 'distances'

layers: ['raw_counts']
obsm:   ['X_scVI', 'X_scVI_MDE', '_scvi_extra_categorical_covs']
uns:    ['_scvi_manager_uuid', '_scvi_uuid', 'broad_ct_colors', 'cell_type_colors', 'donor_colors', 'groups1_colors', 'hvg', 'leiden', 'log1p', 'neighbors', 'tech_colors']
```

```python  # cell 4
print("obs columns:")
for c in adata.obs.columns:
    n = adata.obs[c].nunique(dropna=True)
    print(f"  {c:30s}  dtype={str(adata.obs[c].dtype):20s}  nunique={n}")
```

```
obs columns:
  tissue                          dtype=category              nunique=3
  sex                             dtype=category              nunique=2
  study                           dtype=category              nunique=4
  donor                           dtype=category              nunique=36
  tech                            dtype=category              nunique=2
  cell_type                       dtype=category              nunique=49
  stage                           dtype=category              nunique=7
  n_counts                        dtype=float32               nunique=20314
```

```python  # cell 5
def first_present(cols, options):
    return next((c for c in options if c in cols), None)

OBS = adata.obs.columns
COL_DISEASE = first_present(OBS, ["disease", "Disease", "condition", "Condition"])
COL_STAGE   = first_present(OBS, ["stage", "Stage", "disease_stage", "clinical_stage"])
COL_DONOR   = first_present(OBS, ["donor", "donor_id", "patient", "patient_id", "sample", "sample_id"])
COL_CT      = first_present(OBS, ["cell_type", "celltype", "annotation", "broad_celltype", "cell_type_fine"])
print(dict(disease=COL_DISEASE, stage=COL_STAGE, donor=COL_DONOR, celltype=COL_CT))

for label, col in [("disease", COL_DISEASE), ("stage", COL_STAGE), ("celltype", COL_CT)]:
    if col is None: continue
    print(f"\n=== {label} ({col}) ===")
    print(adata.obs[col].value_counts(dropna=False).head(20))
```

```
{'disease': None, 'stage': 'stage', 'donor': 'donor', 'celltype': 'cell_type'}

=== stage (stage) ===
stage
IA      138572
IIB     132153
IB       99981
IIIB     19639
IV       17508
IIA       7438
IVA2      4288
Name: count, dtype: int64

=== celltype (cell_type) ===
cell_type
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
Name: count, dtype: int64
```

```python  # cell 6
for c in ['tissue','study', 'tech', 'donor']:
    if c in OBS:
        print(f"\n=== {c} ===")
        print(adata.obs[c].value_counts(dropna=False).head(20))
```

```
=== tissue ===
tissue
Dermis        193063
whole_skin    142421
Epidermis      84095
Name: count, dtype: int64

=== study ===
study
Sanger_Ncl_Fresh    277158
PKU                  60164
Sanger_Ncl_FFPE      49634
MDA                  32623
Name: count, dtype: int64

=== tech ===
tech
10x     369945
flex     49634
Name: count, dtype: int64

=== donor ===
donor
CTCL7     77712
CTCL5     41201
CTCL6     37073
CTCL8     34108
CTCL3     26623
CTCL4     25187
CTCL1     21684
MF28      19639
CTCL13    16744
MF30      14477
CTCL2     13570
CTCL18    10082
PT56       8826
CTCL12     7426
MF17       6768
PT35       6522
CTCL9      5984
MF14       5314
PT50       5161
PT11       4288
Name: count, dtype: int64
```

```python  # cell 7
pd.crosstab(adata.obs['study'], adata.obs['tech'], dropna=False)
```

```
tech                 10x   flex
study                          
MDA                32623      0
PKU                60164      0
Sanger_Ncl_FFPE        0  49634
Sanger_Ncl_Fresh  277158      0
```

## 4. UMAP overview

Prefer the existing integrated embedding (`X_scVI` / `X_harmony` / `X_umap`). Recompute only if missing — full integration on 420 k cells is expensive.

```python  # cell 9
if "X_umap" in adata.obsm:
    print("Using published X_umap")
else:
    int_keys = [k for k in adata.obsm if any(s in k.lower() for s in ["scvi", "harmony", "mnn", "scanorama", "integrated"])]
    rep = int_keys[0] if int_keys else None
    print(f"Recomputing neighbors+UMAP from rep={rep or 'X (raw PCA)'}")
    if rep is None:
        sc.pp.pca(adata, n_comps=50)
        rep = "X_pca"
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=30)
    sc.tl.umap(adata, min_dist=0.3)

umap_colors = [c for c in [COL_DISEASE, COL_STAGE, COL_CT] if c is not None]
sc.pl.umap(adata, color=umap_colors, wspace=0.4, save="_overview.png", show=True)
```

```
Recomputing neighbors+UMAP from rep=X_scVI

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

WARNING: saving figure to file /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/umap_overview.png

<Figure size 1120x240 with 2 Axes>
```

## 5. B-cell enrichment: CTCL vs healthy / AD / psoriasis (Fig. 7)

Per-donor B-cell fraction across disease groups + Wilcoxon CTCL vs each control.

```python  # cell 11
from scipy.stats import mannwhitneyu

assert COL_CT and COL_DONOR and COL_DISEASE, "need celltype + donor + disease columns"

is_b = adata.obs[COL_CT].astype(str).str.contains("B[ _]?cell|Bcell|plasma", case=False, regex=True)
adata.obs["_is_B"] = is_b
b_frac = (adata.obs.groupby([COL_DONOR, COL_DISEASE], observed=True)["_is_B"]
          .mean().rename("b_frac").reset_index())
print("B-cell fraction per disease group:")
print(b_frac.groupby(COL_DISEASE)["b_frac"].describe())

ctcl_mask = b_frac[COL_DISEASE].astype(str).str.contains("CTCL|MF", case=False)
ctcl = b_frac.loc[ctcl_mask, "b_frac"].values
print(f"\nCTCL donors: n={len(ctcl)}, median b_frac={np.median(ctcl):.3f}")
print("\nWilcoxon CTCL > group:")
for g in b_frac.loc[~ctcl_mask, COL_DISEASE].unique():
    sub = b_frac.loc[b_frac[COL_DISEASE] == g, "b_frac"].values
    if len(sub) < 3: continue
    u, p = mannwhitneyu(ctcl, sub, alternative="greater")
    print(f"  CTCL vs {g!s:30s}  U={u:7.0f}  p={p:.2e}  n_ref={len(sub)}")

fig, ax = plt.subplots(figsize=(6, 3))
b_frac.boxplot(column="b_frac", by=COL_DISEASE, ax=ax, rot=45, grid=False)
ax.set_title("B-cell fraction per donor"); plt.suptitle("")
ax.set_ylabel("B-cell fraction")
fig.tight_layout(); fig.savefig(FIG_DIR / "bcell_fraction.png", dpi=150)
plt.show()
b_frac.to_csv(FIG_DIR / "bcell_fraction.csv", index=False)
```

```
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[11], line 3
      1 from scipy.stats import mannwhitneyu
----> 3 assert COL_CT and COL_DONOR and COL_DISEASE, "need celltype + donor + disease columns"
      5 is_b = adata.obs[COL_CT].astype(str).str.contains("B[ _]?cell|Bcell|plasma", case=False, regex=True)
      6 adata.obs["_is_B"] = is_b

AssertionError: need celltype + donor + disease columns
```

## 6. TH1 / TH2 / TH17 scoring in T cells, stratified by stage (Fig. 3)

Score gene programs on log-normalised counts. If the object only has scVI latents, fall back to recomputing log-norm in a copy. This applies the test on *all* T cells — for the malignant-only test (paper's p ≈ 3×10⁻⁴) you need inferCNV + TCR labels; see the follow-up notebook.

```python  # cell 13
from scipy.stats import mannwhitneyu

t_mask = adata.obs[COL_CT].astype(str).str.contains("T[ _]?cell|Tc|Th|Treg", case=False, regex=True)
adata_t = adata[t_mask].copy()
print(f"T-cell subset: {adata_t.shape}")

needs_lognorm = adata_t.X.max() > 50 if hasattr(adata_t.X, "max") else True
if needs_lognorm and "counts" in adata_t.layers:
    adata_t.X = adata_t.layers["counts"].copy()
    sc.pp.normalize_total(adata_t, target_sum=1e4)
    sc.pp.log1p(adata_t)

PROGRAMS = {
    "Th1":  ["TBX21", "IFNG", "CXCR3"],
    "Th2":  ["GATA3", "IL4", "IL13", "IL5", "CCR4"],
    "Th17": ["RORC", "IL17A", "IL17F", "CCR6"],
}
for name, genes in PROGRAMS.items():
    present = [g for g in genes if g in adata_t.var_names]
    missing = set(genes) - set(present)
    if missing: print(f"  {name}: missing {missing}")
    sc.tl.score_genes(adata_t, present, score_name=f"{name}_score")

score_cols = [f"{n}_score" for n in PROGRAMS]
adata_t.obs["Th_subtype"] = adata_t.obs[score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)

per_donor = (adata_t.obs.groupby([COL_DONOR, COL_STAGE or COL_DISEASE], observed=True)["Th_subtype"]
             .value_counts(normalize=True).unstack(fill_value=0).reset_index())
print(per_donor.head())
per_donor.to_csv(FIG_DIR / "th_subtype_fractions.csv", index=False)

stage_col = COL_STAGE or COL_DISEASE
if stage_col in per_donor and "Th2" in per_donor:
    early = per_donor.loc[per_donor[stage_col].astype(str).str.contains("early|<IIB|I-IIA|IA|IB|IIA", case=False), "Th2"]
    adv   = per_donor.loc[per_donor[stage_col].astype(str).str.contains("advanced|IIB|III|IV", case=False), "Th2"]
    print(f"\nTh2 fraction: early n={len(early)}, advanced n={len(adv)}")
    if len(early) >= 3 and len(adv) >= 3:
        u, p = mannwhitneyu(early, adv, alternative="less")
        print(f"Wilcoxon early < advanced: U={u:.0f}  p={p:.2e}")

fig, ax = plt.subplots(figsize=(6, 3))
per_donor.boxplot(column=[c for c in ["Th1", "Th2", "Th17"] if c in per_donor],
                  by=stage_col, ax=ax, rot=45, grid=False)
ax.set_title("Th subtype fractions by stage"); plt.suptitle("")
fig.tight_layout(); fig.savefig(FIG_DIR / "th_subtype_by_stage.png", dpi=150)
plt.show()
```
