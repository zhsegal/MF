# old/37_semantic_clonality_transfer.prev.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/37_semantic_clonality_transfer.prev.ipynb` and is never run._

# 37 — Transferring TCR clonality to TCR-less skin samples via the SemanticSCVI latent

TCR exists for **36 of the 82 skin donors** (nb30). The rest — 21 li2024 + 8 gaydosik2019 donors —
have no clonality data *and* no published malignant label, so they are dark. This notebook asks
whether the malignant/benign split can be **transferred** into them through the semantic latent.

The mechanism is the δ-arithmetic from `notebooks/perturbation_trial.ipynb`, where a held-out
unit's latent state is recovered as `z_ctrl + δ` with `δ = mean(z|stim) − mean(z|ctrl)` fitted on
training units. Here:

$$\delta_d = \mathrm{mean}(z \mid \text{ALICE-clonal cells of } d) - \mathrm{mean}(z \mid \text{benign anchors of } d),
\qquad \hat\delta = \mathrm{mean}_d\, \delta_d$$

Per-donor differencing cancels donor offsets. For a donor with no TCR we do not know its benign
centroid, so it is **estimated** from that donor's own cells (2-component GMM on the $\hat\delta$
projection — the per-donor trick nb30 uses on `cnv_score`), and the clonal centroid is
**reconstructed** as $\hat\mu_{mal} = \hat\mu_{ben} + \hat\delta$. Each cell goes to the nearer
centroid.

**Validation is the point of this notebook.** Repeated **leave-3-samples-out** over the labelled
donors: each fold retrains SemanticSCVI with the 3 held-out donors' cells fully removed
(`perturbation_trial`'s protocol), encodes everything with that fold's model, and scores the
transfer on the 3 unseen donors. The applied call on the dark donors comes last and is explicitly
**not** validated against external truth — they have none.

> **HEAVY** cells are marked. The object is ~20 GB and the fold training is 11 GPU runs — run
> those on the GPU/compute kernel (`neural_nmf_env`) or via `jobs/run_clonality_folds.sh`, never
> on the login node.

## Part 0 — parameters

```python  # cell 2
# ============================================================
# Parameters — cohort filters, SemanticSCVI knobs (nb18 recipe), fold layout.
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
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
print("NB_DIR =", NB_DIR)

# ---- inputs ----
OBJ            = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_annotated_v3.h5ad"   # 82 donors, nb30
MALIG          = NB_DIR / "data" / "atlas_joint" / "skin_T_malignancy_v3.parquet"   # nb30 per-cell calls
GENE_ID_SOURCE = NB_DIR / "data" / "cache" / "cnmf_malignant_counts.h5ad"           # gene_name -> Ensembl
SEMANTIC_CACHE = NB_DIR / "data" / "mf_clonality_geneformer.pt"                     # full-gene Geneformer map

# ---- outputs ----
OUT_DIR   = NB_DIR / "benchmark_results" / "clonality_transfer"
JOB_INPUT = OUT_DIR / "_job_input"
FIG_DIR   = NB_DIR / "figures"
CONFIG    = NB_DIR / "jobs" / "clonality_folds_config.json"
MODEL_CACHE = NB_DIR / "models" / ".model_cache_semantic_clonality"
FOLD_PARQUET  = NB_DIR / "data" / "atlas_joint" / "semantic_clonality_folds_v1.parquet"
TRANS_PARQUET = NB_DIR / "data" / "atlas_joint" / "semantic_clonality_transfer_v1.parquet"
for d in (OUT_DIR, JOB_INPUT, FIG_DIR, MODEL_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---- cohort (nb18's rules) ----
CD4_TYPE      = "CD4"                                    # cell_type_T; excludes CD4_Treg / CD8 / NK
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}  # TCRb caller is blind to these
DROP_DONORS   = {"D1__P303"}                             # duplicate of D5__MFIVB
MIN_TRB_FRAC  = 0.0                                      # no extra recovery gate; nb30 already filtered

# ---- preprocessing / model (nb18 recipe; epochs halved, one run per fold + reference) ----
HVG_TOP_N, HVG_FLAVOR = 2500, "seurat_v3"
N_LATENT   = 10
BATCH_KEY  = "study"        # NOT sample_id: a held-out donor's sample_id would be an unseen batch
LABELS_KEY = "cell_type_T"
MAX_EPOCHS, WARMUP_EPOCHS, KL_WARMUP = 100, 20, 100
SEMANTIC_KWARGS = dict(
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

# ---- folds ----
SEED               = 0
FOLD_SIZE          = 3       # leave-3-samples-out
TRAIN_FRAC         = 1 / 3   # per-donor training subsample (nb19 sweep used the same)
TRAIN_MIN_CELLS    = 500     # never subsample a donor below this
EVAL_MIN_CELLS     = 200     # holdout eligibility: per-donor metrics need enough of both classes
EVAL_MIN_CLONAL    = 25
EVAL_MIN_ANCHOR    = 25


def _cache_slug(n=10):
    """Stable hash of every param that affects a trained model (shared by all folds)."""
    blob = json.dumps({"kwargs": dict(sorted(SEMANTIC_KWARGS.items())),
                       "max_epochs": MAX_EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
                       "n_epochs_kl_warmup": KL_WARMUP, "hvg": HVG_TOP_N,
                       "batch_key": BATCH_KEY, "labels_key": LABELS_KEY},
                      default=str, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


PARAM_SLUG = _cache_slug()
print("param slug:", PARAM_SLUG, "| model cache:", MODEL_CACHE / PARAM_SLUG)
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
param slug: 38c5ae855b | model cache: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_clonality/38c5ae855b
```

```python  # cell 3
import gc
import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

for _p in (str(NB_DIR), str(NB_DIR.parent)):     # MF/ helpers, then notebooks/ helpers
    if _p not in sys.path:
        sys.path.insert(0, _p)

import semantic_clonality_helpers as SC
import semantic_malig_helpers as M

importlib.reload(M)
importlib.reload(SC)

np.random.seed(SEED)
sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 120
```

## Part 1 — cohort + labels · HEAVY (compute kernel)

Read the nb30 TCR object **backed**, keep only the CD4 non-Treg cells of the included donors, then
pull that subset into memory (the full file is ~20 GB). Join nb30's per-cell calls and build the two
label columns:

- `is_clonal` — ALICE malignant (`tcr_malignant_alice`), the positives.
- `is_anchor` — `has_tcr & ~ALICE & ~tcr_is_expanded`, the benign negatives (nb16/17's anchor rule).

Cells that are neither (TCR-negative cells of TCR-covered donors, and every cell of a dark donor)
have unknown status and are **excluded from scoring** — they still train the encoder.

```python  # cell 5
# Backed read -> mask -> to_memory: never materialise the whole 20 GB object.  HEAVY
ad = sc.read_h5ad(OBJ, backed="r")
obs_all = ad.obs

keep = (obs_all["cell_type_T"].astype(str) == CD4_TYPE).to_numpy()
keep &= ~obs_all["entity"].astype(str).isin(DROP_ENTITIES).to_numpy()
keep &= ~obs_all["donor"].astype(str).isin(DROP_DONORS).to_numpy()
print(f"CD4 non-Treg of included donors: {int(keep.sum())}/{ad.n_obs} cells, "
      f"{obs_all.loc[keep, 'donor'].nunique()} donors")

adata = ad[keep].to_memory()
del ad
gc.collect()
adata.X = adata.layers["raw_counts"].copy()          # NB likelihood wants raw counts
for lay in list(adata.layers):
    del adata.layers[lay]
gc.collect()
print(adata)
```

```
CD4 non-Treg of included donors: 287175/395233 cells, 70 donors
AnnData object with n_obs × n_vars = 287175 × 40821
    obs: 'cell_id', 'dataset', 'study', 'donor', 'real_donor', 'sample_id', 'sex', 'disease', 'disease_stage', 'compartment', 'tissue', 'tech', 'cell_type_orig', 'cell_type', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'predicted_doublet', 'has_tcr', 'clone_id', 'tra_cdr3', 'trb_cdr3', 'clone_size', 'is_expanded', 'is_dominant_clone', 'is_malignant', 'repo', 'accession', 'organ', 'tissue_detail', 'malignant_call_method', 'tcr_available', 'malignant_labeled', 'malignant_frac', 'tcr_recovery', 'stage_class', 'stage_clean', 'lineage', 'entity', 'fmf_status', 'lineage_resolve', 'treatment_context', 'blood_involvement', 'lesion_type', 'cohort_country', 'cohort_region', 'origin_resolve', 'east_asian', 'lineage_orig', 'lineage_score_cd4', 'lineage_score_cd8', 'lineage_score_gd', 'lineage_score_ab', 'lineage_method', 'lineage_disagree', 'lineage_disagreement', 'leiden_broad', '_sig_T', '_sig_B', '_sig_Plasma', '_sig_Myeloid', '_sig_Mast', '_sig_Keratinocyte', 'cell_type_broad', 'leiden_T', 'cell_type_T', '_li_clone_key'
    uns: 'adt_panel', 'cell_type_T_colors', 'cell_type_broad_colors', 'cell_type_colors', 'dendrogram_leiden_T', 'dendrogram_leiden_broad', 'disease_colors', 'leiden_T', 'leiden_T_colors', 'leiden_broad', 'leiden_broad_colors', 'log1p', 'neighbors', 'pca', 'sample_id_colors', 'study_colors', 'umap'
    obsm: 'X_adt', 'X_mrvi_u', 'X_mrvi_z', 'X_pca', 'X_scVI', 'X_scVI_MDE', 'X_umap', '_scvi_extra_categorical_covs'
    varm: 'PCs'
    obsp: 'connectivities', 'distances'
```

```python  # cell 6
# nb30 per-cell TCR/CNV calls -> the two label columns.
mal = pd.read_parquet(MALIG)
for c in ["tcr_malignant_alice", "has_tcr", "tcr_is_expanded", "cnv_malig_cluster"]:
    v = mal[c].reindex(adata.obs_names)
    adata.obs[c] = v.astype("boolean").fillna(False).to_numpy(dtype=bool)
adata.obs["cnv_cell_score"] = mal["cnv_cell_score"].reindex(adata.obs_names).to_numpy(float)

adata.obs["is_clonal"] = adata.obs["tcr_malignant_alice"].to_numpy()
adata.obs["is_anchor"] = (adata.obs["has_tcr"].to_numpy()
                          & ~adata.obs["is_clonal"].to_numpy()
                          & ~adata.obs["tcr_is_expanded"].to_numpy())
# HC donors are benign by definition (nb18 does the same)
hc = adata.obs["disease"].astype(str).eq("HC").to_numpy()
n_flip = int(adata.obs["is_clonal"].to_numpy()[hc].sum())
adata.obs.loc[hc, "is_clonal"] = False
adata.obs.loc[hc & adata.obs["has_tcr"].to_numpy(), "is_anchor"] = True

print(f"clonal {int(adata.obs.is_clonal.sum()):,} | anchor {int(adata.obs.is_anchor.sum()):,} | "
      f"unlabelled {int((~adata.obs.is_clonal & ~adata.obs.is_anchor).sum()):,} "
      f"(HC cells forced benign: {n_flip})")
```

```
clonal 76,349 | anchor 35,573 | unlabelled 175,253 (HC cells forced benign: 0)
```

```python  # cell 7
# Donor roles: labelled (delta + eval) / no_clone (training + negative control) / dark (transfer target).
donor_tbl = SC.donor_label_table(adata.obs, eval_min_cells=EVAL_MIN_CELLS,
                                 eval_min_clonal=EVAL_MIN_CLONAL, eval_min_anchor=EVAL_MIN_ANCHOR)
print(donor_tbl.groupby("group", observed=True)
      .agg(donors=("n_cells", "size"), cells=("n_cells", "sum"),
           eligible=("eligible", "sum")).to_string())
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(donor_tbl.to_string())
donor_tbl.to_csv(OUT_DIR / "donor_table.csv")
```

```
donors   cells  eligible
group                                   
dark                34   78961         0
labelled            30  197932        27
labelled_small       1      72         0
no_clone             5   10210         0
                      n_cells  n_clonal  n_anchor  n_tcr            study     disease           group  clonal_frac  eligible
donor                                                                                                                       
D3__P87                   494         0         0      0      brunner2024  CTCL_other            dark       0.0000     False
D1__P112                  255         0         0      0  chennareddy2025          HC            dark       0.0000     False
D1__P121                   78         0         0      0  chennareddy2025          HC            dark       0.0000     False
D6__MF2                  1457         0         0      0     gaydosik2019          MF            dark       0.0000     False
D6__MF5                  6973         0         0      0     gaydosik2019          MF            dark       0.0000     False
D6__NOR1                   54         0         0      0     gaydosik2019          HC            dark       0.0000     False
D6__NOR2                   57         0         0      0     gaydosik2019          HC            dark       0.0000     False
D6__NOR3                   92         0         0      0     gaydosik2019          HC            dark       0.0000     False
D6__NOR4                   63         0         0      0     gaydosik2019          HC            dark       0.0000     False
D6__SC157                1456         0         0      0     gaydosik2019          MF            dark       0.0000     False
D6__SC158                 461         0         0      0     gaydosik2019          MF            dark       0.0000     False
D6__SC205                1287         0         0      0     gaydosik2019          MF            dark       0.0000     False
Li2024_atlas__CTCL9      2575         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL10     1373         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL11      102         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL12     3436         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL13     9488         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL14      394         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL15      136         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL16     1893         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL17      341         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__CTCL18     8122         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF14       4301         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF15       1248         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF17       5143         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF18        555         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF21       2581         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF22        937         0         0      0           li2024          MF            dark       0.0000     False
Li2024_atlas__MF26  
... [5229 more chars truncated]
```

## Part 2 — Geneformer map, HVG, training pool, slim job input · HEAVY

nb18's preprocessing exactly: map symbols → Ensembl, build the **full-gene** Geneformer map (cached),
drop out-of-vocab genes *before* HVG so HVG ranks variance only among semantically-grounded genes,
then HVG-subset adata and the map together.

`train_pool` marks the per-donor training subsample (≈1/3, floor `TRAIN_MIN_CELLS`). Held-out
donors are removed per fold *on top of* this pool, and **encoding always uses every cell** — the
subsample only shrinks what the encoder is fit on.

```python  # cell 9
# gene_name -> Ensembl (Geneformer's vocabulary is keyed by Ensembl id).
src = sc.read_h5ad(GENE_ID_SOURCE, backed="r")
sym2ens = dict(zip(src.var["gene_name"].astype(str), src.var["gene_id"].astype(str)))
del src
gc.collect()

ribo = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
print(f"dropping {int(ribo.sum())} ribosomal protein genes")
adata = adata[:, ~ribo].copy()
adata.var["gene_id"] = [sym2ens.get(s, s) for s in adata.var_names.astype(str)]
adata.var["feature_name"] = adata.var_names.astype(str)
print(f"gene_id mapped to Ensembl: {int(sum(g.startswith('ENSG') for g in adata.var['gene_id']))}"
      f"/{adata.n_vars}")
```

```
dropping 105 ribosomal protein genes
gene_id mapped to Ensembl: 9766/40716
```

```python  # cell 10
import torch

from benchmark_helpers import get_or_build_geneformer_map

# Geneformer's vocabulary is 20,275 Ensembl tokens, so on this FULL-gene panel (~40.7k, most of it
# antisense/lncRNA/novel loci) coverage is capped at vocab/n_vars ~ 0.50 — the builder's default
# min_coverage=0.5 can never pass and is the wrong guard here. Disable it and assert instead on the
# quantity that actually detects a key mismatch: how much of the vocabulary we hit. A wrong id/symbol
# column collapses that to a few hundred genes; a healthy run recovers most of the 20,275.
# symbol_key gives the Ensembl lookup a real symbol to fall back on rather than reusing the id.
GF_MIN_IN_VOCAB = 15_000
GF_KW = dict(var_id_key="gene_id", symbol_key="feature_name", min_coverage=0.0)

semantic_map = get_or_build_geneformer_map(adata, SEMANTIC_CACHE, **GF_KW)
if semantic_map.shape[0] != adata.n_vars:            # stale-cache guard (nb18)
    print(f"map rows {semantic_map.shape[0]} != n_vars {adata.n_vars} — rebuilding")
    SEMANTIC_CACHE.unlink()
    semantic_map = get_or_build_geneformer_map(adata, SEMANTIC_CACHE, **GF_KW)

in_vocab = (semantic_map.norm(dim=1) > 0).cpu().numpy()
print(f"in-vocab (non-zero Geneformer row): {int(in_vocab.sum())}/{adata.n_vars}")
assert int(in_vocab.sum()) >= GF_MIN_IN_VOCAB, (
    f"only {int(in_vocab.sum())} genes hit Geneformer's vocabulary — adata.var ids/symbols "
    "probably do not match it; a near-empty map silently disables the semantic prior")
adata = adata[:, in_vocab].copy()
semantic_map = semantic_map[torch.as_tensor(in_vocab)]

sc.pp.highly_variable_genes(adata, n_top_genes=HVG_TOP_N, flavor=HVG_FLAVOR, subset=False)
hv = adata.var["highly_variable"].to_numpy()
adata = adata[:, hv].copy()
semantic_map = semantic_map[torch.as_tensor(hv)]
print("after HVG:", adata.shape, "| semantic_map", tuple(semantic_map.shape))
assert semantic_map.shape[0] == adata.n_vars
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

Loading cached Geneformer map from /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/mf_clonality_geneformer.pt
in-vocab (non-zero Geneformer row): 18053/40716
after HVG: (287175, 2500) | semantic_map (2500, 1152)
```

```python  # cell 11
# Per-donor training subsample (encoding always uses all cells; this only shrinks the fit set).
rng = np.random.default_rng(SEED)
pool = np.zeros(adata.n_obs, dtype=bool)
donors = adata.obs["donor"].astype(str).to_numpy()
for d in np.unique(donors):
    idx = np.where(donors == d)[0]
    n = int(min(len(idx), max(TRAIN_MIN_CELLS, round(TRAIN_FRAC * len(idx)))))
    pool[rng.choice(idx, n, replace=False)] = True
adata.obs["train_pool"] = pool
print(f"train_pool {int(pool.sum()):,}/{adata.n_obs:,} cells "
      f"({pool.mean():.1%}) across {len(np.unique(donors))} donors")
```

```
train_pool 100,367/287,175 cells (34.9%) across 70 donors
```

```python  # cell 12
# Write the slim job input. Keeps X_mrvi_u / X_scVI so the baselines and the CPU dry run are free.
slim = sc.AnnData(
    X=adata.X.copy(),
    obs=adata.obs[["donor", "sample_id", "study", "disease", "entity", "cell_type_T",
                   "has_tcr", "tcr_is_expanded", "tcr_malignant_alice",
                   "cnv_malig_cluster", "cnv_cell_score",
                   "is_clonal", "is_anchor", "train_pool"]].copy(),
    var=adata.var[["gene_id", "feature_name", "highly_variable"]].copy(),
)
for k in ("X_mrvi_u", "X_scVI"):
    if k in adata.obsm:
        slim.obsm[k] = np.asarray(adata.obsm[k], dtype=np.float32)
for c in ("study", "cell_type_T"):
    slim.obs[c] = slim.obs[c].astype(str).astype("category")

INPUT_H5AD = JOB_INPUT / "clonality_input.h5ad"
MAP_PT = JOB_INPUT / "clonality_semantic_map.pt"
slim.write_h5ad(INPUT_H5AD)
torch.save(semantic_map, MAP_PT)
print("wrote", INPUT_H5AD, slim.shape, "| obsm:", list(slim.obsm))
print("wrote", MAP_PT, tuple(semantic_map.shape))
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/clonality_transfer/_job_input/clonality_input.h5ad (287175, 2500) | obsm: ['X_mrvi_u', 'X_scVI']
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/clonality_transfer/_job_input/clonality_semantic_map.pt (2500, 1152)
```

## Part 3 — folds + job config

`SC.make_triples` deals the eligible labelled donors largest-first into the currently smallest fold,
under the hard constraint that **no fold may hold out every donor of a `study`** — a held-out donor
whose batch category vanished from training has no trained batch embedding and could not be encoded.
`SC.check_batch_coverage` re-asserts this on the actual cells, and the job asserts it again before
training.

`runs` = one entry per fold plus `all` (no holdout) — the leakage reference and the model used for
the applied call.

```python  # cell 14
folds = SC.make_triples(donor_tbl, size=FOLD_SIZE, seed=SEED, batch_key=BATCH_KEY)
SC.check_batch_coverage(slim.obs, folds, batch_key=BATCH_KEY)
print(f"{len(folds)} folds of {sorted({len(f) for f in folds})} | "
      f"{len({d for f in folds for d in f})} donors tested once")
print(f"submit with:  cd {NB_DIR / 'jobs'} && ARRAY=1 N_FOLDS={len(folds)} ./run_clonality_folds.sh")
for k, f in enumerate(folds):
    print(f"  fold{k}: " + ", ".join(f"{d} ({donor_tbl.loc[d, 'n_clonal']}c/"
                                     f"{donor_tbl.loc[d, 'n_anchor']}a)" for d in f))

# δ̂ is fitted on every clone-bearing donor (fit_delta skips any that lack one of the classes);
# `eligible` — a strictly smaller set — governs who may be *held out*.
LABELLED = list(donor_tbl.index[donor_tbl["group"].isin(["labelled", "labelled_small"])])
DARK = list(donor_tbl.index[donor_tbl["group"] == "dark"])
NO_CLONE = list(donor_tbl.index[donor_tbl["group"] == "no_clone"])
print(f"\nδ fitted on {len(LABELLED)} clone-bearing donors "
      f"({int(donor_tbl['eligible'].sum())} of them holdout-eligible) | "
      f"no_clone {len(NO_CLONE)} | dark {len(DARK)}")
```

```
make_triples: training-only (sole donor of its study): D5__MFIVB [rindler2021_fi]
8 folds of [3, 4] | 26 donors tested once
submit with:  cd /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs && ARRAY=1 N_FOLDS=8 ./run_clonality_folds.sh
  fold0: Li2024_atlas__PT11 (1619c/141a), Li2024_atlas__CTCL3 (8162c/1174a), B5__PT23 (3702c/164a), H__SS4 (178c/26a)
  fold1: D3__P105 (65c/266a), Li2024_atlas__PT53 (1240c/657a), Li2024_atlas__CTCL7 (3122c/7347a), Li2024_atlas__CTCL1 (7090c/160a)
  fold2: Li2024_atlas__CTCL5 (6604c/2839a), Li2024_atlas__CTCL6 (3119c/4889a), Li2024_atlas__PT56 (1156c/1404a)
  fold3: D3__P192 (3028c/179a), Li2024_atlas__CTCL4 (743c/4111a), Li2024_atlas__CTCL8 (5314c/1238a)
  fold4: B5__PT09 (5253c/1289a), D1__P312 (724c/518a), Li2024_atlas__PT35 (2865c/46a)
  fold5: D1__P171 (5235c/107a), D3__P196 (434c/88a), D1__P204 (2343c/1965a)
  fold6: D1__P76 (4599c/1121a), Li2024_atlas__PT50 (413c/401a), D1__PGS (2259c/137a)
  fold7: Li2024_atlas__PT47 (257c/88a), Li2024_atlas__CTCL2 (1840c/1581a), D1__P311 (3815c/175a)

δ fitted on 31 clone-bearing donors (27 of them holdout-eligible) | no_clone 5 | dark 34
```

```python  # cell 15
cfg = {
    "input_h5ad": str(INPUT_H5AD),
    "semantic_map": str(MAP_PT),
    "model_cache_dir": str(MODEL_CACHE / PARAM_SLUG),
    "out_dir": str(OUT_DIR),
    "donor_key": "donor",
    "batch_key": BATCH_KEY,
    "labels_key": LABELS_KEY,
    "train_pool_key": "train_pool",
    "max_epochs": MAX_EPOCHS,
    "warmup_epochs": WARMUP_EPOCHS,
    "n_epochs_kl_warmup": KL_WARMUP,
    "semantic_kwargs": SEMANTIC_KWARGS,
    "runs": ([{"name": f"fold{k}", "cache_key": f"fold{k}", "held_out": list(f)}
              for k, f in enumerate(folds)]
             + [{"name": "all", "cache_key": "all", "held_out": []}]),
    "folds": folds,
    "notes": (f"nb37 clonality transfer: leave-{FOLD_SIZE}-out over {len(LABELLED)} labelled "
              f"donors ({len(folds)} folds) + all-donor reference. Geneformer geometric prior, "
              f"HVG={HVG_TOP_N}, n_latent={N_LATENT}, batch={BATCH_KEY}."),
}
CONFIG.write_text(json.dumps(cfg, indent=2))
print("wrote", CONFIG)
print(cfg["notes"])
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs/clonality_folds_config.json
nb37 clonality transfer: leave-3-out over 31 labelled donors (8 folds) + all-donor reference. Geneformer geometric prior, HVG=2500, n_latent=10, batch=study.
```

## Part 4 — dry run on the MrVI latent · CPU, no GPU

**Verification gate — run this before spending any GPU time.** `X_mrvi_u` is already in the object,
so the entire fold/δ/eval path can be exercised for free. What this checks: the folds are disjoint
and cover every eligible donor, labels and cell indices line up, all four rules produce metrics on
an *identical* cell set per donor, and the shuffled-label null collapses to chance.

It is also a real baseline: whatever the semantic latent scores later has to beat this.

```python  # cell 17
res_mrvi, pc_mrvi = SC.run_cv(np.asarray(slim.obsm["X_mrvi_u"], dtype=float), slim.obs, folds,
                              latent_name="mrvi_u", train_donors_all=LABELLED, seed=SEED)
print(SC.summarize(res_mrvi).to_string())

# same-cells fairness check (nb18): every rule scored on identical cells for a given donor
assert res_mrvi.groupby(["fold", "donor"])["n"].nunique().eq(1).all(), "rules disagree on the cell set"
print(f"\nOK: {res_mrvi.donor.nunique()} held-out donors, identical cell sets across rules")
```

```
n_donors  auc_mean  auc_sd  auc_raw_mean  auc_raw_sd  f1_mean  f1_sd  balanced_acc_mean  balanced_acc_sd  recall_mean  recall_sd  specificity_mean  specificity_sd  centroid_r2_mean  centroid_r2_sd
latent rule                                                                                                                                                                                                                     
mrvi_u logistic_z                 26     0.961   0.059         0.961       0.059    0.792  0.237              0.834            0.134        0.791      0.306             0.878           0.091             0.974           0.029
       delta_centroid             26     0.955   0.060         0.955       0.060    0.784  0.265              0.837            0.148        0.769      0.326             0.904           0.066             0.974           0.029
       delta_threshold            26     0.955   0.060         0.955       0.060    0.797  0.229              0.831            0.125        0.801      0.295             0.862           0.091             0.974           0.029
       delta_centroid_null        26     0.717   0.134         0.437       0.251    0.709  0.203              0.494            0.129        0.795      0.168             0.193           0.214             0.968           0.047

OK: 26 held-out donors, identical cell sets across rules
```

## Part 5 — per-fold SemanticSCVI training · HEAVY (bsub / GPU)

`len(folds) + 1` runs: one per fold plus the all-donor reference (the fold count falls out of
how many donors clear the eligibility gate — it is printed in Part 3, not fixed at 10). Each removes its held-out donors' cells from training
entirely, then encodes **all** cells (held-out and dark included) with that fold's model.

```bash
cd notebooks/MF/jobs
ARRAY=1 N_FOLDS=<n> ./run_clonality_folds.sh   # one GPU job per run, in parallel
# or sequentially in one job:  ./run_clonality_folds.sh
# resume a subset:             ./run_clonality_folds.sh --folds 3 7 all
```

Models cache under `models/.model_cache_semantic_clonality/<slug>/<run>` and latents land in
`benchmark_results/clonality_transfer/z_<run>.npy`, so re-runs are free. The next cell just waits
for the files.

```python  # cell 19
# Survive a kernel restart while the bsub job ran: everything below needs only the slim input.
if "slim" not in dir():
    slim = sc.read_h5ad(INPUT_H5AD)
    folds = json.loads(CONFIG.read_text())["folds"]
    donor_tbl = pd.read_csv(OUT_DIR / "donor_table.csv", index_col=0)
    LABELLED = list(donor_tbl.index[donor_tbl["group"].isin(["labelled", "labelled_small"])])
    DARK = list(donor_tbl.index[donor_tbl["group"] == "dark"])
    NO_CLONE = list(donor_tbl.index[donor_tbl["group"] == "no_clone"])
    print("reloaded slim input:", slim.shape)

# label arrays used by every cell below
donors_np = slim.obs["donor"].astype(str).to_numpy()
ic = slim.obs["is_clonal"].to_numpy(dtype=bool)
ia = slim.obs["is_anchor"].to_numpy(dtype=bool)

Z_PATHS = {k: OUT_DIR / f"z_fold{k}.npy" for k in range(len(folds))}
Z_PATHS["all"] = OUT_DIR / "z_all.npy"
missing = {k: p for k, p in Z_PATHS.items() if not p.exists()}
if missing:
    raise FileNotFoundError(
        f"{len(missing)}/{len(Z_PATHS)} latents missing: {sorted(map(str, missing))}\n"
        f"submit with:  cd {NB_DIR / 'jobs'} && "
        f"ARRAY=1 N_FOLDS={len(folds)} ./run_clonality_folds.sh")

z_by_fold = {k: np.load(Z_PATHS[k]) for k in range(len(folds))}
z_all = np.load(Z_PATHS["all"])
for k, z in z_by_fold.items():
    assert z.shape[0] == slim.n_obs, f"fold{k}: {z.shape} vs {slim.n_obs}"
print(f"loaded {len(z_by_fold)} fold latents + reference | shape {z_all.shape}")
```

```
loaded 8 fold latents + reference | shape (287175, 10)
```

## Part 6 — evaluate

Four rules on the semantic latent (per-fold models), plus the all-donor semantic latent as the
leakage reference, plus the two latents already in the object as baselines. Every arm uses the same
folds, donors, labels and cells.

- `delta_centroid` — the headline: nearer-centroid to $\hat\mu_{ben}$ / $\hat\mu_{ben}+\hat\delta$.
- `delta_threshold` — same axis, cut transferred from the training donors instead of the midpoint.
- `logistic_z` — supervised ceiling on donor-centred $z$.
- `delta_centroid_null` — labels shuffled within training donors; `auc_raw` should sit at 0.5.
  (`auc` is polarity-folded by `M.binary_scores`, which floors a random direction above 0.5 —
  read `auc_raw` for the null.)

```python  # cell 21
arms = [("semantic_perfold", z_by_fold), ("semantic_alldonor", z_all),
        ("mrvi_u", np.asarray(slim.obsm["X_mrvi_u"], dtype=float))]
if "X_scVI" in slim.obsm:
    arms.append(("scvi", np.asarray(slim.obsm["X_scVI"], dtype=float)))

res_parts, pc_parts = [], []
for name, z in arms:
    r, pc = SC.run_cv(z, slim.obs, folds, latent_name=name, train_donors_all=LABELLED, seed=SEED)
    res_parts.append(r)
    pc_parts.append(pc)
    print(f"[{name}] {len(r)} rows")
res = pd.concat(res_parts, ignore_index=True)
percell = pd.concat(pc_parts)

summary = SC.summarize(res)
with pd.option_context("display.width", 220, "display.max_columns", None):
    print(summary.to_string())
summary.to_csv(OUT_DIR / "transfer_summary.csv")
res.to_csv(OUT_DIR / "transfer_per_donor.csv", index=False)
```

```
[semantic_perfold] 104 rows
[semantic_alldonor] 104 rows
[mrvi_u] 104 rows
[scvi] 104 rows
                                       n_donors  auc_mean  auc_sd  auc_raw_mean  auc_raw_sd  f1_mean  f1_sd  balanced_acc_mean  balanced_acc_sd  recall_mean  recall_sd  specificity_mean  specificity_sd  centroid_r2_mean  centroid_r2_sd
latent            rule                                                                                                                                                                                                                     
mrvi_u            logistic_z                 26     0.961   0.059         0.961       0.059    0.792  0.237              0.834            0.134        0.791      0.306             0.878           0.091             0.974           0.029
                  delta_centroid             26     0.955   0.060         0.955       0.060    0.784  0.265              0.837            0.148        0.769      0.326             0.904           0.066             0.974           0.029
                  delta_threshold            26     0.955   0.060         0.955       0.060    0.797  0.229              0.831            0.125        0.801      0.295             0.862           0.091             0.974           0.029
semantic_alldonor logistic_z                 26     0.937   0.059         0.937       0.059    0.726  0.195              0.795            0.114        0.669      0.264             0.921           0.071             0.867           0.152
semantic_perfold  logistic_z                 26     0.931   0.078         0.931       0.078    0.800  0.151              0.813            0.084        0.791      0.225             0.835           0.107             0.790           0.199
                  delta_centroid             26     0.873   0.125         0.869       0.135    0.741  0.208              0.772            0.137        0.687      0.276             0.856           0.107             0.790           0.199
                  delta_threshold            26     0.873   0.125         0.869       0.135    0.822  0.154              0.760            0.109        0.858      0.171             0.662           0.180             0.790           0.199
                  delta_centroid_null        26     0.830   0.131         0.385       0.342    0.662  0.265              0.428            0.173        0.723      0.252             0.133           0.141             0.852           0.146
semantic_alldonor delta_threshold            26     0.782   0.146         0.773       0.163    0.805  0.177              0.641            0.112        0.932      0.060             0.350           0.221             0.867           0.152
                  delta_centroid             26     0.782   0.146         0.773       0.163    0.629  0.223              0.691            0.122        0.577      0.281             0.806           0.098             0.867           0.152
scvi              logistic_z                 26     0.755   0.243         0.755       0.243    0.850  0.168              0.653            0.157        0.992      0.025             0.314           0.325             0.643           0.282
                  delta_threshold            26     0.739   0.229         0.739       0.229    0.842  0.168              0.633            0.139        0.985      0.037             0.281           0.294             0.643           0.282
                  delta_centroid             26     0.739   0.229         0.739       0.229    0.463  0.445              0.678            0.179        0.498      0.479             0.857           0.178             0.643           0.282
semantic_alldonor delta_centroid_null        26     0.732   0.130         0.422       0.258    0.670  0.190              0.484            0.143        0.724      0.158             0.244           0.274             0.863           0.193
mrvi_u            delta_centroid_null        26     0.717   0.134         0.437       0.251    0.709  0.203              0.494       
... [338 more chars truncated]
```

```python  # cell 22
# The published CNV call on the same held-out donors/cells — the external bar to clear.
cnv_rows = []
for k, held in enumerate(folds):
    for h in held:
        m = (slim.obs["donor"].astype(str) == h).to_numpy()
        lab = m & (slim.obs.is_clonal | slim.obs.is_anchor).to_numpy()
        y = slim.obs["is_clonal"].to_numpy()[lab]
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        r = M.binary_scores(y, slim.obs["cnv_malig_cluster"].to_numpy()[lab],
                            score=slim.obs["cnv_cell_score"].to_numpy()[lab])
        r.update(latent="cnv_published", rule="cnv_malig_cluster", fold=k, donor=h,
                 study=str(slim.obs.loc[m, "study"].iloc[0]),
                 true_frac=float(y.mean()),
                 pred_frac=float(slim.obs["cnv_malig_cluster"].to_numpy()[lab].mean()),
                 auc_raw=SC._auc_raw(y, slim.obs["cnv_cell_score"].to_numpy()[lab]),
                 centroid_r2=np.nan)
        cnv_rows.append(r)
res_cnv = pd.DataFrame(cnv_rows)
print(f"CNV reference on {len(res_cnv)} held-out donors "
      f"(cells with a CNV score: {int(np.isfinite(slim.obs.cnv_cell_score).sum()):,})")
print(SC.summarize(res_cnv).to_string())
res_full = pd.concat([res, res_cnv], ignore_index=True)
```

```
CNV reference on 26 held-out donors (cells with a CNV score: 207,289)
                                 n_donors  auc_mean  auc_sd  auc_raw_mean  auc_raw_sd  f1_mean  f1_sd  balanced_acc_mean  balanced_acc_sd  recall_mean  recall_sd  specificity_mean  specificity_sd  centroid_r2_mean  centroid_r2_sd
latent        rule                                                                                                                                                                                                                   
cnv_published cnv_malig_cluster        26     0.758    0.13         0.751       0.143    0.761  0.236              0.696             0.14        0.774      0.244             0.618           0.361               NaN             NaN
```

## Part 7 — figures

```python  # cell 24
# Fig 1 — does the transfer work? AUROC per arm x rule, mean+-sd over held-out donors.
ORDER = [a[0] for a in arms] + ["cnv_published"]
RULE_ORDER = ["delta_centroid", "delta_threshold", "logistic_z", "delta_centroid_null",
              "cnv_malig_cluster"]
piv = (res_full.groupby(["latent", "rule"], observed=True)["auc"]
       .agg(["mean", "std"]).reindex(pd.MultiIndex.from_product([ORDER, RULE_ORDER])))
lbl = [f"{a}\n{r}" for a, r in piv.index if np.isfinite(piv.loc[(a, r), "mean"])]
vals = piv["mean"].dropna().to_numpy()
errs = piv["std"].reindex(piv["mean"].dropna().index).fillna(0).to_numpy()
col = ["#4c78a8" if l.startswith("semantic") else
       "#888" if l.startswith(("mrvi", "scvi")) else "#c0392b" for l in lbl]
best = float(np.nanmax(vals))

fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(lbl)), 3))
ax.bar(np.arange(len(lbl)), vals, yerr=errs, capsize=2.5, color=col, width=0.7)
ax.axhline(0.5, ls=":", lw=0.8, c="k")
ax.set_xticks(np.arange(len(lbl)))
ax.set_xticklabels(lbl, rotation=60, ha="right", fontsize=5.5)
ax.set_ylabel("AUROC vs ALICE (held-out donor)")
ax.set_ylim(0.4, 1.02)
ax.set_title(f"Clonality transfers to unseen samples (best AUROC {best:.2f})", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_auroc.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 1122x360 with 1 Axes>
```

```python  # cell 25
# Fig 2 — is the tumour BURDEN right? predicted vs true clonal fraction, one dot per held-out donor.
HEAD = "delta_centroid"
fig, ax = plt.subplots(figsize=(5, 3))
for name, c in zip(["semantic_perfold", "mrvi_u"], ["#4c78a8", "#888"]):
    d = res_full[(res_full.latent == name) & (res_full.rule == HEAD)]
    if not len(d):
        continue
    r2 = SC.r2_latent(d["pred_frac"], d["true_frac"])
    ax.scatter(d["true_frac"], d["pred_frac"], s=22, alpha=0.8, color=c,
               label=f"{name} (R²={r2:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("true ALICE clonal fraction")
ax.set_ylabel("predicted clonal fraction")
ax.set_title(f"Per-donor tumour burden, {HEAD} on held-out samples", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_fraction.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

```python  # cell 26
# Fig 3 — one fold's 3 held-out donors: where the reconstructed centroids land on the delta axis.
EX_FOLD = 0
train_ex = [d for d in LABELLED if d not in set(folds[EX_FOLD])]
delta_ex, per_donor_delta = SC.fit_delta(z_by_fold[EX_FOLD], donors_np, ic, ia, train_ex)

fig, axes = plt.subplots(1, len(folds[EX_FOLD]), figsize=(5, 3), sharey=True)
for ax, h in zip(np.atleast_1d(axes), folds[EX_FOLD]):
    m = donors_np == h
    mu_ben, mu_mal, s, thr = SC.donor_centroids(z_by_fold[EX_FOLD][m], delta_ex, seed=SEED)
    lab = (ic | ia)[m]
    ax.hist(s[lab & ~ic[m]], bins=40, alpha=0.6, color="#888", label="ALICE benign")
    ax.hist(s[lab & ic[m]], bins=40, alpha=0.6, color="#c0392b", label="ALICE clonal")
    ax.axvline(0, ls="-", lw=1, c="k")
    ax.axvline(thr, ls="--", lw=1, c="#4c78a8")
    ax.axvline(2 * thr, ls=":", lw=1, c="#4c78a8")
    ax.set_title(h, fontsize=7)
    ax.set_xlabel("projection on δ̂")
np.atleast_1d(axes)[0].set_ylabel("cells")
np.atleast_1d(axes)[0].legend(fontsize=5.5, frameon=False)
fig.suptitle("Reconstructed clonal centroid (dotted) splits unseen donors", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_exemplar.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 4 Axes>
```

```python  # cell 27
# Fig 4 — perturbation_trial Fig-3 analogue: predicted vs real clonal centroid, per latent dim.
fig, ax = plt.subplots(figsize=(5, 3))
pred, real = [], []
for k, held in enumerate(folds):
    z = z_by_fold[k]
    train = [d for d in LABELLED if d not in set(held)]
    dl, _ = SC.fit_delta(z, donors_np, ic, ia, train)
    for h in held:
        m = donors_np == h
        lab_h = m & (ic | ia)
        if not ic[lab_h].any():
            continue
        _, mu_mal, _, _ = SC.donor_centroids(z[m], dl, seed=SEED)
        pred.append(mu_mal)
        real.append(z[lab_h][ic[lab_h]].mean(0))
pred, real = np.asarray(pred).ravel(), np.asarray(real).ravel()
ax.scatter(real, pred, s=10, alpha=0.5, color="#4c78a8")
lo, hi = float(min(real.min(), pred.min())), float(max(real.max(), pred.max()))
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax.set_xlabel("real clonal centroid (per latent dim)")
ax.set_ylabel("reconstructed μ̂_ben + δ̂")
ax.set_title(f"Latent centroid recovery on unseen donors (R²={SC.r2_latent(pred, real):.2f})",
             fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_centroid_r2.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

## Part 8 — negative controls

The rule must *not* fire where there is nothing to find:

1. **`no_clone` donors** — have TCR and *zero* ALICE-clonal cells. Predicted clonal fraction
   should be low. (Donors with a clone too small to be a labelled unit are `labelled_small`, not
   `no_clone` — they carry a clone and would be the wrong thing to assert cleanliness on.)
2. **HC donors** — healthy skin, benign by definition.
3. **CD8 cells** — read straight off the full object; MF malignancy is CD4, so these should be
   near-zero. Scored with the all-donor model's δ̂ (they were never in the CD4 training set).

```python  # cell 29
delta_all, delta_tbl = SC.fit_delta(z_all, donors_np, ic, ia, LABELLED)
print("per-donor δ_d norms:", delta_tbl["norm"].describe().round(2).to_dict())

ctrl_donors = NO_CLONE + [d for d in donor_tbl.index
                          if str(donor_tbl.loc[d, "disease"]) == "HC" and d not in NO_CLONE]
_, ctrl_tbl = SC.apply_to_donors(z_all, slim.obs, ctrl_donors, delta_all, seed=SEED)
ctrl_tbl = ctrl_tbl.merge(donor_tbl[["group", "n_clonal", "n_anchor"]], left_on="donor",
                          right_index=True, how="left")
print("\nnegative-control donors (predicted clonal fraction should be low):")
print(ctrl_tbl.sort_values("pred_clonal_frac", ascending=False).round(3).to_string(index=False))

lab_frac = res_full[(res_full.latent == "semantic_alldonor")
                    & (res_full.rule == "delta_centroid")]["pred_frac"].mean()
print(f"\nmean predicted fraction — labelled held-out donors {lab_frac:.3f} "
      f"vs negative controls {ctrl_tbl.pred_clonal_frac.mean():.3f}")
```

```
per-donor δ_d norms: {'count': 31.0, 'mean': 1.14, 'std': 0.54, 'min': 0.32, '25%': 0.74, '50%': 1.09, '75%': 1.45, 'max': 2.55}
skip H__HC1: 11 cells

negative-control donors (predicted clonal fraction should be low):
   donor  n_cells  pred_clonal_frac           study    disease    group  n_clonal  n_anchor
D6__NOR3       92             0.478    gaydosik2019         HC     dark         0         0
B5__PT22     6380             0.431        buus2025         SS no_clone         0      1688
D6__NOR4       63             0.349    gaydosik2019         HC     dark         0         0
D6__NOR1       54             0.333    gaydosik2019         HC     dark         0         0
D6__NOR2       57             0.298    gaydosik2019         HC     dark         0         0
D1__P115      463             0.261 chennareddy2025         HC no_clone         0       327
D1__P112      255             0.259 chennareddy2025         HC     dark         0         0
D1__P116      451             0.239 chennareddy2025         HC no_clone         0       384
D1__P121       78             0.231 chennareddy2025         HC     dark         0         0
 D3__P94     2905             0.155     brunner2024 CTCL_other no_clone         0       861

mean predicted fraction — labelled held-out donors 0.453 vs negative controls 0.304
```

## Part 9 — apply to the dark donors

All-donor model, `δ̂` from every labelled donor, `delta_centroid` per dark donor with that donor's
own GMM-estimated benign centroid.

**These donors have no TCR and no published malignant label**, so nothing below is validated
against external truth — the accuracy to quote is Part 6's held-out number, not anything here.
The checks available are internal: does the predicted burden track disease stage, and do the
predicted-clonal cells carry the MF tumour program.

```python  # cell 31
dark_percell, dark_tbl = SC.apply_to_donors(z_all, slim.obs, DARK, delta_all, seed=SEED)
dark_tbl = dark_tbl.merge(donor_tbl[["n_cells"]].rename(columns={"n_cells": "n_cd4"}),
                          left_on="donor", right_index=True, how="left")
print(dark_tbl.sort_values("pred_clonal_frac", ascending=False).round(3).to_string(index=False))
print(f"\n{len(dark_tbl)} dark donors | {len(dark_percell):,} cells | "
      f"predicted clonal {int(dark_percell.clonal_call.sum()):,} "
      f"({dark_percell.clonal_call.mean():.1%})")
```

```
donor  n_cells  pred_clonal_frac           study    disease  n_cd4
Li2024_atlas__CTCL10     1373             0.862          li2024         MF   1373
Li2024_atlas__CTCL18     8122             0.819          li2024         MF   8122
  Li2024_atlas__MF21     2581             0.709          li2024         MF   2581
 Li2024_atlas__CTCL9     2575             0.698          li2024         MF   2575
  Li2024_atlas__PT55     2418             0.697          li2024         MF   2418
Li2024_atlas__CTCL17      341             0.689          li2024         MF    341
Li2024_atlas__CTCL15      136             0.676          li2024         MF    136
Li2024_atlas__CTCL12     3436             0.654          li2024         MF   3436
  Li2024_atlas__MF15     1248             0.615          li2024         MF   1248
  Li2024_atlas__PT52      246             0.561          li2024         MF    246
             D6__MF2     1457             0.539    gaydosik2019         MF   1457
  Li2024_atlas__MF26     2053             0.537          li2024         MF   2053
             D3__P87      494             0.522     brunner2024 CTCL_other    494
  Li2024_atlas__MF22      937             0.496          li2024         MF    937
  Li2024_atlas__MF30     4317             0.479          li2024         MF   4317
            D6__NOR3       92             0.478    gaydosik2019         HC     92
Li2024_atlas__CTCL11      102             0.451          li2024         MF    102
  Li2024_atlas__MF17     5143             0.442          li2024         MF   5143
Li2024_atlas__CTCL14      394             0.416          li2024         MF    394
  Li2024_atlas__MF14     4301             0.410          li2024         MF   4301
Li2024_atlas__CTCL16     1893             0.400          li2024         MF   1893
Li2024_atlas__CTCL13     9488             0.393          li2024         MF   9488
  Li2024_atlas__MF27     1036             0.392          li2024         MF   1036
  Li2024_atlas__MF18      555             0.391          li2024         MF    555
  Li2024_atlas__MF28    13539             0.379          li2024         MF  13539
           D6__SC158      461             0.354    gaydosik2019         MF    461
            D6__NOR4       63             0.349    gaydosik2019         HC     63
            D6__NOR1       54             0.333    gaydosik2019         HC     54
           D6__SC205     1287             0.319    gaydosik2019         MF   1287
            D6__NOR2       57             0.298    gaydosik2019         HC     57
           D6__SC157     1456             0.282    gaydosik2019         MF   1456
            D1__P112      255             0.259 chennareddy2025         HC    255
            D1__P121       78             0.231 chennareddy2025         HC     78
             D6__MF5     6973             0.230    gaydosik2019         MF   6973

34 dark donors | 78,961 cells | predicted clonal 38,567 (48.8%)
```

```python  # cell 32
# Predicted burden per dark donor, next to the labelled donors' true burden for scale.
fig, ax = plt.subplots(figsize=(5, 3))
d = dark_tbl.sort_values("pred_clonal_frac", ascending=False)
ax.bar(np.arange(len(d)), d["pred_clonal_frac"],
       color=["#4c78a8" if s == "li2024" else "#f58518" for s in d["study"]], width=0.75)
ax.axhline(donor_tbl.loc[LABELLED, "clonal_frac"].median(), ls="--", lw=0.8, c="k",
           label="median true burden, labelled donors")
ax.set_xticks(np.arange(len(d)))
ax.set_xticklabels(d["donor"], rotation=75, ha="right", fontsize=4.5)
ax.set_ylabel("predicted clonal fraction")
ax.set_title("Projected tumour burden in the TCR-less samples (unvalidated)", fontsize=9)
ax.legend(fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_dark_burden.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

```python  # cell 33
# Do the predicted-clonal cells carry the MF tumour program? rank_genes_groups within the dark set.
ad_dark = slim[dark_percell.index].copy()
ad_dark.obs["pred_clonal"] = pd.Categorical(
    np.where(dark_percell["clonal_call"].to_numpy(), "clonal", "benign"),
    categories=["benign", "clonal"])
sc.pp.normalize_total(ad_dark, target_sum=1e4)
sc.pp.log1p(ad_dark)
sc.tl.rank_genes_groups(ad_dark, "pred_clonal", groups=["clonal"], reference="benign",
                        method="wilcoxon")
top = pd.DataFrame(ad_dark.uns["rank_genes_groups"]["names"])["clonal"].head(25).tolist()
print("top 25 markers of predicted-clonal cells (dark donors):")
print(", ".join(top))
del ad_dark
gc.collect()
```

```
top 25 markers of predicted-clonal cells (dark donors):
TYMS, MCM7, LMNB1, EZH2, MKI67, LMNA, HELLS, SLC7A5, NCAPG2, CENPM, HMGB2, TOP2A, NUSAP1, FANCI, RRM2, MCM4, BHLHE40, ZWINT, KPNA2, KIFC1, CDK1, KIF22, MCM3, SMC4, CENPF

12039
```

## Part 10 — persist

```python  # cell 35
# (a) held-out fold scores/calls, every arm x rule; (b) the applied call on the dark donors.
fold_out = percell.copy()
fold_out.index.name = "obs_name"
fold_out.to_parquet(FOLD_PARQUET)
print("wrote", FOLD_PARQUET, fold_out.shape)

trans = dark_percell.copy()
trans["study"] = slim.obs["study"].astype(str).reindex(trans.index)
trans["disease"] = slim.obs["disease"].astype(str).reindex(trans.index)
trans["model"] = "semantic_alldonor/delta_centroid"
trans.index.name = "obs_name"
trans.to_parquet(TRANS_PARQUET)
print("wrote", TRANS_PARQUET, trans.shape)

res_full.to_csv(OUT_DIR / "transfer_per_donor_full.csv", index=False)
dark_tbl.to_csv(OUT_DIR / "dark_donor_predictions.csv", index=False)
print("wrote", OUT_DIR / "transfer_per_donor_full.csv", "and dark_donor_predictions.csv")
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/semantic_clonality_folds_v1.parquet (3141104, 8)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/semantic_clonality_transfer_v1.parquet (78961, 6)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/clonality_transfer/transfer_per_donor_full.csv and dark_donor_predictions.csv
```

### Reading the results

- **Headline** = `semantic_perfold` / `delta_centroid` in `transfer_summary.csv`: AUROC and F1 vs
  ALICE on donors the encoder never saw. That is the number that licenses (or does not license)
  Part 9.
- **`auc_raw` of `delta_centroid_null`** should sit at ~0.5. If it does not, the fold/label wiring
  is leaking.
- **`semantic_alldonor` − `semantic_perfold`** is the leakage margin. A large positive gap means the
  encoder was memorising donors and the honest number is the per-fold one.
- **vs `mrvi_u` / `scvi`** — whether the semantic latent earns its keep over the embeddings already
  in the object. **vs `cnv_published`** — whether it beats the production CNV call on the same cells.
- `centroid_r2` answers the perturbation-trial question directly: does `μ̂_ben + δ̂` land on the real
  clonal centroid of an unseen donor.
