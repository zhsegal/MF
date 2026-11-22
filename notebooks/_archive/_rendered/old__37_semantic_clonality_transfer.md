# old/37_semantic_clonality_transfer.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/37_semantic_clonality_transfer.ipynb` and is never run._

# 37 — Transferring TCR clonality to TCR-less skin samples: semantic vs LDVAE vs scVI

TCR exists for **36 of the 82 skin donors** (nb30). The rest — 21 li2024 + 8 gaydosik2019 donors —
have no clonality data *and* no published malignant label, so they are dark. This notebook asks
whether the malignant/benign split can be **transferred** into them through the latent, and which
latent does it best.

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
donors: each fold retrains the encoder with the 3 held-out donors' cells fully removed
(`perturbation_trial`'s protocol), encodes everything with that fold's model, and scores the
transfer on the 3 unseen donors. The applied call on the dark donors comes last and is explicitly
**not** validated against external truth — they have none.

**Three arms, one protocol.** Every arm sees the same folds, cells, HVGs, epochs, KL schedule,
batch/labels keys and seed — the only differences are the ones named:

| arm | model | difference |
|---|---|---|
| `semantic` | `SemanticSCVI` | geometric Geneformer prior, `coherence_weight=1000` |
| `ldvae` | `SemanticSCVI` | **`coherence_weight=0`** — architecture-identical no-prior ablation |
| `scvi` | `SCVI` | nonlinear decoder, no positivity constraint |

`ldvae` isolates the semantic prior with one knob: the ELBO, the positive-weight decoder, the
decorrelation term and the KL warm-up are byte-identical to `semantic`. `scvi` cannot be matched
that tightly — it has no `weights_positive`, hence no decorrelation penalty, and its decoder is
nonlinear by construction. Its batch norm is set to `"encoder"`, the faithful translation of the
LDVAE family's `use_batch_norm=False` (which disables *decoder* BN only).

> **HEAVY** cells are marked. The object is ~20 GB and the fold training is 27 GPU runs (3 arms ×
> 8 folds + 3 all-donor references) — run those on the GPU/compute kernel (`neural_nmf_env`) or via
> `jobs/run_clonality_folds.sh`, never on the login node.

## Part 0 — parameters

```python  # cell 2
# ============================================================
# Parameters — cohort filters, the three arms' knobs (nb18 recipe), fold layout.
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
INPUT_H5AD = JOB_INPUT / "clonality_input.h5ad"   # written by Part 2; read back by Part 3 onward
MAP_PT     = JOB_INPUT / "clonality_semantic_map.pt"
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

# The three arms. Everything outside `kwargs` (epochs, KL warm-up, batch/labels keys, folds, HVGs,
# train pool, seed) is shared, so each arm differs only by what is written here.
#   ldvae — SemanticSCVI with the semantic loss switched off. `SemanticLDVAE.loss` still computes
#           the coherence term but adds `scale * 0 * raw_loss`, so the gradient contribution is
#           exactly zero while the ELBO / positive decoder / decorrelation / KL schedule stay
#           identical to `semantic`. (`train_or_load_nonneg_ldvae` would NOT match: it trains bare,
#           leaving n_epochs_kl_warmup at scvi's default 400.)
#   scvi  — use_batch_norm="encoder" is the faithful translation of the LDVAE family's
#           use_batch_norm=False, which only disables the decoder BN (encoder BN is hard-coded on).
#           gene_likelihood must be given explicitly: SCVI defaults to zinb, not nb.
SHUFFLE_MAP_SEED = 0   # row permutation seed for the no-prior control below

# ---- the STRUCTURAL decoder (Rounds A-E on haniffa_cd8) ----
# Loadings are W = softplus(s_k cos(S_g, alpha_k) - b_k), an explicit function of the gene
# embedding, rather than a free matrix that the semantic loss merely regularizes. Config comes
# from Round E's final_config.json when that sweep has run, else the frozen Round C default.
# n_layers is 2 here (the frozen value) against the penalty arm's 1; the `ldvae` twin below is
# bumped to match so that comparison stays controlled.
ROUND_E_CONFIG = (NB_DIR.parent / "benchmark_results" / "haniffa_cd8_roundE_sweep"
                  / "final_config.json")
_ROUND_C_STRUCT = dict(
    decoder_mode="structural", scale_mode="decoupled",
    logit_scale_init=15.0, logit_scale_min=5.0, logit_scale_max=30.0, learn_scale=True,
    b_mode="quantile", b_quantile=0.90, b_slack=0.0, b_detach=True,
    alpha_init="kmeans_expr", learn_alpha=True,
    missing_residual=False, usage_balance_weight=0.0, usage_balance_mode="entropy",
    n_latent=N_LATENT, n_layers=2, n_hidden=128, dropout_rate=0.1,
    gene_likelihood="nb", weights_positive=True, use_batch_norm=False,
)
if ROUND_E_CONFIG.exists():
    _re = json.loads(ROUND_E_CONFIG.read_text())
    STRUCT_KWARGS, STRUCT_WARMUP = dict(_re["kwargs"]), _re["warmup_epochs"]
    STRUCT_PROVENANCE = f"Round E '{_re['label']}'"
    if _re["embedding"] != "geneformer":
        raise ValueError(
            f"Round E selected the '{_re['embedding']}' embedding, but this notebook's semantic "
            f"map ({SEMANTIC_CACHE.name}) is Geneformer. Build the matching map before enabling "
            f"the structural arm.")
else:
    STRUCT_KWARGS, STRUCT_WARMUP = dict(_ROUND_C_STRUCT), 0
    STRUCT_PROVENANCE = "Round C frozen default (Round E has not run)"

# The arms. Everything outside `kwargs` (epochs, KL warm-up, batch/labels keys, folds, HVGs,
# train pool, seed) is shared, so each arm differs only by what is written here.
#   structural_shuffled — the matched no-prior control FOR THE STRUCTURAL ARM. It must be a
#           row-permuted embedding, not coherence_weight=0: in decoder_mode='structural' the
#           semantic penalty is not used at all (_vae.py:1192, 1207), so coherence_weight is
#           INERT and an `ldvae`-style twin of the structural arm would be the identical model.
#           Permuting rows leaves the embedding's geometry exactly intact and destroys only the
#           gene-to-vector assignment.
#   ldvae — SemanticSCVI with the semantic loss switched off. This is a valid no-prior control
#           for the PENALTY arm only. `SemanticLDVAE.loss` still computes the coherence term but
#           adds `scale * 0 * raw_loss`, so the gradient contribution is exactly zero while the
#           ELBO / positive decoder / decorrelation / KL schedule stay identical to `semantic`.
#           (`train_or_load_nonneg_ldvae` would NOT match: it trains bare, leaving
#           n_epochs_kl_warmup at scvi's default 400.)
#   scvi  — use_batch_norm="encoder" is the faithful translation of the LDVAE family's
#           use_batch_norm=False, which only disables the decoder BN (encoder BN is hard-coded on).
#           gene_likelihood must be given explicitly: SCVI defaults to zinb, not nb.
MODEL_NAMES = ["structural", "structural_shuffled", "semantic", "ldvae", "scvi"]
MODELS = {
    "structural": {"kwargs": STRUCT_KWARGS, "warmup_epochs": STRUCT_WARMUP},
    "structural_shuffled": {"kwargs": STRUCT_KWARGS, "warmup_epochs": STRUCT_WARMUP,
                            "shuffle_map": SHUFFLE_MAP_SEED},
    "semantic": {"kwargs": SEMANTIC_KWARGS, "warmup_epochs": WARMUP_EPOCHS},
    "ldvae": {"kwargs": {**SEMANTIC_KWARGS, "coherence_weight": 0.0, "n_layers": 2},
              "warmup_epochs": WARMUP_EPOCHS},
    "scvi": {"kwargs": dict(n_latent=N_LATENT, n_layers=1, n_hidden=128, dropout_rate=0.1,
                            gene_likelihood="nb", use_batch_norm="encoder")},
}

# ---- folds / scoring ----
SEED               = 0
# The shuffled-delta null is a RANDOM DIRECTION in a 10-d latent, so one draw is not a null -- its
# AUROC spread across draws is sd ~0.37 and it only converges to 0.5 after ~50 draws. run_cv used
# a single permutation per fold, which is how a null whose expectation is 0.5 came back at 0.385
# and got read as a bug. Average over N_NULL_DRAWS and read mean +/- sd, not the point estimate.
N_NULL_DRAWS       = 20
# Two-mode gate for the per-donor benign centroid, as a fraction of ||delta_hat||. Without it
# gmm_benign_side force-fits 2 components to EVERY donor and falls back to a median split, so a
# donor with zero clonal cells is cut in half by construction -- healthy controls came back at
# pred_clonal_frac 0.59-0.87. Set to None to reproduce the pre-fix behaviour.
BENIGN_GATE_FRAC   = 0.5
FOLD_SIZE          = 3       # leave-3-samples-out
TRAIN_FRAC         = 1 / 3   # per-donor training subsample (nb19 sweep used the same)
TRAIN_MIN_CELLS    = 500     # never subsample a donor below this
EVAL_MIN_CELLS     = 200     # holdout eligibility: per-donor metrics need enough of both classes
EVAL_MIN_CLONAL    = 25
EVAL_MIN_ANCHOR    = 25


def _cache_slug(n=10):
    """Stable hash of every param that affects a trained model (shared by all folds)."""
    # hash EVERY arm, not just SEMANTIC_KWARGS: the cache path is <slug>/<arm>/<fold>, so a
    # changed structural config under an unchanged slug reloads the previous variant's checkpoint
    blob = json.dumps({"models": {k: MODELS[k] for k in sorted(MODELS)},
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
param slug: 81e1097a79 | model cache: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/models/.model_cache_semantic_clonality/81e1097a79
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

`runs` = one entry per (arm, fold) plus one `all` per arm (no holdout) — the leakage reference and
the model used for the applied call.

The first cell re-reads the slim input if the kernel restarted, so the config can be regenerated
without re-running the ~20 GB Part 1–2.

```python  # cell 14
# Reload guard: Part 3 onward needs only the slim input, so a fresh kernel can start here.  HEAVY-ish
if "slim" not in dir():
    slim = sc.read_h5ad(INPUT_H5AD)
    donor_tbl = pd.read_csv(OUT_DIR / "donor_table.csv", index_col=0)
    print("reloaded slim input:", slim.shape)

folds = SC.make_triples(donor_tbl, size=FOLD_SIZE, seed=SEED, batch_key=BATCH_KEY)
SC.check_batch_coverage(slim.obs, folds, batch_key=BATCH_KEY)
print(f"{len(folds)} folds of {sorted({len(f) for f in folds})} | "
      f"{len({d for f in folds for d in f})} donors tested once")
print(f"submit with:  cd {NB_DIR / 'jobs'} && ARRAY=1 ./run_clonality_folds.sh")
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
submit with:  cd /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs && ARRAY=1 ./run_clonality_folds.sh
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
# Latent-free fold gate — costs nothing and catches the bugs worth catching before 18 GPU jobs.
_ic = slim.obs["is_clonal"].to_numpy(dtype=bool)
_ia = slim.obs["is_anchor"].to_numpy(dtype=bool)
_dn = slim.obs["donor"].astype(str).to_numpy()

held_all = [d for f in folds for d in f]
assert len(held_all) == len(set(held_all)), "a donor is held out by more than one fold"
assert set(held_all) <= set(donor_tbl.index[donor_tbl["eligible"]]), "an ineligible donor is held out"
for k, f in enumerate(folds):
    for h in f:
        lab = (_dn == h) & (_ic | _ia)
        y = _ic[lab]
        assert 0 < y.sum() < y.size, f"fold{k} {h}: labelled cells are one class ({y.sum()}/{y.size})"
print(f"OK: {len(folds)} disjoint folds, {len(held_all)} donors tested once, "
      f"both classes present in every held-out donor")
```

```
OK: 8 disjoint folds, 26 donors tested once, both classes present in every held-out donor
```

```python  # cell 16
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
    "semantic_kwargs": SEMANTIC_KWARGS,          # kept for reference; the job reads "models"
    "models": MODELS,
    # one run per (arm, fold) plus one all-donor reference per arm. `cache_key` is the fold, so the
    # cache path is <slug>/<arm>/<fold> and `--folds 3` still selects fold 3 of every arm.
    "runs": [{"name": f"{m}_{key}", "cache_key": key, "model": m, "held_out": list(held)}
             for m in MODEL_NAMES
             for key, held in ([(f"fold{k}", f) for k, f in enumerate(folds)] + [("all", [])])],
    "folds": folds,
    "notes": (f"nb37 clonality transfer: leave-{FOLD_SIZE}-out over {len(LABELLED)} labelled "
              f"donors ({len(folds)} folds) + all-donor reference, x {len(MODEL_NAMES)} arms "
              f"({', '.join(MODEL_NAMES)}). Geneformer geometric prior, HVG={HVG_TOP_N}, "
              f"n_latent={N_LATENT}, batch={BATCH_KEY}."),
}
CONFIG.write_text(json.dumps(cfg, indent=2))
print("wrote", CONFIG, f"| {len(cfg['runs'])} runs")
print(cfg["notes"])
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/jobs/clonality_folds_config.json | 45 runs
nb37 clonality transfer: leave-3-out over 31 labelled donors (8 folds) + all-donor reference, x 5 arms (structural, structural_shuffled, semantic, ldvae, scvi). Geneformer geometric prior, HVG=2500, n_latent=10, batch=study.
```

## Part 4 — per-fold training of the three arms · HEAVY (bsub / GPU)

`3 × (len(folds) + 1)` runs: every arm × every fold, plus one all-donor reference per arm (the fold
count falls out of how many donors clear the eligibility gate — Part 3 prints it). Each run removes
its held-out donors' cells from training entirely, then encodes **all** cells (held-out and dark
included) with that run's model.

```bash
cd notebooks/MF/jobs
ARRAY=1 ./run_clonality_folds.sh          # one GPU job per run, in parallel
# or sequentially in one job:  ./run_clonality_folds.sh
# one run (smoke test):        ./run_clonality_folds.sh --folds scvi_fold5
# a fold across every arm:     ./run_clonality_folds.sh --folds 3
```

Models cache under `models/.model_cache_semantic_clonality/<slug>/<arm>/<fold>` and latents land in
`benchmark_results/clonality_transfer/z_<arm>_<fold>.npy`; a run whose latent exists is skipped, so
re-runs are free. The next cell just waits for the files.

```python  # cell 18
# Survive a kernel restart while the bsub jobs ran: everything below needs only the slim input.
if "slim" not in dir():
    slim = sc.read_h5ad(INPUT_H5AD)
    donor_tbl = pd.read_csv(OUT_DIR / "donor_table.csv", index_col=0)
    print("reloaded slim input:", slim.shape)
if "folds" not in dir():
    folds = json.loads(CONFIG.read_text())["folds"]
LABELLED = list(donor_tbl.index[donor_tbl["group"].isin(["labelled", "labelled_small"])])
DARK = list(donor_tbl.index[donor_tbl["group"] == "dark"])
NO_CLONE = list(donor_tbl.index[donor_tbl["group"] == "no_clone"])

# label arrays used by every cell below
donors_np = slim.obs["donor"].astype(str).to_numpy()
ic = slim.obs["is_clonal"].to_numpy(dtype=bool)
ia = slim.obs["is_anchor"].to_numpy(dtype=bool)

Z_PATHS = {(m, k): OUT_DIR / f"z_{m}_fold{k}.npy"
           for m in MODEL_NAMES for k in range(len(folds))}
Z_PATHS.update({(m, "all"): OUT_DIR / f"z_{m}_all.npy" for m in MODEL_NAMES})
missing = {k: p for k, p in Z_PATHS.items() if not p.exists()}
if missing:
    raise FileNotFoundError(
        f"{len(missing)}/{len(Z_PATHS)} latents missing: {sorted(p.name for p in missing.values())}\n"
        f"submit with:  cd {NB_DIR / 'jobs'} && ARRAY=1 ./run_clonality_folds.sh")

Z = {m: {k: np.load(Z_PATHS[(m, k)]) for k in list(range(len(folds))) + ["all"]}
     for m in MODEL_NAMES}
for m, zs in Z.items():
    for k, z in zs.items():
        assert z.shape[0] == slim.n_obs, f"{m} {k}: {z.shape} vs {slim.n_obs}"
print(f"loaded {len(Z_PATHS)} latents | "
      + " | ".join(f"{m} {Z[m]['all'].shape}" for m in MODEL_NAMES))
```

```
loaded 45 latents | structural (287175, 10) | structural_shuffled (287175, 10) | semantic (287175, 10) | ldvae (287175, 10) | scvi (287175, 10)
```

## Part 5 — evaluate

Four rules × three arms, all per-fold, all on the same folds, donors, labels and cells.

- `delta_centroid` — the headline: nearer-centroid to $\hat\mu_{ben}$ / $\hat\mu_{ben}+\hat\delta$.
  Parameter-free at test time.
- `delta_threshold` — same axis, cut transferred from the training donors instead of the midpoint.
- `logistic_z` — supervised ceiling on donor-centred $z$.
- `delta_centroid_null` — labels shuffled within each training donor, **refit per arm per fold from
  that arm's own latent**, so each arm carries its own floor. `auc_raw` should sit at 0.5.

Read **`auc_raw`**, not `auc`: `M.binary_scores` reports the polarity-folded $\max(a, 1-a)$, which
floors a random direction well above 0.5 and would make the nulls look informative.

```python  # cell 20
arms = [(m, Z[m]) for m in MODEL_NAMES]      # per-fold latents; dict fold -> z, "all" ignored by run_cv

res_parts, pc_parts = [], []
for name, z in arms:
    r, pc = SC.run_cv(z, slim.obs, folds, latent_name=name, train_donors_all=LABELLED, seed=SEED,
                      n_null_draws=N_NULL_DRAWS, gate_frac=BENIGN_GATE_FRAC)
    res_parts.append(r)
    pc_parts.append(pc)
    print(f"[{name}] {len(r)} rows")
res = pd.concat(res_parts, ignore_index=True)
percell = pd.concat(pc_parts)

summary = SC.summarize(res).sort_values("auc_raw_mean", ascending=False)
with pd.option_context("display.width", 240, "display.max_columns", None):
    print(summary.to_string())
summary.to_csv(OUT_DIR / "transfer_summary.csv")
res.to_csv(OUT_DIR / "transfer_per_donor.csv", index=False)
```

```
[structural] 598 rows
[structural_shuffled] 598 rows
[semantic] 598 rows
[ldvae] 598 rows
[scvi] 598 rows
                                         n_donors  n_null_draws  frac_single_mode  auc_mean  auc_sd  auc_raw_mean  auc_raw_sd  f1_mean  f1_sd  balanced_acc_mean  balanced_acc_sd  recall_mean  recall_sd  specificity_mean  specificity_sd  centroid_r2_mean  centroid_r2_sd
latent              rule                                                                                                                                                                                                                                                     
scvi                logistic_z                 26             1             0.269     0.951   0.056         0.951       0.056    0.746  0.262              0.797            0.135        0.707      0.333             0.888           0.101             0.540           0.255
ldvae               logistic_z                 26             1             0.231     0.950   0.057         0.950       0.057    0.729  0.278              0.796            0.146        0.700      0.345             0.891           0.117             0.749           0.213
scvi                delta_centroid             26             1             0.269     0.945   0.065         0.945       0.065    0.697  0.329              0.795            0.171        0.646      0.374             0.945           0.054             0.540           0.255
                    delta_threshold            26             1             0.269     0.945   0.065         0.945       0.065    0.758  0.261              0.805            0.142        0.725      0.336             0.884           0.098             0.540           0.255
structural          logistic_z                 26             1             0.192     0.939   0.064         0.939       0.064    0.781  0.227              0.813            0.123        0.763      0.297             0.863           0.108             0.695           0.252
semantic            logistic_z                 26             1             0.308     0.931   0.078         0.931       0.078    0.708  0.252              0.773            0.126        0.688      0.334             0.859           0.116             0.766           0.194
structural          delta_threshold            26             1             0.192     0.926   0.073         0.926       0.073    0.792  0.199              0.798            0.104        0.792      0.269             0.803           0.145             0.695           0.252
                    delta_centroid             26             1             0.192     0.926   0.073         0.926       0.073    0.742  0.270              0.799            0.144        0.701      0.324             0.897           0.105             0.695           0.252
ldvae               delta_centroid             26             1             0.231     0.903   0.094         0.891       0.138    0.682  0.327              0.759            0.166        0.647      0.366             0.872           0.127             0.749           0.213
                    delta_threshold            26             1             0.231     0.903   0.094         0.891       0.138    0.736  0.247              0.750            0.136        0.747      0.325             0.752           0.177             0.749           0.213
semantic            delta_centroid             26             1             0.308     0.873   0.125         0.869       0.135    0.614  0.341              0.727            0.167        0.575      0.380             0.878           0.122             0.766           0.194
                    delta_threshold            26             1             0.308     0.873   0.125         0.869       0.135    0.728  0.235              0.736            0.119        0.737      0.302             0.736           0.209             0.766           0.194
structural_shuffled logistic_z                 26             1             0.038     0.835   0.102         0.835 
... [2045 more chars truncated]
```

```python  # cell 21
# Gate + winner selection. Every assert here has caught a real bug in this pipeline's history.
for m in MODEL_NAMES:                       # a latent that is identically zero for some study makes
    for k, z in Z[m].items():               # every metric on those donors meaningless (see the old
        dead = int((np.abs(z).sum(1) == 0).sum())   # published X_scVI arm: 30% of cells were zero)
        assert dead == 0, f"{m} {k}: {dead:,} all-zero latent rows"

n_held = len({d for f in folds for d in f})
_n_rows = n_held * (len([r for r in SC.RULES if not r.endswith("_null")])
                    + len([r for r in SC.RULES if r.endswith("_null")]) * N_NULL_DRAWS)
for m in MODEL_NAMES:
    sub = res[res.latent == m]
    assert len(sub) == _n_rows, f"{m}: {len(sub)} rows, expected {_n_rows}"
    assert sub.groupby(["fold", "donor"])["n"].nunique().eq(1).all(), f"{m}: rules disagree on cells"

# The null as a DISTRIBUTION. A random direction in a 10-d latent can score anywhere from ~0.05 to
# ~0.95 on a single draw, so the number to compare the real arm against is the null's upper tail,
# not 0.5. If the real arm does not clear p95 it has not beaten chance at this donor count.
_nl = res[res.rule == "delta_centroid_null"]
_null_stats = _nl.groupby("latent")["auc_raw"].agg(
    mean="mean", sd="std", p05=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))
print(f"shuffled-δ null over {N_NULL_DRAWS} draws x {n_held} donors:")
print(_null_stats.round(3).to_string())
for m, r in _null_stats.iterrows():
    flag = "" if 0.35 <= r["mean"] <= 0.65 else "   <-- WARN, mean is off 0.5"
    print(f"  [{m}] mean {r['mean']:.3f} +/- {r['sd']:.3f} (p95 {r['p95']:.3f}){flag}")

# how often the two-mode gate fired on a HELD-OUT donor. These donors all carry clonal cells, so
# a high rate here means the gate is too strict and is suppressing real calls.
if "single_mode" in res:
    _sm = res[~res.rule.str.endswith("_null")].groupby("latent")["single_mode"].mean()
    print("\nfrac of held-out (clone-bearing) donors gated as single-mode — should be LOW:")
    print(_sm.round(3).to_string())

# best RULE per arm (nulls excluded), then the best arm — these drive the figures and Part 7-8
real = res[~res.rule.str.endswith("_null")]
mean_auc = real.groupby(["latent", "rule"])["auc_raw"].mean()
BEST_RULE = {m: mean_auc[m].idxmax() for m in MODEL_NAMES}
BEST_LATENT = max(MODEL_NAMES, key=lambda m: mean_auc[(m, BEST_RULE[m])])
print(f"\nbest rule per arm: " + ", ".join(f"{m}={BEST_RULE[m]} ({mean_auc[(m, BEST_RULE[m])]:.3f})"
                                          for m in MODEL_NAMES))
print(f"winner: {BEST_LATENT} / {BEST_RULE[BEST_LATENT]} -> used for the dark-donor transfer")

# the prior question, asked directly: does the real embedding beat its own row-permuted twin?
if {"structural", "structural_shuffled"} <= set(MODEL_NAMES):
    _s = mean_auc[("structural", BEST_RULE["structural"])]
    _ss = mean_auc[("structural_shuffled", BEST_RULE["structural_shuffled"])]
    print(f"structural {_s:.3f} vs structural_shuffled {_ss:.3f} -> gap {_s - _ss:+.3f}. "
          f"This gap, not the AUROC itself, is what the semantic prior buys here.")
```

```
shuffled-δ null over 20 draws x 26 donors:
                      mean     sd    p05    p95
latent                                         
ldvae                0.507  0.303  0.035  0.972
scvi                 0.528  0.280  0.066  0.943
semantic             0.521  0.310  0.050  0.969
structural           0.528  0.303  0.024  0.971
structural_shuffled  0.508  0.279  0.072  0.953
  [ldvae] mean 0.507 +/- 0.303 (p95 0.972)
  [scvi] mean 0.528 +/- 0.280 (p95 0.943)
  [semantic] mean 0.521 +/- 0.310 (p95 0.969)
  [structural] mean 0.528 +/- 0.303 (p95 0.971)
  [structural_shuffled] mean 0.508 +/- 0.279 (p95 0.953)

frac of held-out (clone-bearing) donors gated as single-mode — should be LOW:
latent
ldvae                  0.231
scvi                   0.269
semantic               0.308
structural             0.192
structural_shuffled    0.038

best rule per arm: structural=logistic_z (0.939), structural_shuffled=logistic_z (0.835), semantic=logistic_z (0.931), ldvae=logistic_z (0.950), scvi=logistic_z (0.951)
winner: scvi / logistic_z -> used for the dark-donor transfer
structural 0.939 vs structural_shuffled 0.835 -> gap +0.103. This gap, not the AUROC itself, is what the semantic prior buys here.
```

## Part 6 — figures

```python  # cell 23
# Fig 1 — does the transfer work? ONE fixed rule per arm next to that arm's own shuffled-δ null.
# The rule is fixed to `delta_centroid` rather than each arm's best: per-arm best-rule selection let
# a different rule win per arm (the semantic arm's was logistic_z), so the bars were not a
# like-for-like comparison of latents — and delta_centroid is the rule the null is defined against.
# BEST_RULE / BEST_LATENT still drive the winner selection in Part 7-8; only this figure is fixed.
# Colours are looked up with a fallback so adding an arm to MODEL_NAMES cannot KeyError here.
_PALETTE = ["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#888", "#b279a2", "#9d755d"]
ARM_COL = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(MODEL_NAMES)}   # cd24 reuses this
FIG_RULE = "delta_centroid"
# The penalty arm is off this panel: `structural` is the model the notebook is about, and showing
# both semantic decoders under the same name is what made the old figure hard to read. It is still
# trained and still scored in the cd21 table -- only hidden here.
FIG_DROP = {"semantic"}
# `structural` IS the semantic model in this figure's vocabulary; its row-permuted twin keeps the
# name so the pair reads as one comparison.
FIG_LABEL = {"structural": "semantic", "structural_shuffled": "semantic\n(shuffled map)"}
# the row-permuted control is the reference the real arms are read against, so it sits rightmost
ARM_ORDER = ([m for m in MODEL_NAMES if m not in FIG_DROP and not m.endswith("_shuffled")]
             + [m for m in MODEL_NAMES if m not in FIG_DROP and m.endswith("_shuffled")])
piv = res.groupby(["latent", "rule"])["auc_raw"].agg(["mean", "std"])
x = np.arange(len(ARM_ORDER))
fig, ax = plt.subplots(figsize=(5, 3))
# real bars carry the arm colour; the nulls are one neutral colour so the legend reads unambiguously
ax.bar(x - 0.19, [piv.loc[(m, FIG_RULE), "mean"] for m in ARM_ORDER], width=0.36,
       yerr=[piv.loc[(m, FIG_RULE), "std"] for m in ARM_ORDER], capsize=2.5,
       color=[ARM_COL[m] for m in ARM_ORDER], label=FIG_RULE)
ax.bar(x + 0.19, [piv.loc[(m, "delta_centroid_null"), "mean"] for m in ARM_ORDER], width=0.36,
       yerr=[piv.loc[(m, "delta_centroid_null"), "std"] for m in ARM_ORDER], capsize=2.5,
       color="#d9d9d9", hatch="///", edgecolor="#999", label="shuffled-δ null")
ax.axhline(0.5, ls=":", lw=0.8, c="k")
ax.set_xticks(x)
ax.set_xticklabels([FIG_LABEL.get(m, m) for m in ARM_ORDER], fontsize=7)
ax.set_ylabel("AUROC vs ALICE (held-out donor)")
ax.set_ylim(0.3, 1.02)
_best = max(ARM_ORDER, key=lambda m: piv.loc[(m, FIG_RULE), "mean"])
ax.set_title(f"Clonality transfers to unseen samples ({FIG_RULE}) — best: "
             f"{FIG_LABEL.get(_best, _best).replace(chr(10), ' ')} "
             f"{piv.loc[(_best, FIG_RULE), 'mean']:.2f}", fontsize=9)
ax.legend(fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_auroc.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

```python  # cell 24
# Fig 2 — is the tumour BURDEN right? predicted vs true clonal fraction, one dot per held-out donor.
# Same arm set, order, rule and naming as Fig 1: FIG_RULE for every arm (not per-arm BEST_RULE),
# the penalty arm hidden via FIG_DROP, `structural` shown as "semantic", its row-permuted twin last.
fig, ax = plt.subplots(figsize=(5, 3))
for m in ARM_ORDER:
    d = res[(res.latent == m) & (res.rule == FIG_RULE)]
    r2 = SC.r2_latent(d["pred_frac"], d["true_frac"])
    ax.scatter(d["true_frac"], d["pred_frac"], s=22, alpha=0.8, color=ARM_COL[m],
               label=f"{FIG_LABEL.get(m, m).replace(chr(10), ' ')} (R²={r2:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("true ALICE clonal fraction")
ax.set_ylabel("predicted clonal fraction")
ax.set_title(f"Per-donor tumour burden on held-out samples ({FIG_RULE})", fontsize=9)
ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_transfer_fraction.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

## Part 7 — negative controls

The rule must *not* fire where there is nothing to find:

1. **`no_clone` donors** — have TCR and *zero* ALICE-clonal cells. Predicted clonal fraction
   should be low. (Donors with a clone too small to be a labelled unit are `labelled_small`, not
   `no_clone` — they carry a clone and would be the wrong thing to assert cleanliness on.)
2. **HC donors** — healthy skin, benign by definition.
3. **CD8 cells** — read straight off the full object; MF malignancy is CD4, so these should be
   near-zero. Scored with the all-donor model's δ̂ (they were never in the CD4 training set).

Everything from here on uses the **winning arm and rule** chosen in Part 5, on that arm's all-donor
latent.

```python  # cell 26
z_win = Z[BEST_LATENT]["all"]
rule_win = BEST_RULE[BEST_LATENT]
delta_all, delta_tbl = SC.fit_delta(z_win, donors_np, ic, ia, LABELLED)
print(f"winner {BEST_LATENT} / {rule_win} | per-donor δ_d norms:",
      delta_tbl["norm"].describe().round(2).to_dict())

# delta_centroid needs nothing fitted; the other rules carry a cut learned on the labelled donors
fitted_win = (None if rule_win == "delta_centroid" else
              SC.fit_rules(z_win, donors_np, ic, ia, LABELLED, delta_all, seed=SEED)[rule_win])

ctrl_donors = NO_CLONE + [d for d in donor_tbl.index
                          if str(donor_tbl.loc[d, "disease"]) == "HC" and d not in NO_CLONE]
_, ctrl_tbl = SC.apply_to_donors(z_win, slim.obs, ctrl_donors, delta_all, seed=SEED,
                                 rule=rule_win, fitted=fitted_win)
ctrl_tbl = ctrl_tbl.merge(donor_tbl[["group", "n_clonal", "n_anchor"]], left_on="donor",
                          right_index=True, how="left")
print("\nnegative-control donors (predicted clonal fraction should be low):")
print(ctrl_tbl.sort_values("pred_clonal_frac", ascending=False).round(3).to_string(index=False))

lab_frac = res[(res.latent == BEST_LATENT) & (res.rule == rule_win)]["pred_frac"].mean()
print(f"\nmean predicted fraction — labelled held-out donors {lab_frac:.3f} "
      f"vs negative controls {ctrl_tbl.pred_clonal_frac.mean():.3f}")

# The two-mode gate should fire on donors that have no malignant population. A control run where
# NO control is gated means the gate is not working, not that every healthy donor carries a clone
# -- which is how D1__P112 (HC) previously came back at pred_clonal_frac 0.867.
if "single_mode" in ctrl_tbl:
    n_gated = int(ctrl_tbl["single_mode"].sum())
    print(f"\ntwo-mode gate fired on {n_gated}/{len(ctrl_tbl)} negative-control donors "
          f"(gate_frac={BENIGN_GATE_FRAC}); ungated, every one of them is split ~in half by "
          f"construction")
    _hi = ctrl_tbl[ctrl_tbl.pred_clonal_frac > 0.25]
    if len(_hi):
        print(f"WARN — {len(_hi)} control donor(s) still above 0.25: "
              f"{_hi.sort_values('pred_clonal_frac', ascending=False)[['donor', 'disease', 'pred_clonal_frac', 'single_mode']].to_string(index=False)}")
    else:
        print("all negative controls below 0.25")
```

```
winner scvi / logistic_z | per-donor δ_d norms: {'count': 31.0, 'mean': 2.44, 'std': 0.77, 'min': 0.89, '25%': 1.97, '50%': 2.28, '75%': 2.77, 'max': 4.42}
skip H__HC1: 11 cells

negative-control donors (predicted clonal fraction should be low):
   donor  n_cells  pred_clonal_frac  single_mode           study    disease    group  n_clonal  n_anchor
D1__P112      255             0.867        False chennareddy2025         HC     dark         0         0
D6__NOR2       57             0.702        False    gaydosik2019         HC     dark         0         0
B5__PT22     6380             0.497        False        buus2025         SS no_clone         0      1688
D6__NOR3       92             0.152         True    gaydosik2019         HC     dark         0         0
D1__P115      463             0.143         True chennareddy2025         HC no_clone         0       327
D6__NOR1       54             0.111         True    gaydosik2019         HC     dark         0         0
D1__P121       78             0.090         True chennareddy2025         HC     dark         0         0
D1__P116      451             0.082         True chennareddy2025         HC no_clone         0       384
D6__NOR4       63             0.079         True    gaydosik2019         HC     dark         0         0
 D3__P94     2905             0.072         True     brunner2024 CTCL_other no_clone         0       861

mean predicted fraction — labelled held-out donors 0.505 vs negative controls 0.279

two-mode gate fired on 7/10 negative-control donors (gate_frac=0.5); ungated, every one of them is split ~in half by construction
WARN — 3 control donor(s) still above 0.25:    donor disease  pred_clonal_frac  single_mode
D1__P112      HC          0.866667        False
D6__NOR2      HC          0.701754        False
B5__PT22      SS          0.497492        False
```

## Part 8 — apply to the dark donors

Winning arm's all-donor model, `δ̂` from every labelled donor, the winning rule per dark donor with
that donor's own GMM-estimated benign centroid.

**These donors have no TCR and no published malignant label**, so nothing below is validated
against external truth — the accuracy to quote is Part 5's held-out number, not anything here.
The checks available are internal: does the predicted burden track disease stage, and do the
predicted-clonal cells carry the MF tumour program.

```python  # cell 28
dark_percell, dark_tbl = SC.apply_to_donors(z_win, slim.obs, DARK, delta_all, seed=SEED,
                                           rule=rule_win, fitted=fitted_win)
dark_tbl = dark_tbl.merge(donor_tbl[["n_cells"]].rename(columns={"n_cells": "n_cd4"}),
                          left_on="donor", right_index=True, how="left")
print(dark_tbl.sort_values("pred_clonal_frac", ascending=False).round(3).to_string(index=False))
print(f"\n{len(dark_tbl)} dark donors | {len(dark_percell):,} cells | "
      f"predicted clonal {int(dark_percell.clonal_call.sum()):,} "
      f"({dark_percell.clonal_call.mean():.1%})")
```

```
donor  n_cells  pred_clonal_frac  single_mode           study    disease  n_cd4
Li2024_atlas__CTCL10     1373             0.905        False          li2024         MF   1373
Li2024_atlas__CTCL12     3436             0.886        False          li2024         MF   3436
            D1__P112      255             0.867        False chennareddy2025         HC    255
  Li2024_atlas__MF18      555             0.793        False          li2024         MF    555
  Li2024_atlas__MF21     2581             0.756        False          li2024         MF   2581
            D6__NOR2       57             0.702        False    gaydosik2019         HC     57
Li2024_atlas__CTCL13     9488             0.677        False          li2024         MF   9488
Li2024_atlas__CTCL16     1893             0.671        False          li2024         MF   1893
  Li2024_atlas__PT52      246             0.650        False          li2024         MF    246
           D6__SC157     1456             0.650        False    gaydosik2019         MF   1456
  Li2024_atlas__MF30     4317             0.636        False          li2024         MF   4317
           D6__SC205     1287             0.558        False    gaydosik2019         MF   1287
  Li2024_atlas__MF22      937             0.419        False          li2024         MF    937
 Li2024_atlas__CTCL9     2575             0.388        False          li2024         MF   2575
             D6__MF2     1457             0.382        False    gaydosik2019         MF   1457
  Li2024_atlas__PT55     2418             0.380        False          li2024         MF   2418
Li2024_atlas__CTCL15      136             0.360        False          li2024         MF    136
           D6__SC158      461             0.278        False    gaydosik2019         MF    461
  Li2024_atlas__MF28    13539             0.177         True          li2024         MF  13539
  Li2024_atlas__MF27     1036             0.164         True          li2024         MF   1036
            D6__NOR3       92             0.152         True    gaydosik2019         HC     92
Li2024_atlas__CTCL11      102             0.137         True          li2024         MF    102
             D6__MF5     6973             0.123         True    gaydosik2019         MF   6973
Li2024_atlas__CTCL17      341             0.111         True          li2024         MF    341
            D6__NOR1       54             0.111         True    gaydosik2019         HC     54
  Li2024_atlas__MF14     4301             0.110         True          li2024         MF   4301
            D1__P121       78             0.090         True chennareddy2025         HC     78
Li2024_atlas__CTCL14      394             0.081         True          li2024         MF    394
            D6__NOR4       63             0.079         True    gaydosik2019         HC     63
  Li2024_atlas__MF26     2053             0.069         True          li2024         MF   2053
             D3__P87      494             0.069         True     brunner2024 CTCL_other    494
  Li2024_atlas__MF17     5143             0.056         True          li2024         MF   5143
Li2024_atlas__CTCL18     8122             0.043         True          li2024         MF   8122
  Li2024_atlas__MF15     1248             0.020         True          li2024         MF   1248

34 dark donors | 78,961 cells | predicted clonal 27,107 (34.3%)
```

```python  # cell 29
# Fig 3 — predicted burden per dark donor, next to the labelled donors' true burden for scale.
fig, ax = plt.subplots(figsize=(5, 3))
d = dark_tbl.sort_values("pred_clonal_frac", ascending=False)
_studies = list(dict.fromkeys(d["study"]))
_pal = dict(zip(_studies, ["#4c78a8", "#f58518", "#54a24b", "#c0392b", "#888", "#9467bd"] * 3))
ax.bar(np.arange(len(d)), d["pred_clonal_frac"], color=[_pal[s] for s in d["study"]], width=0.75)
ax.axhline(donor_tbl.loc[LABELLED, "clonal_frac"].median(), ls="--", lw=0.8, c="k",
           label="median true burden, labelled donors")
ax.set_xticks(np.arange(len(d)))
ax.set_xticklabels(d["donor"], rotation=75, ha="right", fontsize=4.5)
ax.set_ylabel("predicted clonal fraction")
ax.set_title(f"Projected tumour burden in the TCR-less samples — {BEST_LATENT} (unvalidated)",
             fontsize=9)
handles, labels = ax.get_legend_handles_labels()
handles += [plt.Line2D([], [], lw=0, marker="s", color=_pal[s]) for s in _studies]
ax.legend(handles, labels + _studies, fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "clonality_dark_burden.png", dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

```python  # cell 30
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
CCR4, PHF19, COL6A3, LMNA, PLEC, RNF19A, VIM, ITM2A, FAM107B, ABCC1, MCM7, KCNN4, SESN3, TCF7, APP, HDAC7, TFRC, HBS1L, MRC2, TRAF1, PTP4A3, TNFRSF4, TYMS, HACD1, EHBP1L1

17498
```

## Part 9 — persist

```python  # cell 32
# (a) held-out fold scores/calls, every arm x rule; (b) the applied call on the dark donors.
fold_out = percell.copy()
fold_out.index.name = "obs_name"
fold_out.to_parquet(FOLD_PARQUET)
print("wrote", FOLD_PARQUET, fold_out.shape)

trans = dark_percell.copy()
trans["study"] = slim.obs["study"].astype(str).reindex(trans.index)
trans["disease"] = slim.obs["disease"].astype(str).reindex(trans.index)
trans["model"] = f"{BEST_LATENT}_alldonor/{rule_win}"
trans.index.name = "obs_name"
trans.to_parquet(TRANS_PARQUET)
print("wrote", TRANS_PARQUET, trans.shape)

dark_tbl.to_csv(OUT_DIR / "dark_donor_predictions.csv", index=False)
print("wrote", OUT_DIR / "dark_donor_predictions.csv")
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/semantic_clonality_folds_v1.parquet (3926380, 8)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/semantic_clonality_transfer_v1.parquet (78961, 6)
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/benchmark_results/clonality_transfer/dark_donor_predictions.csv
```

### Reading the results

- **Headline** = the winning arm/rule in `transfer_summary.csv`: `auc_raw` vs ALICE on donors the
  encoder never saw. That is the number that licenses (or does not license) Part 8.
- **`auc_raw` of each arm's `delta_centroid_null`** should sit at ~0.5. If one does not, that arm's
  fold/label wiring is leaking — the result is a bug, not a finding.
- **`semantic` − `ldvae`** is the semantic prior's contribution, isolated: the two differ by
  `coherence_weight` alone. **`ldvae` − `scvi`** is the linear positive-weight decoder's
  contribution, though scvi also loses the decorrelation penalty (see the header).
- Rank by `auc_raw` and `balanced_acc`, **not `f1`**: most held-out donors are majority-clonal, so
  an all-positive call scores F1 ≈ 0.85 while being uninformative.
- `centroid_r2` answers the perturbation-trial question directly — does `μ̂_ben + δ̂` land on the real
  clonal centroid of an unseen donor — but note it stays high even for the shuffled-δ null, because
  `μ̂_mal ≈ μ̂_ben` and the donor offset dominates. Compare it across arms, never against 1.0.
