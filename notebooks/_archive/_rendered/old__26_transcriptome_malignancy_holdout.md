# old/26_transcriptome_malignancy_holdout.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/26_transcriptome_malignancy_holdout.ipynb` and is never run._

# 26 — Transcriptome-only malignancy calling, tested on a held-out study

`23_malignancy_tcr_cnv` gives two *evidence-based* channels, and both run out of data: 7 of the 14 v2 skin cohorts
carry no V(D)J at all, and the CNV caller abstains on any donor without a clone-level event. The
question here is whether the **transcriptome alone** carries the malignant CD4 signal well enough to
cover those cohorts — and whether it survives compression to a ~1000-gene panel, i.e. whether the
call would transfer to Xenium.

`old/16_mrvi_labelspreading_malignancy.ipynb` prototyped this on the v1 cohort and reported 0.943
AUC — but grouped by **donor**, which overstates transfer badly: a held-out donor's neighbours in
the graph still come from its own study, same chemistry, same site, often the same processing batch.
This notebook redoes it on the v2 atlas against the `23_malignancy_tcr_cnv` TCR baseline with the honest split —
**leave one whole study out** — and three arms:

| arm | representation | question it answers |
|---|---|---|
| `mrvi10k` | MrVI `u`, 10,000 HVG | does the latent transfer across studies at all? |
| `leiden_free` | per-study Leiden + CTCL marker score, **no labels at all** | does "cluster and call the tumour cluster" work with nothing to learn from? |
| `mrvi1k` | MrVI `u`, 1,000-gene panel | does a Xenium-sized panel keep the signal? |

Anchors come from TCR clonality only (`tcr_clonal`), so the CNV channel stays untouched and
available as independent confirmation later.

The middle arm is deliberately label-free. An earlier version scored a held-out cell by the
malignant fraction of its Leiden cluster *among training anchors*; its shuffled-label null spanned
0.197–0.961 (mean ≈ 0.76), so its 0.96 said nothing. `leiden_free` replaces it: cluster the
held-out study on its own cells, then rank clusters by a directional CTCL marker score. It reads no
anchor, no other study and no label — the anchors only score it afterwards — which is also why it
is the one arm that would work on a cohort with no repertoire whatsoever.

> **HEAVY** cells (object load, panel build, MrVI training, the label-spreading folds) run
> on the GPU/compute kernel — not the login node. `skin_T_annotated.h5ad` is 12 GB.

```python  # cell 1
# ============================================================
# Parameters — inputs, cohort, study roles, arms.
# ============================================================
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
OUT_DIR = NB_DIR / "data" / "atlas_joint"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = NB_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)

# ---- inputs ----
# Read nb30's TCR-COMPLETE cache, not the plain nb10b object. nb30 Step 1 folds the Li2024 V(D)J
# in and rewrites `has_tcr` for that study: 0.074 -> 0.491, i.e. 114,752 TCR+ li2024 cells instead
# of 17,274. li2024 is 43% of the atlas and one of the two panel-design studies, so reading the
# plain object would silently discard ~97k benign anchors and design the panel on a crippled study.
TCR_OBJ   = OUT_DIR / "skin_T_tcr_annotated_v4.h5ad"   # nb30 Step 1 cache (12 GB), layer raw_counts
ALICE_MAL = OUT_DIR / "alice_malignancy_v4.parquet"    # nb30 TCR baseline: cell_id, tcr_clonal
COUNTS    = "raw_counts"                               # the cache renames nb10b's `counts` layer

SEED = 0
# nb30's clone rule, reused verbatim (Step 1). The benign anchors are defined by clone size, so
# they have to be computed on the SAME clone key the malignant anchors came from.
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2
CD4_TYPES = ["CD4"]   # nb30's TCR call is CD4-only, so CD4_Treg / CD8 would be *structural*
                      # negatives — they would inflate every AUROC without testing anything.

# ---- study roles ----
# The 1k panel's DE genes come only from PANEL_STUDIES, and those two are in the training set of
# every fold. That is what keeps gene selection from ever seeing a held-out study's labels, and it
# is also how a real panel is designed: once, on pilot data, then applied everywhere.
PANEL_STUDIES    = ["li2024", "brunner2024"]
EVAL_STUDIES     = ["chennareddy2025", "buus2025", "il4ra2026", "gaydosik2022"]
SHUFFLE_FOLD     = "chennareddy2025"   # the fold that also gets the shuffled-label negative control
MIN_FOLD_ANCHORS = 300                 # per class; a fold below this is dropped, loudly

# ---- arms ----
LS_KW      = dict(kernel="knn", n_neighbors=20, alpha=0.2, max_iter=60)   # as in old/nb16
MRVI_ARMS  = {"mrvi10k": "hvg10k", "mrvi1k": "panel1k"}                   # arm -> latent tag
ARMS       = ["mrvi10k", "leiden_free", "mrvi1k"]
CALL_THR   = 0.5     # MrVI arms only, fixed and never tuned — see Step 4
# Shuffled-label control, MrVI arms only: one full LabelSpreading refit per draw. `leiden_free`
# never reads a training label, so permuting them is a no-op for it — its null is a set of
# expression-matched RANDOM marker sets instead (LF_N_CTRL draws), which is the question that
# actually matters for it: would any gene set of this size rank the clusters this well?
N_SHUFFLE  = {"mrvi10k": 1, "mrvi1k": 1}

# ---- leiden_free arm ----
# Directional CTCL marker sets, fixed a priori. The sign is NEVER flipped to whichever direction
# scores better — that would be label peeking. TOX / KIR3DL2 / TRAF1 / CD7 / DPP4 are the members
# shared with subclone_helpers.PAPER_PANELS["signaling"]; that panel mixes gains and losses in one
# list, so it cannot serve as a directional score, but the overlap keeps nb43 consistent with it.
CTCL_UP   = ["TOX", "GATA3", "PLS3", "TWIST1", "KIR3DL2", "TNFRSF8", "CDK6", "TRAF1", "DNM3"]
CTCL_DOWN = ["CD7", "DPP4", "SATB1", "STAT4"]
LF_HVG, LF_PCS, LF_RES = 2000, 50, 0.2   # per-study clustering; one study, so no batch_key
LF_N_CTRL = 5                            # random-marker-set control draws per fold
# The CALL is the LARGEST GAP between cluster median scores: sort the clusters by their median
# directional marker score, split at the widest gap, everything above it is malignant.
#
# The gap has to be MASS-GUARDED — only gaps leaving at least LF_MIN_SIDE of the cells on each
# side are admissible. Unguarded it degenerated on 3 of the 4 folds at this resolution: the widest
# gap is at the BOTTOM of the ranking, where one small outlier cluster sits far below the rest, so
# it peels that cluster off and leaves the real benign population on the malignant side —
# specificity 0.007 / 0.003 / 0.001, recall ~1.0, balanced_acc 0.50, "everything is tumour".
#
# The guard is not a tuned parameter: measured on all four folds, 10% / 15% / 20% / 30% pick the
# IDENTICAL cut everywhere (balanced_acc 0.846-0.983), and only 5% is too permissive to fix
# gaydosik2022. The value below sits in the middle of that plateau, not on an edge.
LF_MIN_SIDE = 0.15                       # min share of cells on each side of an admissible gap
LF_DIR    = OUT_DIR / "cd4_leiden_free"  # one parquet per study AND resolution: cl + scores

N_HVG_FULL, N_PANEL, N_PANEL_HVG = 10000, 1000, 700
# The 10k gene space is READ, not recomputed: run_mrvi_joint.py already ran seurat_v3
# (batch_key="study", n_top_genes=10000, on raw counts) over the 1.35M-cell skin atlas and cached
# the result as this file's `var`. Recomputing it costs 14 per-batch loess fits for no new
# information, and this is the gene space the atlas's own MrVI latents were trained on.
MRVI_HVG_H5 = OUT_DIR / "joint_mrvi_input_skin.h5ad"
UMAP_N = 150_000

# ---- outputs ----
# Caches. The kNN graph is the ~5-minute step LabelSpreading would otherwise rebuild inside every
# fit, and the probabilities are keyed by a hash of the label vector, so a re-run of Step 4 costs
# an np.load while a change of anchors upstream invalidates itself.
MRVI_LS_DIR = OUT_DIR / "cd4_mrvi_ls"                # LabelSpreading probabilities, one .npy/fit
KNN_DIR     = OUT_DIR / "cd4_knn"                    # kNN affinity per arm, one .npz
MRVI_INPUT  = OUT_DIR / "cd4_mrvi_input.h5ad"        # slim job input (union of both gene lists)
HVG_CSV     = TAB_DIR / "cd4_hvg_10000.csv"
PANEL_CSV   = TAB_DIR / "cd4_panel_1000.csv"
LOSO_CSV    = TAB_DIR / "transcriptome_malignancy_loso.csv"
OUT_PARQUET = OUT_DIR / "cd4_transcriptome_malignancy_v1.parquet"
UMAP_NPZ    = OUT_DIR / "cd4_umap_mrvi10k.npz"
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 2
import gc
import hashlib
import os
import re
import subprocess

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelSpreading

import sys
sys.path.insert(0, str(NB_DIR / "helpers"))
import atlas_join_helpers as H        # clone_id_from_cdr3, used by recompute_dominant_clone
import semantic_malig_helpers as SM   # binary_scores: call metrics vs the TCR baseline
import skin_T_cnv_helpers as C        # recompute_dominant_clone — nb30's clone key, verbatim
import subclone_helpers as SUB        # present_panels: filter marker sets to genes in var_names

np.random.seed(SEED)
sc.settings.verbosity = 1


def mem(tag=""):
    """Peak RSS so far. Step 2 is the memory high-water mark; print it after each block so an
    OOM is diagnosable instead of a guess."""
    import resource
    print(f"[mem] {tag:<22} peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

## Step 1 — CD4 cohort + TCR anchors · HEAVY (compute kernel)

Reads **`skin_T_tcr_annotated_v4.h5ad`** — `23_malignancy_tcr_cnv`'s own Step 1 cache — not the plain `21_reannotation` object.
`23_malignancy_tcr_cnv` folds the Li2024 V(D)J into `has_tcr` there, taking li2024 from 7.4 % to 49.1 % TCR coverage;
on the plain object this notebook would lose ~97k of li2024's benign anchors and design the 1k panel
on a crippled version of the study. Backed read → CD4 mask → `to_memory`, so the 12 GB object is
never fully materialised.

`tcr_is_expanded` is recomputed here with `23_malignancy_tcr_cnv`'s own `recompute_dominant_clone`, on the unified
TRB-primary clone key. The atlas ships a `clone_size` built from each study's original clone key —
using it would define the benign anchors on a different clone definition than the one the ALICE
malignant anchors came from.

`alice_malignancy_v4.parquet` has 492,653 of the object's 539,916 rows. The missing 47,263 are
**gated-out donors** (`MF_gamma_delta`, `CD8_aggressive_epidermotropic_CTCL`, donors under 200 T
cells, the `D1__P303` duplicate) — they are dropped here, never filled `False`.

Anchors follow old/`old/16_mrvi_labelspreading_malignancy`, on the v2 columns:

```
malignant = tcr_clonal                                   # ALICE founder family, `23_malignancy_tcr_cnv` Step 3
benign    = has_tcr & ~malignant & ~tcr_is_expanded       # TCR+, non-expanded (`23_malignancy_tcr_cnv` clone key)
unlabeled = everything else                              # no TCR is NOT a negative
```

```python  # cell 4
alice = pd.read_parquet(ALICE_MAL).set_index("cell_id")["tcr_clonal"].astype(bool)
print("alice baseline:", len(alice), "cells |", int(alice.sum()), "clonal")

assert TCR_OBJ.exists(), (
    f"{TCR_OBJ.name} is missing — run nb30 Step 1 first. It is the object that carries the Li2024 "
    "V(D)J fold-in; the plain skin_T_annotated.h5ad reports li2024 has_tcr = 0.074 instead of 0.491")
a = sc.read_h5ad(TCR_OBJ, backed="r")
a.obs["cell_type_T2"] = a.obs["cell_type_T"].astype(str)
is_cd4 = a.obs["cell_type_T2"].isin(CD4_TYPES).to_numpy()
is_li = a.obs["study"].astype(str).eq("li2024").to_numpy()

# nb30 cell 4, verbatim: rebuild the unified TRB-primary clone key over ALL TCR+ T cells (clone
# sizes use every lineage; only the dominance test narrows to CD4). The atlas ships its own
# `clone_size` built from each study's original key — using that here would define the benign
# anchors on a different clone definition than the ALICE malignant anchors came from.
C.recompute_dominant_clone(a, H, is_li, FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN, dom_mask=is_cd4)

keep = is_cd4 & a.obs_names.isin(alice.index)   # Index.isin already returns an ndarray
print(f"{a.n_obs} T -> {int(keep.sum())} CD4 in the gated cohort "
      f"({int((~a.obs_names.isin(alice.index)).sum())} cells belong to gated-out donors)")
adata = a[keep].to_memory()
a.file.close(); del a; gc.collect()

adata.obs["tcr_clonal"] = alice.reindex(adata.obs_names).to_numpy()
mal = adata.obs["tcr_clonal"].to_numpy()
ben = (adata.obs["has_tcr"].to_numpy() & ~mal
       & ~adata.obs["tcr_is_expanded"].to_numpy())
adata.obs["tcr_label"] = pd.Categorical(
    np.where(mal, "malignant", np.where(ben, "benign", "unlabeled")),
    categories=["malignant", "benign", "unlabeled"])

STUDY  = adata.obs["study"].astype(str)
ANCHOR = adata.obs["tcr_label"].isin(["malignant", "benign"]).to_numpy()
Y      = adata.obs["tcr_label"].eq("malignant").to_numpy()

# ---- empty-X cells ----
# gaydosik2023 is effectively obs-only in this object: 1083 of its 1147 T cells — and ALL 634 of
# its gated CD4 cells — have a completely empty X row, even though obs.total_counts averages ~3.5k,
# i.e. the atlas join lost that study's matrix. Every other study has min nnz > 0. Those rows have
# to be excluded explicitly: a batch of all-zero rows leaves scanpy's per-batch HVG with zero
# expressed genes and it dies in `pd.cut` ("Cannot cut empty array"), and any score assigned to a
# zero row is noise anyway. Not dropped from the cohort — the two MrVI latents were trained on all
# n_obs cells and `load_latent` checks that alignment — but masked out of clustering and calls.
_nnz = (np.diff(adata.X.indptr) if hasattr(adata.X, "indptr")
        else (adata.X != 0).sum(axis=1))
adata.obs["has_expr"] = np.asarray(_nnz).ravel() > 0
EXPR = adata.obs["has_expr"].to_numpy()
if not EXPR.all():
    _e = adata.obs.loc[~EXPR, "study"].astype(str).value_counts()
    print(f"\nempty-X cells: {int((~EXPR).sum())} — excluded from clustering and from the "
          f"reported calls\n{_e[_e > 0].to_string()}")
# an empty anchor would be a zero row with a label on it, i.e. pure noise fed into training
assert not (ANCHOR & ~EXPR).any(), f"{int((ANCHOR & ~EXPR).sum())} anchors have an empty X row"

ct = pd.crosstab(STUDY, adata.obs["tcr_label"])
print("\n", ct.to_string())
TCR_STUDIES    = sorted(ct.index[ct[["malignant", "benign"]].sum(axis=1) > 0])
NO_TCR_STUDIES = sorted(set(ct.index) - set(TCR_STUDIES))
print(f"\nTCR studies ({len(TCR_STUDIES)}): {TCR_STUDIES}"
      f"\nno-V(D)J studies ({len(NO_TCR_STUDIES)}): {NO_TCR_STUDIES}")

# every no-V(D)J study must be 100% unlabeled — if not, a label leaked in from somewhere
for s in NO_TCR_STUDIES:
    assert ct.loc[s, ["malignant", "benign"]].sum() == 0, f"{s} has anchors but no V(D)J"
assert set(PANEL_STUDIES) <= set(TCR_STUDIES), f"panel studies need labels: {PANEL_STUDIES}"

# a fold with too few anchors of either class cannot be scored; drop it loudly, never silently
FOLDS = []
for s in EVAL_STUDIES:
    n_m, n_b = int(ct.loc[s, "malignant"]), int(ct.loc[s, "benign"])
    if min(n_m, n_b) < MIN_FOLD_ANCHORS:
        print(f"DROPPED fold {s}: malignant={n_m} benign={n_b} < {MIN_FOLD_ANCHORS}")
    else:
        FOLDS.append(s)
print("evaluation folds:", FOLDS)
adata
```

```
alice baseline: 492653 cells | 81963 clonal
```

## Step 2 — the 1000-gene panel, and the slim MrVI input · HEAVY

Two gene lists, both written to `tables/`:

- **`cd4_hvg_10000.csv`** — the 10,000 `seurat_v3` HVGs (`batch_key="study"`, raw counts) that
  `run_mrvi_joint.py` already selected over the 1.35M-cell skin atlas, read out of
  `joint_mrvi_input_skin.h5ad`'s `var`. Unsupervised, so no label enters it — and since it is the
  gene space the atlas's own MrVI latents were trained on, `mrvi10k` is the canonical arm rather
  than a private variant. (`old/17_mrvi_pseudosample_malignancy` pulled its HVGs the same way.) Recomputing it would mean 14
  per-batch loess fits for no new information.
- **`cd4_panel_1000.csv`** — the Xenium-sized panel: the top 700 of those HVGs by
  `highly_variable_rank`, plus 300 genes differential between malignant and benign anchors
  **computed on `li2024 + brunner2024` only**. The DE runs inside the 10k space: a panel gene has
  to be robustly detected to be worth an probe, and it keeps the whole step 4x smaller.
  Those two studies sit in every fold's training set, so no held-out study's labels reach the panel.

`TR[ABGD][VDJC]` and `IG[HKL][VDJC]` genes are excluded from the DE half: they track the clonotype
itself rather than the tumour program, so keeping them would smuggle the label into the gene space
(and they are not on a real Xenium panel either).

The slim input carries the **union** of both lists, so the two training jobs read one small file.

Split into two cells, both **cached by their output file**: the gene lists and the slim input are
each skipped if already on disk, so a crash in one does not cost the other. Every intermediate here
is built counts-only or lognorm-only rather than as a full `adata` copy — holding `X` *and*
`raw_counts` *and* a batch slice *and* a column slice at once is ~25 GB on this cohort, which is
what killed the first attempt.

```python  # cell 6
if HVG_CSV.exists() and PANEL_CSV.exists():
    HVG10K = pd.read_csv(HVG_CSV)["gene"].astype(str).tolist()
    PANEL = pd.read_csv(PANEL_CSV)["gene"].astype(str).tolist()
    print(f"loaded cached gene lists: {len(HVG10K)} HVG, {len(PANEL)} panel")
else:
    # ---- the 10k gene space: read the cached selection, do not recompute it ----
    with h5py.File(MRVI_HVG_H5, "r") as _f:
        _v = _f["var"]
        _k = _v.attrs.get("_index", "_index")
        _hv = pd.Series(_v["highly_variable_rank"][:],
                        index=[x.decode() for x in _v[_k][:]])
    HVG10K = _hv.sort_values().index.tolist()
    assert len(HVG10K) == N_HVG_FULL, f"expected {N_HVG_FULL} cached HVGs, got {len(HVG10K)}"
    _absent = [g for g in HVG10K if g not in set(adata.var_names)]
    assert not _absent, f"{len(_absent)} cached HVGs absent from this object: {_absent[:5]}"
    print(f"HVG: {len(HVG10K)} genes read from {MRVI_HVG_H5.name} "
          "(seurat_v3, batch_key=study, n_top_genes=10000)")
    mem("after HVG read")

    # ---- DE half: malignant vs benign anchors, PANEL_STUDIES only, lognorm only ----
    _pm = np.flatnonzero(STUDY.isin(PANEL_STUDIES).to_numpy() & ANCHOR)
    _gi = adata.var_names.get_indexer(HVG10K)
    # row-slice first, then column-slice the much smaller block
    sub = ad.AnnData(X=adata.X[_pm][:, _gi],
                     obs=pd.DataFrame({"grp": adata.obs["tcr_label"].astype(str).to_numpy()[_pm]},
                                      index=adata.obs_names[_pm]),
                     var=pd.DataFrame(index=pd.Index(HVG10K)))
    print("panel-design cells:", dict(sub.obs["grp"].value_counts()))
    sc.tl.rank_genes_groups(sub, "grp", groups=["malignant"], reference="benign",
                            method="wilcoxon")
    de = sc.get.rank_genes_groups_df(sub, "malignant")
    del sub; gc.collect(); mem("after DE")

    _CLONO = re.compile(r"^(TR[ABGD][VDJC]|IG[HKL][VDJC])")
    de = de[~de["names"].astype(str).str.match(_CLONO)]
    de = de.reindex(de["scores"].abs().sort_values(ascending=False).index)

    PANEL = HVG10K[:N_PANEL_HVG]
    PANEL += [g for g in de["names"].astype(str)
              if g not in set(PANEL)][:N_PANEL - N_PANEL_HVG]
    PANEL += [g for g in HVG10K if g not in set(PANEL)][:N_PANEL - len(PANEL)]   # top up to 1000
    assert len(PANEL) == N_PANEL == len(set(PANEL)), (len(PANEL), len(set(PANEL)))
    assert set(PANEL) <= set(HVG10K), "panel must live inside the 10k gene space"
    print(f"panel: {N_PANEL_HVG} top-rank HVG + {N_PANEL - N_PANEL_HVG} DE/top-up "
          f"= {len(PANEL)} genes")
    print("top DE genes kept:", de["names"].head(15).tolist())

    pd.DataFrame({"gene": HVG10K}).to_csv(HVG_CSV, index=False)
    pd.DataFrame({"gene": PANEL}).to_csv(PANEL_CSV, index=False)
    print("wrote", HVG_CSV.name, "and", PANEL_CSV.name)
```

```
loaded cached gene lists: 10000 HVG, 1000 panel
```

```python  # cell 7
# ---- slim MrVI job input: the 10k gene space, raw counts in X (no duplicate `counts` layer —
# that would double the file for nothing; run_mrvi_cd4_panel.py reads X when the layer is absent).
# Both training arms subset their genes out of this one file. ----
UNION = HVG10K                       # PANEL is a subset of HVG10K, asserted above
if MRVI_INPUT.exists():
    print("have", MRVI_INPUT.name)
else:
    gi = adata.var_names.get_indexer(UNION)
    assert (gi >= 0).all(), f"{int((gi < 0).sum())} panel/HVG genes are not in adata.var_names"
    slim = ad.AnnData(X=adata.layers[COUNTS][:, gi],
                      obs=adata.obs[["sample_id", "study", "donor", "cell_type_T",
                                     "tcr_label"]].copy(),
                      var=pd.DataFrame(index=pd.Index(UNION)))
    for c in ("sample_id", "study", "donor"):
        slim.obs[c] = slim.obs[c].astype(str).astype("category")
    # write via .tmp + rename: an OOM part-way through a direct write would leave a truncated
    # h5ad that the `exists()` guard above would then happily skip over
    _tmp = MRVI_INPUT.with_suffix(".tmp.h5ad")
    slim.write_h5ad(_tmp)
    _tmp.replace(MRVI_INPUT)
    print("wrote", MRVI_INPUT, slim.shape)
    del slim; gc.collect()

# The counts layer is not read again — Steps 4-6 use adata.X (Leiden), adata.obs and the job
# latents. Dropping it here frees ~6 GB for the rest of the notebook; re-running Step 2 is safe
# because both cells above are guarded by their output files.
if COUNTS in adata.layers:
    del adata.layers[COUNTS]
    gc.collect()
mem("end of Step 2")
```

```
have cd4_mrvi_input.h5ad
[mem] end of Step 2          peak RSS 18.4 GB
```

## Step 3 — train the two MrVI models · HEAVY (bsub, 1 GPU each)

Two runs of `jobs/run_mrvi_cd4_panel.py`: same cells, same `sample_key="sample_id"` /
`batch_key="study"`, same hyper-parameters, **only the gene list differs**. That is what makes the
10k-vs-1k gap attributable to the panel and nothing else.

MrVI is unsupervised, so fitting it on every cell — including the held-out study's — leaks no
malignancy label. This is the design's load-bearing assumption: it is what lets the latent be
trained once and evaluated leave-one-study-out.

Re-run this cell to poll; it submits only what is missing and Step 4 refuses to start until both
latents exist.

```python  # cell 9
JOBS = [("hvg10k", HVG_CSV), ("panel1k", PANEL_CSV)]
for tag, csv in JOBS:
    u = OUT_DIR / f"cd4_X_mrvi_u_{tag}.npy"
    if u.exists():
        print(f"[{tag}] have {u.name}")
        continue
    subprocess.run(["bash", "run_mrvi_cd4_panel.sh",
                    "--gene-list", str(csv.relative_to(NB_DIR)), "--tag", tag],
                   cwd=NB_DIR / "jobs", env={**os.environ, "JOB_TAG": tag}, check=True)

print(subprocess.run(["bjobs", "-a", "-J", "mrvi_cd4_hvg10k"],
                     capture_output=True, text=True).stdout or "(no hvg10k job)")
print(subprocess.run(["bjobs", "-a", "-J", "mrvi_cd4_panel1k"],
                     capture_output=True, text=True).stdout or "(no panel1k job)")
for tag, _ in JOBS:
    log = NB_DIR / "jobs" / f"run_mrvi_cd4_{tag}.bsub.log"
    if log.exists():
        print(f"\n--- {log.name} (tail) ---")
        print(subprocess.run(["tail", "-6", str(log)], capture_output=True, text=True).stdout)
```

```
[hvg10k] have cd4_X_mrvi_u_hvg10k.npy
[panel1k] have cd4_X_mrvi_u_panel1k.npy
(no hvg10k job)
(no panel1k job)

--- run_mrvi_cd4_hvg10k.bsub.log (tail) ---
    Max Threads :                                28
    Run time :                                   2794 sec.
    Turnaround time :                            2844 sec.

The output (if any) is above this job summary.



--- run_mrvi_cd4_panel1k.bsub.log (tail) ---
    Max Threads :                                28
    Run time :                                   2116 sec.
    Turnaround time :                            2166 sec.

The output (if any) is above this job summary.
```

## Step 4 — leave-one-study-out · HEAVY

Two kinds of arm, evaluated the same way but built very differently.

**MrVI arms (`mrvi10k`, `mrvi1k`).** `y = -1` everywhere except the **training** anchors — the
held-out study's anchors are masked back to unlabeled alongside the genuinely unlabelled cells —
then `LabelSpreading` runs over the full CD4 latent and is scored on the held-out anchors only.
The call is `prob >= CALL_THR = 0.5`, fixed and **never tuned**: with `alpha=0.2` the in-sample
anchors clamp to ~0/1 so a train-side F1-optimal cut is meaningless, and tuning on the held-out
fold would be the exact leak this notebook exists to avoid. Their negative control is the
`SHUFFLE_FOLD` row with training labels permuted.

**`leiden_free` — the crude baseline.** No labels anywhere, four steps. For each held-out study,
on that study's cells alone: HVG → scale → PCA → neighbours → **Leiden at `LF_RES = 0.2`** (no
`batch_key`; it is one study), then every cell gets a directional CTCL marker score via
`sc.tl.score_genes` on the **full** gene space (`CTCL_UP` minus `CTCL_DOWN`), then each cluster
takes the median of that score, and finally the **largest mass-guarded gap** between those
cluster medians splits the clusters into malignant and benign. That is the whole arm: cluster
coarsely, rank the clusters by markers, cut where the ranking breaks. There is no threshold and
no fitted parameter, and the marker direction is fixed a priori and never flipped to whichever
sign scores better — so nothing about the call can have been chosen on the fold it is scored
against.

**The mass guard is the load-bearing part**, and it is there because the unguarded rule failed
here, measurably. At `LF_RES = 0.2` the widest gap sits at the *bottom* of the cluster ranking on
3 of the 4 folds: one small outlier cluster lies far below the rest, the gap peels it off, and the
genuinely benign population stays on the malignant side. That reads as specificity **0.007**
(chennareddy2025), **0.003** (il4ra2026) and **0.001** (gaydosik2022) at recall ≈ 1.0 — the arm
degenerates into "everything is tumour" while its cluster ranking is in fact excellent (an oracle
that labels each cluster by its own majority truth reaches `balanced_acc` 0.984 on
chennareddy2025). The ranking was never the problem; only the cut was.

Requiring at least `LF_MIN_SIDE` of the cells on each side of the split moves the cut to the next
admissible gap — the one between the benign and malignant blocks — and recovers `balanced_acc`
0.846–0.983 on all four folds. The guard value is not tuned: 10 %, 15 %, 20 % and 30 % pick the
**identical** cut on every fold, and only 5 % is too permissive to fix gaydosik2022, so 0.15 sits
in the middle of a plateau rather than on a peak. `called_frac` and `specificity` in the table
below are the columns that would show the degeneration coming back.

Permuting training labels is a no-op for `leiden_free`, so its null is `LF_N_CTRL` **random marker
sets** drawn from the same expression deciles as the real ones and pushed through the same
clustering. `f1_cell` applies a zero cut to the raw per-cell marker score instead of the cluster
median, which separates "the markers work" from "the clustering plus the cut work".

**Everything expensive is cached on disk**, so a second run of this cell is seconds:
`cd4_knn/cd4_knn_{tag}_k20.npz` (the kNN graph — the ~5-minute step, rebuilt inside every
`LabelSpreading.fit` if you let it), `cd4_mrvi_ls/{arm}__{key}__{hash}.npy` (one file per
LabelSpreading fit, keyed by a hash of the label vector so new anchors miss automatically) and
`cd4_leiden_free/{study}_res0.2.parquet` (clusters + marker scores per study and resolution).

### Reading the numbers — F1, and its floor

**F1 is the headline metric**, reported next to precision and recall, so the threshold is
load-bearing (it was not, when this was scored by AUROC). Neither cut is fitted to any label: 0.5
for the MrVI arms, the largest cluster-score gap for `leiden_free`. The reported F1 is therefore a
*floor*, not the best achievable.

The anchors are 68–85 % malignant, and F1 inherits that: a caller that says "malignant" to
**everything** scores `2p/(1+p)` — 0.90 on chennareddy2025, 0.92 on gaydosik2022. Every row and
every bar carries that number as `f1_all_positive`, and an F1 at or below it means the arm has
added nothing to a constant call. `balanced_acc`, `precision` and `recall` are reported next to it
because they do not inherit the floor: a 100 %-recall / prevalence-precision caller sits at
`balanced_acc = 0.5` no matter how good its F1 looks.

This is also what to check on the controls. A shuffled-label or random-marker row landing at
F1 ≈ 0.90 is *not* evidence of leakage — that is the all-positive floor. Leakage shows up as
`balanced_acc` well above 0.5, or as `f1` meaningfully above `f1_all_positive`.


```python  # cell 11
# ============================================================
# Arms. `mrvi10k` / `mrvi1k` spread the training anchors over a MrVI latent; `leiden_free` reads
# no label at all and is computed per study from scratch.
# ============================================================
def load_latent(tag):
    """Reindex a job latent onto adata.obs_names; a stale .npy fails here, not silently."""
    u = np.load(OUT_DIR / f"cd4_X_mrvi_u_{tag}.npy")
    bc = np.load(OUT_DIR / f"cd4_mrvi_barcodes_{tag}.npy", allow_pickle=True).astype(str)
    assert len(bc) == u.shape[0] == adata.n_obs, (len(bc), u.shape, adata.n_obs)
    idx = pd.Index(bc).get_indexer(adata.obs_names)
    assert (idx >= 0).all(), f"{int((idx < 0).sum())} cells missing from the {tag} latent"
    return u[idx]


REP = {arm: load_latent(tag) for arm, tag in MRVI_ARMS.items()}
print({k: v.shape for k, v in REP.items()})

# `LabelSpreading(kernel="knn")` rebuilds the whole kNN graph inside every single `.fit()`, and
# that one step IS the fit: measured on this machine at n=150k, the graph takes 83 s while the
# laplacian and all 60 spreading iterations together take 0.4 s. At 400k cells it is ~5 min, and
# Step 4 + Step 5 fit 12 times over the SAME two latents — an hour of rebuilding one matrix. So
# build it once per arm, CACHE IT ON DISK, and hand it over as a callable kernel; the matrix is
# exactly what the estimator would have built for itself (connectivity, k = n_neighbors, each
# point counting itself). n_jobs is deliberately left at its default: joblib parallelism measured
# 25x SLOWER here (2091 s vs 83 s at n=150k), so passing n_jobs=-1 is a large pessimisation.
_AFF = {}


def affinity(arm):
    if arm not in _AFF:
        f = KNN_DIR / f"cd4_knn_{MRVI_ARMS[arm]}_k{LS_KW['n_neighbors']}.npz"
        if f.exists():
            A = sparse.load_npz(f)
            assert A.shape[0] == REP[arm].shape[0], (A.shape, REP[arm].shape)
            print(f"[{arm}] loaded cached kNN graph {f.name} nnz={A.nnz}")
        else:
            A = kneighbors_graph(REP[arm], LS_KW["n_neighbors"], mode="connectivity",
                                 include_self=True)
            KNN_DIR.mkdir(exist_ok=True)
            sparse.save_npz(f, A)
            print(f"built {arm} kNN graph: {A.shape} nnz={A.nnz} -> {f.name}")
        _AFF[arm] = A
    return _AFF[arm]


def spread(arm, y):
    """LabelSpreading over a MrVI latent. y in {1, 0, -1}, -1 = unlabeled."""
    kw = {k: v for k, v in LS_KW.items() if k not in ("kernel", "n_neighbors")}
    ls = LabelSpreading(kernel=lambda X, Y=None, _A=affinity(arm): _A, **kw).fit(REP[arm], y)
    return ls.label_distributions_[:, list(ls.classes_).index(1)]   # never [:, 1]


def _yhash(y):
    """Fingerprint of a label vector. y is in {1, 0, -1}, so int8 is lossless."""
    return hashlib.sha1(np.ascontiguousarray(y, dtype=np.int8).tobytes()).hexdigest()[:10]


def spread_cached(arm, y, key):
    """`spread()`, memoised on disk — a re-run of this notebook costs an np.load, not an hour.

    `key` only makes the filename readable; the HASH of the label vector is what makes the cache
    valid. Change the anchors upstream (a new alice_malignancy parquet, a different cohort gate,
    a different fold) and every affected file misses automatically instead of serving a stale
    score under the right-looking name.
    """
    MRVI_LS_DIR.mkdir(exist_ok=True)
    f = MRVI_LS_DIR / f"{arm}__{key}__{_yhash(y)}.npy"
    if f.exists():
        s = np.load(f)
        assert s.shape[0] == adata.n_obs, (s.shape, adata.n_obs)
        print(f"[{arm}] loaded cached probabilities {f.name}")
        return s
    s = spread(arm, y)
    np.save(f, s)
    return s


# ---------------------------------------------------------------- leiden_free
LF_DIR.mkdir(exist_ok=True)
_LF = {}


def _lf_embed(held):
    """Cluster ONE study on its own cells and give every cell a directional CTCL marker score.

    The only argument is a study name: no anchor, no label and no other study can reach this
    function, which is what makes the arm evaluable leave-one-study-out with no leak to argue
    about. Also computes LF_N_CTRL random-marker-set scores over the same clustering — the null
    for this arm, since permuting training labels does nothing to something that never reads them.
    Cached per study AND per resolution as a parquet (cell_id, cl, mscore, ctrl0..).
    """
    if held in _LF:
        return _LF[held]
    f = LF_DIR / f"{held}_res{LF_RES:g}.parquet"
    if f.exists():
        _LF[held] = pd.read_parquet(f)
        print(f"[{held}] loaded cached {f.name} ({_LF[held].shape[0]} cells, "
              f"{_LF[held]['cl'].nunique()} clusters)")
        return _LF[held]

    idx = np.flatnonzero((STUDY == held).to_numpy() & EXPR)
    sub = ad.AnnData(X=adata.X[idx],
                     obs=adata.obs.iloc[idx][["study"]].copy(),
                     var=pd.DataFrame(index=adata.var_names))
    assert sub.obs["study"].nunique() == 1, "leiden_free must see exactly one study"

    # markers on the FULL gene space: score_genes matches control genes by expression bin, so the
    # score is centred at 0, and a marker can be informative without being one of the 2k HVGs.
    pan = SUB.present_panels(sub, {"up": CTCL_UP, "dn": CTCL_DOWN})
    for k, want in (("up", CTCL_UP), ("dn", CTCL_DOWN)):
        got = pan.get(k, [])
        assert got, f"none of the {k} markers are in var_names: {want}"
        miss = [g for g in want if g not in got]
        if miss:
            print(f"[{held}] {k}: {len(miss)}/{len(want)} marker(s) absent from the atlas -> {miss}")
        sc.tl.score_genes(sub, got, score_name=k, random_state=SEED)

    # expression-decile-matched random marker sets, same sizes as the real ones
    mu = pd.Series(np.asarray(sub.X.mean(axis=0)).ravel(), index=sub.var_names)
    dec = pd.qcut(mu.rank(method="first"), 10, labels=False)
    real = set(pan["up"]) | set(pan["dn"])
    rng = np.random.default_rng(SEED)
    for j in range(LF_N_CTRL):
        for k in ("up", "dn"):
            pick = []
            for g in pan[k]:
                cand = dec.index[(dec == dec[g]).to_numpy()].difference(real | set(pick))
                if len(cand):
                    pick.append(str(rng.choice(np.asarray(cand))))
            sc.tl.score_genes(sub, pick, score_name=f"_c{j}{k}", random_state=SEED)

    hv = sc.pp.highly_variable_genes(sub, n_top_genes=LF_HVG, inplace=False)   # one study: no batch
    lb = ad.AnnData(X=sub.X[:, np.asarray(hv["highly_variable"])].copy(),
                    obs=pd.DataFrame(index=sub.obs_names))
    sc.pp.scale(lb, max_value=10)
    sc.tl.pca(lb, n_comps=min(LF_PCS, min(lb.shape) - 1), random_state=SEED)
    sc.pp.neighbors(lb, random_state=SEED)
    sc.tl.leiden(lb, resolution=LF_RES, random_state=SEED, key_added="cl",
                 flavor="igraph", n_iterations=2, directed=False)

    out = pd.DataFrame({"cl": lb.obs["cl"].astype(str).to_numpy(),
                        "mscore": (sub.obs["up"] - sub.obs["dn"]).to_numpy()},
                       index=sub.obs_names)
    for j in range(LF_N_CTRL):
        out[f"ctrl{j}"] = (sub.obs[f"_c{j}up"] - sub.obs[f"_c{j}dn"]).to_numpy()
    out.index.name = "cell_id"
    out.to_parquet(f)
    print(f"[{held}] {out.shape[0]} cells, {out['cl'].nunique()} clusters, "
          f"mscore {out.mscore.min():.2f}..{out.mscore.max():.2f} -> wrote {f.name}")
    del sub, lb; gc.collect()
    _LF[held] = out
    return out


def largest_gap_cut(scores, sizes, min_side=None):
    """Split the sorted cluster scores at their widest gap, considering only gaps that leave at
    least `min_side` of the CELLS on each side; return the lower edge of the upper group.

    The mass guard is what makes the rule work at all. Unguarded, at LF_RES = 0.2, the widest gap
    is at the BOTTOM of the ranking on 3 of the 4 folds — one small outlier cluster sits far below
    the rest, the gap peels it off, and the genuinely benign population stays on the malignant
    side: specificity 0.007 (chennareddy2025), 0.003 (il4ra2026), 0.001 (gaydosik2022) at recall
    ~1.0, i.e. "everything is tumour". Guarding it moves the cut to the next-widest admissible
    gap, which is the one between the benign and the malignant blocks.

    The guard value is not fitted: 10% / 15% / 20% / 30% pick the identical cut on all four folds
    (balanced_acc 0.846-0.983); only 5% is too permissive to fix gaydosik2022. LF_MIN_SIDE sits in
    the middle of that plateau.

    inf (fewer than 2 clusters, or every gap too lopsided) means the arm abstains: nothing is
    called.
    """
    min_side = LF_MIN_SIDE if min_side is None else min_side
    s = pd.Series(scores, dtype=float).sort_values()
    v = s.to_numpy()
    if len(v) < 2:
        return np.inf
    w = pd.Series(sizes, dtype=float).reindex(s.index).to_numpy()
    frac = np.cumsum(w) / w.sum()                       # cell mass at or below each cluster
    ok = (frac[:-1] >= min_side) & (frac[:-1] <= 1 - min_side)
    if not ok.any():
        return np.inf
    return float(v[int(np.argmax(np.where(ok, np.diff(v), -np.inf))) + 1])


def leiden_free_score(held, col="mscore"):
    """(cluster score per cell, call, raw per-cell score, diagnostics).

    Per held-out study, on that study's cells alone: Leiden at LF_RES -> each cluster's median
    directional CTCL marker score -> the largest mass-guarded gap between those medians splits
    the clusters into malignant and benign. The marker direction is fixed a priori and is never flipped to
    whichever sign scores better, and no label reaches any step, so nothing about the call can be
    tuned on the fold it is scored against.

    Cells outside `held` are NaN / False — this arm is per study by construction.
    """
    lf = _lf_embed(held).reindex(adata.obs_names)
    g = lf.groupby("cl", observed=True)[col]
    cs, sz = g.median(), g.size()
    s = lf["cl"].map(cs).to_numpy(dtype=float)
    cut = largest_gap_cut(cs, sz)
    return s, s >= cut, lf[col].to_numpy(dtype=float), {
        "n_clusters": int(len(cs)), "n_mal_clusters": int((cs >= cut).sum()), "cut": cut}


def shuffled(y, train, i):
    """Same pipeline, training labels permuted: the leakage check for the MrVI arms."""
    yy = y.copy()
    yy[train] = np.random.default_rng(SEED + 100 + i).permutation(y[train])
    return yy


rows, OOF = [], {}
for held in FOLDS:
    ho = (STUDY == held).to_numpy()
    train, test = ANCHOR & ~ho, ANCHOR & ho
    assert not (train & ho).any(), f"{held} leaked into the training anchors"
    y = np.full(adata.n_obs, -1); y[train] = Y[train].astype(int)
    for arm in ARMS:
        if arm == "leiden_free":
            jobs = ([("real", "mscore", None)]
                    + [("ctrl_markers", f"ctrl{j}", None) for j in range(LF_N_CTRL)])
        else:
            jobs = [("real", y, f"loso_{held}")]
            if held == SHUFFLE_FOLD:
                jobs += [("shuffled", shuffled(y, train, i), f"shuf_{held}_{i}")
                         for i in range(N_SHUFFLE[arm])]
        for kind, arg, key in jobs:
            if arm == "leiden_free":
                s, call, cell, extra = leiden_free_score(held, col=arg)
            else:
                s = spread_cached(arm, arg, key)
                call, cell, extra = s >= CALL_THR, None, {}
            p = float(Y[test].mean())            # malignant prevalence among this fold's anchors
            r = {"arm": arm, "held_out_study": held, "labels": kind,
                 "n_anchors": int(test.sum()), "n_malignant": int((test & Y).sum()),
                 "n_benign": int((test & ~Y).sum()),
                 # F1 of the trivial "everything is malignant" caller. At 68-85% prevalence that
                 # is 0.81-0.92, so it is the floor this fold's F1 has to clear to mean anything.
                 "f1_all_positive": 2 * p / (1 + p),
                 "called_frac": float(call[ho & EXPR].mean())}
            r.update(extra)
            r.update({k: v for k, v in SM.binary_scores(Y[test], call[test]).items()
                      if k in ("f1", "precision", "recall", "balanced_acc",
                               "sensitivity", "specificity")})
            # the raw per-cell marker score at its own zero cut, instead of the cluster median:
            # separates "the markers work" from "the clustering plus the gap cut work"
            r["f1_cell"] = (np.nan if cell is None else
                            SM.binary_scores(Y[test], cell[test] >= 0.0)["f1"])
            rows.append(r)
            if kind == "real":
                OOF[(arm, held)] = s[test]
            tail = ("" if arm != "leiden_free" else
                    f" | gap cut {extra['cut']:.3f}: {extra['n_mal_clusters']}/"
                    f"{extra['n_clusters']} clusters, {r['called_frac']:.1%} of cells,"
                    f" per-cell F1={r['f1_cell']:.3f}")
            print(f"[{held}] {arm:<11} {kind:<12} P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} (all-pos {r['f1_all_positive']:.3f}) "
                  f"bal.acc={r['balanced_acc']:.3f} (n={r['n_anchors']}){tail}")

loso = pd.DataFrame(rows)
loso.to_csv(LOSO_CSV, index=False)
print("\nwrote", LOSO_CSV)
real = loso[loso.labels.eq("real")]
print("\nmedian over held-out studies — F1 has to clear f1_all_positive to mean anything:")
print(real.groupby("arm")[["precision", "recall", "f1", "f1_all_positive",
                           "balanced_acc", "specificity", "f1_cell"]]
      .median().round(3).to_string())
print("\nspread across the 4 folds — the axis on which the arms actually differ:")
print(real.groupby("arm")[["precision", "recall", "f1"]]
      .agg(["min", "max"]).round(3).to_string())
print("\nall-positive F1 floor per fold (prevalence-driven, nothing to do with the model):")
print(real.drop_duplicates("held_out_study").set_index("held_out_study")["f1_all_positive"]
      .round(3).to_string())
for kind, note in [("shuffled", f"training labels permuted, {SHUFFLE_FOLD} only"),
                   ("ctrl_markers", "expression-matched random marker sets, every fold")]:
    sub_ = loso[loso.labels.eq(kind)]
    if len(sub_):
        print(f"\nnull control — {kind} ({note}):")
        print(sub_.groupby("arm")[["f1", "balanced_acc", "f1_cell"]]
              .agg(["mean", "min", "max"]).round(3).to_string())
# A null row sitting at F1 ~ 0.9 is NOT leakage — that is the all-positive floor, which any
# degenerate caller reaches at this prevalence. Leakage is `balanced_acc` well above 0.5, or `f1`
# meaningfully above `f1_all_positive`. Same warning for `leiden_free`'s cluster-level F1: the
# score is constant within a cluster, so wherever the clustering already separates malignant from
# benign, any labelling of those clusters inherits the separation — on synthetic data a random
# marker set matched the real one at cluster level while its per-cell F1 collapsed. `f1_cell` is
# the column that discriminates.
loso[loso.labels.eq("real")]
```

```
{'mrvi10k': (401851, 10), 'mrvi1k': (401851, 10)}
[mrvi10k] loaded cached probabilities mrvi10k__loso_chennareddy2025__491204d372.npy
[chennareddy2025] mrvi10k     real         P=0.977 R=0.950 F1=0.963 (all-pos 0.897) bal.acc=0.926 (n=22250)
[mrvi10k] loaded cached probabilities mrvi10k__shuf_chennareddy2025_0__3bcca0c202.npy
[chennareddy2025] mrvi10k     shuffled     P=0.815 R=0.812 F1=0.814 (all-pos 0.897) bal.acc=0.505 (n=22250)
[chennareddy2025] loaded cached chennareddy2025_res0.2.parquet (31643 cells, 10 clusters)
[chennareddy2025] leiden_free real         P=0.993 R=0.994 F1=0.994 (all-pos 0.897) bal.acc=0.983 (n=22250) | gap cut 0.132: 8/10 clusters, 69.5% of cells, per-cell F1=0.901
[chennareddy2025] leiden_free ctrl_markers P=0.991 R=0.594 F1=0.742 (all-pos 0.897) bal.acc=0.785 (n=22250) | gap cut -0.397: 4/10 clusters, 42.1% of cells, per-cell F1=0.188
[chennareddy2025] leiden_free ctrl_markers P=0.994 R=0.593 F1=0.743 (all-pos 0.897) bal.acc=0.789 (n=22250) | gap cut 0.367: 4/10 clusters, 38.0% of cells, per-cell F1=0.856
[chennareddy2025] leiden_free ctrl_markers P=0.990 R=0.243 F1=0.390 (all-pos 0.897) bal.acc=0.616 (n=22250) | gap cut 0.356: 2/10 clusters, 16.9% of cells, per-cell F1=0.839
[chennareddy2025] leiden_free ctrl_markers P=0.990 R=0.835 F1=0.906 (all-pos 0.897) bal.acc=0.900 (n=22250) | gap cut 0.210: 7/10 clusters, 62.3% of cells, per-cell F1=0.806
[chennareddy2025] leiden_free ctrl_markers P=0.995 R=0.530 F1=0.691 (all-pos 0.897) bal.acc=0.759 (n=22250) | gap cut 0.172: 4/10 clusters, 36.3% of cells, per-cell F1=0.763
[mrvi1k] loaded cached probabilities mrvi1k__loso_chennareddy2025__491204d372.npy
[chennareddy2025] mrvi1k      real         P=0.943 R=0.883 F1=0.912 (all-pos 0.897) bal.acc=0.825 (n=22250)
[mrvi1k] loaded cached probabilities mrvi1k__shuf_chennareddy2025_0__3bcca0c202.npy
[chennareddy2025] mrvi1k      shuffled     P=0.814 R=0.815 F1=0.814 (all-pos 0.897) bal.acc=0.502 (n=22250)
[mrvi10k] loaded cached probabilities mrvi10k__loso_buus2025__091659689e.npy
[buus2025] mrvi10k     real         P=0.960 R=0.929 F1=0.944 (all-pos 0.850) bal.acc=0.910 (n=11505)
[buus2025] loaded cached buus2025_res0.2.parquet (30965 cells, 8 clusters)
[buus2025] leiden_free real         P=0.991 R=0.989 F1=0.990 (all-pos 0.850) bal.acc=0.982 (n=11505) | gap cut 0.350: 4/8 clusters, 57.8% of cells, per-cell F1=0.897
[buus2025] leiden_free ctrl_markers P=0.625 R=0.585 F1=0.604 (all-pos 0.850) bal.acc=0.295 (n=11505) | gap cut -0.029: 7/8 clusters, 79.6% of cells, per-cell F1=0.548
[buus2025] leiden_free ctrl_markers P=0.552 R=0.426 F1=0.481 (all-pos 0.850) bal.acc=0.224 (n=11505) | gap cut 0.193: 5/8 clusters, 62.5% of cells, per-cell F1=0.682
[buus2025] leiden_free ctrl_markers P=0.996 R=0.415 F1=0.586 (all-pos 0.850) bal.acc=0.705 (n=11505) | gap cut 0.233: 1/8 clusters, 20.4% of cells, per-cell F1=0.704
[buus2025] leiden_free ctrl_markers P=0.987 R=0.574 F1=0.726 (all-pos 0.850) bal.acc=0.776 (n=11505) | gap cut 0.007: 4/8 clusters, 38.6% of cells, per-cell F1=0.582
[buus2025] leiden_free ctrl_markers P=0.028 R=0.010 F1=0.014 (all-pos 0.850) bal.acc=0.027 (n=11505) | gap cut 0.390: 3/8 clusters, 35.4% of cells, per-cell F1=0.780
[mrvi1k] loaded cached probabilities mrvi1k__loso_buus2025__091659689e.npy
[buus2025] mrvi1k      real         P=0.896 R=0.886 F1=0.891 (all-pos 0.850) bal.acc=0.797 (n=11505)
[mrvi10k] loaded cached probabilities mrvi10k__loso_il4ra2026__0129f06a06.npy
[il4ra2026] mrvi10k     real         P=0.931 R=0.959 F1=0.945 (all-pos 0.813) bal.acc=0.903 (n=8453)
[il4ra2026] loaded cached il4ra2026_res0.2.parquet (20067 cells, 9 clusters)
[il4ra2026] leiden_free real         P=0.982 R=0.999 F1=0.990 (all-pos 0.813) bal.acc=0.979 (n=8453) | gap cut 0.161: 7/9 clusters, 55.5% of cells, per-cell F1=0.888
[il4ra2026] leiden_free ctrl_markers P=0.970 R=0.553 F1=0.705 (all-pos 0.813) bal.acc=0.758 (n=8453) | gap cut -0.014: 4/9 clusters, 36.1% of cells, per-cell F1=0.566
[il4ra2026] leiden_free ctrl_mar
... [7463 more chars truncated]
```

## Step 5 — final calls, persisted

The LOSO table above is the honest estimate of transfer. The calls that get *used* come from one
final fit per **MrVI** arm on **all** anchors, which is what gives the 7 no-V(D)J cohorts a
malignancy score at all — they carry no anchors of their own and exist in this analysis purely as
prediction targets.

`leiden_free` has no such step: an arm that reads no anchors has nothing to refit, and by decision
it runs on the 4 evaluation folds only. Its columns are therefore NaN outside those studies rather
than a manufactured zero, and the figures below mask each arm to the cells it actually scored.

```python  # cell 13
y_all = np.full(adata.n_obs, -1); y_all[ANCHOR] = Y[ANCHOR].astype(int)
out = adata.obs[["study", "donor", "sample_id", "cell_type_T", "has_tcr", "tcr_clonal",
                 "tcr_label", "has_expr"]].copy()
for arm in ARMS:
    if arm == "leiden_free":
        # There is no "fit on all anchors" for an arm that reads no anchors — it is computed per
        # study, and by decision it runs on the 4 evaluation folds only. Every other study stays
        # NaN rather than being handed a number this arm never produced.
        s = np.full(adata.n_obs, np.nan)
        c = np.zeros(adata.n_obs, dtype=bool)
        for held in FOLDS:
            si, ci, _, _ = leiden_free_score(held)
            m = (STUDY == held).to_numpy()
            s[m], c[m] = si[m], ci[m]
    else:
        # empty-X rows carry no information, so their score is written as NaN rather than as a
        # number that looks like a call (NaN >= thr is False, so `call_*` is False for them too)
        s = np.where(EXPR, spread_cached(arm, y_all, "all"), np.nan)
        c = s >= CALL_THR
    out[f"prob_{arm}"] = s
    out[f"call_{arm}"] = c
    n = int(np.isfinite(s).sum())
    print(f"{arm:<11} malignant {int(c.sum()):>7} / {n} scored cells "
          f"({c.sum() / max(n, 1):.1%})")

out.index.name = "cell_id"
out.to_parquet(OUT_PARQUET)
print("\nwrote", OUT_PARQUET, out.shape)

print("\ncalled malignant in the no-V(D)J cohorts (the cells this notebook is for; leiden_free is"
      "\nblank there by design — it runs on the evaluation folds only):")
nv = out[out.has_expr.to_numpy() & out.study.astype(str).isin(NO_TCR_STUDIES).to_numpy()]
print(pd.DataFrame({a: nv[nv[f"prob_{a}"].notna()]
                    .groupby("study", observed=True)[f"call_{a}"].mean()
                    for a in ARMS}).round(3).to_string())
out.head()
```

```
[mrvi10k] loaded cached probabilities mrvi10k__all__2a85d63b1e.npy
mrvi10k     malignant  259612 / 401217 scored cells (64.7%)
leiden_free malignant   55043 / 90794 scored cells (60.6%)
[mrvi1k] loaded cached probabilities mrvi1k__all__2a85d63b1e.npy
mrvi1k      malignant  265935 / 401217 scored cells (66.3%)

wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/cd4_transcriptome_malignancy_v1.parquet (401851, 14)

called malignant in the no-V(D)J cohorts (the cells this notebook is for; leiden_free is
blank there by design — it runs on the evaluation folds only):
                 mrvi10k  leiden_free  mrvi1k
study                                        
alkon2024          0.556          NaN   0.649
brentuximab2026    0.788          NaN   0.760
gaydosik2019       0.815          NaN   0.770
jonak2021          0.360          NaN   0.619
lyp2026            0.617          NaN   0.660
rindler2021_mc     0.617          NaN   0.629

                                                    study  \
cell_id                                                     
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  li2024   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1  li2024   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  li2024   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  li2024   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  li2024   

                                                                 donor  \
cell_id                                                                  
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   

                                                             sample_id  \
cell_id                                                                  
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1  Li2024_atlas__CTCL1   

                                                  cell_type_T  has_tcr  \
cell_id                                                                  
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1         CD4    False   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1         CD4    False   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1         CD4    False   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1         CD4    False   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1         CD4    False   

                                                   tcr_clonal  tcr_label  \
cell_id                                                                    
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1       False  unlabeled   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1       False  unlabeled   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1       False  unlabeled   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1       False  unlabeled   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTCL1       False  unlabeled   

                                                   has_expr  prob_mrvi10k  \
cell_id                                                                     
Li2024_atlas|AAACCTGAGCCAACAG-0_CTCL1_CTCL1_CTCL1      True      0.992039   
Li2024_atlas|AACTTTCTCTATGTGG-0_CTCL1_CTCL1_CTCL1      True      0.999971   
Li2024_atlas|AAGCCGCAGAAACGAG-0_CTCL1_CTCL1_CTCL1      True      0.996496   
Li2024_atlas|AAGGAGCAGACAGGCT-0_CTCL1_CTCL1_CTCL1      True      0.999245   
Li2024_atlas|AATCCAGTCACGAAGG-0_CTCL1_CTCL1_CTC
... [2040 more chars truncated]
```

## Step 6 — figures

First the cohort itself on the `mrvi10k` latent (150k-cell subsample) coloured by study, stage and
TCR status — that embedding is computed once and every panel below reuses it, so all the figures
are in the same coordinates.

Then one UMAP per arm: TCR anchors, then each arm's probability and call.


```python  # cell 15
# The embedding every panel in Step 6 is drawn on: UMAP of the mrvi10k latent over a 150k-cell
# subsample of the CD4 cohort. Computed once here, cached, and reused by every figure below.
if UMAP_NPZ.exists():
    _z = np.load(UMAP_NPZ); SEL, XY = _z["sel"], _z["xy"]
    print("loaded cached", UMAP_NPZ.name)
else:
    SEL = np.sort(np.random.default_rng(SEED).choice(
        adata.n_obs, min(UMAP_N, adata.n_obs), replace=False))
    ue = ad.AnnData(X=np.zeros((len(SEL), 1), dtype="float32"),
                    obs=adata.obs.iloc[SEL][["study"]].copy())
    ue.obsm["X_mrvi_u"] = REP["mrvi10k"][SEL]
    sc.pp.neighbors(ue, use_rep="X_mrvi_u", random_state=SEED)
    sc.tl.umap(ue, random_state=SEED)
    XY = ue.obsm["X_umap"]
    np.savez(UMAP_NPZ, sel=SEL, xy=XY)
    del ue; gc.collect()
    print("wrote", UMAP_NPZ, XY.shape)

# What the cohort looks like before any malignancy call is made. Read left to right: MrVI is
# integrating across studies (a study-coloured panel with no study-shaped islands is the whole
# point of the batch_key), stage tells you which axis of the map is disease progression, and the
# TCR panel says which cells could ever have served as an anchor — the 7 no-V(D)J cohorts are the
# grey mass, and they are the reason this notebook exists.
_st = adata.obs["study"].astype(str).to_numpy()[SEL]
_sc = adata.obs["stage_class"].astype(str).to_numpy()[SEL]
_ht = adata.obs["has_tcr"].to_numpy()[SEL]
_cl = adata.obs["tcr_clonal"].to_numpy()[SEL]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
fig.subplots_adjust(left=0.16)   # room for the study legend, which sits outside on the left

pal = plt.get_cmap("tab20")
for i, s in enumerate(sorted(pd.unique(_st))):
    m = _st == s
    ax[0].scatter(*XY[m].T, s=2, linewidths=0, color=pal(i % 20), label=f"{s} ({int(m.sum())})")
ax[0].legend(markerscale=4, fontsize=7, ncol=1, loc="center right",
             bbox_to_anchor=(-0.03, 0.5), frameon=False)
ax[0].set_title(f"study ({len(pd.unique(_st))} cohorts)", fontsize=10)

STAGE_COL = {"early": "#4daf4a", "advanced": "#e41a1c", "unknown": "#dddddd", "HC": "#377eb8"}
for k, col in STAGE_COL.items():
    m = _sc == k
    if m.any():
        ax[1].scatter(*XY[m].T, s=2, linewidths=0, c=col, label=f"{k} ({int(m.sum())})")
ax[1].legend(markerscale=4, fontsize=7, loc="best")
ax[1].set_title("stage_class", fontsize=10)

for m, col, nm in [(~_ht, "#dddddd", "no TCR"),
                   (_ht & ~_cl, "#1f77b4", "TCR, non-clonal"),
                   (_ht & _cl, "#d62728", "TCR, clonal")]:
    ax[2].scatter(*XY[m].T, s=2, linewidths=0, c=col, label=f"{nm} ({int(m.sum())})")
ax[2].legend(markerscale=4, fontsize=7, loc="best")
ax[2].set_title("TCR status (the anchor supply)", fontsize=10)

for x in ax:
    x.set_xticks([]); x.set_yticks([])
fig.suptitle(f"CD4 cohort on the mrvi10k latent — {len(SEL):,} of {adata.n_obs:,} cells",
             fontsize=11)
fig.savefig(FIG_DIR / "nb43_umap_descriptive.png", dpi=150, bbox_inches="tight")
plt.show()

print(pd.crosstab(_sc, np.where(~_ht, "no TCR", np.where(_cl, "TCR clonal", "TCR non-clonal")))
      .to_string())
```

```
loaded cached cd4_umap_mrvi10k.npz

<Figure size 1500x440 with 3 Axes>
col_0     TCR clonal  TCR non-clonal  no TCR
row_0                                       
HC                 0             332    1156
advanced       19754            9784   52971
early           7946           15141   16920
other              0               0   13479
unknown         2270            1742    8505
```

```python  # cell 16
# What the 1000-gene panel costs, seen rather than inferred. The per-arm panels further down all
# share the mrvi10k layout so the CALLS are comparable, which means they say nothing about the
# panel latent's own geometry. Here each latent gets its own UMAP over the SAME SEL cells, so the
# only thing that differs between the two rows is the gene space the model was trained on.
N_GENES = {"mrvi10k": len(HVG10K), "mrvi1k": len(PANEL)}
AXY = {}
for arm in MRVI_ARMS:
    f = OUT_DIR / f"cd4_umap_{arm}.npz"
    if f.exists():
        _z = np.load(f)
        assert np.array_equal(_z["sel"], SEL), f"{f.name} was built on a different subsample"
        AXY[arm] = _z["xy"]
        print("loaded cached", f.name)
    else:
        ue = ad.AnnData(X=np.zeros((len(SEL), 1), dtype="float32"),
                        obs=pd.DataFrame(index=adata.obs_names[SEL]))
        ue.obsm["X_u"] = REP[arm][SEL]
        sc.pp.neighbors(ue, use_rep="X_u", random_state=SEED)
        sc.tl.umap(ue, random_state=SEED)
        AXY[arm] = ue.obsm["X_umap"]
        np.savez(f, sel=SEL, xy=AXY[arm])
        del ue; gc.collect()
        print("wrote", f.name, AXY[arm].shape)

# The number behind the picture: among the TCR anchors, how often do a cell's k nearest neighbours
# IN THAT LATENT carry its own label? That is exactly what label spreading consumes, so it is the
# quantity the panel either preserves or throws away — and it is measured on the same k the arms
# are scored with, not a new hyper-parameter.
_anc = ANCHOR[SEL]
_ya = Y[SEL][_anc].astype(float)
PURITY = {}
for arm in MRVI_ARMS:
    A = kneighbors_graph(REP[arm][SEL][_anc], LS_KW["n_neighbors"], mode="connectivity",
                         include_self=False)
    same = (A @ _ya) * _ya + (A @ (1 - _ya)) * (1 - _ya)      # neighbours sharing the cell's label
    PURITY[arm] = float((same / LS_KW["n_neighbors"]).mean())
    print(f"[{arm}] kNN label purity over {int(_anc.sum()):,} anchors "
          f"(k={LS_KW['n_neighbors']}): {PURITY[arm]:.3f}")

_ht = adata.obs["has_tcr"].to_numpy()[SEL]
_cl = adata.obs["tcr_clonal"].to_numpy()[SEL]
_st = adata.obs["study"].astype(str).to_numpy()[SEL]
INK, INK2 = "#0b0b0b", "#52514e"

fig, ax = plt.subplots(2, 2, figsize=(9.5, 9))
pal = plt.get_cmap("tab20")
for r, arm in enumerate(MRVI_ARMS):
    XYa = AXY[arm]
    for m, col, nm in [(~_ht, "#dddddd", "no TCR"),
                       (_ht & ~_cl, "#2a78d6", "TCR, non-clonal"),
                       (_ht & _cl, "#eb6834", "TCR, clonal")]:
        ax[r, 0].scatter(*XYa[m].T, s=2, linewidths=0, c=col, label=nm)
    ax[r, 0].set_title(f"TCR status — kNN label purity {PURITY[arm]:.3f}", fontsize=9, color=INK)
    ax[r, 0].set_ylabel(f"{arm}\n{N_GENES[arm]:,} genes", fontsize=10, color=INK)
    for i, s in enumerate(sorted(pd.unique(_st))):
        m = _st == s
        ax[r, 1].scatter(*XYa[m].T, s=2, linewidths=0, color=pal(i % 20), label=s)
    ax[r, 1].set_title("study", fontsize=9, color=INK)
ax[0, 0].legend(markerscale=4, fontsize=7, loc="best", frameon=False, labelcolor=INK2)
ax[0, 1].legend(markerscale=4, fontsize=6, ncol=1, loc="center left",
                bbox_to_anchor=(1.01, 0.5), frameon=False, labelcolor=INK2)
for x in ax.ravel():
    x.set_xticks([]); x.set_yticks([])
fig.suptitle(f"Each MrVI latent on its own UMAP — same {len(SEL):,} cells, only the gene space "
             "differs", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG_DIR / "nb43_umap_mrvi10k_vs_1k.png", dpi=200, bbox_inches="tight")
plt.show()
```

```
loaded cached cd4_umap_mrvi10k.npz
wrote cd4_umap_mrvi1k.npz (150000, 2)
[mrvi10k] kNN label purity over 43,679 anchors (k=20): 0.896
[mrvi1k] kNN label purity over 43,679 anchors (k=20): 0.826

<Figure size 950x900 with 4 Axes>
```

```python  # cell 17
# SEL / XY come from the descriptive cell above — same embedding, so the panels line up.
lab = adata.obs["tcr_label"].to_numpy()[SEL]
fig, ax = plt.subplots(len(ARMS), 3, figsize=(13, 4 * len(ARMS)))
for i, arm in enumerate(ARMS):
    p, c = out[f"prob_{arm}"].to_numpy()[SEL], out[f"call_{arm}"].to_numpy()[SEL]
    for cls, col in [("unlabeled", "#dddddd"), ("benign", "#1f77b4"), ("malignant", "#d62728")]:
        m = lab == cls
        ax[i, 0].scatter(*XY[m].T, s=2, linewidths=0, c=col, label=cls)
    ax[i, 0].legend(markerscale=3, fontsize=7, loc="best")
    ax[i, 0].set_title("TCR anchors" if i == 0 else "")
    h = ax[i, 1].scatter(*XY.T, s=2, linewidths=0, c=p, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(h, ax=ax[i, 1], fraction=0.04)
    ax[i, 1].set_title(f"{arm} — P(malignant)")
    ax[i, 2].scatter(*XY.T, s=2, linewidths=0,
                     c=np.where(c, "#d62728", "#1f77b4"))
    ax[i, 2].set_title(f"{arm} — call (thr={CALL_THR})")
    ax[i, 0].set_ylabel(arm, fontsize=11)
for x in ax.ravel():
    x.set_xticks([]); x.set_yticks([])
fig.tight_layout(); fig.savefig(FIG_DIR / "nb43_umap_panels.png", dpi=150)
plt.show()
```

```
<Figure size 1300x1200 with 12 Axes>
```

### Step 6b — what each arm actually does

The panels above show *where* the calls land, all three arms on one shared MrVI UMAP of 150k
subsampled CD4 cells. They do not show *how* the calls are made, and the two arms make them in
completely different spaces — `leiden_free` never sees the MrVI latent at all, so its row above is
its output drawn in someone else's coordinates.

The next two figures take one exemplar fold (`EXPLAIN_STUDY`) and walk each mechanism end to end,
each in its own space:

- **label spreading** — mask the held-out study's anchors back to unlabeled, let the remaining
  anchors diffuse over the kNN graph of the MrVI latent, threshold at 0.5. Drawn on the global
  UMAP, because that graph is the global one.
- **`leiden_free`** — cluster the held-out study on its own cells, score each cell against the
  CTCL markers, collapse to one median per cluster, split the clusters at the widest gap. Drawn on
  a UMAP of that study's own PCA, i.e. the space the clusters were actually found in.


```python  # cell 19
# How the MrVI arms make a call: labels diffuse from the TRAINING anchors over the kNN graph of
# the latent, and the held-out study contributes nothing to the fit — it only gets scored.
EXPLAIN_STUDY = SHUFFLE_FOLD          # chennareddy2025: the fold with the most anchors
EXPLAIN_ARM = "mrvi10k"

_ho = (STUDY == EXPLAIN_STUDY).to_numpy()
_train = ANCHOR & ~_ho
_y = np.full(adata.n_obs, -1); _y[_train] = Y[_train].astype(int)
_p = spread_cached(EXPLAIN_ARM, _y, f"loso_{EXPLAIN_STUDY}")   # cached by Step 4, free here

ho_s, y_s, p_s = _ho[SEL], Y[SEL], _p[SEL]
seen_s = _train[SEL]                       # anchors the fit was allowed to see
test_s = (ANCHOR & _ho)[SEL]               # held-out anchors: scored, never seen

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

# 1. the input. Everything the fit sees is a training anchor; the whole held-out study is masked
#    back to unlabeled alongside the genuinely unlabelled cells.
ax[0].scatter(*XY[~seen_s & ~ho_s].T, s=2, linewidths=0, c="#dddddd", label="unlabeled")
ax[0].scatter(*XY[ho_s].T, s=2, linewidths=0, c="#f0c000", label=f"{EXPLAIN_STUDY} (masked)")
for v, col, nm in [(0, "#1f77b4", "benign anchor"), (1, "#d62728", "malignant anchor")]:
    m = seen_s & (y_s == bool(v))
    ax[0].scatter(*XY[m].T, s=2, linewidths=0, c=col, label=nm)
ax[0].legend(markerscale=4, fontsize=7, loc="best")
ax[0].set_title("1. input: training anchors only\nthe held-out study is unlabeled", fontsize=9)

# 2. the diffusion. Every cell ends up with a probability, including the masked study.
h = ax[1].scatter(*XY.T, s=2, linewidths=0, c=p_s, cmap="viridis", vmin=0, vmax=1)
fig.colorbar(h, ax=ax[1], fraction=0.04)
ax[1].set_title(f"2. label spreading over the {EXPLAIN_ARM} kNN graph\nP(malignant), "
                f"alpha={LS_KW['alpha']}, k={LS_KW['n_neighbors']}", fontsize=9)

# 3. the verdict, on the held-out anchors only — the cells the LOSO number is computed from.
call_s = p_s >= CALL_THR
ax[2].scatter(*XY[~test_s].T, s=2, linewidths=0, c="#eeeeee")
for m, col, nm in [(test_s & (call_s == y_s), "#2ca02c", "correct"),
                   (test_s & call_s & ~y_s, "#d62728", "false positive"),
                   (test_s & ~call_s & y_s, "#9467bd", "false negative")]:
    ax[2].scatter(*XY[m].T, s=3, linewidths=0, c=col, label=f"{nm} ({int(m.sum())})")
ax[2].legend(markerscale=4, fontsize=7, loc="best")
ax[2].set_title(f"3. call >= {CALL_THR} vs the TCR label\nheld-out anchors of "
                f"{EXPLAIN_STUDY} only", fontsize=9)
for x in ax:
    x.set_xticks([]); x.set_yticks([])
fig.suptitle(f"Label spreading — {EXPLAIN_ARM}, held out {EXPLAIN_STUDY}", fontsize=11)
fig.tight_layout(); fig.savefig(FIG_DIR / "nb43_umap_explain_labelspread.png", dpi=150)
plt.show()
```

```
[mrvi10k] loaded cached probabilities mrvi10k__loso_chennareddy2025__491204d372.npy

<Figure size 1300x420 with 4 Axes>
```

```python  # cell 20
# Headline: does the call transfer to a study it never saw a label from?
#
# Three panels, per held-out study, one bar per arm. specificity and recall are the two halves of
# balanced accuracy, so the third panel is literally the mean of the first two — which is the
# point: at 68-85% malignant anchors an arm can max out recall by calling everything and only the
# specificity panel shows the bill. F1 inherits that prevalence (its floor here is 0.81-0.92), so
# it stays in the table below rather than on the axes.
#
# ARM_SRC says which rows each bar is drawn from. The MrVI arms show their real call. `leiden_free`
# shows its NULL instead: LF_N_CTRL expression-decile-matched random marker sets pushed through
# the same clustering and the same cut, plotted as mean +- sd over the draws. So the figure reads
# "two real callers next to the score any gene set of that size would have got".
ARM_SRC = {"mrvi10k": "real", "leiden_free": "ctrl_markers", "mrvi1k": "real"}
real = loso[loso.labels.eq("real")]                        # the tables below stay on the real call

PLOT_STUDIES = [s for s in FOLDS if s not in ("buus2025",)]   # dropped from the FIGURE only;
PLOT_ARMS = {"mrvi10k": "mrvi10k", "leiden_free": "leiden", "mrvi1k": "mrvi1k"}  # every fold is
                                                              # still scored and still in the CSV
# Fixed hue order, assigned per arm and never cycled, so a bar's colour means the same thing in
# every figure. Colourblind-checked: worst adjacent pair dE 9.2 (deutan), 27.6 (normal vision).
ARM_COL = {"mrvi10k": "#2a78d6", "leiden_free": "#eb6834", "mrvi1k": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e4"
PANELS = [("specificity", "specificity"), ("recall", "recall"),
          ("balanced_acc", "balanced accuracy\n= (recall + specificity) / 2")]

for arm, src in ARM_SRC.items():
    assert len(loso[loso.arm.eq(arm) & loso.labels.eq(src)]), f"no {src!r} rows for {arm}"
_nd = {arm: int(loso[loso.arm.eq(arm) & loso.labels.eq(src)]
                .groupby("held_out_study", observed=True).size().max())
       for arm, src in ARM_SRC.items()}

x = np.arange(len(PLOT_STUDIES)); w = 0.26
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (m, lab) in zip(axes, PANELS):
    for i, arm in enumerate(ARMS):
        g = (loso[loso.arm.eq(arm) & loso.labels.eq(ARM_SRC[arm])]
             .groupby("held_out_study", observed=True)[m])
        mu = g.mean().reindex(PLOT_STUDIES)
        sd = g.std(ddof=1).reindex(PLOT_STUDIES).fillna(0.0)   # single row -> no whisker
        b = ax.bar(x + (i - 1) * w, mu.to_numpy(), width=w - 0.02, color=ARM_COL[arm],
                   label=PLOT_ARMS[arm], zorder=3, yerr=sd.to_numpy(),
                   error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2.5, zorder=4))
        ax.bar_label(b, labels=[f"{v:.2f}" for v in mu], fontsize=6, padding=1, color=INK2)
    ax.set_title(lab, fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(PLOT_STUDIES, fontsize=8, color=INK, rotation=20,
                                         ha="right")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
axes[0].set_ylim(0, 1.08); axes[0].set_yticks(np.arange(0, 1.01, 0.25))
axes[0].set_ylabel("vs the TCR call", fontsize=9, color=INK2)
axes[1].legend(fontsize=8, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.12), labelcolor=INK2)
fig.suptitle(f"Held-out study, never seen in training — leiden bar is the random-marker null "
             f"(mean +- sd, n={_nd['leiden_free']} draws)", fontsize=10, color=INK, y=1.14)
fig.savefig(FIG_DIR / "nb43_balacc_by_study.png", dpi=200, bbox_inches="tight")
plt.show()

print("bar source per arm:", {a: f"{s} (n={_nd[a]})" for a, s in ARM_SRC.items()})
for m, lab in [("specificity", "specificity"), ("recall", "recall"),
               ("balanced_acc", "balanced accuracy"), ("precision", "precision"), ("f1", "F1")]:
    print(f"\n{lab} — the REAL call, every arm, all folds:")
    print(real.pivot(index="held_out_study", columns="arm", values=m)[ARMS].round(3).to_string())
print("\nall-positive F1 floor per fold (why F1 is tabulated, not plotted):")
print(real.drop_duplicates("held_out_study").set_index("held_out_study")["f1_all_positive"]
      .round(3).to_string())
```

```
<Figure size 1200x360 with 3 Axes>
bar source per arm: {'mrvi10k': 'real (n=1)', 'leiden_free': 'ctrl_markers (n=5)', 'mrvi1k': 'real (n=1)'}

specificity — the REAL call, every arm, all folds:
arm              mrvi10k  leiden_free  mrvi1k
held_out_study                               
buus2025           0.891        0.975   0.708
chennareddy2025    0.901        0.971   0.767
gaydosik2022       0.844        0.822   0.733
il4ra2026          0.847        0.960   0.748

recall — the REAL call, every arm, all folds:
arm              mrvi10k  leiden_free  mrvi1k
held_out_study                               
buus2025           0.929        0.989   0.886
chennareddy2025    0.950        0.994   0.883
gaydosik2022       0.961        0.870   0.896
il4ra2026          0.959        0.999   0.904

balanced accuracy — the REAL call, every arm, all folds:
arm              mrvi10k  leiden_free  mrvi1k
held_out_study                               
buus2025           0.910        0.982   0.797
chennareddy2025    0.926        0.983   0.825
gaydosik2022       0.902        0.846   0.814
il4ra2026          0.903        0.979   0.826

precision — the REAL call, every arm, all folds:
arm              mrvi10k  leiden_free  mrvi1k
held_out_study                               
buus2025           0.960        0.991   0.896
chennareddy2025    0.977        0.993   0.943
gaydosik2022       0.971        0.964   0.949
il4ra2026          0.931        0.982   0.886

F1 — the REAL call, every arm, all folds:
arm              mrvi10k  leiden_free  mrvi1k
held_out_study                               
buus2025           0.944        0.990   0.891
chennareddy2025    0.963        0.994   0.912
gaydosik2022       0.966        0.915   0.922
il4ra2026          0.945        0.990   0.895

all-positive F1 floor per fold (why F1 is tabulated, not plotted):
held_out_study
chennareddy2025    0.897
buus2025           0.850
il4ra2026          0.813
gaydosik2022       0.917
```
