# 10_atlas/14_annotate_new_data.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/10_atlas/14_annotate_new_data.ipynb` and is never run._

# Calling malignant cells in new CTCL data

This notebook takes a new single-cell dataset of skin CD4 T cells and labels each cell malignant
or benign, two independent ways.

**(a) Your markers, per cluster.** We cluster your cells on their own, score every cell against a
gene list you supply, and split the clusters at the largest gap in their median score. This never
looks at our data.

**(b) Our anchors, spread through a shared latent.** We merge your cells with a CTCL atlas of
401,217 CD4 T cells from fourteen published cohorts, train MrVI on the merged object, and diffuse
our TCR-clonality labels along the nearest-neighbour graph until they reach your cells. This never
looks at your markers.

The two arms share no inputs, so where they agree you have real evidence, and where they disagree
you have a short list of clusters worth looking at by hand. That is the point of running both.

### What you need

- An `.h5ad` with **raw integer counts**, **gene symbols**, one `obs` column identifying each
  library or biopsy, and your cells gated to **CD4 T cells**.
- One GPU with at least 24 GB. Budget two to three hours end to end; MrVI training is most of it.
- `scvi-tools`, `scanpy`, `anndata`, `scikit-learn`, and a CUDA build of `jax`.

### What you edit

Two cells, both marked `### INSERT ###`: the loader in Step 2, and the marker list in Step 6.
Everything else is the `CONFIG` cell below. The notebook stops with an explicit message if either
insert point is still empty.

Accuracy you should expect, and the caveats that go with it, are in the last section. Read it
before you quote a number.

```python  # cell 1
# ============================================================================
# CONFIG — edit this cell, then run the notebook top to bottom.
# ============================================================================
import os, sys

# Preallocate the GPU before jax is imported. Left on demand, MrVI training dies partway through
# with a non-deterministic CUDA_ERROR_ILLEGAL_ADDRESS — allocator fragmentation, not your data.
# This has to happen before anything pulls in jax or scvi-tools.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
assert "jax" not in sys.modules, \
    "jax is already imported — restart the kernel and run this cell first"

from pathlib import Path

# --- our atlas, shipped alongside this notebook -----------------------------
ATLAS = Path("ctcl_cd4_atlas_v1.h5ad")   # 401,217 CD4 T cells x 42,347 genes, raw counts in .X

# --- your data --------------------------------------------------------------
NEW_H5AD   = Path("CHANGE_ME.h5ad")
NEW_COUNTS = None            # None if raw counts are in .X, else the layer name, e.g. "counts"
NEW_SAMPLE = "sample"        # obs column: one value per library/biopsy. Not one value overall.
NEW_STUDY  = "newdata"       # a single label for your whole dataset; this is MrVI's batch
NEW_SYMBOL = None            # None if var_names are gene symbols, else the var column that is
CELLTYPE   = "cell_type"     # obs column used to gate down to CD4 T cells
CD4_VALUES = ["CD4 T", "CD4"]  # the values in CELLTYPE that mean "CD4 T cell"

# --- outputs ----------------------------------------------------------------
OUT_DIR = Path("mrvi_out"); OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
SEED = 0

# --- settings that reproduce our published run; you should not need to touch these ---
MIN_SHARED_GENES = 300       # below this there is not enough signal to cluster on — hard stop
HVG_TRIGGER      = 12_000    # if you share more genes than this, we reduce to HVGs...
HVG_CAP          = 10_000    # ...this many, which is the regime our accuracy was measured in
MRVI_KW  = dict(n_latent=30, n_latent_u=10)
TRAIN_KW = dict(max_epochs=100, batch_size=256, early_stopping=True,
                early_stopping_patience=15, check_val_every_n_epoch=1, train_size=0.9)
K_NEIGHBORS, LS_ALPHA, LS_MAX_ITER = 20, 0.2, 60     # label spreading
CALL_THR = 0.5                                        # fixed, never tuned
LF_HVG, LF_PCS, LF_RES = 2000, 50, 0.2                # clustering, for arm (a)
LF_MIN_SIDE = 0.15                                    # min share of cells each side of the cut
UMAP_N = 150_000
```

```python  # cell 2
import gc, warnings
import anndata as ad, numpy as np, pandas as pd, scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelSpreading

np.random.seed(SEED)
sc.settings.verbosity = 1


def present(adata, genes):
    """The genes of `genes` that exist in adata.var_names, order kept, duplicates dropped."""
    have, seen, keep = set(adata.var_names), set(), []
    for g in genes:
        if g in have and g not in seen:
            keep.append(g); seen.add(g)
    return keep


def binary_scores(y, yhat):
    """Agreement of a boolean call against a boolean reference."""
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp, tn = int((yhat & y).sum()), int((~yhat & ~y).sum())
    fp, fn = int((yhat & ~y).sum()), int((~yhat & y).sum())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    return dict(precision=pr, recall=rc, specificity=sp,
                f1=2 * pr * rc / max(1e-9, pr + rc), balanced_acc=0.5 * (rc + sp))
```

## Step 1 · The atlas

401,217 CD4 T cells from fourteen published skin CTCL and control cohorts, 143 samples, over the
atlas's **complete 42,347-gene space**. Raw counts are in `.X`.

The full gene space is deliberate. If you are bringing a targeted panel — Xenium, CosMx, a custom
1,000-plex — then whichever genes you have, we can pull those same genes out of our side, and the
merged object is your panel rather than the intersection of your panel with some list we picked
first. Loading it needs roughly 8 GB of RAM; Step 3 cuts it down to your genes immediately.

The column that matters is `tcr_label`, and it is worth being precise about what it means, because
arm (b) is only as good as these anchors:

| value | meaning | cells |
|---|---|---|
| `malignant` | the cell carries its donor's dominant T-cell clone | 81,963 |
| `benign` | the cell has a recovered TCR, is not in that clone, and is not in any expanded clone | 34,890 |
| `unlabeled` | everything else | 284,364 |

A cell with no recovered TCR is **not** a negative — it is unlabeled. Treating missing V(D)J as
evidence of benignity is the most common way to get this wrong, and it inflates every specificity
you would report afterwards.

```python  # cell 4
assert ATLAS.exists(), f"{ATLAS} not found — it ships alongside this notebook"
atlas = sc.read_h5ad(ATLAS)
atlas.obs["dataset_source"] = "atlas"

print(f"atlas: {atlas.n_obs:,} cells x {atlas.n_vars:,} genes")
print(f"       {atlas.obs.sample_id.nunique()} samples, {atlas.obs.study.nunique()} studies, "
      f"{atlas.obs.donor.nunique()} donors\n")
print(atlas.obs.tcr_label.value_counts().to_string())

assert (atlas.n_obs, atlas.n_vars) == (401217, 42347), \
    f"got {atlas.shape}; this is not the atlas this notebook was written against"
assert set(atlas.obs.tcr_label.unique()) == {"malignant", "benign", "unlabeled"}

# For scale, when you look at your own sample sizes in a moment:
s = atlas.obs.sample_id.value_counts()
print(f"\ncells per atlas sample: min {s.min():,}, median {int(s.median()):,}, max {s.max():,}")
```

## Step 2 · Load your data · `### INSERT ###`

Replace the body of the next cell so that it produces an `AnnData` for your dataset. Four things
have to be true of it, and the cell after checks all four.

**Raw integer counts.** MrVI models counts with a negative binomial. Log-normalised input does not
error — it trains happily and returns a latent space that means nothing, which is the worst
failure mode available here. If your counts are in a layer, name it in `NEW_COUNTS`; if they are
in `.raw`, use `adata.raw.to_adata()`.

**Gene symbols.** HUGO symbols in `var_names`, or in a `var` column you name in `NEW_SYMBOL`.

**One value per library or biopsy in `NEW_SAMPLE`.** This becomes MrVI's sample key. If your whole
dataset is a single sample, MrVI's sample-level correction has nothing to correct and its `u`
latent collapses toward the ordinary `z`.

**CD4 T cells only.** Our anchors are CD4 T cells. A keratinocyte pushed through arm (b) still
comes out with a number, and the number is meaningless.

```python  # cell 6
# ==========================  ### INSERT ###  ================================
# Load your data here. Return an AnnData called `new`.
# Replace this body if your data does not live in a single .h5ad.
# ============================================================================
new = sc.read_h5ad(NEW_H5AD)
# new = new[new.obs["qc_pass"]].copy()      # <- your own QC, if you have not applied it yet

print(new)
```

```python  # cell 7
# --- obs index and the columns CONFIG refers to -----------------------------
assert new.obs_names.is_unique, "obs_names are not unique — run new.obs_names_make_unique()"
for col in (NEW_SAMPLE, CELLTYPE):
    assert col in new.obs, (
        f"obs column {col!r} not found. Your columns are: {list(new.obs.columns)}. "
        "Either fix NEW_SAMPLE / CELLTYPE in CONFIG, or add the column.")

# --- gate to CD4 T cells ----------------------------------------------------
ct = new.obs[CELLTYPE].astype(str)
keep = ct.isin(CD4_VALUES).to_numpy()
assert keep.any(), (
    f"none of CD4_VALUES={CD4_VALUES} occur in obs[{CELLTYPE!r}]. "
    f"Values present: {sorted(ct.unique())[:30]}")
print(f"CD4 gate: keeping {keep.sum():,} of {new.n_obs:,} cells ({keep.sum() / new.n_obs:.1%}); "
      "the rest are dropped and will not be scored")
new = new[keep].copy()

# --- counts must be raw, integral, non-negative -----------------------------
assert NEW_COUNTS is None or NEW_COUNTS in new.layers, \
    f"layer {NEW_COUNTS!r} not found; layers present: {list(new.layers)}"
X = new.layers[NEW_COUNTS] if NEW_COUNTS else new.X
d = (X.data if sparse.issparse(X) else np.asarray(X).ravel())[:2_000_000]
assert (d >= 0).all(), "negative values — this looks like scaled data, not counts"
assert np.allclose(d, np.round(d)), (
    "non-integer values — this looks log-normalised or scaled, and MrVI needs raw counts. "
    "If you kept the counts in a layer, set NEW_COUNTS in CONFIG; if they are in .raw, "
    "use adata.raw.to_adata().")
assert d.max() > 30, f"largest value is {d.max():.1f}, which is small for raw counts — check NEW_COUNTS"

tot = np.asarray(X.sum(1)).ravel()
assert (tot > 0).all(), f"{int((tot == 0).sum())} cells have zero total counts — drop them first"
print(f"counts: median {np.median(tot):,.0f} UMI/cell, "
      f"range {tot.min():,.0f}-{tot.max():,.0f}")
```

```python  # cell 8
# --- gene identifiers -------------------------------------------------------
if NEW_SYMBOL:
    new.var_names = new.var[NEW_SYMBOL].astype(str)
vn = pd.Index(new.var_names.astype(str))
frac_ens = float(vn.str.startswith("ENS").mean())
print(f"genes: {new.n_vars:,} | {frac_ens:.0%} look like Ensembl IDs")

assert frac_ens < 0.5, (
    "your var_names are Ensembl IDs and the atlas is indexed by HUGO symbol. Convert them "
    "upstream — from the features.tsv of your own CellRanger reference, which is the mapping "
    "that actually matches your counts — and either set var_names to the symbols or point "
    "NEW_SYMBOL at the var column holding them. We deliberately do not ship a symbol table: "
    "using someone else's mapping silently drops the genes where the two references disagree.")

if not pd.Series(vn).str.isupper().mean() > 0.5:
    warnings.warn("most gene names are not upper case — mouse symbols? This atlas is human.")

ndup = int(new.n_vars - vn.nunique())
if ndup:
    print(f"  {ndup} duplicated gene names — making them unique; note that the '-1' copies "
          "cannot match the atlas and are effectively dropped in the next step")
    new.var_names_make_unique()

# --- samples ----------------------------------------------------------------
sz = new.obs[NEW_SAMPLE].value_counts()
print(f"samples: {len(sz)} | cells per sample: min {sz.min():,}, "
      f"median {int(sz.median()):,}, max {sz.max():,}")
if len(sz) == 1:
    warnings.warn(
        "your dataset is a single sample. MrVI will run, but its sample-level correction has "
        "nothing to correct and the u latent will behave like z. If you have several libraries "
        "or biopsies, point NEW_SAMPLE at the column that distinguishes them.")
if (sz < 50).any():
    warnings.warn(f"{int((sz < 50).sum())} sample(s) have fewer than 50 cells; MrVI's per-sample "
                  "parameters are poorly determined there")
```

## Step 3 · Reduce both datasets to their shared genes

MrVI has to see one gene space, so we keep the genes you and the atlas have in common. Because the
atlas ships its full 42,347 genes, that intersection is essentially *your* gene list: a 1,000-plex
panel stays a 1,000-plex panel rather than being cut down to whatever overlapped a preselected
list.

Two regimes come out of this, and they carry different accuracies:

- **Targeted panel** (a few hundred to a few thousand shared genes). Everything you have is used.
  This is the regime our 1,000-gene measurement covers — expect balanced accuracy around 0.82.
- **Whole transcriptome** (more than `HVG_TRIGGER` shared genes). We reduce to the top `HVG_CAP`
  highly variable genes, selected on the merged object and aware of the study split, because that
  is the regime the 0.906 figure was measured in and because MrVI on 25,000 genes costs a lot more
  for no measured gain.

If the intersection is unexpectedly small for whole-transcriptome data, the cause is almost always
mismatched gene identifiers rather than genuinely sparse data. Check the previous cell first.

```python  # cell 10
shared = [g for g in atlas.var_names if g in set(new.var_names)]   # atlas order: deterministic
print(f"atlas genes : {atlas.n_vars:,}")
print(f"your genes  : {new.n_vars:,}")
print(f"shared      : {len(shared):,}  "
      f"({len(shared) / new.n_vars:.1%} of yours, {len(shared) / atlas.n_vars:.1%} of the atlas)")

assert len(shared) >= MIN_SHARED_GENES, (
    f"only {len(shared)} shared genes — too few to cluster or to train on, so the notebook stops. "
    "For a targeted panel this usually means the gene identifiers do not match (Ensembl vs "
    "symbol, or a different reference build) rather than that the panel is small.")

atlas = atlas[:, shared].copy()
new = new[:, shared].copy()
assert list(atlas.var_names) == list(new.var_names)

REGIME = "panel" if len(shared) <= HVG_TRIGGER else "whole-transcriptome"
print(f"regime      : {REGIME}")

# A cell with nothing detected across the shared genes carries no information and would break
# HVG selection later. Drop it here, where it is visible, rather than scoring noise.
Xn = new.layers[NEW_COUNTS] if NEW_COUNTS else new.X
nz = np.asarray((Xn > 0).sum(1)).ravel()
if (nz == 0).any():
    print(f"dropping {int((nz == 0).sum()):,} of your cells with no detected shared gene")
    new = new[nz > 0].copy()
print(f"your cells: median {int(np.median(nz[nz > 0])):,} shared genes detected")
```

## Step 4 · Merge

Your cells get one `study` label, because to MrVI your dataset is one batch. They keep their own
`sample_id` at library granularity, prefixed `NEW__` so it cannot collide with one of our 143.
Barcodes are prefixed the same way. Every one of your cells is `tcr_label == "unlabeled"` — you
have no anchors, and supplying them is not the point; arm (b) is about transferring ours.

The merged object is written to disk before training, so a crash during the hour that follows
costs you nothing above this line.

```python  # cell 12
Xn = new.layers[NEW_COUNTS] if NEW_COUNTS else new.X
donor_src = new.obs["donor"] if "donor" in new.obs else new.obs[NEW_SAMPLE]
newc = ad.AnnData(
    X=Xn.tocsr() if sparse.issparse(Xn) else sparse.csr_matrix(Xn),
    obs=pd.DataFrame({
        "sample_id": "NEW__" + new.obs[NEW_SAMPLE].astype(str).to_numpy(),
        "study": NEW_STUDY,
        "donor": "NEW__" + donor_src.astype(str).to_numpy(),
        "cell_type_T": "CD4",
        "tcr_label": "unlabeled",
        "dataset_source": "new",
    }, index="NEW__" + new.obs_names.astype(str)),
    var=pd.DataFrame(index=pd.Index(shared)))

merged = ad.concat([atlas, newc], join="inner", merge="same")
for c in ("sample_id", "study", "donor", "tcr_label", "dataset_source"):
    merged.obs[c] = merged.obs[c].astype(str).astype("category")

assert merged.n_obs == atlas.n_obs + newc.n_obs
assert merged.obs_names.is_unique, "barcode collision after the merge"
src = merged.obs.dataset_source.to_numpy()
assert not (set(merged.obs.sample_id[src == "new"]) & set(merged.obs.sample_id[src == "atlas"]))

print(f"merged: {merged.n_obs:,} cells x {merged.n_vars:,} genes | "
      f"{merged.obs.sample_id.nunique()} samples, {merged.obs.study.nunique()} studies")
print(merged.obs.dataset_source.value_counts().to_string())

# Whole-transcriptome input only: reduce to the HVGs, selected on the merged object so both
# datasets inform the choice, and batch-aware so no single study drives it. A panel is already
# the right size and is passed through untouched.
if merged.n_vars > HVG_TRIGGER:
    print(f"reducing {merged.n_vars:,} genes to the top {HVG_CAP:,} HVGs (seurat_v3, by study)")
    sc.pp.highly_variable_genes(merged, n_top_genes=HVG_CAP, flavor="seurat_v3",
                                batch_key="study", subset=True)
    shared = list(merged.var_names)
    print(f"merged is now {merged.n_obs:,} x {merged.n_vars:,}")
else:
    print(f"keeping all {merged.n_vars:,} genes — this is panel-sized, nothing to trim")

MERGED_H5AD = OUT_DIR / "merged_mrvi_input.h5ad"
tmp = MERGED_H5AD.with_suffix(".tmp.h5ad")
merged.write_h5ad(tmp); tmp.replace(MERGED_H5AD)     # tmp + rename: no truncated file on an OOM
del atlas, new, newc, merged
gc.collect()
print("wrote", MERGED_H5AD)
```

## Step 5 · Train MrVI — this is the heavy step

Our reference run, on one NVIDIA A40: **401,851 cells by 10,000 genes, 100 epochs, 44.8 minutes**,
4.3 GB of host RAM and about 17 GB of GPU memory. That is roughly seven minutes per 100,000 cells
at this gene count, so a merged object of 450k is about 50 minutes, 600k about 65, and 800k about
90. Fewer shared genes is faster. Early stopping watches the validation ELBO with a patience of
15 epochs and may finish sooner; ours ran the full hundred.

Those are measurements on an A40, not extrapolations to other cards. You need at least 24 GB of
GPU memory; on a smaller card drop `batch_size` to 128 and set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. On CPU this is about a day, so don't.

The latent is cached to disk next to its barcodes. Re-running this cell after a kernel restart
costs an `np.load`, not another hour.

```python  # cell 14
import jax, scvi
from scvi.external import MRVI

print("jax devices:", jax.devices(), "| backend:", jax.default_backend())
assert jax.default_backend() == "gpu", (
    "jax is running on CPU, where this would take about a day. Fix the CUDA/jaxlib install, "
    "or move the notebook to a GPU node.")

U_NPY, BC_NPY, MODEL = OUT_DIR / "mrvi_u.npy", OUT_DIR / "mrvi_barcodes.npy", OUT_DIR / "mrvi_model"
merged = sc.read_h5ad(MERGED_H5AD)

if U_NPY.exists():
    U = np.load(U_NPY)
    bc = np.load(BC_NPY, allow_pickle=True).astype(str)
    print(f"loaded cached latent {U.shape}")
else:
    scvi.settings.seed = SEED
    MRVI.setup_anndata(merged, layer=None, sample_key="sample_id", batch_key="study")
    model = MRVI(merged, **MRVI_KW)
    model.train(**TRAIN_KW)
    model.save(str(MODEL), overwrite=True, save_anndata=False)
    # u is the sample-corrected latent; give_z=True would return the sample-aware one instead.
    U = model.get_latent_representation(batch_size=512, give_z=False)
    bc = merged.obs_names.to_numpy().astype(str)
    np.save(U_NPY, U); np.save(BC_NPY, bc)
    print("saved", U_NPY, U.shape)

# Storing the barcodes next to the latent makes it self-describing: a stale .npy from a different
# merge fails right here instead of quietly mis-assigning every call downstream.
assert len(bc) == U.shape[0] == merged.n_obs, (len(bc), U.shape, merged.n_obs)
idx = pd.Index(bc).get_indexer(merged.obs_names)
assert (idx >= 0).all(), "the cached latent does not match this merge — delete mrvi_u.npy and re-run"
U = U[idx]

IS_NEW = (merged.obs.dataset_source == "new").to_numpy()
LABEL = merged.obs.tcr_label.astype(str).to_numpy()
print(f"{IS_NEW.sum():,} of your cells, {(~IS_NEW).sum():,} atlas cells")
```

### Did MrVI actually integrate your cells?

Before reading anything into arm (b), check whether your cells landed anywhere near ours. If they
sit in their own island, label spreading has nothing to spread *from* and its probabilities are
extrapolation rather than transfer. This costs one nearest-neighbour query, and the graph it
builds is reused by arm (b), so it is not extra work.

```python  # cell 16
A_KNN = kneighbors_graph(U, K_NEIGHBORS, mode="connectivity", include_self=True)
sparse.save_npz(OUT_DIR / "knn.npz", A_KNN)

deg = np.asarray(A_KNN.sum(1)).ravel()
f_atlas = np.asarray(A_KNN @ (~IS_NEW).astype(float)).ravel() / deg
f_anchor = np.asarray(A_KNN @ (LABEL != "unlabeled").astype(float)).ravel() / deg

med = float(np.median(f_atlas[IS_NEW]))
print(f"of your cells' {K_NEIGHBORS} nearest neighbours:")
print(f"  median {med:.2f} are atlas cells")
print(f"  median {np.median(f_anchor[IS_NEW]):.2f} are TCR anchors")
print(f"  {(f_anchor[IS_NEW] == 0).mean():.1%} of your cells have no anchor among their neighbours")
if med < 0.10:
    warnings.warn(
        "your cells barely neighbour any atlas cell, so MrVI did not integrate them. Arm (b) is "
        "extrapolating and its probabilities should be treated as a weak prior at best. Arm (a), "
        "which never looks at our data, is the one to trust here.")
```

## Step 6 · Arm (a): your markers, per cluster · `### INSERT ###`

Four steps. Cluster your cells on their own. Score every cell as (mean of your `up` genes) minus
(mean of your `dn` genes), using `scanpy`'s expression-binned control correction so the score is
centred near zero. Take the **median** score per cluster. Then split the sorted cluster medians at
the widest gap that still leaves at least 15% of your cells on each side.

**That last constraint is load-bearing.** Without it, on three of the four cohorts we tested, the
widest gap sat at the *bottom* of the cluster ranking: one small outlier cluster lay far below the
rest, the gap peeled it off, and the genuinely benign population stayed on the malignant side.
That reads as specificity 0.007, 0.003 and 0.001 at recall near 1.0 — "everything is tumour" —
while the cluster *ranking* was in fact excellent. The ranking was never the problem, only the
cut. Requiring mass on both sides moves the cut to the next admissible gap and recovers balanced
accuracy between 0.85 and 0.98 on all four. Thresholds of 10%, 15%, 20% and 30% pick the identical
cut on every one of them, so 0.15 sits in the middle of a plateau rather than on a peak.

One discipline: **fix the direction of your markers before you look at the result.** Genes go in
`up` because you expect them higher in malignant cells, not because that orientation scored
better.

We ship this list empty on purpose. The right markers depend on your subtype, your platform and
your own prior work, and a default here would quietly become everybody's answer. Ours is quoted in
the last section if you want a starting point.

```python  # cell 18
# ==========================  ### INSERT ###  ================================
# "up" = expected higher in malignant cells, "dn" = expected lower.
MARKERS = {"up": [], "dn": []}
# ============================================================================

assert MARKERS["up"] and MARKERS["dn"], (
    "MARKERS is empty. Arm (a) needs a directional gene list: put genes you expect HIGHER in "
    "malignant cells into MARKERS['up'] and genes you expect LOWER into MARKERS['dn'], then "
    "re-run this cell. Arm (b) below does not use MARKERS and will run without it.")
```

```python  # cell 19
sub = merged[IS_NEW].copy()
sc.pp.normalize_total(sub, target_sum=1e4)
sc.pp.log1p(sub)

panel = {k: present(sub, v) for k, v in MARKERS.items()}
for k in ("up", "dn"):
    assert panel[k], (
        f"none of MARKERS[{k!r}] survive into the {len(shared)} shared genes: {MARKERS[k]}. "
        "Choose markers that are present in both datasets.")
    miss = [g for g in MARKERS[k] if g not in panel[k]]
    if miss:
        print(f"  {k}: {len(miss)} of {len(MARKERS[k])} absent from the shared gene space -> {miss}")
    sc.tl.score_genes(sub, panel[k], score_name=k, random_state=SEED)
sub.obs["mscore"] = sub.obs["up"] - sub.obs["dn"]

lb = sub.copy()
sc.pp.highly_variable_genes(lb, n_top_genes=min(LF_HVG, lb.n_vars - 1))
lb = lb[:, lb.var.highly_variable].copy()
sc.pp.scale(lb, max_value=10)
sc.tl.pca(lb, n_comps=min(LF_PCS, min(lb.shape) - 1), random_state=SEED)
sc.pp.neighbors(lb, random_state=SEED)
sc.tl.leiden(lb, resolution=LF_RES, random_state=SEED, key_added="cl",
             flavor="igraph", n_iterations=2, directed=False)
sub.obs["cl"] = lb.obs["cl"].astype(str).to_numpy()
del lb; gc.collect()
print(f"{sub.obs.cl.nunique()} clusters over {sub.n_obs:,} of your cells")
```

```python  # cell 20
def largest_gap_cut(scores, sizes, min_side=LF_MIN_SIDE):
    """Split sorted cluster scores at the widest gap that leaves at least `min_side` of the CELLS
    on each side, and return the lower edge of the upper group. inf means abstain."""
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


g = sub.obs.groupby("cl", observed=True)["mscore"]
cs, sz = g.median(), g.size()
cut = largest_gap_cut(cs, sz)
assert np.isfinite(cut), (
    f"arm (a) abstains: there are only {len(cs)} clusters, or every gap leaves less than "
    f"{LF_MIN_SIDE:.0%} of your cells on one side. Raise LF_RES to split the data more finely.")

call_a = (sub.obs.cl.map(cs) >= cut).to_numpy()
print(pd.DataFrame({"median_mscore": cs.round(3), "cells": sz,
                    "call": np.where(cs >= cut, "malignant", "benign")})
      .sort_values("median_mscore").to_string())
print(f"\ncut at {cut:.3f}: {int((cs >= cut).sum())} of {len(cs)} clusters malignant, "
      f"{call_a.mean():.1%} of your cells")
assert 0.01 < call_a.mean() < 0.99, (
    "arm (a) called essentially everything one way. Check the direction of your markers and the "
    "cluster table above before reading anything into this.")
```

## Step 7 · Arm (b): spreading our anchors into your cells

`y` is 1 on our `malignant` cells, 0 on our `benign` cells, and −1 everywhere else — including
every one of your cells. `LabelSpreading` then diffuses those labels along the nearest-neighbour
graph of the MrVI latent until they reach your side of it. Your markers play no part.

One implementation note, because it is a factor of two hundred: the neighbour graph is handed to
`LabelSpreading` as a callable kernel that returns the matrix we already built. With
`kernel="knn"` scikit-learn rebuilds the entire graph inside every call to `.fit()` — we measured
83 seconds of graph construction against 0.4 seconds for the spreading itself at 150,000 cells.
Passing `n_jobs=-1` makes it worse, not better, by about 25 times.

```python  # cell 22
y = np.full(len(U), -1, dtype=int)
y[LABEL == "malignant"] = 1
y[LABEL == "benign"] = 0
assert (y[IS_NEW] == -1).all(), "your cells must all be unlabeled going in"
print(f"anchors: {(y == 1).sum():,} malignant, {(y == 0).sum():,} benign, "
      f"{(y == -1).sum():,} unlabeled")

ls = LabelSpreading(kernel=lambda X, Y=None, _A=A_KNN: _A,
                    alpha=LS_ALPHA, max_iter=LS_MAX_ITER).fit(U, y)
# classes_ is not guaranteed to be ordered [0, 1] — look the column up rather than taking [:, 1].
prob_b = ls.label_distributions_[:, list(ls.classes_).index(1)]
assert np.isfinite(prob_b).all()

call_b = prob_b[IS_NEW] >= CALL_THR
print(f"\narm (b): {call_b.mean():.1%} of your cells called malignant at p >= {CALL_THR}")
print(pd.Series(prob_b[IS_NEW], name="P(malignant)").describe().round(3).to_string())
```

## Step 8 · Compare the two arms

Arm (a) read your markers and your clusters and never saw our labels. Arm (b) read our labels and
never saw your markers. Because they share nothing, their agreement is evidence, and either one
on its own is a hypothesis.

Disagreement is not automatically error, and it is not spread evenly — it concentrates in
particular clusters, which is what makes it useful. A cluster where arm (b) is confidently
malignant and arm (a) is not usually means your marker list is missing that program. The reverse
usually means the cluster has no close atlas neighbours; the diagnostic in Step 5 will tell you.

```python  # cell 24
res = pd.DataFrame({
    "cell_id": pd.Index(merged.obs_names[IS_NEW]).str.removeprefix("NEW__"),
    "sample": pd.Index(merged.obs.sample_id[IS_NEW].astype(str)).str.removeprefix("NEW__"),
    "leiden": sub.obs.cl.to_numpy(),
    "mscore": sub.obs.mscore.to_numpy(),
    "cluster_mscore": sub.obs.cl.map(cs).to_numpy(),
    "atlas_neighbour_frac": f_atlas[IS_NEW],
    "call_markers": call_a,
    "prob_spread": prob_b[IS_NEW],
    "call_spread": call_b,
}).set_index("cell_id")
res["agree"] = res.call_markers == res.call_spread

print("rows = arm (a) markers, columns = arm (b) label spreading")
print(pd.crosstab(res.call_markers, res.call_spread, margins=True), "\n")
print(f"agreement      : {res.agree.mean():.1%}")
print(f"both malignant : {(res.call_markers & res.call_spread).mean():.1%}")
print(f"Cohen's kappa  : {cohen_kappa_score(res.call_markers, res.call_spread):.3f}")

print("\nper cluster — where the two arms part company:")
print(res.groupby("leiden").agg(
    cells=("agree", "size"), cluster_mscore=("cluster_mscore", "first"),
    markers_malig=("call_markers", "mean"), spread_malig=("call_spread", "mean"),
    mean_prob=("prob_spread", "mean"), atlas_nbrs=("atlas_neighbour_frac", "mean"),
    agreement=("agree", "mean")).sort_values("cluster_mscore").round(3).to_string())

print("\nper sample:")
print(res.groupby("sample")[["call_markers", "call_spread", "agree"]].mean().round(3).to_string())
```

```python  # cell 25
# --- figure 1: the latent, subsampled ---------------------------------------
sel = np.sort(np.random.default_rng(SEED).choice(len(U), min(UMAP_N, len(U)), replace=False))
ue = ad.AnnData(X=np.zeros((len(sel), 1), np.float32),
                obs=merged.obs.iloc[sel][["dataset_source", "tcr_label"]].astype(str).copy(),
                obsm={"X_u": np.ascontiguousarray(U[sel], dtype=np.float32)})
sc.pp.neighbors(ue, use_rep="X_u", random_state=SEED)
sc.tl.umap(ue, random_state=SEED)
XY = ue.obsm["X_umap"]

new_s = IS_NEW[sel]
pa = pd.Series(res.call_markers.to_numpy(), index=np.flatnonzero(IS_NEW)).reindex(sel).to_numpy()
pb = prob_b[sel]

fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
ax[0].scatter(*XY[~new_s].T, s=2, lw=0, c="#dddddd", label="atlas", rasterized=True)
ax[0].scatter(*XY[new_s].T, s=2, lw=0, c="#2a78d6", label="your data", rasterized=True)
ax[0].legend(markerscale=5, fontsize=8, frameon=False)
ax[0].set_title("integration")

lab = ue.obs.tcr_label.to_numpy()
for cls, col in [("unlabeled", "#dddddd"), ("benign", "#1f77b4"), ("malignant", "#d62728")]:
    ax[1].scatter(*XY[lab == cls].T, s=2, lw=0, c=col, label=cls, rasterized=True)
ax[1].legend(markerscale=5, fontsize=8, frameon=False)
ax[1].set_title("our TCR anchors")

ax[2].scatter(*XY[~new_s].T, s=2, lw=0, c="#eeeeee", rasterized=True)
ax[2].scatter(*XY[new_s].T, s=2, lw=0,
              c=np.where(pa[new_s].astype(bool), "#d62728", "#1f77b4"), rasterized=True)
ax[2].set_title("arm (a): your markers, per cluster")

ax[3].scatter(*XY[~new_s].T, s=2, lw=0, c="#eeeeee", rasterized=True)
h = ax[3].scatter(*XY[new_s].T, s=2, lw=0, c=pb[new_s], cmap="viridis", vmin=0, vmax=1,
                  rasterized=True)
fig.colorbar(h, ax=ax[3], fraction=0.04)
ax[3].set_title("arm (b): P(malignant)")

for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_box_aspect(1)
fig.tight_layout(); fig.savefig(FIG_DIR / "nb46_umap_panels.png", dpi=150); plt.show()
```

```python  # cell 26
# --- figure 2: the comparison in one picture --------------------------------
o = cs.sort_values()
mean_p = res.groupby("leiden").prob_spread.mean().reindex(o.index)

fig, ax = plt.subplots(figsize=(max(5.5, 0.45 * len(o) + 2.2), 3.6))
bars = ax.bar(np.arange(len(o)), o.values,
              color=plt.get_cmap("viridis")(mean_p.fillna(0).values), edgecolor="#444", lw=0.5)
ax.axhline(cut, color="#c0392b", lw=1.2, ls="--")
ax.annotate(f"cut {cut:.2f}", (len(o) - 0.4, cut), fontsize=8, color="#c0392b",
            va="bottom", ha="right")
ax.set_xticks(np.arange(len(o)), o.index, fontsize=8)
ax.set_xlabel("Leiden cluster"); ax.set_ylabel("median marker score")
fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, fraction=0.03,
             label="mean P(malignant), arm (b)")
ax.set_title("Clusters above the cut are arm (a)'s malignant call;\n"
             "bar colour is arm (b)'s independent opinion of the same cluster", fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG_DIR / "nb46_cluster_cut.png", dpi=180); plt.show()
```

```python  # cell 27
res.to_parquet(OUT_DIR / "new_data_malignancy.parquet")
print("wrote", OUT_DIR / "new_data_malignancy.parquet", res.shape, "\n")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT_DIR)}   {p.stat().st_size / 1e6:,.1f} MB")
```

### What you got

```
mrvi_out/
  new_data_malignancy.parquet    one row per one of YOUR cells — leiden, mscore,
                                 cluster_mscore, atlas_neighbour_frac, call_markers,
                                 prob_spread, call_spread, agree
  mrvi_u.npy / mrvi_barcodes.npy the 10-dimensional latent for every merged cell, plus the
                                 barcode order it belongs to
  mrvi_model/                    the trained MrVI model — MRVI.load() to reuse it
  knn.npz                        the 20-nearest-neighbour graph; delete it to force a rebuild
  merged_mrvi_input.h5ad         the merged, shared-gene counts object
  figures/                       the two figures above
```

Joining the calls back onto your original object:

```python
import pandas as pd, scanpy as sc
calls = pd.read_parquet("mrvi_out/new_data_malignancy.parquet")
adata = sc.read_h5ad("your_original.h5ad")
adata.obs["malignant_markers"] = calls["call_markers"].reindex(adata.obs_names)
adata.obs["malignant_spread"]  = calls["call_spread"].reindex(adata.obs_names)
adata.obs["p_malignant"]       = calls["prob_spread"].reindex(adata.obs_names)
```

Cells you did not gate as CD4 T come back as `NaN`, deliberately — they were never scored.

## What this tells you, and what it does not

### The numbers to quote

Measured on our atlas by holding out one entire **study** at a time — four held-out cohorts, whose
anchors the fit never saw:

| arm | median balanced accuracy | range over the four folds |
|---|---|---|
| MrVI latent, 10,000 genes, label spreading | **0.906** | 0.902 – 0.926 |
| MrVI latent, 1,000-gene panel, label spreading | **0.820** | 0.797 – 0.826 |
| cluster + markers, with our CTCL list | 0.981 | 0.846 – 0.983 |

Which row applies to you is decided by the regime Step 3 printed: a targeted panel is the second
row, whole-transcriptome input the first.

### Five things to keep in mind

**1. The split is leave-one-study-out, not leave-one-lab-out.** The held-out cohorts are still
human skin CTCL on 10x 5'. If your data is nuclei, or Xenium, or blood, or a different disease,
you are further out than anything we measured. An earlier version of this analysis grouped folds
by *donor* and reported 0.943 AUC; that was optimistic, because a held-out donor's neighbours
still come from its own study and its own chemistry. The lower number is the honest one.

**2. Arm (b)'s anchors are ours, and they are TCR-clonality anchors.** `malignant` means "carries
the donor's dominant clone". Malignant cells that lost the clonotype call, and reactive cells
sitting inside an expanded benign clone, are mislabelled at source, and label spreading inherits
that.

**3. Arm (b) inherits a prevalence prior.** Roughly 70% of our anchors are malignant, and
spreading pulls toward that; we called 60–70% of cells malignant on our own cohorts. If your
cohort carries far less tumour burden, expect over-calling, and read `prob_spread` as a ranking
rather than a calibrated probability.

**4. Arm (a) scored beautifully for us, and its null is not zero.** On one fold, expression-matched
*random* marker sets still reached balanced accuracy 0.90 — because the score is constant within a
cluster, so wherever the clustering already separates tumour from benign, almost any labelling of
those clusters inherits the separation. A good cluster-level score is partly a statement about
your clustering. The per-cell `mscore` column is the one that discriminates.

**5. The agreement between the arms is the result.** Cells both arms call malignant are the
defensible set. The clusters where they disagree are a to-do list, not noise.

### Our marker list, if you want a starting point

From the study this notebook is drawn from, direction fixed in advance and never flipped:

- **up** — `TOX`, `GATA3`, `PLS3`, `TWIST1`, `KIR3DL2`, `TNFRSF8`, `CDK6`, `TRAF1`, `DNM3`
- **dn** — `CD7`, `DPP4`, `SATB1`, `STAT4`

It is written here rather than pre-filled into `MARKERS` on purpose: a default in the code becomes
everybody's answer without anybody deciding to use it.

All thirteen are in the atlas's gene space, so whether they survive into your analysis depends
only on whether *your* data carries them. Step 6 prints exactly which ones did, and it is worth
reading that line: in our own hands, losing a single marker from the `dn` side moved this arm's
balanced accuracy from 0.98 to 0.86 on one fold. Marker coverage is not a detail here.
