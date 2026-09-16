# Build spec: TCR-anchored malignant classifiers on the MrVI embedding

## Objective

Build two fresh Jupyter notebooks that classify malignant vs benign T cells in the CTCL atlas using TCR clonotype as semi-supervised anchors on the MrVI `u` (sample-unaware) embedding.

- Notebook 1 (`nb1_labelspreading.ipynb`) = Option 1: `LabelSpreading` on the `u`-KNN graph.
- Notebook 2 (`nb2_nearest_reference.ipynb`) = Option 2A: nearest class-reference in MrVI-denoised expression space.

Each notebook is self-contained (notebooks do not import each other). They communicate only through a shared on-disk calls file. Each notebook ends with the same benchmarking section, which renders whatever method outputs exist so far.

This is a transcriptomic-state classifier (CNV is intentionally dropped). It will be strong on advanced Th2 tumors and weak where malignant ≈ reactive (early stage). Surface that, do not hide it.

---

## CONFIG cell (identical, top of BOTH notebooks)

Create one config cell. Verify every column/key exists before proceeding; if any is missing, print the available `adata.obs.columns` / `adata.obsm.keys()` and stop with a clear message (decision gate).

```python
ADATA_PATH   = "<SET: path to atlas .h5ad with MrVI u, TCR, lineage, donor>"
MRVI_MODEL   = "<SET: trained MrVI model dir, OR None if u already in obsm>"
OUT          = "/mnt/user-data/outputs"
CALLS_CSV    = f"{OUT}/calls.csv"

U_KEY        = "X_mrvi_u"      # obsm key for the sample-unaware u embedding; VERIFY
SAMPLE_KEY   = "donor"         # obs: donor / sample id
STUDY_KEY    = "study"         # obs: cohort / study of origin
STAGE_KEY    = "stage"         # obs: early/advanced (None if absent)
LINEAGE_KEY  = "cell_type"     # obs: column used to select T cells
TCELL_VALUES = ["T cell", "CD4 T", "CD8 T"]   # VERIFY against the data
CLONOTYPE_KEY= "clonotype_id"  # obs: TCR clonotype id, NaN where no TCR

# anchor thresholds
MIN_DOM_SIZE   = 20    # dominant clone must have >= this many cells
DOM_RATIO      = 3.0   # top1/top2 clone-size ratio to accept a dominant clone
BENIGN_MAX_PCT = 5     # benign anchors: clonotype size <= this percentile of TCR+ sizes (and not dominant)
MIN_ANCHORS    = 30    # per class, global floor before trusting the fit

SEED = 0
```

---

## SHARED PREPROCESSING (identical first cells in BOTH notebooks)

1. Load `adata`. Restrict the classification pool to T cells: `adata = adata[adata.obs[LINEAGE_KEY].isin(TCELL_VALUES)].copy()`. Keep barcodes in `adata.obs_names`.

2. Clone sizes: `clone_size` = count of cells per `(SAMPLE_KEY, CLONOTYPE_KEY)`. No-TCR cells → `clone_size = 0`.

3. Dominant clone per donor: within each donor, rank clonotypes by size. Accept a dominant clone only if `top1_size >= MIN_DOM_SIZE` and `top1_size/top2_size >= DOM_RATIO`. Donors failing this have no positive anchors (flag them).

4. Anchors → `adata.obs["tcr_label"]` in {`malignant`,`benign`,`unlabeled`}:
   - `malignant`: cell belongs to its donor's accepted dominant clone.
   - `benign`: TCR+ cell, not in any dominant clone, with clone size ≤ the `BENIGN_MAX_PCT` percentile of all TCR+ clone sizes.
   - `unlabeled`: everything else, including all no-TCR cells and all Flex cells. NEVER label no-TCR as benign.
   - Print anchor counts per class overall, per donor, per study. Warn if either class < `MIN_ANCHORS`.

```python
import numpy as np, pandas as pd
o = adata.obs
size = o.groupby([SAMPLE_KEY, CLONOTYPE_KEY]).size().rename("sz")
o = o.join(size, on=[SAMPLE_KEY, CLONOTYPE_KEY]); o["sz"] = o["sz"].fillna(0)
adata.obs["clone_size"] = o["sz"].values

dom = {}
for d, g in o.dropna(subset=[CLONOTYPE_KEY]).groupby(SAMPLE_KEY):
    s = g.groupby(CLONOTYPE_KEY)["sz"].first().sort_values(ascending=False)
    if len(s) and s.iloc[0] >= MIN_DOM_SIZE and (len(s) == 1 or s.iloc[0]/s.iloc[1] >= DOM_RATIO):
        dom[d] = s.index[0]

tcrpos = o[CLONOTYPE_KEY].notna()
ben_cut = np.percentile(o.loc[tcrpos, "sz"], BENIGN_MAX_PCT)
is_mal = o.apply(lambda r: dom.get(r[SAMPLE_KEY]) == r[CLONOTYPE_KEY], axis=1)
is_ben = tcrpos & (~is_mal) & (o["sz"] <= ben_cut)
lab = np.where(is_mal, "malignant", np.where(is_ben, "benign", "unlabeled"))
adata.obs["tcr_label"] = lab
```

5. `u` embedding: `U = adata.obsm[U_KEY]`. If absent, load `MRVI_MODEL` and call `MRVI.get_latent_representation(give_z=False)` (return the sample-unaware `u`, NOT `z`); verify the method signature for the installed scvi-tools version.

6. UMAP on `u`: if no `u`-derived UMAP exists, `sc.pp.neighbors(adata, use_rep=U_KEY)`, `sc.tl.umap(adata)`, store as `adata.obsm["X_umap_u"]`. Reuse if present.

7. Handoff base: write/merge `barcode, SAMPLE_KEY, STUDY_KEY, STAGE_KEY, tcr_label, clone_size` into `CALLS_CSV` (index = barcode). Each method later adds its own columns to this file.

---

## NOTEBOOK 1 — Option 1: LabelSpreading on the u-KNN graph

Params: `K=20`, `ALPHA=0.2`.

```python
from sklearn.semi_supervised import LabelSpreading
y = np.where(adata.obs.tcr_label.eq("malignant"), 1,
     np.where(adata.obs.tcr_label.eq("benign"), 0, -1))
ls = LabelSpreading(kernel="knn", n_neighbors=K, alpha=ALPHA, max_iter=60).fit(U, y)
mal_col = list(ls.classes_).index(1)
prob_m1 = ls.label_distributions_[:, mal_col]
```

- Calibrate the threshold on anchors via the CV in the Validation section; default 0.5 if CV unavailable.
- `call_m1 = (prob_m1 >= thr_m1).astype(int)`.
- Merge `prob_m1, call_m1` into `CALLS_CSV` on barcode.
- Notebook-1 UMAP panel: 3 subplots on `X_umap_u` colored by `tcr_label` | `prob_m1` | `call_m1`.
- Run the SHARED BENCHMARKING section.

Edge cases: report anchor coverage per Leiden/cell-type cluster; if a region has no anchors within K hops, `prob_m1` stays near the prior there — flag those cells. Confirm `ls.classes_` ordering before indexing the malignant column.

---

## NOTEBOOK 2 — Option 2A: nearest class-reference in denoised expression

Params: `metric="pearson"`, `STANDARDIZE=True`, `MARGIN_MIN=0.02`.

1. Denoised expression `D` (cells × genes), restricted to the MrVI HVG set:
   - Preferred: MrVI normalized expression if the installed version exposes it (check `MRVI.get_normalized_expression`).
   - Fallback: `sc.pp.normalize_total(adata, 1e4); sc.pp.log1p(adata)` on the MrVI HVGs.
   - Document which path was used in a printed line.
2. If `STANDARDIZE`, z-score each gene across the T-cell pool so high-expression genes don't dominate the correlation.
3. References from anchors: `m = D[malignant].mean(0)`, `b = D[benign].mean(0)`.
4. Per-cell score and probability:

```python
def corr_rows(X, v):           # Pearson of each row of X with vector v
    Xc = X - X.mean(1, keepdims=True); vc = v - v.mean()
    return (Xc @ vc) / (np.linalg.norm(Xc,axis=1)*np.linalg.norm(vc) + 1e-9)
s = corr_rows(D, m) - corr_rows(D, b)          # >0 leans malignant
# map s -> prob via 1D logistic fit on anchors
from sklearn.linear_model import LogisticRegression
amask = adata.obs.tcr_label.isin(["malignant","benign"]).values
ya = (adata.obs.tcr_label[amask] == "malignant").astype(int).values
lr = LogisticRegression().fit(s[amask].reshape(-1,1), ya)
prob_m2 = lr.predict_proba(s.reshape(-1,1))[:,1]
```

- Threshold `thr_m2` from anchor CV; `call_m2 = (prob_m2 >= thr_m2).astype(int)`.
- Uncertain flag: `abs(s) < MARGIN_MIN` → `uncertain_m2 = True` (this concentrates on early-stage overlap — expected).
- Merge `prob_m2, call_m2` into `CALLS_CSV`.
- Notebook-2 UMAP panel: `tcr_label` | `prob_m2` | `call_m2` on `X_umap_u`.
- Run the SHARED BENCHMARKING section.

Optional extension (note only, do not build unless asked): fit stage-specific references `m_early`, `m_adv` since the malignant program differs by stage in this disease.

---

## SHARED BENCHMARKING (append verbatim to BOTH notebooks; idempotent)

Load `CALLS_CSV`, merge onto `adata.obs` by barcode. Use whatever method columns are present.

### A) UMAP benchmark → `{OUT}/umap_benchmark.png`
On `X_umap_u`:
- Always: `tcr_label` (malignant / benign / unlabeled — fixed 3-color map) and this notebook's proposed `call_*`.
- If both `call_m1` and `call_m2` exist: 4-panel figure [`tcr_label` | `call_m1` | `call_m2` | agreement], where agreement ∈ {agree-malignant, agree-benign, disagree}.

### B) Per-cell track heatmap → `{OUT}/cell_track_heatmap.png`
Cells along the x-axis, 4 stacked horizontal tracks (one row each), each with its own colormap + colorbar:
- `prob_m1` (or `call_m1` if prob absent) — sequential 0→1.
- `prob_m2` (or `call_m2`) — sequential 0→1.
- `tcr_status` — discrete: malignant / benign / grey for unlabeled.
- `clone_size` — `log1p`, sequential.

Sort cells by ensemble score `mean(prob_m1, prob_m2)` descending (if only one method present, sort by it) so malignant cells cluster at the left. Keep all anchors; if rendering is slow, subsample unlabeled cells to ≤ 20k but never drop anchor cells. The figure should make visible: do high-prob cells coincide with TCR-malignant status and large clones, and where do M1 and M2 disagree.

```python
import matplotlib.pyplot as plt
tracks = ["prob_m1","prob_m2","tcr_num","clone_log"]   # build tcr_num: mal=1,ben=0,unl=nan; clone_log=log1p(clone_size)
order = np.argsort(-ens)                                 # ens = nanmean of available probs
fig, axes = plt.subplots(len(tracks), 1, figsize=(14, 4), sharex=True)
cmaps = {"prob_m1":"viridis","prob_m2":"viridis","tcr_num":"coolwarm","clone_log":"magma"}
for ax, t in zip(axes, tracks):
    im = ax.imshow(df[t].values[order][None,:], aspect="auto", cmap=cmaps[t])
    ax.set_yticks([]); ax.set_ylabel(t, rotation=0, ha="right", va="center")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
axes[-1].set_xlabel("cells (sorted by ensemble malignant score)")
plt.tight_layout(); plt.savefig(f"{OUT}/cell_track_heatmap.png", dpi=150)
```

### C) Concordance summary → `{OUT}/concordance_summary.csv` (and print)
- Each available method vs TCR anchors (anchors only): precision, recall, F1, balanced accuracy.
- If both present: M1↔M2 agreement rate overall, and broken out per `STAGE_KEY` and per `STUDY_KEY`.

---

## VALIDATION (anchor CV — sets the call thresholds in both notebooks)

- `StratifiedGroupKFold`, 5 folds, groups = `SAMPLE_KEY`. Repeat once more with groups = `STUDY_KEY` to test cross-study transfer.
- Per fold: fit/score on train anchors, choose the threshold maximizing F1 on train, evaluate held-out anchors. Report mean ± sd F1 and AUC overall, per stage, per study.
- Use the CV-selected threshold as the notebook's `thr_m1` / `thr_m2`.
- Expect early-stage and cross-study folds to be the weak points; report them rather than averaging them away.

---

## OUTPUTS

- `nb1_labelspreading.ipynb`, `nb2_nearest_reference.ipynb`
- `{OUT}/calls.csv` — barcode, donor, study, stage, tcr_label, clone_size, prob_m1, call_m1, prob_m2, call_m2
- `{OUT}/umap_benchmark.png`
- `{OUT}/cell_track_heatmap.png`
- `{OUT}/concordance_summary.csv`

## DECISION GATES (summary)

- Stop if any CONFIG column/key is missing (print what's available).
- Warn if a class has < `MIN_ANCHORS` global anchors.
- Donors with no accepted dominant clone contribute no positive anchors; their cells are predicted by transfer and flagged low-confidence.
- Never label no-TCR cells as benign.
- Flag `uncertain` where M1 and M2 disagree, or 2A margin < `MARGIN_MIN`, or prob ∈ [0.4, 0.6].
- Run globally (the `u` space is already integrated); validate and report per study and per stage.
