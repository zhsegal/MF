# old/16_mrvi_labelspreading_malignancy.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/16_mrvi_labelspreading_malignancy.ipynb` and is never run._

# 16 · MrVI LabelSpreading malignancy classifier

TCR-anchored **malignant vs benign** T cells via `LabelSpreading` on the **MrVI sample-unaware `u`** KNN graph — a CNV-free, transcriptomic-state classifier (Option 1 of `data/build_mrvi_malignant_classifiers.md`).

- **Anchors** reuse the canonical per-donor dominant TCR clone (mirrors nb14/15): malignant = dominant clone; benign = TCR+, non-dominant, unexpanded clone; everything else (incl. all no-TCR cells) = unlabeled.
- Self-contained; shares `data/atlas_joint/mrvi_malignancy/calls.csv` with `17_mrvi_nearest_reference_malignancy`.
- Expected to be **strong on advanced Th2 tumors, weak where malignant ≈ reactive** (early stage) — surfaced per stage/study, not hidden.

```python  # cell 1
# ============================ CONFIG (identical in nb16/nb17 except method params) ============================
import sys, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(start)


NB_DIR = _resolve_nb_dir()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))
import importlib
import atlas_join_helpers as H
import skin_T_cnv_helpers as C
importlib.reload(H); importlib.reload(C)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1

# ---- paths ----
V2        = NB_DIR / "data/atlas_joint/skin_T_tcr_malig_v2.h5ad"       # curated TCR-complete T object (has X_mrvi_u)
HVG_INPUT = NB_DIR / "data/atlas_joint/joint_mrvi_input.h5ad"          # MrVI HVG var set (nb17 only)
UMAP_NPY  = NB_DIR / "data/atlas_joint/skin_T_umap_u.npy"              # precomputed u-UMAP
OUT       = NB_DIR / "data/atlas_joint/mrvi_malignancy"; OUT.mkdir(parents=True, exist_ok=True)
CALLS_CSV = OUT / "calls.csv"

# ---- obs / obsm keys ----
U_KEY = "X_mrvi_u"; SAMPLE_KEY = "donor"; STUDY_KEY = "study"; STAGE_KEY = "stage_class"; LINEAGE_KEY = "lineage"

# ---- anchors: malignant = carried ALICE call; benign = TCR+ non-malignant unexpanded ----
MIN_ANCHORS = 30

# ---- method params (nb16: LabelSpreading on the u-KNN graph) ----
K, ALPHA = 20, 0.2

# ---- M1 heavy-step cache (CV + final fit) ----
REFIT_M1 = False                       # reuse nb16_m1_cache.npz; set True to rebuild when the cell set / tcr_label anchors / K / ALPHA change
M1_CACHE = OUT / "nb16_m1_cache.npz"   # {thr_m1, prob_m1, call_m1, prob_m1_oof} keyed by barcode
```

```python  # cell 2
# ============================ SHARED PREPROCESSING (identical in nb16/nb17) ============================
# ---- load the curated TCR-complete T object (242,959 cells; carries ALICE call + X_mrvi_u) ----
adata = sc.read_h5ad(V2)
# ALICE tumor call is the malignant truth; it also drives the benign anchor exclusion below
alice = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["tcr_is_malignant"]      = alice
adata.obs["tcr_is_dominant_clone"] = alice
# tcr_is_expanded / has_tcr / tcr_clone_size / tcr_clone_id come from the file

# ---- decision gate: required obs columns + u embedding ----
need = [SAMPLE_KEY, STUDY_KEY, LINEAGE_KEY, "has_tcr", "tra_cdr3", "trb_cdr3"]
miss = [c for c in need if c not in adata.obs.columns]
stage_key = (STAGE_KEY if STAGE_KEY in adata.obs.columns
             else "disease_stage" if "disease_stage" in adata.obs.columns else None)
if miss or U_KEY not in adata.obsm or stage_key is None:
    print("obs.columns:", list(adata.obs.columns))
    print("obsm.keys :", list(adata.obsm.keys()))
    raise SystemExit(f"DECISION GATE — missing cols={miss}  u_present={U_KEY in adata.obsm}  stage_key={stage_key}")
if "cell_type_broad" in adata.obs.columns:
    assert adata.obs["cell_type_broad"].astype(str).eq("T").all(), "object should be all T cells"
print(f"stage_key = {stage_key}  |  cells = {adata.n_obs}  |  u dim = {adata.obsm[U_KEY].shape[1]}")

# ---- anchors: malignant = ALICE call; benign = TCR+ non-malignant, unexpanded clone ----
o = adata.obs
has = o["has_tcr"].astype(bool).to_numpy()
mal = o["tcr_is_malignant"].astype(bool).to_numpy()
ben = has & ~mal & ~o["tcr_is_expanded"].astype(bool).to_numpy()
lab = np.where(mal, "malignant", np.where(ben, "benign", "unlabeled"))   # no-TCR -> unlabeled, never benign
adata.obs["tcr_label"] = pd.Categorical(lab, categories=["malignant", "benign", "unlabeled"])

vc = adata.obs["tcr_label"].value_counts()
print("\nanchor counts:\n", vc.to_string())
for cls in ("malignant", "benign"):
    if int(vc.get(cls, 0)) < MIN_ANCHORS:
        warnings.warn(f"class '{cls}' has {int(vc.get(cls, 0))} < MIN_ANCHORS={MIN_ANCHORS}")
print("\nanchors per study:\n",
      adata.obs.groupby(STUDY_KEY, observed=True)["tcr_label"]
      .value_counts().unstack(fill_value=0).to_string())

# ---- u embedding + UMAP on u (recompute if the cached npy is stale for this cell set) ----
U = np.asarray(adata.obsm[U_KEY])
if UMAP_NPY.exists() and np.load(UMAP_NPY).shape[0] == adata.n_obs:
    adata.obsm["X_umap_u"] = np.load(UMAP_NPY)
    print("\nloaded u-UMAP:", UMAP_NPY.name)
else:
    sc.pp.neighbors(adata, use_rep=U_KEY, random_state=SEED)   # HEAVY
    sc.tl.umap(adata, random_state=SEED)
    adata.obsm["X_umap_u"] = adata.obsm["X_umap"]
    np.save(UMAP_NPY, adata.obsm["X_umap_u"]); print("\ncomputed + saved u-UMAP ->", UMAP_NPY.name)

# ---- handoff base into shared calls.csv (index = barcode) ----
base = pd.DataFrame({
    SAMPLE_KEY: o[SAMPLE_KEY].astype(str).values,
    STUDY_KEY:  o[STUDY_KEY].astype(str).values,
    "stage":    o[stage_key].astype(str).values,
    LINEAGE_KEY: o[LINEAGE_KEY].astype(str).values,
    "tcr_label": adata.obs["tcr_label"].astype(str).values,
    "clone_size": o["tcr_clone_size"].astype(int).values,
}, index=adata.obs_names)
base.index.name = "barcode"
if CALLS_CSV.exists():
    prev = pd.read_csv(CALLS_CSV, index_col=0)
    extra = [c for c in prev.columns if c not in base.columns]   # prob_*/call_* written by the other nb
    base = base.join(prev[extra])                                # align on barcode; rows absent in prev -> NaN
base.to_csv(CALLS_CSV)
print("calls base ->", CALLS_CSV, base.shape)
```

```
stage_key = stage_class  |  cells = 242959  |  u dim = 10

anchor counts:
 tcr_label
unlabeled    133669
malignant     68620
benign        40670

anchors per study:
 tcr_label        malignant  benign  unlabeled
study                                        
brunner2024           5091    2477      18210
chennareddy2025      20335    6893      25087
herrera2021            295     271        353
li2024               41818   30870      89125
rindler2021_fi        1081     159        894

loaded u-UMAP: skin_T_umap_u.npy
calls base -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/mrvi_malignancy/calls.csv (242959, 11)
```

## Validation — anchor CV (sets `thr_m1`)

`StratifiedGroupKFold` (5), groups = donor then study. Each fold refits LabelSpreading on the train anchors (held-out anchors masked to unlabeled) and predicts the held-out anchors. The F1-optimal threshold is picked on the **pooled out-of-fold** probabilities — the diluted regime that matches the unlabeled cells we classify, *not* the clamped in-sample anchors (`alpha=0.2` pins those to ~0/1, so an in-sample threshold is degenerate and mis-transfers). Reports F1/AUC overall, per stage, per study — early-stage & cross-study folds are the expected weak points. **HEAVY** (~10 full fits).

```python  # cell 4
# anchor CV -> thr_m1 (OOF-calibrated)  ·  HEAVY (~10 LabelSpreading fits) unless cached
if (not REFIT_M1) and M1_CACHE.exists():
    z = np.load(M1_CACHE, allow_pickle=True)
    assert list(z["barcodes"]) == list(adata.obs_names.astype(str)), "M1 cache barcode mismatch -> set REFIT_M1=True"
    thr_m1 = float(z["thr_m1"]); adata.obs["prob_m1_oof"] = z["prob_m1_oof"]
    print(f"loaded M1 cache -> thr_m1={thr_m1:.3f}  (set REFIT_M1=True to recompute)")
else:
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

    amask = adata.obs["tcr_label"].isin(["malignant", "benign"]).to_numpy()
    aidx = np.where(amask)[0]
    ya = (adata.obs["tcr_label"].to_numpy()[amask] == "malignant").astype(int)
    stage_a = adata.obs[stage_key].astype(str).to_numpy()[amask]
    study_a = adata.obs[STUDY_KEY].astype(str).to_numpy()[amask]

    def _best_f1_thr(y, p):
        ok = np.isfinite(p); y, p = y[ok], p[ok]
        best = (0.5, 0.0)
        for t in np.unique(np.quantile(p, np.linspace(0.05, 0.95, 19))):
            f = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f > best[1]:
                best = (float(t), f)
        return best[0]

    def run_cv(group_key):
        """5-fold StratifiedGroupKFold; returns pooled out-of-fold anchor probs.
        Threshold is picked on the OOF (diluted) probs — the same regime as the unlabeled
        cells we actually classify — NOT on the clamped in-sample anchors (alpha=0.2 pins
        those to ~0/1, so an in-sample threshold is degenerate and mis-transfers)."""
        groups = adata.obs[group_key].astype(str).to_numpy()[amask]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.full(amask.sum(), np.nan)
        for tr, te in sgkf.split(np.zeros(amask.sum()), ya, groups):
            y_full = np.full(adata.n_obs, -1)
            y_full[aidx[tr]] = ya[tr]                       # held-out anchors masked to unlabeled
            ls = LabelSpreading(kernel="knn", n_neighbors=K, alpha=ALPHA, max_iter=60).fit(U, y_full)
            p = ls.label_distributions_[:, list(ls.classes_).index(1)]
            oof[te] = p[aidx[te]]
        thr = _best_f1_thr(ya, oof)                         # single F1-optimal threshold on pooled OOF
        yhat = (oof >= thr).astype(int)
        def rep(mask, name):
            if mask.sum() < 20 or len(np.unique(ya[mask])) < 2:
                print(f"  {name:22s} n={int(mask.sum()):6d}  (too few / one class)"); return
            print(f"  {name:22s} n={int(mask.sum()):6d}  F1={f1_score(ya[mask], yhat[mask], zero_division=0):.3f}"
                  f"  AUC={roc_auc_score(ya[mask], oof[mask]):.3f}"
                  f"  balAcc={balanced_accuracy_score(ya[mask], yhat[mask]):.3f}")
        print(f"[CV grouped by {group_key}]  thr={thr:.3f}")
        rep(np.ones(len(ya), bool), "overall")
        for s in np.unique(stage_a): rep(stage_a == s, f"stage={s}")
        for s in np.unique(study_a): rep(study_a == s, f"study={s}")
        return thr, oof

    thr_m1, oof_m1 = run_cv(SAMPLE_KEY)        # donor-grouped -> sets the threshold + OOF validation preds
    _ = run_cv(STUDY_KEY)                      # study-grouped -> cross-study transfer report

    # persist OOF (honest, held-out) anchor probs for benchmarking — NaN for non-anchors
    oof_full = np.full(adata.n_obs, np.nan); oof_full[aidx] = oof_m1
    adata.obs["prob_m1_oof"] = oof_full
    print("\nselected thr_m1 =", round(thr_m1, 3), " (OOF-calibrated)")
```

```
[CV grouped by donor]  thr=0.489
  overall                n=109290  F1=0.909  AUC=0.943  balAcc=0.869
  stage=HC               n=   714  (too few / one class)
  stage=advanced         n= 62152  F1=0.921  AUC=0.943  balAcc=0.875
  stage=early            n= 46424  F1=0.891  AUC=0.969  balAcc=0.887
  study=brunner2024      n=  7568  F1=0.800  AUC=0.798  balAcc=0.710
  study=chennareddy2025  n= 27228  F1=0.929  AUC=0.949  balAcc=0.880
  study=herrera2021      n=   566  F1=0.889  AUC=0.953  balAcc=0.876
  study=li2024           n= 72688  F1=0.910  AUC=0.956  balAcc=0.883
  study=rindler2021_fi   n=  1240  F1=0.991  AUC=0.998  balAcc=0.983
[CV grouped by study]  thr=0.504
  overall                n=109290  F1=0.878  AUC=0.904  balAcc=0.807
  stage=HC               n=   714  (too few / one class)
  stage=advanced         n= 62152  F1=0.917  AUC=0.931  balAcc=0.854
  stage=early            n= 46424  F1=0.816  AUC=0.928  balAcc=0.794
  study=brunner2024      n=  7568  F1=0.806  AUC=0.808  balAcc=0.717
  study=chennareddy2025  n= 27228  F1=0.911  AUC=0.947  balAcc=0.877
  study=herrera2021      n=   566  F1=0.884  AUC=0.948  balAcc=0.870
  study=li2024           n= 72688  F1=0.870  AUC=0.926  balAcc=0.808
  study=rindler2021_fi   n=  1240  F1=0.984  AUC=0.998  balAcc=0.976

selected thr_m1 = 0.489  (OOF-calibrated)
```

```python  # cell 5
# final LabelSpreading on ALL anchors  ·  HEAVY (1 full fit) unless cached -> writes M1_CACHE
if (not REFIT_M1) and M1_CACHE.exists():
    z = np.load(M1_CACHE, allow_pickle=True)
    assert list(z["barcodes"]) == list(adata.obs_names.astype(str)), "M1 cache barcode mismatch -> set REFIT_M1=True"
    prob_m1, call_m1 = z["prob_m1"], z["call_m1"]
    print(f"loaded M1 cache -> called malignant {int(call_m1.sum())}/{adata.n_obs}  thr_m1={float(z['thr_m1']):.3f}")
else:
    from sklearn.semi_supervised import LabelSpreading
    y = np.where(adata.obs["tcr_label"].eq("malignant"), 1,
         np.where(adata.obs["tcr_label"].eq("benign"), 0, -1))
    ls = LabelSpreading(kernel="knn", n_neighbors=K, alpha=ALPHA, max_iter=60).fit(U, y)
    prob_m1 = ls.label_distributions_[:, list(ls.classes_).index(1)]   # confirm class ordering
    call_m1 = (prob_m1 >= thr_m1).astype(int)                          # thr_m1 OOF-calibrated in CV cell
    np.savez(M1_CACHE, barcodes=np.array(adata.obs_names.astype(str)), thr_m1=thr_m1,
             prob_m1=prob_m1, call_m1=call_m1, prob_m1_oof=adata.obs["prob_m1_oof"].to_numpy())
    print(f"called malignant: {int(call_m1.sum())}/{adata.n_obs}  | saved M1 cache -> {M1_CACHE.name}")

adata.obs["prob_m1"] = prob_m1
adata.obs["call_m1"] = call_m1
adata.obs["lowconf_m1"] = np.abs(prob_m1 - 0.5) < 0.05                  # anchor-sparse / undecided region
print(f"low-confidence: {int(adata.obs['lowconf_m1'].sum())}")

# anchor coverage per cell-type cluster (flag anchor-sparse regions: prob stays near prior there)
if "cell_type" in adata.obs.columns:
    cov = (adata.obs.assign(anchor=adata.obs["tcr_label"].ne("unlabeled"))
           .groupby("cell_type", observed=True)["anchor"].mean().sort_values())
    print("\nlowest anchor coverage (cell_type):\n", cov.head(8).round(3).to_string())

# merge into calls.csv (prob_m1 = in-sample final fit; prob_m1_oof = honest held-out for anchors)
calls = pd.read_csv(CALLS_CSV, index_col=0)
calls.loc[adata.obs_names, "prob_m1"] = prob_m1
calls.loc[adata.obs_names, "call_m1"] = call_m1
calls.loc[adata.obs_names, "prob_m1_oof"] = adata.obs["prob_m1_oof"].values
calls.to_csv(CALLS_CSV)
print("\nmerged prob_m1/call_m1/prob_m1_oof ->", CALLS_CSV)
```

```
called malignant: 126240/242959  | saved M1 cache -> nb16_m1_cache.npz
low-confidence: 5152

lowest anchor coverage (cell_type):
 cell_type
B_cell     0.000
DC2        0.000
F2         0.000
Macro_2    0.000
NK         0.012
ILC1_NK    0.014
ILC1_3     0.039
DC1        0.075

merged prob_m1/call_m1/prob_m1_oof -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/mrvi_malignancy/calls.csv
```

```python  # cell 6
# nb16 UMAP panel on the u-derived UMAP: tcr_label | prob_m1 | call_m1  (mirrors nb17)
import matplotlib.pyplot as plt
UM = adata.obsm["X_umap_u"]; lab = adata.obs["tcr_label"].astype(str).values
TCRC = {"malignant": "#d62728", "benign": "#1f77b4", "unlabeled": "#dddddd"}
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for k in ["unlabeled", "benign", "malignant"]:
    m = lab == k
    ax[0].scatter(UM[m, 0], UM[m, 1], s=2, c=TCRC[k], label=k, linewidths=0)
ax[0].legend(markerscale=3, fontsize=7); ax[0].set_title("tcr_label")
s1 = ax[1].scatter(UM[:, 0], UM[:, 1], s=2, c=prob_m1, cmap="viridis", linewidths=0)
plt.colorbar(s1, ax=ax[1], fraction=0.04); ax[1].set_title("prob_m1")
cm = np.where(call_m1 == 1, "#d62728", "#1f77b4")
ax[2].scatter(UM[:, 0], UM[:, 1], s=2, c=cm, linewidths=0)
ax[2].set_title(f"call_m1 (thr={thr_m1:.2f})")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "nb16_umap_panel.png", dpi=150); plt.show()
```

```
<Figure size 1300x400 with 4 Axes>
```

## Shared benchmarking (identical in nb16 / nb17)

Reloads `calls.csv` and renders whatever method columns exist so far: UMAP benchmark, per-cell track
heatmap, concordance vs TCR anchors. Run nb16 then nb17 to populate both methods. The out-of-fold
(honest, held-out) validation for both methods is at the **end of nb17**.

```python  # cell 8
# ============================ SHARED BENCHMARKING (identical in nb16/nb17) ============================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

calls = pd.read_csv(CALLS_CSV, index_col=0).reindex(adata.obs_names)
UM = adata.obsm["X_umap_u"]
TCRC = {"malignant": "#d62728", "benign": "#1f77b4", "unlabeled": "#dddddd"}
callmap = {0: "benign", 1: "malignant"}
has_m1 = ("call_m1" in calls.columns) and calls["call_m1"].notna().any()
has_m2 = ("call_m2" in calls.columns) and calls["call_m2"].notna().any()

# ---- A) UMAP benchmark ----
def _disc_scatter(ax, series, title, colors):
    order = [k for k in colors if k in ("unlabeled", "agree-benign")] + \
            [k for k in colors if k not in ("unlabeled", "agree-benign")]
    for k in order:
        m = (series == k).values
        if m.any():
            ax.scatter(UM[m, 0], UM[m, 1], s=2, c=colors[k], label=k, linewidths=0)
    ax.legend(markerscale=3, fontsize=6, loc="best"); ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

panels = [("tcr_label", calls["tcr_label"], TCRC)]
if has_m1: panels.append(("call_m1", calls["call_m1"].map(callmap), TCRC))
if has_m2: panels.append(("call_m2", calls["call_m2"].map(callmap), TCRC))
if has_m1 and has_m2:
    agree = np.where((calls.call_m1 == 1) & (calls.call_m2 == 1), "agree-malignant",
             np.where((calls.call_m1 == 0) & (calls.call_m2 == 0), "agree-benign", "disagree"))
    panels.append(("agreement", pd.Series(agree, index=calls.index),
                   {"agree-malignant": "#d62728", "agree-benign": "#1f77b4", "disagree": "#ff7f0e"}))
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
axes = np.atleast_1d(axes)
for ax, (t, c, cols) in zip(axes, panels):
    _disc_scatter(ax, c, t, cols)
plt.tight_layout(); plt.savefig(OUT / "umap_benchmark.png", dpi=150); plt.show()

# ---- B) per-cell track heatmap ----
prob1 = calls["prob_m1"].values if "prob_m1" in calls.columns else np.full(len(calls), np.nan)
prob2 = calls["prob_m2"].values if "prob_m2" in calls.columns else np.full(len(calls), np.nan)
tcr_num = calls["tcr_label"].map({"malignant": 1.0, "benign": 0.0}).values   # unlabeled -> nan
clone_log = np.log1p(calls["clone_size"].fillna(0).values)
probs = [p for p in (prob1, prob2) if np.isfinite(p).any()]
ens = np.nanmean(np.vstack(probs), axis=0) if probs else clone_log

is_anchor = calls["tcr_label"].ne("unlabeled").values
keep = np.ones(len(calls), bool)
unl = np.where(~is_anchor)[0]
if len(unl) > 20000:
    drop = np.random.default_rng(SEED).choice(unl, len(unl) - 20000, replace=False)
    keep[drop] = False
idx = np.where(keep)[0]
order = idx[np.argsort(-ens[idx])]

tracks = [("prob_m1", prob1, "viridis"), ("prob_m2", prob2, "viridis"),
          ("tcr_status", tcr_num, "coolwarm"), ("clone_size (log1p)", clone_log, "magma")]
fig, axes = plt.subplots(len(tracks), 1, figsize=(14, 4), sharex=True)
for ax, (name, arr, cm) in zip(axes, tracks):
    im = ax.imshow(arr[order][None, :], aspect="auto", cmap=cm)
    ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
axes[-1].set_xlabel("cells (sorted by ensemble malignant score)")
plt.tight_layout(); plt.savefig(OUT / "cell_track_heatmap.png", dpi=150); plt.show()

# ---- C) concordance vs TCR anchors ----
am = calls["tcr_label"].isin(["malignant", "benign"]).values
yt = (calls["tcr_label"] == "malignant").astype(int).values
rows = []
for name in ["call_m1", "call_m2"]:
    if name not in calls.columns or calls[name].isna().all():
        continue
    yp = calls[name].values.astype(float)
    m = am & ~np.isnan(yp)
    rows.append(dict(method=name, n_anchor=int(m.sum()),
                     precision=round(precision_score(yt[m], yp[m], zero_division=0), 3),
                     recall=round(recall_score(yt[m], yp[m], zero_division=0), 3),
                     f1=round(f1_score(yt[m], yp[m], zero_division=0), 3),
                     bal_acc=round(balanced_accuracy_score(yt[m], yp[m]), 3)))
conc = pd.DataFrame(rows)
print("concordance vs TCR anchors:\n", conc.to_string(index=False))
if has_m1 and has_m2:
    agreed = calls["call_m1"] == calls["call_m2"]
    print(f"\nM1<->M2 agreement overall: {agreed.mean():.3f}")
    for k in ["stage", STUDY_KEY]:
        print(f"\nM1<->M2 agreement by {k}:")
        print(calls.groupby(k).apply(lambda d: (d.call_m1 == d.call_m2).mean()).round(3).to_string())
conc.to_csv(OUT / "concordance_summary.csv", index=False)
print("\nwrote", OUT / "concordance_summary.csv")
```

```
<Figure size 1600x400 with 4 Axes>
<Figure size 1400x400 with 8 Axes>
concordance vs TCR anchors:
  method  n_anchor  precision  recall    f1  bal_acc
call_m1    109290        1.0   1.000 1.000    1.000
call_m2    109290        0.9   0.937 0.918    0.881

M1<->M2 agreement overall: 0.827

M1<->M2 agreement by stage:
stage
HC          0.751
advanced    0.838
early       0.818

M1<->M2 agreement by study:
study
brunner2024        0.778
chennareddy2025    0.854
herrera2021        0.756
li2024             0.825
rindler2021_fi     0.920

wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/mrvi_malignancy/concordance_summary.csv

/tmp/ipykernel_3649591/1142658847.py:85: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  print(calls.groupby(k).apply(lambda d: (d.call_m1 == d.call_m2).mean()).round(3).to_string())
/tmp/ipykernel_3649591/1142658847.py:85: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  print(calls.groupby(k).apply(lambda d: (d.call_m1 == d.call_m2).mean()).round(3).to_string())
```

```python  # cell 9
# ---- cache the final M1 call for nb18 (bool methods parquet, indexed by obs_name) ----
MRVI_M1_PARQUET = NB_DIR / "data/atlas_joint/skin_T_mrvi_m1_malignancy_methods.parquet"
mcalls = (adata.obs[["call_m1"]]
          .rename(columns={"call_m1": "mrvi_m1_labelspread"}).astype(bool))
mcalls.index.name = "obs_name"
mcalls.to_parquet(MRVI_M1_PARQUET)
print("wrote", MRVI_M1_PARQUET, mcalls.shape, int(mcalls["mrvi_m1_labelspread"].sum()))
```

```python  # cell 10
# ============================ VALIDATION — OOF (honest) for both methods ============================
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             balanced_accuracy_score, roc_auc_score, confusion_matrix)

cv = pd.read_csv(CALLS_CSV, index_col=0)
yt_all = (cv["tcr_label"] == "malignant").astype(int).to_numpy()
anchor = cv["tcr_label"].isin(["malignant", "benign"]).to_numpy()


def _best_f1_thr(y, p):
    best = (0.5, -1.0)
    for t in np.unique(np.quantile(p, np.linspace(0.05, 0.95, 19))):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best[1]:
            best = (float(t), f)
    return best[0]


def _metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(n=int(len(y)), thr=round(thr, 3),
                acc=round(accuracy_score(y, yh), 3),
                precision=round(precision_score(y, yh, zero_division=0), 3),
                recall=round(recall_score(y, yh, zero_division=0), 3),
                f1=round(f1_score(y, yh, zero_division=0), 3),
                balAcc=round(balanced_accuracy_score(y, yh), 3),
                AUC=round(roc_auc_score(y, p), 3) if len(np.unique(y)) > 1 else np.nan)


summary = []
for m, col in [("M1_labelspreading", "prob_m1_oof"), ("M2_pseudosample_mrvi", "prob_m2_oof")]:
    if col not in cv.columns or cv[col].notna().sum() == 0:
        print(f"{m}: {col} absent in calls.csv — run that notebook first"); continue
    p = cv[col].to_numpy()
    val = anchor & np.isfinite(p)
    y, pv = yt_all[val], p[val]
    thr = _best_f1_thr(y, pv)
    cm = confusion_matrix(y, (pv >= thr).astype(int), labels=[1, 0])
    print(f"\n=== {m} — OOF anchors (n={int(val.sum())}, thr={thr:.3f}) ===")
    print(pd.DataFrame(cm, index=["TCR malignant", "TCR benign"],
                       columns=["pred malignant", "pred benign"]).to_string())
    row = _metrics(y, pv, thr)
    summary.append({"scope": "overall", "method": m, **row})
    print("  overall:", {k: row[k] for k in ["acc", "precision", "recall", "f1", "balAcc", "AUC"]})
    for key in ["stage", STUDY_KEY]:
        g = cv[key].astype(str).to_numpy()[val]
        for v in sorted(np.unique(g)):
            mm = g == v
            if mm.sum() < 20 or len(np.unique(y[mm])) < 2:
                continue
            r = _metrics(y[mm], pv[mm], thr)
            summary.append({"scope": f"{key}={v}", "method": m, **r})
            print(f"    {key}={v:18s} acc={r['acc']:.3f}  f1={r['f1']:.3f}  AUC={r['AUC']:.3f}  balAcc={r['balAcc']:.3f}  (n={r['n']})")

val_df = pd.DataFrame(summary)
val_df.to_csv(OUT / "oof_validation_both_methods.csv", index=False)
print("\nwrote", OUT / "oof_validation_both_methods.csv")
print("\noverall quality table (both methods):")
print(val_df[val_df["scope"] == "overall"].to_string(index=False))
val_df[val_df["scope"] == "overall"]
```

```
=== M1_labelspreading — OOF anchors (n=109290, thr=0.489) ===
               pred malignant  pred benign
TCR malignant           63463         5157
TCR benign               7575        33095
  overall: {'acc': 0.884, 'precision': 0.893, 'recall': 0.925, 'f1': 0.909, 'balAcc': 0.869, 'AUC': 0.943}
    stage=advanced           acc=0.886  f1=0.921  AUC=0.943  balAcc=0.875  (n=62152)
    stage=early              acc=0.884  f1=0.891  AUC=0.969  balAcc=0.887  (n=46424)
    study=brunner2024        acc=0.736  f1=0.800  AUC=0.798  balAcc=0.710  (n=7568)
    study=chennareddy2025    acc=0.896  f1=0.929  AUC=0.949  balAcc=0.880  (n=27228)
    study=herrera2021        acc=0.878  f1=0.889  AUC=0.953  balAcc=0.876  (n=566)
    study=li2024             acc=0.893  f1=0.910  AUC=0.956  balAcc=0.883  (n=72688)
    study=rindler2021_fi     acc=0.984  f1=0.991  AUC=0.998  balAcc=0.983  (n=1240)
M2_pseudosample_mrvi: prob_m2_oof absent in calls.csv — run that notebook first

wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/mrvi_malignancy/oof_validation_both_methods.csv

overall quality table (both methods):
  scope            method      n   thr   acc  precision  recall    f1  balAcc   AUC
overall M1_labelspreading 109290 0.489 0.884      0.893   0.925 0.909   0.869 0.943

     scope             method       n    thr    acc  precision  recall     f1  \
0  overall  M1_labelspreading  109290  0.489  0.884      0.893   0.925  0.909   

   balAcc    AUC  
0   0.869  0.943
```
