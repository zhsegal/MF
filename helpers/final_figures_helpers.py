"""Final publication figures for the MF / CTCL atlas (nb22).

Re-renders chosen panels from nb14 (CNV heatmaps), nb16/17 (per-cell track heatmap) and
nb18 (latent UMAP, factor-separation heatmap, OOF F1 / ROC) as editable, hybrid vector/raster
SVGs: text stays vector (``svg.fonttype='none'``), heavy data layers (scatter, ``imshow``) are
rasterized and saved at high dpi, so axes/labels stay light-weight vector paths.

Loads cached results only; the per-factor OOF evaluation (``evaluate_factors``) is the same
logic as nb18 cells 32-33, kept here so nb22 and nb18 share one implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_curve

import semantic_malig_helpers as M


# ---------------------------------------------------------------------------
# SVG style + save
# ---------------------------------------------------------------------------
def svg_style():
    """Editable-text vector defaults: keep glyphs as <text>, not outlined paths."""
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def save_svg(fig, path, dpi=300):
    """Hybrid SVG save: vector axes/text + the rasterized data layers (set per-artist).

    ``dpi`` controls the resolution of the rasterized scatter/imshow blobs only.
    """
    path = Path(path)
    fig.savefig(path, format="svg", dpi=dpi, bbox_inches="tight")
    print("saved", path)


# ---------------------------------------------------------------------------
# (c) latent UMAP — exact nb18 sc.pl.umap call, saved as hybrid SVG (points rasterized)
# ---------------------------------------------------------------------------
def plot_latent_umap(ad_emb, color, out, ncols=2, wspace=0.3, dpi=300):
    """Latent UMAP via the same ``sc.pl.umap`` definition as nb18 cell 13; saved as hybrid SVG.

    Identical call/styling to nb18 (``ncols``, ``wspace``, ``frameon=False``); the scatter
    point collections are flagged rasterized so the SVG keeps text/axes as vector.
    """
    color = [c for c in color if c in ad_emb.obs]
    fig = sc.pl.umap(ad_emb, color=color, ncols=ncols, wspace=wspace, frameon=False,
                     show=False, return_fig=True)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    save_svg(fig, out, dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# (d1) per-factor separation heatmap for one call  (ports M.plot_factor_heatmaps)
# ---------------------------------------------------------------------------
def plot_factor_sep_heatmap(zmat, ad_emb, col, out, n_bins=400, max_per_class=20000, dpi=300):
    y = ad_emb.obs[col].astype(bool).to_numpy().astype(int)
    if y.sum() in (0, len(y)):
        raise ValueError(f"{col}: only one class present")
    aucs = M.call_aucs(zmat, y)
    idx = M.balanced_idx(y, max_per_class)
    sub_z, sub_y = zmat[idx], y[idx]
    n_factors = zmat.shape[1]

    heat = np.zeros((n_factors, n_bins), dtype=float)
    for k in range(n_factors):
        order = np.argsort(sub_z[:, k])
        y_sorted = sub_y[order].astype(float)
        heat[k] = np.array([chunk.mean() for chunk in np.array_split(y_sorted, n_bins)])

    fig, ax = plt.subplots(figsize=(11, 0.45 * n_factors + 1.5))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest", rasterized=True)
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels([f"Z_{k}  (AUROC={aucs[f'Z_{k}']:.3f})" for k in range(n_factors)],
                       fontsize=9)
    ax.set_xticks([0, n_bins // 2, n_bins - 1])
    ax.set_xticklabels(["low Z_k", "mid", "high Z_k"], fontsize=9)
    ax.set_xlabel(f"cells sorted by Z_k (binned into {n_bins} quantiles)", fontsize=9)
    ax.set_title(f"Per-factor separation — {col}  (class-balanced, "
                 f"{int(sub_y.sum())} per class)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"fraction {col} in bin"); cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["all benign", "0.5", "all malignant"])
    fig.tight_layout()
    save_svg(fig, out, dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# (d2) OOF factor evaluation vs TCR (ports nb18 cells 32-33) + F1 bar / ROC overlay
# ---------------------------------------------------------------------------
def evaluate_factors(zmat, ad_emb, cnv_parquet, m2_csv, show_factors=(4, 5, 7, 1), seed=0):
    """Per-factor OOF threshold classifiers + all-factor logistic vs the two published calls.

    Returns ``(res, oof_scores, y, show_factors, res_full)`` — ``res`` is the surfaced subset
    used by the figures; ``res_full`` covers all factors (matches nb18's saved table).
    """
    from sklearn.linear_model import LogisticRegression

    cnv = pd.read_parquet(cnv_parquet)
    m2 = pd.read_csv(m2_csv).set_index("barcode")
    bc = ad_emb.obs_names
    factor_cols = [f"Z_{k}" for k in range(zmat.shape[1])]
    E = pd.DataFrame(zmat, index=bc, columns=factor_cols)
    E["donor"] = ad_emb.obs["donor"].to_numpy()
    E["study"] = ad_emb.obs["study"].astype(str).to_numpy()
    E["disease"] = ad_emb.obs["disease"].astype(str).to_numpy()

    E["alice"] = ad_emb.obs["tcr_malignant_alice"].astype(bool).to_numpy()
    E["has_tcr"] = cnv["has_tcr"].reindex(bc).astype("boolean").fillna(False).to_numpy(bool)
    E["tcr_expanded"] = cnv["tcr_is_expanded"].reindex(bc).astype("boolean").fillna(False).to_numpy(bool)
    E["cnv_score"] = cnv["cnv_cell_score"].reindex(bc).to_numpy(float)
    E["cnv_call"] = cnv["cnv_malig_cluster"].reindex(bc)
    E["m2_score"] = m2["prob_m2"].reindex(bc).to_numpy(float)
    E["m2_call"] = m2["call_m2"].reindex(bc)

    hc = E["disease"].eq("HC").to_numpy()
    mal = E["alice"].to_numpy() & ~hc
    ben = E["has_tcr"].to_numpy() & ~E["alice"].to_numpy() & ~E["tcr_expanded"].to_numpy() & ~mal
    have = (mal | ben) & np.isfinite(E["cnv_score"].to_numpy()) & np.isfinite(E["m2_score"].to_numpy()) \
        & E["cnv_call"].notna().to_numpy() & E["m2_call"].notna().to_numpy()
    E = E.loc[have].copy()
    E["y"] = mal[have].astype(int)
    fold = M.make_folds(E["y"].to_numpy(), E["donor"].to_numpy(), n_splits=5, seed=seed)

    y = E["y"].to_numpy()
    rows, oof_scores = {}, {}

    for c in factor_cols:
        s = E[c].to_numpy(float)
        oof, form, params = M.oof_eval(s, y, fold)
        r = M.binary_scores(y, oof, score=s)
        r["form"] = form
        r["threshold"] = ", ".join(f"{p:.3g}" for p in params)
        rows[c] = r
        oof_scores[c] = s if M.roc_auc_score(y, s) >= 0.5 else -s

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

    r = M.binary_scores(y, E["cnv_call"].astype(bool).to_numpy(), score=E["cnv_score"].to_numpy(float))
    r["form"] = "published"; r["threshold"] = "nb15 per-donor GMM"; rows["cnv_cd4_cd8ref"] = r
    oof_scores["cnv_cd4_cd8ref"] = (E["cnv_score"].to_numpy(float)
                                    if M.roc_auc_score(y, E["cnv_score"]) >= 0.5
                                    else -E["cnv_score"].to_numpy(float))
    r = M.binary_scores(y, E["m2_call"].astype(bool).to_numpy(), score=E["m2_score"].to_numpy(float))
    r["form"] = "published"; r["threshold"] = "0.429 (nb17 OOF)"; rows["mrvi_m2_pseudosample"] = r
    oof_scores["mrvi_m2_pseudosample"] = E["m2_score"].to_numpy(float)

    order_cols = ["form", "threshold", "f1", "auc", "balanced_acc", "precision",
                  "recall", "specificity", "jaccard", "n"]
    res_full = pd.DataFrame(rows).T[order_cols].sort_values("f1", ascending=False)
    show = [f"Z_{k}" for k in show_factors] + \
        ["semantic_all_factors", "cnv_cd4_cd8ref", "mrvi_m2_pseudosample"]
    res = res_full.loc[[k for k in show if k in res_full.index]]
    print(f"evaluable anchor cells: {len(E):,} | malignant {int(y.sum()):,} ({y.mean():.1%})"
          f" | donors {E['donor'].nunique()}")
    return res, oof_scores, y, list(show_factors), res_full


def plot_f1_bar(res, out, dpi=300):
    cnv_f1, m2_f1 = res.loc["cnv_cd4_cd8ref", "f1"], res.loc["mrvi_m2_pseudosample", "f1"]
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
    fig.tight_layout()
    save_svg(fig, out, dpi=dpi)
    return fig


def plot_roc(oof_scores, y, res, show_factors, out, dpi=300):
    roc_factors = [f"Z_{k}" for k in show_factors]
    fig, ax = plt.subplots(figsize=(5, 3))
    for name in roc_factors + ["semantic_all_factors", "cnv_cd4_cd8ref", "mrvi_m2_pseudosample"]:
        s = oof_scores[name]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.2, label=f"{name} ({res.loc[name, 'auc']:.2f})")
    ax.plot([0, 1], [0, 1], c="k", lw=0.5, ls=":")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Malignancy ROC vs TCR: factors vs CNV / M2")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    save_svg(fig, out, dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# (b) per-cell track heatmap  (ports section "B" of nb16/17 shared benchmarking)
# ---------------------------------------------------------------------------
def plot_cell_track_heatmap(calls, out, seed=0, max_unlabeled=20000, dpi=300):
    """4 tracks (prob_m1, prob_m2, tcr_status, clone_size) over cells sorted by malignant score."""
    prob1 = calls["prob_m1"].to_numpy() if "prob_m1" in calls.columns else np.full(len(calls), np.nan)
    prob2 = calls["prob_m2"].to_numpy() if "prob_m2" in calls.columns else np.full(len(calls), np.nan)
    tcr_num = calls["tcr_label"].map({"malignant": 1.0, "benign": 0.0}).to_numpy()  # unlabeled -> nan
    clone_log = np.log1p(calls["clone_size"].fillna(0).to_numpy())

    probs = [p for p in (prob1, prob2) if np.isfinite(p).any()]
    ens = np.nanmean(np.vstack(probs), axis=0) if probs else clone_log

    is_anchor = calls["tcr_label"].ne("unlabeled").to_numpy()
    keep = np.ones(len(calls), bool)
    unl = np.where(~is_anchor)[0]
    if len(unl) > max_unlabeled:
        drop = np.random.default_rng(seed).choice(unl, len(unl) - max_unlabeled, replace=False)
        keep[drop] = False
    idx = np.where(keep)[0]
    order = idx[np.argsort(-ens[idx])]

    tracks = [("prob_m1", prob1, "viridis"), ("prob_m2", prob2, "viridis"),
              ("tcr_status", tcr_num, "coolwarm"), ("clone_size (log1p)", clone_log, "magma")]
    fig, axes = plt.subplots(len(tracks), 1, figsize=(14, 4), sharex=True)
    for ax, (name, arr, cm) in zip(axes, tracks):
        im = ax.imshow(arr[order][None, :], aspect="auto", cmap=cm, rasterized=True)
        ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    axes[-1].set_xlabel("cells (sorted by ensemble malignant score)")
    fig.tight_layout()
    save_svg(fig, out, dpi=dpi)
    return fig
