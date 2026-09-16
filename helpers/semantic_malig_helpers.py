"""Latent-projection malignancy analysis for the SemanticSCVI CD4 atlas (nb18).

Loads the precomputed malignancy calls written by nb14-17 (CNV + transcriptomic) and
nb21 (TCR ALICE), and holds the per-factor / factor-geometry plotting used to inspect how
the SemanticSCVI latent separates a chosen malignancy label. The notebook stays thin: pick
one column via ``MALIGNANCY_SOURCES``, drive every panel off it.

Companion to ``skin_T_cnv_helpers`` (CNV workflow) — this module is the projection side.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from numpy.linalg import lstsq
from scipy.sparse import issparse
from scipy.stats import fisher_exact
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Malignancy-call registry — {name: (path rel. to NB_DIR, column in file, join key)}.
# join key None -> the file is indexed by obs_name already. Every name becomes a bool
# obs column after load_malignancy_calls; pick one as MALIGNANCY_COL in the notebook.
# ---------------------------------------------------------------------------
MALIGNANCY_SOURCES = {
    # TCR ALICE (nb21) — the default driving call
    "tcr_malignant_alice": (
        "data/atlas_joint/alice_malignancy_v2.parquet", "tcr_malignant_alice", "cell_id"),
    # CNV, full skin-T shared ref (nb14)
    "cnvT_cell_gmm": (
        "data/atlas_joint/skin_T_malignancy_methods.parquet", "cnvT_cell_gmm", None),
    "cnvT_cell_mrvi": (
        "data/atlas_joint/skin_T_malignancy_methods.parquet", "cnvT_cell_mrvi", None),
    "cnvT_cnvcluster": (
        "data/atlas_joint/skin_T_malignancy_methods.parquet", "cnvT_cnvcluster", None),
    "cnvT_cnvcluster_mrvi": (
        "data/atlas_joint/skin_T_malignancy_methods.parquet", "cnvT_cnvcluster_mrvi", None),
    # CNV, CD4 vs same-sample CD8 ref (nb15) — CD4-only coverage
    "cnv_cd4_cd8ref": (
        "data/atlas_joint/skin_cd4_cd8ref_malignancy.parquet", "cnv_malig_cluster", None),
    # transcriptomic MrVI (nb16 LabelSpreading, nb17 pseudosample)
    "mrvi_m1_labelspread": (
        "data/atlas_joint/mrvi_malignancy/calls.csv", "call_m1", "barcode"),
    "mrvi_m2_pseudosample": (
        "data/atlas_joint/skin_T_mrvi_m2_malignancy_methods.parquet", "mrvi_m2_pseudosample", None),
}


def load_malignancy_calls(adata, nb_dir, sources=MALIGNANCY_SOURCES):
    """Join every registry call onto ``adata.obs`` by cell name; uncovered cells -> False.

    Each call lands as a bool obs column named by its registry key. Missing files are
    skipped (column filled False) with a warning. Returns ``adata`` (modified in place).
    """
    nb_dir = Path(nb_dir)
    for name, (rel, col, key) in sources.items():
        p = nb_dir / rel
        if not p.exists():
            print(f"MISSING (skip {name}): {p}")
            adata.obs[name] = False
            continue
        df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
        if key is not None and key in df.columns:
            df = df.set_index(key)
        vals = df[col].reindex(adata.obs_names)
        cov = float(vals.notna().mean())
        adata.obs[name] = vals.astype("boolean").fillna(False).to_numpy(dtype=bool)
        print(f"{name:22s} <- {p.name:42s} {cov:6.1%} covered | "
              f"{int(adata.obs[name].sum()):>7,} malignant")
    return adata


def malignancy_summary(adata, cols=None):
    """Short summary table: per-call n/% malignant + coverage, and pairwise Jaccard.

    Returns (counts_df, jaccard_df). Coverage here = fraction of cells called malignant
    (uncovered cells were zero-filled at load, so this doubles as a prevalence check).
    """
    cols = list(MALIGNANCY_SOURCES) if cols is None else list(cols)
    cols = [c for c in cols if c in adata.obs]
    n = adata.n_obs
    Y = {c: adata.obs[c].astype(bool).to_numpy() for c in cols}
    counts = pd.DataFrame(
        {"n_malignant": [int(Y[c].sum()) for c in cols],
         "pct_malignant": [100.0 * Y[c].mean() for c in cols]},
        index=cols,
    )
    jac = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            inter = int((Y[a] & Y[b]).sum())
            union = int((Y[a] | Y[b]).sum())
            jac.loc[a, b] = inter / union if union else np.nan
    return counts.round(2), jac.astype(float).round(3)


# ---------------------------------------------------------------------------
# Per-factor separation
# ---------------------------------------------------------------------------
def call_aucs(zmat, y):
    """Per-factor AUROC of Z_k separating y, polarity-folded to >= 0.5."""
    return {f"Z_{k}": max(a, 1 - a)
            for k in range(zmat.shape[1])
            for a in [roc_auc_score(y, zmat[:, k])]}


def balanced_idx(y, cap, seed=0):
    """Indices of a class-balanced subsample: min(cap, n_pos, n_neg) per class."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = min(cap, len(pos), len(neg))
    return np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])


def plot_call_umap(ad_emb, cols, extra=("cell_type_T", "study", "disease"), save=None):
    """Latent UMAP colored by each malignancy call (benign/malignant) + extra obs cols."""
    for c in cols:
        ad_emb.obs[c] = ad_emb.obs[c].astype(bool)
        ad_emb.obs[c + "_lbl"] = pd.Categorical(
            np.where(ad_emb.obs[c].to_numpy(), "malignant", "benign"),
            categories=["benign", "malignant"],
        )
    color = [c + "_lbl" for c in cols] + [c for c in extra if c in ad_emb.obs]
    sc.pl.umap(ad_emb, color=color, ncols=2, wspace=0.3, frameon=False, save=save)


def plot_factor_heatmaps(zmat, ad_emb, cols, fig_dir, prefix,
                         n_bins=400, max_per_class=20000):
    """One per-factor separation heatmap per call (cells sorted by Z_k, color=frac malignant)."""
    fig_dir = Path(fig_dir)
    n_factors = zmat.shape[1]
    for col in cols:
        y = ad_emb.obs[col].astype(bool).to_numpy().astype(int)
        if y.sum() in (0, len(y)):
            print(f"skip {col}: only one class present"); continue
        aucs = call_aucs(zmat, y)
        idx = balanced_idx(y, max_per_class)
        sub_z, sub_y = zmat[idx], y[idx]

        heat = np.zeros((n_factors, n_bins), dtype=float)
        for k in range(n_factors):
            order = np.argsort(sub_z[:, k])
            y_sorted = sub_y[order].astype(float)
            heat[k] = np.array([chunk.mean() for chunk in np.array_split(y_sorted, n_bins)])

        fig, ax = plt.subplots(figsize=(11, 0.45 * n_factors + 1.5))
        im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_yticks(range(n_factors))
        ax.set_yticklabels([f"Z_{k}  (AUROC={aucs[f'Z_{k}']:.3f})"
                            for k in range(n_factors)], fontsize=9)
        ax.set_xticks([0, n_bins // 2, n_bins - 1])
        ax.set_xticklabels(["low Z_k", "mid", "high Z_k"], fontsize=9)
        ax.set_xlabel(f"cells sorted by Z_k (binned into {n_bins} quantiles)", fontsize=9)
        ax.set_title(f"Per-factor separation — {col}  (class-balanced, "
                     f"{int(sub_y.sum())} per class)", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(f"fraction {col} in bin"); cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["all benign", "0.5", "all malignant"])
        fig.tight_layout()
        out = fig_dir / f"{prefix}_sep_heatmap_{col}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out)
        plt.show()


# ---------------------------------------------------------------------------
# Factor geometry — 2-factor grid + 3-D scatter + gene-direction plane sweep
# ---------------------------------------------------------------------------
def plot_factor_grid(zmat, is_mal, plot_idx, factors, aucs, call,
                     pos_label, neg_label, fig_dir, seed=0):
    """2-factor scatter grid over all pairs of ``factors``, colored by the call."""
    fig_dir = Path(fig_dir)
    pairs = list(combinations(factors, 2))
    ncols = 5
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    rng = np.random.default_rng(seed)
    shuf = rng.permutation(len(plot_idx))
    sub_mal = is_mal[plot_idx][shuf]
    colors = np.where(sub_mal == 1, "tab:red", "tab:blue")

    for ax, (fx, fy) in zip(axes, pairs):
        ax.scatter(zmat[plot_idx, fx][shuf], zmat[plot_idx, fy][shuf],
                   c=colors, s=2, alpha=0.3, linewidths=0, rasterized=True)
        ax.set_xlabel(f"Z_{fx}", fontsize=9); ax.set_ylabel(f"Z_{fy}", fontsize=9)
        ax.set_title(f"Z_{fx} vs Z_{fy}  (AUROC {aucs[f'Z_{fx}']:.2f}/{aucs[f'Z_{fy}']:.2f})",
                     fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label=pos_label, markersize=6),
        plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label=neg_label, markersize=6),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.suptitle(f"2-factor scatter grid — colored by {call}", y=1.01, fontsize=11)
    fig.tight_layout()
    out = fig_dir / f"{call}_scatter_grid_{'_'.join(map(str, factors))}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    plt.show()


def make_geom3d(zmat, fx, fy, fz, plot_idx, seed=0):
    """Standardized 3-axis subspace + a shuffled balanced draw, reused by the 3-D plots.

    Returns a dict carrying the raw axes, per-axis (mu, sd), the standardized matrix Z3z,
    and the plotting indices. Pass it to plot_3d_threshold / plane_z / plot_best_plane*.
    """
    x3, y3, zZ3 = zmat[:, fx], zmat[:, fy], zmat[:, fz]
    mu = (float(x3.mean()), float(y3.mean()), float(zZ3.mean()))
    sd = (float(x3.std()), float(y3.std()), float(zZ3.std()))
    Z3z = np.column_stack([(x3 - mu[0]) / sd[0], (y3 - mu[1]) / sd[1], (zZ3 - mu[2]) / sd[2]])
    rng = np.random.default_rng(seed)
    shuf = rng.permutation(len(plot_idx))
    return dict(fx=fx, fy=fy, fz=fz, x=x3, y=y3, zc=zZ3, mu=mu, sd=sd,
                Z3z=Z3z, plot_idx=plot_idx, shuf=shuf)


def _draw3d(geom):
    """Subsampled, shuffled raw coords for the 3-D scatter (xx, yy, zz)."""
    pi, sh = geom["plot_idx"], geom["shuf"]
    return geom["x"][pi][sh], geom["y"][pi][sh], geom["zc"][pi][sh]


def plot_3d_threshold(geom, is_mal, call, pos_label, neg_label, fig_dir):
    """3-D scatter on (Z_fx, Z_fy, Z_fz): left=call label, right=diagonal score s3, with
    median + Youden-best constant-s3 planes drawn on both."""
    fig_dir = Path(fig_dir)
    fx, fy, fz = geom["fx"], geom["fy"], geom["fz"]
    mu, sd = geom["mu"], geom["sd"]
    s3 = geom["Z3z"].sum(axis=1)
    y_true = is_mal

    xx3, yy3, zz3p = _draw3d(geom)
    ss3 = s3[geom["plot_idx"]][geom["shuf"]]
    cc3 = np.where(is_mal[geom["plot_idx"]][geom["shuf"]] == 1, "tab:red", "tab:blue")

    median_s3 = float(np.median(s3))
    fpr3, tpr3, thrs3 = roc_curve(y_true, s3)
    best_thr3 = float(thrs3[np.argmax(tpr3 - fpr3)])
    auc_s3 = float(roc_auc_score(y_true, s3)); auc_s3 = max(auc_s3, 1 - auc_s3)

    def s_plane(c, xg, yg):
        return sd[2] * (c - (xg - mu[0]) / sd[0] - (yg - mu[1]) / sd[1]) + mu[2]

    xg = np.linspace(xx3.min(), xx3.max(), 20)
    yg = np.linspace(yy3.min(), yy3.max(), 20)
    XG, YG = np.meshgrid(xg, yg)
    ZG_med, ZG_best = s_plane(median_s3, XG, YG), s_plane(best_thr3, XG, YG)
    s_vmax3 = float(np.quantile(np.abs(s3), 0.99))

    fig = plt.figure(figsize=(14, 7))
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
    axL.plot_surface(XG, YG, ZG_med, color="black", alpha=0.10, edgecolor="none")
    axL.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
    axL.set_xlabel(f"Z_{fx}"); axL.set_ylabel(f"Z_{fy}"); axL.set_zlabel(f"Z_{fz}")
    axL.set_title(f"colored by {call}   AUROC(s3) = {auc_s3:.3f}", fontsize=10)
    axL.legend(handles=[
        plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label=pos_label, markersize=6),
        plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label=neg_label, markersize=6),
        plt.Line2D([], [], marker="s", linestyle="", color="black", alpha=0.10,
                   label=f"plane: s3 = median ({median_s3:+.2f})", markersize=10),
        plt.Line2D([], [], marker="s", linestyle="", color="black", alpha=0.20,
                   label=f"plane: s3 = best   ({best_thr3:+.2f}, Youden)", markersize=10),
    ], loc="upper left", fontsize=8, frameon=False)

    axR = fig.add_subplot(1, 2, 2, projection="3d")
    sc3 = axR.scatter(xx3, yy3, zz3p, c=ss3, cmap="RdBu_r", vmin=-s_vmax3, vmax=s_vmax3,
                      s=3, alpha=0.55, depthshade=False)
    axR.plot_surface(XG, YG, ZG_med, color="black", alpha=0.10, edgecolor="none")
    axR.plot_surface(XG, YG, ZG_best, color="black", alpha=0.20, edgecolor="none")
    axR.set_xlabel(f"Z_{fx}"); axR.set_ylabel(f"Z_{fy}"); axR.set_zlabel(f"Z_{fz}")
    axR.set_title(f"colored by s3 = z(Z_{fx})+z(Z_{fy})+z(Z_{fz})", fontsize=10)
    fig.colorbar(sc3, ax=axR, fraction=0.03, pad=0.10).set_label("s3 value")

    fig.suptitle(f"3-D scatter Z_{fx}, Z_{fy}, Z_{fz} — {call}", fontsize=12, y=1.02)
    fig.tight_layout()
    out = fig_dir / f"{call}_scatter3d_Z{fx}_Z{fy}_Z{fz}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    plt.show()


def gene_lognorm(adata, name, size_factor=None):
    """log1p(cp10k) expression for one gene from raw counts (adata.X assumed raw)."""
    X_raw = adata.X
    if size_factor is None:
        size_factor = np.asarray(X_raw.sum(axis=1)).ravel().astype(np.float64)
        size_factor[size_factor == 0] = 1.0
    j = adata.var_names.get_loc(name)
    col = X_raw[:, j]
    col = col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()
    return np.log1p(col / size_factor * 1e4).astype(np.float64)


def fit_direction(Z3z, y):
    """OLS coeffs (first 3) of y on standardized axes Z3z (+ intercept), ignoring NaNs."""
    A = np.column_stack([Z3z, np.ones(len(Z3z))])
    mask = np.isfinite(y)
    coef, *_ = lstsq(A[mask], y[mask], rcond=None)
    return coef[:3]


def eval_direction(name, v, Z3z, y_true):
    """Score direction v on Z3z: AUROC (polarity-folded), Youden threshold + class stats."""
    s = Z3z @ v
    auc = roc_auc_score(y_true, s)
    flipped = auc < 0.5
    if flipped:
        s, auc, v = -s, 1 - auc, -v
    fpr, tpr, thrs = roc_curve(y_true, s)
    best_thr = float(thrs[np.argmax(tpr - fpr)])
    y_pred = (s > best_thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    odds, pval = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    return dict(name=name, AUROC=auc, acc=acc, prec=p, rec=r, f1=f1,
                TP=tp, FP=fp, FN=fn, TN=tn, OR=float(odds), p=float(pval),
                vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
                thr=best_thr, flipped=flipped)


def plane_z(v, thr, x_raw, y_raw, geom):
    """z-coordinate of the plane v·(standardized x,y,z)=thr, in raw Z_fz units."""
    mu, sd = geom["mu"], geom["sd"]
    if abs(v[2]) < 1e-6:
        return np.full_like(x_raw, np.nan, dtype=float)
    xz_ = (x_raw - mu[0]) / sd[0]
    yz_ = (y_raw - mu[1]) / sd[1]
    zz_ = (thr - v[0] * xz_ - v[1] * yz_) / v[2]
    return sd[2] * zz_ + mu[2]


def _top_plane(results, geom):
    """Extract top-AUROC direction/threshold + its meshed plane from an eval table."""
    top = results.iloc[0]
    v = np.array([top["vx"], top["vy"], top["vz"]], dtype=float)
    v = v / np.linalg.norm(v)
    thr = float(top["thr"])
    xx3, yy3, _ = _draw3d(geom)
    xg = np.linspace(xx3.min(), xx3.max(), 20)
    yg = np.linspace(yy3.min(), yy3.max(), 20)
    XG, YG = np.meshgrid(xg, yg)
    return top, v, thr, XG, YG, plane_z(v, thr, XG, YG, geom)


def plot_best_plane(geom, results, is_mal, call, pos_label, neg_label,
                    mark_expr, mark_gene, fig_dir):
    """Static 3-D scatter of the top-AUROC plane: left=call label, right=marker-gene expr."""
    fig_dir = Path(fig_dir)
    fx, fy, fz = geom["fx"], geom["fy"], geom["fz"]
    top, v, thr, XG, YG, ZG = _top_plane(results, geom)
    top_name = top["name"]
    xx3, yy3, zz3p = _draw3d(geom)
    cc3 = np.where(is_mal[geom["plot_idx"]][geom["shuf"]] == 1, "tab:red", "tab:blue")
    mark_plot = mark_expr[geom["plot_idx"]][geom["shuf"]]
    mark_vmax = float(np.quantile(mark_expr, 0.99))

    fig = plt.figure(figsize=(14, 7))
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axL.scatter(xx3, yy3, zz3p, c=cc3, s=3, alpha=0.3, depthshade=False)
    axL.plot_surface(XG, YG, ZG, color="tab:green", alpha=0.25, edgecolor="none")
    axL.set_xlabel(f"Z_{fx}"); axL.set_ylabel(f"Z_{fy}"); axL.set_zlabel(f"Z_{fz}")
    axL.set_title(f"colored by {call}   top: {top_name}   AUROC={top['AUROC']:.3f}", fontsize=10)
    axL.legend(handles=[
        plt.Line2D([], [], marker="o", linestyle="", color="tab:red", label=pos_label, markersize=6),
        plt.Line2D([], [], marker="o", linestyle="", color="tab:blue", label=neg_label, markersize=6),
        plt.Line2D([], [], marker="s", linestyle="", color="tab:green", alpha=0.25,
                   label=f"{top_name} best (AUROC={top['AUROC']:.3f})", markersize=10),
    ], loc="upper left", fontsize=8, frameon=False)

    axR = fig.add_subplot(1, 2, 2, projection="3d")
    scR = axR.scatter(xx3, yy3, zz3p, c=mark_plot, cmap="viridis",
                      vmin=0, vmax=mark_vmax, s=3, alpha=0.55, depthshade=False)
    axR.plot_surface(XG, YG, ZG, color="tab:green", alpha=0.25, edgecolor="none")
    axR.set_xlabel(f"Z_{fx}"); axR.set_ylabel(f"Z_{fy}"); axR.set_zlabel(f"Z_{fz}")
    axR.set_title(f"colored by {mark_gene} log-norm expression", fontsize=10)
    fig.colorbar(scR, ax=axR, fraction=0.03, pad=0.10).set_label(f"{mark_gene} log1p(cp10k)")

    fig.suptitle(f"Best-direction plane ({call}): {top_name}", fontsize=12, y=1.02)
    fig.tight_layout()
    safe = _safe(top_name)
    out = fig_dir / f"{call}_scatter3d_best_plane_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    plt.show()


def plot_best_plane_plotly(geom, results, is_mal, call, pos_label, neg_label,
                           mark_expr, mark_gene, fig_dir):
    """Interactive rotatable 3-D version of the top-AUROC plane (standalone .html)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from IPython.display import IFrame

    fig_dir = Path(fig_dir)
    fx, fy, fz = geom["fx"], geom["fy"], geom["fz"]
    top, v, thr, XG, YG, ZG = _top_plane(results, geom)
    top_name = top["name"]
    xx3, yy3, zz3p = _draw3d(geom)
    mark_plot = mark_expr[geom["plot_idx"]][geom["shuf"]]
    mark_vmax = float(np.quantile(mark_expr, 0.99))
    mal_mask = is_mal[geom["plot_idx"]][geom["shuf"]] == 1
    ben_mask = ~mal_mask

    fig3d = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"colored by {call}   top: {top_name}   AUROC={top['AUROC']:.3f}",
            f"colored by {mark_gene} log-norm expression"),
        horizontal_spacing=0.05,
    )
    fig3d.add_trace(go.Scatter3d(
        x=xx3[mal_mask], y=yy3[mal_mask], z=zz3p[mal_mask], mode="markers", name=pos_label,
        marker=dict(size=2, color="crimson", opacity=0.45)), row=1, col=1)
    fig3d.add_trace(go.Scatter3d(
        x=xx3[ben_mask], y=yy3[ben_mask], z=zz3p[ben_mask], mode="markers", name=neg_label,
        marker=dict(size=2, color="royalblue", opacity=0.45)), row=1, col=1)
    fig3d.add_trace(go.Surface(
        x=XG, y=YG, z=ZG, showscale=False, opacity=0.35,
        colorscale=[[0, "green"], [1, "green"]],
        name=f"{top_name} best (AUROC={top['AUROC']:.3f})", showlegend=True), row=1, col=1)
    fig3d.add_trace(go.Scatter3d(
        x=xx3, y=yy3, z=zz3p, mode="markers", name=f"{mark_gene}", showlegend=False,
        marker=dict(size=2, color=mark_plot, colorscale="Viridis", cmin=0, cmax=mark_vmax,
                    opacity=0.55,
                    colorbar=dict(title=f"{mark_gene} log1p(cp10k)", thickness=12, x=1.02))),
        row=1, col=2)
    fig3d.add_trace(go.Surface(
        x=XG, y=YG, z=ZG, showscale=False, opacity=0.35,
        colorscale=[[0, "green"], [1, "green"]], name="top plane", showlegend=False),
        row=1, col=2)

    axes_kw = dict(xaxis_title=f"Z_{fx}", yaxis_title=f"Z_{fy}", zaxis_title=f"Z_{fz}")
    fig3d.update_layout(title=f"Best-direction plane ({call}): {top_name}",
                        width=1300, height=700, scene=axes_kw, scene2=axes_kw,
                        legend=dict(itemsizing="constant"))
    html_out = fig_dir / f"{call}_scatter3d_best_plane_{_safe(top_name)}.html"
    fig3d.write_html(html_out, include_plotlyjs="cdn", full_html=True)
    print("saved", html_out)
    return IFrame(html_out.as_posix(), width=1320, height=720)


def _safe(name):
    return (name.replace("(", "").replace(")", "").replace(",", "_")
            .replace(" ", "_").replace("/", "_"))


# ---------------------------------------------------------------------------
# Fair OOF malignancy comparison (nb18): turn each factor into a TCR-trained
# classifier, evaluate out-of-fold, and compare to the published CNV / M2 calls
# on identical cells / labels / folds. See plan in 18_*.ipynb "Part N".
# ---------------------------------------------------------------------------
def _f1(yp, y):
    tp = int((yp & y).sum()); fp = int((yp & ~y).sum()); fn = int((~yp & y).sum())
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)


def fit_threshold_form(x, y, n_grid=99, n_band=24):
    """F1-optimal threshold for one score, allowing 4 shapes of the malignant region.

    Searches: one-sided ``x > t``, one-sided ``x < t``, two-sided inside-band
    ``lo < x < hi``, and two-sided outside-band ``x < lo | x > hi``. Returns
    ``(form, params, train_f1)`` where form in {"gt","lt","band_in","band_out"} and
    params is ``(t,)`` for one-sided or ``(lo, hi)`` for bands.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if x.size < 50 or not (0 < y.sum() < y.size):
        return "gt", (np.nan,), 0.0
    best = ("gt", (np.nan,), -1.0)
    g1 = np.unique(np.percentile(x, np.linspace(1, 99, n_grid)))
    for t in g1:
        for form, yp in (("gt", x > t), ("lt", x < t)):
            f = _f1(yp, y)
            if f > best[2]:
                best = (form, (float(t),), f)
    gb = np.unique(np.percentile(x, np.linspace(2, 98, n_band)))
    for i in range(len(gb)):
        for j in range(i + 1, len(gb)):
            lo, hi = float(gb[i]), float(gb[j])
            inb = (x > lo) & (x < hi)
            for form, yp in (("band_in", inb), ("band_out", ~inb)):
                f = _f1(yp, y)
                if f > best[2]:
                    best = (form, (lo, hi), f)
    return best


def apply_threshold_form(x, form, params):
    """Boolean malignant prediction for a fitted (form, params); NaN scores -> False."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    out = np.zeros(x.shape, dtype=bool)
    if not np.isfinite(params[0]):
        return out
    if form == "gt":
        out[ok] = x[ok] > params[0]
    elif form == "lt":
        out[ok] = x[ok] < params[0]
    elif form == "band_in":
        out[ok] = (x[ok] > params[0]) & (x[ok] < params[1])
    elif form == "band_out":
        out[ok] = (x[ok] < params[0]) | (x[ok] > params[1])
    return out


def make_folds(y, groups, n_splits=5, seed=0):
    """One StratifiedGroupKFold(by group) assignment reused for every predictor.

    Returns an int ndarray of held-out fold id per sample (donor-disjoint folds).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y).astype(int); groups = np.asarray(groups)
    fold = np.full(y.shape, -1, dtype=int)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros_like(y), y, groups)):
        fold[te] = k
    return fold


def oof_eval(score, y, fold):
    """Out-of-fold threshold classifier for one score over shared folds.

    For each fold: fit ``fit_threshold_form`` on the other folds, predict the held-out
    fold. Returns ``(oof_pred_bool, form, params)`` where form/params are the modal
    (most frequent across folds) shape, for reporting.
    """
    score = np.asarray(score, dtype=float); y = np.asarray(y, dtype=bool)
    oof = np.zeros(y.shape, dtype=bool)
    forms = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        form, params, _ = fit_threshold_form(score[tr], y[tr])
        oof[te] = apply_threshold_form(score[te], form, params)
        forms.append((form, params))
    # modal form; median threshold(s) among folds sharing that form (for display only)
    from collections import Counter
    mform = Counter(f for f, _ in forms).most_common(1)[0][0]
    ps = [p for f, p in forms if f == mform]
    mparams = tuple(np.median([p[i] for p in ps]) for i in range(len(ps[0])))
    return oof, mform, mparams


def binary_scores(y, yhat, score=None):
    """Agreement metrics of a boolean call vs truth y; AUC (polarity-folded) if score given."""
    y = np.asarray(y, dtype=bool); yhat = np.asarray(yhat, dtype=bool)
    tp = int((yhat & y).sum()); tn = int((~yhat & ~y).sum())
    fp = int((yhat & ~y).sum()); fn = int((~yhat & y).sum())
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    jac = tp / max(1, tp + fp + fn)
    out = {
        "f1": 2 * pr * rc / max(1e-9, pr + rc),
        "balanced_acc": 0.5 * (rc + spec),
        "precision": pr,
        "recall": rc,
        "sensitivity": rc,
        "specificity": spec,
        "jaccard": jac,
        "n": int(y.size),
    }
    if score is not None:
        s = np.asarray(score, dtype=float); ok = np.isfinite(s)
        if ok.sum() > 0 and 0 < y[ok].sum() < ok.sum():
            a = roc_auc_score(y[ok], s[ok]); out["auc"] = max(a, 1 - a)
        else:
            out["auc"] = np.nan
    return out
