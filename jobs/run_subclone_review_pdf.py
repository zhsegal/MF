#!/usr/bin/env python
"""Eyeball review of the two v4 subclone labellings: one PDF per method, one page per donor.

  cnv_subclone     KMeans on the per-cell arm-CNV matrix (inferCNV only), null-anchored gates
  nested_subclone  Leiden on the MRVI latent (transcriptomic MAJOR) -> CNV split within each major

Page: header with the call statistics; row 1 = per-donor UMAP of the malignant cells (MRVI u latent)
coloured by this method, the other method, sample, tissue, top TCR clonotypes, CNV score;
row 2 = per-cell arm-CNV heatmap (z vs the held-out diploid null, fixed +-3 SD) grouped by cluster with the
donor's benign CD4 cells as reference, and the cluster-mean arm profile.
CPU job (reads obs + obsm of the 12 GB object through h5py only). Writes figures/subclone_review_<method>_v4.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "helpers"))
import scanpy as sc
import subclone_helpers as S
try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem

AJ = ROOT / "data" / "atlas_joint"
FIG = ROOT / "figures"
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 9})
SEED, N_NEIGHBORS, MIN_MODEL = 0, 15, 150
MAX_HEAT = 4000          # malignant cells drawn per donor heatmap (stratified by cluster)
OBS_COLS = ["donor", "sample_id", "tissue", "study", "disease", "disease_stage", "cell_type_T", "clin_timepoint", "trb_cdr3"]

# ------------------------------------------------------------------ load (obs + latent only)
with h5py.File(AJ / "skin_T_tcr_annotated_v4.h5ad", "r") as f:
    idx = np.asarray(read_elem(f["obs"][f["obs"].attrs["_index"]])).astype(str)
    obs = pd.DataFrame({c: np.asarray(read_elem(f["obs"][c])) for c in OBS_COLS if c in f["obs"]}, index=idx)
    lat = np.asarray(f["obsm"]["X_mrvi_u"], dtype=np.float32)
for c in obs.columns:
    obs[c] = obs[c].astype(str)
print("obs", obs.shape, "latent", lat.shape, flush=True)

sub = pd.read_parquet(AJ / "subclones_v4.parquet")
if "cell_id" in sub.columns:
    sub = sub.set_index("cell_id")
for c in ("cnv_subclone", "nested_subclone"):
    obs[c] = sub[c].astype(str).reindex(obs.index).fillna("").to_numpy()
mal = (obs["cnv_subclone"] != "").to_numpy()

def _idx(df, col):
    return df.set_index(col) if col in df.columns else df


mc = _idx(pd.read_parquet(AJ / "skin_T_malignancy_cluster_v1.parquet"), "cell_id")
obs["mal_cluster"] = mc["mal_cluster"].reindex(obs.index).to_numpy()
obs["mal_prob"] = mc["mal_cluster_prob"].reindex(obs.index).to_numpy(dtype=float)
m5 = _idx(pd.read_parquet(AJ / "skin_T_malignancy_v5.parquet"), "obs_name")
obs["cnv_score"] = m5["cnv_score"].reindex(obs.index).to_numpy(dtype=float)
import lineage as LIN
obs = LIN.apply_lineage(obs, "skin")          # obs comes straight from the h5ad, not read_obs
benign = ((obs["mal_cluster"].astype(str) == "False").to_numpy()
          & obs["cell_type_T"].astype(str).isin(LIN.CD4_BENIGN["skin"]).to_numpy())

arm_raw = pd.read_parquet(AJ / "skin_T_arm_cnv_v5.parquet")
if "obs_name" in arm_raw.columns:
    arm_raw = arm_raw.set_index("obs_name")
ARM_COLS = S.arm_order([c for c in arm_raw.columns if c.startswith("chr")])
arm_q, null_arm = S.split_null_rows(arm_raw)
NS = S.arm_null_stats(null_arm, ARM_COLS)
ARM_COLS = [c for c in ARM_COLS if np.isfinite(NS.loc[c, "sd"]) and arm_q[c].notna().any()]      # drop arms with no positioned genes
arm_z = ((arm_q[ARM_COLS] - NS.loc[ARM_COLS, "mean"].to_numpy()) / NS.loc[ARM_COLS, "sd"].to_numpy()).reindex(obs.index)
del arm_raw, arm_q, null_arm

summ = pd.read_csv(AJ / "subclone_summary_v4.csv").set_index("donor")
tb = pd.read_csv(AJ / "subclone_trunk_branch_v4.csv").set_index("donor")
model = pd.read_csv(AJ / "pseq30_donor_table_v1.csv")
model = set(model.loc[model["retained"].astype(bool), "donor"].astype(str))
donors = sorted(summ.index, key=lambda d: (d not in model, d))

# ------------------------------------------------------------------ per-donor UMAP (as nb24: kNN on the latent)
donor_arr = obs["donor"].to_numpy()
pos = {d: np.where(mal & (donor_arr == d))[0] for d in donors}
UM = {}
for d in donors:
    ii = pos[d]
    a = sc.AnnData(lat[ii]); a.obsm["X_mu"] = a.X.copy()
    sc.pp.neighbors(a, use_rep="X_mu", n_neighbors=min(N_NEIGHBORS, a.n_obs - 1), random_state=SEED)
    sc.tl.umap(a, random_state=SEED)
    UM[d] = a.obsm["X_umap"]
print("umaps done", flush=True)


def short(lab, d):
    return np.array([x[len(d) + 1:] if x.startswith(d + "_") else x for x in lab])


def palette(labels, method):
    labels = sorted(set(labels), key=lambda s: (len(s), s))
    if method == "nested_subclone":
        return S.major_minor_palette(labels)
    return {l: plt.get_cmap("tab10")(i % 10) for i, l in enumerate(labels)}


def scatter_cat(ax, xy, cats, pal, title, legend=True, max_leg=8, ncol=2, lab_len=22):
    order = np.random.default_rng(0).permutation(len(xy))
    ax.scatter(xy[order, 0], xy[order, 1], s=max(1.5, min(8, 12000 / len(xy))), lw=0,
               c=[pal.get(c, (0.85, 0.85, 0.85)) for c in cats[order]], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    if legend:
        vc = pd.Series(cats).value_counts()
        h = [Patch(color=pal.get(k, (0.85, 0.85, 0.85)), label=f"{str(k)[:lab_len]} ({v})") for k, v in vc.head(max_leg).items()]
        ax.legend(handles=h, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=ncol,
                  handlelength=0.8, columnspacing=0.8)


def page(pdf, d, method):
    other = "cnv_subclone" if method == "nested_subclone" else "nested_subclone"
    ii = pos[d]; o = obs.iloc[ii]; xy = UM[d]
    lab = short(o[method].to_numpy(), d); lab_o = short(o[other].to_numpy(), d)
    pal = palette(lab, method); pal_o = palette(lab_o, other)
    sizes = pd.Series(lab).value_counts()
    r = summ.loc[d]

    fig = plt.figure(figsize=(17, 10.5))
    gs = GridSpec(2, 12, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.6, left=0.04, right=0.98, top=0.88, bottom=0.07)

    # ---- header
    in_model = d in model
    n_ok = int((sizes >= MIN_MODEL).sum())
    head = (f"{d}   [{r.get('study', '')} | {r.get('disease', '')} {r.get('disease_stage', '')}]   "
            f"{'IN the 30-donor regimprint set' if in_model else 'not in the regimprint set'}")
    l2 = (f"{len(ii):,} malignant cells (mal_cluster)   |   {method}: {len(sizes)} clusters, {n_ok} with >= {MIN_MODEL} cells "
          f"({', '.join(f'{k}={v}' for k, v in sizes.items())})")
    l3 = (f"inferCNV-only split: k={int(r['k'])}, silhouette {r['silhouette']:.3f} (null gate {r['sil_thr']:.3f}), "
          f"max arm delta {r['max_arm_delta']:.3f} (gate {r['arm_delta_thr']:.3f})   |   transcriptomic majors: {int(r['n_major'])}, "
          f"NMI(major, CNV) = {r['nmi_major_vs_minor']:.2f}   |   cells CNV-confirmed malignant: {r['frac_cnv_confirmed']:.0%}")
    t = tb.loc[d] if d in tb.index else None
    l4 = "" if t is None else (f"trunk arms: {str(t['trunk_arms']) if pd.notna(t['trunk_arms']) else '-'}    "
                               f"branch arms: {str(t['branch_arms'])[:150] if pd.notna(t['branch_arms']) else '-'}")
    fig.text(0.04, 0.975, head, fontsize=13, fontweight="bold", va="top")
    fig.text(0.04, 0.95, l2 + "\n" + l3 + "\n" + l4, fontsize=8.5, va="top")

    # ---- row 1: UMAPs
    ax = fig.add_subplot(gs[0, 0:2]); scatter_cat(ax, xy, lab, pal, f"{method}")
    ax = fig.add_subplot(gs[0, 2:4]); scatter_cat(ax, xy, lab_o, pal_o, f"other method: {other}")
    samp = o["sample_id"].str.replace(d + "__", "", regex=False).to_numpy()
    if "clin_timepoint" in o and o["clin_timepoint"].nunique() > 1:
        samp = np.array([f"{s} [{t}]" for s, t in zip(samp, o["clin_timepoint"])])
    ps = {k: plt.get_cmap("Set2")(i % 8) for i, k in enumerate(pd.Series(samp).value_counts().index)}
    ax = fig.add_subplot(gs[0, 4:6]); scatter_cat(ax, xy, samp, ps, "sample [time point]", ncol=1, lab_len=40)
    tis = o["tissue"].to_numpy()
    pt = {k: plt.get_cmap("Dark2")(i % 8) for i, k in enumerate(pd.Series(tis).value_counts().index)}
    ax = fig.add_subplot(gs[0, 6:8]); scatter_cat(ax, xy, tis, pt, "tissue")
    trb = o["trb_cdr3"].replace({"nan": "", "None": ""}).to_numpy() if "trb_cdr3" in o else np.full(len(o), "")
    top = [k for k in pd.Series(trb[trb != ""]).value_counts().head(4).index]
    tc = np.where(trb == "", "no TCR", np.where(np.isin(trb, top), trb, "other clonotype"))
    ptc = {k: plt.get_cmap("tab10")(i) for i, k in enumerate(top)}; ptc.update({"other clonotype": (0.45, 0.45, 0.45), "no TCR": (0.88, 0.88, 0.88)})
    ax = fig.add_subplot(gs[0, 8:10]); scatter_cat(ax, xy, tc, ptc, "TRB clonotype (top 4)", max_leg=6, ncol=1)
    ax = fig.add_subplot(gs[0, 10:12])
    cs = o["cnv_score"].to_numpy(dtype=float)
    scp = ax.scatter(xy[:, 0], xy[:, 1], s=max(1.5, min(8, 12000 / len(xy))), lw=0, c=cs, cmap="viridis",
                     vmin=np.nanpercentile(cs, 2) if np.isfinite(cs).any() else 0,
                     vmax=np.nanpercentile(cs, 98) if np.isfinite(cs).any() else 1, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("per-cell inferCNV score")
    fig.colorbar(scp, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)

    # ---- row 2 left: per-cell arm heatmap grouped by cluster (+ benign CD4 reference)
    ax = fig.add_subplot(gs[1, 1:8])
    rng = np.random.default_rng(SEED)
    Z = arm_z.iloc[ii].to_numpy(dtype=float)
    clusters = list(sizes.index)
    blocks, names, cols = [], [], []
    frac = min(1.0, MAX_HEAT / len(ii))
    for c in clusters:
        w = np.where(lab == c)[0]
        w = rng.choice(w, max(20, int(round(len(w) * frac))), replace=False) if len(w) > 20 and frac < 1 else w
        zc = Z[w]; zc = zc[np.argsort(-np.nan_to_num(np.abs(zc)).mean(1))]
        blocks.append(zc); names.append(f"{c}  (n={sizes[c]}{'' if sizes[c] >= MIN_MODEL else ', <150'})"); cols.append(pal[c])
    bi = np.where(benign & (donor_arr == d))[0]
    if len(bi) >= 20:
        bi = rng.choice(bi, min(len(bi), max(150, int(0.15 * sum(len(b) for b in blocks)))), replace=False)
        blocks.append(arm_z.iloc[bi].to_numpy(dtype=float)); names.append(f"benign CD4 (n={len(bi)} shown)"); cols.append((0.6, 0.6, 0.6))
    M = np.vstack(blocks)
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="none", rasterized=True)
    y = 0
    for b, nm, col in zip(blocks, names, cols):
        ax.add_patch(plt.Rectangle((-1.6, y - 0.5), 1.0, len(b), color=col, clip_on=False, lw=0))
        ax.text(-2.0, y + len(b) / 2, nm, ha="right", va="center", fontsize=8, color="k", fontweight="bold")
        y += len(b)
        ax.axhline(y - 0.5, color="k", lw=0.8)
    for j, a_ in enumerate(ARM_COLS):                       # chromosome separators
        if j and a_.rstrip("pq") != ARM_COLS[j - 1].rstrip("pq"):
            ax.axvline(j - 0.5, color="k", lw=0.3, alpha=0.5)
    ax.set_xlim(-0.5, len(ARM_COLS) - 0.5)
    ax.set_xticks(range(len(ARM_COLS))); ax.set_xticklabels([a_.replace("chr", "") for a_ in ARM_COLS], rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title(f"per-cell arm CNV, grouped by {method} (z vs held-out diploid null; "
                 f"{'all' if frac >= 1 else f'{frac:.0%} of'} malignant cells drawn)")
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label="diploid SDs")

    # ---- row 2 right: cluster-mean arm profile
    ax = fig.add_subplot(gs[1, 9:12])
    prof = np.vstack([np.nanmean(Z[lab == c], axis=0) for c in clusters])
    rows = [f"{c} ({sizes[c]})" for c in clusters]
    if len(bi) >= 20:
        prof = np.vstack([prof, np.nanmean(arm_z.iloc[bi].to_numpy(dtype=float), axis=0)]); rows.append("benign CD4")
    im2 = ax.imshow(np.ma.masked_invalid(prof.T), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="none")
    ax.set_yticks(range(len(ARM_COLS))); ax.set_yticklabels([a_.replace("chr", "") for a_ in ARM_COLS], fontsize=6)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(rows, rotation=60, ha="right", fontsize=7)
    ax.set_title("cluster-mean arm profile")
    fig.colorbar(im2, ax=ax, fraction=0.05, pad=0.03, label="diploid SDs")
    pdf.savefig(fig, dpi=140); plt.close(fig)


def cover(pdf, method):
    rows = []
    for d in donors:
        lab = obs[method].to_numpy()[pos[d]]; sz = pd.Series(lab).value_counts()
        rows.append(dict(donor=d, in_model=d in model, n_malignant=len(lab), n_clusters=len(sz), n_ge150=int((sz >= MIN_MODEL).sum()),
                         cnv_k=int(summ.loc[d, "k"]), n_major=int(summ.loc[d, "n_major"])))
    T = pd.DataFrame(rows)
    T.to_csv(FIG / f"subclone_review_{method}_v4_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(17, 10.5), gridspec_kw=dict(width_ratios=[1, 2.2]))
    ax = axs[0]
    for flag, col, nm in ((True, "#c44e52", "30 regimprint donors"), (False, "#8c8c8c", "other donors")):
        v = T.loc[T.in_model == flag, "n_ge150"].value_counts().sort_index()
        ax.bar(v.index + (-0.2 if flag else 0.2), v.values, width=0.4, color=col, label=nm)
    ax.set_xlabel(f"clusters with >= {MIN_MODEL} cells per donor"); ax.set_ylabel("donors"); ax.legend(frameon=False)
    m30 = T[T.in_model]
    ax.set_title(f"{method}: {int((m30.n_ge150 >= 2).sum())}/30 model donors and {int((T.n_ge150 >= 2).sum())}/{len(T)} "
                 f"of all donors have >= 2 usable clusters", fontsize=10)
    ax = axs[1]; ax.axis("off")
    Tt = T.assign(donor=T.donor.str.replace("Li2024_atlas__", "Li__"), in_model=T.in_model.map({True: "yes", False: ""}))
    Tt.columns = ["donor", "model", "n_mal", "clusters", ">=150", "cnv_k", "majors"]
    lines = Tt.to_string(index=False, justify="right").split("\n"); half = (len(Tt) + 1) // 2
    ax.text(0.0, 1.0, "\n".join(lines[:half + 1]), family="monospace", fontsize=8, va="top")
    ax.text(0.52, 1.0, "\n".join([lines[0]] + lines[half + 1:]), family="monospace", fontsize=8, va="top")
    fig.suptitle(f"Subclone review — {method} (subclones_v4, mal_cluster arms). One page per donor; the 30 model donors come first.",
                 fontsize=13, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)


for method in ("cnv_subclone", "nested_subclone"):
    out = FIG / f"subclone_review_{method}_v4.pdf"
    with PdfPages(out) as pdf:
        cover(pdf, method)
        for n, d in enumerate(donors):
            page(pdf, d, method)
            if n % 10 == 0:
                print(method, n, d, flush=True)
    print("wrote", out, flush=True)
