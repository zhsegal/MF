"""Subclonal (divergent-evolution) analysis of MF malignant clones — nb23.

Scales nb20's Q1b prototype (KMeans on the cached arm-level inferCNV matrix, top-3 donors) to the
whole eligible cohort and adds the Cancer-Discovery-2025 divergent-evolution characterization:

- `detect_subclones`   : per-donor CNV subclones, data-driven k (silhouette + size/separation guards),
                         gates calibrated against the held-out diploid null (`null_split_thresholds`)
- `split_null_rows` / `arm_null_stats` / `arm_z` / `healthy_ref_profiles`
                       : the healthy-CD4 diploid null carried through the v3 inferCNV pass, and the
                         z-scale + reference rows the arm-profile plots are drawn on
- `plot_cohort_arm_profiles` / `plot_sample_arm_profile` : cohort and per-sample arm profiles
- `major_minor`        : major (distinct transcriptional cluster) vs minor (intermixed) per subclone
- `call_arm_events`    : per-subclone arm gain/loss vs the same-donor benign-T diploid baseline (nb20 cell-15 logic)
- `trunk_branch`       : trunk (pan-subclonal) vs branch (private) arm events -> the divergent tree edges
- `subclone_linkage`   : hierarchical linkage on arm centroids (rooted at diploid) for a per-donor dendrogram
- `extract_trb` / `tcr_variant_class` : founder vs ALICE <=1-aa variant vs other (reconcile CNV subclones w/ nb21 TCR)
- `pseudobulk_counts` / `run_pydeseq2` : nb03 pseudobulk-DESeq2 helpers, lifted here for cohort-level subclone DE
- `subclone_markers_wilcoxon`          : within-donor single-cell subclone DE (presto-style baseline)
- `PROGRAM_SETS` / `score_programs`    : metabolism / proliferation / cytokine / Th / homing signatures per subclone

Reuses `skin_T_cnv_helpers.HG38_CENTROMERE` (arm coords) and `alice_helpers.founder_family` (TCR families).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as _sp
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_samples, silhouette_score

# hg38 arm order for stable column display (mirrors nb20 cell 11)
def arm_order(arm_cols):
    """Sort arm labels ('chr8q'...) into genomic order: chrom number, then p before q."""
    def key(a):
        stem = a.replace("chr", "")
        num = "".join(ch for ch in stem if ch.isdigit())
        chrom = int(num) if num else (23 if "X" in stem else 24)
        return (chrom, 0 if a.endswith("p") else 1)
    return sorted(arm_cols, key=key)


# --------------------------------------------------------------------------- #
# §1  CNV subclone detection
# --------------------------------------------------------------------------- #
_NULL_THR_CACHE: dict = {}


def null_split_thresholds(null_X: np.ndarray, n: int, k: int, *, n_draws: int = 8,
                          null_z: float = 3.0, seed: int = 0, min_sub: int = 50,
                          sil_sample: int = 2000):
    """Silhouette / arm-delta a *known-diploid* split of the same size and k produces.

    `null_X` = arm matrix of held-out diploid reference cells (the `<donor>|NULL|<cell>` rows the
    v3 inferCNV pass carries through the identical estimator). Draws `n_draws` random subsets of
    `n` cells, forces KMeans at `k`, and returns `mean + null_z * sd` of the resulting silhouette
    and max per-arm centroid range. Those are the noise floors a real split has to clear.

    This replaces hard-coded gates: absolute thresholds tuned on one inferCNV estimator do not
    transfer to another. On the blood v3 matrix a diploid split at k=2, n~4k gives arm_delta
    0.047 +/- 0.001 and silhouette 0.065 +/- 0.002 — so nb31's `min_arm_delta=0.03` sits *below*
    the noise floor (it rejects nothing) while `sil_min=0.15` sits at ~2x it (it rejects nearly
    everything).

    `mean + z*sd` rather than a high empirical quantile because the draw-to-draw spread is tiny
    (sd/mean ~3% for silhouette, ~2% for arm_delta), so the parametric tail is stable at 8 draws
    where a 0.99 quantile would just be the max of 8 and would move run to run.

    Cached on (k, n bucketed in half-powers of 2, n_draws, seed): the null statistics are flat in n
    (arm_delta 0.047 at n=250 and at n=4000), so exact-n draws would be wasted work.
    """
    if null_X is None or len(null_X) < max(2 * min_sub, 50):
        return {"sil": None, "arm_delta": None, "n_draws": 0,
                "n_null": 0 if null_X is None else len(null_X)}
    n_eff = int(min(n, len(null_X)))
    key = (k, int(np.round(np.log2(max(n_eff, 2)) * 2)), n_draws, seed)
    if key not in _NULL_THR_CACHE:
        rng = np.random.default_rng(seed)
        sils, deltas = [], []
        for _ in range(n_draws):
            idx = rng.choice(len(null_X), n_eff, replace=len(null_X) < n_eff)
            Xn = null_X[idx]
            km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xn)
            if np.bincount(km.labels_, minlength=k).min() < min_sub:
                continue
            ss = min(sil_sample, n_eff - 1) if n_eff > sil_sample else None
            sils.append(float(silhouette_score(Xn, km.labels_, sample_size=ss, random_state=seed)))
            cen = km.cluster_centers_
            deltas.append(float((cen.max(0) - cen.min(0)).max()))
        _NULL_THR_CACHE[key] = ({"sil_mean": float(np.mean(sils)), "sil_sd": float(np.std(sils)),
                                 "delta_mean": float(np.mean(deltas)), "delta_sd": float(np.std(deltas)),
                                 "n_draws": len(sils)} if sils else {"n_draws": 0})
    m = _NULL_THR_CACHE[key]
    if not m["n_draws"]:
        return {"sil": None, "arm_delta": None, "n_draws": 0, "n_null": len(null_X)}
    return {"sil": m["sil_mean"] + null_z * m["sil_sd"],
            "arm_delta": m["delta_mean"] + null_z * m["delta_sd"],
            "sil_mean": m["sil_mean"], "sil_sd": m["sil_sd"],
            "delta_mean": m["delta_mean"], "delta_sd": m["delta_sd"],
            "n_draws": m["n_draws"], "n_null": len(null_X)}


def detect_subclones(A: pd.DataFrame, *, k_max: int = 4, min_sub: int = 50, seed: int = 0,
                     sil_min: float = 0.15, min_arm_delta: float = 0.03, sil_sample: int = 2000,
                     null_arms: pd.DataFrame | np.ndarray | None = None,
                     null_z: float = 3.0, null_draws: int = 8):
    """Cluster one donor's malignant cells on the arm-CNV matrix into subclones.

    `A` = cells x arms (arm columns only), malignant cells of a single donor; NaN -> 0.
    Sweeps k = 2..k_max (KMeans, n_init=10); picks the k whose partition satisfies ALL of:
      (a) every cluster >= `min_sub` cells,
      (b) silhouette >= the threshold,
      (c) subclone centroids differ on at least one arm by >= the arm-delta threshold (a CNA-scale
          difference, not transcriptional-state noise) — arm-level inferCNV picks up
          activation/cycling programs, so a split whose centroids differ by less than that on every
          arm is not a genuine copy-number subclone.
    If no k qualifies -> k=1 ("genomically uniform clone"), avoiding noise-driven over-splitting.

    **Thresholds.** Pass `null_arms` (held-out diploid reference cells, same arm columns, same
    inferCNV estimator) and both thresholds are calibrated per (n, k) from what that known-diploid
    matrix produces when forced to split: `mean + null_z * sd` over `null_draws` draws. The
    absolute `sil_min` / `min_arm_delta` are then **ignored** — they are the fallback for a matrix
    with no null (nb31/skin behaviour), not a floor. Deliberately so: absolute gates do not transfer
    between inferCNV estimators, since changing the reference rescales the whole matrix. On blood v3
    `min_arm_delta=0.03` sits below the diploid noise floor (inert) while `sil_min=0.15` sits at ~2x
    it (prohibitive), and taking the max of the two would keep the prohibitive one in force.

    Silhouette is estimated on a random subsample of `sil_sample` cells (silhouette_score is O(n^2)).

    Among the k that qualify, the winner maximizes `sil_z` = (silhouette - null mean) / null sd for
    that k. Raw silhouette is not comparable across k (the null silhouette falls as k rises) and the
    ratio silhouette/threshold merely rewards whichever k has the loosest gate; the z-score is
    standardized per k, so it is the only one of the three that compares like with like.

    Returns (labels, meta): `labels` = int array 0..k-1 aligned to A rows; `meta` = dict with
    k, silhouette, centroid_spread, max_arm_delta, sizes, centroids (k x arms DataFrame), and
    `sil_thr` / `arm_delta_thr` / `sil_margin` / `sil_z` / `null_calibrated` for the gates applied.
    """
    X = A.fillna(0.0).to_numpy(dtype=float)
    arms = list(A.columns)
    n = X.shape[0]
    if null_arms is not None:
        NX = (null_arms[arms].fillna(0.0).to_numpy(dtype=float)
              if isinstance(null_arms, pd.DataFrame) else np.asarray(null_arms, dtype=float))
    else:
        NX = None
    best = None
    # gates for k=2 — reported when no k qualifies, i.e. what a split would have had to clear
    thr_used = {"sil_thr": sil_min, "arm_delta_thr": min_arm_delta, "null_calibrated": False}
    if n >= 2 * min_sub:
        for k in range(2, min(k_max, n // min_sub) + 1):
            nul = null_split_thresholds(NX, n, k, n_draws=null_draws, null_z=null_z, seed=seed,
                                        min_sub=min_sub, sil_sample=sil_sample)
            sil_thr = nul["sil"] if nul["sil"] is not None else sil_min
            delta_thr = nul["arm_delta"] if nul["arm_delta"] is not None else min_arm_delta
            if k == 2:
                thr_used = {"sil_thr": round(sil_thr, 4), "arm_delta_thr": round(delta_thr, 4),
                            "null_calibrated": nul["sil"] is not None}
            km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
            lab = km.labels_
            sizes = np.bincount(lab, minlength=k)
            if sizes.min() < min_sub:
                continue
            ss = min(sil_sample, n - 1) if n > sil_sample else None
            sil = float(silhouette_score(X, lab, sample_size=ss, random_state=seed)) if k < n else -1.0
            cen = km.cluster_centers_
            spread = float(np.mean([np.linalg.norm(cen[i] - cen[j])
                                    for i in range(k) for j in range(i + 1, k)]))
            arm_delta = float((cen.max(0) - cen.min(0)).max())   # largest per-arm centroid range
            if sil < sil_thr or arm_delta < delta_thr:
                continue
            # Rank k by excess over its OWN null, in units of that null's spread. Neither
            # alternative works: raw silhouette is not comparable across k (the null silhouette
            # falls with k), and the ratio sil/sil_thr just rewards whichever k has the lowest
            # gate — also large k. The z-score is the one quantity standardized per k.
            margin = sil / sil_thr if sil_thr > 0 else np.nan
            zscore = ((sil - nul["sil_mean"]) / nul["sil_sd"]
                      if nul.get("sil_sd") else margin)
            if best is None or zscore > best["sil_z"]:
                best = {"k": k, "labels": lab, "silhouette": sil, "centroid_spread": spread,
                        "max_arm_delta": arm_delta, "sizes": sizes, "centers": cen,
                        "margin": margin, "sil_z": zscore,
                        "sil_thr": round(sil_thr, 4), "arm_delta_thr": round(delta_thr, 4),
                        "null_calibrated": nul["sil"] is not None}
    if best is None:  # k = 1
        cen = X.mean(0, keepdims=True) if n else np.zeros((1, len(arms)))
        return np.zeros(n, dtype=int), {
            "k": 1, "silhouette": np.nan, "centroid_spread": 0.0, "max_arm_delta": 0.0,
            "sizes": np.array([n]), "centroids": pd.DataFrame(cen, columns=arms),
            "sil_margin": np.nan, "sil_z": np.nan, **thr_used}
    return best["labels"], {
        "k": best["k"], "silhouette": round(best["silhouette"], 4),
        "centroid_spread": round(best["centroid_spread"], 4),
        "max_arm_delta": round(best["max_arm_delta"], 4), "sizes": best["sizes"],
        "centroids": pd.DataFrame(best["centers"], columns=arms),
        "sil_thr": best["sil_thr"], "arm_delta_thr": best["arm_delta_thr"],
        "sil_margin": round(best["margin"], 3), "sil_z": round(float(best["sil_z"]), 2),
        "null_calibrated": best["null_calibrated"]}


# --------------------------------------------------------------------------- #
# §2  major vs minor (transcriptional separation)
# --------------------------------------------------------------------------- #
def major_minor(labels: np.ndarray, latent: np.ndarray, *, auc_thr: float = 0.75,
                sil_thr: float = 0.10):
    """Classify each subclone major (distinct transcriptional cluster) vs minor (intermixed).

    `labels` = subclone ids for a donor's malignant cells; `latent` = matching transcriptional
    embedding rows (e.g. X_scVI). A subclone is **major** if it separates from the rest in the
    latent — either 1-vs-rest LogisticRegression AUC >= `auc_thr` OR its mean silhouette >= `sil_thr`.
    k=1 -> {0: ("single", nan, nan)}. Returns {label: (kind, auc, sil)}.
    """
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {int(uniq[0]) if len(uniq) else 0: ("single", np.nan, np.nan)}
    sil = silhouette_samples(latent, labels)
    out = {}
    for c in uniq:
        m = labels == c
        try:
            auc = roc_auc_score(m.astype(int),
                                LogisticRegression(max_iter=1000, class_weight="balanced")
                                .fit(latent, m.astype(int)).decision_function(latent))
            auc = max(auc, 1 - auc)  # direction-agnostic separability
        except Exception:
            auc = np.nan
        s = float(sil[m].mean())
        kind = "major" if ((np.isfinite(auc) and auc >= auc_thr) or s >= sil_thr) else "minor"
        out[int(c)] = (kind, round(float(auc), 3) if np.isfinite(auc) else np.nan, round(s, 3))
    return out


def major_minor_palette(labels, *, cmap: str = "tab10"):
    """Colour map for combined MAJOR×MINOR subclone labels ('A1','A2','B1'...).

    Same MAJOR letter -> same base hue (from `cmap`); MINOR number -> lightness/saturation shade,
    so A1/A2 read as one transcriptomic family while B1 is a clearly distinct hue. Returns
    {label: rgb}. `labels` = combined strings, first char = major letter, rest = minor number.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    majors = sorted({l[0] for l in labels})
    base = plt.get_cmap(cmap)
    out = {}
    for mi, m in enumerate(majors):
        h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(base(mi % base.N)))
        minors = sorted({l[1:] for l in labels if l[0] == m}, key=lambda x: (len(x), x))
        for ji, mn in enumerate(minors):
            f = 1.0 if len(minors) == 1 else 1.0 - 0.5 * ji / (len(minors) - 1)   # 1.0 -> 0.5
            out[f"{m}{mn}"] = mcolors.hsv_to_rgb((h, s * (0.55 + 0.45 * f), v * (0.55 + 0.45 * f)))
    return out


def major_minor_tracks(adata, latent_key, mal_mask, donors, cnv_col="cnv_subclone", *,
                       leiden_res: float = 0.2, n_neighbors: int = 15, seed: int = 0):
    """Per-donor transcriptomic MAJOR (Leiden on `latent_key`) + CNV MINOR (`cnv_col`) tracks.

    Substrate for the §2c nested subclones. Per donor: kNN + Leiden on the malignant cells' latent
    embedding -> MAJOR relabelled A,B,C.. by size; the §1 CNV subclones -> MINOR relabelled 1,2,3..
    by size; a per-donor UMAP for display. Returns (obs_df, umap_by_donor, mm):
      obs_df       = per-cell DataFrame (major_subclone/minor_subclone/subclone_label = "{donor}_.."
                     for malignant cells, "" elsewhere; sub_umap1/sub_umap2 = NaN elsewhere).
      umap_by_donor[d] = DataFrame(umap1, umap2, label="{Letter}{n}") indexed by that donor's cells.
      mm           = per-donor summary (n_major, n_minor, has_major, has_minor, nmi_major_vs_minor).
    """
    import string
    import scanpy as sc
    from sklearn.metrics import normalized_mutual_info_score
    obs_names = adata.obs_names
    donor_str = adata.obs["donor"].astype(str).values
    major_lab = pd.Series("", index=obs_names, dtype=object)
    minor_lab = pd.Series("", index=obs_names, dtype=object)
    comb_lab = pd.Series("", index=obs_names, dtype=object)
    umap1 = pd.Series(np.nan, index=obs_names)
    umap2 = pd.Series(np.nan, index=obs_names)
    umap_by_donor, mm_rows = {}, []
    lat = adata.obsm[latent_key]
    for d in donors:
        cells = obs_names[mal_mask & (donor_str == d)]
        ci = obs_names.get_indexer(cells)
        sub = sc.AnnData(np.asarray(lat[ci], dtype="float32"),
                         obs=adata.obs.loc[cells, [cnv_col]].copy())
        sub.obsm["X_mu"] = sub.X.copy()
        sc.pp.neighbors(sub, use_rep="X_mu", n_neighbors=min(n_neighbors, sub.n_obs - 1), random_state=seed)
        sc.tl.leiden(sub, resolution=leiden_res, random_state=seed, key_added="_leiden")
        sc.tl.umap(sub, random_state=seed)
        maj_order = sub.obs["_leiden"].value_counts().index.tolist()
        maj = sub.obs["_leiden"].map({cl: string.ascii_uppercase[i] for i, cl in enumerate(maj_order)}).astype(str)
        cnv = sub.obs[cnv_col].astype(str)
        min_order = cnv.value_counts().index.tolist()
        minr = cnv.map({cl: str(i + 1) for i, cl in enumerate(min_order)}).astype(str)
        major_lab.loc[cells] = (d + "_" + maj).values
        minor_lab.loc[cells] = (d + "_" + minr).values
        comb_lab.loc[cells] = d + "_" + maj.values + minr.values
        umap1.loc[cells] = sub.obsm["X_umap"][:, 0]
        umap2.loc[cells] = sub.obsm["X_umap"][:, 1]
        umap_by_donor[d] = pd.DataFrame({"umap1": sub.obsm["X_umap"][:, 0],
                                         "umap2": sub.obsm["X_umap"][:, 1],
                                         "label": maj.values + minr.values}, index=cells)
        k_maj, k_min = len(maj_order), len(min_order)
        nmi = normalized_mutual_info_score(maj.values, minr.values) if min(k_maj, k_min) > 1 else 0.0
        mm_rows.append({"donor": d, "n_major": k_maj, "n_minor": k_min,
                        "has_major": k_maj > 1, "has_minor": k_min > 1,
                        "nmi_major_vs_minor": round(float(nmi), 3)})
    obs_df = pd.DataFrame({"major_subclone": major_lab, "minor_subclone": minor_lab,
                           "subclone_label": comb_lab, "sub_umap1": umap1, "sub_umap2": umap2})
    return obs_df, umap_by_donor, pd.DataFrame(mm_rows)


def nested_subclone_labels(umap_by_donor, arm_df, arm_cols, donors, *, k_max: int = 4,
                           min_sub: int = 50, seed: int = 0, sil_min: float = 0.15,
                           min_arm_delta: float = 0.03, null_arms=None,
                           null_z: float = 3.0, null_draws: int = 8):
    """Cohort-wide §2c nested subclones: transcriptomic MAJOR -> CNV MINOR within each major.

    Generalises nb23 §2c (which runs for a single selected sample) to every donor. For each donor,
    the MAJOR letter is the first char of `umap_by_donor[d]["label"]` (§2 Leiden on the mu embedding);
    within each major we try `detect_subclones` on the arm-CNV matrix. A major that genuinely splits
    on CNV gets minors 1,2,.. by descending size -> nested label "{donor}_{Letter}{n}"; a major with
    no CNV substructure stays "{donor}_{Letter}".

    `null_arms` is forwarded to `detect_subclones` so the nested (within-major) split is judged
    against the same diploid noise floor as the top-level one.

    Returns (labels, split_summary): `labels` = Series (cell_id -> "{donor}_{Letter}{n}") over the
    supplied donors' malignant cells; `split_summary` = DataFrame(donor, major, n_cells, k).
    """
    out, rows = {}, []
    for d in donors:
        U = umap_by_donor.get(d)
        if U is None:
            continue
        maj_of = U["label"].astype(str).str[0]                 # MAJOR letter per cell
        for mj in sorted(maj_of.unique()):
            cm = maj_of.index[maj_of.values == mj]
            lab, meta = detect_subclones(arm_df.reindex(cm)[arm_cols], k_max=k_max, min_sub=min_sub,
                                         seed=seed, sil_min=sil_min, min_arm_delta=min_arm_delta,
                                         null_arms=null_arms, null_z=null_z,
                                         null_draws=null_draws)
            if meta["k"] > 1:
                remap = {old: i + 1 for i, old in enumerate(pd.Series(lab).value_counts().index)}
                for cid, l in zip(cm, lab):
                    out[cid] = f"{d}_{mj}{remap[l]}"
            else:
                for cid in cm:
                    out[cid] = f"{d}_{mj}"
            rows.append({"donor": d, "major": mj, "n_cells": len(cm), "k": meta["k"]})
    return pd.Series(out, dtype=object), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# §3  trunk / branch arm events + tree
# --------------------------------------------------------------------------- #
def call_arm_events(A_sub: pd.DataFrame, A_benign: pd.DataFrame, *, z_thr: float = 2.5,
                    eps: float = 0.01) -> pd.Series:
    """Per-arm gain(+1)/loss(-1)/none(0) for one subclone vs same-donor benign-T diploid baseline.

    Robust MAD z-score of the subclone-mean arm CNV against the benign-cell distribution
    (nb20 cell-15 logic). Requires |z| > z_thr AND |subclone mean| > eps (magnitude floor).
    """
    b_med = A_benign.median()
    b_mad = (A_benign - b_med).abs().median() * 1.4826 + 1e-6
    z = (A_sub.mean() - b_med) / b_mad
    mean = A_sub.mean()
    return pd.Series({a: (1 if (z[a] > z_thr and mean[a] > eps) else
                          -1 if (z[a] < -z_thr and mean[a] < -eps) else 0) for a in A_sub.columns})


def trunk_branch(clone_events: pd.Series, centroids: pd.DataFrame, *, branch_delta: float = 0.03):
    """Split arm events into trunk (pan-clonal, ancestral) vs branch (subclone-divergent, private).

    trunk  = arms altered in the CLONE AS A WHOLE (`clone_events` != 0; call_arm_events of all the
             donor's malignant cells vs benign) — the shared ancestral events.
    branch = arms whose subclone `centroids` diverge by >= `branch_delta` (max minus min across
             subclones) — the private / variably-amplified events that define the divergence edges.
    An arm can be both (a shared event further amplified in one subclone). Robust to a near-diploid
    "transitional" subclone that would otherwise erase an all-subclones-unanimous trunk definition.

    Returns dict(trunk, branch, trunk_signed).
    """
    trunk = [a for a in clone_events.index if clone_events[a] != 0]
    trunk_signed = {a: int(clone_events[a]) for a in trunk}
    rng = centroids.max() - centroids.min()
    branch = [a for a in centroids.columns if float(rng[a]) >= branch_delta]
    return {"trunk": trunk, "branch": branch, "trunk_signed": trunk_signed}


def subclone_linkage(centroids: pd.DataFrame, *, root_diploid: bool = True):
    """Average-linkage hierarchy on subclone arm centroids for a rooted divergent-evolution tree.

    With `root_diploid`, prepend a diploid (all-zero) ancestor so the dendrogram is rooted at the
    normal genome. Returns (Z, leaf_labels) for scipy.cluster.hierarchy.dendrogram.
    """
    labels = list(centroids.index)
    M = centroids.to_numpy(dtype=float)
    if root_diploid:
        M = np.vstack([np.zeros((1, M.shape[1])), M])
        labels = ["diploid"] + labels
    Z = linkage(M, method="average", metric="euclidean")
    return Z, labels


# --------------------------------------------------------------------------- #
# §4  TCR-beta variant class (reconcile CNV subclones with nb21 ALICE families)
# --------------------------------------------------------------------------- #
def extract_trb(obs: pd.DataFrame) -> pd.Series:
    """TRB CDR3 per cell from the unified clone key (`TRB:<cdr3>`) falling back to `trb_cdr3`."""
    trb = pd.Series("", index=obs.index)
    if "tcr_clone_id" in obs.columns:
        trb = obs["tcr_clone_id"].astype(str).str.extract(r"^TRB:([A-Z]+)$", expand=False).fillna("")
    if "trb_cdr3" in obs.columns:
        trb = trb.mask(trb == "", obs["trb_cdr3"].astype(str))
    return trb


def tcr_variant_class(trb: pd.Series, founder: set, family: set) -> pd.Series:
    """founder / alice_variant (in family, not founder) / other, per cell given its TRB CDR3."""
    fam_only = set(family) - set(founder)
    return pd.Series(np.where(trb.isin(founder), "founder",
                     np.where(trb.isin(fam_only), "alice_variant", "other")), index=trb.index)


# --------------------------------------------------------------------------- #
# §5  differential expression
# --------------------------------------------------------------------------- #
def pseudobulk_counts(adata, sample_col, layer="raw_counts", min_cells=20):
    """Sum raw counts per pseudo-sample. (Lifted from nb03 03_mrvi_replication cell 2.)"""
    X = adata.layers[layer]
    samples = adata.obs[sample_col].astype(str).values
    rows, idx, ncells = [], [], {}
    for s in pd.unique(samples):
        mask = samples == s
        n = int(mask.sum())
        if n < min_cells:
            continue
        sub = X[mask]
        v = np.asarray(sub.sum(axis=0)).ravel() if _sp.issparse(sub) else np.asarray(sub).sum(0)
        rows.append(v); idx.append(s); ncells[s] = n
    counts = pd.DataFrame(np.vstack(rows), index=idx, columns=adata.var_names.astype(str)).round().astype(int)
    return counts, pd.Series(ncells, name="n_cells")


def run_pydeseq2(counts, metadata, design, contrast, min_total=10, n_cpus=4):
    """Pseudobulk DESeq2 (NB GLM + Wald, BH). (Lifted from nb03.) Returns results_df with aliases."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts = counts.loc[:, counts.sum(axis=0) >= min_total]
    try:
        dds = DeseqDataSet(counts=counts, metadata=metadata, design=design, quiet=True, n_cpus=n_cpus)
    except TypeError:  # older pydeseq2 API
        factors = [t.strip() for t in design.replace("~", "").split("+") if t.strip()]
        dds = DeseqDataSet(counts=counts, metadata=metadata, design_factors=factors, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    res = st.results_df.copy()
    res["logfoldchanges"] = res["log2FoldChange"]
    res["pvals_adj"] = res["padj"]
    return res


def subclone_markers_wilcoxon(adata_donor, group_col="cnv_subclone", n_genes=25):
    """Within-donor single-cell subclone DE (Wilcoxon; the paper's presto::wilcoxauc baseline).

    No biological replicates exist within one patient, so pseudobulk DESeq2 is not valid here —
    use single-cell Wilcoxon per subclone (cross-donor major/minor contrasts use pseudobulk instead).
    Returns a long DataFrame: subclone, rank, gene, log2fc, pval_adj, score.
    """
    import scanpy as sc
    ad = adata_donor.copy()
    sc.tl.rank_genes_groups(ad, group_col, method="wilcoxon", n_genes=n_genes)
    r = ad.uns["rank_genes_groups"]
    rows = []
    for grp in r["names"].dtype.names:
        for i in range(len(r["names"][grp])):
            rows.append({"subclone": grp, "rank": i + 1, "gene": r["names"][grp][i],
                         "log2fc": float(r["logfoldchanges"][grp][i]),
                         "pval_adj": float(r["pvals_adj"][grp][i]),
                         "score": float(r["scores"][grp][i])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# §6  functional programs (score genes per subclone)
# --------------------------------------------------------------------------- #
# S / G2M cell-cycle genes: Tirosh/Regev list shipped with Seurat (used by sc.tl.score_genes_cell_cycle).
CC_S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
              "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1",
              "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
              "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
              "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
CC_G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
                "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L",
                "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B",
                "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2",
                "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1",
                "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]

# Divergent-subclone programs (Cancer Discovery 2025 Fig 4/5). Missing genes are dropped by score_genes.
PROGRAM_SETS = {
    "prolif":       ["MKI67", "TOP2A", "PCNA", "CCNB1", "CDK1", "TYMS", "BIRC5"],
    "oxphos":       ["NDUFA1", "NDUFB1", "COX6C", "ATP5F1A", "ATP5MC1", "UQCRB", "SDHB", "CYCS"],
    "glycolysis":   ["HK2", "PKM", "LDHA", "PGK1", "ENO1", "SLC2A1", "ALDOA"],
    "cytokine":     ["IL4", "IL13", "IL21", "IL22", "TNF"],
    "th1":          ["TBX21", "CXCR3", "STAT1"],
    "th2":          ["GATA3", "IL4", "IL5", "IL13", "CCR4"],
    "th17":         ["RORC", "IL17A", "IL23R", "CCR6"],
    "skin_homing":  ["CCR4", "CCR6", "CD69", "CCR10", "SELPLG"],                # tissue-resident / skin
    "recirc_homing": ["SELL", "CCR7", "S1PR1", "KLF2", "TCF7"],                 # lymphoid recirculation
    # ECM-remodeling / protease program (tumor-intrinsic). TIMP1 (inhibitor) is kept OUT of the
    # summed score — read the protease:TIMP1 ratio in the dot plot instead.
    "protease":     ["MMP14", "MMP25", "MMP9", "MMP2", "PLAU", "PLAUR", "GZMB",
                     "ADAM17", "ADAM10", "ADAMTS4", "CTSC"],
}


def score_programs(adata, program_sets=PROGRAM_SETS, prefix="prog_", cell_cycle=True, seed=0):
    """Score each program on the (log-normalised X) cells with sc.tl.score_genes; add obs columns.

    Adds `<prefix><name>` per program and (if cell_cycle) `S_score`/`G2M_score`/`phase`.
    Returns the list of added obs column names.
    """
    import scanpy as sc
    added = []
    for name, genes in program_sets.items():
        present = [g for g in genes if g in adata.var_names]
        col = f"{prefix}{name}"
        if len(present) >= 2:
            sc.tl.score_genes(adata, present, score_name=col, random_state=seed)
        else:
            adata.obs[col] = np.nan
        added.append(col)
    if cell_cycle:
        s = [g for g in CC_S_GENES if g in adata.var_names]
        g2m = [g for g in CC_G2M_GENES if g in adata.var_names]
        if len(s) >= 2 and len(g2m) >= 2:
            sc.tl.score_genes_cell_cycle(adata, s_genes=s, g2m_genes=g2m)
            added += ["S_score", "G2M_score", "phase"]
    return added


# --------------------------------------------------------------------------- #
# §7  paper functional-marker panels + descriptive dot plots (Cancer Discovery 2025 Fig 4)
# --------------------------------------------------------------------------- #
# Marker panels for the paper's subclonal characterization, grouped into the five functional
# categories. Each list = the paper's named genes + canonical members of that program. Missing
# genes are dropped by `subclone_dotplot`. (skin/tissue-resident homing | lymphoid recirculation).
PAPER_PANELS = {
    "homing":     ["CCR4", "CCR6", "CCR8", "CCR10", "CD69", "ITGA1", "SELPLG", "CXCR3",
                   "SELL", "CCR7", "S1PR1", "KLF2", "TCF7", "ITGB2", "RUNX1"],
    "spatial":    ["CCR4", "CCR10", "CD69", "CXCR3", "S1PR1", "SELL", "CCR7", "KLF2"],
    "cytokine":   ["IL4", "IL5", "IL13", "IL17A", "IL21", "IL22", "IL26",
                   "TNF", "TGFB1", "CSF2"],
    "metabolism": PROGRAM_SETS["oxphos"] + PROGRAM_SETS["glycolysis"],
    "signaling":  ["TOX", "SESN3", "KIR3DL2", "FCRL3", "CD7", "DPP4",
                   "NFKB1", "RELB", "NFKBIA", "TNFAIP3", "TRAF1",
                   "CTLA4", "PDCD1", "TIGIT", "ICOS", "CD40LG"],
    # core tumor-intrinsic discrimination panel (ECM-remodeling proteases + TIMP1 inhibitor;
    # read the protease:TIMP1 ratio across subclones).
    "protease":   ["MMP14", "MMP25", "MMP9", "MMP2", "PLAU", "PLAUR", "GZMB",
                   "ADAM17", "ADAM10", "ADAMTS4", "CTSC", "TIMP1"],
}


def present_panels(adata, panels=PAPER_PANELS):
    """Filter each category to genes present in `adata.var_names` (order + dedup preserved)."""
    vs = set(adata.var_names)
    out = {}
    for cat, genes in panels.items():
        seen, keep = set(), []
        for g in genes:
            if g in vs and g not in seen:
                keep.append(g); seen.add(g)
        if keep:
            out[cat] = keep
    return out


def subclone_dotplot(ad, groupby, panels=PAPER_PANELS, *, title=None, save=None,
                     standard_scale="var", figsize=None, categories_order=None):
    """Descriptive paper-style dot plot (mean expr = colour, % expressing = size) over `panels`.

    `sc.pl.dotplot` renders the category dict as grouped columns with brackets = the paper Fig 4
    layout. `standard_scale='var'` scales each gene 0..1 across groups so subclone differences pop.
    Returns the DotPlot object (already saved to `save` if given).
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    filt = present_panels(ad, panels)
    dp = sc.pl.dotplot(ad, filt, groupby=groupby, standard_scale=standard_scale,
                       categories_order=categories_order, figsize=figsize,
                       title=title, return_fig=True)
    dp.make_figure()
    if save is not None:
        dp.fig.savefig(str(save), bbox_inches="tight")
    plt.show()
    plt.close(dp.fig)          # show once; don't let the inline backend re-render at cell end


def layer_split_key(obs, subclone_col, *, min_cells: int = 30, tissue_col: str = "tissue"):
    """Build a "{subclone} | {epidermis|dermis}" key for the cross-tissue dot plot.

    Only cells whose `tissue` matches epidermis/dermis are keyed, and only for subclones present in
    BOTH layers with >= `min_cells` each (matches nb23 §7's contains('epiderm')/contains('derm')).
    Returns (key Series over obs.index, "" elsewhere; list of qualifying subclones).
    """
    layer = obs[tissue_col].astype(str).str.lower()
    is_epi = layer.str.contains("epiderm")
    is_derm = layer.str.contains("derm") & ~is_epi
    lname = pd.Series("", index=obs.index, dtype=object)
    lname[is_epi.values] = "epidermis"
    lname[is_derm.values] = "dermis"
    sub = obs[subclone_col].astype(str)
    ok = []
    for s in sub[lname != ""].unique():
        if s == "":
            continue
        m = sub.values == s
        n_epi = int((lname[m] == "epidermis").sum())
        n_derm = int((lname[m] == "dermis").sum())
        if n_epi >= min_cells and n_derm >= min_cells:
            ok.append(s)
    key = pd.Series("", index=obs.index, dtype=object)
    m = sub.isin(ok).values & (lname != "").values
    key[m] = (sub[m] + " | " + lname[m]).values
    return key, sorted(ok)


def classify_subclone_tropism(obs, subclone_col, *, tissue_col="tissue", min_cells=30,
                              epi_hi=0.7, epi_lo=0.3):
    """Label each subclone epidermal / dermal / mixed by its epidermis-fraction among layer-resolved
    cells. A subclone needs >= `min_cells` layer-resolved cells else 'unclassified'. epi_frac >=
    `epi_hi` -> epidermal, <= `epi_lo` -> dermal, else mixed. Returns (table, cell_class):
      table      = per-subclone DataFrame(subclone, n_epi, n_derm, n_layer, epi_frac, tropism).
      cell_class = Series over obs.index: each cell -> its subclone's tropism, or '' if the subclone
                   is unclassified (so it drops out of any groupby).
    """
    layer = obs[tissue_col].astype(str).str.lower()
    is_epi = layer.str.contains("epiderm").values
    is_derm = (layer.str.contains("derm") & ~layer.str.contains("epiderm")).values
    sub = obs[subclone_col].astype(str)
    rows = []
    for s in sub.unique():
        if s == "":
            continue
        m = sub.values == s
        n_epi = int((is_epi & m).sum()); n_derm = int((is_derm & m).sum())
        n_layer = n_epi + n_derm
        if n_layer < min_cells:
            trop, frac = "unclassified", np.nan
        else:
            frac = n_epi / n_layer
            trop = "epidermal" if frac >= epi_hi else "dermal" if frac <= epi_lo else "mixed"
        rows.append({"subclone": s, "n_epi": n_epi, "n_derm": n_derm, "n_layer": n_layer,
                     "epi_frac": frac, "tropism": trop})
    table = pd.DataFrame(rows)
    cmap = table.set_index("subclone")["tropism"].to_dict()
    cell_class = pd.Series([cmap.get(s, "") for s in sub.values], index=obs.index, dtype=object)
    cell_class = cell_class.where(cell_class.isin(["epidermal", "dermal", "mixed"]), "")
    return table, cell_class


def plot_nested_sample(adata, sample, umap_by_donor, arm_df, arm_cols, mal_mask, *,
                       nested_col="nested_subclone", tissue_col="tissue", fig_dir=None):
    """One-sample illustration of the §2c nested subclones (moved out of the notebook).

    Row 1 = per-MAJOR UMAP panels (highlight that major's nested minors, grey the rest) + a
    cell-location UMAP (tissue); row 2 = the §1 arm-CNV heatmap grouped by the nested label.
    Reads the cohort `nested_col` from `adata.obs`. Returns the Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    donor_str = adata.obs["donor"].astype(str).values
    cells = adata.obs_names[mal_mask & (donor_str == sample)]
    U = umap_by_donor[sample].copy()
    U["nlabel"] = adata.obs.loc[U.index, nested_col].astype(str).str.rsplit("_", n=1).str[-1].values
    maj_of = pd.Series([l[0] for l in U["nlabel"].values], index=U.index)
    labels_all = sorted(U["nlabel"].unique(), key=lambda s: (s[0], s[1:]))
    cpal = {lb: plt.get_cmap("tab20")(i % 20) for i, lb in enumerate(labels_all)}
    majors = sorted(maj_of.unique())

    ncol = len(majors) + 1
    fig = plt.figure(figsize=(3.2 * ncol, 6.4))
    gs = GridSpec(2, ncol, height_ratios=[3.0, 2.4], hspace=0.30, wspace=0.12)

    for j, mj in enumerate(majors):
        ax = fig.add_subplot(gs[0, j])
        inmaj = maj_of.values == mj
        ax.scatter(U["umap1"].values[~inmaj], U["umap2"].values[~inmaj], s=5, lw=0, color="lightgrey")
        for lb in sorted(U["nlabel"][inmaj].unique(), key=lambda s: s[1:]):
            m = U["nlabel"].values == lb
            ax.scatter(U["umap1"].values[m], U["umap2"].values[m], s=9, lw=0, color=cpal[lb], label=lb)
        n_min = U["nlabel"][inmaj].nunique()
        tag = f"{n_min} CNV minors" if n_min > 1 else "no CNV split"
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"major {mj} — {tag}", fontsize=8)
        ax.legend(markerscale=2, fontsize=6, frameon=False, loc="best", title="subclone", title_fontsize=6)

    ax = fig.add_subplot(gs[0, len(majors)])
    loc = adata.obs.loc[U.index, tissue_col].astype(str)
    lcats = sorted(c for c in loc.unique() if c.lower() not in ("nan", "", "none"))
    lpal = {c: plt.get_cmap("Set2")(i % 8) for i, c in enumerate(lcats)}
    unk = ~loc.isin(lcats).values
    if unk.any():
        ax.scatter(U["umap1"].values[unk], U["umap2"].values[unk], s=5, lw=0, color="lightgrey", label="unknown")
    for c in lcats:
        m = loc.values == c
        ax.scatter(U["umap1"].values[m], U["umap2"].values[m], s=8, lw=0, color=lpal[c], label=c)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("cell location (tissue)", fontsize=9)
    ax.legend(markerscale=2, fontsize=6, frameon=False, loc="best", title="tissue", title_fontsize=6)

    ax = fig.add_subplot(gs[1, :])
    Ad = arm_df.reindex(cells)[arm_cols].fillna(0.0)
    lab_cell = U["nlabel"].reindex(cells).values
    order = np.argsort(lab_cell, kind="stable")
    vlim = float(np.nanpercentile(np.abs(Ad.values), 98)) or 0.05
    ax.imshow(Ad.values[order], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="none")
    sorted_lab = lab_cell[order]
    bounds = np.where(sorted_lab[:-1] != sorted_lab[1:])[0] + 1
    for b in bounds:
        ax.axhline(b - 0.5, color="k", lw=0.6)
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(sorted_lab)]])
    for s, e in zip(starts, ends):
        lb = sorted_lab[s]
        ax.text(-1.2, (s + e) / 2, lb, color=cpal.get(lb, "k"), ha="right", va="center",
                fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(arm_cols))); ax.set_xticklabels(arm_cols, rotation=90, fontsize=5)
    ax.set_yticks([]); ax.set_ylabel(f"{len(cells)} malignant cells")
    ax.set_title(f"{sample}: arm-CNV per cell (grouped by nested major->CNV label)", fontsize=8)

    fig.suptitle(f"{sample}: nested subclones — Leiden MAJOR then CNV MINOR within each major", y=0.99)
    if fig_dir is not None:
        fig.savefig(f"{fig_dir}/subclone_nested_{sample}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8  cohort-overview + sample-menu plots (lifted from the notebook for readability)
# --------------------------------------------------------------------------- #
def layer_split_donors(obs, subclone_col, *, donor_col="donor", tissue_col="tissue",
                       min_cells: int = 30):
    """Donors testable for epidermis/dermis subclone tropism -> the Section-1/2 sample menu.

    Among each donor's labelled malignant cells that are layer-resolved (`tissue` contains
    'epiderm' or 'derm'), keep donors with cells in BOTH layers, >= 2 subclones, and
    >= `min_cells` layer-resolved cells (matches nb23 §7's Fisher gate). Returns a DataFrame
    (donor, n_subclones, n_epi, n_derm, subclones) sorted by n_subclones then n_epi; the
    `subclones` column lists the short labels (donor prefix stripped) for the printed menu.
    """
    layer = obs[tissue_col].astype(str).str.lower()
    is_epi = layer.str.contains("epiderm")
    is_derm = layer.str.contains("derm") & ~is_epi
    df = pd.DataFrame({"donor": obs[donor_col].astype(str).values,
                       "sub": obs[subclone_col].astype(str).values,
                       "epi": is_epi.values, "derm": is_derm.values})
    df = df[(df["sub"] != "") & (df["epi"] | df["derm"])]      # layer-resolved, labelled cells
    rows = []
    for d, g in df.groupby("donor"):
        n_epi, n_derm = int(g["epi"].sum()), int(g["derm"].sum())
        subs = sorted(g["sub"].unique())
        if n_epi >= 1 and n_derm >= 1 and len(subs) >= 2 and (n_epi + n_derm) >= min_cells:
            rows.append({"donor": d, "n_subclones": len(subs), "n_epi": n_epi, "n_derm": n_derm,
                         "subclones": ",".join(s.rsplit("_", 1)[-1] for s in subs)})
    out = pd.DataFrame(rows, columns=["donor", "n_subclones", "n_epi", "n_derm", "subclones"])
    return out.sort_values(["n_subclones", "n_epi"], ascending=False).reset_index(drop=True)


NULL_TAG = "|NULL|"


def split_null_rows(arm_raw: pd.DataFrame, *, hc_prefix: str = "H__HC"):
    """Split a raw arm-CNV matrix into (query cells, held-out healthy-CD4 diploid null).

    The v3 inferCNV pass relabels half the shared reference as `ref_null` and renames those rows
    `<run>|NULL|<cell>` (`skin_T_cnv_helpers._split_null`), so they traverse the identical estimator
    as the tumour cells and are the empirical diploid null. The shared reference is healthy CD4:
    `hc_atlas` (same-atlas healthy blood CD4, cell names `H__HC1_Blood|...`) plus `healthy_pbmc`
    (external OneK1K healthy PBMC CD4, bare barcodes).

    Returns (query, null): `null` carries `ref_source` in {'hc_atlas', 'healthy_pbmc'} and keeps its
    `donor` column (the inferCNV run the null was held out in — the null is redrawn per run).
    """
    idx = pd.Series(arm_raw.index.astype(str), index=arm_raw.index)
    is_null = idx.str.contains(NULL_TAG, regex=False)
    null = arm_raw[is_null.to_numpy()].copy()
    orig = idx[is_null.to_numpy()].str.split(NULL_TAG, regex=False).str[1]
    null["ref_source"] = np.where(orig.str.startswith(hc_prefix).to_numpy(),
                                  "hc_atlas", "healthy_pbmc")
    return arm_raw[~is_null.to_numpy()], null


def arm_null_stats(null_arms: pd.DataFrame, arm_cols):
    """Per-arm diploid mean and single-cell SD over the held-out healthy-CD4 reference cells.

    Arms with no positioned genes in the null (acrocentric p-arms) come back NaN and plot blank
    rather than as a spurious zero.
    """
    N = null_arms[list(arm_cols)].to_numpy(dtype=float)
    ok = np.isfinite(N).sum(axis=0) >= 2
    mu = np.full(N.shape[1], np.nan)
    sd = np.full(N.shape[1], np.nan)
    if ok.any():
        mu[ok] = np.nanmean(N[:, ok], axis=0)
        sd[ok] = np.nanstd(N[:, ok], axis=0)
    return pd.DataFrame({"mean": mu, "sd": np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)},
                        index=list(arm_cols))


def arm_z(prof: pd.DataFrame, null_stats: pd.DataFrame, arm_cols):
    """Z-score arm profiles against the diploid null: (profile - null mean) / null single-cell SD.

    Units are single-cell diploid SDs, so the scale is fixed by the null rather than by the
    profiles themselves — the point being that a heatmap of *raw* arm means with a percentile
    `vlim` self-normalizes, which makes a change of inferCNV reference (which rescales the whole
    matrix) invisible. Dividing by the null's own spread rather than by sqrt(n) keeps this a
    profile, not a significance score, so it stays comparable across donors of different sizes.
    """
    cols = [c for c in arm_cols]
    mu = null_stats.loc[cols, "mean"].to_numpy(dtype=float)
    sd = null_stats.loc[cols, "sd"].to_numpy(dtype=float)
    return pd.DataFrame((prof[cols].to_numpy(dtype=float) - mu) / sd,
                        index=prof.index, columns=cols)


def healthy_ref_profiles(null_arms: pd.DataFrame, arm_cols, *, run: str | None = None,
                         by_source: bool = True):
    """Mean arm profile of the healthy-CD4 reference cells — the diploid row for the heatmaps.

    `run` restricts to the null held out in one inferCNV run (batch-matched to that sample); falls
    back to the pooled null when that run has none. Returns (profiles, counts).
    """
    N = null_arms
    if run is not None and "donor" in N.columns:
        sel = N[N["donor"].astype(str) == str(run)]
        if len(sel) >= 50:
            N = sel
    keys = (["hc_atlas", "healthy_pbmc"] if by_source and "ref_source" in N.columns else [None])
    rows, counts = {}, {}
    for k in keys:
        sub = N if k is None else N[N["ref_source"].astype(str) == k]
        if len(sub) < 20:
            continue
        label = "ref healthy CD4" if k is None else f"ref healthy CD4 ({k})"
        rows[label] = sub[arm_cols].mean(axis=0)
        counts[label] = len(sub)
    if not rows:
        return pd.DataFrame(columns=list(arm_cols)), {}
    return pd.DataFrame(rows).T[list(arm_cols)], counts


def _annot_strip(ax, annot: pd.DataFrame, *, fontsize=5):
    """Draw a categorical colour strip (rows x annotation columns) left of a heatmap."""
    import matplotlib.pyplot as plt
    codes = np.zeros((len(annot), annot.shape[1]))
    for j, c in enumerate(annot.columns):
        vals = annot[c].astype(str)
        cats = {v: i for i, v in enumerate(sorted(vals.unique()))}
        codes[:, j] = [cats[v] / max(1, len(cats) - 1) for v in vals]
    ax.imshow(codes, aspect="auto", cmap="tab20", vmin=0, vmax=1, interpolation="none")
    ax.set_xticks(range(annot.shape[1]))
    ax.set_xticklabels(list(annot.columns), rotation=90, fontsize=fontsize)
    ax.set_yticks([])


def plot_cohort_arm_profiles(centroids, donors, arm_cols, *, null_arms=None, ref_rows=None,
                             row_annot=None, sizes=None, vlim=3.0, cluster=True, save=None,
                             show=True,
                             title="CNV-subclone arm profiles vs healthy-CD4 diploid null"):
    """Cohort heatmap of CNV-subclone mean arm profiles (subclones x arms).

    `centroids` = {donor: DataFrame(subclone x arm)} from `detect_subclones` (§1 axis).

    With `null_arms` (held-out healthy-CD4 diploid cells, see `split_null_rows`) the profiles are
    z-scored against that null (`arm_z`) and `vlim` is a **fixed** number of diploid SDs, so two
    runs of a different inferCNV reference are directly comparable. Reference rows built from the
    same null are appended at the bottom and must read flat — they are the visual zero. Without
    `null_arms` this falls back to raw arm means on a self-normalizing percentile scale (nb31
    behaviour), which is exactly what hides an estimator change.

    `cluster=True` orders subclone rows by ward linkage so recurrent arm blocks group; genomic
    column order is never clustered. `row_annot` = DataFrame(row label -> categorical columns).
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list

    arm_cols = list(arm_cols)
    prof = pd.concat([centroids[d] for d in donors])[arm_cols]
    if null_arms is not None:
        stats = arm_null_stats(null_arms, arm_cols)
        prof = arm_z(prof, stats, arm_cols)
        if ref_rows is None:
            ref_rows, _ = healthy_ref_profiles(null_arms, arm_cols)
        if len(ref_rows):
            ref_rows = arm_z(ref_rows[arm_cols], stats, arm_cols)
        cbar_label = "arm CNV (z vs healthy-CD4 diploid null)"
        vlim = float(vlim)
    else:
        cbar_label = "inferCNV (arm mean)"
        vlim = float(np.nanpercentile(np.abs(prof.values), 98)) or 0.05
        ref_rows = ref_rows if ref_rows is not None else pd.DataFrame(columns=arm_cols)

    if cluster and len(prof) >= 3:
        Z = linkage(np.nan_to_num(prof.to_numpy(dtype=float)), method="ward")
        prof = prof.iloc[leaves_list(Z)]
    if row_annot is not None:
        row_annot = row_annot.reindex(prof.index)

    body = pd.concat([prof, ref_rows[arm_cols]]) if len(ref_rows) else prof
    labels = list(body.index)
    if sizes is not None:
        labels = [f"{s}  (n={sizes.get(s, '')})" if s in sizes else str(s) for s in body.index]

    h = max(4.0, 0.14 * len(body))
    if row_annot is not None and row_annot.shape[1]:
        fig, (axa, ax) = plt.subplots(1, 2, figsize=(11.6, h), sharey=True, layout="constrained",
                                      gridspec_kw={"width_ratios": [0.03 * row_annot.shape[1], 1]})
        pad = pd.DataFrame("", index=ref_rows.index, columns=row_annot.columns)
        _annot_strip(axa, pd.concat([row_annot, pad]) if len(ref_rows) else row_annot)
    else:
        fig, ax = plt.subplots(figsize=(11, h), layout="constrained")
    im = ax.imshow(body.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="none")
    if len(ref_rows):
        ax.axhline(len(prof) - 0.5, color="k", lw=1.2)
    ax.set_xticks(range(len(arm_cols))); ax.set_xticklabels(arm_cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(body)))
    ax.set_yticklabels(labels, fontsize=4)
    for t, s in zip(ax.get_yticklabels(), body.index):
        if s in set(ref_rows.index):
            t.set_fontsize(6); t.set_fontweight("bold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label=cbar_label)
    if save is not None:
        fig.savefig(str(save), bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_sample_arm_profile(centroids_d, sample, arm_cols, *, null_arms=None, sizes=None,
                            run=None, vlim=3.0, band=2.0, save=None, show=True):
    """One sample's CNV-subclone arm profiles: heatmap + step profile over the diploid null band.

    Rows = that sample's subclone centroids plus the healthy-CD4 reference row(s) built from the
    null held out in this sample's own inferCNV run (`run`, defaults to `sample`). The lower panel
    plots the same profiles in genomic order against a +/- `band`-SD diploid envelope, because at
    k=1 a one-row heatmap cannot show whether the clone actually departs from diploid.
    """
    import matplotlib.pyplot as plt

    arm_cols = list(arm_cols)
    prof = centroids_d[arm_cols].copy()
    ref = pd.DataFrame(columns=arm_cols)
    if null_arms is not None:
        stats = arm_null_stats(null_arms, arm_cols)
        prof = arm_z(prof, stats, arm_cols)
        ref, ref_n = healthy_ref_profiles(null_arms, arm_cols, run=run if run is not None else sample)
        if len(ref):
            ref = arm_z(ref[arm_cols], stats, arm_cols)
        ylab = "z vs healthy-CD4 diploid null"
    else:
        vlim = float(np.nanpercentile(np.abs(prof.values), 98)) or 0.05
        band = None
        ylab = "inferCNV (arm mean)"

    body = pd.concat([prof, ref]) if len(ref) else prof
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 2.0 + 0.22 * len(body) + 2.2),
                                  layout="constrained",
                                  gridspec_kw={"height_ratios": [max(1.0, 0.5 * len(body)), 2.2]})
    im = ax.imshow(body.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="none")
    if len(ref):
        ax.axhline(len(prof) - 0.5, color="k", lw=1.2)
    ax.set_xticks([]); ax.set_yticks(range(len(body)))
    ax.set_yticklabels([f"{s}  (n={sizes.get(s, '')})" if sizes and s in sizes else str(s)
                        for s in body.index], fontsize=7)
    ax.set_title(f"{sample}: CNV-subclone arm profiles vs healthy CD4")
    # colorbar spans both axes so the heatmap columns stay aligned with the step plot below
    fig.colorbar(im, ax=[ax, ax2], fraction=0.025, pad=0.01, label=ylab)

    x = np.arange(len(arm_cols))
    if band is not None:
        ax2.fill_between(x, -band, band, color="0.85", step="mid",
                         label=f"diploid +/-{band:g} SD")
        ax2.axhline(0, color="0.4", lw=0.6)
    for s in prof.index:
        ax2.step(x, prof.loc[s].to_numpy(dtype=float), where="mid", lw=1.1, label=str(s))
    for s in ref.index:
        ax2.step(x, ref.loc[s].to_numpy(dtype=float), where="mid", lw=1.0, ls="--",
                 color="k", alpha=0.7, label=str(s))
    ax2.set_xticks(x); ax2.set_xticklabels(arm_cols, rotation=90, fontsize=6)
    ax2.set_xlim(-0.5, len(arm_cols) - 0.5); ax2.set_ylabel(ylab, fontsize=7)
    ax2.legend(fontsize=5, ncol=2, frameon=False, loc="best")
    if save is not None:
        fig.savefig(str(save), bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_donor_cnv_heatmap(adata, donor, arm_df, arm_cols, mal_mask, *,
                           subclone_col="cnv_subclone", save=None):
    """Per-cell arm-CNV heatmap for one donor's malignant cells, ordered/split by `subclone_col`."""
    import matplotlib.pyplot as plt
    cells = adata.obs_names[mal_mask & (adata.obs["donor"].astype(str).values == donor)]
    Ad = arm_df.reindex(cells)[arm_cols].fillna(0.0)
    lab = adata.obs.loc[cells, subclone_col].astype(str).values
    order = np.argsort(lab, kind="stable")
    vlim = float(np.nanpercentile(np.abs(Ad.values), 98)) or 0.05
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(Ad.values[order], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="none")
    sorted_lab = lab[order]
    for b in np.where(sorted_lab[:-1] != sorted_lab[1:])[0] + 1:
        ax.axhline(b - 0.5, color="k", lw=0.6)
    ax.set_xticks(range(len(arm_cols))); ax.set_xticklabels(arm_cols, rotation=90, fontsize=6)
    ax.set_yticks([]); ax.set_ylabel(f"{len(cells)} malignant cells")
    ax.set_title(f"{donor}: CNV subclones within one TCR clone")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_program_heatmap(obs, subclone_col, prog_cols, *, donor_col="donor",
                         donor_order=None, save=None):
    """Centered per-subclone functional-program score heatmap (subclones x programs).

    Subclones are grouped by donor (`donor_order` or all donors sorted); values are mean program
    scores centered per program (column mean subtracted) so within-cohort differences pop.
    """
    import matplotlib.pyplot as plt
    prog_cols = list(prog_cols)
    df = obs[[donor_col, subclone_col, *prog_cols]].copy()
    df[donor_col] = df[donor_col].astype(str); df[subclone_col] = df[subclone_col].astype(str)
    df = df[df[subclone_col] != ""]
    psub = df.groupby(subclone_col)[prog_cols].mean()
    donors = list(donor_order) if donor_order is not None else sorted(df[donor_col].unique())
    order = [s for d in donors for s in sorted(df.loc[df[donor_col] == d, subclone_col].unique())
             if s in psub.index]
    psub = psub.loc[order] if order else psub
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(prog_cols)), max(3, 0.16 * len(psub))))
    centered = psub.values - psub.values.mean(0)
    v = float(np.nanpercentile(np.abs(centered), 98)) or 0.1
    im = ax.imshow(centered, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(prog_cols)))
    ax.set_xticklabels([c.replace("prog_", "") for c in prog_cols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(psub))); ax.set_yticklabels(psub.index, fontsize=4)
    ax.set_title("Program scores per subclone (centered)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_subclone_prevalence(summary, *, k_max, n_multi, save=None):
    """Histogram of #CNV subclones per donor (prevalence of >= 2 vs the paper's 84%)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(summary["k"], bins=np.arange(0.5, k_max + 1.5), rwidth=0.8, color="#4c72b0")
    ax.set_xlabel("# CNV subclones per donor"); ax.set_ylabel("donors")
    ax.set_title(f">=2 subclones in {n_multi}/{len(summary)} = {n_multi/len(summary):.0%} donors (paper 84%)")
    ax.set_xticks(range(1, k_max + 1))
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), bbox_inches="tight")
    plt.show()
    plt.close(fig)
