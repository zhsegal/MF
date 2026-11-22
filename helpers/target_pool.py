"""Pooled malignant-vs-reactive statistics for the blood surface-target screen (nb46).

43-45 compare the arms inside each donor, which spends the cohort on the 52 donors that carry
both. Pooling compares all malignant cells against all reactive cells, so donors with one arm
count, and any clinical stratum can be cut out of the pool and compared with any other. Two
choices decide whether that is honest:

  WEIGHT     A pool is a weighted sum over donors. `weight="cells"` counts every cell once,
             which is what "all cells" literally means -- and in blood it hands half the
             reactive pool to three harro2023 early-MF donors. `weight="donor"` gives every
             donor the same total weight inside an arm (the pooled mean is the mean of donor
             means). nb46's headline is donor-balanced; the cell pool is printed beside it.

  INFERENCE  The donor is the unit. CIs come from a cluster bootstrap over donors (a donor brings
             all of its arms), between-stratum p-values from permuting donor labels -- freely,
             and within study, which is the one to quote because disease and treatment are
             confounded with study. No p-value is computed over cells
             (`degradome_data.CAVEATS["cell_level_pvalues"]`).

Pooling is exact. The long table carries each group's n, mean, SD and detection rate, so the
pooled mean and detection are weighted averages and the pooled variance follows from the law of
total variance. The AUROC needs the distribution, which `jobs/run_target_pool_hist.py` writes as
a per-group histogram (bin 0 = exact zero, then 0.05 log1p-CP10K bins); a pooled histogram is the
weighted sum of per-donor ones, so the AUROC re-pools under any weight, stratum or bootstrap draw.

    P = target_pool
    mal = P.panel(long, "CD4_malignant", hist=hist)
    rea = P.panel(long, "CD4_reactive", hist=hist)
    head = P.contrast(mal, rea, weight="donor", n_boot=1000)
    b = P.between(mal, labels, "MF", "SS", block=study)     # tumour cells, MF vs SS
    d = P.did(mal, rea, labels, "MF", "SS", block=study)    # is the tumour effect different
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import lgamma, log

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

PSEUDO = 0.05           # the pseudocount target_screen / target_het use for log2 ratios
TEST_MIN, DESCRIBE_MIN = 5, 3


# ------------------------------------------------------------------ the per-arm panel
@dataclass
class Panel:
    """One arm as donor-major arrays: everything a pool needs, nothing per cell."""
    arm: str
    genes: pd.Index
    donors: pd.Index
    M: np.ndarray            # donors x genes, mean log1p CP10K
    P: np.ndarray            # donors x genes, detection rate
    E2: np.ndarray           # donors x genes, mean of squares (from sd_all, ddof=1)
    n: np.ndarray            # donors, cells
    F: np.ndarray | None     # donors x genes x bins, per-donor normalised histogram

    def sub(self, donors) -> "Panel":
        keep = self.donors.isin(set(donors))
        return replace(self, donors=self.donors[keep], M=self.M[keep], P=self.P[keep],
                       E2=self.E2[keep], n=self.n[keep],
                       F=None if self.F is None else self.F[keep])


def panel(long, arm, hist=None, donors=None, genes=None) -> Panel:
    d = long[long["arm"].eq(arm)]
    if donors is not None:
        d = d[d["donor"].isin(set(donors))]
    if genes is not None:
        d = d[d["gene"].isin(set(genes))]
    piv = d.pivot_table(index="donor", columns="gene", values=["mean_all", "pct_expr", "sd_all"],
                        observed=True)
    dn, gn = piv.index, piv["mean_all"].columns
    n = d.groupby("donor", observed=True)["n_cells"].first().reindex(dn).to_numpy(float)
    M = piv["mean_all"][gn].to_numpy(float)
    S = piv["sd_all"][gn].to_numpy(float)
    E2 = S ** 2 * ((n[:, None] - 1) / n[:, None]) + M ** 2
    F = None
    if hist is not None:
        h = hist[hist["arm"].astype(str).eq(arm)]
        K = int(hist["bin"].max()) + 1
        F = np.zeros((len(dn), len(gn), K), dtype=np.float32)
        di = dn.get_indexer(h["donor"].astype(str))
        gi = gn.get_indexer(h["gene"].astype(str))
        ok = (di >= 0) & (gi >= 0)
        F[di[ok], gi[ok], h["bin"].to_numpy()[ok]] = h["count"].to_numpy()[ok]
        F /= n[:, None, None].astype(np.float32)
    return Panel(arm, gn, dn, M, piv["pct_expr"][gn].to_numpy(float), E2, n, F)


# ------------------------------------------------------------------ pooling
def _base(pan, weight):
    if weight == "donor":
        return np.ones(len(pan.donors))
    if weight == "cells":
        return pan.n.astype(float)
    raise ValueError(weight)


def _pool(pan, W):
    """Pooled mean / detection / variance for each row of donor weights `W` (draws x donors)."""
    s = W.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (W @ pan.M) / s
        pct = (W @ pan.P) / s
        var = (W @ pan.E2) / s - mean ** 2
    return mean, pct, np.clip(var, 0, None)


def _pool_hist(pan, W):
    s = W.sum(1)[:, None, None]
    D, G, K = pan.F.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W.astype(np.float32) @ pan.F.reshape(D, G * K)).reshape(len(W), G, K) / s


def _auroc(fa, fb):
    """P(X_a > X_b) + 0.5 P(tie), from two pooled histograms on the same bins."""
    cb = np.cumsum(fb, axis=-1) - fb
    return (fa * (cb + 0.5 * fb)).sum(-1)


def _lfc(ma, mb):
    return np.log2((ma + PSEUDO) / (mb + PSEUDO))


def _cohen(ma, mb, va, vb):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ma - mb) / np.sqrt((va + vb) / 2)


def pooled(pan, weight="donor") -> pd.DataFrame:
    m, p, v = _pool(pan, _base(pan, weight)[None, :])
    return pd.DataFrame({"gene": pan.genes, "mean": m[0], "pct": p[0], "sd": np.sqrt(v[0])})


# ------------------------------------------------------------------ one contrast, with CIs
def _boot_counts(strata, n_boot, rng):
    """Cluster bootstrap multiplicities (draws x donors), resampled inside each stratum."""
    C = np.zeros((n_boot, len(strata)), dtype=np.float64)
    for s in pd.unique(strata):
        idx = np.flatnonzero(strata == s)
        C[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return C


def contrast(A: Panel, B: Panel, weight="donor", n_boot=1000, seed=0, auroc=True, chunk=50,
             ci=(2.5, 97.5)) -> pd.DataFrame:
    """Pooled A vs B, per gene: log2FC, detection difference, Cohen's d and AUROC, with
    donor-bootstrap CIs.

    Donors common to A and B (paired donors, when A and B are two arms) are drawn once and bring
    both arms, so the pairing is kept. When A and B share no donor -- two strata -- each side is
    resampled on its own and keeps its size.
    """
    assert A.genes.equals(B.genes)
    U = A.donors.union(B.donors)
    shared = A.donors.intersection(B.donors)
    strata = (np.zeros(len(U), int) if len(shared)
              else np.asarray(U.isin(A.donors), int))
    ia, ib = U.get_indexer(A.donors), U.get_indexer(B.donors)
    wa, wb = _base(A, weight), _base(B, weight)

    ma, pa, va = _pool(A, wa[None, :])
    mb, pb, vb = _pool(B, wb[None, :])
    out = pd.DataFrame({
        "gene": A.genes,
        "mean_a": ma[0], "mean_b": mb[0], "pct_a": pa[0], "pct_b": pb[0],
        "log2fc": _lfc(ma, mb)[0], "d_det": (pa - pb)[0], "cohen_d": _cohen(ma, mb, va, vb)[0],
        "n_donors_a": len(A.donors), "n_donors_b": len(B.donors),
        "n_cells_a": int(A.n.sum()), "n_cells_b": int(B.n.sum()),
    })
    do_auc = auroc and A.F is not None and B.F is not None
    if do_auc:
        out["auroc"] = _auroc(_pool_hist(A, wa[None, :]), _pool_hist(B, wb[None, :]))[0]
    if not n_boot:
        return out

    C = _boot_counts(strata, n_boot, np.random.default_rng(seed))
    lfc, ddet, auc = [], [], []
    for s in range(0, n_boot, chunk):
        Wa, Wb = C[s:s + chunk, ia] * wa, C[s:s + chunk, ib] * wb
        mA, pA, _ = _pool(A, Wa)
        mB, pB, _ = _pool(B, Wb)
        lfc.append(_lfc(mA, mB))
        ddet.append(pA - pB)
        if do_auc:
            auc.append(_auroc(_pool_hist(A, Wa), _pool_hist(B, Wb)))
    for name, arr in (("log2fc", lfc), ("d_det", ddet), ("auroc", auc)):
        if arr:
            lo, hi = np.nanpercentile(np.vstack(arr), ci, axis=0)
            out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    # share of draws on the other side of the null -- a CI read, not a test
    null = {"log2fc": 0.0, "auroc": 0.5}
    for name, arr in (("log2fc", lfc), ("auroc", auc)):
        if arr:
            X = np.vstack(arr) - null[name]
            obs = np.sign(out[name].to_numpy() - null[name])
            out[f"{name}_boot_flip"] = np.nanmean(np.sign(X) != obs, axis=0)
    return out


# ------------------------------------------------------------------ strata: permutation
def n_configs_log10(labels, block=None) -> float:
    """log10 of the number of distinct label arrangements the permutation can reach.

    Within-study permutation of a contrast that is almost a study split has very few: a p-value
    from it has a floor, and this number says how high.
    """
    lab = pd.Series(np.asarray(labels))
    blk = pd.Series(np.zeros(len(lab), int) if block is None else np.asarray(block))
    tot = 0.0
    for _, l in lab.groupby(blk.to_numpy()):
        c = l.value_counts().to_numpy()
        tot += lgamma(c.sum() + 1) - sum(lgamma(k + 1) for k in c)
    return tot / log(10)


def _perm_labels(labels, block, n_perm, rng):
    lab = np.asarray(labels)
    out = np.tile(lab, (n_perm, 1))
    if block is None:
        return rng.permuted(out, axis=1)
    blk = np.asarray(block)
    for b in pd.unique(blk):
        idx = np.flatnonzero(blk == b)
        if len(idx) > 1:
            out[:, idx] = rng.permuted(out[:, idx], axis=1)
    return out


def _labels(labels, x, y) -> pd.Series:
    lab = pd.Series(labels).dropna().astype(str)
    return lab[lab.isin([x, y])]


def _side_pool(pan, L, U, x, weight):
    """Pooled means for the donors of `pan` whose label (rows of L, over U) equals x."""
    pos = U.get_indexer(pan.donors)
    keep = pos >= 0
    W = np.zeros((len(L), len(pan.donors)))
    W[:, keep] = (L[:, pos[keep]] == x) * _base(pan, weight)[keep]
    return _pool(pan, W)


def _perm_columns(out, stat_col, lab, blk, n_perm, seed, stat, chunk=5000):
    """p_free / p_study for `stat_col`, from `stat(L)` on permuted label rows.

    Two-sided, (1 + #|T*| >= |T|) / (1 + #valid draws); a draw that leaves a side empty is
    undefined and counts neither way. Drawn in chunks, so 50,000 permutations cost the memory
    of 5,000 -- and 50,000 is what BH over ~1,000 genes needs: the smallest attainable p is
    1/(n+1), and a single gene clears q < 0.05 only below 0.05/1,098 = 4.6e-5.
    """
    rng = np.random.default_rng(seed)
    obs = out[stat_col].to_numpy()
    for tag, b in (("free", None), ("study", blk)):
        if tag == "study" and b is None:
            continue
        lc = n_configs_log10(lab, b)
        out[f"log10_configs_{tag}"] = lc
        if lc < np.log10(20):
            out[f"p_{tag}"] = np.nan            # fewer than 20 arrangements: no usable p
            continue
        hit = np.zeros(len(obs))
        valid = np.zeros(len(obs))
        for s in range(0, n_perm, chunk):
            null = stat(_perm_labels(lab, b, min(chunk, n_perm - s), rng))
            ok = np.isfinite(null)
            hit += ((np.abs(null) >= np.abs(obs)[None, :] - 1e-12) & ok).sum(0)
            valid += ok.sum(0)
        p = (1 + hit) / (1 + valid)
        p[~np.isfinite(obs)] = np.nan
        out[f"p_{tag}"] = p
    out["n_perm"] = n_perm
    return _bh(out)


def between(pan: Panel, labels, x, y, block=None, n_perm=50000, weight="donor", seed=0):
    """One arm, stratum x vs stratum y: pooled log2FC, detection difference and AUROC, with
    permutation p-values over donors -- free, and within `block` (donor -> study).

    `labels` maps donor -> stratum. Donors labelled anything else, or unlabelled, are left out.
    """
    lab = _labels(labels, x, y)
    pan = pan.sub(lab.index)
    U = pan.donors
    lab = lab.reindex(U).to_numpy()
    blk = None if block is None else pd.Series(block).reindex(U).fillna("NA").to_numpy()
    nx, ny = int((lab == x).sum()), int((lab == y).sum())
    out = pd.DataFrame({"gene": pan.genes, "n_x": nx, "n_y": ny})
    if min(nx, ny) == 0:
        return out
    mx, px, vx = _side_pool(pan, lab[None, :], U, x, weight)
    my, py, vy = _side_pool(pan, lab[None, :], U, y, weight)
    out["mean_x"], out["mean_y"] = mx[0], my[0]
    out["pct_x"], out["pct_y"] = px[0], py[0]
    out["log2fc"] = _lfc(mx, my)[0]
    out["d_det"] = (px - py)[0]
    out["cohen_d"] = _cohen(mx, my, vx, vy)[0]
    if pan.F is not None:
        wx = (lab == x) * _base(pan, weight)
        wy = (lab == y) * _base(pan, weight)
        out["auroc"] = _auroc(_pool_hist(pan, wx[None, :]), _pool_hist(pan, wy[None, :]))[0]
    if min(nx, ny) < DESCRIBE_MIN or not n_perm:
        return out

    def stat(L):
        mX, _, _ = _side_pool(pan, L, U, x, weight)
        mY, _, _ = _side_pool(pan, L, U, y, weight)
        return _lfc(mX, mY)

    return _perm_columns(out, "log2fc", lab, blk, n_perm, seed, stat)


def did(mal: Panel, rea: Panel, labels, x, y, block=None, n_perm=50000, weight="donor", seed=0):
    """Difference in differences: is the pooled malignant-vs-reactive log2FC different in
    stratum x than in stratum y. Labels are permuted over donors, which carry both arms.

    Read beside `between` on each arm: a treatment that moves every T cell moves the malignant
    arm too, and only the difference says the tumour responds on its own.
    """
    lab = _labels(labels, x, y)
    mal, rea = mal.sub(lab.index), rea.sub(lab.index)
    U = mal.donors.union(rea.donors)
    lab = lab.reindex(U).to_numpy()
    blk = None if block is None else pd.Series(block).reindex(U).fillna("NA").to_numpy()
    n = {f"n_{a}_{t}": int(np.isin(p.donors, U[lab == s]).sum())
         for a, p in (("mal", mal), ("rea", rea)) for t, s in (("x", x), ("y", y))}
    out = pd.DataFrame({"gene": mal.genes, **n})

    def stat(L, both=False):
        mmx, _, _ = _side_pool(mal, L, U, x, weight)
        mrx, _, _ = _side_pool(rea, L, U, x, weight)
        mmy, _, _ = _side_pool(mal, L, U, y, weight)
        mry, _, _ = _side_pool(rea, L, U, y, weight)
        lx, ly = _lfc(mmx, mrx), _lfc(mmy, mry)
        return (lx, ly) if both else lx - ly

    lx, ly = stat(lab[None, :], both=True)
    out["log2fc_x"], out["log2fc_y"] = lx[0], ly[0]
    out["did"] = (lx - ly)[0]
    if min(n.values()) < DESCRIBE_MIN or not n_perm:
        return out
    return _perm_columns(out, "did", lab, blk, n_perm, seed, stat)


def _bh(out):
    for c in [c for c in out.columns if c.startswith("p_")]:
        q = np.full(len(out), np.nan)
        m = out[c].notna().to_numpy()
        if m.any():
            q[m] = multipletests(out.loc[m, c], method="fdr_bh")[1]
        out["q_" + c[2:]] = q
    return out


# ------------------------------------------------------------------ diagnostics
def composition(pan: Panel, study_of) -> tuple[pd.DataFrame, dict]:
    """Who the pool is made of, under each weighting, and its effective number of donors."""
    st = pd.Series(study_of).reindex(pan.donors).fillna("NA")
    d = pd.DataFrame({"donor": pan.donors, "study": st.to_numpy(), "n_cells": pan.n})
    d["share_cells"] = d["n_cells"] / d["n_cells"].sum()
    d["share_donor"] = 1 / len(d)
    by_study = d.groupby("study")[["n_cells", "share_cells", "share_donor"]].sum()
    by_study["n_donors"] = d.groupby("study").size()
    kish = d["n_cells"].sum() ** 2 / (d["n_cells"] ** 2).sum()
    top = d.sort_values("share_cells", ascending=False).iloc[0]
    summ = {"arm": pan.arm, "n_donors": len(d), "n_cells": int(d["n_cells"].sum()),
            "kish_n_eff_cells": float(kish), "kish_n_eff_donor": float(len(d)),
            "top_donor": top["donor"], "top_donor_share_cells": float(top["share_cells"]),
            "top_study_share_cells": float(by_study["share_cells"].max()),
            "top_study": by_study["share_cells"].idxmax()}
    return by_study.sort_values("share_cells", ascending=False), summ


def loso(A: Panel, B: Panel, study_of, genes=None, weight="donor") -> pd.DataFrame:
    """Pooled log2FC with each study left out in turn: genes x studies."""
    st = pd.Series(study_of)
    gi = A.genes.get_indexer(genes) if genes is not None else np.arange(len(A.genes))
    cols = {}
    for s in sorted(set(st.reindex(A.donors.union(B.donors)).dropna())):
        keep = st.index[st.ne(s)]
        a, b = A.sub(keep), B.sub(keep)
        if not len(a.donors) or not len(b.donors):
            continue
        ma, _, _ = _pool(a, _base(a, weight)[None, :])
        mb, _, _ = _pool(b, _base(b, weight)[None, :])
        cols[s] = _lfc(ma, mb)[0][gi]
    return pd.DataFrame(cols, index=A.genes[gi])


def tier(n_x, n_y) -> int:
    """1 = test, 2 = describe (effect and CI, no p), 3 = count only. 44's tiers."""
    m = min(n_x, n_y)
    return 1 if m >= TEST_MIN else 2 if m >= DESCRIBE_MIN else 3
