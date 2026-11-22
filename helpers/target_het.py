"""Which antigens are up in *every* patient, and which are up in only some of them.

CTCL is heterogeneous, and a cohort median hides the shape of that heterogeneity. Three
different questions, three different statistics, and they do not agree with each other -- which
is the point:

  1 HOW CONSISTENT IS IT?  A random-effects meta-analysis across donors. `tau2` is the spread of
    the true per-donor effect once the sampling noise of each donor's own cells is taken out,
    `I2` is the share of the observed spread that is not sampling, and the prediction interval
    is what the next patient's effect would plausibly be. A gene with a large pooled effect and
    I2 near 1 is not a cohort-wide antigen; it is a subgroup antigen with a high average.

  2 IS IT A SUBGROUP, OR A GRADIENT?  COPA (Tomlins 2005) and the outlier sum
    (Tibshirani & Hastie) were built for exactly this: a gene activated in a fraction of tumours,
    which a t-test averages away. Both standardise by median and MAD and then score the upper
    tail, so a gene that is extreme in six donors beats a gene that is mildly up in all of them.
    The null here is a sign flip of the paired within-donor differences, which is the same
    exchangeability the Wilcoxon test assumes.

  3 IS THE SUBGROUP A REAL SPLIT?  Gaussian-mixture BIC (one component vs two) and the
    bimodality coefficient on the per-donor effects. `diptest` is not installed in
    `neural_nmf_env`, so Hartigan's dip is not available and these two stand in for it.

And the question a clinician asks instead of all three: at what detection threshold, in what
fraction of patients, would this antigen be present? That is `eligibility`, and it is the
heterogeneity result that can be read without a statistics background.

    H = target_het
    M  = H.lfc_matrix(long)                 # genes x donors
    re = H.meta_random_effects(M, se=H.lfc_se(long))
    cp = H.copa(M, n_perm=2000)
    el = H.eligibility(long, "CD4_malignant")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PSEUDO = 0.05


# ------------------------------------------------------------------ per-donor effect matrix
def lfc_matrix(long, arm_pos="CD4_malignant", arm_neg="CD4_reactive", value="mean_all",
               pseudo=PSEUDO, genes=None, min_donors=5) -> pd.DataFrame:
    """Genes x donors log2 ratio of the two arms, NaN where the donor lacks one."""
    d = long[long["arm"].isin([arm_pos, arm_neg])]
    if genes is not None:
        d = d[d["gene"].isin(set(genes))]
    w = d.pivot_table(index=["gene", "donor"], columns="arm", values=value)
    if arm_pos not in w.columns or arm_neg not in w.columns:
        return pd.DataFrame()
    lfc = np.log2((w[arm_pos] + pseudo) / (w[arm_neg] + pseudo))
    M = lfc.unstack("donor")
    # Donors that carry only one arm survive the unstack as an all-NaN column and would show up
    # as an empty half of every heatmap.
    M = M.loc[:, M.notna().any()]
    return M[M.notna().sum(1) >= min_donors]


def lfc_se(long, arm_pos="CD4_malignant", arm_neg="CD4_reactive", value="mean_all",
           sd_col="sd_all", pseudo=PSEUDO, genes=None) -> pd.DataFrame:
    """Delta-method standard error of each donor's log2 ratio, from the per-arm cell SDs.

    Without this the meta-analysis cannot separate real between-patient spread from the noise of
    a donor who contributed 25 cells. Requires `sd_all` and `n_cells`, which the QC build emits
    and the v1 build does not -- returns an empty frame if they are missing, and the caller then
    falls back to the unweighted variance decomposition.
    """
    if sd_col not in long.columns:
        return pd.DataFrame()
    d = long[long["arm"].isin([arm_pos, arm_neg])]
    if genes is not None:
        d = d[d["gene"].isin(set(genes))]
    m = d.pivot_table(index=["gene", "donor"], columns="arm", values=value)
    s = d.pivot_table(index=["gene", "donor"], columns="arm", values=sd_col)
    n = d.pivot_table(index=["gene", "donor"], columns="arm", values="n_cells")
    var = 0.0
    for arm in (arm_pos, arm_neg):
        sem2 = (s[arm] ** 2) / n[arm].clip(lower=1)
        var = var + sem2 / (m[arm] + pseudo) ** 2
    return np.sqrt(var / np.log(2) ** 2).unstack("donor")


# ------------------------------------------------------------------ 1. random effects
def _dl(y, v):
    """DerSimonian-Laird random-effects pool of effects `y` with sampling variances `v`."""
    k = len(y)
    wf = 1.0 / v
    mu_f = float((wf * y).sum() / wf.sum())
    Q = float((wf * (y - mu_f) ** 2).sum())
    c = float(wf.sum() - (wf ** 2).sum() / wf.sum())
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    w = 1.0 / (v + tau2)
    mu = float((w * y).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return mu, se, tau2, I2, Q


def meta_random_effects(M: pd.DataFrame, se: pd.DataFrame | None = None,
                        min_donors=5) -> pd.DataFrame:
    """Per gene: pooled effect, tau2, I2, Q and the 95% prediction interval.

    With `se` the sampling term is the donor's own measurement error. Without it, every donor is
    given the same nominal variance, `tau2` then absorbs the sampling noise as well, and `I2` is
    an upper bound -- which the `se_source` column records so a figure cannot quietly overstate
    heterogeneity.
    """
    rows = []
    for gene, y in M.iterrows():
        y = y.dropna()
        if len(y) < min_donors:
            continue
        if se is not None and len(se) and gene in se.index:
            v = (se.loc[gene].reindex(y.index) ** 2).to_numpy(float)
            src = "delta-method from per-arm cell SD"
        else:
            v = np.full(len(y), np.nan)
            src = "none -- tau2 absorbs sampling noise, I2 is an upper bound"
        if not np.isfinite(v).all() or (v <= 0).any():
            v = np.full(len(y), float(np.var(y, ddof=1)) / max(len(y), 2) or 1e-6)
            src = "none -- tau2 absorbs sampling noise, I2 is an upper bound"
        mu, sem, tau2, I2, Q = _dl(y.to_numpy(float), v)
        tcrit = stats.t.ppf(0.975, max(len(y) - 2, 1))
        pi = tcrit * np.sqrt(tau2 + sem ** 2)
        rows.append({"gene": gene, "n_donors": int(len(y)), "mu": mu, "se": sem,
                     "tau2": tau2, "tau": float(np.sqrt(tau2)), "I2": I2, "Q": Q,
                     # I2 saturates when each donor contributes thousands of cells: the
                     # sampling term is then ~0 and almost all spread is "real", so I2 sits
                     # near 1 for everything and stops discriminating. tau is the absolute
                     # between-patient SD of the effect and keeps its meaning; tau/|mu| is the
                     # scale-free version.
                     "tau_over_mu": float(np.sqrt(tau2) / abs(mu)) if mu != 0 else np.inf,
                     "p_Q": float(stats.chi2.sf(Q, len(y) - 1)),
                     "pi_lo": mu - pi, "pi_hi": mu + pi,
                     "frac_donors_up": float((y > 0).mean()), "se_source": src})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_Q"] = multipletests(out["p_Q"], method="fdr_bh")[1]
    return out.sort_values("I2", ascending=False)


# ------------------------------------------------------------------ 2. subgroup statistics
def _standardise(y):
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826
    return (y - med) / mad if mad > 0 else np.zeros_like(y)


def _standardise_rows(Y):
    """`_standardise` applied down each row of a 2-D array, without a Python loop.

    The permutation null needs this for thousands of resamples per gene; the per-row version
    costs two medians per resample and dominates the runtime.
    """
    med = np.median(Y, axis=1, keepdims=True)
    mad = np.median(np.abs(Y - med), axis=1, keepdims=True) * 1.4826
    return np.divide(Y - med, mad, out=np.zeros_like(Y), where=mad > 0)


def _copa_score(y, pct):
    return float(np.percentile(_standardise(y), pct))


def _os_score(y):
    z = _standardise(y)
    q75, q25 = np.percentile(z, [75, 25])
    thr = q75 + (q75 - q25)
    hit = z[z > thr]
    return float(hit.sum()), int(len(hit))


def _os_scores_rows(Z):
    q75, q25 = np.percentile(Z, [75, 25], axis=1)
    thr = (q75 + (q75 - q25))[:, None]
    return np.where(Z > thr, Z, 0.0).sum(1)


def copa(M: pd.DataFrame, pct=75, n_perm=0, seed=0, min_donors=5) -> pd.DataFrame:
    """COPA and the outlier sum per gene, with a sign-flip permutation null.

    The null flips the sign of each donor's effect independently, which is the exchangeability a
    paired test already assumes -- so a gene that scores high here scores high *beyond* what its
    own overall shift explains.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gene, y in M.iterrows():
        y = y.dropna().to_numpy(float)
        if len(y) < min_donors:
            continue
        c = _copa_score(y, pct)
        osum, n_out = _os_score(y)
        rec = {"gene": gene, "n_donors": len(y), f"copa_p{pct}": c,
               "outlier_sum": osum, "n_outlier_donors": n_out,
               "frac_outlier": n_out / len(y)}
        if n_perm:
            Z = _standardise_rows(rng.choice([-1.0, 1.0], size=(n_perm, len(y))) * y)
            null_c = np.percentile(Z, pct, axis=1)
            null_o = _os_scores_rows(Z)
            rec["p_copa"] = float((np.sum(null_c >= c) + 1) / (n_perm + 1))
            rec["p_outlier_sum"] = float((np.sum(null_o >= osum) + 1) / (n_perm + 1))
        rows.append(rec)
    out = pd.DataFrame(rows)
    for c in ("p_copa", "p_outlier_sum"):
        if c in out:
            out["q" + c[1:]] = multipletests(out[c], method="fdr_bh")[1]
    return out.sort_values(f"copa_p{pct}", ascending=False) if not out.empty else out


# ------------------------------------------------------------------ 3. is it a real split
def bimodality(M: pd.DataFrame, min_donors=8, seed=0) -> pd.DataFrame:
    """One- vs two-component Gaussian mixture BIC, plus the bimodality coefficient.

    `delta_bic > 0` favours two components. `bc > 5/9` is the classical bimodality-coefficient
    threshold. Neither is a test; with 40-odd donors both are suggestive at best, which is why
    they are reported beside `I2` rather than instead of it.
    """
    from sklearn.mixture import GaussianMixture
    rows = []
    for gene, y in M.iterrows():
        y = y.dropna().to_numpy(float)
        if len(y) < min_donors or np.ptp(y) == 0:
            continue
        X = y.reshape(-1, 1)
        bic = {}
        for k in (1, 2):
            gm = GaussianMixture(k, covariance_type="full", random_state=seed,
                                 n_init=3).fit(X)
            bic[k] = gm.bic(X)
            if k == 2:
                mu = np.sort(gm.means_.ravel())
                w = gm.weights_[np.argsort(gm.means_.ravel())]
        g1 = float(stats.skew(y, bias=False))
        g2 = float(stats.kurtosis(y, fisher=True, bias=False))
        n = len(y)
        bc = (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        rows.append({"gene": gene, "n_donors": n, "delta_bic": float(bic[1] - bic[2]),
                     "two_component": bool(bic[2] < bic[1]),
                     "mode_lo": float(mu[0]), "mode_hi": float(mu[1]),
                     "w_hi": float(w[1]), "bimodality_coef": float(bc)})
    out = pd.DataFrame(rows)
    return out.sort_values("delta_bic", ascending=False) if not out.empty else out


def donor_clusters(M: pd.DataFrame, k=None, max_k=6, seed=0, min_size=3):
    """Cluster donors on their effect profile across genes. Returns (labels, linkage, order).

    Correlation distance and average linkage, with k chosen by silhouette when not given. The
    point is not the dendrogram but the test afterwards: do these clusters line up with
    diagnosis, treatment, burden or study.
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    X = M.dropna(axis=1, how="all").fillna(M.mean(axis=1).median())
    A = X.to_numpy(float).T                       # donors x genes
    C = np.corrcoef(A)
    C = np.nan_to_num(C, nan=0.0)
    D = np.clip(1 - C, 0, 2)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    if k is None:
        # Silhouette alone happily returns a 51-vs-1 split: one outlier donor is trivially
        # well separated. Only cuts where every cluster has `min_size` members are considered,
        # and if none qualifies the function says so by returning a single cluster.
        best, k = -np.inf, None
        for kk in range(2, min(max_k, len(A) - 1) + 1):
            lab = fcluster(Z, kk, criterion="maxclust")
            sizes = np.bincount(lab)[1:]
            if len(set(lab)) < 2 or sizes.min() < min_size:
                continue
            s = silhouette_score(D, lab, metric="precomputed")
            if s > best:
                best, k = s, kk
        if k is None:
            return (pd.Series(1, index=X.columns, name="donor_cluster"), Z,
                    list(X.columns[leaves_list(Z)]))
    labels = pd.Series(fcluster(Z, k, criterion="maxclust"), index=X.columns, name="donor_cluster")
    return labels, Z, list(X.columns[leaves_list(Z)])


# ------------------------------------------------------------------ the clinical form
def eligibility(long, arm="CD4_malignant", thresholds=(0.10, 0.25, 0.50, 0.75),
                genes=None, donors=None) -> pd.DataFrame:
    """Fraction of donors whose tumour has at least X% of its cells positive for the gene.

    One column per threshold. This is the curve a biomarker-selected trial is designed against:
    "antigen-positive in 60% of patients at a 25% tumour-cell cut" is a recruitment statement,
    where a log2FC is not.
    """
    a = long[long["arm"].eq(arm)]
    if genes is not None:
        a = a[a["gene"].isin(set(genes))]
    if donors is not None:
        a = a[a["donor"].isin(set(donors))]
    g = a.groupby("gene")["pct_expr"]
    out = pd.DataFrame({"n_donors": g.size(), "median_pct_expr": g.median()})
    for t in thresholds:
        out[f"frac_donors_ge{int(t * 100)}"] = (
            a.assign(_h=a["pct_expr"] >= t).groupby("gene")["_h"].mean())
    return out.reset_index().sort_values(f"frac_donors_ge{int(thresholds[0] * 100)}",
                                         ascending=False)


def subgroup_table(M: pd.DataFrame, meta: pd.DataFrame, fields, min_donors=5,
                   thr=0.5) -> pd.DataFrame:
    """For each gene, the donors where it is up (lfc > thr) described by their clinical fields.

    Answers "who are the patients this marker works in" directly, which the summary statistics
    above only imply.
    """
    rows = []
    for gene, y in M.iterrows():
        y = y.dropna()
        if len(y) < min_donors:
            continue
        hi = y.index[y > thr]
        sub = meta.reindex(hi)
        rec = {"gene": gene, "n_donors": len(y), "n_up": len(hi),
               "frac_up": len(hi) / len(y)}
        for f in fields:
            if f not in meta.columns:
                continue
            v = sub[f].dropna()
            if v.empty:
                rec[f] = ""
            elif pd.api.types.is_numeric_dtype(v):
                rec[f] = f"{v.median():.3g}"
            else:
                rec[f] = ", ".join(f"{k}:{n}" for k, n in v.value_counts().head(3).items())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("frac_up", ascending=False)
