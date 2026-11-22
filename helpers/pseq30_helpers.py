"""Per-donor pert2state: patient-specific regulators of the MF malignant-CD4 program.

Helpers for jobs/run_pseq30.py (heavy: signatures, design, fits) and
notebooks/_archive/20_skin/30_perturbseq_per_donor.ipynb (light: statistics, figures, report).
Extends nb29 (`perturbseq_helpers`): the Perturb-seq design X is built ONCE (cis-masked z, column
standardization, TruncatedSVD D = 60) and every donor is one elastic net with FROZEN alpha / l1_ratio
on the same 60-dim gene projection and the same gene folds, so all weights share one coordinate system.
Sign convention as nb29: w_k > 0 -- knocking down k mimics the malignant program; w_k < 0 -- the program
depends on k.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "atlas_joint"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
VER = "v1"
CONDITION = "Stim48hr"

# ---- knobs ----
SEED = 0
MIN_CELLS_ARM = 600       # donor needs >= this many malignant AND reactive CD4 cells
N_PSEUDO = 3              # pseudo-replicates per arm per donor
GENE_FRAC_DONORS = 0.9    # gene testable in >= this fraction of donors
MIN_CELLS_GENE = 10       # 'testable in a donor' = non-zero in >= this many cells of BOTH arms
N_PCS, N_SPLITS, N_REPEATS = 60, 5, 3
TOP_K = 25                # per-donor top / bottom regulators for recurrence
MIN_RECUR = 15            # 'recurrent' = in the top/bottom set of >= this many donors (of 30)
K_RANGE = range(2, 6)     # donor clusters tried
FDR_SIG = 0.01            # cohort signature genes (LIC ordering, downstream rule)
FDR_KD = 0.1              # knockdown DE threshold (authors' downstream rule)
MARKERS_REACT = ("CD8A", "FOXP3", "GZMK", "NKG7")

DESIGN = dict(teal="#0F766E", dark="#1F2937", muted="#6B7280", grey="#D1D5DB", scr="#9CA3AF", red="#B91C1C")

# ---- inputs (nb29 caches + project tables) ----
CELLS_PQ = OUT / "pseq29_cells_v1.parquet"
CD4_H5 = OUT / "pseq29_cd4_cells_v1.h5ad"
CNV_SUMMARY = TAB / "skin_cnv_v5_summary.csv"
PSEQ_H5 = ROOT / "data" / "external_ref" / "perturbseq_gwcd4i" / f"GWCD4i_DE_{CONDITION}.h5ad"


def cache(name, ext):
    return OUT / f"pseq30_{name}_{VER}.{ext}"


def table(name):
    return TAB / f"pseq30_{name}_{VER}.csv"


DONOR_TAB = cache("donor_table", "csv")
GENE_SET_PRE = cache("gene_set_prelim", "txt")
GENE_SET = cache("gene_set", "txt")
DONOR_Z = cache("donor_z", "parquet")
DONOR_Z_PARTS = OUT / f"pseq30_donor_z_parts_{VER}"
COHORT_DE = cache("cohort30_de", "csv")
REACT_COMP = cache("reactive_composition", "csv")
SVD_NPZ = cache("svd", "npz")
LIC_ORDER = cache("lic_order", "txt")
HYPER = cache("hyperparams", "json")
W_SEM = cache("regulator_weights_sem", "parquet")
MODEL_FIT = table("model_fit")
WEIGHTS = table("regulator_weights")
RECURRENCE = table("recurrence")
CLUSTERS = table("donor_clusters")
PROGRAMS = table("program_tests")
DOWNSTREAM = table("downstream_genes")
REPORT = TAB / f"pseq30_full_report_{VER}.txt"


# ============================================================================ cohort
def donor_table(cells_pq: Path = CELLS_PQ, min_cells: int = MIN_CELLS_ARM) -> pd.DataFrame:
    """One row per nb29-eligible donor (CD4 clone, MF, de-duplicated): cell counts per arm, study, stage,
    nb29 split; `retained` = min(n_malignant, n_reactive) >= min_cells. Sorted by n_min, descending."""
    import target_cohort as TC
    cells = pd.read_parquet(cells_pq)
    ct = (cells.groupby(["donor", "arm"]).size().unstack(fill_value=0)
          .reindex(columns=["malignant", "reactive"], fill_value=0))
    ct.columns = ["n_malignant", "n_reactive"]
    tab = ct.join(cells.groupby("donor")[["study", "split"]].first())
    don = TC.skin_donor_table()
    tab["stage"] = don["clin_stage_class"].reindex(tab.index).astype(str).replace({"nan": "unknown", "None": "unknown"})
    tab["n_min"] = tab[["n_malignant", "n_reactive"]].min(axis=1)
    tab["malignant_frac"] = tab["n_malignant"] / (tab["n_malignant"] + tab["n_reactive"])
    tab["retained"] = tab["n_min"] >= min_cells
    tab.index.name = "donor"
    return tab.sort_values("n_min", ascending=False)


def donor_annotation(tab: pd.DataFrame, react: pd.DataFrame | None = None,
                     cnv_summary: Path = CNV_SUMMARY) -> pd.DataFrame:
    """Covariates for the clustering figure: study, stage, cell numbers, CNV event-arm count (callable
    donors only, from skin_cnv_v5_summary) and the reactive-pool composition control."""
    a = tab[tab["retained"]].copy()
    s = pd.read_csv(cnv_summary).set_index("donor")
    s = s[s["cnv_state"].astype(str).eq("callable")]
    a["n_event_arms"] = s["n_event_arms"].reindex(a.index)
    if react is not None:
        a = a.join(react)
    return a


# ============================================================================ signatures
def pseq_genes(path: Path = PSEQ_H5, basemean_min: float = 0.1) -> pd.Index:
    """Genes measured in the Perturb-seq resource that pass the authors' baseMean filter (var only)."""
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        var = read_elem(f["var"])
    cols = [c for c in var.columns if c.startswith("mean_baseMean_")]
    return var.index[(var[cols] > basemean_min).all(axis=1)].astype(str)


def nonzero_cells_per_arm(adata, layer: str = "raw_counts") -> pd.DataFrame:
    """(donor|arm) x genes: number of cells with a non-zero count."""
    import scipy.sparse as sp
    X = adata.layers[layer]
    X = X.tocsr() if sp.issparse(X) else X
    key = (adata.obs["donor"].astype(str) + "|" + adata.obs["arm"].astype(str)).to_numpy()
    rows = {}
    for k in pd.unique(key):
        sub = X[key == k]
        rows[k] = np.asarray((sub > 0).sum(axis=0)).ravel()
    return pd.DataFrame(rows, index=adata.var_names.astype(str)).T


def testable_genes(nz: pd.DataFrame, donors, frac: float = GENE_FRAC_DONORS, min_cells: int = MIN_CELLS_GENE) -> pd.Index:
    """Genes non-zero in >= min_cells cells of BOTH arms in >= frac of donors."""
    ok = pd.DataFrame({d: (nz.loc[f"{d}|malignant"] >= min_cells) & (nz.loc[f"{d}|reactive"] >= min_cells)
                       for d in donors})
    return ok.index[ok.mean(axis=1) >= frac]


def deseq_wald(counts: pd.DataFrame, meta: pd.DataFrame, design: str, contrast: list, n_cpus: int = 4,
               min_total: int = 10) -> pd.DataFrame:
    """pydeseq2 (NB GLM + Wald, BH) with an EXPLICIT `DefaultInference(n_cpus)` for both the dataset and
    the stats object. `subclone_helpers.run_pydeseq2` leaves DeseqStats without an inference object, so it
    spawns one worker per core on the host (hundreds on the login node) and stalls; this mirrors its
    gene filter (`min_total`) and returns the same results frame."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
    counts = counts.loc[:, counts.sum(axis=0) >= min_total]
    inf = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, quiet=True, inference=inf)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True, inference=inf)
    st.summary()
    return st.results_df.copy()


def pseudo_replicate_signatures(adata, donors, n: int = N_PSEUDO, seed: int = SEED, n_cpus: int = 4,
                                layer: str = "raw_counts", part_dir: Path | None = None):
    """Per donor: split each arm into `n` seeded, balanced groups, sum raw counts (2n pseudobulks),
    pydeseq2 `~ arm`, take the Wald `stat`. Returns (genes x donors frame, list of failed donors).
    The pseudo-replicate variance is sampling-only, so |z| is inflated; `prepare_y` standardizes each
    donor's vector, so only the per-gene relative weighting matters. With `part_dir`, each donor's
    stat is checkpointed to `<part_dir>/<donor>.parquet` and skipped on rerun (the queues preempt)."""
    import scipy.sparse as sp
    X = adata.layers[layer]
    X = X.tocsr() if sp.issparse(X) else X
    donor = adata.obs["donor"].astype(str).to_numpy()
    arm = adata.obs["arm"].astype(str).to_numpy()
    out, failed = {}, []
    if part_dir is not None:
        part_dir.mkdir(exist_ok=True)
    for d in donors:
        ckpt = part_dir / f"{d}.parquet" if part_dir is not None else None
        if ckpt is not None and ckpt.exists():
            out[d] = pd.read_parquet(ckpt)["stat"]
            continue
        cols, idx, md = [], [], []
        rng = np.random.default_rng([seed, list(donors).index(d)])  # per-donor stream: independent of skipped donors
        for a in ("malignant", "reactive"):
            cells = rng.permutation(np.where((donor == d) & (arm == a))[0])
            for g, part in enumerate(np.array_split(cells, n)):
                cols.append(np.asarray(X[np.sort(part)].sum(axis=0)).ravel())
                idx.append(f"{d}|{a}|{g}"); md.append(a)
        counts = pd.DataFrame(np.vstack(cols), index=idx, columns=adata.var_names.astype(str)).round().astype(int)
        meta = pd.DataFrame({"arm": md}, index=idx)
        try:
            res = deseq_wald(counts, meta, "~ arm", ["arm", "malignant", "reactive"], n_cpus=n_cpus)
            out[d] = res["stat"]
            if ckpt is not None:
                res[["stat"]].rename_axis("gene").to_parquet(ckpt)
        except Exception as e:  # noqa: BLE001 -- flagged, not silently handled
            import traceback
            failed.append((d, f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-1500:]))
    Z = pd.DataFrame(out)
    Z.index.name = "gene"
    return Z, failed


def cohort_signature(adata, donors, n_cpus: int = 4, layer: str = "raw_counts") -> pd.DataFrame:
    """nb29's cohort DE (`~ donor + arm`, one pseudobulk per donor x arm) on exactly these donors."""
    import subclone_helpers as S
    adata.obs["pb"] = adata.obs["donor"].astype(str) + "|" + adata.obs["arm"].astype(str)
    counts, _ = S.pseudobulk_counts(adata, "pb", layer=layer, min_cells=1)
    counts = counts.loc[[i for i in counts.index if i.split("|")[0] in set(donors)]]
    meta = pd.DataFrame({"donor": [s.split("|")[0] for s in counts.index],
                         "arm": [s.split("|")[1] for s in counts.index]}, index=counts.index)
    res = deseq_wald(counts, meta, "~ donor + arm", ["arm", "malignant", "reactive"], n_cpus=n_cpus)
    res.index.name = "gene"
    return res


def reactive_composition(adata, donors, markers=MARKERS_REACT, layer: str = "raw_counts") -> pd.DataFrame:
    """Control covariate: fraction of each donor's REACTIVE CD4 cells positive for CD8A, FOXP3 and
    GZMK|NKG7 (cytotoxic). If donors cluster by this, the clustering reads the reference, not the tumour."""
    import scipy.sparse as sp
    genes = [g for g in markers if g in adata.var_names]
    sub = adata[adata.obs["arm"].astype(str) == "reactive", genes]
    M = sub.layers[layer]
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    df = pd.DataFrame(M > 0, columns=genes, index=sub.obs_names)
    df["donor"] = sub.obs["donor"].astype(str).to_numpy()
    g = df.groupby("donor")
    out = pd.DataFrame({"react_CD8A_frac": g["CD8A"].mean() if "CD8A" in genes else np.nan,
                        "react_FOXP3_frac": g["FOXP3"].mean() if "FOXP3" in genes else np.nan,
                        "react_cytotoxic_frac": (df["GZMK"] | df["NKG7"]).groupby(df["donor"]).mean()
                        if {"GZMK", "NKG7"} <= set(genes) else np.nan})
    out.index.name = "donor"
    return out.reindex(donors)


# ============================================================================ design (built once)
def build_design(pseq, genes, n_pcs: int = N_PCS, seed: int = SEED, min_de_genes: int = 10) -> dict:
    """X = genes x regulators (`zscore_masked`: on-target entry masked, column-mean imputed) ->
    StandardScaler -> TruncatedSVD(n_pcs). Returns X, Z (genes x n_pcs), V (n_pcs x regulators),
    sigma, mu, sd, explained variance."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    import perturbseq_helpers as P
    regs = pseq.obs_names[P.regulator_mask(pseq, min_de_genes)]
    X = P.build_X(pseq, list(genes), regulators=regs, layer="zscore_masked")
    sc = StandardScaler().fit(X.to_numpy())
    Xs = sc.transform(X.to_numpy())
    svd = TruncatedSVD(n_components=n_pcs, random_state=seed).fit(Xs)
    return dict(X=X, Z=svd.transform(Xs), V=svd.components_, sigma=svd.singular_values_,
                mu=sc.mean_, sd=sc.scale_, explained=svd.explained_variance_ratio_,
                genes=X.index.to_numpy().astype(str), regulators=X.columns.to_numpy().astype(str))


def scrambled_Z(X: pd.DataFrame, d: dict, seed: int = SEED) -> np.ndarray:
    """One shared null design: each gene's z permuted across regulators (authors' scramble), then
    standardized with the REAL mu/sd and projected with the REAL V."""
    import perturbseq_helpers as P
    Xp = P.scramble_rows(X, seed).to_numpy()
    return ((Xp - d["mu"]) / d["sd"]) @ d["V"].T


def save_design(d: dict, path: Path | None = None, Z_scr: np.ndarray | None = None):
    path = path or SVD_NPZ
    arrs = {k: d[k] for k in ("Z", "V", "sigma", "mu", "sd", "explained", "genes", "regulators")}
    if Z_scr is not None:
        arrs["Z_scr"] = Z_scr
    np.savez(path, **arrs)


def load_design(path: Path | None = None) -> dict:
    with np.load(path or SVD_NPZ, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    d["genes"] = d["genes"].astype(str)
    d["regulators"] = d["regulators"].astype(str)
    return d


def lic_order(X: pd.DataFrame, sig_genes, method: str = "average", metric: str = "correlation") -> list:
    """The single x-axis of every LIC plot: regulators clustered on their knockdown z over the cohort
    signature genes (average linkage, correlation distance)."""
    import scipy.cluster.hierarchy as sch
    g = X.index.intersection(pd.Index(list(sig_genes)))
    M = X.loc[g].T.to_numpy()
    link = sch.linkage(M, method=method, metric=metric)
    return X.columns[sch.leaves_list(link)].tolist()


# ============================================================================ fits
def gene_folds(n_genes: int, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED) -> list:
    """The ONE fold assignment shared by every fit (cohort, donors, scrambled)."""
    from sklearn.model_selection import RepeatedKFold
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(tr, te) for tr, te in rkf.split(np.arange(n_genes))]


def freeze_hyperparams(Z: np.ndarray, y: np.ndarray, seed: int = SEED, n_jobs: int = 4) -> dict:
    """The authors' inner 4-fold ElasticNetCV (their grids, l1_ratio capped at 1 as in nb29) on the
    cohort signature over ALL genes -> one (alpha, l1_ratio) reused by every fit."""
    from sklearn.linear_model import ElasticNetCV
    import perturbseq_helpers as P
    m = ElasticNetCV(l1_ratio=P.L1_RATIO, alphas=P.ALPHAS, cv=4, tol=1e-3, max_iter=3000,
                     random_state=seed, n_jobs=n_jobs).fit(Z, y)
    return {"alpha": float(m.alpha_), "l1_ratio": float(m.l1_ratio_)}


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fit_fixed(Z: np.ndarray, y: pd.Series | np.ndarray, folds: list, V: np.ndarray, alpha: float,
              l1_ratio: float, seed: int = SEED, max_iter: int = 3000) -> dict:
    """One signature, fixed hyper-parameters, the shared gene folds. Returns held-out / train Pearson r
    (mean, SE over folds), the regulator weights w = V^T coef (mean, SEM over folds) and a convergence
    flag (any fold that hit max_iter)."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet
    import perturbseq_helpers as P
    yv = P.prepare_y(pd.Series(np.asarray(y, dtype=float))).to_numpy()
    W, tr, te, nit = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for train, test in folds:
            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-3, max_iter=max_iter, random_state=seed)
            m.fit(Z[train], yv[train])
            tr.append(_pearson(yv[train], m.predict(Z[train])))
            te.append(_pearson(yv[test], m.predict(Z[test])))
            W.append(V.T @ m.coef_)
            nit.append(int(m.n_iter_))
    W = np.asarray(W)
    k = len(folds)
    return dict(test_r=float(np.mean(te)), test_r_se=float(np.std(te, ddof=1) / np.sqrt(k)),
                train_r=float(np.mean(tr)), w_mean=W.mean(0), w_sem=W.std(0, ddof=1) / np.sqrt(k),
                converged=bool(max(nit) < max_iter), n_iter_max=int(max(nit)), n_folds=k)


# ============================================================================ statistics
def recurrence(W: pd.DataFrame, top_k: int = TOP_K, min_recur: int = MIN_RECUR) -> pd.DataFrame:
    """Per donor, the top_k and bottom_k regulators by w; per regulator, in how many donors it sits in
    the top set (n_top) or the bottom set (n_bottom). n_recur = max of the two (sign-consistent by
    construction). Binomial p against p0 = top_k / n_regulators per direction, BH across regulators."""
    n_regs, n_don = W.shape
    ranks = W.rank(axis=0, ascending=False)
    n_top = (ranks <= top_k).sum(axis=1)
    n_bottom = (ranks > n_regs - top_k).sum(axis=1)
    df = pd.DataFrame({"n_top": n_top, "n_bottom": n_bottom})
    df["n_recur"] = df[["n_top", "n_bottom"]].max(axis=1)
    df["direction"] = np.where(df["n_top"] >= df["n_bottom"], "positive", "negative")
    df["mean_w"] = W.mean(axis=1)
    df["sd_w"] = W.std(axis=1)
    df["frac_same_sign"] = (np.sign(W).eq(np.sign(df["mean_w"]), axis=0)).mean(axis=1)
    p0 = top_k / n_regs
    df["p_binom"] = [ss.binomtest(int(k), n_don, p0, alternative="greater").pvalue for k in df["n_recur"]]
    df["q_binom"] = multipletests(df["p_binom"], method="fdr_bh")[1]
    df["recurrent"] = df["n_recur"] >= min_recur
    df.index.name = "regulator"
    df.attrs["expected_chance"] = n_don * 2 * top_k / n_regs
    return df.sort_values(["n_recur", "frac_same_sign"], ascending=False)


def donor_clustering(W: pd.DataFrame, k_range=K_RANGE) -> dict:
    """Spearman correlation between donors over all weights -> distance 1 - rho -> Ward linkage;
    k chosen by mean silhouette over k_range (profile returned so the choice is visible)."""
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score
    rho = W.corr(method="spearman")
    D = 1 - rho.to_numpy()
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    link = sch.linkage(squareform(D, checks=False), method="ward")
    prof = {}
    for k in k_range:
        lab = sch.fcluster(link, k, criterion="maxclust")
        prof[k] = silhouette_score(D, lab, metric="precomputed") if len(set(lab)) > 1 else np.nan
    prof = pd.Series(prof, name="silhouette")
    prof.index.name = "k"
    best = int(prof.idxmax())
    labels = pd.Series(sch.fcluster(link, best, criterion="maxclust"), index=rho.index, name="cluster")
    return dict(rho=rho, link=link, labels=labels, silhouette=prof, k=best,
                leaves=rho.index[sch.leaves_list(link)].tolist())


def cluster_covariate_tests(labels: pd.Series, annot: pd.DataFrame) -> pd.DataFrame:
    """Does cluster membership track study / stage (Fisher for 2x2, else chi-square) or cell number,
    CNV burden, malignant fraction, reactive-pool composition (Kruskal-Wallis)?"""
    rows = []
    for col in ("study", "stage"):
        if col not in annot:
            continue
        ct = pd.crosstab(annot[col].reindex(labels.index), labels)
        if ct.shape == (2, 2):
            p, test = ss.fisher_exact(ct.to_numpy())[1], "fisher"
        else:
            p, test = ss.chi2_contingency(ct.to_numpy())[1], "chi2"
        rows.append({"covariate": col, "test": test, "p": p,
                     "detail": "; ".join(f"{i}: " + "/".join(str(v) for v in ct.loc[i]) for i in ct.index)})
    for col in ("n_min", "malignant_frac", "n_event_arms", "react_CD8A_frac", "react_FOXP3_frac", "react_cytotoxic_frac"):
        if col not in annot:
            continue
        v = annot[col].reindex(labels.index)
        groups = [v[labels == k].dropna() for k in sorted(labels.unique())]
        groups = [g for g in groups if len(g)]
        p = ss.kruskal(*groups).pvalue if len(groups) > 1 and sum(len(g) for g in groups) > len(groups) else np.nan
        rows.append({"covariate": col, "test": "kruskal", "p": p,
                     "detail": "; ".join(f"c{k}: median {g.median():.3g} (n={len(g)})"
                                         for k, g in zip(sorted(labels.unique()), groups))})
    return pd.DataFrame(rows)


def program_tests(w_by_group: dict, sets: dict, min_n: int = 5) -> pd.DataFrame:
    """nb29's signed pathway tests (one-sided Mann-Whitney, members' w > rest = 'knockdown mimics the
    state', members' w < rest = 'state depends on the set'; BH across sets per test) for each group of
    weights (consensus, cluster means)."""
    import perturbseq_helpers as P
    out = []
    for grp, w in w_by_group.items():
        t = P.pathway_tests(pd.DataFrame({"coef_mean": w}), sets, min_n=min_n)
        if not len(t):
            continue
        t.insert(0, "group", grp)
        out.append(t)
    df = pd.concat(out, ignore_index=True)
    df["direction"] = np.where(df["q_signed_greater"] <= df["q_signed_less"], "KD mimics MF (w > rest)",
                               "MF depends on set (w < rest)")
    df["q_signed"] = df[["q_signed_greater", "q_signed_less"]].min(axis=1)
    cols = ["group", "set", "n", "mean_w", "delta_signed", "direction", "q_signed", "q_signed_greater",
            "q_signed_less", "p_signed_greater", "p_signed_less", "members"]
    return df[cols].sort_values(["group", "q_signed"])


def controlled_genes(pseq, regulator: str, w_k: float, cohort_de: pd.DataFrame,
                     fdr_sig: float = FDR_SIG, fdr_kd: float = FDR_KD) -> pd.DataFrame:
    """Authors' rule: a signature gene (cohort padj < fdr_sig) is CONTROLLED by regulator k if (a) it is
    DE at fdr_kd on knockdown of k and (b) sign(z_kd) * sign(w_k) == sign(cohort log2FC). Returns every
    signature gene that satisfies (a) with a `controlled` flag for (b)."""
    o = pseq.obs["regulator"].astype(str).to_numpy()
    i = np.where(o == regulator)[0]
    if not i.size:
        raise KeyError(regulator)
    i = int(i[0])
    z = pd.Series(np.asarray(pseq.layers["zscore"][i]).ravel(), index=pseq.var_names.astype(str))
    q = pd.Series(np.asarray(pseq.layers["adj_p_value"][i]).ravel(), index=pseq.var_names.astype(str))
    sig = cohort_de[cohort_de["padj"] < fdr_sig]
    g = sig.index.intersection(q.index[q < fdr_kd])
    g = g[g != regulator]
    df = pd.DataFrame({"regulator": regulator, "w": w_k, "z_kd": z.reindex(g).to_numpy(),
                       "padj_kd": q.reindex(g).to_numpy(), "lfc_cohort": sig.loc[g, "log2FoldChange"].to_numpy(),
                       "padj_cohort": sig.loc[g, "padj"].to_numpy()}, index=pd.Index(g, name="gene"))
    df["controlled"] = np.sign(df["z_kd"]) * np.sign(w_k) == np.sign(df["lfc_cohort"])
    return df.sort_values("z_kd", key=np.abs, ascending=False)


def downstream_sets(condition: str = CONDITION, min_upstream: int = 3, min_coherence: float = 0.5) -> dict:
    """The authors' per-cluster downstream programs (coherent genes only), named like `P.load_clusters`."""
    import perturbseq_helpers as P
    df = pd.read_csv(P.DOWNSTREAM_CSV, index_col=0)
    df = df[(df["condition"] == condition) & (df["num_of_upstream"] >= min_upstream)
            & (df["sign_coherence"].abs() >= min_coherence) & (df["hdbscan_cluster"] >= 0)]  # -1 = HDBSCAN noise
    names = P.load_clusters().set_index("cluster")["name"]
    return {names.get(c, f"c{c}"): set(g) for c, g in df.groupby("hdbscan_cluster")["downstream_gene"]}


def hypergeom_enrichment(hits, universe, sets: dict, min_overlap: int = 3) -> pd.DataFrame:
    """Over-representation of `hits` within `universe` for each set (hypergeometric, BH)."""
    U = set(universe)
    H = set(hits) & U
    rows = []
    for name, s in sets.items():
        S = set(s) & U
        k = len(S & H)
        if k < min_overlap:
            continue
        p = ss.hypergeom.sf(k - 1, len(U), len(S), len(H))
        rows.append({"set": name, "n_set": len(S), "n_hits": len(H), "overlap": k,
                     "expected": len(S) * len(H) / len(U), "p": p, "genes": ",".join(sorted(S & H)[:15])})
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = multipletests(df["p"], method="fdr_bh")[1]
        df = df.sort_values("p")
    return df


# ============================================================================ figures
def set_style():
    """Design system: white, teal accent, Carlito (falls back to what is installed), 400 dpi, editable PDF text."""
    import matplotlib as mpl
    import target_figs as TF
    TF.rc()
    mpl.rcParams.update({"text.color": DESIGN["dark"], "axes.labelcolor": DESIGN["dark"],
                         "axes.edgecolor": DESIGN["muted"], "xtick.color": DESIGN["muted"],
                         "ytick.color": DESIGN["muted"], "savefig.dpi": 400, "pdf.fonttype": 42,
                         "figure.dpi": 110})


def savefig(fig, name: str, show: bool = True):
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"pseq30_{name}.{ext}", bbox_inches="tight", dpi=400)
    if show:
        plt.show()
    plt.close(fig)


def lic_panel(ax, w: pd.Series, order: list, highlight=(), labels=(), colors=None, ylim=None,
              yerr: pd.Series | None = None, s: float = 2.5, label_fs: float = 5):
    """One Long-Island-City panel: every regulator at its shared x position, grey; `highlight`
    regulators in teal (or per-regulator `colors` dict); `labels` annotated."""
    w = w.reindex(order)
    x = np.arange(len(order))
    pos = {r: i for i, r in enumerate(order)}
    if yerr is not None:
        ax.errorbar(x, w.to_numpy(), yerr=yerr.reindex(order).to_numpy(), fmt="none", ecolor=DESIGN["grey"],
                    elinewidth=0.3, alpha=0.6, rasterized=True)
    ax.scatter(x, w.to_numpy(), s=s, c=DESIGN["grey"], lw=0, rasterized=True)
    hi = [r for r in highlight if r in pos]
    if hi:
        c = [colors.get(r, DESIGN["teal"]) for r in hi] if colors else DESIGN["teal"]
        ax.scatter([pos[r] for r in hi], w.loc[hi].to_numpy(), s=s * 3, c=c, lw=0, zorder=3)
    for r in labels:
        if r in pos:
            ax.annotate(r, (pos[r], w.loc[r]), fontsize=label_fs, color=DESIGN["dark"],
                        xytext=(1, 2), textcoords="offset points")
    ax.axhline(0, lw=0.5, ls="--", color=DESIGN["muted"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_xlim(-10, len(order) + 10)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(obj: dict, path: Path):
    Path(path).write_text(json.dumps(obj, indent=1))
