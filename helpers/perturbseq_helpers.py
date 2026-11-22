"""Perturb-seq (GWCD4i) reverse-engineering of the MF malignant-CD4 program -- helpers for
notebooks/_archive/20_skin/29_perturbseq_reverse_engineering.ipynb.

The model is the authors' (Zhu, Dann et al., Cell 2026, Methods "Cell state signature analysis"),
vendored verbatim in `pert2state_model.py`; everything here is the glue that reproduces their
driver notebooks (`src/7_1k1k_analysis/1k1k_pert2state_model.ipynb`,
`src/4_polarization_signatures/pert2state_polarization.ipynb`) and the project-specific extension
to the inferCNV arm calls. Functions copied from the authors' drivers are marked `# authors'`.

Sign convention (authors' Fig 4/5): w_k > 0 -- knocking down k pushes the transcriptome toward the
target state (loss-of-function-like); w_k < 0 -- k's activity is needed for the target state.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
PSEQ_DIR = ROOT / "data" / "external_ref" / "perturbseq_gwcd4i"
GTF = ROOT / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
ARCE_CSV = PSEQ_DIR / "Arce2024_20230130_DESeq2_output_AAVS1_Teff_Stimulation_vs_Resting.csv"
CLUSTERS_CSV = PSEQ_DIR / "suppl_tables" / "clustering_results_and_annotations.csv"
DOWNSTREAM_CSV = PSEQ_DIR / "suppl_tables" / "clustering_downstream_genes.csv.gz"
CNV_SUMMARY = ROOT / "tables" / "skin_cnv_v5_summary.csv"

# The authors' hyper-parameters (Methods + driver notebooks); the l1_ratio grid is the one change:
# their default logspace(-2, 2) exceeds 1 and scikit-learn >= 1.2 rejects it.
N_PCS = 60
ALPHAS = np.logspace(-3, 3, 30)
L1_RATIO = np.logspace(-2, 0, 10)

# CTCL literature control panel -- the analogue of the authors' `known_regulators` yaml. Expected
# sign of w under "arm loss / LoF mutation <-> knockdown" (w>0) and "gain / GoF / dependency" (w<0).
CONTROL_PANEL = {
    "LoF in CTCL (expect w>0)": ["PTEN", "TNFAIP3", "ZEB1", "CDKN2A", "TP53", "ARID1A", "DNMT3A",
                                 "TET2", "KMT2D", "CD58", "FAS", "NFKBIA", "RB1", "CYLD", "PRDM1"],
    "GoF / dependency (expect w<0)": ["PLCG1", "CARD11", "STAT3", "STAT5B", "JAK3", "JAK1", "CD28",
                                      "NFKB2", "MYC", "TOX", "GATA3", "IRF4", "BATF", "TNFRSF1B"],
}
TCR_CASCADE = ["CD3D", "CD3E", "CD3G", "CD247", "LCK", "ZAP70", "LAT", "LCP2", "ITK", "PLCG1",
               "VAV1", "PRKCQ", "CARD11", "BCL10", "MALT1", "PPP3R1", "PPP3CA", "NFATC1", "NFATC2",
               "RASGRP1", "MAP3K7", "IKBKB", "NFKB1", "RELA", "CD28", "ICOS", "PTPRC", "CD2"]
MF_MARKERS = {"down": ["CD7", "CD2", "CD5", "CD26", "DPP4", "IL7R", "KLF2", "SELL"],
              "up": ["TOX", "GATA3", "PLS3", "KIR3DL2", "TWIST1", "TFAP2A", "CCR4", "TNFRSF8", "DNM3"]}


# ----------------------------------------------------------------------------- Perturb-seq input
def load_pseq(condition: str = "Stim48hr", path: Path | None = None,
              layers=("zscore_masked", "zscore", "adj_p_value", "log_fc")):
    """The compact per-condition h5ad written by jobs/run_perturbseq_prep.py, reading only the
    `layers` the notebook needs (each is ~0.45 GB float32)."""
    import anndata as ad
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:  # older anndata
        from anndata.experimental import read_elem
    path = Path(path) if path else PSEQ_DIR / f"GWCD4i_DE_{condition}.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run jobs/download_perturbseq_gwcd4i.sh then jobs/run_perturbseq_prep.sh")
    with h5py.File(path, "r") as f:
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        L = {k: np.asarray(f["layers"][k], dtype=np.float32) for k in layers if k in f["layers"]}
        uns = {k: read_elem(f["uns"][k]) for k in f["uns"].keys()} if "uns" in f else {}
    a = ad.AnnData(obs=obs, var=var, layers=L, uns=uns, shape=(len(obs), len(var)))
    a.obs["regulator"] = a.obs["regulator"].astype(str)
    return a


def gene_mask(pseq, basemean_min: float = 0.1) -> np.ndarray:
    """authors': keep genes with mean baseMean > 0.1 in every condition ('very low base mean
    messes up the reliability estimate')."""
    cols = [c for c in pseq.var.columns if c.startswith("mean_baseMean_")]
    return (pseq.var[cols] > basemean_min).all(axis=1).to_numpy()


def regulator_mask(pseq, min_de_genes: int = 10, drop_flagged: bool = False) -> np.ndarray:
    """authors': regulators with significant effects on >= 10 measured genes in the condition.
    `drop_flagged` additionally removes distal off-target / neighbouring-gene knockdown flags
    (not in the authors' filter; a sensitivity knob)."""
    o = pseq.obs
    m = (o["n_total_de_genes"].astype(float) >= min_de_genes).to_numpy()
    if drop_flagged:
        for c in ("distal_offtarget_flag", "neighboring_gene_KD"):
            if c in o:
                m &= ~o[c].astype(str).str.lower().eq("true").to_numpy()
    return m


def build_X(pseq, genes, regulators=None, layer: str = "zscore_masked") -> pd.DataFrame:
    """Genes x regulators matrix (the authors' X_perturbs orientation), regulator symbols as columns."""
    sub = pseq[regulators if regulators is not None else slice(None), list(genes)]
    X = pd.DataFrame(np.asarray(sub.layers[layer], dtype=np.float32).T,
                     index=pd.Index(sub.var_names, name="gene"),
                     columns=pd.Index(sub.obs["regulator"].astype(str), name="regulator"))
    if not X.columns.is_unique:
        X = X.loc[:, ~X.columns.duplicated()]
    return X


def mask_cis_effects(X_perturbs: pd.DataFrame) -> pd.DataFrame:
    """authors' (verbatim): mask diagonal (cis) elements and impute with column means."""
    X_perturbs_masked = X_perturbs.copy()
    mask = pd.DataFrame(False, index=X_perturbs_masked.index, columns=X_perturbs_masked.columns)
    for idx in X_perturbs_masked.index:
        if idx in X_perturbs_masked.columns:
            mask.loc[idx, idx] = True
    X_perturbs_masked = X_perturbs_masked.mask(mask)
    X_perturbs_masked = X_perturbs_masked.fillna(X_perturbs_masked.mean())
    return X_perturbs_masked


def scramble_rows(X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """authors': scramble z-scores for each gene across regulators (keeps the gene's mean z)."""
    rng = np.random.default_rng(seed)
    return X.apply(lambda row: rng.permutation(row.values), axis=1, result_type="broadcast")


# ----------------------------------------------------------------------------- signatures (y)
def signature_from_deseq(res: pd.DataFrame, lfc="log2FoldChange", se="lfcSE", gene=None) -> pd.Series:
    """authors': z = log2FC / lfcSE; drop NaN/inf; one row per gene (keep the largest z)."""
    df = res.copy()
    if gene is not None:
        df = df.set_index(gene)
    z = (df[lfc] / df[se]).replace([np.inf, -np.inf], np.nan).dropna()
    z = z.groupby(level=0).max()
    z.index = z.index.astype(str)
    return z.rename("zscore")


def prepare_y(y: pd.Series, scale: bool = True) -> pd.Series:
    """authors' `scale_y=True`: standardize the signature (also removes a mean-LFC bias)."""
    y = y.astype(float)
    return (y - y.mean()) / y.std() if scale else y


def load_activation_signature(path: Path = ARCE_CSV) -> pd.Series:
    """authors': Arce 2024 Teff stimulated-vs-resting DESeq2 -> z, deduplicated per gene symbol."""
    df = pd.read_csv(path).dropna(subset=["log2FoldChange", "lfcSE"])
    df["zscore"] = df["log2FoldChange"] / df["lfcSE"]
    df = df.set_index("gene_name").sort_values("zscore").groupby(level=0).last()
    return df["zscore"].rename("zscore")


# ----------------------------------------------------------------------------- the model
def fit_pert2state(X: pd.DataFrame, y: pd.Series, *, n_pcs: int = N_PCS, n_splits: int = 5,
                   n_repeats: int = 3, seed: int = 42, scramble: bool = False,
                   model_id: str = "MF", l1_ratio=L1_RATIO, alphas=ALPHAS):
    """authors' `_train_model`: Perturb2StateModel(pca_transform=True, n_pcs=60, 5-fold x repeats,
    positive=False) on the cis-masked z matrix and the standardized signature."""
    from pert2state_model import Perturb2StateModel
    common = X.index.intersection(y.index)
    Xc, yc = X.loc[common], prepare_y(y.loc[common])
    if scramble:
        Xc = scramble_rows(Xc, seed)
    m = Perturb2StateModel(pca_transform=True, n_pcs=n_pcs, n_splits=n_splits, n_repeats=n_repeats,
                           random_state=seed, positive=False, l1_ratio=l1_ratio, alpha=alphas)
    m.fit(Xc, yc, model_id=model_id)
    m.X_, m.y_ = Xc, yc
    return m


def heldout_summary(models: dict) -> pd.DataFrame:
    """One row per fitted model: held-out-gene Pearson R (mean +- SEM over splits), chosen
    hyper-parameters. Argument: {label: model}."""
    rows = []
    for k, m in models.items():
        s = m.summarize_eval().iloc[0]
        rows.append({"model": k, "n_genes": len(m.y_), "n_regulators": m.X_.shape[1],
                     "test_pearson": s["test_pearson"], "test_pearson_se": s["test_pearson_se"],
                     "train_pearson": s["train_pearson"], "test_r2": s["test_r2"],
                     "alpha": s["alpha"], "l1_ratio": s["l1_ratio"], "n_splits": len(m.eval)})
    return pd.DataFrame(rows).set_index("model")


def replication_eval(model, y_repl: pd.Series) -> pd.DataFrame:
    """authors' (1k1k notebook, 'Get effect of aging in replication dataset'): per split, correlate the
    held-out-gene prediction with the *replication* signature; `upper_correlation` is the
    discovery-vs-replication signature correlation on the same genes (the ceiling)."""
    X, y = model.X_, model.y_
    splits_pred = model.get_prediction(X, return_splits=True)
    rows = []
    for i, (_, test_idx) in enumerate(model.split_ixs):
        test_pred = splits_pred.iloc[test_idx, i]
        test_input = y.iloc[test_idx]
        test_true = y_repl.reindex(test_input.index)
        ok = test_true.notna().to_numpy()
        rows.append({"split": i, "n_test_genes": int(ok.sum()),
                     "correlation": np.corrcoef(test_pred[ok], test_true[ok])[0, 1],
                     "upper_correlation": np.corrcoef(test_input[ok], test_true[ok])[0, 1],
                     "pred_input_correlation": np.corrcoef(test_pred, test_input)[0, 1]})
    return pd.DataFrame(rows)


def coef_table(model, clusters: pd.Series | None = None) -> pd.DataFrame:
    """authors': mean/SEM of the regulator weights over folds and `coef_rank` (0 = largest w)."""
    c = model.get_coefs().copy()
    c["coef_rank"] = (c["coef_mean"].rank(ascending=False) - 1) / (len(c) - 1)
    c["abs_rank"] = c["coef_mean"].abs().rank(ascending=False).astype(int)
    c.index.name = "regulator"
    if clusters is not None:
        c["cluster"] = clusters.reindex(c.index)
    return c.sort_values("coef_mean", ascending=False)


def lic_order(X: pd.DataFrame, sig_genes) -> list:
    """authors' 'Long Island City' x-axis: regulators ordered by Ward clustering of their z effects
    on the significant signature genes."""
    import scipy.cluster.hierarchy as sch
    g = X.index.intersection(pd.Index(sig_genes))
    M = X.loc[g].T.to_numpy()
    link = sch.linkage(M, method="ward")
    return X.columns[sch.leaves_list(link)].tolist()


def controlled_genes(X: pd.DataFrame, y: pd.Series, pseq, regulators, w: pd.Series, fdr=0.1):
    """authors': signature genes 'controlled by' a regulator set -- significant for >=1 regulator and
    sign(mean z) * sign(w) consistent with sign(y)."""
    sub = pseq[pseq.obs["regulator"].isin(regulators)]
    sig = (np.asarray(sub.layers["adj_p_value"]) < fdr).any(axis=0)
    genes = sub.var_names[sig].intersection(X.index).intersection(y.index)
    zmean = X.loc[genes, list(regulators)].mean(axis=1)
    wsign = np.sign(w.reindex(list(regulators)).mean())
    ok = np.sign(zmean) * wsign == np.sign(y.loc[genes])
    return genes[ok.to_numpy()]


# ----------------------------------------------------------------------------- pathway tests
def _cliffs_delta(a, b):
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def pathway_tests(coefs: pd.DataFrame, sets: dict, min_n: int = 5) -> pd.DataFrame:
    """authors' pathway test -- one-sided Mann-Whitney of |w| for set members vs all regulators
    (`mannwhitneyu(abs_set, abs_all, alternative='greater')`) -- plus the signed versions we need
    for direction (members' w > rest: 'KD mimics MF'; < rest: 'KD opposes MF'). BH per column."""
    w = coefs["coef_mean"]
    rows = []
    for name, genes in sets.items():
        g = w.index.intersection(pd.Index(list(genes)))
        if len(g) < min_n:
            continue
        a, b = w.loc[g], w.drop(g)
        rows.append({"set": name, "n": len(g), "n_total": len(w), "mean_w": a.mean(),
                     "mean_abs_w": a.abs().mean(), "delta_signed": _cliffs_delta(a, b),
                     "p_abs_greater": ss.mannwhitneyu(a.abs(), w.abs(), alternative="greater").pvalue,
                     "p_signed_greater": ss.mannwhitneyu(a, b, alternative="greater").pvalue,
                     "p_signed_less": ss.mannwhitneyu(a, b, alternative="less").pvalue,
                     "members": ",".join(g[:12]) + ("..." if len(g) > 12 else "")})
    out = pd.DataFrame(rows)
    if len(out):
        for c in ("p_abs_greater", "p_signed_greater", "p_signed_less"):
            out["q" + c[1:]] = multipletests(out[c], method="fdr_bh")[1]
    return out.sort_values("p_abs_greater")


def load_clusters(path: Path = CLUSTERS_CSV) -> pd.DataFrame:
    """The authors' 111 regulator clusters (Suppl. Table 13) with member lists parsed."""
    df = pd.read_csv(path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df["members"] = df["cluster_member"].apply(lambda s: re.findall(r"'([^']+)'", str(s)))
    df["name"] = "c" + df["cluster"].astype(str) + " " + df["manual_annotation"].astype(str)
    return df


def cluster_sets(clusters: pd.DataFrame) -> dict:
    return dict(zip(clusters["name"], clusters["members"]))


def regulator_cluster_map(clusters: pd.DataFrame) -> pd.Series:
    m = {}
    for _, r in clusters.iterrows():
        for g in r["members"]:
            m.setdefault(g, r["name"])
    return pd.Series(m, name="cluster")


def load_downstream(cluster: int, condition: str = "Stim48hr", path: Path = DOWNSTREAM_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df[(df["hdbscan_cluster"] == cluster) & (df["condition"] == condition)].copy()


def program_projection(y: pd.Series, downstream: pd.DataFrame, *, min_upstream: int = 3,
                       min_coherence: float = 0.5, n_perm: int = 2000, seed: int = 0) -> dict:
    """Where does the MF signature sit relative to a cluster's downstream program?
    `sign_coherence` > 0 = the cluster's knockdowns UP-regulate the gene. Projection =
    Pearson(y, sign) over coherent genes plus mean y of KD-up vs KD-down genes, with a gene-label
    permutation p. A positive projection means the MF program resembles the knockdown of the
    cluster (for the TCR-signaling cluster: 'below' the stimulated CD4 state)."""
    d = downstream[(downstream["num_of_upstream"] >= min_upstream)
                   & (downstream["sign_coherence"].abs() >= min_coherence)]
    d = d.set_index("downstream_gene")
    g = d.index.intersection(y.index)
    s = np.sign(d.loc[g, "sign_coherence"]).astype(float)
    yy = y.loc[g].astype(float)
    r = float(np.corrcoef(yy, s)[0, 1]) if len(g) > 2 else np.nan
    obs = yy[s > 0].mean() - yy[s < 0].mean()
    rng = np.random.default_rng(seed)
    sv = s.to_numpy()
    null = np.array([(lambda p: p[sv > 0].mean() - p[sv < 0].mean())(rng.permutation(yy.to_numpy()))
                     for _ in range(n_perm)]) if len(g) > 2 else np.array([np.nan])
    return {"n_genes": int(len(g)), "n_kd_up": int((s > 0).sum()), "n_kd_down": int((s < 0).sum()),
            "pearson_y_vs_kd_sign": r, "mean_y_kd_up": float(yy[s > 0].mean()),
            "mean_y_kd_down": float(yy[s < 0].mean()), "diff": float(obs),
            "p_perm": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))}


# ----------------------------------------------------------------------------- genes -> arms
def gene_arms(gtf: Path = GTF, genes=None) -> pd.Series:
    """gene symbol -> chromosome arm ('chr10q'), from the same GTF the inferCNV pipeline used and
    `skin_T_cnv_helpers.HG38_CENTROMERE` / `_arm_labels`."""
    import skin_T_cnv_helpers as H
    want = set(genes) if genes is not None else None
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', f[8])
            if not m or (want is not None and m.group(1) not in want):
                continue
            chrom = f[0] if f[0].startswith("chr") else "chr" + f[0]
            rows.append((m.group(1), chrom, int(f[3])))
    var = pd.DataFrame(rows, columns=["gene", "chromosome", "start"]).drop_duplicates("gene").set_index("gene")
    arm = pd.Series(H._arm_labels(var), index=var.index, name="arm")
    return arm[arm != ""]


def parse_signature(sig) -> dict:
    """'chr7p+, chr10q-' -> {'chr7p': 1, 'chr10q': -1}."""
    out = {}
    for tok in str(sig).split(","):
        tok = tok.strip()
        if tok and tok[-1] in "+-":
            out[tok[:-1]] = 1 if tok[-1] == "+" else -1
    return out


def arm_event_table(donors, summary_csv: Path = CNV_SUMMARY) -> pd.DataFrame:
    """Donor x arm matrix of signed events (+1 gain / -1 loss / 0) for the callable donors among
    `donors`, from the nb23 v5 per-donor signature."""
    s = pd.read_csv(summary_csv).set_index("donor")
    s = s.loc[s.index.intersection(pd.Index(donors))]
    s = s[s["cnv_state"].astype(str).eq("callable")]
    ev = {d: parse_signature(s.loc[d, "signature"]) for d in s.index}
    arms = sorted({a for e in ev.values() for a in e})
    return pd.DataFrame([[ev[d].get(a, 0) for a in arms] for d in ev], index=ev.keys(), columns=arms)


def arm_classes(events: pd.DataFrame, min_frac: float = 0.25) -> pd.DataFrame:
    """Cohort arm class: 'lost' / 'gained' if the event recurs in >= min_frac of callable donors
    (ties go to the more frequent direction), else 'neutral'."""
    n = len(events)
    df = pd.DataFrame({"frac_lost": (events == -1).sum() / n, "frac_gained": (events == 1).sum() / n})
    df["n_donors"] = n
    df["class"] = "neutral"
    df.loc[(df["frac_lost"] >= min_frac) & (df["frac_lost"] >= df["frac_gained"]), "class"] = "lost"
    df.loc[(df["frac_gained"] >= min_frac) & (df["frac_gained"] > df["frac_lost"]), "class"] = "gained"
    df.index.name = "arm"
    return df.sort_values(["class", "frac_lost", "frac_gained"], ascending=[True, False, False])


def arm_class_test(coefs: pd.DataFrame, gene_arm: pd.Series, classes: pd.DataFrame, *,
                   n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Do regulators on recurrently lost arms have w > neutral-arm regulators (KD mimics MF), and
    those on gained arms w < neutral? Signed Mann-Whitney + Cliff's delta + an arm-label permutation
    p (shuffles the arm assignment across regulators, keeps the w distribution); plus the authors'
    unsigned |w| test for any-event arms."""
    w = coefs["coef_mean"].copy()
    arm = gene_arm.reindex(w.index)
    cls = classes["class"].reindex(arm.values).to_numpy()
    cls = pd.Series(np.where(pd.isna(cls), "unplaced", cls), index=w.index)
    rng = np.random.default_rng(seed)
    neutral = w[cls == "neutral"]
    rows = []
    for name, side, use_abs in (("lost vs neutral", "greater", False), ("gained vs neutral", "less", False),
                                ("any event vs neutral (|w|, authors' test)", "greater", True)):
        sel = cls.isin(["lost", "gained"]) if use_abs else cls.eq(name.split()[0])
        a = w[sel].abs() if use_abs else w[sel]
        b = neutral.abs() if use_abs else neutral
        if len(a) < 5:
            continue
        obs = a.mean() - b.mean()
        pool = pd.concat([a, b]).to_numpy()
        na = len(a)
        null = np.array([(lambda p: p[:na].mean() - p[na:].mean())(rng.permutation(pool)) for _ in range(n_perm)])
        p_perm = ((null >= obs).sum() + 1) / (n_perm + 1) if side == "greater" else ((null <= obs).sum() + 1) / (n_perm + 1)
        rows.append({"test": name, "n_set": na, "n_neutral": len(b), "mean_set": a.mean(), "mean_neutral": b.mean(),
                     "cliffs_delta": _cliffs_delta(a, b), "p_mw": ss.mannwhitneyu(a, b, alternative=side).pvalue,
                     "p_perm": p_perm, "expected": side})
    return pd.DataFrame(rows)


def subset_prediction_matrix(model) -> pd.DataFrame:
    """Genes x regulators matrix W such that W[:, S].sum(1) equals the authors'
    `get_prediction(X, subset_regulators=S)["pred_mean"]` (mean over the fitted folds of the
    scaled X times the back-transformed coefficients) -- precomputed once so thousands of
    regulator subsets can be scored."""
    X = model.X_
    Xa = X.values
    W = np.zeros(X.shape, dtype=float)
    for mod, pca, scaler in zip(model.models, model.pcas, model.scalers):
        coefs = np.matmul(pca.components_.T, mod.coef_) if pca is not None else mod.coef_
        W += scaler.transform(Xa) * coefs[None, :]
    W /= len(model.models)
    return pd.DataFrame(W, index=X.index, columns=X.columns)


def dosage_predicted_program(model, gene_arm: pd.Series, events: pd.DataFrame, Y_donor: pd.DataFrame,
                             *, n_null: int = 200, seed: int = 0, min_regs: int = 5) -> pd.DataFrame:
    """authors' `get_prediction(X, subset_regulators=...)` used per donor: reconstruct the signature
    from the regulators sitting on that donor's own lost arms (and, separately, gained arms) and
    correlate with the donor's own log2FC; null = random regulator subsets of the same size."""
    W = subset_prediction_matrix(model)
    arm = gene_arm.reindex(W.columns)
    rng = np.random.default_rng(seed)
    rows = []
    cols = np.arange(W.shape[1])
    Wv = W.values
    for d in events.index:
        if d not in Y_donor.columns:
            continue
        yd = Y_donor[d].reindex(W.index)
        ok = yd.notna().to_numpy()
        yv = yd.to_numpy()[ok]
        for cls, val in (("lost", -1), ("gained", 1)):
            arms = events.columns[(events.loc[d] == val).to_numpy()]
            idx = np.where(arm.isin(arms).to_numpy())[0]
            if len(idx) < min_regs:
                rows.append({"donor": d, "class": cls, "n_arms": len(arms), "n_regulators": len(idx),
                             "r_obs": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p_null": np.nan})
                continue
            r_obs = np.corrcoef(Wv[ok][:, idx].sum(1), yv)[0, 1]
            null = np.array([np.corrcoef(Wv[ok][:, rng.choice(cols, size=len(idx), replace=False)].sum(1), yv)[0, 1]
                             for _ in range(n_null)])
            rows.append({"donor": d, "class": cls, "n_arms": len(arms), "n_regulators": len(idx),
                         "r_obs": r_obs, "null_mean": null.mean(), "null_sd": null.std(),
                         "p_null": ((null >= r_obs).sum() + 1) / (n_null + 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- per-cell scoring
def kd_gene_sets(pseq, regulators, gene_arm: pd.Series, *, n: int = 50, fdr: float = 0.1,
                 drop_cis_arm: bool = True) -> dict:
    """Top-n up / down genes of each regulator's knockdown (FDR < 0.1, ranked by z), the authors'
    Fig 4G construction; genes on the regulator's own arm are removed so the per-cell score and the
    inferCNV arm value do not share genes."""
    out = {}
    o = pseq.obs["regulator"].astype(str)
    for r in regulators:
        i = np.where(o.to_numpy() == r)[0]
        if not i.size:
            continue
        i = int(i[0])
        z = pd.Series(np.asarray(pseq.layers["zscore"][i]).ravel(), index=pseq.var_names)
        q = pd.Series(np.asarray(pseq.layers["adj_p_value"][i]).ravel(), index=pseq.var_names)
        z = z[(q < fdr) & (z.index != r)]
        if drop_cis_arm and r in gene_arm.index:
            z = z[gene_arm.reindex(z.index).ne(gene_arm[r]).to_numpy()]
        up, down = z.nlargest(n), z.nsmallest(n)
        out[r] = {"up": up[up > 0].index.tolist(), "down": down[down < 0].index.tolist()}
    return out


def kd_scores(expr: pd.DataFrame, donor: pd.Series, reference: pd.Series, sets: dict,
              min_genes: int = 10, min_ref: int = 20) -> pd.DataFrame:
    """authors' Fig 4G score, with same-donor reactive CD4 in place of NTC cells: per donor,
    standardize each gene by the mean/SD of the reference cells, score = mean(z_up) - mean(z_down).
    `expr` = cells x genes log1p CP10K (dense DataFrame), `reference` = bool per cell."""
    scores = pd.DataFrame(index=expr.index, columns=list(sets), dtype=float)
    for d in donor.unique():
        cells = donor.index[donor == d]
        ref = cells[reference.loc[cells].to_numpy()]
        if len(ref) < min_ref:
            continue
        mu = expr.loc[ref].mean()
        sd = expr.loc[ref].std().replace(0, np.nan)
        zc = (expr.loc[cells] - mu) / sd
        for r, s in sets.items():
            up = [g for g in s["up"] if g in zc.columns and np.isfinite(sd.get(g, np.nan))]
            dn = [g for g in s["down"] if g in zc.columns and np.isfinite(sd.get(g, np.nan))]
            if len(up) + len(dn) < min_genes:
                continue
            val = (zc[up].mean(axis=1) if up else 0) - (zc[dn].mean(axis=1) if dn else 0)
            scores.loc[cells, r] = val.to_numpy()
    return scores


def per_cell_arm_test(scores: pd.DataFrame, armz: pd.DataFrame, donor: pd.Series, reg_arm: pd.Series,
                      events: pd.DataFrame, *, min_cells: int = 50) -> tuple:
    """Within each donor, Spearman rho between a regulator's KD-mimicry score and the per-cell
    inferCNV z of the regulator's arm; rows carry the donor's event on that arm (+1/-1/0) so
    event arms and no-event arms (the control) are tested the same way. Returns (long, summary):
    BH over all rows; summary per regulator x event class with Stouffer z across donors."""
    rows = []
    for r in scores.columns:
        a = reg_arm.get(r)
        if a is None or a not in armz.columns:
            continue
        for d in donor.unique():
            cells = donor.index[donor == d]
            s = scores.loc[cells, r]
            z = armz.reindex(cells)[a]
            ok = s.notna() & z.notna()
            if ok.sum() < min_cells:
                continue
            rho, p = ss.spearmanr(s[ok], z[ok])
            ev = int(events.loc[d, a]) if (d in events.index and a in events.columns) else 0
            rows.append({"regulator": r, "arm": a, "donor": d, "event": ev, "n_cells": int(ok.sum()),
                         "rho": rho, "p": p})
    long = pd.DataFrame(rows)
    if not len(long):
        return long, long
    long["q"] = multipletests(long["p"], method="fdr_bh")[1]
    long["z"] = ss.norm.isf(long["p"] / 2) * np.sign(long["rho"])
    long["event_class"] = long["event"].map({-1: "lost", 1: "gained", 0: "no event"})
    summ = (long.groupby(["regulator", "arm", "event_class"])
                .agg(n_donors=("donor", "nunique"), mean_rho=("rho", "mean"), median_rho=("rho", "median"),
                     stouffer_z=("z", lambda v: v.sum() / np.sqrt(len(v))))
                .reset_index())
    summ["p_stouffer"] = 2 * ss.norm.sf(summ["stouffer_z"].abs())
    summ["q_stouffer"] = multipletests(summ["p_stouffer"], method="fdr_bh")[1]
    return long, summ.sort_values("p_stouffer")


# ----------------------------------------------------------------------------- misc
def control_ranks(coefs: pd.DataFrame, panel: dict = CONTROL_PANEL) -> pd.DataFrame:
    """The control panel's w, coef_rank and whether the sign matches expectation (the authors'
    known-regulator highlighting, made explicit)."""
    rows = []
    for grp, genes in panel.items():
        expect = 1 if "w>0" in grp else -1
        for g in genes:
            if g in coefs.index:
                r = coefs.loc[g]
                rows.append({"group": grp, "regulator": g, "coef_mean": r["coef_mean"], "coef_sem": r["coef_sem"],
                             "coef_rank": r["coef_rank"], "abs_rank": int(r["abs_rank"]),
                             "sign_ok": bool(np.sign(r["coef_mean"]) == expect)})
            else:
                rows.append({"group": grp, "regulator": g, "coef_mean": np.nan, "coef_sem": np.nan,
                             "coef_rank": np.nan, "abs_rank": np.nan, "sign_ok": np.nan})
    return pd.DataFrame(rows)


def stratified_donor_split(don: pd.DataFrame, by: str = "study", frac: float = 0.2, seed: int = 0) -> pd.Series:
    """authors' discovery/replication split (OneK1K: 80/20 stratified by age bin), here stratified
    by study; every stratum with >= 2 donors contributes at least one replication donor."""
    rng = np.random.default_rng(seed)
    split = pd.Series("discovery", index=don.index)
    for _, g in don.groupby(by):
        n = len(g)
        k = int(round(n * frac)) if n >= 5 else (1 if n >= 2 else 0)
        if k:
            split.loc[rng.choice(g.index, size=k, replace=False)] = "replication"
    return split.rename("split")
