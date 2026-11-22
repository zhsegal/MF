"""Cluster-level malignancy for skin T cells (nb26b).

Per sample: Leiden (res 1.0) over all T cells (CD4 + Treg + CD8), then one malignant/benign call
per cluster.

  jobs/run_cluster_malig.py        per-sample clustering, UMAP, cluster pseudobulks, per-cell
                                   V-gene calls and ALICE family membership.
  jobs/run_cluster_malig_model.py  Step 1: on the ALICE-labelled samples, learn which score
                                   separates malignant from benign clusters (leave-one-study-out)
                                   and set two cutoffs from held-out scores. Step 2: score every
                                   cluster; clear ones are called, unclear ones go to a review sheet
                                   + CSV for a manual call.
  nb26b                            the decisions (which score, how strict) and the manual calls.

Cluster keys are `<sample_id>__<k>` with k ordered by cluster size (0 = largest).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"

TAB_DIR, FIG_DIR = NB_DIR / "tables", NB_DIR / "figures"

# ------------------------------------------------------------------ tissue configuration
# Everything that differs between skin and blood lives here. The functions below and both job
# scripts read module-level names, so `use_tissue()` rebinds those names rather than threading a
# config object through 15 signatures. Skin is applied at import, so nb26b and the existing skin
# jobs behave exactly as before.
#
# The grouping unit is NOT the same in the two compartments. Skin clusters per `sample_id`; blood
# clusters per `donor`, because geskin `sample_id` is an HTO lane pooling 5-6 donors and a cluster
# spanning donors is not a clone. The job writes the chosen key into a column still called
# `sample_id`, so the cluster key `<sample_id>__<k>` and every table keep their shape.
CONFIG = {
    "skin": dict(
        TCR_OBJ=ATLAS_JOINT / "skin_T_tcr_annotated_v4.h5ad",   # nb23 cache: li2024 V(D)J folded in
        ALICE_PQ=ATLAS_JOINT / "alice_malignancy_v4.parquet",   # cell_id, tcr_clonal (CD4 only)
        ALICE_COL="tcr_clonal",
        CLUSTERS_PQ=ATLAS_JOINT / "skin_T_clusters_v1.parquet",             # per cell
        PSEUDOBULK_H5AD=ATLAS_JOINT / "skin_T_cluster_pseudobulk_v1.h5ad",  # per cluster
        CALLS_PQ=ATLAS_JOINT / "skin_T_malignancy_cluster_v1.parquet",      # final per-cell call
        MODEL_JSON=ATLAS_JOINT / "skin_T_cluster_malig_model_v1.json",      # settings + cutoffs
        MANIFEST_JSON=ATLAS_JOINT / "skin_T_clusters_v1_manifest.json",
        COUNTS="raw_counts",
        GROUP_KEY="sample_id",
        REVIEW_DIR=FIG_DIR / "nb26b_review",
        CLONALITY_CSV=TAB_DIR / "skin_tcr_clonality_per_sample.csv",  # nb23 Step 4: ALICE coverage
        KNN_PQ=ATLAS_JOINT / "cd4_transcriptome_malignancy_v1.parquet",  # mal_knn, comparison only
        CELLTYPE_FIX_CSV=None,                                  # see the blood entry
        TAB_PREFIX="cluster_malig_",
        OBS_COLS=["sample_id", "donor", "study", "disease", "stage_class", "entity", "tech",
                  "cell_type_T", "has_tcr", "tra_cdr3", "trb_cdr3", "_li_clone_key"],
        META_COLS=["donor", "study", "disease", "stage_class", "entity", "tech"],
        CELL_TYPES=["CD4", "CD4_Treg", "CD8"],
        # nb26 a-priori markers.
        CTCL_UP=["TOX", "GATA3", "PLS3", "TWIST1", "KIR3DL2", "TNFRSF8", "CDK6", "TRAF1", "DNM3"],
        CTCL_DOWN=["CD7", "DPP4", "SATB1", "STAT4"],
        MIN_CD4=0.8,
    ),
    "blood": dict(
        TCR_OBJ=ATLAS_JOINT / "blood_T_annotated.h5ad",         # nb31 v2 re-annotation
        ALICE_PQ=ATLAS_JOINT / "alice_malignancy_blood_v2.parquet",
        ALICE_COL="tcr_malignant_alice",                        # nb32 names it differently
        CLUSTERS_PQ=ATLAS_JOINT / "blood_T_clusters_v1.parquet",
        PSEUDOBULK_H5AD=ATLAS_JOINT / "blood_T_cluster_pseudobulk_v1.h5ad",
        CALLS_PQ=ATLAS_JOINT / "blood_T_malignancy_cluster_v1.parquet",
        MODEL_JSON=ATLAS_JOINT / "blood_T_cluster_malig_model_v1.json",
        MANIFEST_JSON=ATLAS_JOINT / "blood_T_clusters_v1_manifest.json",
        COUNTS="counts",                                        # nb31 writes `counts`, not raw_counts
        GROUP_KEY="donor",
        REVIEW_DIR=FIG_DIR / "nb32b_review",
        CLONALITY_CSV=TAB_DIR / "blood_tcr_clonality_per_sample.csv",   # nb32
        KNN_PQ=None,                                            # no mal_knn in blood
        # Cluster-level cell-type corrections from the nb32b review (cell_id, cell_type_T).
        # nb31 called 12 cytotoxic clusters CD8 that carry the NK signature and almost no TCR;
        # read_obs overlays this file so every consumer of `cell_type_T` sees the same labels.
        CELLTYPE_FIX_CSV=ATLAS_JOINT / "blood_T_celltype_corrections_v1.csv",
        TAB_PREFIX="blood_cluster_malig_",
        OBS_COLS=["donor", "sample_id", "study", "disease", "entity",
                  "cell_type_T", "has_tcr", "tra_cdr3", "trb_cdr3"],
        META_COLS=["donor", "study", "disease", "entity"],
        # Measured from the object 2026-10-22: blood `cell_type_T` is only CD4 / CD8 / NK
        # (554,369 / 112,609 / 606). There are no CD4_Treg, gdT or MAIT labels, so listing them
        # would only add all-zero composition columns. CD4_Treg is kept anyway because
        # run_cluster_malig_model.py's REV_COLS reads `frac_CD4_Treg` by name and would KeyError
        # without it; it will be zero throughout, which is honest.
        CELL_TYPES=["CD4", "CD4_Treg", "CD8", "NK"],
        # nb31 Sezary panel. CCR7/SELL/TCF7 are deliberately NOT in CTCL_DOWN: Sezary cells are
        # central-memory-like, so their positivity is not evidence that a cluster is benign.
        CTCL_UP=["KIR3DL2", "CADM1", "TWIST1", "PLS3", "DNM3", "TNFRSF8", "TNFRSF1B", "TOX",
                 "GATA3"],
        CTCL_DOWN=["CD7", "DPP4", "CD27", "CD5", "CD6"],
        # Blood is ~92% malignant CD4: a benign cluster that is >= 80% CD4 barely exists, and
        # consensus_signature would raise "no sample has both". HC donors and CD8 anchor instead.
        MIN_CD4=0.5,
    ),
}

TISSUE = None


def use_tissue(tissue="skin"):
    """Rebind this module's tissue-dependent globals (paths, obs columns, markers, prefixes)."""
    global TISSUE
    if tissue not in CONFIG:
        raise ValueError(f"unknown tissue {tissue!r}; have {sorted(CONFIG)}")
    g = globals()
    g.update(CONFIG[tissue])
    pre = CONFIG[tissue]["TAB_PREFIX"]
    g["CV_CSV"] = TAB_DIR / f"{pre}cv_v1.csv"              # held-out, candidate x fold
    g["QUEUE_CSV"] = TAB_DIR / f"{pre}queue_v1.csv"        # review load, candidate x MAX_ERR
    g["PROGRAM_CSV"] = TAB_DIR / f"{pre}program_v1.csv"
    g["CLUSTERS_CSV"] = TAB_DIR / f"{pre}clusters_v1.csv"
    g["REVIEW_CSV"] = TAB_DIR / f"{pre}review_v1.csv"
    g["EXCLUDE_CSV"] = TAB_DIR / f"{pre}excluded_samples_v1.csv"  # sample_id, note: sample is out
    TISSUE = tissue
    return tissue


RES, N_HVG, N_PCS, N_NEIGHBORS = 1.0, 2000, 30, 15
MIN_SAMPLE_CELLS = 31      # cluster samples with > 30 T cells
MIN_CLUSTER_CELLS = 10     # smaller Leiden clusters are merged into the nearest centroid
SEED = 0

# ---- Step 1 / Step 2 settings ----
MIN_TCR, FAM_HI, FAM_LO = 10, 0.8, 0.1     # cluster ALICE label: family share of its TCR+ cells
MAX_ERR = 0.02                            # default: <= this share of auto-called cells wrong
MAX_ERR_GRID = (0.01, 0.02, 0.05)         # review load reported at each, for the decision
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}   # tumour not ab-CD4
N_UP = N_DOWN = 50
CANDIDATES = {"program": ["program"],             # learned gene program
              "ctcl_apriori": ["ctcl"],           # nb26 marker lists
              "vskew": ["v_z"],                   # TRBV/TRAV skew in the expression reads
              "program+vskew": ["program", "v_z"]}
MARKERS = ["CD4", "CD8A", "FOXP3", "MKI67", "CD7", "DPP4", "TOX", "KIR3DL2", "GTSF1", "PLS3"]
# clone-V check: a sample's clone V = top V of its most skewed non-CD8 cluster (v_z >= Z_CLONE); an
# auto-benign cluster whose top V is that gene (v_z >= Z_SHARE, >= SHARE_MIN of its V calls) goes to
# review instead. Without SHARE_MIN it flagged 17% of ALICE-benign clusters; with 0.3 it flags only
# the one held-out auto-benign ALICE cluster that is malignant (2026-09-28 run).
Z_CLONE, Z_SHARE, SHARE_MIN = 5.0, 3.0, 0.3

use_tissue("skin")      # default; jobs/notebooks call use_tissue("blood") before anything else

# out of clustering HVGs and out of any program: a clone would otherwise cluster on its own TRBV
_CLUSTER_EXCLUDE = re.compile(r"^(TR[ABGD][VDJC]|IG[HKL][VDJC]|MT-|RP[SL]\d|RPLP|MRP[SL])")
# additionally out of the program: sex-linked, stress/dissociation, hemoglobin, CD8 lineage
_PROGRAM_EXCLUDE = re.compile(
    r"^(XIST|TSIX|RPS4Y|DDX3Y|UTY|KDM5D|EIF1AY|USP9Y|ZFY|NLGN4Y|TXLNGY|HSPA|HSPH|HSPB1|HSPD1|"
    r"HSPE1|DNAJB1|FOS|JUN|IER[235]|EGR[123]|ATF3|HB[ABDGMQZ]\d?$|HBE1$|CD8[AB]$)")


def cluster_excluded(genes):
    return np.array([bool(_CLUSTER_EXCLUDE.match(g)) for g in genes])


def program_excluded(genes):
    return np.array([bool(_CLUSTER_EXCLUDE.match(g) or _PROGRAM_EXCLUDE.match(g)) for g in genes])


# ------------------------------------------------------------------ h5ad reading (job)
def _read_elem(g):
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    return read_elem(g)


def read_obs(h5, cols=None):
    """Selected obs columns of an open h5ad, indexed by obs_names.

    Resolved at call time, not at import: `use_tissue` rebinds OBS_COLS. Columns the object does
    not carry are returned empty rather than raising -- blood has no `stage_class` or `tech`, and
    a missing design column must surface as an empty column in the manifest, not a KeyError four
    hours into a job.
    """
    cols = list(OBS_COLS if cols is None else cols)
    g = h5["obs"]
    idx = _read_elem(g[g.attrs["_index"]])
    have = set(g.keys())
    missing = [c for c in cols if c not in have]
    if missing:
        import warnings
        warnings.warn(f"obs columns absent from the object, filled empty: {missing}")
    out = pd.DataFrame({c: (_read_elem(g[c]) if c in have else "") for c in cols},
                       index=pd.Index(idx, name="cell_id"))
    return apply_celltype_fix(out)


def apply_celltype_fix(obs, path=None):
    """Overlay the manual `cell_type_T` corrections for this tissue, if any.

    The h5ad is the record of what nb31/nb21 called; this CSV is the record of what the nb32b/nb26b
    cluster review found wrong. Keeping them apart means the correction is one reviewable file
    instead of a rewrite of a 600k-cell object, and `cell_type_T_nb31` stays on the frame so a
    notebook can show what changed.
    """
    path = CELLTYPE_FIX_CSV if path is None else Path(path)
    if path is None or "cell_type_T" not in obs.columns or not Path(path).exists():
        return obs
    fix = pd.read_csv(path, usecols=["cell_id", "cell_type_T"], dtype=str).set_index("cell_id")
    hit = obs.index.intersection(fix.index)
    if len(hit) == 0:
        return obs
    obs = obs.copy()
    obs["cell_type_T_nb31"] = obs["cell_type_T"].astype(str)
    obs.loc[hit, "cell_type_T"] = fix.loc[hit, "cell_type_T"].to_numpy()
    return obs


def read_csr_rows(g, rows):
    """Rows `rows` (sorted ascending) of a CSR h5ad matrix group, one read per contiguous run."""
    rows = np.asarray(rows)
    n_cols = int(g.attrs["shape"][1])
    indptr = g["indptr"]
    parts = []
    for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
        a, b = int(run[0]), int(run[-1]) + 1
        ip = indptr[a:b + 1].astype(np.int64)
        lo, hi = int(ip[0]), int(ip[-1])
        parts.append(sparse.csr_matrix(
            (g["data"][lo:hi].astype(np.float32), g["indices"][lo:hi], ip - lo),
            shape=(b - a, n_cols)))
    return sparse.vstack(parts, format="csr")


# ------------------------------------------------------------------ TCR family (job)
def trb_key(obs):
    """TRB CDR3 per cell, built as nb23 builds it: li2024 from `_li_clone_key`, else trb_cdr3.

    `_li_clone_key` is a skin-only column (the li2024 V(D)J fold-in). Blood has no li2024 cells,
    so its absence is expected there and `trb_cdr3` is used throughout.
    """
    raw = obs["trb_cdr3"].astype(str).replace({"nan": "", "None": ""}).str.strip()
    if "_li_clone_key" not in obs.columns:
        return raw
    li = (obs["_li_clone_key"].astype(str).str.extract(r"^TRB:([A-Z]+)$", expand=False)
          .fillna(""))
    is_li = obs["study"].astype(str).eq("li2024")
    trb = raw.where(~is_li, li)
    return trb.where(trb != "", raw)


def family_mask(obs, tcr_clonal):
    """has_tcr & TRB in the donor's ALICE founder family, for every T type.

    nb23 calls the family on CD4 only. The family of a clonal donor is exactly the set of TRB
    CDR3s its `tcr_clonal` cells carry, so it is read back from them and matched on every T cell:
    a tumour cell annotated Treg or CD8 then still counts. `has_tcr` is required, which also drops
    the 1,635 li2024 cells that alice_malignancy_v4 marks clonal without a recovered TCR.
    """
    trb = trb_key(obs).to_numpy()
    has = obs["has_tcr"].astype(bool).to_numpy() & (trb != "")
    donor = obs["donor"].astype(str).to_numpy()
    clonal = tcr_clonal.reindex(obs.index).fillna(False).astype(bool).to_numpy() & has
    fam = pd.MultiIndex.from_arrays([donor[clonal], trb[clonal]]).unique()
    return pd.MultiIndex.from_arrays([donor, trb]).isin(fam) & has


# ------------------------------------------------------------------ per-sample clustering (job)
def merge_small(labels, emb, min_size=MIN_CLUSTER_CELLS):
    """Fold clusters under `min_size` into the nearest centroid; relabel 0..k-1 by size."""
    labels = np.asarray(labels).copy()
    while True:
        ids, n = np.unique(labels, return_counts=True)
        if len(ids) <= 1 or n.min() >= min_size:
            break
        small = ids[np.argmin(n)]
        cent = np.stack([emb[labels == i].mean(0) for i in ids])
        d = np.linalg.norm(cent - cent[ids == small], axis=1)
        d[ids == small] = np.inf
        labels[labels == small] = ids[np.argmin(d)]
    ids, n = np.unique(labels, return_counts=True)
    remap = {old: new for new, old in enumerate(ids[np.argsort(-n, kind="stable")])}
    return np.array([remap[x] for x in labels])


def cluster_sample(counts, genes, allowed, seed=SEED):
    """Leiden (RES) over one sample's cells on its own HVGs + PCA.

    counts = raw CSR (cells x genes); `allowed` = genes eligible as HVGs. Returns (labels, umap).
    """
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=counts.copy(), var=pd.DataFrame(index=pd.Index(genes)))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    detected = np.asarray((counts > 0).sum(0)).ravel() >= 3
    a = a[:, allowed & detected].copy()
    sc.pp.highly_variable_genes(a, n_top_genes=min(N_HVG, a.n_vars - 1), flavor="seurat")
    a = a[:, a.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(a, max_value=10)
    n_pcs = int(min(N_PCS, a.n_obs - 1, a.n_vars - 1))
    sc.tl.pca(a, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(N_NEIGHBORS, a.n_obs - 1), n_pcs=n_pcs, random_state=seed)
    sc.tl.leiden(a, resolution=RES, flavor="igraph", n_iterations=2, random_state=seed)
    labels = merge_small(a.obs["leiden"].astype(int).to_numpy(), a.obsm["X_pca"])
    sc.tl.umap(a, random_state=seed)
    return labels, a.obsm["X_umap"].astype(np.float32)


def cluster_sums(counts, labels):
    """(clusters x genes) summed counts and number of cells detecting each gene."""
    k = int(labels.max()) + 1
    ind = sparse.csr_matrix((np.ones(len(labels), np.float32), (labels, np.arange(len(labels)))),
                            shape=(k, len(labels)))
    det = counts.copy()
    det.data = (det.data > 0).astype(np.float32)
    return sparse.csr_matrix(ind @ counts), sparse.csr_matrix(ind @ det)


def v_calls(counts, genes, prefix):
    """Per cell, the V gene with most UMIs among `prefix` genes ('' if none detected)."""
    idx = np.flatnonzero(pd.Index(genes).str.startswith(prefix))
    sub = counts[:, idx].toarray()
    top = np.asarray(genes)[idx][sub.argmax(1)]
    return np.where(sub.max(1) > 0, top, "")


# ------------------------------------------------------------------ cluster tables (notebook)
def logcpm(X, keep=None):
    """log1p(CPM) per row. Library size over ALL columns first, then subset to `keep`."""
    X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=np.float64)
    lib = X.sum(1, keepdims=True)
    lib[lib == 0] = 1
    if keep is not None:
        X = X[:, keep]
    return np.log1p(X / lib * 1e6).astype(np.float32)


def tcr_cluster_labels(n_tcr, n_family, min_tcr=10, hi=0.8, lo=0.1):
    """malignant / benign / mixed from the ALICE family share of a cluster's TCR+ cells."""
    n_tcr, n_family = np.asarray(n_tcr), np.asarray(n_family)
    frac = np.divide(n_family, n_tcr, out=np.zeros(len(n_tcr)), where=n_tcr > 0)
    return np.select([n_tcr < min_tcr, frac >= hi, frac <= lo], ["none", "malignant", "benign"],
                     "mixed")


def consensus_signature(counts, genes, cl, use, n_up=50, n_down=50, min_frac=0.75,
                        min_lfc=0.25, min_cpm=10.0, min_cd4=None):
    """Genes separating malignant from benign clusters *within a sample*, consistent across donors.

    Per sample with both classes among `use` clusters: pooled malignant clusters vs pooled benign
    clusters that are >= `min_cd4` conventional CD4 (default: the tissue's `MIN_CD4`), log1p(CPM)
    difference. At a 0.5 cut the down
    list filled up with CD8 / cytotoxic genes (CD8A, NKG7, GZMA): lineage, not malignancy.
    Samples are averaged per donor, then each gene gets the median logFC across donors and the
    share of donors where it moves > `min_lfc` in each direction. Returns the per-gene table with
    `direction` in {up, down, ''}.
    """
    min_cd4 = MIN_CD4 if min_cd4 is None else min_cd4
    lab = cl["tcr_label"].to_numpy()
    mal = use & (lab == "malignant")
    ben = use & (lab == "benign") & (cl["frac_CD4"].to_numpy() >= min_cd4)
    m_rows, b_rows, donors = [], [], []
    for _, idx in cl.groupby("sample_id", observed=True).indices.items():
        im, ib = idx[mal[idx]], idx[ben[idx]]
        if len(im) and len(ib):
            m_rows.append(np.asarray(counts[im].sum(0)).ravel())
            b_rows.append(np.asarray(counts[ib].sum(0)).ravel())
            donors.append(cl["donor"].iat[idx[0]])
    if not donors:
        raise ValueError("no sample has both a malignant and a benign CD4 cluster")
    M, B = logcpm(np.vstack(m_rows)), logcpm(np.vstack(b_rows))
    g = pd.Index(np.asarray(donors), name="donor")
    lfc = pd.DataFrame(M - B, index=g, columns=genes).groupby(level=0).mean()
    cpm_m = pd.DataFrame(np.expm1(M), index=g).groupby(level=0).mean().mean(0).to_numpy()
    cpm_b = pd.DataFrame(np.expm1(B), index=g).groupby(level=0).mean().mean(0).to_numpy()
    tab = pd.DataFrame({"gene": np.asarray(genes),
                        "lfc_median": lfc.median(0).to_numpy(),
                        "frac_up": (lfc > min_lfc).mean(0).to_numpy(),
                        "frac_down": (lfc < -min_lfc).mean(0).to_numpy(),
                        "cpm_mal": cpm_m, "cpm_ben": cpm_b})
    ok = ~program_excluded(tab["gene"])
    up = tab[ok & (tab["frac_up"] >= min_frac) & (tab["cpm_mal"] >= min_cpm)
             & (tab["lfc_median"] > 0)].nlargest(n_up, "lfc_median").index
    dn = tab[ok & (tab["frac_down"] >= min_frac) & (tab["cpm_ben"] >= min_cpm)
             & (tab["lfc_median"] < 0)].nsmallest(n_down, "lfc_median").index
    tab["direction"] = ""
    tab.loc[up, "direction"] = "up"
    tab.loc[dn, "direction"] = "down"
    tab.attrs["n_donors"] = lfc.shape[0]
    return tab


def program_score(lc, genes, up, down):
    """Mean log1p(CPM) of `up` minus mean of `down`, inside each cluster's own pseudobulk.

    Genes are compared with genes within one cluster, so the score needs no benign reference in
    the same sample (Sezary-heavy samples have none).
    """
    gi = pd.Index(genes)
    u, d = gi.get_indexer(list(up)), gi.get_indexer(list(down))
    u, d = u[u >= 0], d[d >= 0]
    s = lc[:, u].mean(1) if len(u) else np.zeros(lc.shape[0])
    return s - (lc[:, d].mean(1) if len(d) else 0.0)


def vskew(cells, col, min_calls=10, n_sim=500, seed=SEED):
    """Top-V share per cluster vs the max share a polyclonal repertoire of that size would give.

    Reference usage = mean of per-sample V-call frequencies, so no single tumour clone dominates
    it. z = (observed top share - null mean) / null sd, NaN under `min_calls` V-called cells.
    """
    c = cells.loc[cells[col].astype(str) != "", ["sample_id", "cluster", col]]
    per_sample = pd.crosstab(c["sample_id"], c[col])
    per_sample = per_sample[per_sample.sum(1) >= 20]
    ref = per_sample.div(per_sample.sum(1), axis=0).mean(0)
    tab = pd.crosstab(c["cluster"], c[col]).reindex(columns=ref.index, fill_value=0)
    n = tab.sum(1).to_numpy()
    share = tab.max(1).to_numpy() / np.maximum(n, 1)
    rng, p, cache = np.random.default_rng(seed), ref.to_numpy() / ref.sum(), {}
    for k in np.unique(n[n >= min_calls]):
        draws = rng.multinomial(int(k), p, size=n_sim).max(1) / k
        cache[k] = (draws.mean(), draws.std() + 1e-6)
    z = np.array([(s - cache[k][0]) / cache[k][1] if k in cache else np.nan
                  for s, k in zip(share, n)])
    return pd.DataFrame({"top_v": tab.idxmax(1).to_numpy(), "top_v_share": share,
                         "n_vcalls": n, "v_z": z}, index=tab.index)


# ------------------------------------------------------------------ model + cutoffs (notebook)
def fit_predict(feats, cols, train, pred, y, w):
    """Fit a logistic model on `train` rows, return P(malignant) on `pred` rows (NaN elsewhere).

    When `v_z` is combined with other features, clusters without a V skew fall back to a model on
    the other features alone. A missing-value flag instead extrapolates badly: li2024, the largest
    training study, has no V reads at all, so held out it is all "missing" while training is not.
    """
    p = np.full(len(feats), np.nan)
    ok = feats[cols].notna().all(1).to_numpy()
    parts = [(cols, ok, ok)]                                  # (features, train rows, predict rows)
    if "v_z" in cols and len(cols) > 1:
        rest = [c for c in cols if c != "v_z"]
        ok_rest = feats[rest].notna().all(1).to_numpy()
        parts.append((rest, ok_rest, ok_rest & feats["v_z"].isna().to_numpy()))
    for cc, ok_tr, ok_pr in parts:
        tr, pr = train & ok_tr, pred & ok_pr
        if pr.any() and len(np.unique(y[tr])) == 2:
            m = fit_model(feats.loc[tr, cc], y[tr], w[tr])
            p[pr] = m.predict_proba(feats.loc[pr, cc].to_numpy())[:, 1]
    return p


def fit_model(F, y, w):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    m.fit(F.to_numpy(), y, logisticregression__sample_weight=w)
    return m


def pick_cutoffs(p, y, w, max_err=0.02):
    """p_hi / p_lo so that <= `max_err` of the cells auto-called on each side are wrong.

    p_hi: lowest cutoff whose clusters with p >= p_hi hold <= max_err benign cells (largest such
    set, step-up); p_lo mirrored for malignant cells among p <= p_lo. NaN p never counts as clear.
    """
    ok = ~np.isnan(p)
    p, y, w = p[ok], np.asarray(y)[ok].astype(bool), np.asarray(w, float)[ok]

    def _side(order, wrong):
        err = np.cumsum(w[order] * wrong[order]) / np.cumsum(w[order])
        k = np.flatnonzero(err <= max_err)
        return p[order][k.max()] if len(k) else None

    hi = _side(np.argsort(-p, kind="stable"), ~y)
    lo = _side(np.argsort(p, kind="stable"), y)
    p_hi = 1.0 + 1e-9 if hi is None else hi
    p_lo = -1e-9 if lo is None else lo
    if p_lo >= p_hi:                                   # sides overlap: one cut, everything clear
        p_lo = p_hi = (p_lo + p_hi) / 2
        p_lo -= 1e-12
    return float(p_lo), float(p_hi)


def evaluate(p, y, w, sample, p_lo, p_hi):
    """Held-out summary of one candidate over labelled clusters (NaN p = not clear)."""
    from sklearn.metrics import roc_auc_score

    def _auc(yy, pp, ww=None):
        return roc_auc_score(yy, pp, sample_weight=ww) if 0 < yy.sum() < len(yy) else np.nan

    y, w, sample = np.asarray(y).astype(bool), np.asarray(w, float), np.asarray(sample)
    ok = ~np.isnan(p)
    hi, lo = ok & (p >= p_hi), ok & (p <= p_lo)
    within = [roc_auc_score(y[i], p[i]) for i in pd.Series(sample).groupby(sample).indices.values()
              if ok[i].all() and 0 < y[i].sum() < len(i)]
    return {
        "coverage": ok.mean(),
        "auc_cluster": _auc(y[ok], p[ok]),
        "auc_cells": _auc(y[ok], p[ok], w[ok]),
        "auc_within_median": np.median(within) if within else np.nan,
        "n_within": len(within),
        "auto_cells_frac": w[hi | lo].sum() / w.sum(),
        "err_mal": w[hi & ~y].sum() / max(w[hi].sum(), 1),
        "err_ben": w[lo & y].sum() / max(w[lo].sum(), 1),
        "p_lo": p_lo, "p_hi": p_hi,
    }


# ------------------------------------------------------------------ review (notebook)
STATUS_COLORS = {"malignant": "#c0392b", "benign": "#2e86c1", "review": "#f39c12"}


def review_figure(sid, cells, k, expr, pct, p_lo, p_hi, breaks=(), out=None):
    """One review sheet per sample.

    cells: this sample's cells (umap_1, umap_2, cluster key). k: its clusters, indexed by cluster
    key, with cluster_idx, n_cells, prob, status in {malignant, benign, review}, top_v, top_v_share,
    and optionally n_tcr / n_family / alice_ok (ALICE family share is printed where alice_ok).
    expr / pct: (cluster key x genes) log1p(CPM) and fraction of cells detecting, for the dotplot;
    `breaks` = column positions of dashed separators (up | down | markers).
    """
    import matplotlib.pyplot as plt

    k = k.sort_values("cluster_idx")
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_u, ax_p, ax_b = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_d = fig.add_subplot(gs[1, :])

    st = cells["cluster"].map(k["status"])
    ax_u.scatter(cells["umap_1"], cells["umap_2"], s=3, c=st.map(STATUS_COLORS).fillna("#bbb"),
                 linewidths=0, rasterized=True)
    pr = cells["cluster"].map(k["prob"])
    sca = ax_p.scatter(cells["umap_1"], cells["umap_2"], s=3, c=pr, cmap="viridis", vmin=0,
                       vmax=1, linewidths=0, rasterized=True)
    fig.colorbar(sca, ax=ax_p, fraction=0.04, label="P(malignant)")
    med = cells.groupby("cluster", observed=True)[["umap_1", "umap_2"]].median()
    for key, (x, yv) in med.iterrows():
        for ax in (ax_u, ax_p):
            ax.text(x, yv, str(k.loc[key, "cluster_idx"]), fontsize=9, weight="bold",
                    ha="center", va="center")
    for ax, t in ((ax_u, "call (orange = you decide)"), (ax_p, "cluster score")):
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    x = np.arange(len(k))
    ax_b.bar(x, k["prob"].fillna(0), color=k["status"].map(STATUS_COLORS))
    for yv in (p_lo, p_hi):
        ax_b.axhline(yv, ls="--", c="k", lw=0.8)
    for xi, (_, r) in zip(x, k.iterrows()):
        txt = [f"n={r['n_cells']}"]
        if pd.notna(r.get("top_v_share")):
            txt.append(f"{r['top_v']} {r['top_v_share']:.0%}")
        if r.get("alice_ok", False) and r.get("n_tcr", 0) > 0:
            txt.append(f"ALICE {int(r['n_family'])}/{int(r['n_tcr'])}")
        ax_b.text(xi, min(float(np.nan_to_num(r["prob"])) + 0.02, 1.02), "\n".join(txt),
                  rotation=90, fontsize=6, ha="center", va="bottom")
    ax_b.set_xticks(x, k["cluster_idx"].astype(str))
    ax_b.set_ylim(0, 1.5)
    ax_b.set_xlabel("cluster")
    ax_b.set_ylabel("P(malignant)")
    ax_b.set_title("score per cluster (dashed = auto cutoffs)", fontsize=10)

    e, f = expr.loc[k.index], pct.loc[k.index]
    z = (e - e.mean(0)) / (e.std(0) + 1e-6)
    gx, gy = np.meshgrid(np.arange(e.shape[1]), np.arange(e.shape[0]))
    ax_d.scatter(gx.ravel(), gy.ravel(), s=(f.to_numpy().ravel() * 250) + 1,
                 c=z.to_numpy().ravel(), cmap="RdBu_r", vmin=-2, vmax=2, linewidths=0)
    for b in breaks:
        ax_d.axvline(b - 0.5, ls="--", c="grey", lw=0.8)
    ax_d.set_xticks(np.arange(e.shape[1]), e.columns, rotation=90, fontsize=7)
    ax_d.set_yticks(np.arange(e.shape[0]),
                    [f"{c} ({s[0]})" for c, s in zip(k["cluster_idx"], k["status"])], fontsize=7)
    ax_d.invert_yaxis()
    ax_d.set_title("program genes (up | down) + lineage markers; colour = z across clusters, "
                   "size = % cells", fontsize=10)
    n_rev = int(k["status"].eq("review").sum())
    fig.suptitle(f"{sid}: {n_rev} of {len(k)} clusters need a call", fontsize=12)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=110)
        plt.close(fig)
    return fig


_CALL_MAP = {"malignant": True, "m": True, "1": True, "true": True, "mal": True,
             "benign": False, "b": False, "0": False, "false": False, "ben": False}


def parse_user_call(s):
    """'malignant'/'m'/'1' -> True, 'benign'/'b'/'0' -> False, anything else -> NA."""
    return s.astype(str).str.strip().str.lower().map(_CALL_MAP).astype("boolean")


def clone_v(cl, z_clone=Z_CLONE, z_share=Z_SHARE, share_min=SHARE_MIN):
    """Per cluster: the sample's clone V gene and whether the cluster carries it (see Z_CLONE)."""
    seed = cl[(cl["v_z"] >= z_clone) & (cl["frac_CD8"] < 0.5)]
    cv = seed.sort_values("v_z").groupby("sample_id")["top_v"].last()
    v = cl["sample_id"].map(cv)
    return v, cl["top_v"].eq(v) & (cl["v_z"] >= z_share) & (cl["top_v_share"] >= share_min)


# ---- cluster-level cell-type QC -------------------------------------------------------------
# An NK cell is CD8A+ CD8B-, and a CD8 T cell is not. nb31 labelled on CD8A alone, so cytotoxic
# NK clusters came through as CD8. The pair of scores below separates them on the cluster
# pseudobulk; `n_tcr / n_cells` is the independent check, because an NK cluster has no TRB.
T_SCORE_GENES = ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC2", "CD2"]
NK_SCORE_GENES = ["KLRF1", "TYROBP", "FCER1G", "NCAM1", "KLRD1", "FCGR3A", "GNLY", "NKG7"]
NK_DELTA, NK_MAX_TCR = 1.0, 0.15    # nk_score - t_score, and TCR recovery rate


def nk_vs_t_scores(pseudobulk_h5ad=None):
    """Per cluster: mean log1p CP10K over the T and NK marker sets, plus the TCR recovery rate.

    `delta = nk - t` and `tcr_rate` are read together: delta alone would move a genuinely
    cytotoxic CD8 T cluster, and tcr_rate alone would move a cluster that simply lost its V(D)J.
    """
    import h5py
    import scipy.sparse as sp
    path = Path(pseudobulk_h5ad or PSEUDOBULK_H5AD)
    with h5py.File(path, "r") as f:
        gi, vi = f["obs"].attrs["_index"], f["var"].attrs["_index"]
        names = pd.Index(_read_elem(f["obs"][gi])).astype(str)
        genes = pd.Index(_read_elem(f["var"][vi])).astype(str)
        obs = pd.DataFrame({c: _read_elem(f["obs"][c]) for c in ("n_cells", "n_tcr", "frac_CD4",
                                                                 "frac_CD8", "frac_NK")},
                           index=names)
        g = f["X"]
        X = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                          shape=tuple(int(v) for v in g.attrs["shape"]))
    lib = np.asarray(X.sum(1)).ravel()

    def score(gene_list):
        cols = [genes.get_loc(x) for x in gene_list if x in set(genes)]
        return np.log1p(X[:, cols].toarray() / lib[:, None] * 1e4).mean(1)

    out = obs.copy()
    out["t_score"] = score(T_SCORE_GENES)
    out["nk_score"] = score(NK_SCORE_GENES)
    out["delta"] = out["nk_score"] - out["t_score"]
    out["tcr_rate"] = out["n_tcr"] / out["n_cells"].clip(lower=1)
    out["nk_like"] = (out["delta"] >= NK_DELTA) & (out["tcr_rate"] < NK_MAX_TCR)
    return out


def finalize_calls(clusters, cells, review_csv=None, exclude_csv=None):
    """Merge manual calls into the cluster calls and expand to cells.

    clusters: indexed by cluster key, with source, call, prob (as written to CLUSTERS_CSV). A filled
    `user_call` overrides whatever the cluster had; open review rows become `review_pending`
    (mal_cluster undefined); every cluster of a sample in `exclude_csv` becomes `excluded_user`
    (undefined), whatever else it had. Returns (per-cell frame for CALLS_PQ, updated cluster frame).
    """
    review_csv, exclude_csv = Path(review_csv or REVIEW_CSV), Path(exclude_csv or EXCLUDE_CSV)
    final = clusters.copy()
    final["call"] = final["call"].astype("boolean")
    if review_csv.exists():
        u = pd.read_csv(review_csv, index_col="cluster", dtype=str, keep_default_na=False)
        uc = parse_user_call(u["user_call"]).dropna()
        uc = uc[uc.index.isin(final.index)]
        final.loc[uc.index, "call"] = uc
        final.loc[uc.index, "source"] = "user"
    final.loc[final["source"].eq("review"), "source"] = "review_pending"
    if exclude_csv.exists():
        gone = final["sample_id"].isin(pd.read_csv(exclude_csv)["sample_id"].astype(str))
        final.loc[gone, "source"] = "excluded_user"
        final.loc[gone, "call"] = pd.NA
    out = (cells[["cell_id", "sample_id", "cluster"]]
           .join(final[["source", "call", "prob"]], on="cluster")
           .rename(columns={"source": "call_source", "call": "mal_cluster",
                            "prob": "mal_cluster_prob"}))
    out["mal_cluster"] = out["mal_cluster"].astype("boolean")
    return out, final
