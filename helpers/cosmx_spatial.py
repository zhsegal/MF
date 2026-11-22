"""CosMx early-MF spatial layer (GSE337336) for the in-situ degradome notebooks (28d, 28e).

The dataset is 16 MF skin biopsies (8 patch, 8 plaque) on 4 CosMx slides, 1000-plex, 65,717 cells,
shipped only as per-sample flat CSVs. Two properties decide how anything here may be used:

1. The GEO `patient` label is not a biopsy label. Within a slide, a cell's nearest neighbour
   carries the same patient label no more often than random labelling would (`patient_mixing`),
   and patients 1-5 tile the same FOVs. Expression and coordinates *are* coherent (KRT14 and PanCK
   are strongly spatially autocorrelated), so the tissue is real and only the per-patient split is
   broken. The unit of inference is therefore the `piece` -- a spatially contiguous group of FOVs
   on one slide -- nested in `slide`.
2. Stage is a slide property: slides 3/4 are patch, 7/8 plaque. Patch vs plaque is 2 vs 2 slides,
   so it is described, never tested.

Neighbourhoods are radius graphs on slide-global micrometre coordinates, built per slide, so they
may cross adjacent FOVs but never slides.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

NB_DIR = Path(__file__).resolve().parent.parent
S1_DIR = NB_DIR / "data" / "S1_cosmx_earlymf_spatial"
RAW_DIR = S1_DIR / "raw"
RAW_TAR = RAW_DIR / "GSE337336_RAW.tar"
CACHE = S1_DIR / "processed" / "cosmx_s1.h5ad"
N_CELLS_EXPECTED = 65_717

META_KEEP = ["fov", "x_FOV_px", "y_FOV_px", "x_slide_mm", "y_slide_mm", "Area", "AspectRatio",
             "nCount_RNA", "nFeature_RNA", "Mean.PanCK", "Mean.CD68", "Mean.CD45", "Mean.DAPI",
             "nCount_negprobes", "nCount_falsecode", "propNegative", "complexity", "stim", "patient",
             "InSituTypeIDs2"]
FOV_PX = 4256       # CosMx FOV is 4256 × 4256 px at 0.12028 µm/px ≈ 0.512 mm square

# Lineage marker sets, intersected with the panel at call time. No melanocyte set: PMEL, MLANA,
# TYRP1, DCT, MITF and SOX10 are all off the 1000-plex panel, so melanocytes cannot be called.
LINEAGE_MARKERS = {
    "Keratinocyte": ["KRT5", "KRT14", "KRT10", "KRT1", "KRT15"],
    "Fibroblast":   ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Endothelial":  ["PECAM1", "VWF", "CDH5", "LYVE1", "PROX1"],
    "Pericyte_SMC": ["ACTA2", "RGS5", "MYH11", "TAGLN"],
    "T":            ["CD3E", "CD3D", "CD3G", "CD2"],
    "Myeloid":      ["CD68", "CD163", "LYZ", "C1QA", "C1QB", "CD14", "HLA-DRA"],
    "B":            ["MS4A1", "CD79A", "CD19"],
    "Plasma":       ["MZB1", "JCHAIN", "IGKC", "IGHG1"],
    "Mast":         ["TPSAB1/B2", "CPA3", "KIT"],
}
IMMUNE = {"T", "Myeloid", "B", "Plasma", "Mast"}

LINEAGE_COLORS = {
    "Keratinocyte": "#c9a66b", "Fibroblast": "#3a78b5", "Endothelial": "#2ca02c",
    "Pericyte_SMC": "#17becf", "T": "#d62728", "Myeloid": "#f28e2b", "B": "#e377c2",
    "Plasma": "#8c564b", "Mast": "#9467bd", "Unassigned": "#d9d9d9",
}
# Both are early (T1-T2) lesion types, so these are deliberately not 28b's early/advanced colours.
STAGE_COLORS = {"patch": "#e8a33d", "plaque": "#8c510a"}

GATE_MIN_ENRICH = 2.0
GATE_MIN_DET = 0.15

# 28e gene sets (intersected with the panel at use). The 28c amoeboid programs are off this panel
# (of 37 genes only AQP3 and LMNA are measured), so 28e reads position / shape proxies instead.
RETENTION = ["ITGAE", "CD69", "CXCR3", "CCR10", "CXCR6"]      # epidermal residence / homing
EGRESS = ["KLF2", "SELL", "CCR7", "CXCR4"]                    # recirculation / exit
LAMINA = ["LMNA"]                                             # 28c lamina ratio: LMNB1/2 off panel
MALIG_APRIORI = ["GATA3", "TOX", "TWIST1", "DPP4", "CCR7", "CXCR4", "SELL", "IL7R"]
T_SPILL = ["CD3E", "CD2"]                                     # T genes read in stroma: T spillover
KER_SPILL = ["KRT14", "KRT5"]                                 # keratinocyte genes read in T cells
AXIS_PAIRS = [("TNFSF14", "LTBR"), ("TNFSF14", "TNFRSF14"), ("CD40LG", "CD40"),
              ("TGFB1", "TGFBR2"), ("ITGAE", "CDH1"), ("CXCL10", "CXCR3"), ("CXCL9", "CXCR3")]
RESOLVI_H5AD = S1_DIR / "processed" / "cosmx_s1_resolvi.h5ad"
CELLS_28D = S1_DIR / "processed" / "cosmx_s1_cells_28d.parquet"
CELLS_28E = S1_DIR / "processed" / "cosmx_s1_cells_28e.parquet"


# ================================================================ ingest


def _sample_table(patients=None):
    """One row per GSM: its three CSVs, from the loose files or from inside RAW.tar."""
    with tarfile.open(RAW_TAR) as tf:
        in_tar = tf.getnames()
    pat = re.compile(r"(GSM\d+)_MF_Patient(\d+)_(patch|plaque)_(exprMat|metadata)_file\.csv\.gz")
    rows = {}
    for name in in_tar:
        m = pat.fullmatch(name)
        if m:
            gsm, pid, stage, kind = m.groups()
            rows.setdefault(gsm, {"gsm": gsm, "patient": int(pid), "stage": stage})[kind] = name
    tab = pd.DataFrame(rows.values()).sort_values("patient").reset_index(drop=True)
    tab["loose"] = [(RAW_DIR / e).exists() and (RAW_DIR / m).exists()
                    for e, m in zip(tab.exprMat, tab.metadata)]
    if patients is not None:
        tab = tab[tab.patient.isin(patients)].reset_index(drop=True)
    return tab


def _read_csv_gz(name, loose, tf, **kw):
    if loose:
        return pd.read_csv(RAW_DIR / name, **kw)
    return pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(name).read())), **kw)


def load_s1(cache=True, patients=None, min_counts=0, verbose=True):
    """AnnData of GSE337336: `layers['counts']` raw, `obsm['spatial']` in slide-global µm.

    Samples not extracted next to the tar (Patient16) are read from inside it; nothing is written
    to raw/. `cache` reads/writes CACHE and only applies to the full cohort.
    """
    import anndata as ad

    full = patients is None
    adata = None
    if cache and full and CACHE.exists():
        adata = ad.read_h5ad(CACHE)
        if not {"x_FOV_px", "y_FOV_px"} <= set(adata.obs.columns):   # cache from an older META_KEEP
            adata = None
    if adata is None:
        tab = _sample_table(patients)
        mats, metas, genes = [], [], None
        with tarfile.open(RAW_TAR) as tf:
            for r in tab.itertuples():
                meta = _read_csv_gz(r.metadata, r.loose, tf, usecols=["cell_ID", *META_KEEP])
                ex = _read_csv_gz(r.exprMat, r.loose, tf).set_index("cell_id")
                ex = ex.reindex(meta.cell_ID)
                assert not ex.isna().any().any(), f"{r.gsm}: metadata cells missing from exprMat"
                if genes is None:
                    genes = ex.columns
                assert ex.columns.equals(genes), f"{r.gsm}: gene panel differs"
                mats.append(sp.csr_matrix(ex.to_numpy(dtype=np.int32)))
                metas.append(meta.assign(gsm=r.gsm))
                if verbose:
                    print(f"  {r.gsm} P{r.patient:02d} {r.stage:6s} {len(meta):6,d} cells"
                          f"{'' if r.loose else '  (read from tar)'}")
        obs = pd.concat(metas, ignore_index=True)
        assert obs.cell_ID.is_unique
        obs["slide"] = obs.cell_ID.str.split("_").str[1]
        obs = obs.rename(columns={"stim": "stage", "patient": "patient_geo",
                                  "InSituTypeIDs2": "insitutype"})
        obs["fov_key"] = obs.slide + "_" + obs.fov.astype(str)
        obs["piece"] = assign_pieces(obs)
        obs = obs.set_index("cell_ID")
        for c in ["slide", "stage", "insitutype", "fov_key", "piece", "gsm"]:
            obs[c] = obs[c].astype("category")
        obs["patient_geo"] = obs.patient_geo.astype(str).astype("category")
        X = sp.vstack(mats).tocsr()
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=genes.astype(str)))
        adata.layers["counts"] = X
        adata.obsm["spatial"] = obs[["x_slide_mm", "y_slide_mm"]].to_numpy() * 1000.0
        adata.uns["s1_note"] = ("GSE337336 CosMx 1000-plex. patient_geo is the GEO label and is NOT "
                                "a biopsy label (spatially interleaved). Unit = piece within slide.")
        if full:
            assert adata.n_obs == N_CELLS_EXPECTED, adata.n_obs
            if cache:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(CACHE, compression="gzip")
    if min_counts:
        adata = adata[adata.obs.nCount_RNA >= min_counts].copy()
    return adata


def assign_pieces(obs, gap_um=1000.0):
    """Single-linkage over FOV centroids within each slide; FOVs closer than `gap_um` share a piece."""
    cen = obs.groupby(["slide", "fov"], observed=True)[["x_slide_mm", "y_slide_mm"]].mean() * 1000.0
    lab = {}
    for s, g in cen.groupby(level="slide"):
        k = (fcluster(linkage(g.values, "single"), gap_um, "distance")
             if len(g) > 1 else np.array([1]))
        for (sl, fov), kk in zip(g.index, k):
            lab[(sl, fov)] = f"{sl}_p{kk:02d}"
    return pd.Series([lab[(s, f)] for s, f in zip(obs.slide, obs.fov)], index=obs.index)


def patient_mixing(adata):
    """Per slide × FOV: fraction of cells whose nearest neighbour has the same GEO patient label,
    against the random-labelling expectation sum(p_i^2). Equal values = labels carry no space."""
    rows = []
    obs = adata.obs
    for (s, f), idx in obs.groupby(["slide", "fov"], observed=True).indices.items():
        if len(idx) < 20:
            continue
        _, nn = cKDTree(adata.obsm["spatial"][idx]).query(adata.obsm["spatial"][idx], k=2)
        p = obs.patient_geo.to_numpy()[idx]
        rows.append({"slide": s, "fov": f, "n": len(idx), "n_patients": len(set(p)),
                     "nn_same": (p[nn[:, 1]] == p).mean(),
                     "random": (pd.Series(p).value_counts(normalize=True) ** 2).sum()})
    return pd.DataFrame(rows)


# ================================================================ layout (for maps)


def fov_boxes(adata):
    """True FOV squares in slide µm, one row per slide × FOV.

    GEO gives no FOV offsets, but every cell carries both its FOV-pixel and slide-mm position, so
    the offset is recovered per FOV from those pairs (slope = the instrument's µm/px, shared by all
    FOVs; the FOV y-pixel axis runs opposite to slide y). The box spans the full 4256-px FOV, not
    just its cells, so tissue-free parts of an imaged FOV are visible as empty.
    """
    obs = adata.obs
    x_um, y_um = obs.x_slide_mm.to_numpy() * 1000, obs.y_slide_mm.to_numpy() * 1000
    xp, yp = obs.x_FOV_px.to_numpy(float), obs.y_FOV_px.to_numpy(float)
    grp = obs.groupby(["slide", "fov"], observed=True)
    ctr = lambda v, col: v - grp[col].transform("mean").to_numpy() * (1000 if col.endswith("mm") else 1)
    bx = np.polyfit(ctr(xp, "x_FOV_px"), ctr(x_um, "x_slide_mm"), 1)[0]      # within-FOV slope
    by = np.polyfit(ctr(yp, "y_FOV_px"), ctr(y_um, "y_slide_mm"), 1)[0]
    d = pd.DataFrame({"slide": obs.slide.astype(str).to_numpy(), "fov": obs.fov.to_numpy(),
                      "piece": obs.piece.astype(str).to_numpy(), "ox": x_um - bx * xp, "oy": y_um - by * yp})
    b = d.groupby(["slide", "fov", "piece"], observed=True)[["ox", "oy"]].median().reset_index()
    ends_x = np.stack([b.ox, b.ox + bx * FOV_PX]); ends_y = np.stack([b.oy, b.oy + by * FOV_PX])
    b["x0"], b["x1"] = ends_x.min(0), ends_x.max(0)
    b["y0"], b["y1"] = ends_y.min(0), ends_y.max(0)
    b.attrs["um_per_px"] = (bx, by)
    return b.drop(columns=["ox", "oy"])


def pack_pieces(boxes, pad_um=150.0):
    """Per slide, offsets that lay its pieces side by side at true scale (shelf packing, tallest
    first) so a map shows every imaged region without the empty slide between them. Positions
    between pieces are not preserved; the whole-slide view is `fov_boxes` on slide coordinates.
    Returns one row per piece: dx, dy (add to a cell's µm coordinates) and the packed box."""
    rows = []
    for s, g in boxes.groupby("slide"):
        pb = g.groupby("piece").agg(x0=("x0", "min"), x1=("x1", "max"), y0=("y0", "min"), y1=("y1", "max"))
        pb["w"], pb["h"] = pb.x1 - pb.x0, pb.y1 - pb.y0
        pb = pb.sort_values("h", ascending=False)
        width = max(pb.w.max(), np.sqrt(((pb.w + pad_um) * (pb.h + pad_um)).sum()) * 1.15)
        x = y = row_h = 0.0
        for p, r in pb.iterrows():
            if x > 0 and x + r.w > width:
                y -= row_h + pad_um; x = row_h = 0.0
            rows.append({"slide": s, "piece": p, "dx": x - r.x0, "dy": (y - r.h) - r.y0,
                         "px0": x, "px1": x + r.w, "py0": y - r.h, "py1": y})
            x += r.w + pad_um; row_h = max(row_h, r.h)
    return pd.DataFrame(rows)


# ================================================================ annotation


def lognorm(adata, layer="counts"):
    """log1p of counts scaled to the median library size; the X every display uses."""
    C = adata.layers[layer].astype(np.float32)
    lib = np.asarray(C.sum(1)).ravel()
    C = sp.diags(np.median(lib) / np.maximum(lib, 1)) @ C
    return C.log1p().tocsr()


def lineage_scores(adata, markers=LINEAGE_MARKERS):
    """Per cell: mean lognorm over each lineage's panel markers minus the cell's panel-wide mean."""
    X = adata.X
    base = np.asarray(X.mean(1)).ravel()
    out, used = {}, {}
    for lin, gs in markers.items():
        gs = [g for g in gs if g in adata.var_names]
        used[lin] = gs
        if gs:
            out[lin] = np.asarray(X[:, adata.var_names.get_indexer(gs)].mean(1)).ravel() - base
    return pd.DataFrame(out, index=adata.obs_names), used


def call_clusters(adata, scores, cluster_key="leiden", min_margin=0.05):
    """Cluster → lineage by top mean score, with the stain means beside it for the manual check.

    `call` is 'Unassigned' when the top lineage wins by less than `min_margin`, or when the stain
    contradicts it. Raw stain intensities differ several-fold between slides, and slide
    composition differs too (slide 3 is mostly epidermis), so each stain is divided by the median
    of that slide's stromal cells -- cells in clusters whose top is Fibroblast / Endothelial /
    Pericyte_SMC (`*_rel`, stroma = 1). Keratinocyte needs PanCK_rel > 3; immune needs
    CD45_rel > 1 (myeloid may pass on CD68_rel > 1 instead).

    `flag_epi_immune`: a keratinocyte call whose CD45_rel reaches the T clusters' median. In MF
    these are likely intraepidermal T cells carrying keratinocyte spillover (epidermotropism).
    Not auto-reassigned; the notebook reviews the table and may override.
    """
    obs = adata.obs
    S = scores.groupby(obs[cluster_key].to_numpy()).mean()
    top2 = np.sort(S.to_numpy(), 1)[:, -2:]
    tab = pd.DataFrame({"n": obs[cluster_key].value_counts().reindex(S.index).to_numpy(),
                        "top": S.idxmax(1).to_numpy(), "margin": top2[:, 1] - top2[:, 0]},
                       index=S.index)
    grp = obs[cluster_key].to_numpy()
    in_top = lambda lins: pd.Series(grp).isin(tab.index[tab.top.isin(lins)]).to_numpy()
    stroma = in_top(["Fibroblast", "Endothelial", "Pericyte_SMC"])
    rel = pd.DataFrame(index=obs.index)
    for c in ["Mean.PanCK", "Mean.CD45", "Mean.CD68"]:
        ref_s = obs.loc[stroma, c].groupby(obs.slide.to_numpy()[stroma]).median()
        ref_c = obs.slide.astype(str).map(ref_s.rename(index=str)).fillna(obs.loc[stroma, c].median())
        rel[c.split(".")[1] + "_rel"] = obs[c] / ref_c.to_numpy()
    tab = tab.join(rel.groupby(grp).median())
    tab["nCount"] = obs.groupby(cluster_key, observed=True).nCount_RNA.median()
    tab["top_slide_frac"] = obs.groupby(cluster_key, observed=True).slide.agg(
        lambda x: x.value_counts(normalize=True).iloc[0])
    tab["insitutype_mode"] = obs.groupby(cluster_key, observed=True).insitutype.agg(
        lambda x: x.value_counts().index[0])
    t_cd45 = rel.CD45_rel[in_top(["T"])].median()
    immune_ok = (tab.CD45_rel > 1) | ((tab.top == "Myeloid") & (tab.CD68_rel > 1))
    bad = ((tab.margin < min_margin)
           | ((tab.top == "Keratinocyte") & (tab.PanCK_rel < 3))
           | (tab.top.isin(IMMUNE) & ~immune_ok))
    tab["call"] = np.where(bad, "Unassigned", tab.top)
    tab["flag_epi_immune"] = (tab.call == "Keratinocyte") & (tab.CD45_rel >= t_cd45)
    return tab.join(S.add_prefix("s_")).sort_values(["call", "n"], ascending=[True, False])


def t_subset(adata):
    """Count-based CD4/CD8 split of T cells; most stay T_other at this depth. Descriptive only."""
    C = adata.layers["counts"]
    get = lambda g: np.asarray(C[:, adata.var_names.get_loc(g)].todense()).ravel() if g in adata.var_names else 0
    cd4, cd8 = get("CD4"), get("CD8A") + get("CD8B")
    sub = np.select([(cd8 > cd4), (cd4 > cd8)], ["T_CD8", "T_CD4"], "T_other")
    return pd.Series(np.where(adata.obs.lineage.to_numpy() == "T", sub, ""), index=adata.obs_names)


# ================================================================ gene gate


def gene_gate(adata, genes, lineage_key="lineage", exclude=("Unassigned",)):
    """Is the gene distinguishable from the ~7-10% ambient floor in its top lineage?

    Means are on library-normalised counts (scaled to the median nCount): keratinocytes carry
    ~2.5× the counts of other cells, so raw means would make them the top lineage of every gene.
    enrichment = top-lineage mean / median of lineage means. Pass: enrichment >= 2 and raw
    detection >= 0.15 in the top lineage.
    """
    genes = [g for g in genes if g in adata.var_names]
    keep = ~adata.obs[lineage_key].isin(exclude).to_numpy()
    C = adata.layers["counts"][keep][:, adata.var_names.get_indexer(genes)].toarray()
    lib = adata.obs.nCount_RNA.to_numpy()[keep]
    lab = adata.obs[lineage_key].to_numpy()[keep]
    df = pd.DataFrame(C * (np.median(lib) / lib)[:, None], columns=genes)
    mean, det = df.groupby(lab).mean(), (df > 0).groupby(lab).mean()
    top = mean.idxmax()
    rows = pd.DataFrame({
        "top_lineage": top,
        "top_mean": [mean.loc[top[g], g] for g in genes],
        "top_det": [det.loc[top[g], g] for g in genes],
        "median_mean": mean.median(),
        "floor_det": det.min(),
    })
    rows["enrichment"] = rows.top_mean / rows.median_mean.clip(lower=1e-3)
    rows["pass"] = (rows.enrichment >= GATE_MIN_ENRICH) & (rows.top_det >= GATE_MIN_DET)
    return rows, mean, det


# ================================================================ neighbourhoods


def neighbors(adata, radius_um, key="spatial"):
    """Symmetric radius graph (no self edges) built per slide; CSR, n_obs × n_obs, 0/1."""
    xy = adata.obsm[key]
    rows, cols = [], []
    for _, idx in adata.obs.groupby("slide", observed=True).indices.items():
        pairs = cKDTree(xy[idx]).query_pairs(radius_um, output_type="ndarray")
        if len(pairs):
            a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
            rows += [a, b]; cols += [b, a]
    r = np.concatenate(rows) if rows else np.array([], int)
    c = np.concatenate(cols) if cols else np.array([], int)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(adata.n_obs, adata.n_obs))


def nbr_composition(G, labels):
    """Neighbour counts per label (DataFrame n_obs × labels) and the total neighbour count."""
    lab = pd.Categorical(labels)
    M = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab.codes)),
                      shape=(len(lab), len(lab.categories)))
    cnt = pd.DataFrame((G @ M).toarray(), columns=lab.categories)
    return cnt, cnt.sum(1)


def smooth(G, values):
    """Mean of a per-cell value over the cell and its graph neighbours."""
    v = np.asarray(values, float)
    return (G @ v + v) / (np.asarray(G.sum(1)).ravel() + 1)


def dist_to_lineage(adata, lineage="Keratinocyte", lineage_key="lineage", min_cells=50):
    """µm from each cell to the nearest `lineage` cell in the same piece (0 for those cells).
    NaN in pieces holding fewer than `min_cells` of them -- no epidermis captured, no depth."""
    out = np.full(adata.n_obs, np.nan)
    xy = adata.obsm["spatial"]
    is_l = adata.obs[lineage_key].to_numpy() == lineage
    for _, idx in adata.obs.groupby("piece", observed=True).indices.items():
        tgt = idx[is_l[idx]]
        if len(tgt) < min_cells:
            continue
        d, _ = cKDTree(xy[tgt]).query(xy[idx])
        out[idx] = np.where(is_l[idx], 0.0, d)
    return out


def nhood_enrichment(G, labels, groups, n_perm=500, seed=0):
    """Edge counts between labels vs label permutations within `groups` (pieces). Returns
    (z, observed, expected) as label × label DataFrames. Shuffling within pieces keeps each
    piece's composition, so these measure arrangement, not abundance. z grows with group size;
    compare groups of different size on log2(observed / expected)."""
    rng = np.random.default_rng(seed)
    lab = pd.Categorical(labels)
    codes, k, n = lab.codes.copy(), len(lab.categories), len(lab)
    Gc = G.tocsr()

    def count(c):
        M = sp.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, k))
        return (M.T @ Gc @ M).toarray()

    obs = count(codes)
    grp = [np.flatnonzero(np.asarray(groups) == g) for g in pd.unique(np.asarray(groups))]
    perm = np.empty((n_perm, k, k))
    for i in range(n_perm):
        c = codes.copy()
        for idx in grp:
            c[idx] = rng.permutation(c[idx])
        perm[i] = count(c)
    z = (obs - perm.mean(0)) / np.maximum(perm.std(0), 1e-9)
    cats = list(lab.categories)
    return (pd.DataFrame(z, cats, cats), pd.DataFrame(obs, cats, cats),
            pd.DataFrame(perm.mean(0), cats, cats))


# ================================================================ niche GLM


def niche_glm(y, X, offset, cluster, fe=None):
    """Poisson GLM with a log library-size offset, optional fixed effects, and SEs clustered on
    `cluster` (the FOV). Clustering on FOV is what keeps neighbouring cells -- which share a
    neighbourhood and so are not independent -- from inflating the evidence.

    X: DataFrame of covariates (the first column is the term of interest). Returns a dict for it.
    """
    import statsmodels.api as sm

    D = X.astype(float).copy()
    if fe is not None:
        D = D.join(pd.get_dummies(pd.Series(np.asarray(fe), index=D.index).astype(str),
                                  prefix="fe", drop_first=True, dtype=float))
    D = sm.add_constant(D, has_constant="add")
    term = X.columns[0]
    out = {"term": term, "n": len(y), "n_pos": int((np.asarray(y) > 0).sum())}
    if out["n_pos"] < 20 or D[term].std() == 0:
        return out | {"coef": np.nan, "se": np.nan, "p": np.nan}
    try:
        res = sm.GLM(np.asarray(y, float), D, family=sm.families.Poisson(),
                     offset=np.asarray(offset, float)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(np.asarray(cluster))[0]})
        return out | {"coef": res.params[term], "se": res.bse[term], "p": res.pvalues[term]}
    except Exception as e:  # separation / singular design on a thin slide
        return out | {"coef": np.nan, "se": np.nan, "p": np.nan, "error": str(e)[:80]}


def ols_fe(y, X, cluster, fe=None):
    """OLS counterpart of `niche_glm` for a continuous per-cell outcome (e.g. AspectRatio): same
    fixed effects and FOV-clustered SEs. Returns a dict for X's first column."""
    import statsmodels.api as sm

    D = X.astype(float).copy()
    if fe is not None:
        D = D.join(pd.get_dummies(pd.Series(np.asarray(fe), index=D.index).astype(str),
                                  prefix="fe", drop_first=True, dtype=float))
    D = sm.add_constant(D, has_constant="add")
    term = X.columns[0]
    y = np.asarray(y, float)
    out = {"term": term, "n": len(y), "n_pos": int(np.isfinite(y).sum())}
    if len(y) < 50 or D[term].std() == 0:
        return out | {"coef": np.nan, "se": np.nan, "p": np.nan}
    try:
        res = sm.OLS(y, D).fit(cov_type="cluster",
                               cov_kwds={"groups": pd.factorize(np.asarray(cluster))[0]})
        return out | {"coef": res.params[term], "se": res.bse[term], "p": res.pvalues[term]}
    except Exception as e:
        return out | {"coef": np.nan, "se": np.nan, "p": np.nan, "error": str(e)[:80]}


# ================================================================ compartments (28e)


def compartments(G, lineage, depth, epi_frac=0.5, dej_um=50.0, lineage_ker="Keratinocyte"):
    """Per cell: 'epidermis' when >= `epi_frac` of its graph neighbours are keratinocytes (and
    every keratinocyte), 'DEJ' within `dej_um` of the nearest keratinocyte, 'dermis' beyond,
    'no_epidermis' where the piece holds no epidermis (depth NaN).

    Depth alone cannot separate intraepidermal T cells from papillary-dermis T cells touching
    the basal layer (both sit ~0-15 µm from a keratinocyte); the neighbour fraction can.
    Returns (labels, keratinocyte neighbour fraction)."""
    lab = np.asarray(lineage)
    is_k = (lab == lineage_ker).astype(np.float32)
    n_nb = np.asarray(G.sum(1)).ravel()
    frac = np.divide(G @ is_k, n_nb, out=np.zeros(len(lab)), where=n_nb > 0)
    d = np.asarray(depth, float)
    out = np.select([np.isnan(d), (lab == lineage_ker) | (frac >= epi_frac), d <= dej_um],
                    ["no_epidermis", "epidermis", "DEJ"], "dermis")
    return out, frac


def pautrier(G, is_t, compartment, min_t=3):
    """Intraepidermal T cells in a cluster of >= `min_t` T cells (the cell plus >= min_t-1 T
    neighbours on G) -- a Pautrier-like aggregate, by position only."""
    is_t = np.asarray(is_t, bool)
    n_t = G @ is_t.astype(np.float32)
    return is_t & (np.asarray(compartment) == "epidermis") & (n_t >= min_t - 1)


def field_vs_null(G, values, anchor, pool, strata, min_pool=10):
    """Sum of `values` over each cell's graph neighbours (the 'field'), for `anchor` cells vs the
    mean field of `pool` cells in the same stratum (piece × depth bin × density bin).

    Returns a per-anchor DataFrame: obs field, expected field (stratum mean over the pool) and the
    stratum. Anchors whose stratum holds fewer than `min_pool` pool cells are dropped. Aggregate
    as log2(sum obs / sum exp) per slide; the pool must exclude the anchors' own lineage.
    """
    field = G @ np.asarray(values, np.float32)
    strata = pd.Series(np.asarray(strata).astype(str))
    pool, anchor = np.asarray(pool, bool), np.asarray(anchor, bool)
    ref = pd.DataFrame({"s": strata[pool].to_numpy(), "f": field[pool]}).groupby("s").f.agg(["mean", "size"])
    ref = ref[ref["size"] >= min_pool]
    idx = np.flatnonzero(anchor)
    s = strata.iloc[idx].to_numpy()
    ok = pd.Series(s).isin(ref.index).to_numpy()
    idx, s = idx[ok], s[ok]
    return pd.DataFrame({"cell": idx, "obs": field[idx], "exp": ref["mean"].reindex(s).to_numpy(),
                         "stratum": s})


# ================================================================ reporting (shared with 28d)


def _stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def robust_flags(df, key, min_agree=3):
    """Pooled BH q within the figure; robust = q < 0.05 and >= `min_agree` slides share the pooled
    sign. `df` holds a 'pooled' row and one row per slide for each key; 'control' rows are not
    tested. Same rule as 28d §U."""
    from statsmodels.stats.multitest import multipletests

    pooled = df[df.slide == "pooled"].copy()
    test = ~pooled.control
    pooled["q"] = np.nan
    pooled.loc[test, "q"] = multipletests(pooled.loc[test, "p"].fillna(1), method="fdr_bh")[1]
    per = df[df.slide != "pooled"].merge(pooled[key + ["coef"]].rename(columns={"coef": "pooled_coef"}), on=key)
    agree = (np.sign(per.coef) == np.sign(per.pooled_coef)).groupby([per[k] for k in key]).sum()
    pooled = pooled.merge(agree.rename("n_slides_agree").reset_index(), on=key, how="left")
    pooled["robust"] = (pooled.q < 0.05) & (pooled.n_slides_agree >= min_agree)
    return pooled


SLIDE_MARKERS = ["o", "s", "^", "D", "v", "P"]


def coef_panel(ax, df, labels, slides, slide_stage, spill=None, title="", ylabel="log rate ratio"):
    """Pooled coefficient ± 95% CI (black) with per-slide fits as stage-coloured markers. `df`
    needs label / slide / coef / se (and q on pooled rows). `spill`: (lo, hi) grey band."""
    if spill is not None and np.isfinite(spill).all():
        ax.axhspan(*spill, color="#bdbdbd", alpha=0.35, lw=0, label="spillover reference")
    ax.axhline(0, color="k", lw=0.6)
    mark = dict(zip(slides, SLIDE_MARKERS))
    for j, lab in enumerate(labels):
        d = df[df.label == lab]
        if not (d.slide == "pooled").any() or not np.isfinite(d[d.slide == "pooled"].coef.iloc[0]):
            ax.text(j, 0, "n/a", ha="center", va="bottom", fontsize=6, color="#9a9a9a")
            continue
        p = d[d.slide == "pooled"].iloc[0]
        ax.errorbar(j, p.coef, yerr=1.96 * p.se, fmt="o", color="k", ms=3.5, lw=1, capsize=2, zorder=3)
        star = _stars(p.q) if "q" in d and np.isfinite(p.get("q", np.nan)) else ""
        if star:
            ax.text(j, p.coef + 1.96 * p.se, star, ha="center", va="bottom", fontsize=7)
        for k, s in enumerate(slides):
            r = d[d.slide == s]
            if len(r) and np.isfinite(r.coef.iloc[0]):
                ax.scatter(j - 0.24 + 0.16 * k, r.coef.iloc[0], s=11, color=STAGE_COLORS[slide_stage[s]],
                           marker=mark[s], edgecolor="none", zorder=2)
    for s in slides:
        ax.scatter([], [], s=11, color=STAGE_COLORS[slide_stage[s]], marker=mark[s],
                   label=f"slide {s} ({slide_stage[s]})")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7); ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
