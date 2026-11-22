"""Ligand-receptor traffic on the malignant CD4 <-> fibroblast / myeloid axis (nb28 section 10).

Why this exists next to `ccc_utils.py` rather than inside it. `27_ccc_subtype` runs LIANA over
the whole consensus resource on `data/ccc/ccc_skin.h5ad` and ranks edges by specificity. That
object is **v1-era** (727k cells, 82 donors, 7 studies, against v2's 985k / 112 / 12) and splits
CD4 on the ALICE TCR call, not on `mal_knn`. Importing it into a v2 notebook is the exact
failure mode `11_atlas_v2_qc` exists to catch, and LIANA is not installed in `neural_nmf_env`
anyway. So this module does the smallest thing that answers the question on v2 data:

  * expression comes from `skin_degradome_pseudobulk_full.parquet`, the FULL-gene (level x
    sample) count cube the degradome build writes in the same pass as the object. Same cells,
    same scope, same `degradome_level`, same `mal_knn` split -- one vintage;
  * the panel is `ccc_panel.PRIORITY_PAIRS`, 30 curated edges, not a 4,700-pair resource;
  * the score is an abundance product, `sqrt(ligand * receptor)`, per sample;
  * the statistic is the one nb28 section 4d uses -- one malignant-minus-reactive delta per
    sample, Wilcoxon signed-rank -- so the CCC section and the expression sections rest on the
    same design and can be read against each other.

What this is NOT. There is no specificity *rank* over a resource, because there is no resource
here -- 30 curated edges cannot supply one, and nothing below is comparable to
`tables/ccc40_*`. What there is, since `edge_specificity` was added, is a null drawn from the
transcriptome rather than from the panel: an edge is asked whether its paired increase exceeds
what abundance-matched random gene pairs show in the same two levels. That separates "this edge
moved" from "everything in this arm moved" and from "that level is simply deep" -- the
magnitude/specificity split LIANA makes, rebuilt for a curated panel. A high *score* still means
only "both partners are abundant in the same sample".
"""

from pathlib import Path

import numpy as np
import pandas as pd

import ccc_panel as panel
import degradome_data as cfg
from degradome_helpers import _finish, _level_colors  # noqa: F401  (shared palette + save)


# ================================================================ the panel, made machine-readable

# `PRIORITY_PAIRS` is written for a reader: `|` separates alternatives, `.` joins the subunits
# of a complex, and a few entries are prose rather than symbols. Every deviation is listed here
# explicitly rather than guessed at by a regex -- there are 30 pairs, so the map is auditable.
TOKEN_FIXES = {
    "S100A8.A9": ["S100A8", "S100A9"],   # the panel's shorthand for the calprotectin dimer
    "BSG (homophilic)": ["BSG"],
    "HLA-DR": ["HLA-DRA"],               # the alpha chain is the invariant one
}

# Real biology the cube cannot score: a shed ectodomain, a rearranged receptor, and a zymogen
# conversion are not transcripts. Reported in the run log, never silently dropped.
NON_GENE_TOKENS = {
    "ADAM17->sIL6R": "trans-signalling by a shed receptor, not a transcript pair",
    "TCR": "a rearranged receptor, not a gene symbol",
    "plasminogen->plasmin": "a zymogen conversion, not a transcript pair",
    "TGFB1_latent": "a latent complex, not a distinct transcript",
}

# The panel names cell types in prose. Nothing in `helpers/` uses these spellings.
CELL_MAP = {
    "malignant_CD4": [cfg.CD4_MALIGNANT],
    "CD4": [cfg.CD4_MALIGNANT],
    "fibroblast": list(cfg.FIBRO_LEVELS),
    "fibroblast_apCAF": ["F_apCAF"],
    "myeloid": list(cfg.MYELOID_LEVELS),
    "keratinocyte": ["Keratinocyte"],
}

# The two CD4 arms the contrast runs over. Whichever side of an edge the panel calls a CD4 cell
# is expanded into both, so every edge is scored twice per sample and the delta is paired.
CD4_ARMS = [cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE]

# Which fibrosis axis each priority pair speaks to, for the per-axis hierarchy plots. Keyed by
# position in `PRIORITY_PAIRS` so it cannot drift if a rationale string is reworded.
PAIR_AXIS = {
    0: "th2_fibrotic", 1: "gp130_stat3", 2: "gp130_stat3", 3: "tgfb_activation",
    4: "bsg_resolution", 5: "gp130_stat3", 6: "th2_fibrotic", 7: "th2_fibrotic",
    8: "th2_fibrotic", 9: "juxtacrine", 10: "th2_fibrotic", 11: "juxtacrine",
    12: "myeloid_innate", 13: "infl_matrix_suppression", 14: "tgfb_activation",
    # the nb28 panel extension pairs; `other` where the spec family has no fibrosis axis
    15: "juxtacrine", 16: "juxtacrine", 17: "juxtacrine", 18: "gp130_stat3",
    19: "tgfb_activation", 20: "caf_expansion", 21: "other", 22: "other",
    23: "other", 24: "other", 25: "gp130_stat3", 26: "other", 27: "other",
    28: "other", 29: "other",
}


def _cells(field):
    """`"fibroblast|myeloid"` -> the degradome levels it covers, deduped, order kept."""
    out = []
    for name in (t.strip() for t in str(field).split("|")):
        for lv in CELL_MAP.get(name, []):
            if lv not in out:
                out.append(lv)
    return out, [t.strip() for t in str(field).split("|") if t.strip() not in CELL_MAP]


def _subunits(token):
    """One alternative -> its subunit symbols, or `(None, reason)` when it is not transcripts."""
    token = token.strip()
    if token in TOKEN_FIXES:
        return TOKEN_FIXES[token], None
    if token in NON_GENE_TOKENS:
        return None, NON_GENE_TOKENS[token]
    parts = [p.strip() for p in token.split(".") if p.strip()]
    bad = [p for p in parts if p in NON_GENE_TOKENS]
    if bad:
        return None, NON_GENE_TOKENS[bad[0]]
    return parts, None


def priority_pairs_frame(pairs=None, var_names=None, stroma_levels=None, verbose=True):
    """Expand `PRIORITY_PAIRS` into one row per concrete (sender, ligand, receiver, receptor).

    Returns `(frame, dropped)`. `frame` carries `pair_id, axis, sender, receiver, ligand,
    receptor, ligand_genes, receptor_genes, direction, rationale`; `dropped` is one row per
    alternative that could not be resolved, with the reason -- a shed ectodomain, a rearranged
    receptor, or a symbol absent from the object. Nothing is discarded quietly.

    `direction` is `cd4_to_stroma` when the panel puts the CD4 cell on the ligand side and
    `stroma_to_cd4` when it puts it on the receptor side; both are scored.

    `stroma_levels` restricts the non-CD4 side to a vocabulary. nb28 passes fibroblast +
    myeloid, which is what removes the `keratinocyte` resolution of `LAMA3 ->
    fibroblast|keratinocyte`; the removal is recorded in `dropped`, not silent.
    """
    pairs = panel.PRIORITY_PAIRS if pairs is None else pairs
    have = None if var_names is None else set(map(str, var_names))
    allowed = None if stroma_levels is None else set(map(str, stroma_levels))

    rows, dropped = [], []
    for i, (src, lig, tgt, rec, why) in enumerate(pairs):
        src_lv, src_bad = _cells(src)
        tgt_lv, tgt_bad = _cells(tgt)
        for name in src_bad + tgt_bad:
            dropped.append({"pair_id": i, "side": "cell", "token": name,
                            "reason": "cell name not in CELL_MAP"})
        cd4_side = ("ligand" if any(lv in CD4_ARMS for lv in src_lv)
                    else "receptor" if any(lv in CD4_ARMS for lv in tgt_lv) else None)
        if cd4_side is None:
            dropped.append({"pair_id": i, "side": "cell", "token": f"{src}/{tgt}",
                            "reason": "neither side is a CD4 level"})
            continue
        stroma = tgt_lv if cd4_side == "ligand" else src_lv
        if allowed is not None:
            for lv in [x for x in stroma if x not in allowed]:
                dropped.append({"pair_id": i, "side": "cell", "token": lv,
                                "reason": "stromal level outside the requested vocabulary"})
            stroma = [lv for lv in stroma if lv in allowed]
        if not stroma:
            dropped.append({"pair_id": i, "side": "cell", "token": f"{src}/{tgt}",
                            "reason": "no stromal level resolved"})
            continue

        for lig_tok in str(lig).split("|"):
            lg, lreason = _subunits(lig_tok)
            if lg is None:
                dropped.append({"pair_id": i, "side": "ligand", "token": lig_tok.strip(),
                                "reason": lreason})
                continue
            for rec_tok in str(rec).split("|"):
                rg, rreason = _subunits(rec_tok)
                if rg is None:
                    dropped.append({"pair_id": i, "side": "receptor", "token": rec_tok.strip(),
                                    "reason": rreason})
                    continue
                miss = [] if have is None else [g for g in lg + rg if g not in have]
                if miss:
                    dropped.append({"pair_id": i, "side": "both",
                                    "token": f"{lig_tok.strip()}->{rec_tok.strip()}",
                                    "reason": f"absent from the object: {miss}"})
                    continue
                for lv in stroma:
                    rows.append({
                        "pair_id": i, "axis": PAIR_AXIS.get(i, "other"),
                        "direction": "cd4_to_stroma" if cd4_side == "ligand"
                                     else "stroma_to_cd4",
                        "stroma_level": lv,
                        "ligand": "_".join(lg), "receptor": "_".join(rg),
                        "ligand_genes": lg, "receptor_genes": rg,
                        "rationale": why,
                    })

    frame = pd.DataFrame(rows)
    if len(frame):
        frame["edge"] = frame["ligand"] + "->" + frame["receptor"]
        frame["key"] = frame["edge"] + "@" + frame["stroma_level"]
        frame = frame.drop_duplicates("key").reset_index(drop=True)
    drop = pd.DataFrame(dropped).drop_duplicates() if dropped else pd.DataFrame(
        columns=["pair_id", "side", "token", "reason"])
    if verbose:
        print(f"priority pairs: {len(pairs)} curated -> {frame['edge'].nunique() if len(frame) else 0} "
              f"concrete edges over {frame['stroma_level'].nunique() if len(frame) else 0} "
              f"stromal levels ({len(frame)} edge x level rows)")
        if len(drop):
            print(f"  not scored ({len(drop)}):")
            for _, r in drop.iterrows():
                print(f"    [{r.pair_id}] {r.side}: {r.token} -- {r.reason}")
    return frame, drop


# ================================================================ the cube

def load_cube(genes=None, cube_path=None, meta_path=None, disease="MF", min_cells=None,
              block=4000, verbose=True):
    """The (level x sample) full-gene count cube, as log1p CP10K per pseudo-sample.

    The cube is summed RAW counts, so it is normalised here and not at build time. Normalising
    per pseudo-sample is the right unit for this question: the score compares a ligand in one
    level against a receptor in another within the same biopsy, so each pseudo-sample has to be
    on its own per-10k scale first.

    **The size factor is summed over all 42,347 columns before `genes` is applied**, the same
    rule the object build follows -- normalising after subsetting would rescale every
    pseudo-sample by its panel fraction, which varies by cell type and is precisely the axis
    this section measures. The sum is accumulated `block` columns at a time so the peak stays
    around 100 MB instead of the ~2 GB a whole-cube read costs; that matters, the login node is
    shared.

    Rows below `min_cells` are dropped, the same gate every donor median in this notebook uses:
    a mean over eight cells is not an expression level.
    """
    import pyarrow.parquet as pq

    cube_path = cfg.PSEUDOBULK_PQ if cube_path is None else Path(cube_path)
    meta_path = cfg.PSEUDOBULK_META if meta_path is None else Path(meta_path)
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells

    meta = pd.read_parquet(meta_path)
    meta[cfg.GROUPBY] = meta[cfg.GROUPBY].astype(str)
    keep = meta["n_cells"] >= min_cells
    if disease is not None:
        keep &= meta["disease"].astype(str) == str(disease)
    meta = meta[keep]

    pf = pq.ParquetFile(cube_path)
    names = list(pf.schema_arrow.names)
    index_col = next((c for c in names if c.startswith("__index_level")), None)
    gene_cols = [c for c in names if c != index_col]
    idx = pd.Index(pq.read_table(cube_path, columns=[index_col])
                   .column(0).to_pandas().astype(str), name="pseudo_sample")

    lib = np.zeros(len(idx), dtype=np.float64)
    for k in range(0, len(gene_cols), block):
        t = pq.read_table(cube_path, columns=gene_cols[k:k + block])
        lib += np.asarray(t.to_pandas(), dtype=np.float64).sum(axis=1)
    lib[lib == 0] = 1.0

    want = gene_cols if genes is None else [g for g in dict.fromkeys(genes) if g in set(gene_cols)]
    sub = pq.read_table(cube_path, columns=want).to_pandas()
    sub.index = idx
    rows = idx.intersection(meta.index)
    sub, meta = sub.loc[rows], meta.loc[rows]
    expr = pd.DataFrame(np.log1p(sub.to_numpy(dtype=np.float64) / lib[idx.get_indexer(rows), None] * 1e4),
                        index=rows, columns=sub.columns)
    if verbose:
        print(f"cube: {expr.shape[0]:,} pseudo-samples x {expr.shape[1]:,} genes "
              f"(size factor over all {len(gene_cols):,}) | disease={disease} "
              f"| >= {min_cells} cells | {meta[cfg.SAMPLE_KEY].nunique()} samples, "
              f"{meta[cfg.DONOR_KEY].nunique()} donors, {meta[cfg.STUDY_KEY].nunique()} studies")
    return expr, meta


def _cell_libsize(adata, verbose=True):
    """The per-cell library size the build divided by, recovered from the object itself.

    `X` is `log1p(count / lib * 1e4)` and the raw counts are on `cfg.COUNTS_LAYER`, so for any
    gene a cell actually expresses, `lib = count / (expm1(X) / 1e4)`. Taking the median over the
    expressed panel genes recovers the size factor that was summed over all 42,347 genes
    **before** the column subset -- exactly, and without trusting `obs.total_counts`, which was
    computed upstream on a possibly different gene set. Cells with no panel counts get NaN and
    contribute nothing to a pseudobulk.
    """
    import scipy.sparse as sp

    C = adata.layers[cfg.COUNTS_LAYER]
    X = adata.X
    C = sp.csr_matrix(C) if not sp.issparse(C) else C.tocsr()
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    C.sort_indices(); X.sort_indices()
    if not np.array_equal(C.indices, X.indices) or not np.array_equal(C.indptr, X.indptr):
        raise ValueError("counts and X do not share a sparsity pattern -- cannot recover lib")
    ratio = C.data / (np.expm1(X.data) / 1e4)
    ratio[~np.isfinite(ratio)] = 0.0
    R = sp.csr_matrix((ratio, X.indices, X.indptr), shape=X.shape)
    cnt = np.diff(X.indptr).astype(np.float64)
    # every expressed gene of a cell gives the same ratio up to float error, so the mean over
    # them is the size factor, not an estimate of it.
    lib = np.asarray(R.sum(axis=1)).ravel() / np.maximum(cnt, 1.0)
    lib[cnt == 0] = np.nan
    if verbose:
        ok = np.isfinite(lib)
        print(f"library size recovered for {int(ok.sum()):,}/{adata.n_obs:,} cells "
              f"(median {np.nanmedian(lib):,.0f})")
    return lib


def _adata_pseudobulk(adata, level, levels, genes, lib, min_cells, disease=None):
    """(level x sample) log1p CP10K from the object, on `load_cube`'s contract and index."""
    import scipy.sparse as sp

    lvl = pd.Series(level).astype(str).to_numpy()
    samp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
    keep = np.isin(lvl, list(levels)) & np.isfinite(lib)
    if disease is not None:
        keep &= adata.obs["disease"].astype(str).to_numpy() == str(disease)
    idx = np.flatnonzero(keep)
    if not len(idx):
        raise ValueError(f"no cells for levels {sorted(levels)}")

    tags = pd.Index(lvl[idx] + "||" + samp[idx])
    codes, uniq = pd.factorize(tags)
    n_cells = np.bincount(codes, minlength=len(uniq))
    var = pd.Index(map(str, adata.var_names))
    have = [g for g in genes if g in var]
    C = adata.layers[cfg.COUNTS_LAYER][idx][:, var.get_indexer(np.asarray(have))]
    C = sp.csr_matrix(C) if not sp.issparse(C) else C
    G = sp.csr_matrix((np.ones(len(idx)), (codes, np.arange(len(idx)))),
                      shape=(len(uniq), len(idx)))
    tot = np.asarray((G @ C).todense(), dtype=np.float64)
    libsum = np.asarray(G @ lib[idx], dtype=np.float64)
    libsum[libsum == 0] = 1.0

    expr = pd.DataFrame(np.log1p(tot / libsum[:, None] * 1e4),
                        index=pd.Index(uniq, name="pseudo_sample"), columns=have)
    meta = pd.DataFrame({cfg.GROUPBY: [t.split("||")[0] for t in uniq],
                         cfg.SAMPLE_KEY: [t.split("||", 1)[1] for t in uniq],
                         "n_cells": n_cells}, index=expr.index)
    # carry the design columns the cube's own metadata has, so the two halves join cleanly
    first = pd.DataFrame({c: adata.obs[c].astype(str).to_numpy()[idx]
                          for c in (cfg.DONOR_KEY, cfg.STUDY_KEY, "disease")
                          if c in adata.obs}, index=tags)
    first = first.groupby(level=0).first().reindex(uniq)
    meta = meta.join(first.set_axis(expr.index))
    ok = meta["n_cells"] >= min_cells
    return expr[ok.to_numpy()], meta[ok.to_numpy()]


def cube_tcr(adata, level, genes=None, pool_genes=2000, seed=0, disease="MF", min_cells=None,
             tol=1e-4, verbose=True):
    """`load_cube`, with the two CD4 arm rows re-derived from `level` instead of the build's.

    The cube on disk is keyed on `degradome_level`, whose CD4 rows were split by the build's
    headline call (`cfg.MALIG_HEADLINE`, `mal_knn`). Its **stromal rows do not depend on that
    call at all** -- a fibroblast is a fibroblast under either -- so only the two CD4 arm rows
    have to be rebuilt when the analysis runs on `mal_tcr`. Rebuilding the whole cube would mean
    re-running the bsub build for two rows.

    So: stromal and context rows come from the cube, where all 42,347 columns are available and
    the stromal-side null can therefore be drawn from the transcriptome; the CD4 arm rows are
    summed from the object, where only the panel exists. Columns absent from the object are NaN
    on the CD4 rows, which is why `pools["cd4"]` is the panel and `pools["stroma"]` is not.

    The two halves must be on one scale or every delta below is a units artefact, so a stromal
    level is recomputed from the object and asserted equal to the cube's own row within `tol`.
    That gate is the reason this function returns rather than raises on the happy path.

    Returns `(expr, meta, pools)`.
    """
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    rng = np.random.default_rng(seed)
    level = pd.Series(np.asarray(level).astype(str), index=adata.obs_names)

    import pyarrow.parquet as pq
    cube_cols = [c for c in pq.ParquetFile(cfg.PSEUDOBULK_PQ).schema_arrow.names
                 if not c.startswith("__index_level")]
    obj_genes = [g for g in map(str, adata.var_names)]
    want = list(dict.fromkeys(list(genes or []) + obj_genes))
    pool = [g for g in cube_cols if g not in set(want)]
    draw = sorted(rng.choice(len(pool), min(int(pool_genes), len(pool)), replace=False))
    stroma_pool = [pool[i] for i in draw]

    expr, meta = load_cube(genes=want + stroma_pool, disease=disease, min_cells=min_cells,
                           verbose=verbose)
    lib = _cell_libsize(adata, verbose=verbose)

    # ---- the scale gate: one stromal level, recomputed from the object
    stro = [lv for lv in meta[cfg.GROUPBY].unique() if lv not in CD4_ARMS]
    probe = (meta[meta[cfg.GROUPBY].isin(stro)].groupby(cfg.GROUPBY)["n_cells"].sum()
             .sort_values().index[-1])
    pe, pm = _adata_pseudobulk(adata, level, [probe], obj_genes, lib, min_cells,
                               disease=disease)
    shared_rows = pe.index.intersection(expr.index)
    shared_cols = pe.columns.intersection(expr.columns)
    if not len(shared_rows) or not len(shared_cols):
        raise AssertionError(f"scale gate cannot run: no overlap on level {probe}")
    diff = float(np.nanmax(np.abs(pe.loc[shared_rows, shared_cols].to_numpy()
                                  - expr.loc[shared_rows, shared_cols].to_numpy())))
    if verbose:
        print(f"scale gate on {probe}: max |object - cube| = {diff:.3g} over "
              f"{len(shared_rows)} pseudo-samples x {len(shared_cols)} genes")
    if not (diff < tol):
        raise AssertionError(
            f"the object and the cube are on different scales ({diff:.3g} >= {tol:.3g}) -- "
            f"every section 10 delta would be a units artefact. Check that the object and "
            f"{cfg.PSEUDOBULK_PQ.name} came from the same build pass.")

    # ---- swap the CD4 rows
    ce, cm = _adata_pseudobulk(adata, level, CD4_ARMS, want, lib, min_cells, disease=disease)
    drop = meta.index[meta[cfg.GROUPBY].isin(CD4_ARMS)]
    expr, meta = expr.drop(index=drop), meta.drop(index=drop)
    ce = ce.reindex(columns=expr.columns)                 # non-panel columns stay NaN
    for c in meta.columns:
        if c not in cm.columns:
            cm[c] = np.nan
    expr = pd.concat([expr, ce])

    meta = pd.concat([meta, cm[meta.columns]])

    pools = {"cd4": [g for g in obj_genes if g in expr.columns], "stroma": stroma_pool}
    if verbose:
        n = cm.groupby(cfg.GROUPBY)[cfg.SAMPLE_KEY].nunique()
        print(f"CD4 arms re-derived from the supplied level: "
              + ", ".join(f"{k} {v} samples" for k, v in n.items())
              + f" | null pools: CD4 side {len(pools['cd4'])} panel genes, "
                f"stromal side {len(pools['stroma'])} cube genes")
    return expr, meta, pools


def _side_expr(expr, meta, level, genes):
    """Mean-of-min: one value per sample for a (level, complex). Missing rows stay missing."""
    rows = meta.index[meta[cfg.GROUPBY] == level]
    if not len(rows):
        return pd.Series(dtype=float)
    sub = expr.loc[rows, genes]
    # A complex signals only if every subunit is present, so the complex takes the minimum.
    val = sub.min(axis=1) if len(genes) > 1 else sub.iloc[:, 0]
    val.index = meta.loc[rows, cfg.SAMPLE_KEY].astype(str).to_numpy()
    return val[~val.index.duplicated()]


def edge_scores(expr, meta, pairs, min_samples=None, verbose=True):
    """`sqrt(ligand * receptor)` per (sample, edge, stromal level, CD4 arm).

    The geometric mean is the conventional abundance score: it is zero unless both partners are
    present, and it is on the same scale as the inputs. Returns a long frame with one row per
    (sample, CD4 arm, edge, stromal level); a sample that lacks either level above the cell gate
    simply has no row, and the paired test then skips it rather than imputing a zero.

    `min_samples` drops an (edge x level) key carried by too few paired samples -- Neutrophils
    and Mono_CD16 clear the 25-cell gate in a handful of biopsies each, and an edge measured in
    two samples draws a wide ribbon and a dark cell with no test behind either.
    """
    out = []
    for _, r in pairs.iterrows():
        stroma_l = _side_expr(expr, meta, r.stroma_level, r.ligand_genes)
        stroma_r = _side_expr(expr, meta, r.stroma_level, r.receptor_genes)
        for arm in CD4_ARMS:
            cd4_l = _side_expr(expr, meta, arm, r.ligand_genes)
            cd4_r = _side_expr(expr, meta, arm, r.receptor_genes)
            lig, rec = ((cd4_l, stroma_r) if r.direction == "cd4_to_stroma"
                        else (stroma_l, cd4_r))
            common = lig.index.intersection(rec.index)
            if not len(common):
                continue
            out.append(pd.DataFrame({
                cfg.SAMPLE_KEY: common,
                "cd4_arm": arm,
                "key": r.key, "edge": r.edge, "axis": r.axis,
                "direction": r.direction, "stroma_level": r.stroma_level,
                "score": np.sqrt(lig[common].to_numpy() * rec[common].to_numpy()),
            }))
    scores = (pd.concat(out, ignore_index=True) if out else
              pd.DataFrame(columns=[cfg.SAMPLE_KEY, "cd4_arm", "key", "edge", "axis",
                                    "direction", "stroma_level", "score"]))
    min_samples = cfg.MIN_SAMPLES if min_samples is None else min_samples
    if len(scores) and min_samples:
        paired = (scores.pivot_table(index=[cfg.SAMPLE_KEY, "key"], columns="cd4_arm",
                                     values="score", aggfunc="median")
                  .dropna().reset_index().groupby("key")[cfg.SAMPLE_KEY].nunique())
        thin = sorted(set(scores["key"]) - set(paired[paired >= min_samples].index))
        scores = scores[~scores["key"].isin(thin)].reset_index(drop=True)
        if verbose and thin:
            lv = sorted({k.split("@")[-1] for k in thin})
            print(f"  dropped {len(thin)} edge x level keys below {min_samples} paired "
                  f"samples (levels touched: {lv})")
    if verbose and len(scores):
        n_paired = (scores.pivot_table(index=[cfg.SAMPLE_KEY, "key"], columns="cd4_arm",
                                       values="score", aggfunc="median")
                    .dropna().index.get_level_values(cfg.SAMPLE_KEY).nunique())
        print(f"scored {len(scores):,} (sample x arm x edge x level) rows | "
              f"{scores['key'].nunique()} edge x level keys | "
              f"{n_paired} samples carry both CD4 arms")
    return scores


def edge_paired_test(scores, ref=None, alt=None, verbose=True):
    """The section 4d design, one edge at a time: malignant - reactive within the same sample.

    Delegates to `degradome_cells.paired_within`, so the p-value, the sign test alongside it and
    the `n_pairs` printed next to both are produced by exactly the code the expression sections
    use. Every edge is one score column; the arm is the CD4 level; the pair key is the sample.
    """
    import degradome_cells as DC

    ref = cfg.CD4_REACTIVE if ref is None else ref
    alt = cfg.CD4_MALIGNANT if alt is None else alt

    wide = scores.pivot_table(index=[cfg.SAMPLE_KEY, "cd4_arm"], columns="key",
                              values="score", aggfunc="median").reset_index()
    wide["eligible"] = True
    keys = [c for c in wide.columns if c not in (cfg.SAMPLE_KEY, "cd4_arm", "eligible")]
    res = DC.paired_within(wide, keys, pair_key=cfg.SAMPLE_KEY, arm_col="cd4_arm",
                           ref=ref, alt=alt)

    meta = (scores.drop_duplicates("key")
            .set_index("key")[["edge", "axis", "direction", "stroma_level"]])
    res = res.rename(columns={"score": "key"}).join(meta, on="key")
    res["mean_score"] = res["key"].map(scores.groupby("key")["score"].mean())
    # These products are zero-inflated: median_delta can be exactly 0 while the signed-rank is
    # decisive, so direction is read off the nonzero pairs, not off the median.
    res["higher"] = np.where(res["n_pos"] > res["n_neg"], str(alt), str(ref))
    res.loc[res["q"].isna(), "higher"] = ""        # abstained: no direction to report
    res = res.sort_values(["axis", "edge", "stroma_level"]).reset_index(drop=True)
    if verbose:
        sig = int((res["q"] < 0.05).sum())
        print(f"paired edge tests: {len(res)} edge x level, {sig} at q < 0.05 | "
              f"median n_pairs = {res['n_pairs'].median():.0f}")
    return res


def _level_frame(expr, meta, level, genes):
    """Samples x genes for one level, indexed by sample -- the matrix form of `_side_expr`."""
    rows = meta.index[meta[cfg.GROUPBY].astype(str) == str(level)]
    if not len(rows):
        return pd.DataFrame(dtype=float)
    sub = expr.loc[rows, [g for g in genes if g in expr.columns]].copy()
    sub.index = pd.Index(meta.loc[rows, cfg.SAMPLE_KEY].astype(str).to_numpy())
    return sub[~sub.index.duplicated()]


def edge_specificity(expr, meta, pairs, tests, pools, band=0.1, min_band=60, ref=None,
                     alt=None, verbose=True):
    """Is the increase specific to *this* pair, or would any pair of that abundance show it?

    `edge_paired_test` answers "did this edge's score rise in the clonal arm". Two uninteresting
    things produce that answer:

      * **the whole arm moved.** If the clonal CD4 pseudobulk is broader or deeper than the
        non-clonal one, every CD4-side gene rises and so does every edge that has one.
      * **that level is simply deep.** The score is `sqrt(ligand x receptor)`, so a stromal
        level with a lot of counts lifts every edge landing on it.

    Neither is a statement about the pair. So each side is replaced, in turn, by the genes of
    the same abundance **in the same level** -- the nearest `band` of the relevant pool by mean
    expression there, at least `min_band` genes -- and the real edge's mean paired delta is
    ranked against the deltas those substitutes produce over the same samples:

        spec_p = (1 + #{null >= real}) / (n_band + 1)          one-sided, the direction claimed

    `pools` comes from `cube_tcr`: the CD4-side pool is the panel, because the TCR-split CD4
    rows exist only for the panel; the stromal-side pool is drawn from the whole cube. The CD4
    side is therefore matched against matrix and cytokine genes rather than the transcriptome,
    which makes `spec_p_cd4_side` **conservative** -- it under-calls specificity, it cannot
    invent it. Say so wherever the column is printed.

    `spec_p = max(spec_p_cd4_side, spec_p_stroma_side)`: an edge has to beat both nulls.
    Returns `tests` with the specificity columns joined on `key`.
    """
    ref = cfg.CD4_REACTIVE if ref is None else ref
    alt = cfg.CD4_MALIGNANT if alt is None else alt
    pinfo = pairs.set_index("key")
    cd4_pool = [g for g in pools.get("cd4", []) if g in expr.columns]
    str_pool = [g for g in pools.get("stroma", []) if g in expr.columns]

    cache = {}

    def _frame(level, genes):
        k = (str(level), tuple(genes))
        if k not in cache:
            cache[k] = _level_frame(expr, meta, level, genes)
        return cache[k]

    def _stat(m, r, other, common):
        """Mean paired (alt - ref) delta of `sqrt(arm * other)` over `common` samples.

        Either side may be a frame of candidate genes instead of one value, in which case the
        fixed side broadcasts and the result is one delta per candidate.
        """
        M = np.asarray(m.loc[common], dtype=float)
        R = np.asarray(r.loc[common], dtype=float)
        O = np.asarray(other.loc[common], dtype=float)
        if M.ndim > O.ndim:
            O = O[:, None]
        elif O.ndim > M.ndim:
            M, R = M[:, None], R[:, None]
        return np.nanmean(np.sqrt(M * O) - np.sqrt(R * O), axis=0)

    def _match(pool_frame, target):
        """The `band` of pool genes whose mean in this level sits closest to `target`."""
        if not pool_frame.shape[1]:
            return []
        mu = pool_frame.mean(axis=0)
        n = max(int(min_band), int(round(band * len(mu))))
        return list(mu.reindex((mu - target).abs().sort_values().index[:n]).index)

    out = []
    for key in tests["key"]:
        if key not in pinfo.index:
            continue
        r = pinfo.loc[key]
        cd4_genes = list(r.ligand_genes if r.direction == "cd4_to_stroma" else r.receptor_genes)
        str_genes = list(r.receptor_genes if r.direction == "cd4_to_stroma" else r.ligand_genes)

        cm = _side_expr(expr, meta, alt, cd4_genes)
        cr = _side_expr(expr, meta, ref, cd4_genes)
        st = _side_expr(expr, meta, r.stroma_level, str_genes)
        common = cm.index.intersection(cr.index).intersection(st.index)
        rec = {"key": key, "n_spec_samples": len(common)}
        if len(common) < cfg.MIN_DONORS_PER_ARM:
            out.append(rec)
            continue
        real = float(_stat(cm, cr, st, common))
        rec["mean_delta"] = real

        # ---- null 1: the CD4 side is replaced (rules out "the whole arm moved")
        cand = _match(_frame(alt, cd4_pool).reindex(common), float(cm.loc[common].mean()))
        if cand:
            nm = _frame(alt, cand).reindex(common)
            nr = _frame(ref, cand).reindex(common)
            null = _stat(nm, nr, st, common)
            null = null[np.isfinite(null)]
            if len(null):
                rec["spec_p_cd4_side"] = (1 + int((null >= real).sum())) / (len(null) + 1)
                rec["spec_z_cd4_side"] = ((real - null.mean()) / null.std(ddof=1)
                                          if null.std(ddof=1) > 0 else np.nan)
                rec["n_null_cd4_side"] = len(null)

        # ---- null 2: the stromal side is replaced (rules out "that level is simply deep")
        cand = _match(_frame(r.stroma_level, str_pool).reindex(common),
                      float(st.loc[common].mean()))
        if cand:
            ns = _frame(r.stroma_level, cand).reindex(common)
            null = _stat(cm, cr, ns, common)
            null = null[np.isfinite(null)]
            if len(null):
                rec["spec_p_stroma_side"] = (1 + int((null >= real).sum())) / (len(null) + 1)
                rec["spec_z_stroma_side"] = ((real - null.mean()) / null.std(ddof=1)
                                             if null.std(ddof=1) > 0 else np.nan)
                rec["n_null_stroma_side"] = len(null)
        out.append(rec)

    spec = pd.DataFrame(out)
    for c in ("spec_p_cd4_side", "spec_p_stroma_side", "spec_z_cd4_side", "spec_z_stroma_side",
              "n_null_cd4_side", "n_null_stroma_side", "mean_delta"):
        if c not in spec.columns:
            spec[c] = np.nan
    # skipna=False on purpose: a null that could not be computed is not a null the edge beat.
    spec["spec_p"] = spec[["spec_p_cd4_side", "spec_p_stroma_side"]].max(axis=1, skipna=False)
    spec["spec_z"] = spec[["spec_z_cd4_side", "spec_z_stroma_side"]].min(axis=1, skipna=False)
    res = tests.merge(spec, on="key", how="left")
    if verbose:
        n = int((res["spec_p"] < 0.05).sum())
        print(f"specificity: {n}/{int(res['spec_p'].notna().sum())} edge x level rows beat both "
              f"nulls at p < 0.05 (CD4-side pool {len(cd4_pool)} panel genes -- conservative; "
              f"stromal-side pool {len(str_pool)} cube genes)")
    return res


# ================================================================ aggregation for the pictures

def cell_edges(scores, arm=None, axis=None, stat="mean"):
    """Collapse the LR table to cell-level traffic: one weight per (source level, target level).

    The three summary pictures all want the same object -- who talks to whom and how loudly --
    and differ only in how they lay it out. `arm` restricts to one CD4 arm; leaving it None
    keeps both and returns a `cd4_arm` column so a plot can colour or difference by it.
    """
    d = scores if axis is None else scores[scores["axis"] == axis]
    if arm is not None:
        d = d[d["cd4_arm"] == arm]
    if not len(d):
        return pd.DataFrame(columns=["cd4_arm", "source", "target", "weight", "n_edges"])

    d = d.copy()
    fwd = d["direction"] == "cd4_to_stroma"
    d["source"] = np.where(fwd, d["cd4_arm"], d["stroma_level"])
    d["target"] = np.where(fwd, d["stroma_level"], d["cd4_arm"])
    per_edge = d.groupby(["cd4_arm", "source", "target", "key"], observed=True)["score"]
    per_edge = per_edge.mean() if stat == "mean" else per_edge.median()
    out = (per_edge.groupby(["cd4_arm", "source", "target"], observed=True)
           .agg(weight="sum", n_edges="size").reset_index())
    return out


def _delta_edges(scores, axis=None, ref=None, alt=None):
    """Cell-level weight for each arm side by side, plus the malignant - reactive difference."""
    ref = cfg.CD4_REACTIVE if ref is None else ref
    alt = cfg.CD4_MALIGNANT if alt is None else alt
    w = cell_edges(scores, axis=axis)
    if not len(w):
        return w
    w["stroma"] = np.where(w["source"].isin(CD4_ARMS), w["target"], w["source"])
    w["direction"] = np.where(w["source"].isin(CD4_ARMS), "cd4_to_stroma", "stroma_to_cd4")
    piv = w.pivot_table(index=["stroma", "direction"], columns="cd4_arm",
                        values="weight", aggfunc="sum").fillna(0.0)
    for a in (ref, alt):
        if a not in piv.columns:
            piv[a] = 0.0
    piv["delta"] = piv[alt] - piv[ref]
    return piv.reset_index()


def _pal(levels):
    """The notebook palette, with the two CD4 arms forced to the arm colours used everywhere."""
    col = dict(_level_colors([lv for lv in levels if lv not in CD4_ARMS]))
    col[cfg.CD4_MALIGNANT] = "#d62728"
    col[cfg.CD4_REACTIVE] = "#2ca02c"
    return {lv: col.get(lv, "#969696") for lv in levels}


# ================================================================ plots

def plot_chord(scores, axis=None, arm=None, title=None, save=None, show=True,
               min_frac=0.01, figsize=(6.0, 6.0)):
    """Chord diagram: node arc = total traffic through a level, ribbon = one directed edge.

    Read it for shape, not for significance. The ribbon width is an abundance product summed
    over the curated pairs, so a wide ribbon means "these two levels both express a lot of this
    panel in the same biopsies". The claim lives in `plot_edge_delta`, which carries the paired
    test; this figure is the map you read that table against.
    """
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    e = cell_edges(scores, arm=arm, axis=axis)
    if not len(e):
        raise ValueError("no edges to draw -- check the axis / arm filter")
    e = e.groupby(["source", "target"], observed=True)["weight"].sum().reset_index()
    e = e[e["weight"] >= e["weight"].sum() * min_frac].sort_values("weight", ascending=False)

    seen = set(e["source"]) | set(e["target"])
    nodes = ([lv for lv in CD4_ARMS if lv in seen] +
             [lv for lv in list(cfg.FIBRO_LEVELS) + list(cfg.MYELOID_LEVELS) if lv in seen] +
             sorted(seen - set(CD4_ARMS) - set(cfg.FIBRO_LEVELS) - set(cfg.MYELOID_LEVELS)))
    nodes = list(dict.fromkeys(nodes))
    tot = {n: e.loc[e["source"] == n, "weight"].sum() + e.loc[e["target"] == n, "weight"].sum()
           for n in nodes}
    grand = sum(tot.values())
    col = _pal(nodes)

    pad = np.deg2rad(3.0)
    span = 2 * np.pi - pad * len(nodes)
    ang, cursor = {}, 0.0
    for n in nodes:
        w = span * tot[n] / grand
        ang[n] = [cursor, cursor + w, cursor]          # start, end, running offset
        cursor += w + pad

    fig, ax = plt.subplots(figsize=figsize)
    R, T = 1.0, 0.06
    for n in nodes:
        a0, a1, _ = ang[n]
        th = np.linspace(a0, a1, 64)
        ax.fill(np.r_[np.cos(th) * R, np.cos(th[::-1]) * (R + T)],
                np.r_[np.sin(th) * R, np.sin(th[::-1]) * (R + T)],
                color=col[n], lw=0)
        m = (a0 + a1) / 2
        ax.text(np.cos(m) * (R + T + 0.07), np.sin(m) * (R + T + 0.07), n,
                ha="left" if np.cos(m) >= 0 else "right", va="center", fontsize=7,
                rotation=np.rad2deg(m) + (0 if np.cos(m) >= 0 else 180),
                rotation_mode="anchor")

    for _, r in e.iterrows():
        w = span * r.weight / grand
        s0 = ang[r.source][2]; ang[r.source][2] += w
        t0 = ang[r.target][2]; ang[r.target][2] += w
        N = 24
        s = np.linspace(s0, s0 + w, N)
        t = np.linspace(t0 + w, t0, N)          # reversed, so the ribbon does not self-cross
        arc = lambda a: (np.cos(a) * R, np.sin(a) * R)
        pts = [arc(a) for a in s]                                   # N
        pts += [(0, 0), arc(t[0])]                                  # bezier through the centre
        pts += [arc(a) for a in t[1:]]                              # N - 1
        pts += [(0, 0), pts[0]]                                     # and back
        codes = ([MPath.MOVETO] + [MPath.LINETO] * (N - 1) +
                 [MPath.CURVE3, MPath.CURVE3] + [MPath.LINETO] * (N - 1) +
                 [MPath.CURVE3, MPath.CURVE3])
        ax.add_patch(PathPatch(MPath(pts, codes), facecolor=col[r.source],
                               alpha=0.45, lw=0))

    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.75)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"curated LR traffic{'' if axis is None else f' -- {axis}'}",
                 fontsize=9, pad=10)
    return _finish(fig, save, show)


def plot_circle_network(scores, axis=None, title=None, save=None, show=True,
                        min_frac=0.01, figsize=(6.0, 6.0)):
    """Aggregated network: levels on a circle, arrow width = traffic, colour = malignant shift.

    One arrow per (level, direction). Width is the summed abundance product over the curated
    pairs with the malignant clone as the CD4 arm; colour is that minus the same quantity with
    the reactive CD4 arm, so red means the edge is louder around the clone and blue means it is
    louder around the reactive T cells in the same biopsies.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib import cm, colors as mcolors

    d = _delta_edges(scores, axis=axis)
    if not len(d):
        raise ValueError("no edges to draw -- check the axis filter")
    d = d[d[cfg.CD4_MALIGNANT] >= d[cfg.CD4_MALIGNANT].sum() * min_frac]

    seen = set(d["stroma"])
    stroma = ([lv for lv in cfg.FIBRO_LEVELS if lv in seen] +
              [lv for lv in cfg.MYELOID_LEVELS if lv in seen] +
              sorted(seen - set(cfg.FIBRO_LEVELS) - set(cfg.MYELOID_LEVELS)))
    nodes = [cfg.CD4_MALIGNANT] + stroma
    col = _pal(nodes)
    th = {n: a for n, a in zip(nodes, np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi,
                                                  len(nodes), endpoint=False))}
    pos = {n: (np.cos(a), np.sin(a)) for n, a in th.items()}

    vmax = float(np.abs(d["delta"]).max()) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    wmax = float(d[cfg.CD4_MALIGNANT].max()) or 1.0

    fig, ax = plt.subplots(figsize=figsize)
    for _, r in d.iterrows():
        a, b = ((cfg.CD4_MALIGNANT, r.stroma) if r.direction == "cd4_to_stroma"
                else (r.stroma, cfg.CD4_MALIGNANT))
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], connectionstyle="arc3,rad=0.18",
            arrowstyle="-|>", mutation_scale=11, shrinkA=13, shrinkB=13,
            lw=0.6 + 4.5 * r[cfg.CD4_MALIGNANT] / wmax,
            color=cmap(norm(r.delta)), alpha=0.85, zorder=1))
    for n in nodes:
        x, y = pos[n]
        ax.scatter([x], [y], s=380 if n == cfg.CD4_MALIGNANT else 220,
                   color=col[n], zorder=3, edgecolor="white", lw=1.2)
        a = th[n]
        ax.text(np.cos(a) * 1.16, np.sin(a) * 1.16, n, fontsize=7,
                ha="left" if np.cos(a) >= 0 else "right", va="center",
                rotation=np.rad2deg(a) + (0 if np.cos(a) >= 0 else 180),
                rotation_mode="anchor")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55,
                 label="malignant - reactive")
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.9)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"aggregated LR network{'' if axis is None else f' -- {axis}'}",
                 fontsize=9, pad=12)
    return _finish(fig, save, show)


def plot_hierarchy(scores, axis=None, title=None, save=None, show=True,
                   min_frac=0.01, figsize=(6.5, 4.6)):
    """Hierarchy plot: the two CD4 arms on the left, fibroblast and myeloid targets on the right.

    This is the layout that makes the comparison the section is about legible -- the same target
    column is fed by the clone and by the reactive T cells of the same biopsies, so the two
    fans are read against each other rather than in separate figures. Only the outgoing
    direction is drawn; incoming traffic is a different question and gets its own call.
    """
    import matplotlib.pyplot as plt

    e = cell_edges(scores, axis=axis)
    e = e[e["source"].isin(CD4_ARMS)]
    if not len(e):
        raise ValueError("no outgoing CD4 edges to draw -- check the axis filter")
    e = e[e["weight"] >= e["weight"].sum() * min_frac]

    fib = [lv for lv in cfg.FIBRO_LEVELS if lv in set(e["target"])]
    mye = [lv for lv in cfg.MYELOID_LEVELS if lv in set(e["target"])]
    rest = [lv for lv in dict.fromkeys(e["target"]) if lv not in fib + mye]
    col = _pal(CD4_ARMS + fib + mye + rest)

    def _col(levels, x, y0, y1):
        if not levels:
            return {}
        ys = np.linspace(y1, y0, len(levels)) if len(levels) > 1 else [(y0 + y1) / 2]
        return {lv: (x, y) for lv, y in zip(levels, ys)}

    # Share the right-hand height between the two lineages in proportion to how many levels
    # each contributes, so an axis with no myeloid target does not leave half the panel blank.
    low = mye + rest
    gap = 0.06 if (fib and low) else 0.0
    frac = len(fib) / max(len(fib) + len(low), 1)
    cut = 1.0 - frac * (1.0 - gap)
    pos = {}
    pos.update(_col(CD4_ARMS, 0.0, 0.30, 0.70))
    pos.update(_col(fib, 1.0, cut + gap, 1.00))
    pos.update(_col(low, 1.0, 0.00, cut))

    wmax = float(e["weight"].max()) or 1.0
    fig, ax = plt.subplots(figsize=figsize)
    for _, r in e.iterrows():
        if r.source not in pos or r.target not in pos:
            continue
        (x0, y0), (x1, y1) = pos[r.source], pos[r.target]
        t = np.linspace(0, 1, 60)
        x = x0 + (x1 - x0) * (3 * t ** 2 - 2 * t ** 3)
        ax.plot(x, y0 + (y1 - y0) * (3 * t ** 2 - 2 * t ** 3),
                color=col[r.source], lw=0.5 + 4.0 * r.weight / wmax, alpha=0.65,
                solid_capstyle="round")
    for lv, (x, y) in pos.items():
        ax.scatter([x], [y], s=90, color=col[lv], zorder=3, edgecolor="white", lw=1.0)
        ax.text(x + (-0.04 if x == 0 else 0.04), y, lv, fontsize=7, va="center",
                ha="right" if x == 0 else "left")
    if fib:
        ax.text(1.0, 1.06, "fibroblast", fontsize=7.5, ha="center", color="#3182bd")
    if low:
        ax.text(1.0, -0.07, "myeloid", fontsize=7.5, ha="center", color="#e6550d")

    ax.set_xlim(-0.45, 1.5); ax.set_ylim(-0.14, 1.14)
    ax.axis("off")
    ax.set_title(title or f"CD4 -> stroma{'' if axis is None else f' -- {axis}'}", fontsize=9)
    return _finish(fig, save, show)


def plot_edge_delta(tests, direction="cd4_to_stroma", title=None, save=None, show=True,
                    alpha=0.05, figsize=None):
    """The quantitative backbone: edge x stromal level, paired malignant - reactive delta.

    `*` marks q < `alpha` on the Wilcoxon signed-rank over per-sample deltas; grey is an edge
    that was never scored in that level, which is a coverage statement and not a zero. Claims
    come off this figure. The chord, network and hierarchy plots are summaries of the same
    numbers and carry no test.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    t = tests[tests["direction"] == direction]
    if not len(t):
        raise ValueError(f"no {direction} edges in the test table")
    m = t.pivot_table(index="edge", columns="stroma_level", values="median_delta")
    q = t.pivot_table(index="edge", columns="stroma_level", values="q")
    n = t.pivot_table(index="edge", columns="stroma_level", values="n_pairs")
    order = [lv for lv in list(cfg.FIBRO_LEVELS) + list(cfg.MYELOID_LEVELS) + ["Keratinocyte"]
             if lv in m.columns]
    m, q, n = m[order], q.reindex(columns=order), n.reindex(columns=order)
    # An abstained cell is a coverage statement, not an effect: blank it rather than colour it.
    m = m.mask(~np.isfinite(q.to_numpy()))

    vmax = float(np.nanmax(np.abs(m.to_numpy()))) or 1.0
    figsize = figsize or (0.45 * m.shape[1] + 3.0, 0.26 * m.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#e0e0e0")
    im = ax.imshow(m.to_numpy(), cmap=cmap, aspect="auto",
                   norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(q.iat[i, j]) and q.iat[i, j] < alpha:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="malignant - reactive")
    nmed = np.nanmedian(n.to_numpy())
    ax.set_title(title or f"{direction}: paired per-sample delta "
                          f"(* q<{alpha}; median n_pairs={nmed:.0f})", fontsize=9)
    return _finish(fig, save, show)
