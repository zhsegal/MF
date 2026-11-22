"""External healthy-skin reference for the degradome notebook (28).

Reynolds et al. Science 2021 (E-MTAB-8142), 195,739 healthy adult skin cells over 5 donors.
It answers one question that our own HC arm cannot: is what we measure in healthy CTCL-cohort
skin what healthy skin looks like anywhere else, or is it a property of these six cohorts?

What this module will and will not let you claim
------------------------------------------------
It will let you place our arms against an outside band. It will NOT give you a blocked test.
There is no study that carries both sides, so `blocked_ranksum` would find zero strata and
return NaN -- which is the correct answer, not a bug to route around. Every contrast here is
therefore donor-level, unblocked, and stamped `test="donor_level_unblocked"` with its confounds
named. Four of them, all real:

  * one cohort, one lab, one dissociation protocol;
  * 10x 3' v2 against our mostly-5' atlas;
  * all five donors female, all breast skin;
  * n = 5, which is one above MIN_DONORS_PER_ARM.

The calibration control is what makes the comparison readable anyway. `CONTROL_FAMILIES`
(mast, granulocyte, th2_loop, caspase) have no reason to differ between healthy cohorts, so the
shift they show is the platform floor. A claim family that moves no more than the controls has
not moved. `contrast_external` computes both in one call and puts the control shift on every
row, because a reader who has to look it up in a second table will not.

Normalisation is the other load-bearing part. `degradome_cells` scores on X = log1p CP10K with
the library size summed over ALL genes before the panel columns are subset. Normalising the
external object after subsetting would rescale every cell by the ratio of panel counts to total
counts and produce plausible, wrong numbers on both axes of every comparison. `build_external`
does it in the same order, from `raw/X` (CELLxGENE keeps raw counts there; X is already
normalised and is not used).

Panel coverage: 288 of 290 symbols are present. MMP26 is missing (mmp 19/20, fine) and TMEM2 is
missing, which drops `ha_catabolic` to 4 genes -- below `cfg.MIN_FAMILY_SIZE_FOR_SCORE`, so that
family abstains externally rather than being silently computed over a different gene set than
ours. `panel_coverage` returns exactly which families are in that position.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import degradome_data as cfg

LABEL_COL = "full_clustering"     # the source's own fine clustering
DONOR_COL = "donor_id"            # 5 donors; sample_id has 40 libraries over those 5
TARGET_COL = "ext_target"         # the harmonised level both sides are pooled to
CARRY_OBS = [LABEL_COL, DONOR_COL, "sample_id", "assay", "sex", "tissue", "Site", "Location"]


# ---------------------------------------------------------------- the label map
# Built from the file's own 34 categories (printed by `describe`), not from the paper: this
# deposit is the healthy-only subset and is coarser than the published full_clustering (no
# LC_1-4, no moDC_1-3, no Mono_CD14/Mono_CD16, no pDC).
#
# Targets are the granularity at which BOTH sides carry a defensible label. Where ours is finer
# than theirs the target is the coarse one and our levels are pooled up to meet it -- that is
# what TARGET_TO_OURS is for. The alternative, asserting Reynolds' F1/F2/F3 onto our
# papillary/reticular/inflammatory axis, would be a 1:1 claim neither annotation supports.
EXT_TO_TARGET = {
    # -------- lymphoid
    "Th": "CD4",
    "Treg": "CD4_Treg",
    "Tc": "CD8",
    "Plasma": "Plasma",
    # -------- myeloid
    "LC": "LC",
    "migLC": "LC",
    "DC1": "cDC1",
    "DC2": "cDC2",
    "MigDC": "DC_LAMP3",
    "moDC": "moDC",
    "Macro_1": "Mono_Mac",
    "Macro_2": "Mono_Mac",
    "Inf_mac": "Mono_Mac",
    "Mono_mac": "Mono_Mac",
    "Mast_cell": "Mast",
    # -------- stromal / structural
    "F1": "Fibroblast",
    "F2": "Fibroblast",
    "F3": "Fibroblast",
    "VE1": "Vascular",
    "VE2": "Vascular",
    "VE3": "Vascular",
    "LE1": "Vascular",
    "LE2": "Vascular",
    "Pericyte_1": "Vascular",
    "Pericyte_2": "Vascular",
    "Differentiated_KC": "Keratinocyte",
    "Undifferentiated_KC": "Keratinocyte",
    "Melanocyte": "Melanocyte",
}

# Named, not silently dropped: an external label with no counterpart in our roster.
DROPPED_EXT_LABELS = {
    "NK": "no NK level in the nb22/nb21 roster (nb22b would add one)",
    "ILC1_3": "no ILC level in the roster",
    "ILC1_NK": "no ILC level in the roster",
    "ILC2": "no ILC level in the roster",
    "Schwann_1": "no Schwann level in the roster",
    "Schwann_2": "no Schwann level in the roster",
}

# Our degradome_levels pooled to each target. CD4 is deliberately absent: its arms
# (malignant / reactive / unassessed / HC) are the question, so nb28 supplies them one at a
# time rather than pooling them into a single comparator.
TARGET_TO_OURS = {
    "CD4_Treg": ["CD4_Treg"],
    "CD8": ["CD8"],
    "Plasma": ["Plasma"],
    "LC": ["LC"],
    "cDC1": ["cDC1"],
    "cDC2": ["cDC2"],
    "DC_LAMP3": ["DC_LAMP3"],
    "moDC": ["moDC"],
    "Mono_Mac": ["Mono_CD14", "Mono_CD16", "Mac_FOLR2", "Mac_SPP1_TREM2", "Mac_infl"],
    "Mast": ["Mast"],
    "Fibroblast": ["F_papillary", "F_reticular", "F_mesenchymal", "F_myofibroblast",
                   "F_inflammatory", "F_apCAF"],
    "Vascular": ["Vascular"],
    "Keratinocyte": ["Keratinocyte"],
    "Melanocyte": ["Melanocyte"],
}
CD4_TARGET = "CD4"

# Our roster levels with no external counterpart in this deposit.
NO_EXTERNAL = ["pDC", "Neutrophils", "B"]


def describe(path=None, verbose=True):
    """Print what the file actually carries. Run this before trusting EXT_TO_TARGET."""
    import h5py

    path = cfg.EXT_H5AD if path is None else Path(path)
    out = {}
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        for k in obs.keys():
            g = obs[k]
            if isinstance(g, h5py.Group) and "categories" in g:
                cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
                if len(cats) <= 64:
                    out[k] = cats
        out["_shape"] = list(f["X"].attrs["shape"])
    if verbose:
        print(f"{path.name}: {out['_shape'][0]:,} cells x {out['_shape'][1]:,} genes")
        for k in (LABEL_COL, DONOR_COL, "disease", "assay", "sex", "tissue"):
            if k in out:
                print(f"  {k} ({len(out[k])}): {out[k]}")
    return out


def assert_label_map(labels, verbose=True):
    """Every observed label is either mapped or explicitly dropped -- no silent third case."""
    labels = set(map(str, labels))
    known = set(EXT_TO_TARGET) | set(DROPPED_EXT_LABELS)
    unknown = sorted(labels - known)
    assert not unknown, (
        f"external labels with neither a target nor a drop reason: {unknown}. Add them to "
        f"EXT_TO_TARGET or to DROPPED_EXT_LABELS -- a label must not fall through.")
    stale = sorted(known - labels)
    if stale and verbose:
        print(f"  note: {len(stale)} mapped/dropped labels not present in this file: {stale}")
    return True


def panel_coverage(var_symbols, verbose=True):
    """Which families survive the external gene space, and which must abstain.

    A family scored over a different gene set on each side is not a comparison. Families that
    fall below cfg.MIN_FAMILY_SIZE_FOR_SCORE here are returned in `abstain` and nb28 refuses
    them rather than reporting a number computed over four genes against our five.
    """
    have = set(map(str, var_symbols))
    rows, abstain = [], {}
    for fam, genes in cfg.DEGRADOME_FAMILIES.items():
        miss = [g for g in genes if g not in have]
        n_present = len(genes) - len(miss)
        rows.append({"family": fam, "n_panel": len(genes), "n_present": n_present,
                     "missing": ",".join(miss)})
        if n_present < cfg.MIN_FAMILY_SIZE_FOR_SCORE <= len(genes):
            abstain[fam] = (f"{n_present}/{len(genes)} genes in the external var "
                            f"(< MIN_FAMILY_SIZE_FOR_SCORE={cfg.MIN_FAMILY_SIZE_FOR_SCORE}); "
                            f"missing {miss}")
    cov = pd.DataFrame(rows)
    # the balance ratios quietly depend on gene presence too
    bal_missing = {}
    for key, (num, den) in cfg.BALANCE_PAIRS.items():
        m = [g for g in cfg.family_genes(num) + list(den) if g not in have]
        if m:
            bal_missing[key] = m
    if verbose:
        n_miss = int((cov["n_present"] < cov["n_panel"]).sum())
        print(f"panel coverage: {int(cov['n_present'].sum())}/{int(cov['n_panel'].sum())} "
              f"family genes present, {n_miss} families incomplete")
        for fam, why in abstain.items():
            print(f"  ABSTAIN {fam}: {why}")
        for k, m in bal_missing.items():
            print(f"  balance {k}: missing {m} (ratio computed over a reduced gene set)")
    return cov, abstain, bal_missing


def build_external(path=None, out=None, chunk=20_000, verbose=True):
    """Build the external panel object with OUR normalisation contract.

    raw counts -> collapse duplicate symbols -> CP10K over the FULL gene width -> log1p ->
    subset to cfg.panel_genes(). The order is the whole point; see the module docstring.

    Streams the CSR in row blocks: the full 195,739 x 27,867 count matrix does not need to be
    resident, and this runs on a job node rather than the login node either way.
    """
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    path = cfg.EXT_H5AD if path is None else Path(path)
    out = cfg.EXT_DEG_ADATA if out is None else Path(out)

    with h5py.File(path, "r") as f:
        n_obs, n_var = (int(x) for x in f["X"].attrs["shape"])

        def _cat(grp):
            if isinstance(grp, h5py.Group):
                cats = np.array([c.decode() if isinstance(c, bytes) else c
                                 for c in grp["categories"][:]])
                return cats[grp["codes"][:]]
            return np.array([c.decode() if isinstance(c, bytes) else c for c in grp[:]])

        # Symbols come from var, counts from raw/X -- so assert the two var frames are the
        # same frame before trusting a column index built on one against a matrix from the
        # other. (They are here, and raw is the right matrix: CXG zeroes `feature_is_filtered`
        # genes in X but keeps them in raw, and some of the panel is filtered.)
        assert np.array_equal(_cat(f["var"]["_index"]), _cat(f["raw"]["var"]["_index"])), \
            "var and raw/var are not the same gene frame -- the column index would be wrong"
        symbols = _cat(f["var"]["feature_name"])
        # AnnData records the index dataset's NAME in the group's `_index` attribute; it is
        # only literally "_index" when the frame had an unnamed index. This file's is "index".
        obs_index = f["obs"].attrs.get("_index", "_index")
        obs = pd.DataFrame({c: _cat(f["obs"][c]) for c in CARRY_OBS if c in f["obs"]},
                           index=pd.Index(_cat(f["obs"][obs_index]), name="cell_id"))
        assert len(obs) == n_obs, (len(obs), n_obs)

        # ---- symbol space. 12 symbols are duplicated in var; counts are summed, which is the
        # only correct collapse for counts and keeps the library size honest.
        sym = pd.Index(symbols)
        uniq = sym.unique()
        sym_code = pd.Categorical(sym, categories=uniq).codes.astype(np.int64)
        collapse = sp.csr_matrix(
            (np.ones(n_var, dtype=np.float32), (sym_code, np.arange(n_var))),
            shape=(len(uniq), n_var))

        panel = [g for g in cfg.panel_genes() if g in set(uniq)]
        keep = pd.Index(uniq).get_indexer(panel)
        if verbose:
            print(f"{path.name}: {n_obs:,} x {n_var:,} -> {len(uniq):,} symbols; "
                  f"panel {len(panel)}/{len(cfg.panel_genes())}")

        # ---- stream the raw CSR
        X = f["raw"]["X"]
        indptr = X["indptr"][:]
        blocks_norm, blocks_cnt = [], []
        for start in range(0, n_obs, chunk):
            stop = min(start + chunk, n_obs)
            lo, hi = int(indptr[start]), int(indptr[stop])
            block = sp.csr_matrix(
                (X["data"][lo:hi].astype(np.float32),
                 X["indices"][lo:hi], indptr[start:stop + 1] - lo),
                shape=(stop - start, n_var))
            block = block @ collapse.T                      # -> symbol space, counts summed
            lib = np.asarray(block.sum(axis=1)).ravel()     # FULL width, BEFORE the subset
            lib[lib == 0] = 1.0
            sub = block[:, keep].tocsr()
            blocks_cnt.append(sub.astype(np.float32))
            norm = sp.diags(1e4 / lib) @ sub
            norm.data = np.log1p(norm.data)
            blocks_norm.append(norm.astype(np.float32))
            if verbose and (start // chunk) % 4 == 0:
                print(f"  {stop:,}/{n_obs:,}")

    lognorm = sp.vstack(blocks_norm).tocsr()
    counts = sp.vstack(blocks_cnt).tocsr()

    assert_label_map(obs[LABEL_COL], verbose=verbose)
    obs[TARGET_COL] = pd.Categorical(
        obs[LABEL_COL].astype(str).map(EXT_TO_TARGET).fillna("DROPPED"))
    cov, abstain, bal_missing = panel_coverage(panel, verbose=verbose)

    adata = ad.AnnData(X=lognorm, layers={cfg.COUNTS_LAYER: counts}, obs=obs,
                       var=pd.DataFrame(index=pd.Index(panel, name="symbol")))
    adata.uns["external_ref"] = {
        "name": cfg.EXT_NAME,
        "source_file": str(path),
        "dataset_id": cfg.EXT_DATASET_ID,
        "accession": cfg.EXT_ACCESSION,
        "doi": cfg.EXT_DOI,
        "layer_semantics": (f"X = log1p CP10K, library size summed over all {len(uniq)} "
                            f"symbols BEFORE the panel subset -- the same order as "
                            f"degradome_helpers.build_degradome_object"),
        "counts_source": "raw/X (CELLxGENE raw counts); X in the source is already normalised "
                         "and is not used",
        "panel_present": len(panel),
        "panel_total": len(cfg.panel_genes()),
        "panel_missing": [g for g in cfg.panel_genes() if g not in set(uniq)],
        "families_abstaining": abstain,
        "balance_pairs_missing_genes": {k: list(v) for k, v in bal_missing.items()},
        "label_col": LABEL_COL,
        "target_col": TARGET_COL,
        "dropped_labels": DROPPED_EXT_LABELS,
        "no_external_counterpart": NO_EXTERNAL,
        "confounds": ["single cohort/lab", "10x 3' v2 vs our mostly 5'", "all female",
                      "breast skin only", "n_donors=5"],
    }
    kept = adata[adata.obs[TARGET_COL].astype(str) != "DROPPED"].copy()
    if verbose:
        print(f"kept {kept.n_obs:,}/{adata.n_obs:,} cells over "
              f"{kept.obs[TARGET_COL].nunique()} targets; dropped "
              f"{adata.n_obs - kept.n_obs:,} ({sorted(DROPPED_EXT_LABELS)})")
        print(kept.obs[TARGET_COL].value_counts().to_string())
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        kept.write_h5ad(out, compression="gzip")
        cov.to_csv(cfg.tab("external_panel_coverage"), index=False)
        if verbose:
            print(f"wrote {out}  {kept.shape}")
    return kept


def load_external(path=None):
    """Read the built external object, or say how to make it."""
    import anndata as ad

    path = cfg.EXT_DEG_ADATA if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ./jobs/download_external_skin.sh then "
            f"./jobs/run_external_degradome_build.sh (bsub -- not the login node).")
    return ad.read_h5ad(path)


def load_external_donor_median(path=None):
    """The per-(target, donor) medians written by jobs/run_external_degradome_build.py.

    nb28 reads these rather than re-scoring 195,739 cells: the band it draws and the test it
    runs are both donor-level, so the cell matrix is not needed in the notebook at all.
    """
    path = cfg.tab("external_donor_median") if path is None else Path(path)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run ./jobs/download_external_skin.sh then "
            f"./jobs/run_external_degradome_build.sh (bsub).")
    return pd.read_csv(path)


def external_abstentions(path=None, verbose=True):
    """Families that must abstain externally because the gene set is not the same one.

    Returns (abstain_families, score_cols_to_drop). Reads the coverage table the build wrote,
    so the notebook does not have to open the external object to know what it may not compare.
    """
    path = cfg.tab("external_panel_coverage") if path is None else Path(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run the external build first.")
    cov = pd.read_csv(path)
    bad = cov[(cov["n_present"] < cfg.MIN_FAMILY_SIZE_FOR_SCORE)
              & (cov["n_panel"] >= cfg.MIN_FAMILY_SIZE_FOR_SCORE)]
    fams = bad["family"].tolist()
    if verbose and fams:
        for _, r in bad.iterrows():
            print(f"  ABSTAIN fam_{r['family']}: {r['n_present']}/{r['n_panel']} genes present "
                  f"externally (missing {r['missing']}) -- the two sides would be different "
                  f"gene sets, which is not a comparison")
    return fams, [f"fam_{f}" for f in fams]


def external_donor_median(adata, score_cols, min_cells=None):
    """Per-(target, donor) medians, through the same helper our own arms go through."""
    import degradome_cells as DC

    dmed = DC.donor_median(adata, score_cols, groupby=TARGET_COL, donor_key=DONOR_COL,
                           min_cells=min_cells, extra=("assay", "sex", "tissue"))
    dmed = dmed.rename(columns={TARGET_COL: "target"})
    dmed["arm"] = "external_HC"
    return dmed


def _our_target_medians(dmed, target, our_levels, arm_col=None, arm=None, groupby=None):
    """Pool our per-(level, donor) medians up to one value per donor at `target`."""
    groupby = cfg.GROUPBY if groupby is None else groupby
    d = dmed[dmed["eligible"] & dmed[groupby].astype(str).isin(our_levels)]
    if arm_col is not None and arm is not None:
        d = d[d[arm_col].astype(str) == str(arm)]
    num = d.select_dtypes("number").columns
    return d.groupby(cfg.DONOR_KEY, observed=True)[list(num)].median()


def contrast_external(dmed, ext_dmed, score_cols, target, our_levels=None, arm_col=None,
                      arm=None, groupby=None, donor_col=None, female_only=None,
                      control_scores=None, fdr=True):
    """Donor-level Mann-Whitney of our arm against the external cohort at one target.

    Deliberately NOT routed through `compare_levels`. That function's contract is a
    study-blocked p; there is no shared study here, so returning an unblocked number under the
    same column name would make an unblockable comparison look like a blocked one. Every row
    carries `test`, `confound`, and `control_shift` instead.

    `control_shift` is the median |delta| the CONTROL_FAMILIES show at this same target. Read
    every claim delta against it: a claim family that moves no further than the controls has
    not moved, it has changed chemistry. `female_only`, if given, is the set of our donors that
    are female -- the same contrast restricted to them, reported as `delta_female` rather than
    as a second table.
    """
    from scipy import stats

    donor_col = cfg.DONOR_KEY if donor_col is None else donor_col
    our_levels = TARGET_TO_OURS.get(target, [target]) if our_levels is None else our_levels
    ours = _our_target_medians(dmed, target, our_levels, arm_col, arm, groupby)
    ext = ext_dmed[ext_dmed["eligible"] & (ext_dmed["target"].astype(str) == str(target))]
    ext = ext.set_index(DONOR_COL)

    control_scores = ([f"fam_{f}" for f in cfg.CONTROL_FAMILIES]
                      if control_scores is None else list(control_scores))

    def _one(sc):
        a = ours[sc].dropna().to_numpy() if sc in ours else np.array([])
        b = ext[sc].dropna().to_numpy() if sc in ext else np.array([])
        if len(a) < cfg.MIN_DONORS_PER_ARM or len(b) < cfg.MIN_DONORS_PER_ARM:
            return None
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"n_donors_ours": len(a), "n_donors_ext": len(b),
                "median_ours": float(np.median(a)), "median_ext": float(np.median(b)),
                "delta": float(np.median(a) - np.median(b)),
                "u": float(u.statistic), "p": float(u.pvalue)}

    ctrl = [abs(r["delta"]) for r in (_one(sc) for sc in control_scores) if r is not None]
    control_shift = float(np.median(ctrl)) if ctrl else np.nan

    rows = []
    for sc in score_cols:
        rec = {"target": target, "our_levels": "|".join(our_levels), "arm": arm or "all",
               "score": sc, "test": "donor_level_unblocked",
               "confound": "cohort+platform+sex+site", "control_shift": control_shift}
        r = _one(sc)
        if r is None:
            rec.update({"n_donors_ours": int(len(ours)), "n_donors_ext": int(len(ext)),
                        "median_ours": np.nan, "median_ext": np.nan, "delta": np.nan,
                        "u": np.nan, "p": np.nan,
                        "reason": f"< {cfg.MIN_DONORS_PER_ARM} eligible donors on one side"})
        else:
            rec.update(r)
            rec["reason"] = ""
            rec["exceeds_control"] = bool(abs(r["delta"]) > control_shift) \
                if np.isfinite(control_shift) else None
        if female_only is not None and r is not None:
            af = ours.loc[ours.index.isin(female_only), sc].dropna().to_numpy()
            rec["n_donors_ours_female"] = int(len(af))
            rec["delta_female"] = (float(np.median(af) - r["median_ext"])
                                   if len(af) >= cfg.MIN_DONORS_PER_ARM else np.nan)
            rec["female_same_direction"] = (
                bool(np.sign(rec["delta_female"]) == np.sign(r["delta"]))
                if np.isfinite(rec.get("delta_female", np.nan)) else None)
        rows.append(rec)

    res = pd.DataFrame(rows)
    res["q"] = np.nan
    if fdr and len(res):
        from statsmodels.stats.multitest import multipletests
        m = res["p"].notna()
        if m.any():
            res.loc[m, "q"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
    return res


def provenance(path=None):
    """The fetch record written by jobs/download_external_skin.sh."""
    path = cfg.EXT_PROV if path is None else Path(path)
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}
