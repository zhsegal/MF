"""Functions for the skin TME degradome analysis (notebook 42).

Split in two halves.

  BUILD (LSF only, `jobs/run_degradome_build.py`): label construction and the one streaming
  pass over `joint_annotated.h5ad`. Nothing here is called from the notebook except the
  asserts.

  ANALYSIS (notebook): coverage gate, panel coverage, source attribution, spillover control,
  the CTCL-vs-HC / layer / stage contrasts, the lesion-restricted table, the clonal axis
  (nb31's nested subclones), figures and the run log.

Everything that already exists is imported, not rewritten: `ccc_helpers.stream_lognorm_subset`
does the streaming pass (and gets the library size right -- see its docstring),
`ccc_helpers.build_ccc_celltype` does the CD4 malignant/reactive/unassessed split, and
`subclone_helpers.run_pydeseq2` runs the pseudobulk DESeq2.

The one thing this module does differently from `ccc_utils.build_ccc_celltype_sub` is that it
does NOT apply a KEEP_LEVELS drop. nb40 narrows to a claim roster because a ligand-receptor
grid is quadratic in the number of levels. Here the levels outside the claim roster are the
denominator: "cDC produces 40% of the skin's MMP12" is only meaningful if keratinocytes,
mast cells and CD8 are counted in the other 60%. They are carried, and flagged claim=False.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import degradome_data as cfg

# ================================================================ cache


def _cache_stamp(src=None):
    """Identity of the object every cached result was computed from."""
    src = cfg.DEG_ADATA if src is None else Path(src)
    st = src.stat()
    return {"source": src.name, "mtime": int(st.st_mtime), "size": int(st.st_size)}


def cached(name, build, force=False, verbose=True, src=None):
    """Disk-cached result, invalidated when the built object changes.

    `build` is a zero-argument callable. DataFrames land as parquet, anything JSON-serialisable
    as json; a sidecar stamp records the source object's mtime and size, so a rebuilt
    `skin_degradome.h5ad` silently drops every cache rather than serving numbers computed from
    a different object.

    This exists because the three DESeq2 contrasts are ~30 of the notebook's 37 minutes and
    nothing about them changes when a downstream plot is edited.
    """
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp_p = cfg.CACHE_DIR / f"{name}.stamp.json"
    pq_p, js_p = cfg.CACHE_DIR / f"{name}.parquet", cfg.CACHE_DIR / f"{name}.json"
    want = _cache_stamp(src)

    if not force and stamp_p.exists():
        got = json.loads(stamp_p.read_text())
        if got == want:
            if pq_p.exists():
                if verbose:
                    print(f"[cache] {name} <- {pq_p.name}")
                return pd.read_parquet(pq_p)
            if js_p.exists():
                if verbose:
                    print(f"[cache] {name} <- {js_p.name}")
                return json.loads(js_p.read_text())
        elif verbose:
            print(f"[cache] {name} stale ({got.get('mtime')} != {want['mtime']}), recomputing")

    out = build()
    if isinstance(out, pd.DataFrame):
        out.to_parquet(pq_p, index=False)
        js_p.unlink(missing_ok=True)
    else:
        js_p.write_text(json.dumps(out, indent=2, default=str))
        pq_p.unlink(missing_ok=True)
    stamp_p.write_text(json.dumps(want, indent=2))
    if verbose:
        print(f"[cache] {name} -> written")
    return out


def clear_cache(names=None, verbose=True):
    """Delete cached results. `names=None` clears everything."""
    if not cfg.CACHE_DIR.exists():
        return []
    gone = []
    for f in sorted(cfg.CACHE_DIR.iterdir()):
        if names is None or f.name.split(".")[0] in set(names):
            f.unlink()
            gone.append(f.name)
    if verbose:
        print(f"cleared {len(gone)} cache files")
    return gone


# ================================================================ labels


def disease_group(obs, key=cfg.DISEASE_KEY):
    """disease -> "CTCL" | "HC" | "other" (the arm variable for the section 4 contrast)."""
    d = obs["disease"].astype(str)
    out = np.where(d.isin(cfg.CTCL_DISEASES), "CTCL",
                   np.where(d == cfg.HC_DISEASE, "HC", "other"))
    return pd.Series(out, index=obs.index, name=key)


def load_subtype_sidecar(path=None, verbose=True):
    """nb10c cell_id -> subtype_ccc, nulls already removed. 153,164 usable of 160,195."""
    path = cfg.SUBTYPE_CSV if path is None else Path(path)
    side = pd.read_csv(path, dtype=str)
    n_all = len(side)
    bad = side[cfg.SUBTYPE_SRC].isna() | side[cfg.SUBTYPE_SRC].isin(["nan", "None", "NA", ""])
    side = side[~bad]
    if verbose:
        print(f"nb10c sidecar: {len(side):,} of {n_all:,} myeloid/fibroblast cells carry a "
              f"{cfg.SUBTYPE_SRC} ({int(bad.sum()):,} dropped: UNK / proliferating / pericyte)")
    return side.set_index("cell_id")[cfg.SUBTYPE_SRC]


def build_degradome_level(obs, subtype_csv=None, verbose=True):
    """cell_type_final + mal_tcr_alice, with Myeloid/Fibroblast replaced by nb10c subtype_ccc.

    Same construction as `ccc_utils.build_ccc_celltype_sub` minus the KEEP_LEVELS drop: the
    context levels survive because they are the attribution denominator. Returns a categorical
    Series in `cfg.CT_ORDER` order, NaN for cells with no identity (nb10b UNK, or a
    myeloid/fibroblast cell whose nb10c subtype is null).
    """
    import ccc_helpers as H

    base = H.build_ccc_celltype(
        obs,
        key=cfg.GROUPBY,
        celltype_src=cfg.CELLTYPE_SRC,
        malig_src=cfg.MALIG_SRC,
        tcr_assessed_src=cfg.TCR_ASSESSED_SRC,
        keep_unassessed=True,
        drop_levels=cfg.DROP_LEVELS,
        verbose=verbose,
    )

    fine = load_subtype_sidecar(subtype_csv, verbose=verbose)
    ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else obs.index.astype(str)
    mapped = pd.Series(fine.reindex(pd.Index(ids)).to_numpy(), index=obs.index)
    is_split = obs[cfg.CELLTYPE_SRC].astype(str).isin(cfg.SPLIT_LINEAGES)

    label = base.astype(object).mask(is_split, mapped)
    # A myeloid/fibroblast cell the sidecar does not cover has no identity at this resolution.
    # It cannot go back to the pooled level -- that level no longer exists in this vocabulary.
    label = label.where(label.isin(cfg.ALL_LEVELS), np.nan)

    order = [lv for lv in cfg.CT_ORDER if lv in set(label.dropna())]
    out = pd.Series(pd.Categorical(label, categories=order), index=obs.index, name=cfg.GROUPBY)
    if verbose:
        n_drop = int(out.isna().sum())
        print(f"\n{cfg.GROUPBY}: {len(out) - n_drop:,} cells labelled, "
              f"{n_drop:,} dropped ({n_drop / len(out):.1%})")
        print(out.value_counts().to_string())
    return out


def assert_degradome_roster(labels, obs=None, sidecar_counts=None, verbose=True):
    """Raise unless the 13 nb10c levels reconcile cell-for-cell against the sidecar.

    Catches a stale `cfg.MYELOID_LEVELS`/`FIBRO_LEVELS`, a re-run of nb10c that moved cells,
    and a cell_id join that silently matched nothing. Returns the reconciliation frame.
    """
    sidecar_counts = cfg.SIDECAR_COUNTS if sidecar_counts is None else sidecar_counts
    vc = labels.value_counts()

    missing = [lv for lv in cfg.TME_LEVELS if lv not in vc.index]
    if missing:
        raise ValueError(f"nb10c levels absent from the built label: {missing} -- the cell_id "
                         f"join failed or nb10c was re-run with a different vocabulary")

    rows = []
    for lv in cfg.TME_LEVELS:
        got, want = int(vc[lv]), int(sidecar_counts[lv])
        rows.append({"level": lv, "built": got, "sidecar": want, "delta": got - want})
    recon = pd.DataFrame(rows)

    bad = recon[recon["delta"] != 0]
    if len(bad):
        raise ValueError(
            "nb10c reconciliation failed -- the built label and the sidecar disagree:\n"
            f"{bad.to_string(index=False)}\n"
            "If nb10c was deliberately re-run, update degradome_data.SIDECAR_COUNTS."
        )

    extra = sorted(set(labels.dropna().unique()) - set(cfg.ALL_LEVELS))
    if extra:
        raise ValueError(f"levels not in degradome_data.ALL_LEVELS: {extra}")

    if verbose:
        print(f"roster reconciled: {len(cfg.TME_LEVELS)} nb10c levels, "
              f"{int(recon['built'].sum()):,} cells, zero drift")
    return recon


def assert_panel(var_names, include_context=True):
    """Raise unless every configured symbol is present. Run before anything else."""
    have = set(map(str, var_names))
    want = cfg.panel_genes(include_context=include_context)
    missing = [g for g in want if g not in have]
    if missing:
        raise ValueError(f"{len(missing)} panel genes absent from the object: {missing}")
    return want


def assert_nb31_panel():
    """Raise if `cfg.NB31_PANEL` has drifted from `subclone_helpers.PAPER_PANELS["protease"]`.

    The whole point of this notebook is to answer a question nb31 posed, so the two panels
    have to be the same list. If nb31's panel is edited, this fires rather than the comparison
    quietly becoming apples-to-oranges.
    """
    import subclone_helpers as S

    theirs = list(S.PAPER_PANELS["protease"])
    if theirs != list(cfg.NB31_PANEL):
        raise ValueError(
            "degradome_data.NB31_PANEL no longer matches subclone_helpers.PAPER_PANELS"
            f"['protease']:\n  nb31: {theirs}\n  here: {list(cfg.NB31_PANEL)}"
        )
    return theirs


# ================================================================ build (LSF only)


def _pseudobulk_meta(obs, group_levels, groupby=None, sample_key=None):
    """One row per (level, sample) pseudo-sample with the design metadata DESeq2 needs."""
    groupby = cfg.GROUPBY if groupby is None else groupby
    sample_key = cfg.SAMPLE_KEY if sample_key is None else sample_key
    design = [groupby, sample_key, cfg.DONOR_KEY, "real_donor", cfg.STUDY_KEY, "disease",
              cfg.DISEASE_KEY, "entity", "stage_group", "stage_class", "skin_layer", "tissue",
              "sex", "tech", "treatment_context", "lesion_type"]
    design = [c for c in design if c in obs.columns]

    tmp = obs.assign(_g=obs[groupby].astype(str) + "||" + obs[sample_key].astype(str))
    n_cells = tmp.groupby("_g", observed=True).size().rename("n_cells")
    lib = tmp.groupby("_g", observed=True)["total_counts"].sum().rename("total_counts")
    med = tmp.groupby("_g", observed=True)["total_counts"].median().rename("median_lib")
    meta = (tmp.drop_duplicates("_g").set_index("_g")[design]
            .join(n_cells).join(lib).join(med).reindex(group_levels))
    meta.index.name = "pseudo_sample"
    meta["claim"] = meta[groupby].isin(cfg.CLAIM_LEVELS)
    return meta


def build_degradome_object(source=None, obs_parquet=None, out=None, pseudobulk_out=None,
                           pseudobulk_meta_out=None, testset_out=None, manifest_out=None,
                           chunk=50_000, verbose=True):
    """One streaming pass -> the degradome object + a FULL-GENE (level x sample) count cube.

    Called only from `jobs/run_degradome_build.py`. Two things it must get right, both
    inherited from `ccc_helpers.stream_lognorm_subset`:

      * the library size is summed over all 40,821 genes BEFORE the columns are subset.
        Normalising after subsetting would rescale every cell by the ratio of panel counts to
        total counts and produce plausible, wrong numbers everywhere downstream;
      * the pseudobulk cube is accumulated in the same pass over the FULL gene space, so
        DESeq2 estimates dispersions on the whole transcriptome rather than on 156 columns
        (which would be a different, and much worse, size-factor estimate).
    """
    import anndata as ad
    import ccc_helpers as H

    source = cfg.SOURCE_H5AD if source is None else Path(source)
    obs_parquet = cfg.OBS_PARQUET if obs_parquet is None else Path(obs_parquet)
    out = cfg.DEG_ADATA if out is None else Path(out)
    pseudobulk_out = cfg.PSEUDOBULK_PQ if pseudobulk_out is None else Path(pseudobulk_out)
    pseudobulk_meta_out = (cfg.PSEUDOBULK_META if pseudobulk_meta_out is None
                           else Path(pseudobulk_meta_out))
    manifest_out = cfg.MANIFEST if manifest_out is None else Path(manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- obs + label
    obs = H.load_ccc_obs(obs_parquet, compartment=cfg.COMPARTMENT, cols=cfg.OBS_COLS,
                         verbose=verbose)
    obs = obs.set_index("cell_id", drop=False)
    n_skin = len(obs)
    label = build_degradome_level(obs, verbose=verbose)
    obs[cfg.GROUPBY] = label
    obs = obs[label.notna()].copy()
    obs[cfg.DISEASE_KEY] = disease_group(obs)
    recon = assert_degradome_roster(obs[cfg.GROUPBY], verbose=verbose)

    # ---- genes
    var = H.read_source_var(source)
    if not var.index.is_unique:
        raise ValueError("source var index is not unique")
    genes = assert_panel(var.index)
    keep_gene_mask = np.asarray(var.index.isin(genes))
    if verbose:
        print(f"keeping {int(keep_gene_mask.sum())} of {len(var)} genes")

    # ---- rows
    source_index = H.read_source_index(source)
    rows, cell_ids = H.resolve_rows(source_index, obs["cell_id"].to_numpy())
    obs = obs.loc[cell_ids]

    groups = (obs[cfg.GROUPBY].astype(str) + "||" + obs[cfg.SAMPLE_KEY].astype(str)).to_numpy()
    group_levels = pd.unique(groups)
    group_codes = pd.Categorical(groups, categories=group_levels).codes.astype(np.int64)

    # ---- the pass
    lognorm, counts, cube = H.stream_lognorm_subset(
        source, rows, keep_gene_mask, counts_layer=cfg.COUNTS_LAYER, chunk=chunk,
        group_codes=group_codes, n_groups=len(group_levels), verbose=verbose,
    )

    adata = ad.AnnData(X=lognorm, layers={cfg.COUNTS_LAYER: counts},
                       obs=obs.drop(columns=["cell_id"]), var=var.loc[keep_gene_mask].copy())
    adata.uns["degradome_build"] = {
        "panel_size": int(keep_gene_mask.sum()),
        "families": {k: len(v) for k, v in cfg.DEGRADOME_FAMILIES.items()},
        "layer_semantics": f"X = {cfg.LAYER} (log1p CP10K, size factor over all "
                           f"{len(var)} genes, computed BEFORE the column subset)",
        "warning": "DEGRADOME-ONLY object: var is a hand-curated 156-gene panel. Not for "
                   "HVG selection, PCA, UMAP, clustering or genome-wide DE. Use "
                   "skin_degradome_pseudobulk_full.parquet for anything that needs all genes.",
        "claim_levels": list(cfg.CLAIM_LEVELS),
        "context_levels": list(cfg.CONTEXT_LEVELS),
    }
    H.sanitize_obs(adata, verbose=verbose)
    adata.write_h5ad(out, compression="gzip")
    if verbose:
        print(f"wrote {out}  {adata.shape}")

    # ---- full-gene pseudobulk (free by-product of the pass)
    cube_df = pd.DataFrame(cube, index=group_levels, columns=var.index).astype(np.int64)
    meta = _pseudobulk_meta(obs, group_levels)
    cube_df.to_parquet(pseudobulk_out)
    meta.to_parquet(pseudobulk_meta_out)
    if verbose:
        print(f"wrote pseudobulk cube {cube_df.shape} -> {pseudobulk_out.name}")

    if testset_out is not None:
        write_degradome_testset(obs, var, out=testset_out, chunk=chunk, verbose=verbose)

    manifest = {
        "source": str(source),
        "obs_parquet": str(obs_parquet),
        "subtype_csv": str(cfg.SUBTYPE_CSV),
        "compartment": cfg.COMPARTMENT,
        "n_skin_cells": int(n_skin),
        "n_obs": int(adata.n_obs),
        "n_dropped_no_identity": int(n_skin - adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_genes_source": int(len(var)),
        "nnz_lognorm": int(lognorm.nnz),
        "nnz_counts": int(counts.nnz),
        "level_counts": adata.obs[cfg.GROUPBY].value_counts().to_dict(),
        "nb10c_reconciliation": recon.to_dict("records"),
        "n_donors": int(adata.obs[cfg.DONOR_KEY].nunique()),
        "n_samples": int(adata.obs[cfg.SAMPLE_KEY].nunique()),
        "pseudobulk_shape": list(cube_df.shape),
        "testset": None if testset_out is None else str(testset_out),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, default=str))
    if verbose:
        print(f"wrote manifest -> {manifest_out.name}")
    return adata, manifest


def write_degradome_testset(obs, var, out=None, testset_donors=None, chunk=50_000,
                            verbose=True):
    """Small FULL-GENE object for the equivalence gate. Build-time only.

    Its only purpose is to prove that restricting var to the 156-gene panel does not change a
    single expression value -- i.e. that the size factor was summed over all 40,821 genes and
    not over the panel. Delegates to `ccc_helpers._write_testset`, which is exactly this code
    already (private only because nb34 is its sole caller); passing explicit donors avoids its
    ccc_celltype-dependent auto-selection branch.

    Donors: the largest in each of li2024 (CTCL, the biggest cohort), chennareddy2025 and
    gaydosik2019 (the two studies carrying both disease arms), so the gate covers the cells
    the section 4 contrast actually uses.
    """
    import ccc_helpers as H

    out = cfg.TESTSET_H5AD if out is None else Path(out)
    if testset_donors is None:
        wanted = ["li2024"] + list(cfg.PAIRED_STUDIES)
        by_study = (obs[obs[cfg.GROUPBY].astype(str).isin(cfg.TME_LEVELS)]
                    .groupby(cfg.STUDY_KEY, observed=True)[cfg.DONOR_KEY]
                    .agg(lambda s: s.value_counts().idxmax()))
        testset_donors = [by_study[s] for s in wanted if s in by_study.index]
    return H._write_testset(cfg.SOURCE_H5AD, obs, var, out,
                            testset_donors=testset_donors, chunk=chunk, verbose=verbose)


def assert_build_equivalence(adata, testset_path=None, tol=1e-6, verbose=True):
    """Raise unless the panel columns of the built object equal the full-gene test set.

    BLOCKING, and the one check that cannot be skipped: normalising after the column subset
    rescales every cell by (panel counts / total counts) -- roughly 40x here -- and the result
    looks entirely plausible. Every attribution share would be wrong by a per-cell factor that
    varies with cell type, which is precisely the axis being measured.
    """
    import scanpy as sc
    import scipy.sparse as sp

    testset_path = cfg.TESTSET_H5AD if testset_path is None else Path(testset_path)
    if not testset_path.exists():
        raise FileNotFoundError(f"{testset_path} absent -- re-run the build without "
                                f"--skip-testset")
    test = sc.read_h5ad(testset_path)
    shared_cells = adata.obs_names.intersection(test.obs_names)
    if len(shared_cells) == 0:
        raise ValueError("the test set and the object share no cells -- stale build")
    genes = [g for g in adata.var_names if g in set(test.var_names)]

    a = adata[shared_cells, genes].X
    b = test[shared_cells, genes].X
    a = a.toarray() if sp.issparse(a) else np.asarray(a)
    b = b.toarray() if sp.issparse(b) else np.asarray(b)
    d = float(np.nanmax(np.abs(a - b)))
    if d >= tol:
        raise ValueError(
            f"EQUIVALENCE GATE FAILED: max abs diff {d:.3e} over {len(shared_cells):,} cells "
            f"x {len(genes)} genes. The library size was computed on the wrong gene set."
        )
    if verbose:
        print(f"EQUIVALENCE GATE PASSED on {len(shared_cells):,} cells x {len(genes)} genes "
              f"(max abs diff {d:.3e})")
    del test
    return d


def load_degradome_adata(path=None, manifest=None, verbose=True):
    """Read the built object, rebuild the level column, re-run every build-time assert."""
    import scanpy as sc

    path = cfg.DEG_ADATA if path is None else Path(path)
    manifest = cfg.MANIFEST if manifest is None else Path(manifest)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run jobs/run_degradome_build.sh first (one bsub pass "
            f"over the 48 GB source atlas; it is not a notebook step)."
        )
    adata = sc.read_h5ad(path)
    assert_panel(adata.var_names)
    assert_nb31_panel()

    if cfg.GROUPBY not in adata.obs.columns:
        raise ValueError(f"{path.name} carries no {cfg.GROUPBY} column -- stale build")
    adata.obs[cfg.GROUPBY] = pd.Categorical(
        adata.obs[cfg.GROUPBY].astype(str),
        categories=[lv for lv in cfg.CT_ORDER
                    if lv in set(adata.obs[cfg.GROUPBY].astype(str))],
    )
    if cfg.DISEASE_KEY not in adata.obs.columns:
        adata.obs[cfg.DISEASE_KEY] = disease_group(adata.obs)
    assert_degradome_roster(adata.obs[cfg.GROUPBY], verbose=verbose)

    if verbose:
        man = json.loads(manifest.read_text()) if manifest.exists() else {}
        print(f"{path.name}: {adata.n_obs:,} cells x {adata.n_vars} genes, "
              f"{adata.obs[cfg.DONOR_KEY].nunique()} donors, "
              f"{adata.obs[cfg.SAMPLE_KEY].nunique()} samples")
        if man:
            print(f"  build dropped {man.get('n_dropped_no_identity', 0):,} of "
                  f"{man.get('n_skin_cells', 0):,} skin cells with no identity")
        print(f"  {adata.uns['degradome_build']['warning']}")
    return adata


# ================================================================ coverage gate


def coverage_table(obs, min_cells=None, min_samples=None, out=None, verbose=True):
    """Per level: cells, donors, studies, the claim gate, and the study-dominance flag.

    Same shape and same gate as `tables/ccc40_coverage.csv`, extended with `claim` (is this a
    level nb42 makes statements about) and the HC arm, which nb40 never needed. The CTCL
    window is what the gate is evaluated in; HC counts ride along because section 4 needs them.
    """
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    min_samples = cfg.MIN_SAMPLES if min_samples is None else min_samples

    grp = disease_group(obs) if cfg.DISEASE_KEY not in obs.columns else obs[cfg.DISEASE_KEY]
    rows = []
    for lv, sub in obs.groupby(obs[cfg.GROUPBY].astype(str), observed=True):
        g = grp.loc[sub.index]
        ctcl, hc = sub[g == "CTCL"], sub[g == "HC"]
        per_donor = ctcl.groupby(cfg.DONOR_KEY, observed=True).size()
        hc_donor = hc.groupby(cfg.DONOR_KEY, observed=True).size()
        studies = ctcl[cfg.STUDY_KEY].astype(str).value_counts(normalize=True)
        rows.append({
            cfg.GROUPBY: lv,
            "claim": lv in cfg.CLAIM_LEVELS,
            "n_cells": len(sub),
            "n_cells_ctcl": len(ctcl),
            "n_cells_hc": len(hc),
            "n_donors_present": int(ctcl[cfg.DONOR_KEY].nunique()),
            "n_donors_ge25": int((per_donor >= min_cells).sum()),
            "n_hc_donors_ge25": int((hc_donor >= min_cells).sum()),
            "n_studies": int(ctcl[cfg.STUDY_KEY].nunique()),
            "top_study": studies.index[0] if len(studies) else None,
            "top_study_frac": round(float(studies.iloc[0]), 3) if len(studies) else np.nan,
        })
    cov = pd.DataFrame(rows)
    cov["claimable"] = cov["n_donors_ge25"] >= min_samples
    cov["study_dominated"] = cov["top_study_frac"] >= cfg.STUDY_DOMINANCE
    cov["disease_testable"] = cov["n_hc_donors_ge25"] >= cfg.MIN_DONORS_PER_ARM
    order = {lv: i for i, lv in enumerate(cfg.CT_ORDER)}
    cov = cov.sort_values(cfg.GROUPBY, key=lambda s: s.map(order)).reset_index(drop=True)

    if out is not None:
        cov.to_csv(out, index=False)
    if verbose:
        print(cov.to_string(index=False))
        bad = cov[cov["claim"] & ~cov["claimable"]]
        if len(bad):
            print(f"\nclaim levels failing the gate: {list(bad[cfg.GROUPBY])}")
    return cov


def assert_power_statement(cov, verbose=True):
    """Raise unless the measured HC coverage still matches `cfg.HC_DONORS_AT_GATE`.

    The notebook's power statement -- 6 of 13 levels testable against healthy skin -- is
    written into the config and into the prose. If nb10c or nb30 is re-run and the numbers
    move, this fires rather than the prose quietly becoming false.
    """
    got = (cov.set_index(cfg.GROUPBY)["n_hc_donors_ge25"]
           .reindex(cfg.TME_LEVELS).fillna(0).astype(int).to_dict())
    want = {k: int(v) for k, v in cfg.HC_DONORS_AT_GATE.items()}
    drift = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if drift:
        raise ValueError(
            "HC coverage has drifted from degradome_data.HC_DONORS_AT_GATE "
            f"(level: got vs configured) {drift}. Update the config AND the power prose in "
            "nb42 section 0 -- DISEASE_TESTABLE / DISEASE_UNDEFINED are derived from it."
        )
    testable = sorted(k for k, v in want.items() if v >= cfg.MIN_DONORS_PER_ARM)
    if testable != sorted(cfg.DISEASE_TESTABLE):
        raise ValueError(f"DISEASE_TESTABLE is stale: measured {testable}")
    if verbose:
        print(f"power statement holds: {len(testable)} of {len(cfg.TME_LEVELS)} nb10c levels "
              f"have >= {cfg.MIN_DONORS_PER_ARM} HC donors at {cfg.MIN_CELLS} cells "
              f"({', '.join(testable)});\n  the other "
              f"{len(cfg.TME_LEVELS) - len(testable)} have too few or none -- absence of a "
              f"comparator, reported in section 5, never as a fold change")
    return testable


# ================================================================ the (level x donor) cube


class DegradomeCube:
    """(level, donor, gene) raw counts + detection, built by one indicator matmul.

    Donor, not sample, is the replicate unit: several skin samples from one patient are not
    independent observations of that patient's stroma. Samples are summed within donor.

    Two shares are carried and they answer different questions.

      `share_raw`   -- of every transcript of gene g captured in this donor's skin, what
                       fraction came from level l. Abundance is part of the answer: a
                       fibroblast population that is ten times larger genuinely supplies ten
                       times the collagenase. This is the primary attribution.
      `share_cpm`   -- the same, after normalising each level by its own full-gene library, so
                       every level contributes per unit of transcriptome. Abundance removed;
                       this is expression intensity.

    A gene owned on `share_raw` but not on `share_cpm` is supplied by a level because there is
    a lot of it, not because each cell makes much. Both are reported.
    """

    def __init__(self, adata, levels=None, min_cells=None, verbose=True):
        import scipy.sparse as sp

        min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
        obs = adata.obs
        lv = obs[cfg.GROUPBY].astype(str).to_numpy()
        dn = obs[cfg.DONOR_KEY].astype(str).to_numpy()

        self.levels = [x for x in cfg.CT_ORDER if x in set(lv)] if levels is None else list(levels)
        self.donors = sorted(set(dn))
        self.genes = list(map(str, adata.var_names))
        self.li = {x: i for i, x in enumerate(self.levels)}
        self.di = {x: i for i, x in enumerate(self.donors)}
        self.gi = {x: i for i, x in enumerate(self.genes)}
        self.family = np.array([cfg.GENE_TO_FAMILY.get(g, "context") for g in self.genes])

        keep = np.isin(lv, self.levels)
        codes = (np.vectorize(self.li.get)(lv[keep]) * len(self.donors)
                 + np.vectorize(self.di.get)(dn[keep]))
        n_groups = len(self.levels) * len(self.donors)
        ind = sp.csr_matrix((np.ones(keep.sum()), (codes, np.arange(int(keep.sum())))),
                            shape=(n_groups, int(keep.sum())))

        X = adata.layers[cfg.COUNTS_LAYER][keep].tocsr()
        shape = (len(self.levels), len(self.donors), len(self.genes))
        self.counts = np.asarray((ind @ X).todense()).reshape(shape)
        self.detected = np.asarray((ind @ (X > 0).astype(np.int32)).todense()).reshape(shape)
        self.n_cells = np.asarray(ind.sum(axis=1)).ravel().reshape(shape[:2])
        # FULL-gene library from obs, not the panel sum: the panel is 156 of 40,821 columns and
        # its total is not a size factor.
        libv = np.zeros(n_groups)
        np.add.at(libv, codes, obs["total_counts"].to_numpy()[keep].astype(float))
        self.lib = libv.reshape(shape[:2])

        self.eligible = self.n_cells >= min_cells        # (level, donor)
        self.min_cells = min_cells
        if verbose:
            print(f"cube {shape} | {int(self.eligible.sum()):,} eligible (level, donor) pairs "
                  f"at >= {min_cells} cells")

    # -------------------------------------------------------- derived planes
    @property
    def det_frac(self):
        n = np.where(self.n_cells[:, :, None] > 0, self.n_cells[:, :, None], 1)
        return self.detected / n

    @property
    def cpm(self):
        lib = np.where(self.lib > 0, self.lib, 1)[:, :, None]
        return self.counts / lib * 1e6

    @property
    def cell_frac(self):
        tot = self.n_cells.sum(axis=0, keepdims=True)
        return self.n_cells / np.where(tot > 0, tot, 1)

    def _share(self, plane, min_gene_counts):
        """Normalise across levels. The denominator is EVERY level present in the donor.

        The eligibility mask deliberately does NOT enter here. Masking ineligible pairs before
        normalising would make the statistic "share among the levels that happened to clear
        MIN_CELLS in this donor", which is a different quantity in every donor and inflates
        every share by an unknown amount: Mast is eligible in 13 of 62 donors (median 6 cells),
        LC in 15 of 75, pDC in 13 of 66. Their transcripts are real and belong in the
        denominator whether or not the level can be credited with producing them.

        Eligibility is applied downstream, in `attribution`, where it decides which levels may
        be *reported* and which may *win* a donor -- a share estimated from six cells is noise
        and must not become an ownership call.
        """
        tot = plane.sum(axis=0)                                   # (donor, gene), all levels
        enough = self.counts.sum(axis=0) >= min_gene_counts
        tot = np.where((tot > 0) & enough, tot, np.nan)
        return plane / tot[None, :, :]

    def shares(self, min_gene_counts=10):
        return self._share(self.counts.astype(float), min_gene_counts)

    def shares_cpm(self, min_gene_counts=10):
        return self._share(self.cpm, min_gene_counts)


# ================================================================ panel coverage


def panel_coverage(cube, min_detection=None, out=None, verbose=True):
    """Detection fraction per (gene, level), pooled over eligible donors, + the drop list.

    A share computed from three counts is noise, not attribution, so genes never detected
    above `min_detection` in any level are dropped -- and the drop is written down. Silent
    truncation reads as "we looked at everything" when we did not.
    """
    min_detection = cfg.MIN_DETECTION if min_detection is None else min_detection
    det = np.nansum(np.where(cube.eligible[:, :, None], cube.detected, np.nan), axis=1)
    ncell = np.nansum(np.where(cube.eligible, cube.n_cells, np.nan), axis=1)
    frac = det / np.where(ncell[:, None] > 0, ncell[:, None], 1)

    cov = pd.DataFrame(frac.T, index=cube.genes, columns=cube.levels)
    cov.insert(0, "family", cube.family)
    cov["max_detection"] = frac.max(axis=0)
    cov["keep"] = cov["max_detection"] >= min_detection
    cov.index.name = "gene"

    dropped = sorted(cov.index[~cov["keep"]])
    if out is not None:
        cov.to_csv(out)
    if verbose:
        print(f"panel: {int(cov['keep'].sum())} of {len(cov)} genes detected in >= "
              f"{min_detection:.0%} of cells in at least one level")
        if dropped:
            print(f"  dropped ({len(dropped)}): {', '.join(dropped)}")
        nb31_dropped = [g for g in cfg.NB31_PANEL if g in set(dropped)]
        if nb31_dropped:
            print(f"  of which in the nb31 panel: {nb31_dropped}")
    return cov, dropped


# ================================================================ source attribution


def attribution(cube, genes=None, min_gene_counts=10, value="raw", out=None):
    """Long (gene, level) table: median donor share, IQR, wins, per-cell share, detection.

    Shares come off a COMPLETE denominator (every level present in the donor -- see
    `DegradomeCube._share`). Eligibility enters only here, and it does two separate jobs:

      * reported statistics are computed over the donors where the level clears MIN_CELLS,
        because a share estimated from six cells is not an estimate;
      * a donor's `win` is credited only if the top level is eligible in that donor. If the
        biggest producer is a level too sparse to be credited, the donor counts toward
        `n_donors` and toward nobody's `wins` -- it is an abstention, not a promotion of the
        runner-up. That is what stops mast tryptases from being handed to a fibroblast.

    `value` selects which share drives the win: "raw" (abundance-weighted output, the primary
    attribution) or "cpm" (per-transcriptome intensity, abundance divided out).
    """
    genes = cube.genes if genes is None else [g for g in genes if g in cube.gi]
    gidx = [cube.gi[g] for g in genes]

    raw = cube.shares(min_gene_counts)[:, :, gidx]        # (level, donor, gene), complete
    cpm = cube.shares_cpm(min_gene_counts)[:, :, gidx]
    det = cube.det_frac[:, :, gidx]
    elig = cube.eligible[:, :, None]

    drive = raw if value == "raw" else cpm
    with np.errstate(invalid="ignore"):
        valid = ~np.all(np.isnan(drive), axis=0)          # (donor, gene)
        top = np.nanargmax(np.where(np.isnan(drive), -np.inf, drive), axis=0)
        winner = np.full(drive.shape[1:], -1)
        winner[valid] = top[valid]
        # a win is only credited to a level that clears MIN_CELLS in that donor
        top_eligible = np.take_along_axis(
            np.broadcast_to(cube.eligible[:, :, None], drive.shape),
            np.clip(winner, 0, None)[None, :, :], axis=0)[0]
        winner = np.where(valid & top_eligible, winner, -1)

    # reported statistics: eligible donors only
    raw_r = np.where(elig, raw, np.nan)
    cpm_r = np.where(elig, cpm, np.nan)
    det_r = np.where(elig, det, np.nan)

    rows = []
    for li, lv in enumerate(cube.levels):
        for gj, g in enumerate(genes):
            r = raw_r[li, :, gj]
            n_don = int(np.sum(~np.isnan(r)))
            if n_don == 0:
                continue
            rows.append({
                "gene": g,
                "family": cfg.GENE_TO_FAMILY.get(g, "context"),
                cfg.GROUPBY: lv,
                "claim": lv in cfg.CLAIM_LEVELS,
                "n_donors": n_don,
                "n_donors_scored": int(np.sum(valid[:, gj])),
                "share_raw_med": float(np.nanmedian(r)),
                "share_raw_q25": float(np.nanpercentile(r, 25)),
                "share_raw_q75": float(np.nanpercentile(r, 75)),
                "share_cpm_med": float(np.nanmedian(cpm_r[li, :, gj])),
                "det_frac_med": float(np.nanmedian(det_r[li, :, gj])),
                "wins": int(np.sum(winner[:, gj] == li)),
            })
    attr = pd.DataFrame(rows)
    if out is not None:
        attr.to_csv(out, index=False)
    return attr


def call_owners(attr, owner_frac=None, min_donors=None, out=None, verbose=True):
    """One row per gene: the level that produces it, with the evidence that says so.

    A level owns a gene when it is the top producer in at least `owner_frac` of the donors
    where the gene is scoreable at all, over at least `min_donors` donors. Genes that clear
    neither are `owner = None` -- shared production is a real answer, not a missing one.

    The denominator is `n_donors_scored`, not the number of donors in which the winning level
    happened to be eligible. Donors whose top producer is too sparse to be credited are
    abstentions and they count against ownership, which is what keeps a gene made by a rare
    cell type from being awarded to whoever is second.
    """
    owner_frac = cfg.OWNER_DONOR_FRAC if owner_frac is None else owner_frac
    min_donors = cfg.MIN_DONORS_FOR_OWNERSHIP if min_donors is None else min_donors

    rows = []
    for g, sub in attr.groupby("gene", observed=True):
        sub = sub.sort_values("wins", ascending=False)
        n_don = int(sub["n_donors_scored"].max())
        n_abstain = n_don - int(sub["wins"].sum())
        top, second = sub.iloc[0], (sub.iloc[1] if len(sub) > 1 else None)
        frac = top["wins"] / n_don if n_don else np.nan
        owned = bool(n_don >= min_donors and frac >= owner_frac)
        rows.append({
            "gene": g,
            "family": top["family"],
            "owner": top[cfg.GROUPBY] if owned else None,
            "owner_is_claim_level": bool(top["claim"]) if owned else None,
            "owner_win_frac": round(float(frac), 3),
            "owner_share_med": round(float(top["share_raw_med"]), 4),
            "owner_share_cpm_med": round(float(top["share_cpm_med"]), 4),
            "owner_det_frac": round(float(top["det_frac_med"]), 3),
            "runner_up": None if second is None else second[cfg.GROUPBY],
            "runner_up_win_frac": None if second is None else round(
                float(second["wins"] / n_don) if n_don else np.nan, 3),
            "n_donors": n_don,
            "n_donors_abstained": int(n_abstain),
            "top_level_unowned": None if owned else top[cfg.GROUPBY],
        })
    own = pd.DataFrame(rows).sort_values(["family", "gene"]).reset_index(drop=True)
    if out is not None:
        own.to_csv(out, index=False)
    if verbose:
        called = own[own["owner"].notna()]
        print(f"{len(called)} of {len(own)} genes have a single-level owner at "
              f"{owner_frac:.0%} of donors")
        print(called["owner"].value_counts().to_string())
    return own


def shuffle_null(adata, cube, genes=None, n_shuffle=None, seed=None, verbose=True):
    """Null ceiling for ownership: permute the level label within donor, re-run the call.

    Within-donor permutation keeps every level's size and every donor's composition and
    destroys only the cell-to-level link, so the returned win fraction is what this many
    donors and this much sparsity produce by chance. If a real owner's win fraction is not
    above the ceiling, it is not an attribution.
    """
    import scipy.sparse as sp

    n_shuffle = cfg.N_SHUFFLE if n_shuffle is None else n_shuffle
    rng = np.random.default_rng(cfg.SEED if seed is None else seed)
    genes = cube.genes if genes is None else genes
    gidx = [cube.gi[g] for g in genes]

    obs = adata.obs
    lv = obs[cfg.GROUPBY].astype(str).to_numpy()
    dn = obs[cfg.DONOR_KEY].astype(str).to_numpy()
    keep = np.isin(lv, cube.levels)
    lv, dn = lv[keep], dn[keep]
    X = adata.layers[cfg.COUNTS_LAYER][keep].tocsr()
    lcode = np.vectorize(cube.li.get)(lv)
    dcode = np.vectorize(cube.di.get)(dn)
    n_groups = len(cube.levels) * len(cube.donors)
    shape = (len(cube.levels), len(cube.donors), len(cube.genes))
    blocks = [np.where(dcode == d)[0] for d in range(len(cube.donors))]

    ceilings = []
    for it in range(n_shuffle):
        perm = lcode.copy()
        for b in blocks:
            perm[b] = rng.permutation(perm[b])
        codes = perm * len(cube.donors) + dcode
        ind = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                            shape=(n_groups, len(codes)))
        counts = np.asarray((ind @ X).todense()).reshape(shape)
        n_cells = np.asarray(ind.sum(axis=1)).ravel().reshape(shape[:2])
        elig = n_cells >= cube.min_cells

        # same rule as `attribution`: complete denominator, wins credited only to eligible
        # levels, otherwise the ceiling is measured under a different statistic than the call.
        v = counts.astype(float)[:, :, gidx]
        tot = v.sum(axis=0)
        enough = counts[:, :, gidx].sum(axis=0) >= 10
        share = v / np.where((tot > 0) & enough, tot, np.nan)[None, :, :]
        with np.errstate(invalid="ignore"):
            valid = ~np.all(np.isnan(share), axis=0)
            top = np.nanargmax(np.where(np.isnan(share), -np.inf, share), axis=0)
            win = np.full(share.shape[1:], -1)
            win[valid] = top[valid]
            top_elig = np.take_along_axis(
                np.broadcast_to(elig[:, :, None], share.shape),
                np.clip(win, 0, None)[None, :, :], axis=0)[0]
            win = np.where(valid & top_elig, win, -1)
        n_don = valid.sum(axis=0)                  # donors where the share is computable
        best = np.zeros(len(genes))
        for gj in range(len(genes)):
            if n_don[gj] == 0:
                continue
            cnt = np.bincount(win[:, gj][win[:, gj] >= 0], minlength=len(cube.levels))
            best[gj] = cnt.max() / n_don[gj]
        ceilings.append(best)
        if verbose:
            print(f"  shuffle {it + 1}/{n_shuffle}: max win frac {np.nanmax(best):.3f}",
                  flush=True)

    ceil = np.vstack(ceilings)
    out = pd.DataFrame({
        "gene": genes,
        "null_win_frac_med": np.nanmedian(ceil, axis=0),
        "null_win_frac_max": np.nanmax(ceil, axis=0),
    })
    if verbose:
        print(f"null ceiling over {n_shuffle} shuffles: median "
              f"{np.nanmedian(ceil):.3f}, max {np.nanmax(ceil):.3f} "
              f"(real ownership bar is {cfg.OWNER_DONOR_FRAC:.2f})")
    return out


# ================================================================ spillover control


def spillover_flags(cube, own, genes=None, rho_thresh=0.5, p_thresh=0.05, out=None,
                    verbose=True):
    """Flag apparent expression that tracks the owner's abundance rather than the cell.

    Ambient mRNA and doublets put a cell type's transcripts into every other cell type in the
    same droplet suspension, in proportion to how abundant that cell type is. So for a level
    that does NOT own a gene, a per-cell intensity (abundance already divided out) that still
    rises with the owner's donor cell fraction is contamination, not expression.

    Genes are flagged, never dropped: an inflamed donor really does have both more macrophages
    and more macrophage-adjacent signalling, so a positive correlation is suggestive and not
    proof. The second guard is detection -- ambient raises the mean at low detection, so a
    high share with a low `det_frac` is the shape to distrust.
    """
    from scipy.stats import spearmanr

    genes = cube.genes if genes is None else genes
    owner_of = own.set_index("gene")["owner"].to_dict()
    cf = cube.cell_frac
    cpm_share = cube.shares_cpm()
    det = cube.det_frac

    rows = []
    for g in genes:
        owner = owner_of.get(g)
        if owner is None or owner not in cube.li:
            continue
        gj, oi = cube.gi[g], cube.li[owner]
        # abundance is well measured in every donor, so it is not masked; the level's own
        # intensity is, because a share from six cells is not an observation.
        owner_ab = cf[oi]
        for lv, li in cube.li.items():
            if lv == owner:
                continue
            y = np.where(cube.eligible[li], cpm_share[li, :, gj], np.nan)
            ok = ~np.isnan(y) & ~np.isnan(owner_ab)
            if ok.sum() < cfg.MIN_DONORS_FOR_OWNERSHIP:
                continue
            rho, p = spearmanr(owner_ab[ok], y[ok])
            rows.append({
                "gene": g, "family": cfg.GENE_TO_FAMILY.get(g, "context"),
                cfg.GROUPBY: lv, "owner": owner, "n_donors": int(ok.sum()),
                "rho_vs_owner_abundance": round(float(rho), 3),
                "p": float(p),
                "share_cpm_med": round(float(np.nanmedian(y)), 4),
                "det_frac_med": round(float(np.nanmedian(
                    np.where(cube.eligible[li], det[li, :, gj], np.nan))), 3),
            })
    sp_df = pd.DataFrame(rows)
    if len(sp_df):
        sp_df["spillover_suspect"] = ((sp_df["rho_vs_owner_abundance"] >= rho_thresh)
                                      & (sp_df["p"] < p_thresh))
    if out is not None:
        sp_df.to_csv(out, index=False)
    if verbose and len(sp_df):
        n = int(sp_df["spillover_suspect"].sum())
        print(f"{n} of {len(sp_df)} (gene, non-owner level) pairs track the owner's abundance "
              f"(rho >= {rho_thresh}, p < {p_thresh}) -- flagged, not dropped")
        if n:
            top = (sp_df[sp_df["spillover_suspect"]]
                   .sort_values("rho_vs_owner_abundance", ascending=False).head(10))
            print(top[["gene", cfg.GROUPBY, "owner", "rho_vs_owner_abundance",
                       "det_frac_med"]].to_string(index=False))
    return sp_df


def owner_spillover_note(own, cube, det_thresh=0.10):
    """Mark owner calls whose evidence is a high share at a low detection fraction."""
    own = own.copy()
    own["owner_low_detection"] = own["owner_det_frac"] < det_thresh
    return own


# ================================================================ protease : inhibitor balance


def balance_table(cube, pairs=None, out=None, verbose=True):
    """Per (level, donor): summed protease CPM over summed inhibitor CPM, log2.

    The readout `subclone_helpers.PAPER_PANELS["protease"]` promised ("read the protease:TIMP1
    ratio in the dot plot") and nb31 never rendered, computed on families rather than on one
    inhibitor. Both sides are CPM on the level's own full-gene library, so this is net
    degradative tone per unit of transcriptome and does not move with cell number.
    """
    pairs = cfg.BALANCE_PAIRS if pairs is None else pairs
    cpm = cube.cpm
    rows = []
    for name, (fam, inhib) in pairs.items():
        num_idx = [cube.gi[g] for g in cfg.DEGRADOME_FAMILIES[fam] if g in cube.gi]
        den_idx = [cube.gi[g] for g in inhib if g in cube.gi]
        num = cpm[:, :, num_idx].sum(axis=2)
        den = cpm[:, :, den_idx].sum(axis=2)
        ratio = np.log2((num + 1.0) / (den + 1.0))
        for li, lv in enumerate(cube.levels):
            for di, dn in enumerate(cube.donors):
                if not cube.eligible[li, di]:
                    continue
                rows.append({"balance": name, cfg.GROUPBY: lv, "claim": lv in cfg.CLAIM_LEVELS,
                             cfg.DONOR_KEY: dn, "protease_cpm": float(num[li, di]),
                             "inhibitor_cpm": float(den[li, di]),
                             "log2_ratio": float(ratio[li, di])})
    bal = pd.DataFrame(rows)
    if out is not None:
        bal.to_csv(out, index=False)
    if verbose and len(bal):
        med = (bal.groupby(["balance", cfg.GROUPBY], observed=True)["log2_ratio"]
               .median().unstack(0).round(2))
        order = [lv for lv in cfg.CT_ORDER if lv in med.index]
        print(med.reindex(order).to_string())
    return bal


# ================================================================ CTCL vs HC


def donor_pseudobulk(cube_pq=None, meta_pq=None, extra_keys=(), verbose=True):
    """Full-gene (level x sample) parquet -> (level x donor [x extra_keys]) counts + metadata.

    Donor, not sample, is the replicate unit -- several skin biopsies from one patient are not
    independent observations of that patient. Summing to donor before DESeq2 is the difference
    between 8 healthy replicates and a p-value inflated by pseudo-replication.

    `extra_keys` keeps a within-donor split intact. The epidermis/dermis contrast needs
    `extra_keys=("skin_layer",)`, because collapsing all the way to donor would average the two
    layers together and delete the very axis being tested.
    """
    cube_pq = cfg.PSEUDOBULK_PQ if cube_pq is None else Path(cube_pq)
    meta_pq = cfg.PSEUDOBULK_META if meta_pq is None else Path(meta_pq)
    counts = pd.read_parquet(cube_pq)
    meta = pd.read_parquet(meta_pq)

    keys = [cfg.GROUPBY, cfg.DONOR_KEY, *extra_keys]
    key = meta[keys[0]].astype(str)
    for k in keys[1:]:
        key = key + "||" + meta[k].astype(str)
    counts = counts.loc[meta.index]
    agg = counts.groupby(key.to_numpy(), observed=True).sum()

    design = [c for c in [cfg.GROUPBY, cfg.DONOR_KEY, cfg.STUDY_KEY, "disease",
                          cfg.DISEASE_KEY, "entity", "stage_class", "stage_group",
                          cfg.LAYER_KEY, "tissue", "sex", "tech"] if c in meta.columns]
    dmeta = meta.assign(_k=key.to_numpy()).drop_duplicates("_k").set_index("_k")[design]
    dmeta = dmeta.join(meta.assign(_k=key.to_numpy())
                       .groupby("_k", observed=True)["n_cells"].sum())
    dmeta = dmeta.reindex(agg.index)
    if verbose:
        lab = " x ".join(["level", "donor", *extra_keys])
        print(f"pseudobulk collapsed to {lab}: {agg.shape[0]:,} rows x "
              f"{agg.shape[1]:,} genes")
    return agg, dmeta


def pseudobulk_contrast(agg, dmeta, group_col, ref, alt, design, levels, *, panel=None,
                        min_cells=None, min_donors=4, restrict=None, paired_on=None,
                        arm="main", n_cpus=4, verbose=True):
    """Pseudobulk DESeq2 for one two-group contrast, run separately inside each level.

    Generic over the grouping axis so disease, skin layer and stage all go through the same
    code path, the same BH-within-level-over-the-panel rule and the same abstention logging.

    `restrict`  -- {column: allowed values}; applied before counting donors.
    `paired_on` -- keep only levels where this many units carry BOTH arms (the epidermis/dermis
                   case, where the pairing is the whole point of the design).
    """
    import subclone_helpers as S
    from statsmodels.stats.multitest import multipletests

    panel = cfg.panel_genes(include_context=False) if panel is None else panel
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells

    frames = []
    for lv in levels:
        m = dmeta[(dmeta[cfg.GROUPBY].astype(str) == lv)
                  & (dmeta[group_col].astype(str).isin([ref, alt]))
                  & (dmeta["n_cells"] >= min_cells)].copy()
        for col, allowed in (restrict or {}).items():
            m = m[m[col].astype(str).isin(list(allowed))]

        n_arm = m[group_col].astype(str).value_counts()
        note = None
        if n_arm.get(alt, 0) < min_donors or n_arm.get(ref, 0) < min_donors:
            note = f"{dict(n_arm)} units, need >= {min_donors} per arm"
        elif paired_on is not None:
            paired = (m.groupby(m[paired_on].astype(str))[group_col]
                      .agg(lambda s: set(s.astype(str)) >= {ref, alt}))
            n_paired = int(paired.sum())
            if n_paired < min_donors:
                note = f"only {n_paired} {paired_on}s carry both arms, need {min_donors}"
            else:
                # Recount the arms AFTER the pairing filter: the unpaired units are not in the
                # fit, so reporting them as n_alt / n_ref would put a sample size on the volcano
                # panel that no p-value in it was computed from.
                m = m[m[paired_on].astype(str).isin(paired.index[paired])]
                n_arm = m[group_col].astype(str).value_counts()
        if note:
            if verbose:
                print(f"  {lv}: skipped ({note})")
            continue

        md = pd.DataFrame(index=m.index)
        md[group_col] = pd.Categorical(m[group_col].astype(str), categories=[ref, alt])
        use = design
        for term in [t.strip() for t in design.replace("~", "").split("+")]:
            if term and term != group_col:
                md[term] = m[term].astype(str)
                if md[term].nunique() < 2:          # a blocking term with one level is singular
                    use = use.replace(f"+ {term}", "").replace(f"{term} +", "").strip()
                    md = md.drop(columns=term)
        res = S.run_pydeseq2(agg.loc[m.index], md, use, [group_col, alt, ref], n_cpus=n_cpus)

        keep = [g for g in panel if g in res.index]
        r = res.loc[keep].copy()
        r["gene"] = r.index
        r["family"] = [cfg.GENE_TO_FAMILY.get(g, "context") for g in r.index]
        r[cfg.GROUPBY] = lv
        r["contrast"] = f"{alt}_vs_{ref}"
        r["arm"] = arm
        r["n_alt"] = int(n_arm.get(alt, 0))
        r["n_ref"] = int(n_arm.get(ref, 0))
        r["design"] = use
        ok = r["pvalue"].notna()
        r["padj_panel"] = np.nan
        if ok.sum():
            r.loc[ok, "padj_panel"] = multipletests(r.loc[ok, "pvalue"], method="fdr_bh")[1]
        frames.append(r.reset_index(drop=True))
        if verbose:
            print(f"  {lv}: {n_arm.get(alt, 0)} {alt} / {n_arm.get(ref, 0)} {ref}, "
                  f"{len(keep)} genes, {int((r['padj_panel'] < 0.05).sum())} at FDR < 0.05 "
                  f"[{use}]")

    cols = ["gene", cfg.GROUPBY, "contrast", "arm"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def layer_pairing(dmeta_layer, min_cells=None, min_donors=None, out=None, verbose=True):
    """Per level: which donors carry BOTH epidermis and dermis at the gate, and how deeply.

    The gate on this contrast is PAIRING, not cell count. A level with ten epidermal donors and
    ten dermal donors and no donor in common is untestable under `~ donor + skin_layer`, and
    counting units rather than pairs is exactly the error that made the first version of this
    contrast return an empty frame for every level while looking adequately powered on paper
    (see the LAYER block of `degradome_data`). So the testable set is derived here, from the
    data, and `layer_contrast` takes it from this table rather than from a literal.

    `min_cells_in_pair` is the smallest pseudobulk any surviving arm rests on -- read it before
    reading the fold change; at three pairs it is the number most likely to disappoint.
    """
    min_cells = cfg.LAYER_MIN_CELLS if min_cells is None else min_cells
    min_donors = cfg.LAYER_MIN_PAIRED_DONORS if min_donors is None else min_donors
    col, alt, ref = cfg.LAYER_CONTRAST

    m = dmeta_layer[dmeta_layer[col].astype(str).isin([alt, ref])
                    & dmeta_layer["disease"].astype(str).isin(cfg.CTCL_DISEASES)]
    rows = []
    for lv in cfg.CT_ORDER:
        d = m[(m[cfg.GROUPBY].astype(str) == lv) & (m["n_cells"] >= min_cells)]
        if not len(d):
            continue
        paired = (d.groupby(d[cfg.DONOR_KEY].astype(str), observed=True)[col]
                  .agg(lambda s: set(s.astype(str)) >= {alt, ref}))
        donors = sorted(paired.index[paired])
        dp = d[d[cfg.DONOR_KEY].astype(str).isin(donors)]
        rows.append({
            cfg.GROUPBY: lv, "claim": lv in cfg.CLAIM_LEVELS,
            f"n_{alt}_units": int((d[col].astype(str) == alt).sum()),
            f"n_{ref}_units": int((d[col].astype(str) == ref).sum()),
            "n_paired_donors": len(donors),
            "min_cells_in_pair": int(dp["n_cells"].min()) if len(dp) else 0,
            "studies": ",".join(sorted(set(dp[cfg.STUDY_KEY].astype(str)))) if len(dp) else "",
            "paired_donors": ";".join(donors),
            "testable": len(donors) >= min_donors,
        })
    tbl = pd.DataFrame(rows)
    if out is not None:
        tbl.to_csv(out, index=False)
    if verbose and len(tbl):
        t = tbl[tbl["claim"]]
        print(f"epidermis/dermis pairing at >= {min_cells} cells "
              f"(need {min_donors} donors carrying BOTH layers):")
        print(t.drop(columns=["claim", "paired_donors"]).to_string(index=False))
        ab = t[(~t["testable"]) & (t["n_paired_donors"] > 0)]
        if len(ab):
            print(f"  abstained with 1-{min_donors - 1} paired donors (not a null): "
                  f"{dict(zip(ab[cfg.GROUPBY], ab['n_paired_donors']))}")
    return tbl


def layer_testable(tbl, claim_only=True):
    """The levels `layer_pairing` says can carry the contrast, in report order."""
    t = tbl[tbl["testable"] & (tbl["claim"] if claim_only else True)]
    return [lv for lv in cfg.CT_ORDER if lv in set(t[cfg.GROUPBY])]


def layer_contrast(agg_layer, dmeta_layer, levels=None, min_cells=None, out=None, verbose=True,
                   **kw):
    """Epidermis vs dermis, paired within donor. Needs no HC arm, so it reaches levels the
    disease contrast cannot -- DC_LAMP3 among them.

    `levels` defaults to whatever `layer_pairing` finds pairable rather than to a frozen list,
    and `min_cells` to `cfg.LAYER_MIN_CELLS` (10, not 25) because at three paired donors the
    binding constraint is the pairing, not the depth of one pseudobulk.
    """
    if levels is None:
        levels = layer_testable(layer_pairing(dmeta_layer, min_cells=min_cells, verbose=False))
    col, alt, ref = cfg.LAYER_CONTRAST
    de = pseudobulk_contrast(agg_layer, dmeta_layer, col, ref, alt, cfg.LAYER_DESIGN, levels,
                             min_cells=cfg.LAYER_MIN_CELLS if min_cells is None else min_cells,
                             min_donors=cfg.LAYER_MIN_PAIRED_DONORS,
                             restrict={"disease": cfg.CTCL_DISEASES},
                             paired_on=cfg.DONOR_KEY, arm="paired_within_donor",
                             verbose=verbose, **kw)
    if out is not None:
        de.to_csv(out, index=False)
    return de


def stage_contrast(agg, dmeta, levels=None, out=None, verbose=True, **kw):
    """Advanced vs early, inside li2024 only -- the one study that contains both stages."""
    levels = cfg.TME_LEVELS if levels is None else levels
    col, alt, ref = cfg.STAGE_CONTRAST
    de = pseudobulk_contrast(agg, dmeta, col, ref, alt, cfg.STAGE_DESIGN, levels,
                             min_donors=cfg.STAGE_MIN_DONORS_PER_ARM,
                             restrict={cfg.STUDY_KEY: [cfg.STAGE_STUDY]},
                             arm=f"within_{cfg.STAGE_STUDY}", verbose=verbose, **kw)
    if out is not None:
        de.to_csv(out, index=False)
    return de


def disease_contrast(agg, dmeta, levels=None, studies=None, panel=None, min_cells=None,
                     min_donors=None, arm="within_study", n_cpus=4, out=None, verbose=True):
    """CTCL vs HC, run separately inside each level. Thin wrapper over `pseudobulk_contrast`.

    `arm="within_study"` restricts to `cfg.PAIRED_STUDIES` -- the only two skin studies that
    contribute donors to both arms. Five of the seven are CTCL-only, so a pooled fit puts
    disease and study in the same column of the design matrix and reports batch as biology.
    `arm="pooled_sensitivity"` runs it anyway, on every study, and is labelled so it can never
    be read as the headline.

    BH is applied within level over the panel genes, not over all 40,821 -- the panel is the
    hypothesis, the rest of the transcriptome is there so DESeq2 can estimate size factors and
    dispersions properly.
    """
    levels = cfg.DISEASE_TESTABLE if levels is None else levels
    studies = cfg.PAIRED_STUDIES if studies is None else studies
    de = pseudobulk_contrast(
        agg, dmeta, cfg.DISEASE_KEY, "HC", "CTCL", cfg.DESEQ_DESIGN, levels,
        panel=panel, min_cells=min_cells,
        min_donors=cfg.MIN_DONORS_PER_ARM if min_donors is None else min_donors,
        restrict={cfg.STUDY_KEY: studies} if arm == "within_study" else None,
        arm=arm, n_cpus=n_cpus, verbose=verbose)
    if len(de):
        de["n_ctcl_donors"], de["n_hc_donors"] = de["n_alt"], de["n_ref"]
    if out is not None:
        de.to_csv(out, index=False)
    return de


def lesion_restricted_table(cov, cube, out=None, verbose=True):
    """The levels with no healthy comparator: what they are, and what they make in CTCL.

    Absence of a comparator is not a null result. These seven levels clear the claim gate in
    lesional skin and have zero HC donors at that gate, so their degradome output is reported
    as an abundance observation with no fold change attached. Writing them as "not
    significant" would be the single most misleading thing this notebook could do.
    """
    cpm = cube.cpm
    fam_idx = {f: [cube.gi[g] for g in gs if g in cube.gi]
               for f, gs in cfg.DEGRADOME_FAMILIES.items() if f in cfg.CLAIM_FAMILIES}

    rows = []
    for lv in cfg.DISEASE_UNDEFINED:
        if lv not in cube.li:
            continue
        li = cube.li[lv]
        elig = cube.eligible[li]
        c = cov[cov[cfg.GROUPBY] == lv]
        row = {cfg.GROUPBY: lv,
               "n_cells_ctcl": int(c["n_cells_ctcl"].iloc[0]) if len(c) else np.nan,
               "n_donors_ge25_ctcl": int(c["n_donors_ge25"].iloc[0]) if len(c) else np.nan,
               "n_cells_hc": int(c["n_cells_hc"].iloc[0]) if len(c) else np.nan,
               "n_hc_donors_ge25": int(c["n_hc_donors_ge25"].iloc[0]) if len(c) else np.nan,
               "contrast": "undefined -- no healthy comparator at the coverage gate"}
        for f, idx in fam_idx.items():
            row[f"{f}_cpm_med"] = float(np.nanmedian(
                np.where(elig, cpm[li, :, idx].sum(axis=0), np.nan)))
        rows.append(row)
    tbl = pd.DataFrame(rows)
    if out is not None:
        tbl.to_csv(out, index=False)
    if verbose and len(tbl):
        print("levels present in lesional skin with no healthy comparator "
              "(abundance observation, NOT a fold change):")
        print(tbl.to_string(index=False))
    return tbl


# ================================================================ controls


def lineage_level(obs):
    """The 13 nb10c subtypes collapsed back to Myeloid / Fibroblast; everything else as-is.

    Used only by the controls. At subtype granularity the TME is split 13 ways while
    Keratinocyte, Vascular and CD4_unassessed stay pooled, so an abundance-weighted share
    cannot express "COL1A1 comes from fibroblasts" -- each of the five fibroblast subtypes
    carries a fifth of it. Collapsing restores a roster on which the known answers are
    representable.
    """
    lv = obs[cfg.GROUPBY].astype(str)
    m = {x: "Myeloid" for x in cfg.MYELOID_LEVELS}
    m.update({x: "Fibroblast" for x in cfg.FIBRO_LEVELS})
    return pd.Series(lv.map(lambda x: m.get(x, x)), index=obs.index, name=cfg.GROUPBY)


def control_cube(adata, min_cells=None, verbose=True):
    """A lineage-granularity, low-eligibility cube built solely for the positive controls."""
    min_cells = cfg.CONTROL_MIN_CELLS if min_cells is None else min_cells
    ad = adata.copy()
    ad.obs[cfg.GROUPBY] = lineage_level(ad.obs)
    levels = sorted(set(ad.obs[cfg.GROUPBY].astype(str)))
    return DegradomeCube(ad, levels=levels, min_cells=min_cells, verbose=verbose)


def lineage_attribution(adata, genes, min_cells=None, value="raw", verbose=True):
    """Attribution with the 13 TME subtypes collapsed to Myeloid / Fibroblast.

    The subtype roster answers "which macrophage state", but it cannot answer "tumour or
    stroma": splitting the TME 13 ways while Keratinocyte and CD4_unassessed stay pooled means
    a diffusely produced gene has no single-subtype owner even when its lineage is obvious.
    Both granularities are reported because they answer different questions, and the coarse one
    is the one that speaks to nb31.
    """
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    ad = adata.copy()
    ad.obs[cfg.GROUPBY] = lineage_level(ad.obs)
    cube = DegradomeCube(ad, levels=sorted(set(ad.obs[cfg.GROUPBY].astype(str))),
                         min_cells=min_cells, verbose=verbose)
    attr = attribution(cube, genes=genes, value=value)
    return attr, call_owners(attr, verbose=verbose), cube


def assert_positive_controls(adata, cube=None, strict=True, verbose=True, out=None):
    """Raise unless the attribution recovers producers that are not in question.

    Mast tryptases must come from mast cells, granzymes from CD8, the kallikrein cascade and
    SPINK5 from keratinocytes, COL1A1 from fibroblasts, C1QA from myeloid cells. These are not
    results -- they are the only way to know the share arithmetic, the cell_id join and the
    library sizes are right before any actual claim is read.

    Run on `control_cube`: lineage granularity, MIN_CELLS lowered to CONTROL_MIN_CELLS. See
    the comment above `degradome_data.POSITIVE_CONTROLS_RAW` for why testing the arithmetic
    and making the claims need different rosters. Raw-share controls test "who supplies the
    tissue"; CPM-share controls test "who is built to make it", and the two are checked
    separately because they are different statistics.
    """
    cube = control_cube(adata, verbose=verbose) if cube is None else cube

    tiers = [
        ("raw", "ownership", cfg.POSITIVE_CONTROLS_RAW),
        ("cpm", "ownership", cfg.POSITIVE_CONTROLS_CPM),
        ("raw", "plurality", cfg.POSITIVE_CONTROLS_PLURALITY_RAW),
        ("cpm", "plurality", cfg.POSITIVE_CONTROLS_PLURALITY_CPM),
    ]
    rows, fails = [], []
    for value, tier, controls in tiers:
        genes = [g for g in controls if g in cube.gi]
        if not genes:
            continue
        own = call_owners(attribution(cube, genes=genes, value=value), verbose=False)
        idx = own.set_index("gene")
        for g, allowed in controls.items():
            tag = f"{value}/{tier}:{g}"
            if g not in idx.index:
                rows.append({"statistic": value, "tier": tier, "gene": g,
                             "expected": sorted(allowed), "top_level": None, "owner": None,
                             "win_frac": np.nan, "n_donors": np.nan, "pass": False,
                             "note": "not in the panel / not scoreable"})
                fails.append(tag)
                continue
            r = idx.loc[g]
            # `top_level_unowned` carries the argmax when the ownership bar was not cleared
            top = r["owner"] if pd.notna(r["owner"]) else r["top_level_unowned"]
            ok = (r["owner"] in allowed) if tier == "ownership" else (top in allowed)
            rows.append({"statistic": value, "tier": tier, "gene": g,
                         "expected": sorted(allowed), "top_level": top, "owner": r["owner"],
                         "win_frac": r["owner_win_frac"], "n_donors": r["n_donors"],
                         "pass": bool(ok),
                         "note": "" if ok else f"top was {top or 'nothing'}"})
            if not ok:
                fails.append(tag)

    tbl = pd.DataFrame(rows)
    if out is not None:
        tbl.to_csv(out, index=False)
    if verbose:
        print(tbl.to_string(index=False))
        print(f"\n{int(tbl['pass'].sum())}/{len(tbl)} positive controls recovered "
              f"(lineage granularity, MIN_CELLS={cube.min_cells})")
    if fails and strict:
        raise ValueError(
            f"positive controls failed for {fails} -- the attribution does not recover known "
            f"biology, so nothing downstream should be read. Check the cell_id join, the "
            f"library sizes and the share denominator before anything else."
        )
    return tbl


def assert_controls_pass(tbl, verbose=True):
    """Re-raise on a failed control table. Kept separate from `assert_positive_controls` so a
    cached table is still gated -- a cache must not be a way to skip the check that says the
    numbers mean anything."""
    fails = tbl[~tbl["pass"].astype(bool)]
    if verbose:
        print(tbl.to_string(index=False))
        print(f"\n{int(tbl['pass'].sum())}/{len(tbl)} positive controls recovered")
    if len(fails):
        raise ValueError(
            "positive controls failed for "
            f"{list(fails['statistic'] + '/' + fails['tier'] + ':' + fails['gene'])} -- the "
            f"attribution does not recover known biology, so nothing downstream should be read."
        )
    return tbl


def against_null(own, null, out=None, verbose=True):
    """Join the ownership call to the shuffle ceiling and mark calls the null already explains."""
    m = own.merge(null, on="gene", how="left")
    m["above_null"] = m["owner_win_frac"] > m["null_win_frac_max"]
    m.loc[m["owner"].isna(), "above_null"] = np.nan
    if out is not None:
        m.to_csv(out, index=False)
    if verbose:
        called = m[m["owner"].notna()]
        n_bad = int((called["above_null"] == False).sum())  # noqa: E712
        print(f"{len(called) - n_bad} of {len(called)} owner calls exceed the within-donor "
              f"shuffle ceiling; {n_bad} do not and are not attributions")
        if n_bad:
            print(called[called["above_null"] == False]  # noqa: E712
                  [["gene", "owner", "owner_win_frac", "null_win_frac_max"]]
                  .to_string(index=False))
    return m


# ================================================================ the clonal axis


def load_subclone_labels(adata, path=None, min_cells=None, verbose=True):
    """nb31's `nested_subclone` joined onto this object's CD4_malignant cells.

    Returns (labels, meta): `labels` is a per-cell Series ("" outside the malignant compartment
    or below the size gate) and `meta` is one row per eligible subclone.

    Two things are checked rather than assumed. The label is applied ONLY where this object
    calls the cell CD4_malignant -- nb31's compartment is `tcr_malignant_alice` and this one is
    `mal_tcr_alice` gated through `build_degradome_level`, and if they ever diverge the join
    must not silently relabel a reactive cell as tumour. And the cells nb31 could not cover are
    counted and printed, because a subclone analysis that quietly drops cells reads as complete.
    """
    path = cfg.SUBCLONE_PARQUET if path is None else Path(path)
    min_cells = cfg.CLONE_MIN_CELLS if min_cells is None else min_cells
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run nb31 section A first")

    sub = pd.read_parquet(path)
    if cfg.SUBCLONE_COL not in sub.columns:
        if "cell_id" not in sub.columns:
            raise ValueError(f"{path.name} carries no {cfg.SUBCLONE_COL} column")
    if "cell_id" in sub.columns:
        sub = sub.set_index("cell_id")

    mal = (adata.obs[cfg.GROUPBY].astype(str) == cfg.CD4_MALIGNANT).to_numpy()
    lab = (sub[cfg.SUBCLONE_COL].reindex(adata.obs_names).astype(str)
           .replace({"nan": "", "None": ""}).fillna(""))
    outside = int(((lab != "") & ~mal).sum())
    lab[~mal] = ""
    n_unlabelled = int((mal & (lab == "").to_numpy()).sum())

    vc = lab[lab != ""].value_counts()
    eligible = set(vc.index[vc >= min_cells])
    lab[~lab.isin(eligible)] = ""

    donor = adata.obs[cfg.DONOR_KEY].astype(str)
    meta = (pd.DataFrame({"subclone": lab[lab != ""], cfg.DONOR_KEY: donor[lab != ""]})
            .groupby("subclone", observed=True)
            .agg(**{cfg.DONOR_KEY: (cfg.DONOR_KEY, "first"), "n_cells": ("subclone", "size")})
            .reset_index())
    n_per_donor = meta.groupby(cfg.DONOR_KEY, observed=True).size()
    meta["donor_n_subclones"] = meta[cfg.DONOR_KEY].map(n_per_donor)
    meta["testable"] = meta["donor_n_subclones"] >= cfg.CLONE_MIN_SUBCLONES

    if verbose:
        print(f"{path.name}: {int((lab != '').sum()):,} of {int(mal.sum()):,} "
              f"{cfg.CD4_MALIGNANT} cells carry a subclone label at >= {min_cells} cells")
        print(f"  {len(meta)} subclones over {meta[cfg.DONOR_KEY].nunique()} donors "
              f"| {int(meta['testable'].sum())} in donors with >= {cfg.CLONE_MIN_SUBCLONES} "
              f"subclones (the testable set)")
        print(f"  {n_unlabelled:,} malignant cells nb31 could not cover (no arm-CNV row); "
              f"{int(vc.sum() - (lab != '').sum()):,} in subclones below the size gate")
        if outside:
            print(f"  {outside:,} labelled cells are NOT {cfg.CD4_MALIGNANT} here and were "
                  f"dropped -- nb31's compartment and this object's level do not fully agree")
    return lab, meta


def clone_cube(adata, labels, genes=None, verbose=True):
    """(subclone x gene) counts, CPM on the subclone's own FULL-gene library, and detection.

    Same arithmetic as `DegradomeCube`, one axis instead of two: the library comes from
    `obs.total_counts` (all 40,821 genes), never from the 156 panel columns, so a subclone's
    CPM is comparable to a TME level's CPM in section 3b.
    """
    import scipy.sparse as sp

    genes = list(adata.var_names) if genes is None else [g for g in genes
                                                         if g in set(adata.var_names)]
    lab = np.asarray(labels.astype(str))
    subs = sorted(set(lab) - {""})
    si = {s: i for i, s in enumerate(subs)}
    keep = np.array([x != "" for x in lab])
    codes = np.array([si[x] for x in lab[keep]])

    gi = {g: i for i, g in enumerate(map(str, adata.var_names))}
    cols = [gi[g] for g in genes]
    X = adata.layers[cfg.COUNTS_LAYER][keep][:, cols].tocsr()
    ind = sp.csr_matrix((np.ones(int(keep.sum())), (codes, np.arange(int(keep.sum())))),
                        shape=(len(subs), int(keep.sum())))

    counts = pd.DataFrame(np.asarray((ind @ X).todense()), index=subs, columns=genes)
    det = pd.DataFrame(np.asarray((ind @ (X > 0).astype(np.int32)).todense()),
                       index=subs, columns=genes)
    n_cells = np.asarray(ind.sum(axis=1)).ravel()
    lib = np.zeros(len(subs))
    np.add.at(lib, codes, adata.obs["total_counts"].to_numpy()[keep].astype(float))
    cpm = counts.div(np.where(lib > 0, lib, 1), axis=0) * 1e6
    det_frac = det.div(np.where(n_cells > 0, n_cells, 1), axis=0)

    meta = pd.DataFrame({"subclone": subs, "n_cells": n_cells.astype(int), "lib": lib})
    meta[cfg.DONOR_KEY] = [s.rsplit("_", 1)[0] for s in subs]
    if verbose:
        print(f"clone cube: {len(subs)} subclones x {len(genes)} genes "
              f"| {int(n_cells.sum()):,} cells")
    return cpm, det_frac, meta


def clone_balance_table(cpm, meta, pairs=None, out=None, verbose=True):
    """Protease : inhibitor log2 ratio per subclone -- section 3b's readout on the clonal axis.

    Identical arithmetic to `balance_table`, so a subclone's number and a TME level's number are
    on the same scale and can be read against each other.
    """
    pairs = cfg.BALANCE_PAIRS if pairs is None else pairs
    rows = []
    for name, (fam, inhib) in pairs.items():
        num = cpm[[g for g in cfg.DEGRADOME_FAMILIES[fam] if g in cpm.columns]].sum(axis=1)
        den = cpm[[g for g in inhib if g in cpm.columns]].sum(axis=1)
        r = np.log2((num + 1.0) / (den + 1.0))
        rows.append(pd.DataFrame({"balance": name, "subclone": cpm.index,
                                  "protease_cpm": num.to_numpy(),
                                  "inhibitor_cpm": den.to_numpy(),
                                  "log2_ratio": r.to_numpy()}))
    bal = pd.concat(rows, ignore_index=True).merge(
        meta[["subclone", cfg.DONOR_KEY, "n_cells"]], on="subclone", how="left")
    if out is not None:
        bal.to_csv(out, index=False)
    if verbose and len(bal):
        med = bal.groupby("balance", observed=True)["log2_ratio"].median().round(2)
        print("median log2 protease/inhibitor across subclones:")
        print(med.to_string())
    return bal


def clone_panel_de(adata, labels, meta, genes=None, out=None, verbose=True):
    """Within-donor Wilcoxon: each nested subclone against the rest of that donor's tumour.

    `subclone_helpers.subclone_markers_wilcoxon` is reused verbatim -- it is what nb31 and
    `old/23_subclonal_evolution.ipynb` cell 17 ran, so these numbers sit next to theirs. The
    object handed to it carries the panel genes only, which makes scanpy's BH adjustment a
    panel-wide FDR and matches the "BH within level across the panel" rule used by every
    contrast in section 4.

    No pseudobulk, no cross-donor pooling: there are no biological replicates inside one
    patient, and subclones of different patients are not the same entity.
    """
    import subclone_helpers as S

    genes = [g for g in (cfg.panel_genes(False) if genes is None else genes)
             if g in set(adata.var_names)]
    testable = meta[meta["testable"]]
    donors = sorted(testable[cfg.DONOR_KEY].unique())
    lab = labels.astype(str)

    frames = []
    for d in donors:
        cells = ((adata.obs[cfg.DONOR_KEY].astype(str) == d).to_numpy()
                 & lab.isin(set(testable.loc[testable[cfg.DONOR_KEY] == d, "subclone"])).to_numpy())
        if cells.sum() < 2 * cfg.CLONE_MIN_CELLS:
            continue
        ad = adata[cells, genes].copy()
        ad.obs[cfg.SUBCLONE_COL] = pd.Categorical(lab[cells].to_numpy())
        if ad.obs[cfg.SUBCLONE_COL].nunique() < cfg.CLONE_MIN_SUBCLONES:
            continue
        de = S.subclone_markers_wilcoxon(ad, cfg.SUBCLONE_COL, n_genes=ad.n_vars)
        de[cfg.DONOR_KEY] = d
        frames.append(de)
        if verbose:
            n_up = int(((de["pval_adj"] < 0.05) & (de["log2fc"] >= cfg.CLONE_LFC)).sum())
            print(f"  {d}: {ad.obs[cfg.SUBCLONE_COL].nunique()} subclones, {ad.n_obs:,} cells, "
                  f"{n_up} subclone-up calls at FDR < 0.05, log2FC >= {cfg.CLONE_LFC}")

    cols = ["subclone", "gene", "log2fc", "pval_adj", "score", cfg.DONOR_KEY]
    de = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    if len(de):
        de["family"] = de["gene"].map(lambda g: cfg.GENE_TO_FAMILY.get(g, "context"))
        de["up"] = (de["pval_adj"] < 0.05) & (de["log2fc"] >= cfg.CLONE_LFC)
        de["down"] = (de["pval_adj"] < 0.05) & (de["log2fc"] <= -cfg.CLONE_LFC)
    if out is not None:
        de.to_csv(out, index=False)
    if verbose and len(de):
        hit = (de[de["up"]].groupby(["gene", "family"], observed=True)
               .agg(n_subclones=("subclone", "nunique"), n_donors=(cfg.DONOR_KEY, "nunique"))
               .reset_index().sort_values(["n_donors", "n_subclones"], ascending=False))
        print(f"\n{int(de['up'].sum())} (subclone, gene) up-calls over {len(de)} tests; "
              f"genes called up in the most donors:")
        print(hit.head(12).to_string(index=False))
    return de


def level_cpm_median(cube, genes=None):
    """(level x gene) median CPM over the donors where the level is eligible."""
    genes = cube.genes if genes is None else [g for g in genes if g in cube.gi]
    gidx = [cube.gi[g] for g in genes]
    c = np.where(cube.eligible[:, :, None], cube.cpm[:, :, gidx], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN level x gene slices
        med = np.nanmedian(c, axis=1)
    return pd.DataFrame(med, index=cube.levels, columns=genes)


def clone_vs_tme(cpm, cube, own, attr=None, genes=None, out=None, verbose=True):
    """Per gene: the median subclone's CPM against the owning TME level's, in the same units.

    Section 2 says who supplies the tissue in absolute transcripts, which is abundance-weighted
    on purpose. This says whether the tumour clone is even BUILT to make the gene -- intensity,
    abundance divided out, both sides CPM on their own full-gene library. The two answers come
    apart in a way that matters: a clone that matches its owner per transcriptome while
    supplying ~1% of the tissue's transcripts is a population too small to matter, and a clone
    an order of magnitude below its owner is not making the gene at all.
    """
    genes = [g for g in (cfg.NB31_PANEL if genes is None else genes) if g in cpm.columns]
    lvl = level_cpm_median(cube, genes)
    o = own.set_index("gene")
    a = None if attr is None else attr.set_index(["gene", cfg.GROUPBY])

    rows = []
    for g in genes:
        owner = o["owner"].get(g) if g in o.index else None
        owner = None if (owner is None or (isinstance(owner, float) and np.isnan(owner))) else owner
        rec = {"gene": g, "family": cfg.GENE_TO_FAMILY.get(g, "context"),
               "clone_cpm_med": float(cpm[g].median()),
               "clone_cpm_max": float(cpm[g].max()),
               "owner": owner,
               "owner_cpm_med": float(lvl.loc[owner, g]) if owner in lvl.index else np.nan,
               "malignant_level_cpm_med": (float(lvl.loc[cfg.CD4_MALIGNANT, g])
                                           if cfg.CD4_MALIGNANT in lvl.index else np.nan)}
        if a is not None and (g, cfg.CD4_MALIGNANT) in a.index:
            rec["malignant_share_raw_med"] = float(a.loc[(g, cfg.CD4_MALIGNANT),
                                                         "share_raw_med"])
        rows.append(rec)
    tbl = pd.DataFrame(rows)
    tbl["clone_over_owner_log2"] = np.log2((tbl["clone_cpm_med"] + 1.0)
                                           / (tbl["owner_cpm_med"] + 1.0))
    if out is not None:
        tbl.to_csv(out, index=False)
    if verbose and len(tbl):
        print(tbl.round(3).to_string(index=False))
    return tbl


# ================================================================ figures


def _level_colors(levels):
    """Categorical palette: myeloid warm, fibroblast cool, malignant CD4 magenta, rest grey.

    Sequential ramps within a lineage were unreadable -- eight shades of orange do not
    separate at bar-segment size. These are discrete hues chosen for adjacent-segment
    contrast, with lineage still legible by family.
    """
    mye = ["#e6550d", "#fd8d3c", "#fdae6b", "#a63603", "#fdd0a2", "#d94801", "#f16913",
           "#8c2d04"]
    fib = ["#3182bd", "#6baed6", "#9ecae1", "#08519c", "#c6dbef", "#4292c6", "#2171b5"]
    oth = ["#31a354", "#756bb1", "#969696", "#e7ba52", "#17becf", "#8c6d31", "#bcbd22",
           "#7b4173", "#525252", "#ce6dbd"]
    col, i, j, k = {}, 0, 0, 0
    for lv in levels:
        if lv == cfg.CD4_MALIGNANT:
            col[lv] = "#d62728"
        elif lv in cfg.MYELOID_LEVELS:
            col[lv] = mye[i % len(mye)]; i += 1
        elif lv in cfg.FIBRO_LEVELS:
            col[lv] = fib[j % len(fib)]; j += 1
        else:
            col[lv] = oth[k % len(oth)]; k += 1
    return col


def _finish(fig, save, show):
    """Save, show at most once, then close.

    The inline backend re-renders any figure still open when the cell ends, so a bare
    `plt.show()` produces the same figure twice. Closing after showing is what stops it.
    """
    import matplotlib.pyplot as plt

    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_nb31_panel_owner(attr, panel=None, save=None, show=True, top_n=8):
    """Stacked donor-median share per gene, for the genes nb31 scored."""
    import matplotlib.pyplot as plt

    panel = cfg.NB31_PANEL if panel is None else panel
    sub = attr[attr["gene"].isin(panel)]
    piv = (sub.pivot_table(index="gene", columns=cfg.GROUPBY, values="share_raw_med",
                           observed=True)
           .reindex([g for g in panel if g in set(sub["gene"])]).fillna(0.0))
    keep = list(piv.sum(0).sort_values(ascending=False).head(top_n).index)
    if cfg.CD4_MALIGNANT in piv.columns and cfg.CD4_MALIGNANT not in keep:
        keep.append(cfg.CD4_MALIGNANT)
    plot = piv[keep].copy()
    plot["other"] = piv.drop(columns=keep).sum(axis=1)
    plot = plot.div(plot.sum(axis=1), axis=0)

    col = _level_colors(keep)
    col["other"] = "#d9d9d9"
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(plot) + 1.3))
    left = np.zeros(len(plot))
    for c in plot.columns:
        ax.barh(plot.index, plot[c], left=left, color=col.get(c, "#d9d9d9"), label=c,
                edgecolor="white", lw=0.6)
        left += plot[c].to_numpy()
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("share of skin output (donor median)", fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.5, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.set_title("nb31 protease panel: source by level", fontsize=9)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_balance(bal, save=None, show=True):
    """Protease : inhibitor log2 ratio per level, one panel per family pair."""
    import matplotlib.pyplot as plt

    names = list(bal["balance"].unique())
    order = [lv for lv in cfg.CT_ORDER if lv in set(bal[cfg.GROUPBY])]
    col = _level_colors(order)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 0.22 * len(order) + 2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        sub = bal[bal["balance"] == name]
        data = [sub.loc[sub[cfg.GROUPBY] == lv, "log2_ratio"].dropna().to_numpy()
                for lv in order]
        bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True)
        for patch, lv in zip(bp["boxes"], order):
            patch.set_facecolor(col[lv] if lv in cfg.CLAIM_LEVELS else "#e0e0e0")
            patch.set_edgecolor("0.3")
        for med in bp["medians"]:
            med.set_color("k")
        ax.axvline(0, color="0.4", lw=0.7, ls="--")
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels(order, fontsize=6)
        ax.set_xlabel("log2 protease / inhibitor CPM", fontsize=7)
        ax.set_title(name.replace("_", " : "), fontsize=8)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_volcano(de, levels=None, arm=None, alpha=0.05, lfc=1.0, title=None, save=None,
                 show=True, label_n=6):
    """One panel per level, panel genes only. Works for any contrast in `de`."""
    import matplotlib.pyplot as plt

    d = de if arm is None else de[de["arm"] == arm]
    if not len(d):
        print("nothing to plot")
        return None
    order = {lv: i for i, lv in enumerate(cfg.CT_ORDER)}
    levels = (sorted(set(d[cfg.GROUPBY]), key=lambda x: order.get(x, 99)) if levels is None
              else [lv for lv in levels if lv in set(d[cfg.GROUPBY])])
    if not levels:
        print("nothing to plot")
        return None
    lab = str(d["contrast"].iloc[0])
    ncol = min(3, len(levels))
    nrow = int(np.ceil(len(levels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for ax, lv in zip(axes.ravel(), levels):
        s = d[d[cfg.GROUPBY] == lv].dropna(subset=["padj_panel", "log2FoldChange"])
        up = (s["padj_panel"] < alpha) & (s["log2FoldChange"] >= lfc)
        dn = (s["padj_panel"] < alpha) & (s["log2FoldChange"] <= -lfc)
        ns = ~(up | dn)
        ax.scatter(s.loc[ns, "log2FoldChange"], -np.log10(s.loc[ns, "padj_panel"]),
                   s=8, c="#cccccc", lw=0)
        ax.scatter(s.loc[up, "log2FoldChange"], -np.log10(s.loc[up, "padj_panel"]),
                   s=14, c="#d62728", lw=0)
        ax.scatter(s.loc[dn, "log2FoldChange"], -np.log10(s.loc[dn, "padj_panel"]),
                   s=14, c="#3182bd", lw=0)
        sig = s[up | dn]
        for _, r in sig.reindex(sig["padj_panel"].nsmallest(label_n).index).iterrows():
            ax.annotate(r["gene"], (r["log2FoldChange"], -np.log10(r["padj_panel"])),
                        fontsize=5.5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(-np.log10(alpha), color="0.6", lw=0.6, ls="--")
        ax.axvline(0, color="0.6", lw=0.6)
        n = s[["n_alt", "n_ref"]].iloc[0] if len(s) else [0, 0]
        ax.set_title(f"{lv}  ({int(n[0])} vs {int(n[1])})", fontsize=8)
        ax.set_xlabel(f"log2FC {lab}", fontsize=7)
        ax.set_ylabel("-log10 FDR", fontsize=7)
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[len(levels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_disease_volcano(de, arm="within_study", **kw):
    """Back-compat alias for the CTCL-vs-HC panel."""
    return plot_volcano(de, levels=cfg.DISEASE_TESTABLE, arm=arm, **kw)


# ---------------------------------------------------------------- the clonal axis


def plot_clone_panel_dotplot(adata, labels, panels=None, donors=None, save=None, title=None,
                             standard_scale="var"):
    """The dot plot nb31 computed and never consumed, over the nb31 panel.

    `subclone_helpers.subclone_dotplot` is called directly -- same renderer, same
    `standard_scale='var'`, same bracket layout as nb31 cell 11 and old/23 cell 16, so this
    figure can be laid next to theirs. Rows are nested subclones, grouped by donor.
    """
    import subclone_helpers as S

    panels = {"nb31 protease panel": list(cfg.NB31_PANEL)} if panels is None else panels
    lab = labels.astype(str)
    keep = (lab != "").to_numpy()
    if donors is not None:
        keep &= adata.obs[cfg.DONOR_KEY].astype(str).isin(list(donors)).to_numpy()
    ad = adata[keep].copy()
    subs = sorted(set(lab[keep]))
    ad.obs[cfg.SUBCLONE_COL] = pd.Categorical(lab[keep].to_numpy(), categories=subs)
    n = len(subs)
    return S.subclone_dotplot(
        ad, cfg.SUBCLONE_COL, panels=panels, categories_order=subs, save=save,
        standard_scale=standard_scale,
        figsize=(0.34 * sum(len(v) for v in panels.values()) + 2.5, 0.17 * n + 2),
        title=title or (f"degradome panel across {n} nested subclones "
                        f"({adata.obs.loc[keep, cfg.DONOR_KEY].nunique()} donors)"))


def plot_clone_de(de, top_n=25, save=None, show=True):
    """Gene x subclone effect map for the within-donor Wilcoxon, significant genes only.

    Colour is log2FC of the subclone against the rest of its donor's tumour; a dot marks the
    calls that clear FDR < 0.05 and |log2FC| >= cfg.CLONE_LFC. Genes with no call anywhere are
    dropped -- a wall of grey rows is not evidence of anything.
    """
    import matplotlib.pyplot as plt

    if not len(de) or not de[["up", "down"]].to_numpy().any():
        print("no subclone call cleared FDR < 0.05 -- nothing to plot")
        return None
    called = de[de["up"] | de["down"]]
    genes = (called.groupby("gene", observed=True)[cfg.DONOR_KEY].nunique()
             .sort_values(ascending=False).head(top_n).index.tolist())
    genes = [g for g in cfg.panel_genes(False) if g in set(genes)]
    subs = sorted(set(de.loc[de["gene"].isin(genes), "subclone"]))

    piv = (de[de["gene"].isin(genes)]
           .pivot_table(index="gene", columns="subclone", values="log2fc", observed=True)
           .reindex(index=genes, columns=subs))
    sig = (de[de["gene"].isin(genes)]
           .assign(s=lambda d: (d["up"] | d["down"]).astype(float))
           .pivot_table(index="gene", columns="subclone", values="s", observed=True)
           .reindex(index=genes, columns=subs).fillna(0.0))

    v = float(np.nanpercentile(np.abs(piv.to_numpy()), 98)) or 1.0
    fig, ax = plt.subplots(figsize=(max(5.0, 0.14 * len(subs) + 2.5), 0.22 * len(genes) + 1.6))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    yy, xx = np.nonzero(sig.to_numpy() > 0)
    ax.scatter(xx, yy, s=5, c="k", lw=0)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([s.rsplit("_", 1)[-1] for s in subs], fontsize=4, rotation=90)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=6)
    ax.set_xlabel(f"nested subclone ({de[cfg.DONOR_KEY].nunique()} donors, "
                  f"each tested against the rest of its own tumour)", fontsize=7)
    ax.set_title("Degradome genes across malignant subclones\n"
                 "colour = log2FC vs rest of donor's tumour · dot = FDR < 0.05", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_clone_balance(bal_clone, bal_level=None, save=None, show=True):
    """Protease : inhibitor balance per donor's subclones, against the TME levels.

    One box per donor over that donor's nested subclones, so within-patient clonal spread is
    the visible quantity; the TME claim levels from section 3b are drawn as reference lines in
    the same units, which is the entire point of computing both on each population's own
    full-gene library.
    """
    import matplotlib.pyplot as plt

    names = list(bal_clone["balance"].unique())
    donors = sorted(set(bal_clone[cfg.DONOR_KEY]))
    fig, axes = plt.subplots(1, len(names),
                             figsize=(4.2 * len(names), 0.22 * len(donors) + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        sub = bal_clone[bal_clone["balance"] == name]
        data = [sub.loc[sub[cfg.DONOR_KEY] == d, "log2_ratio"].dropna().to_numpy()
                for d in donors]
        bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#d62728")
            patch.set_alpha(0.55)
            patch.set_edgecolor("0.3")
        for med in bp["medians"]:
            med.set_color("k")
        for i, d in enumerate(donors, start=1):
            y = sub.loc[sub[cfg.DONOR_KEY] == d, "log2_ratio"].dropna().to_numpy()
            ax.scatter(y, np.full(len(y), i), s=6, c="0.15", zorder=3)
        if bal_level is not None and len(bal_level):
            ref = (bal_level[(bal_level["balance"] == name) & bal_level["claim"]]
                   .groupby(cfg.GROUPBY, observed=True)["log2_ratio"].median())
            for lv, val in ref.items():
                if lv == cfg.CD4_MALIGNANT:
                    ax.axvline(val, color="#d62728", lw=1.2, ls="-")
                    ax.text(val, len(donors) + 0.8, f" {lv} (level)", fontsize=5,
                            color="#d62728", rotation=90, va="bottom")
                else:
                    ax.axvline(val, color="0.75", lw=0.6, ls=":")
            ax.text(0.02, 0.02, "dotted = TME claim levels (section 3b medians)",
                    transform=ax.transAxes, fontsize=5, color="0.4")
        ax.axvline(0, color="0.4", lw=0.7, ls="--")
        ax.set_yticks(range(1, len(donors) + 1))
        ax.set_yticklabels(donors, fontsize=5)
        ax.set_xlabel("log2 protease / inhibitor CPM", fontsize=7)
        ax.set_title(name.replace("_", " : "), fontsize=8)
    fig.suptitle("Degradative tone of the malignant clone, per donor's subclones", fontsize=9)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_clone_vs_tme(tbl, save=None, show=True):
    """Per gene: median subclone CPM vs the owning level's, both on their own full library."""
    import matplotlib.pyplot as plt

    d = tbl.dropna(subset=["owner_cpm_med"]).copy()
    if not len(d):
        print("no gene in the panel has a called owner -- nothing to plot")
        return None
    d = d.sort_values("clone_over_owner_log2")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(d) + 1.4))
    ax.hlines(y, np.log10(d["owner_cpm_med"] + 1), np.log10(d["clone_cpm_med"] + 1),
              color="0.75", lw=1.0, zorder=1)
    ax.scatter(np.log10(d["owner_cpm_med"] + 1), y, s=26, c="#3182bd", zorder=2,
               label="owning TME level (donor median)")
    ax.scatter(np.log10(d["clone_cpm_med"] + 1), y, s=26, c="#d62728", zorder=2,
               label="malignant subclones (median)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  ({o})" for g, o in zip(d["gene"], d["owner"])], fontsize=6.5)
    ax.set_xlabel("log10 (CPM + 1), each population on its own full-gene library", fontsize=7)
    ax.set_title("Intensity, abundance divided out: does the clone even make it?", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _finish(fig, save, show)


# ================================================================ run log


def append_run_log(title, body, path=None, verbose=True):
    """Append a self-describing block, readable pasted into a chat with no access to nb42."""
    path = (cfg.TAB_DIR / f"{cfg.TAB_PREFIX}run_log.md") if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# deg42 run log\n")
    bar = "=" * 110
    block = f"\n```\n{bar}\n{title}\n{bar}\n{body.rstrip()}\n```\n"
    with path.open("a") as fh:
        fh.write(block)
    if verbose:
        print(f"{bar}\n{title}\n{bar}\n{body.rstrip()}")
    return path
