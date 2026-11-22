"""Build and gate functions for the skin degradome object (notebook 28, atlas v2).

Two halves, and the split is load-bearing.

  BUILD (LSF only, `jobs/run_degradome_build.py`): label construction and the one streaming
  pass over `joint_annotated.h5ad`. Nothing here is called from the notebook except the
  asserts.

  GATE (notebook): the asserts, the coverage table and the panel-detection table. The analysis
  itself -- per-cell family scores, donor medians, study-blocked tests, the figures -- lives in
  `degradome_cells.py`, because the unit of analysis is the cell and everything here is about
  getting a correct object to analyse.

What is NOT here any more. The v1 notebook (`notebooks/_archive/old/28_tme_degradome_v1.ipynb`) carried a
donor-pseudobulk DESeq2 arm, an epidermis/dermis contrast, a source-attribution cube and a
subclone axis. All four are gone with it: the analysis is single-cell and study-blocked now, and
carrying a second statistical framework that nothing calls is how the frozen v1 constants
survived a v2 rebuild in the first place.

Reuse rather than reimplementation: `ccc_helpers.stream_lognorm_subset` does the streaming pass
and gets the library size right (see its docstring), `ccc_helpers.load_ccc_obs` /
`read_source_var` / `resolve_rows` / `sanitize_obs` do the I/O, and `malignancy.load_malignancy`
supplies the three malignancy calls.
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

# ================================================================ the level column


def load_subtype_sidecar(path=None, unassigned_path=None, verbose=True):
    """cell_id -> subtype label, from nb22 and (when it exists) nb22b.

    nb22 covers the myeloid and fibroblast compartments. nb22b covers what nobody had called:
    the never-re-clustered `Unk` broad cluster, the UNK residue of both lineages, and the
    NK-like block that nb21 folded into CD8. Its labels take precedence where the two overlap,
    because it is the later and more specific pass.

    Cells whose label is a non-identity (UNK, a doublet, a proliferating cluster) are dropped
    here rather than downstream: they would sit in the denominator without being attributable
    to anything.
    """
    path = cfg.SUBTYPE_CSV if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run 22_myeloid_fibro_reannotation §10 first.")
    side = pd.read_csv(path, dtype=str)
    lab = side.set_index("cell_id")[cfg.SUBTYPE_SRC]
    n_nb22 = len(lab)

    unassigned_path = cfg.UNASSIGNED_CSV if unassigned_path is None else Path(unassigned_path)
    n_nb22b = 0
    if unassigned_path.exists():
        extra = pd.read_csv(unassigned_path, dtype=str)
        ex = extra.set_index("cell_id")[cfg.RESIDUAL_SRC]
        n_nb22b = len(ex)
        lab = pd.concat([lab[~lab.index.isin(ex.index)], ex])
    elif verbose:
        print(f"  {unassigned_path.name} absent -- no residual levels (NK / ILC / the wider "
              f"neutrophil pool). nb22's {cfg.SIDECAR_COUNTS.get('Neutrophils', 0)} "
              f"neutrophils are all there is.")

    drop = lab.isna() | lab.isin(cfg.DROP_LEVELS)
    lab = lab[~drop]
    if verbose:
        print(f"sidecars: {n_nb22:,} cells from nb22"
              + (f" + {n_nb22b:,} from nb22b" if n_nb22b else "")
              + f"; {len(lab):,} carry an identity ({int(drop.sum()):,} dropped as "
                f"{sorted(set(cfg.DROP_LEVELS) & set(side[cfg.SUBTYPE_SRC].dropna()))})")
    return lab


def build_degradome_level(obs, subtype_csv=None, malignancy=None, verbose=True):
    """The analysis grouping: cell_type_final, with two substitutions.

      * Myeloid / Fibroblast are replaced by the nb22 subtype (and the nb22b residual labels);
      * CD4 is split into malignant / reactive / unassessed by the headline malignancy call.

    `CD4_unassessed` is a real level, not a dustbin -- it is the cells the call could not be
    evaluated on. Folding them into reactive would make "non-malignant CD4" mean "malignant was
    not detected", which is a different statement.

    Context levels (Keratinocyte, CD8, Vascular, ...) are kept: they are the denominator that
    makes a share interpretable. Returns a categorical in `cfg.CT_ORDER` order, NaN for cells
    with no identity.
    """
    import malignancy as MAL

    ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else obs.index.astype(str)

    # ---- CD4 split on the headline call
    if malignancy is None:
        malignancy = MAL.load_malignancy(
            ids, has_tcr=obs["has_tcr"] if "has_tcr" in obs.columns else None,
            verbose=verbose)
    # The lineage column arrives from the degradome object, not through
    # cluster_malig_helpers.read_obs, so the re-annotation overlay has to be applied here or the
    # CD4 split silently rebuilds the old contaminated arm. Non-T levels (Keratinocyte,
    # Fibroblast, ...) are absent from the label table and pass through untouched.
    import lineage as LIN
    _ct = LIN.apply_lineage(
        pd.DataFrame({cfg.CELLTYPE_SRC: obs[cfg.CELLTYPE_SRC].astype(str).to_numpy()},
                     index=pd.Index(ids, name="cell_id")),
        "skin", col=cfg.CELLTYPE_SRC)[cfg.CELLTYPE_SRC]
    base = pd.Series(MAL.cd4_level(_ct.to_numpy(), malignancy,
                                   call=cfg.MALIG_HEADLINE).astype(str), index=obs.index)

    # ---- myeloid / fibroblast subtypes
    fine = load_subtype_sidecar(subtype_csv, verbose=verbose)
    mapped = pd.Series(fine.reindex(pd.Index(ids)).to_numpy(), index=obs.index)
    is_split = obs[cfg.CELLTYPE_SRC].astype(str).isin(cfg.SPLIT_LINEAGES)

    label = base.astype(object).mask(is_split, mapped)
    # nb22b may also relabel cells that were `Unk` or `CD8` at the broad level.
    resid = mapped.notna() & ~is_split & base.isin(["Unk", "UNK", "CD8"])
    label = label.mask(resid, mapped)
    # Anything not in the vocabulary has no identity at this resolution. It cannot fall back to
    # the pooled level -- that level no longer exists here.
    label = label.where(label.isin(cfg.ALL_LEVELS), np.nan)

    order = [lv for lv in cfg.CT_ORDER if lv in set(label.dropna())]
    out = pd.Series(pd.Categorical(label, categories=order), index=obs.index, name=cfg.GROUPBY)
    if verbose:
        n_drop = int(out.isna().sum())
        print(f"\n{cfg.GROUPBY}: {len(out) - n_drop:,} cells labelled, "
              f"{n_drop:,} dropped ({n_drop / len(out):.1%})")
        print(out.value_counts().to_string())
    return out


def assert_degradome_roster(sidecar=None, sidecar_counts=None, verbose=True):
    """Raise unless the nb22 sidecar still has the composition the config was written against.

    This runs on the SIDECAR, not on the built object: the build is scoped to MF + HC, so its
    per-level counts are a subset of the sidecar's and cannot be compared to a frozen total.
    Checking the sidecar catches the thing that actually matters -- a re-run of nb22 that moved
    cells between levels while every downstream file still reads the old numbers.

    Returns the reconciliation frame, which goes into the build manifest.
    """
    sidecar_counts = cfg.SIDECAR_COUNTS if sidecar_counts is None else sidecar_counts
    if sidecar is None:
        side = pd.read_csv(cfg.SUBTYPE_CSV, dtype=str)
        sidecar = side[cfg.SUBTYPE_SRC]
    vc = pd.Series(sidecar).astype(str).value_counts()

    rows = [{"level": lv, "sidecar": int(vc.get(lv, 0)), "expected": int(want),
             "delta": int(vc.get(lv, 0)) - int(want)}
            for lv, want in sidecar_counts.items()]
    recon = pd.DataFrame(rows).sort_values("expected", ascending=False)

    bad = recon[recon["delta"] != 0]
    if len(bad):
        raise ValueError(
            "nb22 sidecar reconciliation failed -- the sidecar and "
            "degradome_data.SIDECAR_COUNTS disagree:\n"
            f"{bad.to_string(index=False)}\n"
            "If nb22 was deliberately re-run, update SIDECAR_COUNTS.")

    extra = sorted(set(vc.index) - set(sidecar_counts))
    if extra:
        raise ValueError(f"sidecar carries levels the config does not know: {extra}")

    if verbose:
        print(f"nb22 sidecar reconciled: {len(sidecar_counts)} levels, "
              f"{int(recon['sidecar'].sum()):,} cells, zero drift")
    return recon


# ================================================================ build (LSF only)


def assert_panel(var_names, include_context=True):
    """Raise unless every configured symbol is present. Run before anything else."""
    have = set(map(str, var_names))
    want = cfg.panel_genes(include_context=include_context)
    missing = [g for g in want if g not in have]
    if missing:
        raise ValueError(f"{len(missing)} panel genes absent from the object: {missing}")
    return want


def _pseudobulk_meta(obs, group_levels):
    """One row per (degradome_level, sample) pseudo-sample with its design metadata.

    `ccc_helpers` has a function of the same name, but it keys on `ccc_helpers.GROUPBY` -- the
    CCC cell-type column, not `cfg.GROUPBY`. Calling that one here would group the cube on a
    column this object does not have, so the degradome build carries its own.
    """
    design = [
        cfg.GROUPBY, cfg.SAMPLE_KEY, cfg.DONOR_KEY, cfg.STUDY_KEY, "disease", "entity",
        cfg.STAGE_KEY, "stage_class", cfg.LAYER_KEY, "tissue", "sex",
        "treatment_context", "lesion_type",
    ]
    design = [c for c in design if c in obs.columns]
    tmp = obs.assign(
        _g=obs[cfg.GROUPBY].astype(str) + "||" + obs[cfg.SAMPLE_KEY].astype(str))
    n_cells = tmp.groupby("_g", observed=True).size().rename("n_cells")
    meta = (tmp.drop_duplicates("_g").set_index("_g")[design]
            .join(n_cells).reindex(group_levels))
    meta.index.name = "pseudo_sample"
    return meta


def build_degradome_object(source=None, obs_parquet=None, out=None, pseudobulk_out=None,
                           pseudobulk_meta_out=None, testset_out=None, manifest_out=None,
                           chunk=50_000, verbose=True):
    """One streaming pass -> the degradome object + a FULL-GENE (level x sample) count cube.

    Called only from `jobs/run_degradome_build.py`. Two things it must get right, both
    inherited from `ccc_helpers.stream_lognorm_subset`:

      * the library size is summed over all 42,347 genes BEFORE the columns are subset.
        Normalising after subsetting would rescale every cell by the ratio of panel counts to
        total counts and produce plausible, wrong numbers everywhere downstream;
      * the pseudobulk cube is accumulated in the same pass over the FULL gene space, so
        DESeq2 estimates dispersions on the whole transcriptome rather than on the panel columns
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

    # Scope: MF + HC. Malignancy is annotated for MF and not for SS or CTCL_other, so
    # including them would put cells of unknown malignant status into a comparison whose axis
    # is malignant status. Applied BEFORE the label so the roster reconciles on what was built.
    in_scope = obs["disease"].astype(str).isin(cfg.DISEASES)
    if verbose:
        print(f"scope {cfg.DISEASES}: {int(in_scope.sum()):,} of {n_skin:,} skin cells "
              f"({int((~in_scope).sum()):,} dropped as out-of-scope disease)")
    obs = obs[in_scope].copy()
    n_scope = len(obs)

    label = build_degradome_level(obs, verbose=verbose)
    obs[cfg.GROUPBY] = label
    obs = obs[label.notna()].copy()
    recon = assert_degradome_roster(verbose=verbose)

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
        "warning": f"DEGRADOME-ONLY object: var is a hand-curated "
                   f"{int(keep_gene_mask.sum())}-gene panel. Not for HVG selection, PCA, "
                   f"UMAP, clustering or genome-wide DE. Use "
                   f"skin_degradome_pseudobulk_full.parquet for anything needing all genes.",
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
        "unassigned_csv": (str(cfg.UNASSIGNED_CSV) if cfg.UNASSIGNED_CSV.exists() else None),
        "malignancy": {"calls": list(cfg.MALIGNANCY), "headline": cfg.MALIG_HEADLINE},
        "compartment": cfg.COMPARTMENT,
        "diseases": list(cfg.DISEASES),
        "atlas": "v2",
        "n_skin_cells": int(n_skin),
        "n_in_scope": int(n_scope),
        "n_obs": int(adata.n_obs),
        "n_dropped_no_identity": int(n_scope - adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_genes_source": int(len(var)),
        "nnz_lognorm": int(lognorm.nnz),
        "nnz_counts": int(counts.nnz),
        "level_counts": adata.obs[cfg.GROUPBY].value_counts().to_dict(),
        "sidecar_reconciliation": recon.to_dict("records"),
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

    Its only purpose is to prove that restricting var to the panel does not change a
    single expression value -- i.e. that the size factor was summed over all 42,347 genes and
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
        wanted = ["li2024", "chennareddy2025", "gaydosik2022"]
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

# ================================================================ load + gate


def load_degradome_adata(path=None, manifest=None, verbose=True):
    """Read the built object and re-run every build-time assert before handing it over."""
    import scanpy as sc

    path = cfg.DEG_ADATA if path is None else Path(path)
    manifest = cfg.MANIFEST if manifest is None else Path(manifest)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run jobs/run_degradome_build.sh first (one bsub pass "
            f"over the 112 GB source atlas; it is not a notebook step).")
    adata = sc.read_h5ad(path)
    assert_panel(adata.var_names)

    if cfg.GROUPBY not in adata.obs.columns:
        raise ValueError(f"{path.name} carries no {cfg.GROUPBY} column -- stale build")
    adata.obs[cfg.GROUPBY] = pd.Categorical(
        adata.obs[cfg.GROUPBY].astype(str),
        categories=[lv for lv in cfg.CT_ORDER
                    if lv in set(adata.obs[cfg.GROUPBY].astype(str))])
    assert_degradome_roster(verbose=verbose)

    out_of_scope = set(adata.obs["disease"].astype(str)) - set(cfg.DISEASES)
    if out_of_scope:
        raise ValueError(f"object carries out-of-scope diseases {sorted(out_of_scope)} -- "
                         f"stale build, rebuild with the current degradome_data.DISEASES")

    if verbose:
        man = json.loads(manifest.read_text()) if manifest.exists() else {}
        print(f"{path.name}: {adata.n_obs:,} cells x {adata.n_vars} genes, "
              f"{adata.obs[cfg.DONOR_KEY].nunique()} donors, "
              f"{adata.obs[cfg.SAMPLE_KEY].nunique()} samples, "
              f"{adata.obs[cfg.STUDY_KEY].nunique()} studies")
        if man:
            print(f"  scope {man.get('diseases')}: kept {man.get('n_in_scope', 0):,} of "
                  f"{man.get('n_skin_cells', 0):,} skin cells; dropped "
                  f"{man.get('n_dropped_no_identity', 0):,} with no identity")
        print(f"  {adata.uns['degradome_build']['warning']}")
    return adata


def coverage_table(obs, min_cells=None, min_samples=None, out=None, verbose=True):
    """Per level x stage arm: cells, eligible donors, studies, and what may be claimed.

    Eligibility is per (level, donor): a donor contributes to a level only if it carries at
    least `min_cells` of it. `n_donors_ge{min_cells}` is therefore the real n of every figure
    that overlays donor medians, and it is the number the abstention table is built from.

    `top_study_frac` is computed on the arm, not on the level, because that is where the
    confound lives: no study in this cohort carries all three of HC, early and advanced.
    """
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    min_samples = cfg.MIN_SAMPLES if min_samples is None else min_samples

    lv_all = obs[cfg.GROUPBY].astype(str)
    stage = obs[cfg.STAGE_KEY].astype(str)
    rows = []
    for lv in [x for x in cfg.CT_ORDER if x in set(lv_all)]:
        sub = obs[lv_all == lv]
        rec = {cfg.GROUPBY: lv, "claim": lv in cfg.CLAIM_LEVELS, "n_cells": len(sub)}
        for arm in cfg.STAGE_ARMS:
            a = sub[stage.loc[sub.index] == arm]
            per_donor = a.groupby(cfg.DONOR_KEY, observed=True).size()
            elig = per_donor[per_donor >= min_cells]
            rec[f"n_cells_{arm}"] = len(a)
            rec[f"n_donors_{arm}"] = int(len(elig))
            if len(elig):
                st = (a[a[cfg.DONOR_KEY].isin(elig.index)][cfg.STUDY_KEY]
                      .astype(str).value_counts(normalize=True))
                rec[f"top_study_{arm}"] = st.index[0]
                rec[f"top_study_frac_{arm}"] = round(float(st.iloc[0]), 3)
            else:
                rec[f"top_study_{arm}"] = None
                rec[f"top_study_frac_{arm}"] = np.nan
        rows.append(rec)
    cov = pd.DataFrame(rows)

    cov["claimable"] = (cov[[f"n_donors_{a}" for a in cfg.STAGE_ARMS]].max(axis=1)
                        >= min_samples)
    for ref, alt in [("HC", "advanced"), ("early", "advanced")]:
        n = cov[[f"n_donors_{ref}", f"n_donors_{alt}"]].min(axis=1)
        cov[f"testable_{ref}_vs_{alt}"] = n >= cfg.MIN_DONORS_PER_ARM

    if out is not None:
        cov.to_csv(out, index=False)
    if verbose:
        show = [cfg.GROUPBY, "claim", "n_cells"] + \
               [f"n_donors_{a}" for a in cfg.STAGE_ARMS] + \
               ["claimable", "testable_HC_vs_advanced", "testable_early_vs_advanced"]
        print(cov[show].to_string(index=False))
        bad = cov[cov["claim"] & ~cov["claimable"]]
        if len(bad):
            print(f"\nclaim levels failing the {min_cells}-cell / {min_samples}-donor gate: "
                  f"{list(bad[cfg.GROUPBY])}")
    return cov


def abstentions(cov, out=None, verbose=True):
    """Every (level, axis) this cohort cannot test, with the measured reason.

    A deliverable, not a footnote. "No HC donor carries 25 of this level" and "the effect was
    null" are different findings and the figures must not make them look alike.
    """
    rows = []
    for _, r in cov.iterrows():
        for ref, alt in [("HC", "advanced"), ("early", "advanced")]:
            if r[f"testable_{ref}_vs_{alt}"]:
                continue
            n_ref, n_alt = int(r[f"n_donors_{ref}"]), int(r[f"n_donors_{alt}"])
            short = ref if n_ref < n_alt else alt
            rows.append({
                "level": r[cfg.GROUPBY], "claim": bool(r["claim"]),
                "axis": f"{ref}_vs_{alt}",
                "n_donors_ref": n_ref, "n_donors_alt": n_alt,
                "reason": (f"{short} arm has {min(n_ref, n_alt)} eligible donors, "
                           f"below MIN_DONORS_PER_ARM={cfg.MIN_DONORS_PER_ARM}"),
            })
    tbl = pd.DataFrame(rows)
    if out is not None:
        tbl.to_csv(out, index=False)
    if verbose and len(tbl):
        n_claim = int(tbl["claim"].sum())
        print(f"{len(tbl)} (level, axis) abstentions, {n_claim} of them on claim levels")
        print(tbl[tbl["claim"]].to_string(index=False))
    return tbl


def panel_coverage(adata, min_detection=None, out=None, verbose=True):
    """Detection fraction per (gene, level) and the drop list.

    A family score built on genes nothing expresses is a score of zero with noise on top, so
    genes never detected above `min_detection` in any level are flagged -- and the flag is
    written down. Silent truncation reads as "we looked at everything" when we did not.
    """
    import scipy.sparse as sp

    min_detection = cfg.MIN_DETECTION if min_detection is None else min_detection
    lv = adata.obs[cfg.GROUPBY].astype(str).to_numpy()
    levels = [x for x in cfg.CT_ORDER if x in set(lv)]
    X = adata.layers[cfg.COUNTS_LAYER] if cfg.COUNTS_LAYER in adata.layers else adata.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()

    frac = np.zeros((len(levels), adata.n_vars))
    for i, level in enumerate(levels):
        m = lv == level
        n = int(m.sum())
        if n:
            frac[i] = np.asarray((X[m] > 0).sum(axis=0)).ravel() / n

    cov = pd.DataFrame(frac.T, index=list(map(str, adata.var_names)), columns=levels)
    cov.insert(0, "family", [cfg.GENE_TO_FAMILY.get(g, "context")
                             for g in map(str, adata.var_names)])
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
            print(f"  below threshold ({len(dropped)}): {', '.join(dropped)}")
            byfam = (cov.loc[dropped].groupby("family").size()
                     .sort_values(ascending=False))
            print(f"  by family: {byfam.to_dict()}")
    return cov, dropped


# ================================================================ misc


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
