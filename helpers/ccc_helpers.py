"""Helpers for the MF / CTCL skin cell-cell-communication analysis (notebooks 34-36).

Ported from MyelomaProject/ccc/ccc_functions.py; the mouse ortholog machinery is
dropped (this atlas is human) and a streaming build is added in its place.

THE 48 GB SOURCE ATLAS IS NEVER READ WHOLE. `build_ccc_object` streams row-chunks
with `anndata.io.sparse_dataset` and keeps only the ~1,832 consensus-resource
genes. That is lossless for LIANA -- `rank_aggregate` subsets to resource genes
internally -- PROVIDED the log-normalisation size factor is computed over all
40,821 genes BEFORE the columns are subset. `stream_lognorm_subset` does that,
and it is the one correctness-critical step in the whole pipeline. Getting it
wrong inflates every value by ~16x and produces plausible, wrong lr_means, which
is why nb34 has an explicit full-gene-vs-subset equivalence gate.

Only the build is heavy (LSF, ~15 min of I/O). Everything downstream runs
interactively in the `mrvi_env` kernel off the ~2 GB CCC object. `neural_nmf_env`
has no liana.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import liana as li

from ccc_data import (
    ANALYSES,
    CD4_MALIGNANT,
    CD4_REACTIVE,
    CD4_UNASSESSED,
    CELLTYPE_SRC,
    CCC_ADATA,
    COMPARTMENT,
    COUNTS_LAYER,
    CT_ORDER,
    DONOR_KEY,
    DOTPLOT_SIZE,
    DOTPLOT_TOP_N,
    DOWNSAMPLE_SIZES,
    DROP_LEVELS,
    EVIDENCE_SRC,
    EXPR_PROP,
    EXPR_PROP_SWEEP,
    FINE_SRC,
    FOCAL_AXES,
    GROUPBY,
    HEATMAP_BAD,
    HEATMAP_CMAP,
    HEATMAP_TOP_N,
    LAYER,
    LR_CAVEATS,
    LR_PANEL,
    MALIG_ALT,
    MALIG_SRC,
    MANIFEST,
    MIN_CELLS,
    MIN_SAMPLES,
    N_JOBS,
    N_PERMS,
    N_SHUFFLES,
    NEGATIVE_CONTROLS,
    OBS_COLS,
    OBS_PARQUET,
    POSITIVE_CONTROLS,
    PSEUDOBULK_META,
    PSEUDOBULK_PQ,
    PVAL_ALPHA,
    RESOURCE_COVERAGE_CHECK,
    RESOURCE_NAME,
    SAMPLE_KEY,
    SEED,
    SOURCE_H5AD,
    STUDY_KEY,
    SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL,
    SUBSAMPLE_MAX_PER_LEVEL,
    SUBSAMPLE_SEED,
    TCR_ASSESSED_SRC,
    TESTSET_H5AD,
    TME_THIN,
    TOP_N,
)

KEY_COLS = ["source", "target", "ligand_complex", "receptor_complex"]


# ================================================================ resource


def resource_genes(resource):
    """Sorted unique subunit symbols; complexes are split on '_'."""
    genes = set()
    for col in ("ligand", "receptor"):
        for entry in resource[col].astype(str):
            genes.update(entry.split("_"))
    return sorted(genes)


def _complex_ok(entry, present):
    return all(sub in present for sub in str(entry).split("_"))


def load_resource(resource_name=RESOURCE_NAME, var_names=None, verbose=True):
    """Human consensus LIANA resource. Returns (resource, coverage).

    No ortholog translation -- the Myeloma pipeline needed HCOP because its atlas
    is mouse. `var_names` turns the coverage dict into a data-specific report:
    how many interactions have every subunit present, and which of the genes the
    controls depend on are absent.
    """
    resource = li.rs.select_resource(resource_name)[["ligand", "receptor"]]
    genes = resource_genes(resource)

    coverage = {
        "resource": resource_name,
        "n_interactions": len(resource),
        "n_subunit_genes": len(genes),
    }
    if var_names is not None:
        present = set(map(str, var_names))
        ok = resource["ligand"].map(lambda x: _complex_ok(x, present)) & resource[
            "receptor"
        ].map(lambda x: _complex_ok(x, present))
        coverage.update(
            n_genes_present=sum(g in present for g in genes),
            n_genes_missing=sum(g not in present for g in genes),
            n_interactions_covered=int(ok.sum()),
            frac_interactions_covered=float(ok.mean()),
            missing_checks=[g for g in RESOURCE_COVERAGE_CHECK if g not in present],
        )
    if verbose:
        print(
            f"{resource_name}: {coverage['n_interactions']} interactions over "
            f"{coverage['n_subunit_genes']} subunit genes"
        )
        if var_names is not None:
            print(
                f"  {coverage['n_genes_present']}/{coverage['n_subunit_genes']} genes in data; "
                f"{coverage['n_interactions_covered']} interactions fully covered "
                f"({coverage['frac_interactions_covered']:.1%})"
            )
            miss = coverage["missing_checks"]
            print("  control genes missing:", miss if miss else "none")
    return resource, coverage


def _subunits(entry):
    return frozenset(str(entry).split("_"))


def resolve_pairs(pairs, resource, verbose=False):
    """Map human-readable (ligand, receptor) pairs onto the resource's spelling.

    Necessary because the consensus resource does not spell interactions the way
    a person writes them:
      * complex subunits are sorted alphabetically  (IL4R_IL2RG -> IL2RG_IL4R);
      * some ligands are themselves complexes       (CSF1 -> CSF1R is CSF1_IL34);
      * some receptors carry extra obligate chains  (IL7R -> IL2RG_IL7R,
                                                     KLRC1 -> KLRC1_KLRD1);
      * a few pairs are stored with the receptor in the ligand column
        (TIGIT -> PVR rather than PVR -> TIGIT).
    Hard-coding these by hand is how a real interaction gets reported as absent.

    A curated side matches a resource side when its subunits are a subset of that
    side's subunits, so IL7 -> IL7R resolves to IL7 -> IL2RG_IL7R. `orientation`
    is "flipped" when the match required swapping the two sides -- callers that
    carry a direction (the controls) must swap source/target with it.

    Returns one row per resource match; a curated pair with no match gets one row
    with n_matches == 0, so a resource gap is visible instead of silent.
    """
    lig_sub = resource["ligand"].map(_subunits)
    rec_sub = resource["receptor"].map(_subunits)

    rows = []
    for ligand, receptor in pairs:
        lig_in = ligand if isinstance(ligand, str) else "_".join(ligand)
        rec_in = receptor if isinstance(receptor, str) else "_".join(receptor)
        want_l, want_r = _subunits(lig_in), _subunits(rec_in)

        fwd = lig_sub.map(want_l.issubset) & rec_sub.map(want_r.issubset)
        if fwd.any():
            hits = [("forward", i) for i in resource.index[fwd]]
        else:
            flip = lig_sub.map(want_r.issubset) & rec_sub.map(want_l.issubset)
            hits = [("flipped", i) for i in resource.index[flip]]
        if not hits:
            rows.append(
                {
                    "ligand_in": lig_in,
                    "receptor_in": rec_in,
                    "ligand_complex": None,
                    "receptor_complex": None,
                    "orientation": "absent",
                    "n_matches": 0,
                }
            )
            continue
        for orientation, i in hits:
            rows.append(
                {
                    "ligand_in": lig_in,
                    "receptor_in": rec_in,
                    "ligand_complex": resource.at[i, "ligand"],
                    "receptor_complex": resource.at[i, "receptor"],
                    "orientation": orientation,
                    "n_matches": len(hits),
                }
            )
    out = pd.DataFrame(rows).drop_duplicates()
    if verbose:
        absent = out[out["orientation"] == "absent"]
        print(f"resolved {out['ligand_in'].nunique() - len(absent)}/{out['ligand_in'].nunique()} pairs")
        if len(absent):
            print("absent from resource:")
            print(absent[["ligand_in", "receptor_in"]].to_string(index=False))
    return out


def resolve_panel(panel=LR_PANEL, resource=None, verbose=False):
    """lr_panel_frame, but with the resource's own complex spelling."""
    frame = lr_panel_frame(panel)
    if resource is None:
        resource, _ = load_resource(verbose=False)
    resolved = resolve_pairs(
        list(zip(frame["ligand_complex"], frame["receptor_complex"])), resource, verbose=verbose
    )
    return frame.rename(
        columns={"ligand_complex": "ligand_in", "receptor_complex": "receptor_in"}
    ).merge(resolved, on=["ligand_in", "receptor_in"], how="left")


def resolve_controls(controls, resource, verbose=False):
    """Resolve (source, target, ligand, receptor) controls to resource spelling.

    When the resource stores a pair flipped, source and target swap with it --
    otherwise the control would look for the edge in the direction liana never
    scores and report a false negative.
    """
    resolved = resolve_pairs([(l, r) for _s, _t, l, r in controls], resource, verbose=verbose)
    out = []
    for source, target, ligand, receptor in controls:
        lig_in = ligand if isinstance(ligand, str) else "_".join(ligand)
        rec_in = receptor if isinstance(receptor, str) else "_".join(receptor)
        hits = resolved[(resolved["ligand_in"] == lig_in) & (resolved["receptor_in"] == rec_in)]
        for h in hits.itertuples(index=False):
            if h.orientation == "absent":
                out.append((source, target, lig_in, rec_in))
            elif h.orientation == "flipped":
                out.append((target, source, h.ligand_complex, h.receptor_complex))
            else:
                out.append((source, target, h.ligand_complex, h.receptor_complex))
    return list(dict.fromkeys(out))


def panel_coverage(resource, panel=LR_PANEL, var_names=None):
    """Per curated pair: is it in the RESOURCE, and are its genes in the DATA?

    Two different failures with the same appearance in a result table. A pair
    absent from the resource can never be scored no matter how well expressed; a
    pair absent from the data is a real negative. Conflating them turns a
    bookkeeping gap into a biological claim, so they get separate columns.
    """
    frame = resolve_panel(panel, resource)
    present = set(map(str, var_names)) if var_names is not None else None

    frame = frame.assign(in_resource=frame["orientation"] != "absent")
    if present is not None:
        def missing_for(row):
            side = (row["ligand_complex"] or row["ligand_in"], row["receptor_complex"] or row["receptor_in"])
            subs = [s for entry in side for s in str(entry).split("_") if s != "*"]
            return ",".join(s for s in subs if s not in present)

        frame["genes_missing_from_data"] = frame.apply(missing_for, axis=1)
        frame["in_data"] = frame["genes_missing_from_data"] == ""
    return frame[
        [c for c in [
            "group", "ligand_in", "receptor_in", "ligand_complex", "receptor_complex",
            "orientation", "in_resource", "in_data", "genes_missing_from_data",
        ] if c in frame.columns]
    ]


# ================================================================ labels


def load_ccc_obs(obs_parquet=OBS_PARQUET, compartment=COMPARTMENT, cols=OBS_COLS, verbose=True):
    """Read the obs parquet and restrict to one compartment. Never opens an h5ad.

    The 48 GB source atlas obs does NOT carry cell_type_final / mal_tcr_alice /
    stage_group -- this parquet (written by nb32) is the only place they exist.
    """
    obs = pd.read_parquet(obs_parquet, columns=[c for c in cols if c != "cell_id"] + ["cell_id"])
    if compartment is not None:
        obs = obs[obs["compartment"].astype(str) == compartment]
    if verbose:
        print(
            f"{len(obs)} {compartment} cells, {obs[DONOR_KEY].nunique()} donors, "
            f"{obs[SAMPLE_KEY].nunique()} samples, {obs[STUDY_KEY].nunique()} studies"
        )
    return obs


def _malig_flag(obs, col):
    """Tri-state malignancy column -> (is_true, is_false, is_null) boolean masks."""
    raw = obs[col]
    is_null = raw.isna()
    as_bool = raw.fillna(False).astype(bool)
    return as_bool & ~is_null, ~as_bool & ~is_null, is_null


def build_ccc_celltype(
    obs,
    key=GROUPBY,
    celltype_src=CELLTYPE_SRC,
    malig_src=MALIG_SRC,
    tcr_assessed_src=TCR_ASSESSED_SRC,
    keep_unassessed=True,
    drop_levels=DROP_LEVELS,
    drop_cd8_malcall=False,
    verbose=True,
):
    """Collapse cell_type_final + a malignancy call into one grouping column.

    ONLY CD4 is split, because ALICE is a CD4-only caller:
        CD4 & call is True  -> CD4_malignant    ( 76,349 with mal_tcr_alice)
        CD4 & call is False -> CD4_reactive     ( 72,207)
        CD4 & call is null  -> CD4_unassessed   (190,142) own level, never pooled

    `tcr_assessed_src` is the second half of the "unassessed" definition and the
    reason the reactive level is smaller than a naive `== False`. 59,658 CD4 have
    no TRB CDR3 at all, so ALICE never tested them; they come back False from the
    column but that is absence of data, not evidence of benignity. They are
    demoted to CD4_unassessed rather than diluting the comparator. Pass
    `tcr_assessed_src=None` to skip the gate -- which is what
    `build_ccc_celltype_from` does, because mal_cnv's missingness is CNV-set
    membership and tumor_cell's is li2024 membership; gating either on CDR3
    recovery would mix two unrelated mechanisms and break the §13 comparison.

    Tregs are not split: 1,936 cells means no split clears MIN_CELLS in any donor,
    and ALICE calls none of them anyway. CD8 is not split either -- ALICE assigns
    it no malignant cell at all, and mal_combined's 21,200 True calls were 100%
    cnv_only with zero TCR support, i.e. the signature of a CNV false positive or
    ambient bleed from the dominant CD4 clone rather than a CD8 malignancy.
    `drop_cd8_malcall=True` removes those cells for the sensitivity run instead of
    pretending they are a separate state.

    Returns a categorical Series on obs.index, NaN where the cell is dropped.
    """
    ct = obs[celltype_src].astype(str)
    label = ct.copy()

    is_cd4 = ct == "CD4"
    mal_true, mal_false, mal_null = _malig_flag(obs, malig_src)

    if tcr_assessed_src is not None:
        untestable = ~obs[tcr_assessed_src].fillna(False).astype(bool)
        demoted = mal_false & untestable
        mal_false = mal_false & ~untestable
        mal_null = mal_null | demoted
        if verbose:
            print(
                f"{int((is_cd4 & demoted).sum())} CD4 negative for {malig_src} with no "
                f"{tcr_assessed_src} -> CD4_unassessed (untestable, not benign)"
            )

    label = label.mask(is_cd4 & mal_true, CD4_MALIGNANT)
    label = label.mask(is_cd4 & mal_false, CD4_REACTIVE)
    label = label.mask(is_cd4 & mal_null, CD4_UNASSESSED if keep_unassessed else np.nan)

    if drop_cd8_malcall:
        n = int(((ct == "CD8") & mal_true).sum())
        label = label.mask((ct == "CD8") & mal_true, np.nan)
        if verbose:
            print(f"drop_cd8_malcall: removed {n} CD8 cells with a cnv_only malignant call")

    label = label.replace({lv: np.nan for lv in drop_levels})
    label = label.where(label.notna() & (label != "nan"), np.nan)

    order = [lv for lv in CT_ORDER if lv in set(label.dropna())]
    extra = sorted(set(label.dropna()) - set(order))
    out = pd.Series(
        pd.Categorical(label, categories=order + extra), index=obs.index, name=key
    )

    if verbose:
        n_drop = int(out.isna().sum())
        print(f"dropping {n_drop} cells ({n_drop / len(obs):.1%}) with no CCC label")
        print(out.value_counts().to_string())
    return out


def build_ccc_celltype_from(obs, malig_col, **kwargs):
    """Same grouping built from an alternative malignancy definition.

    `malig_col` may be one of MALIG_ALT, or the literal string "tumor_cell" to use
    li2024's own paper label (which is `Unknown` outside li2024, so the resulting
    CD4_unassessed level absorbs every non-li2024 CD4).

    The CDR3 gate is switched OFF here on purpose: it encodes "could ALICE test
    this cell", which is meaningless for a CNV- or paper-label-derived call. Each
    alternative definition carries its own missingness and is left to express it.
    """
    kwargs.setdefault("tcr_assessed_src", None)
    if malig_col == "tumor_cell":
        alt = obs[FINE_SRC].astype(str)
        flag = pd.Series(np.nan, index=obs.index, dtype="object")
        flag[alt == "tumor_cell"] = True
        flag[(alt != "tumor_cell") & (alt != "Unknown")] = False
        obs = obs.assign(_alt_malig=flag)
        malig_col = "_alt_malig"
    return build_ccc_celltype(obs, malig_src=malig_col, **kwargs)


def build_ccc_celltype_sub(
    obs,
    subtype_csv,
    keep_levels,
    key="ccc_celltype_sub",
    split_src=("Myeloid", "Fibroblast"),
    celltype_src=CELLTYPE_SRC,
    verbose=True,
    **kwargs,
):
    """`build_ccc_celltype`, then Myeloid/Fibroblast replaced by their nb10c sub-levels.

    `subtype_csv` is data/atlas_joint/skin_myeloid_fibro_subtypes.csv (nb10c): one row per
    myeloid or fibroblast skin cell, columns `cell_id, lineage, subtype_fine, subtype_ccc`.
    `subtype_ccc` is the coverage-driven collapse -- a fine state that cleared MIN_CELLS in
    fewer than MIN_SAMPLES donors was merged into its parent there, so nothing arrives here
    that cannot in principle carry a claim. Cells whose `subtype_ccc` is null (UNK,
    proliferating, pericyte contamination) are dropped, exactly as DROP_LEVELS drops UNK.

    `keep_levels` is the roster. Anything outside it -- CD4_unassessed, Tregs, Plasma,
    Vascular, Mast, Melanocyte in the nb38/39 design -- becomes NaN and is dropped by the
    caller, the same contract as `build_ccc_celltype`. Restricting the roster is NOT the same
    as never having computed those levels: they exist in ccc_skin.h5ad and in the nb35/36
    results, and this is a deliberate narrowing of scope, not a correction of them.

    Returns a categorical Series on obs.index, NaN where the cell is dropped.
    """
    # verbose=False on the base call: its value_counts would print the POOLED distribution, which
    # is not the labelling this function returns and reads as a contradiction two lines later.
    base = build_ccc_celltype(obs, key=key, celltype_src=celltype_src, verbose=False, **kwargs)

    sub = pd.read_csv(subtype_csv, dtype=str)
    sub = sub[sub["subtype_ccc"].notna() & ~sub["subtype_ccc"].isin(["nan", "None", "NA", ""])]
    sub = sub.set_index("cell_id")["subtype_ccc"]

    ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else obs.index.astype(str)
    ct = obs[celltype_src].astype(str)
    is_split = ct.isin(list(split_src)).to_numpy()

    label = base.astype(object)
    fine = sub.reindex(pd.Index(ids)).to_numpy()
    label = label.mask(pd.Series(is_split, index=obs.index), pd.Series(fine, index=obs.index))

    if verbose:
        n_split = int(is_split.sum())
        n_hit = int(pd.notna(fine[is_split]).sum())
        print(
            f"{n_hit}/{n_split} {'/'.join(split_src)} cells carry a subtype_ccc "
            f"({n_split - n_hit} dropped as UNK/prolif/contaminant)"
        )

    keep = list(keep_levels)
    label = label.where(label.isin(keep), np.nan)

    order = [lv for lv in keep if lv in set(label.dropna())]
    out = pd.Series(pd.Categorical(label, categories=order), index=obs.index, name=key)
    if verbose:
        n_drop = int(out.isna().sum())
        print(f"dropping {n_drop} cells ({n_drop / len(obs):.1%}) outside the roster")
        print(out.value_counts().to_string())
    return out


def malignancy_definition_audit(obs, defs=None, celltype_src=CELLTYPE_SRC, verbose=True):
    """Pairwise agreement between the malignancy calls, restricted to CD4.

    Includes li2024's own `tumor_cell` label. Agreement is only moderate
    (mal_combined vs tumor_cell: 48,916 both-positive, 3,288 tumor_cell-only,
    72,676 mal_combined-only), which is why the definition sensitivity run in
    nb35 is mandatory rather than optional.
    """
    defs = [MALIG_SRC, *MALIG_ALT] if defs is None else list(defs)
    cd4 = obs[obs[celltype_src].astype(str) == "CD4"]

    flags = {}
    for col in defs:
        t, _f, _n = _malig_flag(cd4, col)
        flags[col] = t
    tumor = cd4[FINE_SRC].astype(str) == "tumor_cell"
    flags["li2024_tumor_cell"] = tumor

    names = list(flags)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            fa, fb = flags[a], flags[b]
            inter = int((fa & fb).sum())
            union = int((fa | fb).sum())
            rows.append(
                {
                    "def_a": a,
                    "def_b": b,
                    "n_a": int(fa.sum()),
                    "n_b": int(fb.sum()),
                    "both": inter,
                    "a_only": int((fa & ~fb).sum()),
                    "b_only": int((~fa & fb).sum()),
                    "jaccard": inter / union if union else np.nan,
                }
            )
    audit = pd.DataFrame(rows)
    if verbose:
        print(audit.to_string(index=False))
        if EVIDENCE_SRC in cd4.columns:
            print("\nmalignant_evidence within CD4:")
            print(cd4[EVIDENCE_SRC].value_counts(dropna=False).to_string())
    return audit


def _coerce_obs_column(series):
    """Give a column a writable dtype -- h5py cannot write object columns."""
    if series.dtype != object:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if series.notna().any() and numeric.notna().sum() == series.notna().sum():
        return numeric.astype(float)
    return series.astype(str).astype("category")


def sanitize_obs(adata, verbose=True):
    """Coerce leftover object columns in .obs so the object can be written."""
    fixed = []
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = _coerce_obs_column(adata.obs[col])
            fixed.append((col, str(adata.obs[col].dtype)))
    if verbose and fixed:
        print("coerced object columns:", fixed)
    return adata


# ================================================================ build (LSF only)


def _open_source(h5_path):
    import h5py

    return h5py.File(h5_path, "r")


def read_source_var(h5_path=SOURCE_H5AD):
    """var frame of the source atlas, without touching X."""
    from anndata.io import read_elem

    with _open_source(h5_path) as f:
        return read_elem(f["var"])


def read_source_index(h5_path=SOURCE_H5AD):
    """obs index (cell_id) of the source atlas, without touching X."""
    from anndata.io import read_elem

    with _open_source(h5_path) as f:
        idx_key = f["obs"].attrs.get("_index", "_index")
        return pd.Index(np.asarray(read_elem(f["obs"][idx_key])).astype(str))


def resolve_rows(source_index, cell_ids):
    """Positions of `cell_ids` in `source_index`; raises on any miss.

    Returned sorted -- the h5 sparse dataset wants increasing row order, and
    every downstream frame is reindexed to match rather than the other way round.
    """
    pos = pd.Series(np.arange(len(source_index)), index=source_index)
    hit = pos.reindex(pd.Index(cell_ids).astype(str))
    n_miss = int(hit.isna().sum())
    if n_miss:
        raise ValueError(f"{n_miss} cell_ids absent from the source atlas index")
    rows = np.sort(hit.to_numpy().astype(np.int64))
    return rows, source_index[rows]


def _group_indicator(codes, n_groups):
    """(n_rows x n_groups) csr selector for a group-sum via matrix product."""
    keep = codes >= 0
    rows = np.nonzero(keep)[0]
    return sp.csr_matrix(
        (np.ones(len(rows)), (rows, codes[keep])), shape=(len(codes), n_groups)
    )


def stream_lognorm_subset(
    h5_path,
    rows,
    keep_gene_mask,
    counts_layer=COUNTS_LAYER,
    chunk=50_000,
    target_sum=1e4,
    group_codes=None,
    n_groups=0,
    verbose=True,
):
    """One pass over row-chunks -> (lognorm csr float32, counts csr int32, cube).

    The library size is summed over ALL genes of each row and only THEN are the
    columns subset. Normalising after subsetting would rescale every cell by the
    ratio of resource-gene counts to total counts (~16x here) and silently
    corrupt every lr_mean.

    `group_codes` (aligned to `rows`, -1 to skip) additionally accumulates
    full-gene raw-count sums per group into a dense (n_groups x n_genes) cube, so
    the pseudobulk by-product costs no extra I/O.
    """
    import time

    from anndata.io import sparse_dataset

    lognorm_blocks, counts_blocks = [], []
    cube = None
    n_genes_total = int(keep_gene_mask.shape[0])
    if group_codes is not None:
        cube = np.zeros((n_groups, n_genes_total), dtype=np.float64)

    t0 = time.time()
    with _open_source(h5_path) as f:
        X = sparse_dataset(f[f"layers/{counts_layer}"])
        for start in range(0, len(rows), chunk):
            sel = rows[start : start + chunk]
            blk = X[sel]
            if not sp.issparse(blk):
                blk = sp.csr_matrix(blk)
            blk = blk.tocsr()

            if start == 0 and not np.allclose(blk.data, np.rint(blk.data)):
                raise ValueError(
                    f"layers/{counts_layer} is not integral -- it is not raw counts"
                )

            lib = np.asarray(blk.sum(axis=1)).ravel()   # ALL genes, before subsetting
            lib[lib == 0] = 1.0

            if cube is not None:
                codes = group_codes[start : start + chunk]
                cube += (_group_indicator(codes, n_groups).T @ blk).toarray()

            sub = blk[:, keep_gene_mask]
            counts_blocks.append(sub.astype(np.int32))
            lognorm_blocks.append(
                sp.diags(target_sum / lib).dot(sub).log1p().astype(np.float32).tocsr()
            )

            if verbose:
                done = min(start + chunk, len(rows))
                print(
                    f"  {done}/{len(rows)} rows  ({time.time() - t0:.0f}s)", flush=True
                )

    lognorm = sp.vstack(lognorm_blocks, format="csr")
    counts = sp.vstack(counts_blocks, format="csr")
    return lognorm, counts, cube


def build_ccc_object(
    source=SOURCE_H5AD,
    obs_parquet=OBS_PARQUET,
    out=CCC_ADATA,
    pseudobulk_out=PSEUDOBULK_PQ,
    pseudobulk_meta_out=PSEUDOBULK_META,
    testset_out=TESTSET_H5AD,
    manifest_out=MANIFEST,
    resource_name=RESOURCE_NAME,
    compartment=COMPARTMENT,
    chunk=50_000,
    testset_donors=None,
    verbose=True,
):
    """Build the CCC object, the pseudobulk cube, the equivalence test set and a
    manifest, in a single streaming pass. Called only from jobs/run_ccc_build.py.
    """
    import anndata as ad

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    obs = load_ccc_obs(obs_parquet, compartment=compartment, verbose=verbose)
    obs = obs.set_index("cell_id", drop=False)
    label = build_ccc_celltype(obs, verbose=verbose)
    obs[GROUPBY] = label
    obs = obs[label.notna()].copy()
    if verbose:
        print(f"kept {len(obs)} cells after dropping unlabelled")

    var = read_source_var(source)
    if not var.index.is_unique:
        raise ValueError("source var index is not unique")
    resource, coverage = load_resource(resource_name, var_names=var.index, verbose=verbose)
    genes = resource_genes(resource)
    keep_gene_mask = np.asarray(var.index.isin(genes))
    if verbose:
        print(f"keeping {int(keep_gene_mask.sum())} of {len(var)} genes")

    source_index = read_source_index(source)
    rows, cell_ids = resolve_rows(source_index, obs["cell_id"].to_numpy())
    obs = obs.loc[cell_ids]

    groups = (
        obs[GROUPBY].astype(str) + "||" + obs[SAMPLE_KEY].astype(str)
    ).to_numpy()
    group_levels = pd.unique(groups)
    group_codes = pd.Categorical(groups, categories=group_levels).codes.astype(np.int64)

    lognorm, counts, cube = stream_lognorm_subset(
        source,
        rows,
        keep_gene_mask,
        chunk=chunk,
        group_codes=group_codes,
        n_groups=len(group_levels),
        verbose=verbose,
    )

    adata = ad.AnnData(
        X=lognorm,
        layers={COUNTS_LAYER: counts},
        obs=obs.drop(columns=["cell_id"]),
        var=var.loc[keep_gene_mask].copy(),
    )
    adata.uns["ccc_build"] = {
        "resource": resource_name,
        "layer_semantics": f"X = {LAYER} (log1p CP10K, size factor over all {len(var)} genes)",
        "warning": "CCC-ONLY object: var is the LR resource subset. Not for DE/HVG/UMAP.",
    }
    sanitize_obs(adata, verbose=verbose)
    adata.write_h5ad(out, compression="gzip")
    if verbose:
        print(f"wrote {out}  {adata.shape}")

    cube_df = pd.DataFrame(cube, index=group_levels, columns=var.index).astype(np.int64)
    meta = _pseudobulk_meta(obs, group_levels)
    cube_df.to_parquet(pseudobulk_out)
    meta.to_parquet(pseudobulk_meta_out)
    if verbose:
        print(f"wrote pseudobulk cube {cube_df.shape} -> {pseudobulk_out}")

    if testset_out is not None:
        _write_testset(
            source, obs, var, testset_out, testset_donors=testset_donors,
            chunk=chunk, verbose=verbose,
        )

    manifest = {
        "source": str(source),
        "obs_parquet": str(obs_parquet),
        "compartment": compartment,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_genes_source": int(len(var)),
        "nnz_lognorm": int(lognorm.nnz),
        "nnz_counts": int(counts.nnz),
        "level_counts": adata.obs[GROUPBY].value_counts().to_dict(),
        "n_donors": int(adata.obs[DONOR_KEY].nunique()),
        "n_samples": int(adata.obs[SAMPLE_KEY].nunique()),
        "resource_coverage": {
            k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
            for k, v in coverage.items()
        },
        "liana_version": li.__version__,
        "pseudobulk_shape": list(cube_df.shape),
    }
    Path(manifest_out).write_text(json.dumps(manifest, indent=2, default=str))
    if verbose:
        print(f"wrote manifest -> {manifest_out}")
    return adata, manifest


def _pseudobulk_meta(obs, group_levels):
    """One row per (ccc_celltype, sample) pseudo-sample with design metadata."""
    design = [
        GROUPBY, SAMPLE_KEY, DONOR_KEY, STUDY_KEY, "disease", "entity",
        "stage_group", "stage_class", "skin_layer", "tissue", "sex",
        "treatment_context", "lesion_type",
    ]
    design = [c for c in design if c in obs.columns]
    tmp = obs.assign(_g=obs[GROUPBY].astype(str) + "||" + obs[SAMPLE_KEY].astype(str))
    n_cells = tmp.groupby("_g", observed=True).size().rename("n_cells")
    meta = (
        tmp.drop_duplicates("_g").set_index("_g")[design].join(n_cells).reindex(group_levels)
    )
    meta.index.name = "pseudo_sample"
    return meta


def _write_testset(source, obs, var, out, testset_donors=None, chunk=50_000, verbose=True):
    """Small FULL-GENE object for the nb34 equivalence gate.

    Its only purpose is to prove that restricting var to the resource genes does
    not change a single liana statistic. If this object disagrees with the subset
    object, the size factor was computed on the wrong gene set.
    """
    import anndata as ad

    if testset_donors is None:
        by_study = (
            obs[obs[GROUPBY].astype(str) == CD4_MALIGNANT]
            .groupby(STUDY_KEY, observed=True)[DONOR_KEY]
            .agg(lambda s: s.value_counts().idxmax())
        )
        wanted = ["li2024", "buus2025", "chennareddy2025"]
        testset_donors = [by_study[s] for s in wanted if s in by_study.index]
    sub_obs = obs[obs[DONOR_KEY].astype(str).isin([str(d) for d in testset_donors])]
    if verbose:
        print(f"testset donors {list(testset_donors)} -> {len(sub_obs)} cells")

    source_index = read_source_index(source)
    rows, cell_ids = resolve_rows(source_index, sub_obs["cell_id"].to_numpy())
    sub_obs = sub_obs.loc[cell_ids]

    all_genes = np.ones(len(var), dtype=bool)
    lognorm, counts, _ = stream_lognorm_subset(
        source, rows, all_genes, chunk=chunk, verbose=False
    )
    test = ad.AnnData(
        X=lognorm,
        layers={COUNTS_LAYER: counts},
        obs=sub_obs.drop(columns=["cell_id"]),
        var=var.copy(),
    )
    sanitize_obs(test, verbose=False)
    test.write_h5ad(out, compression="gzip")
    if verbose:
        print(f"wrote equivalence testset {test.shape} -> {out}")
    return test


def load_ccc_adata(path=CCC_ADATA, manifest=MANIFEST, groupby=GROUPBY, verbose=True):
    """Load the ~2 GB CCC object and assert the manifest invariants."""
    import scanpy as sc

    adata = sc.read_h5ad(path)
    if Path(manifest).exists():
        man = json.loads(Path(manifest).read_text())
        assert adata.n_obs == man["n_obs"], (adata.n_obs, man["n_obs"])
        assert adata.n_vars == man["n_vars"], (adata.n_vars, man["n_vars"])
        got = adata.obs[groupby].value_counts().to_dict()
        for lv, n in man["level_counts"].items():
            if n:
                assert got.get(lv, 0) == n, (lv, got.get(lv, 0), n)
    if verbose:
        print(adata)
        print(adata.obs[groupby].value_counts().to_string())
    return adata


def assert_ccc_invariants(adata, groupby=GROUPBY, counts_layer=COUNTS_LAYER):
    """Cheap structural checks; run before anything is inferred from the object."""
    cnt = adata.layers[counts_layer]
    head = cnt[:2000].data
    assert np.allclose(head, np.rint(head)), f"{counts_layer} is not integral"
    xs, cs = adata.X[:2000], cnt[:2000]
    assert xs.nnz == cs.nnz, "X and counts have different sparsity patterns"
    assert not np.allclose(xs.data, cs.data), "X looks like raw counts, not lognorm"
    assert adata.X.max() < 15, f"X max {adata.X.max()} is too large for log1p CP10K"
    assert adata.var_names.is_unique
    # A wildcard control (e.g. KRT1 -> *) asserts the pair is NOT in the resource, and var
    # here IS the resource gene set -- so its genes are absent by construction. Checking
    # them would contradict the reason the control exists.
    for genes in (POSITIVE_CONTROLS, NEGATIVE_CONTROLS):
        for _s, _t, lig, rec in genes:
            subs = [*lig.split("_"), *rec.split("_")]
            if "*" in subs:
                continue
            for sub in subs:
                assert sub in adata.var_names, f"control gene {sub} absent from var"
    print("invariants OK:", adata.shape)


# ================================================================ windows & audits


def focal_window(
    adata,
    disease=None,
    studies=None,
    donors=None,
    stage_group=None,
    skin_layer=None,
    entity=None,
    groupby=GROUPBY,
    verbose=True,
):
    """Subset to a context window and print its denominators.

    Every rate this pipeline reports is printed with the number of donors behind
    it, so a window is never used without knowing what it contains.
    """
    mask = pd.Series(True, index=adata.obs_names)
    for col, want in [
        ("disease", disease),
        (STUDY_KEY, studies),
        (DONOR_KEY, donors),
        ("stage_group", stage_group),
        ("skin_layer", skin_layer),
        ("entity", entity),
    ]:
        if want is not None:
            mask &= adata.obs[col].astype(str).isin([str(w) for w in want])

    sub = adata[mask.values].copy()
    if verbose:
        print(
            f"window: {sub.n_obs} cells, {sub.obs[DONOR_KEY].nunique()} donors, "
            f"{sub.obs[SAMPLE_KEY].nunique()} samples, {sub.obs[STUDY_KEY].nunique()} studies"
        )
        per = (
            sub.obs.groupby(sub.obs[groupby].astype(str), observed=True)[DONOR_KEY]
            .nunique()
            .sort_values(ascending=False)
        )
        print("donors per level:\n" + per.to_string())
    return sub


def prepare_analysis(adata, name, groupby=GROUPBY, analyses=None, axes=None, verbose=True):
    """Subset + groupby_pairs for one ANALYSES entry -> (sub, pairs, spec).

    `analyses` / `axes` default to ccc_data's, so nb35/36 are unaffected; nb38/39 pass
    ccc_data_sub's overrides instead of the module globals.
    """
    analyses = ANALYSES if analyses is None else analyses
    axes = FOCAL_AXES if axes is None else axes
    spec = analyses[name]
    senders, receivers = axes[spec["axis"]]
    drop = set(spec.get("drop_levels", []))
    senders = [s for s in senders if s not in drop]
    receivers = [r for r in receivers if r not in drop]

    sub = focal_window(
        adata,
        disease=spec.get("disease"),
        studies=spec.get("studies"),
        stage_group=spec.get("stage_group"),
        skin_layer=spec.get("skin_layer"),
        groupby=groupby,
        verbose=verbose,
    )
    pairs = build_groupby_pairs({spec["axis"]: (senders, receivers)})

    if verbose:
        print(f"\n{name}: {senders} <-> {receivers}")
        print(f"  {spec['note']}")
        present = set(sub.obs[groupby].astype(str))
        missing = [lv for lv in senders + receivers if lv not in present]
        if missing:
            print(f"  !! levels absent from this window: {missing}")
    return sub, pairs, spec


def subsample_levels(
    adata,
    groupby=GROUPBY,
    max_per_level=SUBSAMPLE_MAX_PER_LEVEL,
    max_per_donor_per_level=SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL,
    donor_key=DONOR_KEY,
    seed=SUBSAMPLE_SEED,
    verbose=True,
):
    """Donor-stratified cap, then a per-level cap.

    Two purposes, not one. Compute: 1000 permutations over ~400k cells is slow.
    Bias: liana's permutation p-value shrinks with group size, so an unbalanced
    CD4_malignant (76,349 cells vs Tregs' 1,936) would score p=0 on nearly every
    pair purely on n. The unbalanced run is the sensitivity check, not the default.
    """
    rng = np.random.default_rng(seed)
    obs = adata.obs
    keep = []
    report = []
    for lv, idx in obs.groupby(obs[groupby].astype(str), observed=True).groups.items():
        idx = pd.Index(idx)
        picked = []
        if max_per_donor_per_level:
            for _d, didx in obs.loc[idx].groupby(obs.loc[idx, donor_key].astype(str), observed=True).groups.items():
                didx = np.asarray(didx)
                if len(didx) > max_per_donor_per_level:
                    didx = rng.choice(didx, max_per_donor_per_level, replace=False)
                picked.append(didx)
            picked = np.concatenate(picked) if picked else np.array([], dtype=object)
        else:
            picked = np.asarray(idx)
        if max_per_level and len(picked) > max_per_level:
            picked = rng.choice(picked, max_per_level, replace=False)
        keep.append(picked)
        report.append({"level": lv, "n_before": len(idx), "n_after": len(picked)})

    keep = np.concatenate(keep)
    sub = adata[pd.Index(keep)].copy()
    rep = pd.DataFrame(report).sort_values("n_before", ascending=False)
    if verbose:
        print(f"subsampled {adata.n_obs} -> {sub.n_obs} cells")
        print(rep.to_string(index=False))
    return sub, rep


def cell_count_audit(adata_or_obs, groupby=GROUPBY, sample_key=None, min_cells=MIN_CELLS):
    """Cross-tab level x sample (or donor) and report what min_cells removes.

    Accepts a plain obs frame so nb34 can audit before any h5ad is opened.
    """
    sample_key = SAMPLE_KEY if sample_key is None else sample_key
    obs = adata_or_obs.obs if hasattr(adata_or_obs, "obs") else adata_or_obs
    counts = pd.crosstab(obs[groupby].astype(str), obs[sample_key].astype(str))
    passing = counts >= min_cells
    summary = pd.DataFrame(
        {
            "n_cells": counts.sum(axis=1),
            "n_units_present": (counts > 0).sum(axis=1),
            "n_units_passing": passing.sum(axis=1),
            "frac_units_passing": passing.sum(axis=1) / counts.shape[1],
        }
    ).sort_values("frac_units_passing")
    return counts, summary


def axis_feasibility(
    counts,
    axes=None,
    min_cells=MIN_CELLS,
    min_samples=MIN_SAMPLES,
    thin_levels=None,
    both_directions=True,
):
    """Per (axis, sender, receiver): units where BOTH sides clear min_cells.

    The verdict column is the gate the figures obey:
      claim          -- >= min_samples units on both sides, may carry a claim
      report_only    -- too few units, OR one side is a TME_THIN level: printed
                        so absence is on the record, never claimed
      not_computable -- no unit has both sides present
    """
    axes = FOCAL_AXES if axes is None else axes
    thin_levels = set(TME_THIN if thin_levels is None else thin_levels)
    passing = counts >= min_cells
    rows = []
    for axis, (senders, receivers) in axes.items():
        pairs = [(s, t) for s in senders for t in receivers]
        if both_directions:
            pairs += [(t, s) for s, t in pairs]
        for s, t in dict.fromkeys(pairs):
            if s not in passing.index or t not in passing.index:
                n = 0
            else:
                n = int((passing.loc[s] & passing.loc[t]).sum())
            if n == 0:
                verdict = "not_computable"
            elif n < min_samples or {s, t} & thin_levels:
                verdict = "report_only"
            else:
                verdict = "claim"
            rows.append({"axis": axis, "source": s, "target": t, "n_units": n, "verdict": verdict})
    return pd.DataFrame(rows).drop_duplicates(["source", "target"]).sort_values(
        "n_units", ascending=False
    )


def coverage_table(adata, groupby=GROUPBY, sample_key=None, min_cells=MIN_CELLS, levels=None):
    """Per level: cells, units present, units clearing min_cells.

    Printed under every headline figure. Every grey heatmap cell must line up
    with a zero here -- otherwise the grey is being read as biology.
    """
    sample_key = DONOR_KEY if sample_key is None else sample_key
    counts, summary = cell_count_audit(adata, groupby, sample_key, min_cells)
    if levels is not None:
        summary = summary.reindex([lv for lv in levels if lv in summary.index])
    summary = summary.rename(
        columns={"n_units_present": f"{sample_key}s_present",
                 "n_units_passing": f"{sample_key}s_ge_{min_cells}",
                 "frac_units_passing": "frac_passing"}
    )
    return summary


# ================================================================ liana runs


def build_groupby_pairs(axes, both_directions=True):
    """Expand {axis: (senders, receivers)} into a liana `groupby_pairs` frame."""
    rows = []
    for senders, receivers in axes.values():
        for s in senders:
            for t in receivers:
                rows.append((s, t))
                if both_directions:
                    rows.append((t, s))
    return pd.DataFrame(rows, columns=["source", "target"]).drop_duplicates()


def run_rank_aggregate(
    adata,
    resource,
    groupby=GROUPBY,
    groupby_pairs=None,
    expr_prop=EXPR_PROP,
    min_cells=MIN_CELLS,
    n_perms=N_PERMS,
    key_added="liana_res",
    layer=LAYER,
    n_jobs=N_JOBS,
    seed=SEED,
    verbose=True,
    **kwargs,
):
    """Consensus (RRA) LR inference on one context.

    Ranks and permutation p-values are DESCRIPTIVE ONLY: the unit is the cell and
    they are pseudoreplicated across donors. They size dots and mark cells; they
    are never the evidence for a claim.
    """
    li.mt.rank_aggregate(
        adata,
        groupby=groupby,
        resource=resource,
        groupby_pairs=groupby_pairs,
        expr_prop=expr_prop,
        min_cells=min_cells,
        use_raw=False,
        layer=layer,
        n_perms=n_perms,
        aggregate_method="rra",
        return_all_lrs=False,
        n_jobs=n_jobs,
        seed=seed,
        key_added=key_added,
        verbose=verbose,
        **kwargs,
    )
    return adata.uns[key_added]


def run_by_group(adata, resource, group_key, groups=None, min_cells=MIN_CELLS, **kwargs):
    """rank_aggregate per level of `group_key`; one long frame with denominators.

    Used here as the STUDY diagnostic: expr_prop is a detection-rate threshold and
    chemistry tracks study, so a pair that passes in one study and fails in
    another is a capture artifact rather than a biological difference.
    """
    groups = (
        list(groups)
        if groups is not None
        else list(pd.unique(adata.obs[group_key].astype(str).dropna()))
    )
    out = []
    for g in groups:
        sub = adata[adata.obs[group_key].astype(str) == g].copy()
        n_donors = sub.obs[DONOR_KEY].nunique()
        try:
            res = run_rank_aggregate(
                sub, resource, key_added=f"liana_{group_key}_{g}", min_cells=min_cells, **kwargs
            )
        except Exception as exc:                                    # noqa: BLE001
            print(f"{group_key}={g}: {exc}")
            continue
        out.append(res.assign(**{group_key: g, "n_donors": n_donors, "n_cells": sub.n_obs}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def expr_prop_sweep(adata, resource, props=EXPR_PROP_SWEEP, top_n=TOP_N, **kwargs):
    """Re-run at several expr_prop values and report rank stability.

    expr_prop is the single biggest lever on the result and the low-abundance
    cytokines (IL13, IL4, IL31, IL2) sit right at the threshold.
    """
    tops, frames = {}, []
    for prop in props:
        res = run_rank_aggregate(
            adata, resource, expr_prop=prop, key_added=f"liana_res_{prop}", **kwargs
        ).sort_values("magnitude_rank")
        frames.append(res.assign(expr_prop=prop))
        tops[prop] = set(map(tuple, res.head(top_n)[KEY_COLS].values))
    stable = set.intersection(*tops.values()) if tops else set()
    return pd.concat(frames, ignore_index=True), stable


def shuffle_control(
    adata,
    resource,
    groupby=GROUPBY,
    within=DONOR_KEY,
    n_shuffles=N_SHUFFLES,
    top_n=TOP_N,
    seed=SEED,
    reference=None,
    **kwargs,
):
    """Permute the grouping label WITHIN donor and re-run.

    Shuffling within donor rather than globally holds each donor's cell-type
    composition fixed, so what dissolves is cell-type specificity rather than
    donor identity. Target: <= 2/20 overlap with the real top-20.
    """
    verbose = kwargs.pop("verbose", True)  # popped so it doesn't collide with the hardcoded per-shuffle verbose=False
    if reference is None:
        reference = run_rank_aggregate(adata, resource, groupby=groupby, verbose=verbose, **kwargs)
    real_top = set(map(tuple, reference.sort_values("magnitude_rank").head(top_n)[KEY_COLS].values))

    rng = np.random.default_rng(seed)
    rows, frames = [], []
    for i in range(n_shuffles):
        shuf = adata.copy()
        lab = shuf.obs[groupby].astype(str).to_numpy().copy()
        for _d, idx in shuf.obs.groupby(shuf.obs[within].astype(str), observed=True).groups.items():
            pos = shuf.obs.index.get_indexer(pd.Index(idx))
            lab[pos] = rng.permutation(lab[pos])
        shuf.obs[groupby] = pd.Categorical(lab)
        res = run_rank_aggregate(
            shuf, resource, groupby=groupby, key_added=f"liana_shuffle_{i}", verbose=False, **kwargs
        ).sort_values("magnitude_rank")
        frames.append(res.assign(shuffle=i))
        top = set(map(tuple, res.head(top_n)[KEY_COLS].values))
        rows.append({"shuffle": i, "overlap_with_real": len(top & real_top), "top_n": top_n})

    overlap = pd.DataFrame(rows)
    print(overlap.to_string(index=False))
    return pd.concat(frames, ignore_index=True), overlap


def downsample_stability(
    adata, resource, groupby=GROUPBY, sizes=None, top_n=TOP_N, seed=SEED, **kwargs
):
    """Top-n rank stability as the balanced level size grows.

    Makes the n-dependence of the permutation p-value explicit instead of
    leaving it as an unstated property of the default subsample.
    """
    sizes = DOWNSAMPLE_SIZES if sizes is None else sizes
    tops, rows = {}, []
    for n in sizes:
        sub, _ = subsample_levels(
            adata, groupby=groupby, max_per_level=n, max_per_donor_per_level=None,
            seed=seed, verbose=False,
        )
        res = run_rank_aggregate(
            sub, resource, groupby=groupby, key_added=f"liana_ds_{n}", verbose=False, **kwargs
        ).sort_values("magnitude_rank")
        tops[n] = set(map(tuple, res.head(top_n)[KEY_COLS].values))
        rows.append(
            {
                "max_per_level": n,
                "n_cells": sub.n_obs,
                "median_cellphone_pval": float(res["cellphone_pvals"].median()),
                "frac_pval_zero": float((res["cellphone_pvals"] == 0).mean()),
            }
        )
    stab = pd.DataFrame(rows)
    largest = tops[sizes[-1]]
    stab["overlap_with_largest"] = [len(tops[n] & largest) for n in sizes]
    print(stab.to_string(index=False))
    return stab, tops


# ================================================================ panel


def lr_panel_frame(panel=LR_PANEL):
    """Flatten the curated panel; complexes joined by '_' to match liana."""
    rows = []
    for group, pairs in panel.items():
        for ligand, receptor in pairs:
            rows.append(
                {
                    "group": group,
                    "ligand_complex": ligand if isinstance(ligand, str) else "_".join(ligand),
                    "receptor_complex": receptor
                    if isinstance(receptor, str)
                    else "_".join(receptor),
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def filter_to_panel(liana_res, panel=LR_PANEL, resource=None):
    """Inner-join a liana result to the curated panel, keeping the group labels.

    Joins on the RESOLVED spelling (see resolve_pairs) -- joining on the
    human-readable spelling silently drops CSF1->CSF1R, IL7->IL7R and every other
    pair the resource stores as a complex.
    """
    frame = resolve_panel(panel, resource)
    frame = frame[frame["orientation"] != "absent"]
    return liana_res.merge(
        frame[["group", "ligand_in", "receptor_in", "ligand_complex", "receptor_complex", "orientation"]],
        on=["ligand_complex", "receptor_complex"],
        how="inner",
    )


def forced_panel_expression(
    adata, panel=LR_PANEL, groupby=GROUPBY, layer=LAYER, min_cells=MIN_CELLS, resource=None
):
    """Per (level, gene): detection proportion and mean lognorm, computed directly.

    So a curated pair that failed expr_prop is reported as "ligand detected in 6%
    of the sender" rather than as absent. A negative that is really a coverage
    limit must not be reported as biology. Resolved subunits are included, so a
    pair that fails only because of an obligate extra chain (IL2RG on IL7->IL7R)
    is diagnosable rather than mysterious.
    """
    frame = resolve_panel(panel, resource)
    cols = ["ligand_in", "receptor_in", "ligand_complex", "receptor_complex"]
    genes = sorted(
        {
            sub
            for entry in pd.concat([frame[c] for c in cols]).dropna()
            for sub in str(entry).split("_")
            if sub != "*"
        }
    )
    genes = [g for g in genes if g in adata.var_names]

    X = adata[:, genes].layers[layer] if layer in adata.layers else adata[:, genes].X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    lab = adata.obs[groupby].astype(str).to_numpy()

    rows = []
    for lv in pd.unique(lab):
        idx = np.nonzero(lab == lv)[0]
        if len(idx) == 0:
            continue
        sub = X[idx]
        prop = np.asarray((sub > 0).sum(axis=0)).ravel() / len(idx)
        mean = np.asarray(sub.mean(axis=0)).ravel()
        rows.append(
            pd.DataFrame(
                {
                    "level": lv,
                    "gene": genes,
                    "n_cells": len(idx),
                    "expr_prop": prop,
                    "mean_lognorm": mean,
                    "passes_min_cells": len(idx) >= min_cells,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ================================================================ controls


def _matches(row, sender, receiver, ligand, receptor):
    return (
        row["source"] == sender
        and row["target"] == receiver
        and (ligand == "*" or row["ligand_complex"] == ligand)
        and (receptor == "*" or row["receptor_complex"] == receptor)
    )


def control_report(
    liana_res,
    positives=None,
    negatives=None,
    caveats=None,
    top_n=40,
    orderby="magnitude_rank",
    resource=None,
):
    """One frame per control pair: found, rank, magnitude, p, and top-n membership.

    Pass `resource` to resolve the controls onto its spelling first -- without
    that, CSF1->CSF1R and IL7->IL7R report as absent because the resource stores
    them as CSF1_IL34->CSF1R and IL7->IL2RG_IL7R.

    Also flags any LR_CAVEATS gene sitting in the top 10: that is the MIF->CD74
    failure mode, recorded rather than quietly excluded.
    """
    positives = POSITIVE_CONTROLS if positives is None else positives
    negatives = NEGATIVE_CONTROLS if negatives is None else negatives
    caveats = LR_CAVEATS if caveats is None else caveats
    if resource is not None:
        positives = resolve_controls(positives, resource)
        negatives = resolve_controls(negatives, resource)

    res = liana_res.sort_values(orderby).reset_index(drop=True)
    res["_rank_pos"] = np.arange(1, len(res) + 1)

    rows = []
    for kind, controls in (("positive", positives), ("negative", negatives)):
        for sender, receiver, ligand, receptor in controls:
            hit = res[res.apply(_matches, axis=1, args=(sender, receiver, ligand, receptor))]
            row = {
                "kind": kind,
                "source": sender,
                "target": receiver,
                "ligand_complex": ligand,
                "receptor_complex": receptor,
                "found": len(hit) > 0,
            }
            if len(hit):
                best = hit.iloc[0]
                row.update(
                    rank_position=int(best["_rank_pos"]),
                    magnitude_rank=float(best["magnitude_rank"]),
                    specificity_rank=float(best.get("specificity_rank", np.nan)),
                    lr_means=float(best.get("lr_means", np.nan)),
                    cellphone_pvals=float(best.get("cellphone_pvals", np.nan)),
                    in_top_n=int(best["_rank_pos"]) <= top_n,
                    pct_of_tested=int(best["_rank_pos"]) / len(res),
                )
            rows.append(row)
    report = pd.DataFrame(rows)

    flagged = []
    for r in res.head(10).itertuples(index=False):
        subs = {*str(r.ligand_complex).split("_"), *str(r.receptor_complex).split("_")}
        for g in subs & set(caveats):
            flagged.append(
                {
                    "gene": g,
                    "interaction": f"{r.ligand_complex} -> {r.receptor_complex}",
                    "source": r.source,
                    "target": r.target,
                    "caveat": caveats[g],
                }
            )
    caveat_hits = pd.DataFrame(flagged)

    n_pos = int(report.query("kind == 'positive'")["found"].sum())
    n_pos_top = int(report.query("kind == 'positive'").get("in_top_n", pd.Series(dtype=bool)).sum())
    n_neg = int(report.query("kind == 'negative'")["found"].sum())
    print(
        f"positives found {n_pos}/{len(positives)} ({n_pos_top} in top {top_n}); "
        f"negatives found {n_neg}/{len(negatives)} (want 0 or bottom-decile)"
    )
    if len(caveat_hits):
        print("\ncaveat genes in the top 10 -- documented failure mode, not biology:")
        print(caveat_hits[["gene", "interaction", "source", "target"]].to_string(index=False))
    return report, caveat_hits


def rank_delta(res_a, res_b, label_a="malignant", label_b="reactive", orderby="magnitude_rank"):
    """magnitude_rank with sender A minus with sender B, per (partner, L, R).

    The comparator table. No claim of the form "malignant CD4 signals X to Y" is
    reportable without it: a pair that ranks identically from reactive CD4 is a
    CD4 property, not a malignancy property.
    """
    keys = ["target", "ligand_complex", "receptor_complex"]
    a = res_a[keys + [orderby, "lr_means"]].rename(
        columns={orderby: f"{orderby}_{label_a}", "lr_means": f"lr_means_{label_a}"}
    )
    b = res_b[keys + [orderby, "lr_means"]].rename(
        columns={orderby: f"{orderby}_{label_b}", "lr_means": f"lr_means_{label_b}"}
    )
    out = a.merge(b, on=keys, how="outer")
    out["delta_rank"] = out[f"{orderby}_{label_a}"] - out[f"{orderby}_{label_b}"]
    out["delta_lr_means"] = out[f"lr_means_{label_a}"] - out[f"lr_means_{label_b}"]
    out[f"{label_a}_only"] = out[f"{orderby}_{label_b}"].isna()
    return out.sort_values("delta_rank")


def top_n_overlap_matrix(named_results, top_n=TOP_N, orderby="magnitude_rank"):
    """Symmetric top-n overlap counts across alternative runs (definitions, studies)."""
    tops = {
        name: set(map(tuple, res.sort_values(orderby).head(top_n)[KEY_COLS].values))
        for name, res in named_results.items()
    }
    names = list(tops)
    mat = pd.DataFrame(index=names, columns=names, dtype=int)
    for a in names:
        for b in names:
            mat.loc[a, b] = len(tops[a] & tops[b])
    return mat


# ================================================================ figures


def dotplot_axis(
    liana_res=None,
    adata=None,
    uns_key="liana_res",
    source_labels=None,
    target_labels=None,
    top_n=DOTPLOT_TOP_N,
    colour="lr_means",
    size="cellphone_pvals",
    inverse_size=True,
    orderby="magnitude_rank",
    orderby_ascending=True,
    figure_size=DOTPLOT_SIZE,
    title=None,
):
    """li.pl.dotplot with the MF defaults and an explicit title.

    Panels are the sender and x is the receiver, so the title always says so --
    reading a liana dotplot with the direction reversed is the easiest way to
    report an interaction backwards.
    """
    import plotnine as p9

    fig = li.pl.dotplot(
        adata=adata,
        liana_res=liana_res,
        uns_key=uns_key,
        colour=colour,
        size=size,
        source_labels=source_labels,
        target_labels=target_labels,
        inverse_size=inverse_size,
        orderby=orderby,
        orderby_ascending=orderby_ascending,
        top_n=top_n,
        figure_size=figure_size,
    )
    return fig + p9.labs(
        size="-log10(cellphone p)",
        colour="LR mean expr",
        title=(title or "") + "  | panel = sender, x = receiver",
    )


def partner_heatmap(
    res,
    sender=CD4_MALIGNANT,
    partners=None,
    top_n=HEATMAP_TOP_N,
    figsize=(11, 5),
    title=None,
    value_col="lr_means",
    pval_col="cellphone_pvals",
    order_col="magnitude_rank",
    alpha=PVAL_ALPHA,
):
    """Top interactions x partner cell type, one panel per direction.

    The pooled-design analogue of the Myeloma nb05 `time_heatmap`, with partner
    on the x axis instead of day. Rendering contract, kept deliberately:
      * colour = lr_means on viridis, so a magnitude reads as a magnitude;
      * cmap.set_bad(grey) so a structurally-absent (pair, partner) cell is grey
        rather than reading as a low value -- missing data is not weak signal;
      * shared vmin/vmax across BOTH panels, so left and right are comparable;
      * '*' where the per-cell permutation p < alpha (pseudoreplicated -- it
        marks cells, it is not evidence);
      * row order = min magnitude_rank across partners, so a pair that is strong
        with any single partner still makes the panel.
    Returns the Figure.
    """
    import matplotlib.pyplot as plt

    r = res.copy()
    r["interaction"] = r["ligand_complex"] + " -> " + r["receptor_complex"]
    if partners is None:
        partners = sorted(set(r.loc[r["source"] == sender, "target"]) |
                          set(r.loc[r["target"] == sender, "source"]))
    partners = [p for p in partners if p != sender]

    out = r[
        ((r["source"] == sender) & (r["target"].isin(partners)))
        | ((r["target"] == sender) & (r["source"].isin(partners)))
    ]
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(HEATMAP_BAD)
    vmin, vmax = (
        (out[value_col].min(), out[value_col].max()) if len(out) else (0.0, 1.0)
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im = None
    for ax, direction in zip(axes, ["out", "in"]):
        d = (
            out[out["source"] == sender].assign(partner=lambda x: x["target"])
            if direction == "out"
            else out[out["target"] == sender].assign(partner=lambda x: x["source"])
        )
        arrow = f"{sender} -> partner" if direction == "out" else f"partner -> {sender}"
        if not len(d):
            ax.set_axis_off()
            ax.set_title(f"{arrow}  (no interactions)", fontsize=9)
            continue
        order = (
            d.groupby("interaction")[order_col].min().sort_values().head(top_n).index
        )
        mat = d.pivot_table(index="interaction", columns="partner", values=value_col).reindex(
            index=order, columns=partners
        )
        sig = d.pivot_table(index="interaction", columns="partner", values=pval_col).reindex(
            index=order, columns=partners
        )
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(partners)))
        ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_title(arrow, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if pd.notna(sig.values[i, j]) and sig.values[i, j] < alpha:
                    ax.text(j, i, "*", ha="center", va="center", color="white", fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes, label=value_col, fraction=0.03)
    fig.suptitle(
        title or f"{sender} <-> TME  ({value_col}; * = permutation p<{alpha}; grey = below min_cells)",
        fontsize=10,
    )
    return fig
