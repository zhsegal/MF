"""Everything `40_ccc_subtype.ipynb` calls. No notebook dependency; imports standalone.

Replaces `ccc_helpers.py` (1,614 l) for the merged CCC pipeline. What survived was ported
because it already did the right thing (resource-name resolution, the windowing/coverage
audits, the labelling, the heatmap rendering contract). What did not survive was either
superseded by the donor-level layer (`rank_delta`, `downsample_stability`, `run_by_group`,
`expr_prop_sweep`) or belonged to the one-off build (`build_ccc_object`, the streaming pass).

THE TWO SCORES THIS MODULE REFUSES TO PRODUCE
---------------------------------------------
`magnitude_rank` and `cellphone_pvals`. Every LIANA magnitude score is monotone in absolute
expression, and the permutation null has SD sigma/sqrt(n_cells), so the p measures cells per
level rather than evidence. `run_spec_consensus` drops both columns before returning and
asserts they are gone, so no downstream sort or gate can reach them. Ranking is on
`specificity_rank` (RRA over NATMI spec_weight, Connectome scaled_weight, log2FC lr_logfc);
evidence is the donor-level layer in sections 2-4.

ONE ROSTER, ONE WINDOW, ONE RUN
-------------------------------
`liana_pipe` derives `groupby_subset` from `groupby_pairs` and physically subsets the object;
`min_cells` drops more. The null, `mat_mean`, the Connectome z-scores, the logFC and the NATMI
sums are then all computed on that residual pool, so scores from two different grids are not
comparable. Build one `groupby_pairs` over the whole roster and run once per window. Order is
label -> `focal_window` -> `subsample_common_n` -> run.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as st

import liana as li
from liana.method import AggregateClass, aggregate_meta, connectome, logfc, natmi

import ccc_data_sub as cfg

# ---------------------------------------------------------------- constants
# The config modules are frozen (they are the audited vocabulary); anything this rewrite adds
# lives here.
KEY_COLS = ["source", "target", "ligand_complex", "receptor_complex"]
RANK_COL = "specificity_rank"          # the only score anything sorts or gates on
ABUNDANCE_COL = "expr_prod"            # reported, never ranked (see module docstring)

PSEUDOBULK_SUB = cfg.CCC_DIR / "ccc_skin_pseudobulk_sub.parquet"
PSEUDOBULK_SUB_META = cfg.CCC_DIR / "ccc_skin_pseudobulk_sub_meta.parquet"
TAB_DIR = cfg.TAB_DIR
FIG_DIR = cfg.FINAL_FIG_DIR
RUN_LOG = TAB_DIR / "ccc40_run_log.md"
TABLE_PREFIX = "ccc40_"
FIG_PREFIX = "ccc40_"

DROP_FROM_RUN = ["pDC"]        # claimable but 1,111 cells: it alone sets the common n
COMMON_N_CAP = 3_000
MIN_DONOR_FRAC = 0.5           # gate 2
MIN_STUDIES = 2                # gate 2
TOP_DECILE = 0.10              # gate 5
FDR_ALPHA = 0.05
MIN_DONORS_REPRO = 5
MIN_DONORS_AMBIENT = 8

TEAL = "#0F766E"
GREY_BAD = "0.92"

# Curated biological grouping of the named edges, used only by `plot_dotplot_programs`.
# Keys are "<ligand_complex>-<receptor_complex>" exactly as they appear in the run frame;
# complex subunits keep their '_' ("CSF2-CSF2RA_CSF2RB"), and ligands that carry a '-'
# themselves ("HLA-A") are matched on the full key, never by splitting on the dash.
CCC_PROGRAMS = {
    "tls_b_cell_recruitment": [
        "CXCL13-CXCR5",
        "CD70-TNFRSF13B",
        "CD22-PTPRC",
        "CD72-CD5",
    ],
    "myeloid_dc_reprogramming": [
        "IL22-IL22RA2",
        "IL22-IL10RA",
        "CSF2-CSF1R",
        "CSF2-CSF2RA_CSF2RB",
        "XCL1-XCR1",
        "SPN-SIGLEC1",
    ],
    "immunosuppressive_macrophage_support": [
        "SPP1-PTGER4",
    ],
    "dc_skin_homing": [
        "CCL17-CCR4",
        "CCL22-CCR4",
    ],
    "fibroblast_stromal_retention": [
        "CXCL12-CCR4",
        "CXCL12-CXCR3",
    ],
    "broad_chemokine_network": [
        "CXCL13-CXCR3",
        "CXCL13-ADRA2A",
        "CXCL13-ACKR4",
        "CCL5-CXCR3",
        "CCL5-CCR4",
        "CCL13-CXCR3",
        "CCL19-CXCR3",
        "CCL20-CXCR3",
        "CXCL9-CXCR3",
    ],
    "canonical_state_regulation": [
        "CD80-CTLA4",
        "CD86-CTLA4",
    ],
    "direct_immune_modulation": [
        "CD70-CD27",
        "HLA-A-CD8A",
        "HLA-E-CD8A",
        "LCK-CD8A_CD8B",
    ],
    "reciprocal_epidermotropism": [
        "JAML-CXADR",
        "CXADR-JAML",
    ],
    "structural_and_scaffolding": [
        "ICAM3-ITGAL",
        "CD48-CD2",
        "ALCAM-CD6",
        "C3-IFITM1",
    ],
}

# NATMI + Connectome + log2FC. All three carry permute=False, so no permutation is ever
# computed however n_perms is set. n_perms must NOT be None: liana_pipe overwrites
# consensus_opts with 'Magnitude' when it is, and the frame comes back with magnitude_rank
# as the only aggregate -- the exact column this pipeline refuses to use.
SPEC_CONSENSUS = AggregateClass(aggregate_meta, methods=[natmi, connectome, logfc])


def tab(name: str) -> Path:
    """tables/ccc40_<name>.csv."""
    return TAB_DIR / f"{TABLE_PREFIX}{name}.csv"


def fig_path(name: str) -> Path:
    """figures/final/ccc40_<name>.svg."""
    return FIG_DIR / f"{FIG_PREFIX}{name}.svg"


# ================================================================ resource


def resource_genes(resource) -> list[str]:
    """Sorted unique subunit symbols; complexes split on '_'."""
    genes: set[str] = set()
    for col in ("ligand", "receptor"):
        for entry in resource[col].astype(str):
            genes.update(entry.split("_"))
    return sorted(genes)


def _complex_ok(entry, present) -> bool:
    return all(sub in present for sub in str(entry).split("_"))


def load_resource(resource_name=cfg.RESOURCE_NAME, var_names=None, verbose=True):
    """(resource, coverage dict) for the human consensus resource; no ortholog mapping."""
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
            n_interactions_covered=int(ok.sum()),
            frac_interactions_covered=float(ok.mean()),
            missing_checks=[g for g in cfg.RESOURCE_COVERAGE_CHECK if g not in present],
        )
    if verbose:
        print(f"{resource_name}: {coverage['n_interactions']} interactions, "
              f"{coverage['n_subunit_genes']} subunit genes")
    return resource, coverage


def _subunits(entry) -> frozenset:
    return frozenset(str(entry).split("_"))


def resolve_pairs(pairs, resource, verbose=False):
    """Map readable (ligand, receptor) pairs onto the resource's spelling; one row per match.

    Assumes nothing about how a pair is written. The consensus resource sorts complex subunits
    alphabetically, stores some ligands as complexes (CSF1 -> CSF1R is CSF1_IL34 -> CSF1R),
    adds obligate chains (IL7 -> IL2RG_IL7R) and stores a few pairs with the receptor in the
    ligand column (TIGIT -> PVR). A curated side matches when its subunits are a SUBSET of the
    resource side's. `orientation == "flipped"` means the match needed a swap and any caller
    carrying a direction must swap source/target with it. No match -> one row, n_matches == 0,
    so a resource gap is visible instead of silent.
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
            rows.append({"ligand_in": lig_in, "receptor_in": rec_in, "ligand_complex": None,
                         "receptor_complex": None, "orientation": "absent", "n_matches": 0})
            continue
        for orientation, i in hits:
            rows.append({"ligand_in": lig_in, "receptor_in": rec_in,
                         "ligand_complex": resource.at[i, "ligand"],
                         "receptor_complex": resource.at[i, "receptor"],
                         "orientation": orientation, "n_matches": len(hits)})
    out = pd.DataFrame(rows).drop_duplicates()
    if verbose:
        absent = out[out["orientation"] == "absent"]
        print(f"resolved {out['ligand_in'].nunique() - len(absent)}/{out['ligand_in'].nunique()} pairs")
    return out


def lr_panel_frame(panel=None) -> pd.DataFrame:
    """Flatten the curated LR panel; complexes joined by '_' to match liana."""
    panel = cfg.LR_PANEL if panel is None else panel
    rows = []
    for group, pairs in panel.items():
        for ligand, receptor in pairs:
            rows.append({
                "group": group,
                "ligand_complex": ligand if isinstance(ligand, str) else "_".join(ligand),
                "receptor_complex": receptor if isinstance(receptor, str) else "_".join(receptor),
            })
    return pd.DataFrame(rows).drop_duplicates()


def resolve_panel(panel=None, resource=None, verbose=False) -> pd.DataFrame:
    """`lr_panel_frame` carrying the resource's own complex spelling and orientation."""
    frame = lr_panel_frame(panel)
    if resource is None:
        resource, _ = load_resource(verbose=False)
    resolved = resolve_pairs(list(zip(frame["ligand_complex"], frame["receptor_complex"])),
                             resource, verbose=verbose)
    return frame.rename(columns={"ligand_complex": "ligand_in",
                                 "receptor_complex": "receptor_in"}).merge(
        resolved, on=["ligand_in", "receptor_in"], how="left")


def resolve_controls(controls, resource, verbose=False) -> list[tuple]:
    """Resolve (source, target, ligand, receptor) controls; source/target swap when flipped."""
    resolved = resolve_pairs([(lg, rc) for _s, _t, lg, rc in controls], resource, verbose=verbose)
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


def panel_coverage(resource, panel=None, var_names=None) -> pd.DataFrame:
    """Per curated pair: in the RESOURCE, and are its genes in the DATA -- separate columns.

    A pair absent from the resource can never be scored however well expressed; a pair absent
    from the data is a real negative. Conflating them turns a bookkeeping gap into biology.
    """
    frame = resolve_panel(panel, resource)
    present = set(map(str, var_names)) if var_names is not None else None
    frame = frame.assign(in_resource=frame["orientation"] != "absent")
    if present is not None:
        def missing_for(row):
            side = (row["ligand_complex"] or row["ligand_in"],
                    row["receptor_complex"] or row["receptor_in"])
            subs = [s for entry in side for s in str(entry).split("_") if s != "*"]
            return ",".join(s for s in subs if s not in present)
        frame["genes_missing_from_data"] = frame.apply(missing_for, axis=1)
        frame["in_data"] = frame["genes_missing_from_data"] == ""
    keep = ["group", "ligand_in", "receptor_in", "ligand_complex", "receptor_complex",
            "orientation", "in_resource", "in_data", "genes_missing_from_data"]
    return frame[[c for c in keep if c in frame.columns]]


# ================================================================ object & labels


def load_ccc_adata(path=cfg.CCC_ADATA, groupby=None, verbose=True):
    """The ~2 GB CCC object with `layers['lognorm']` attached. Assumes the nb34 build ran."""
    import scanpy as sc

    groupby = cfg._v1.GROUPBY if groupby is None else groupby
    adata = sc.read_h5ad(path)
    adata.layers[cfg.LAYER] = adata.X
    if verbose:
        print(f"{adata.n_obs:,} cells x {adata.n_vars:,} resource genes")
    return adata


def assert_ccc_invariants(adata, groupby=None, counts_layer=cfg.COUNTS_LAYER) -> None:
    """Raise unless counts are integral, X is lognorm and every control gene is in var."""
    cnt = adata.layers[counts_layer]
    head = cnt[:2000].data
    assert np.allclose(head, np.rint(head)), f"{counts_layer} is not integral"
    xs, cs = adata.X[:2000], cnt[:2000]
    assert xs.nnz == cs.nnz, "X and counts have different sparsity patterns"
    assert not np.allclose(xs.data, cs.data), "X looks like raw counts, not lognorm"
    assert adata.X.max() < 15, f"X max {adata.X.max()} too large for log1p CP10K"
    assert adata.var_names.is_unique
    for controls in (cfg.POSITIVE_CONTROLS, cfg.NEGATIVE_CONTROLS):
        for _s, _t, lig, rec in controls:
            subs = [*lig.split("_"), *rec.split("_")]
            if "*" in subs:            # a wildcard control asserts ABSENCE from the resource
                continue
            for sub in subs:
                assert sub in adata.var_names, f"control gene {sub} absent from var"
    print(f"invariants OK: {adata.shape}")


def _malig_flag(obs, col):
    raw = obs[col]
    is_null = raw.isna()
    as_bool = raw.fillna(False).astype(bool)
    return as_bool & ~is_null, ~as_bool & ~is_null, is_null


def build_ccc_celltype(obs, key=None, verbose=True) -> pd.Series:
    """cell_type_final + the ALICE call -> one grouping column; NaN where the cell is dropped.

    Only CD4 splits (ALICE is a CD4-only caller). A CD4 that came back False purely because no
    TRB CDR3 was recovered is demoted to CD4_unassessed: absence of data, not evidence of
    benignity. Assumes obs carries `cell_type_final`, `mal_tcr_alice`, `assessed_tcr_nb30`.
    """
    key = cfg._v1.GROUPBY if key is None else key
    ct = obs[cfg.CELLTYPE_SRC].astype(str)
    label = ct.copy()
    is_cd4 = ct == "CD4"
    mal_true, mal_false, mal_null = _malig_flag(obs, cfg.MALIG_SRC)

    untestable = ~obs[cfg.TCR_ASSESSED_SRC].fillna(False).astype(bool)
    demoted = mal_false & untestable
    mal_false = mal_false & ~untestable
    mal_null = mal_null | demoted
    if verbose:
        print(f"{int((is_cd4 & demoted).sum()):,} CD4 negative for {cfg.MALIG_SRC} with no "
              f"{cfg.TCR_ASSESSED_SRC} -> CD4_unassessed (untestable, not benign)")

    label = label.mask(is_cd4 & mal_true, cfg.CD4_MALIGNANT)
    label = label.mask(is_cd4 & mal_false, cfg.CD4_REACTIVE)
    label = label.mask(is_cd4 & mal_null, cfg.CD4_UNASSESSED)
    label = label.replace({lv: np.nan for lv in cfg._v1.DROP_LEVELS})
    label = label.where(label.notna() & (label != "nan"), np.nan)
    return pd.Series(label, index=obs.index, name=key)


def build_ccc_celltype_sub(obs, subtype_csv=None, keep_levels=None, key=None,
                           verbose=True) -> pd.Series:
    """`build_ccc_celltype`, then Myeloid/Fibroblast replaced by the nb10c `subtype_ccc`.

    Assumes obs carries a `cell_id` column matching the sidecar. Cells with a null subtype
    (UNK / proliferating / pericyte contamination) and anything outside `keep_levels` become
    NaN for the caller to drop. Returns a categorical Series in roster order.
    """
    subtype_csv = cfg.SUBTYPE_CSV if subtype_csv is None else subtype_csv
    keep_levels = cfg.KEEP_LEVELS if keep_levels is None else keep_levels
    key = cfg.GROUPBY if key is None else key

    base = build_ccc_celltype(obs, key=key, verbose=verbose)
    side = pd.read_csv(subtype_csv, dtype=str)
    side = side[side["subtype_ccc"].notna() & ~side["subtype_ccc"].isin(["nan", "None", "NA", ""])]
    fine = side.set_index("cell_id")["subtype_ccc"]

    ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else obs.index.astype(str)
    is_split = obs[cfg.CELLTYPE_SRC].astype(str).isin(cfg.SPLIT_LINEAGES)
    mapped = pd.Series(fine.reindex(pd.Index(ids)).to_numpy(), index=obs.index)

    label = base.astype(object).mask(is_split, mapped)
    label = label.where(label.isin(list(keep_levels)), np.nan)
    order = [lv for lv in keep_levels if lv in set(label.dropna())]
    out = pd.Series(pd.Categorical(label, categories=order), index=obs.index, name=key)
    if verbose:
        n_drop = int(out.isna().sum())
        print(f"dropping {n_drop:,} cells ({n_drop / len(obs):.1%}) outside the roster")
        print(out.value_counts().to_string())
    return out


def assert_roster(adata, sidecar_path=None, pooled_key=None, sub_key=None) -> pd.DataFrame:
    """Raise unless the sidecar matches the config and the untouched levels carry over exactly.

    Three failures this catches before a single liana run: a stale MYELOID_LEVELS/FIBRO_LEVELS
    in ccc_data_sub.py, a pooled Myeloid/Fibroblast cell that vanished without a stated reason,
    and any drift in the five levels that make the B/CD8 axes a regression test.
    Returns the pooled -> sub reconciliation frame.
    """
    sidecar_path = cfg.SUBTYPE_CSV if sidecar_path is None else sidecar_path
    pooled_key = cfg._v1.GROUPBY if pooled_key is None else pooled_key
    sub_key = cfg.GROUPBY if sub_key is None else sub_key
    assert Path(sidecar_path).exists(), (
        f"missing {sidecar_path} -- run 10c_skin_myeloid_fibro_reannotation.ipynb (section 9)")

    side = pd.read_csv(sidecar_path, dtype=str)
    got_m = sorted(set(side.loc[side.lineage == "Myeloid", "subtype_ccc"].dropna()))
    got_f = sorted(set(side.loc[side.lineage == "Fibroblast", "subtype_ccc"].dropna()))
    assert got_m == sorted(cfg.MYELOID_LEVELS), (
        f"MYELOID_LEVELS is stale.\n  sidecar: {got_m}\n  config: {sorted(cfg.MYELOID_LEVELS)}")
    assert got_f == sorted(cfg.FIBRO_LEVELS), (
        f"FIBRO_LEVELS is stale.\n  sidecar: {got_f}\n  config: {sorted(cfg.FIBRO_LEVELS)}")

    pooled = adata.obs[pooled_key].astype(str)
    sub_lab = adata.obs[sub_key].astype(str)
    rows = []
    for lineage, levels in [("Myeloid", cfg.MYELOID_LEVELS), ("Fibroblast", cfg.FIBRO_LEVELS)]:
        m = pooled == lineage
        n_lab = int(sub_lab[m].isin(levels).sum())
        n_side = int(side[(side.lineage == lineage) & side.subtype_ccc.notna()].shape[0])
        assert n_lab == n_side, (lineage, n_lab, n_side)
        rows.append({"level": lineage, "n_pooled": int(m.sum()), "n_sub_labelled": n_lab,
                     "n_dropped_unk_prolif": int(m.sum()) - n_lab})
    for lv in ["CD8", "B", "Keratinocyte", cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE]:
        n_pool, n_sub = int((pooled == lv).sum()), int((sub_lab == lv).sum())
        assert n_pool == n_sub, (lv, n_pool, n_sub)
        rows.append({"level": lv, "n_pooled": n_pool, "n_sub_labelled": n_sub,
                     "n_dropped_unk_prolif": 0})
    return pd.DataFrame(rows)


# ================================================================ windows & coverage


def focal_window(adata, disease=None, studies=None, donors=None, groupby=None, verbose=True):
    """Subset to a context window and print its denominators. Returns a copy."""
    groupby = cfg.GROUPBY if groupby is None else groupby
    mask = pd.Series(True, index=adata.obs_names)
    for col, want in [("disease", disease), (cfg.STUDY_KEY, studies), (cfg.DONOR_KEY, donors)]:
        if want is not None:
            mask &= adata.obs[col].astype(str).isin([str(w) for w in want])
    sub = adata[mask.values].copy()
    if verbose:
        print(f"window: {sub.n_obs:,} cells, {sub.obs[cfg.DONOR_KEY].nunique()} donors, "
              f"{sub.obs[cfg.STUDY_KEY].nunique()} studies")
    return sub


def coverage_table(adata, groupby=None, sample_key=None, min_cells=None) -> pd.DataFrame:
    """(level x donor) count matrix. Every grey heatmap cell must line up with a zero here."""
    groupby = cfg.GROUPBY if groupby is None else groupby
    sample_key = cfg.DONOR_KEY if sample_key is None else sample_key
    obs = adata.obs if hasattr(adata, "obs") else adata
    return pd.crosstab(obs[groupby].astype(str), obs[sample_key].astype(str))


def claim_gate(adata, counts=None, groupby=None, min_cells=None, min_samples=None,
               levels=None) -> pd.DataFrame:
    """Per level: cells, donors >= min_cells, studies, top-study fraction, claimable.

    The study-dominance flag TRAVELS WITH THE ROW; it never gates. Assumes `adata` is already
    restricted to the window the runs use -- claimability computed on the whole object would
    count HC donors that no CTCL run ever sees.
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    min_samples = cfg.MIN_SAMPLES if min_samples is None else min_samples
    counts = coverage_table(adata, groupby=groupby, min_cells=min_cells) if counts is None else counts

    obs = adata.obs if hasattr(adata, "obs") else adata
    lab = obs[groupby].astype(str)
    by_study = pd.crosstab(lab, obs[cfg.STUDY_KEY].astype(str))
    out = pd.DataFrame({
        "n_cells": counts.sum(axis=1),
        "n_donors_present": (counts > 0).sum(axis=1),
        f"n_donors_ge{min_cells}": (counts >= min_cells).sum(axis=1),
        "n_studies": (by_study > 0).sum(axis=1),
        "top_study": by_study.idxmax(axis=1),
        "top_study_frac": (by_study.max(axis=1) / by_study.sum(axis=1)).round(3),
    })
    out["claimable"] = out[f"n_donors_ge{min_cells}"] >= min_samples
    out["study_dominated"] = out["top_study_frac"] > 0.8
    if levels is not None:
        out = out.reindex([lv for lv in levels if lv in out.index])
    return out.sort_values("n_cells", ascending=False)


def subsample_common_n(adata, roster, groupby=None, cap=COMMON_N_CAP, min_cells=None,
                       donor_key=None, seed=None, verbose=True):
    """Equalise cells per level, then truncate to a common n. Returns (adata, report).

    ASSUMES THE OBJECT IS ALREADY WINDOWED -- windowing after subsampling spends the cap on
    cells no run will see, and spends it unevenly (structural levels carry far more HC than
    CD4_malignant, which carries none). Procedure, all of it logged:
      n_raw       = min cells over the roster levels
      donor cap   = n_raw // 5, applied first so no single donor defines a level
      n           = min level total after the donor cap, capped at `cap`
      every level truncated to n, uniform at random without replacement, seeded.
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    donor_key = cfg.DONOR_KEY if donor_key is None else donor_key
    seed = cfg.SUBSAMPLE_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    obs = adata.obs
    lab = obs[groupby].astype(str)
    keep_lv = [lv for lv in roster if (lab == lv).any()]
    if len(keep_lv) < 2:
        raise ValueError(f"need >= 2 levels present in this window, got {keep_lv}")
    n_raw = int(min((lab == lv).sum() for lv in keep_lv))
    donor_cap = max(n_raw // 5, 1)

    capped = {}
    for lv in keep_lv:
        idx = obs.index[lab == lv]
        picked = []
        for _d, didx in obs.loc[idx].groupby(obs.loc[idx, donor_key].astype(str),
                                             observed=True).groups.items():
            didx = np.asarray(didx)
            if len(didx) > donor_cap:
                didx = rng.choice(didx, donor_cap, replace=False)
            picked.append(didx)
        capped[lv] = np.concatenate(picked) if picked else np.array([], dtype=object)

    n = min(int(min(len(v) for v in capped.values())), cap)
    keep, report = [], []
    for lv in keep_lv:
        picked = capped[lv]
        if len(picked) > n:
            picked = rng.choice(picked, n, replace=False)
        keep.append(picked)
        report.append({"level": lv, "n_window": int((lab == lv).sum()),
                       "n_after_donor_cap": len(capped[lv]), "n_final": len(picked),
                       "n_donors_final": int(obs.loc[picked, donor_key].nunique()),
                       "power_deficient": len(picked) < n})
    sub = adata[pd.Index(np.concatenate(keep))].copy()
    sub.obs[groupby] = sub.obs[groupby].astype(str).astype("category")
    sub.layers[cfg.LAYER] = sub.X
    rep = pd.DataFrame(report).sort_values("n_window", ascending=False)
    rep.attrs["common_n"] = n
    rep.attrs["donor_cap"] = donor_cap
    if verbose:
        print(f"common n = {n} (n_raw {n_raw} on {keep_lv[int(np.argmin([(lab == lv).sum() for lv in keep_lv]))]}, "
              f"per-donor cap {donor_cap}) -> {sub.n_obs:,} cells over {len(keep_lv)} levels")
    return sub, rep


# ================================================================ the single run


def build_groupby_pairs(levels, both_directions=True) -> pd.DataFrame:
    """Every ordered pair over ONE roster -> a liana `groupby_pairs` frame."""
    rows = [(s, t) for s in levels for t in levels if s != t]
    if not both_directions:
        rows = [(s, t) for i, s in enumerate(levels) for t in levels[i + 1:]]
    return pd.DataFrame(rows, columns=["source", "target"]).drop_duplicates()


def run_spec_consensus(adata, resource, pairs, groupby=None, expr_prop=None, min_cells=None,
                       layer=None, seed=None, n_jobs=None, key_added="spec", verbose=True):
    """Permutation-free specificity consensus over ONE grid -> the scored frame.

    `n_perms` is an int on purpose: `liana_pipe` overwrites `consensus_opts` with 'Magnitude'
    when it is None, and `consensus_opts=["Specificity"]` raises KeyError in liana 1.8.1. Since
    NATMI, Connectome and log2FC all carry `permute=False`, no permutation is ever computed and
    the value is inert. `magnitude_rank` and `cellphone_pvals` are dropped and their absence
    asserted, so nothing downstream can sort or gate on them.

    Only the pairs that cleared `expr_prop` come back. `return_all_lrs=True` is NOT used: the
    aggregate discards liana's `lrs_to_keep` flag anyway, and enumerating the whole grid would
    add ~400k mostly-empty rows for a question `grid_coverage` answers per edge for free.
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    expr_prop = cfg.EXPR_PROP if expr_prop is None else expr_prop
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    layer = cfg.LAYER if layer is None else layer
    seed = cfg.SEED if seed is None else seed
    n_jobs = cfg.N_JOBS if n_jobs is None else n_jobs

    SPEC_CONSENSUS(
        adata, groupby=groupby, groupby_pairs=pairs, resource=resource,
        expr_prop=expr_prop, min_cells=min_cells, use_raw=False, layer=layer,
        n_perms=1,                 # inert: every member method carries permute=False
        consensus_opts=None,       # ["Specificity"] raises KeyError in liana 1.8.1
        return_all_lrs=False, aggregate_method="rra", seed=seed, n_jobs=n_jobs,
        key_added=key_added, verbose=False,
    )
    out = adata.uns[key_added].copy()
    out = out.drop(columns=[c for c in ("magnitude_rank", "cellphone_pvals", "lr_means")
                            if c in out.columns])
    assert RANK_COL in out.columns, "specificity_rank absent -- check consensus_opts/n_perms"
    assert "magnitude_rank" not in out.columns and "cellphone_pvals" not in out.columns

    out["passed_expr_prop"] = True          # only scored pairs come back; kept for uniformity
    out["specificity_pct"] = out[RANK_COL].rank(pct=True)
    if verbose:
        print(f"{key_added}: {len(out):,} edges scored at expr_prop={expr_prop} over "
              f"{out['source'].nunique()} sender levels, {len(pairs)} ordered level pairs")
    return out.sort_values(RANK_COL).reset_index(drop=True)


def grid_coverage(adata, resource, edges, groupby=None, expr_prop=None, min_cells=None,
                  layer=None) -> pd.DataFrame:
    """Why a named edge was not scored: coverage failure or genuine absence.

    Recomputes liana's own filter (every subunit of the ligand complex detected in >= expr_prop
    of the sender, every receptor subunit in the receiver, both levels over min_cells) directly
    off the matrix, for a SHORT list of edges -- the controls, the curated panel, the resolution
    tests. `reason` is one of: scored, level_below_min_cells, ligand_below_expr_prop,
    receptor_below_expr_prop, both_below_expr_prop, gene_absent_from_data.
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    expr_prop = cfg.EXPR_PROP if expr_prop is None else expr_prop
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    layer = cfg.LAYER if layer is None else layer

    edges = pd.DataFrame(list(edges), columns=KEY_COLS) if not isinstance(edges, pd.DataFrame) \
        else edges[KEY_COLS]
    genes = sorted({s for cx in pd.concat([edges["ligand_complex"], edges["receptor_complex"]])
                    for s in str(cx).split("_") if s != "*"})
    have = [g for g in genes if g in adata.var_names]
    X = adata[:, have].layers[layer] if layer in adata.layers else adata[:, have].X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    lab = adata.obs[groupby].astype(str).to_numpy()
    levels = sorted(set(lab))
    prop = pd.DataFrame(0.0, index=levels, columns=have)
    n_cells = pd.Series(0, index=levels, dtype=int)
    for lv in levels:
        idx = np.nonzero(lab == lv)[0]
        n_cells[lv] = len(idx)
        prop.loc[lv] = np.asarray((X[idx] > 0).sum(axis=0)).ravel() / max(len(idx), 1)

    def _side(cx, level):
        subs = [s for s in str(cx).split("_") if s != "*"]
        if any(s not in have for s in subs) or level not in prop.index:
            return np.nan
        return float(prop.loc[level, subs].min())

    rows = []
    for src, tgt, lig, rec in edges.itertuples(index=False):
        lp, rp = _side(lig, src), _side(rec, tgt)
        ns, nt = int(n_cells.get(src, 0)), int(n_cells.get(tgt, 0))
        if np.isnan(lp) or np.isnan(rp):
            reason = "gene_absent_from_data"
        elif ns < min_cells or nt < min_cells:
            reason = "level_below_min_cells"
        elif lp < expr_prop and rp < expr_prop:
            reason = "both_below_expr_prop"
        elif lp < expr_prop:
            reason = "ligand_below_expr_prop"
        elif rp < expr_prop:
            reason = "receptor_below_expr_prop"
        else:
            reason = "scored"
        rows.append({"source": src, "target": tgt, "ligand_complex": lig,
                     "receptor_complex": rec, "n_cells_source": ns, "n_cells_target": nt,
                     "ligand_prop_source": lp, "receptor_prop_target": rp, "reason": reason})
    return pd.DataFrame(rows)


def shuffle_control(adata, resource, pairs, n_shuffles=None, top_n=None, groupby=None,
                    within=None, seed=None, reference=None, min_shuffles=1, **kwargs):
    """Permute the label WITHIN donor, re-run, report top-n overlap AND the recurrent edges.

    Within-donor rather than global holds each donor's composition fixed, so what dissolves is
    cell-type specificity rather than donor identity.

    The count alone is a brittle pass/fail on a max over `n_shuffles` seeds, so the identity of
    every edge that reaches the shuffled top-n is returned too: an edge recoverable without
    cell-type identity is abundance-driven, and section 7 gates it out rather than asserting on
    the count. Returns (per-shuffle counts, recurrent edges with `n_shuffles_recovered` and
    `shuffle_recurrent` = recovered in >= `min_shuffles` shuffles).
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    within = cfg.DONOR_KEY if within is None else within
    n_shuffles = cfg.N_SHUFFLES if n_shuffles is None else n_shuffles
    top_n = cfg.TOP_N if top_n is None else top_n
    seed = cfg.SEED if seed is None else seed

    real_top = set(map(tuple, top_edges(reference, top_n)[KEY_COLS].values))
    rng = np.random.default_rng(seed)
    rows, seen = [], Counter()
    for i in range(n_shuffles):
        shuf = adata.copy()
        lab = shuf.obs[groupby].astype(str).to_numpy().copy()
        for _d, idx in shuf.obs.groupby(shuf.obs[within].astype(str), observed=True).groups.items():
            pos = shuf.obs.index.get_indexer(pd.Index(idx))
            lab[pos] = rng.permutation(lab[pos])
        shuf.obs[groupby] = pd.Categorical(lab)
        res = run_spec_consensus(shuf, resource, pairs, groupby=groupby,
                                 key_added=f"shuffle{i}", verbose=False, **kwargs)
        top = set(map(tuple, top_edges(res, top_n)[KEY_COLS].values))
        seen.update(top)
        rows.append({"shuffle": i, "overlap_with_real": len(top & real_top), "top_n": top_n})

    rec = pd.DataFrame([dict(zip(KEY_COLS, k), n_shuffles_recovered=n) for k, n in seen.items()])
    if len(rec):
        rec["shuffle_recurrent"] = rec["n_shuffles_recovered"] >= min_shuffles
        rec["in_real_top_n"] = [tuple(k) in real_top for k in rec[KEY_COLS].values]
        rec = rec.sort_values(["n_shuffles_recovered"], ascending=False).reset_index(drop=True)
    else:
        rec = pd.DataFrame(columns=KEY_COLS + ["n_shuffles_recovered", "shuffle_recurrent",
                                               "in_real_top_n"])
    return pd.DataFrame(rows), rec


def top_edges(res, top_n=None, source=None, target=None) -> pd.DataFrame:
    """The top-n scored edges by specificity_rank, optionally on one side of the grid."""
    top_n = cfg.TOP_N if top_n is None else top_n
    r = res[res["passed_expr_prop"]] if "passed_expr_prop" in res.columns else res
    if source is not None:
        r = r[r["source"].isin(np.atleast_1d(source))]
    if target is not None:
        r = r[r["target"].isin(np.atleast_1d(target))]
    return r.nsmallest(top_n, RANK_COL)


# ================================================================ donor-level layer


def build_pseudobulk_cube(adata, out=None, meta_out=None, groupby=None, donor_key=None,
                          counts_layer=cfg.COUNTS_LAYER, verbose=True):
    """(level, donor, gene) summed counts + detected-cell counts -> parquet. Returns (cube, meta).

    Regenerated for THIS roster; `ccc_skin_pseudobulk_full.parquet` is keyed on the v1 pooled
    levels x sample and carries no detection counts, so it cannot answer C5a. Computed from the
    already-loaded object by indicator matmul -- the 48 GB source atlas is not re-read. Rows
    with zero detected cells are dropped (expr_prop reads them as 0 by absence); `meta` carries
    n_cells, summed library size and study per (level, donor).
    """
    out = PSEUDOBULK_SUB if out is None else Path(out)
    meta_out = PSEUDOBULK_SUB_META if meta_out is None else Path(meta_out)
    groupby = cfg.GROUPBY if groupby is None else groupby
    donor_key = cfg.DONOR_KEY if donor_key is None else donor_key

    obs = adata.obs
    key = obs[groupby].astype(str) + "||" + obs[donor_key].astype(str)
    levels = pd.unique(key)
    codes = pd.Categorical(key, categories=levels).codes
    ind = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                        shape=(len(levels), len(codes)))

    X = adata.layers[counts_layer].tocsr()
    summed = np.asarray((ind @ X).todense())
    detected = np.asarray((ind @ (X > 0).astype(np.int32)).todense())

    genes = adata.var_names.to_numpy()
    lv, dn = zip(*(s.split("||") for s in levels))
    n_cells = np.asarray(ind.sum(axis=1)).ravel()

    keep = detected > 0
    ii, jj = np.nonzero(keep)
    cube = pd.DataFrame({
        "level": np.asarray(lv)[ii], "donor": np.asarray(dn)[ii], "gene": genes[jj],
        "counts": summed[ii, jj].astype(np.int64), "n_detected": detected[ii, jj].astype(np.int32),
    })
    meta = pd.DataFrame({"level": lv, "donor": dn, "n_cells": n_cells.astype(int)})
    extra = obs.groupby([obs[groupby].astype(str), obs[donor_key].astype(str)], observed=True).agg(
        study=(cfg.STUDY_KEY, "first"),
        disease=("disease", "first"),
        total_counts=("total_counts", "sum"),
        median_lib=("total_counts", "median"),
    ).reset_index()
    extra.columns = ["level", "donor", "study", "disease", "total_counts", "median_lib"]
    meta = meta.merge(extra, on=["level", "donor"], how="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    cube.to_parquet(out, index=False)
    meta.to_parquet(meta_out, index=False)
    if verbose:
        print(f"cube {cube.shape} -> {out.name}; meta {meta.shape} -> {meta_out.name}")
    return cube, meta


class _Cube:
    """(level, donor, gene) cube as dense [n_levels, n_donors, n_genes] arrays.

    Built once so the per-edge loops are numpy indexing rather than pandas slicing: at 17
    levels x 73 donors x 1,832 genes a plane is 18 MB, and a per-edge `.loc` on a MultiIndex
    would be O(rows) each of ~100k times. Carries detection proportion, log1p-CPM, cell counts
    and a per-(complex, level) worst-subunit cache.
    """

    def __init__(self, cube, meta, value="prop"):
        self.levels = sorted(meta["level"].unique())
        self.donors = sorted(meta["donor"].unique())
        self.genes = sorted(cube["gene"].unique())
        self.li = {lv: i for i, lv in enumerate(self.levels)}
        self.di = {d: i for i, d in enumerate(self.donors)}
        self.gi = {g: i for i, g in enumerate(self.genes)}

        self.n_cells = np.zeros((len(self.levels), len(self.donors)))
        for lv, dn, nc in meta[["level", "donor", "n_cells"]].itertuples(index=False):
            self.n_cells[self.li[lv], self.di[dn]] = nc

        i = cube["level"].map(self.li).to_numpy()
        j = cube["donor"].map(self.di).to_numpy()
        k = cube["gene"].map(self.gi).to_numpy()
        arr = np.zeros((len(self.levels), len(self.donors), len(self.genes)), dtype=np.float32)
        if value == "prop":
            denom = np.where(self.n_cells > 0, self.n_cells, 1.0)
            arr[i, j, k] = cube["n_detected"].to_numpy()
            arr /= denom[:, :, None]
        else:                                   # log1p CPM, the pseudobulk analogue of a mean
            lib = np.zeros_like(self.n_cells)
            np.add.at(lib, (i, j), cube["counts"].to_numpy())
            arr[i, j, k] = cube["counts"].to_numpy()
            arr = np.log1p(1e6 * arr / np.maximum(lib, 1.0)[:, :, None])
        self.arr = arr
        self._cx: dict = {}

    def complex_plane(self, cx) -> np.ndarray:
        """[n_levels, n_donors] value for a complex = its WORST subunit; 0 if none present."""
        if cx not in self._cx:
            idx = [self.gi[s] for s in str(cx).split("_") if s in self.gi]
            self._cx[cx] = (self.arr[:, :, idx].min(axis=2) if idx
                            else np.zeros(self.n_cells.shape, dtype=np.float32))
        return self._cx[cx]

    def side(self, cx, level) -> np.ndarray:
        """[n_donors] value for one complex on one level."""
        return self.complex_plane(cx)[self.li[level]]

    def eligible(self, level, min_cells) -> np.ndarray:
        """[n_donors] bool: does this level clear min_cells in that donor."""
        return self.n_cells[self.li[level]] >= min_cells


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def donor_reproducibility(res, cube, meta, edges=None, min_cells=None, expr_prop=None,
                          min_donors=MIN_DONORS_REPRO, verbose=True) -> pd.DataFrame:
    """Per edge: fraction of eligible donors where BOTH sides clear expr_prop. C5a, the primary axis.

    Eligible donor = both levels clear `min_cells` in that donor. Success = every ligand subunit
    clears `expr_prop` in the sender AND every receptor subunit in the receiver. Reports the
    proportion, a Wilson 95% CI, and how many distinct studies recovered the edge. Assumes
    `res` and `cube` describe the same window.
    """
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    expr_prop = cfg.EXPR_PROP if expr_prop is None else expr_prop

    C = _Cube(cube, meta, value="prop")
    study_code = (meta.drop_duplicates("donor").set_index("donor")["study"]
                  .reindex(C.donors).astype(str).to_numpy())
    if edges is None:
        edges = res[res["passed_expr_prop"]][KEY_COLS].drop_duplicates()
    elig = {lv: C.eligible(lv, min_cells) for lv in C.levels}

    rows = []
    for src, tgt, lig, rec in edges.itertuples(index=False):
        el = elig[src] & elig[tgt]
        n_el = int(el.sum())
        ok = el & (C.side(lig, src) >= expr_prop) & (C.side(rec, tgt) >= expr_prop)
        k = int(ok.sum())
        lo, hi = _wilson(k, n_el)
        rows.append({
            "source": src, "target": tgt, "ligand_complex": lig, "receptor_complex": rec,
            "n_donors_tested": n_el, "n_donors_recovered": k,
            "donor_frac": k / n_el if n_el else np.nan, "donor_frac_lo": lo, "donor_frac_hi": hi,
            "n_studies_recovered": int(len(np.unique(study_code[ok]))) if k else 0,
        })
    out = pd.DataFrame(rows)
    out["repro_ok"] = ((out["n_donors_tested"] >= min_donors)
                       & (out["donor_frac"] >= MIN_DONOR_FRAC)
                       & (out["n_studies_recovered"] >= MIN_STUDIES))
    if verbose:
        print(f"{len(out):,} edges | {int(out['repro_ok'].sum()):,} clear donor_frac>="
              f"{MIN_DONOR_FRAC} in >={MIN_STUDIES} studies with >={min_donors} donors tested")
    return out.sort_values("donor_frac", ascending=False)


def paired_contrast(cube, meta, res, level_a=None, level_b=None, min_cells=None,
                    n_cpus=None, verbose=True):
    """Paired malignant-vs-reactive pseudobulk contrast -> (per-gene frame, per-edge frame).

    Donors carrying >= min_cells of BOTH levels; design ~ donor + celltype, so donor, study,
    chemistry and stage all cancel -- the one contrast in this atlas free of study confounding.
    Fitted with PyDESeq2 (no R in this environment; the spec's edgeR/limma-voom is unavailable).
    Size factors come from the 1,832 resource genes only -- median-of-ratios is robust to that,
    but the caveat is logged. The per-edge roll-up takes the malignant-side subunit of each
    edge (ligand where malignant sends, receptor where it receives) and, for a complex, its
    WORST subunit; `sign_ok` means that side is up in malignant.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    level_a = cfg.CD4_MALIGNANT if level_a is None else level_a
    level_b = cfg.CD4_REACTIVE if level_b is None else level_b
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells

    m = meta[meta["level"].isin([level_a, level_b]) & (meta["n_cells"] >= min_cells)]
    donors = sorted(set(m.loc[m.level == level_a, "donor"]) & set(m.loc[m.level == level_b, "donor"]))
    m = m[m["donor"].isin(donors)].copy()
    counts = (cube[cube["level"].isin([level_a, level_b]) & cube["donor"].isin(donors)]
              .pivot_table(index=["level", "donor"], columns="gene", values="counts",
                           aggfunc="sum", fill_value=0))
    counts = counts.reindex(pd.MultiIndex.from_frame(m[["level", "donor"]])).fillna(0)
    counts.index = [f"{lv}|{dn}" for lv, dn in counts.index]

    md = pd.DataFrame({"celltype": m["level"].values, "donor": m["donor"].values},
                      index=counts.index).astype("category")
    keep = (counts > 0).sum(axis=0) >= max(3, len(donors) // 2)
    counts = counts.loc[:, keep].astype(int)
    if verbose:
        print(f"paired contrast: {len(donors)} donors, {counts.shape[0]} pseudo-samples, "
              f"{counts.shape[1]} genes (of {keep.size} in the object)")

    n_cpus = cfg.N_JOBS if n_cpus is None else n_cpus
    dds = DeseqDataSet(counts=counts, metadata=md, design="~donor + celltype", quiet=True,
                       n_cpus=n_cpus)
    dds.deseq2()
    stat = DeseqStats(dds, contrast=["celltype", level_a, level_b], quiet=True, n_cpus=n_cpus)
    stat.summary()
    genes = stat.results_df.rename(columns={"log2FoldChange": "lfc", "padj": "fdr"})
    genes = genes[["lfc", "pvalue", "fdr"]].reset_index().rename(columns={"index": "gene"})
    genes.columns = ["gene", "lfc", "pvalue", "fdr"]

    g = genes.set_index("gene")
    edges = res[res["passed_expr_prop"]][KEY_COLS].drop_duplicates().copy()
    mal_side = np.where(edges["source"] == level_a, edges["ligand_complex"],
                        np.where(edges["target"] == level_a, edges["receptor_complex"], None))
    edges["malignant_side_complex"] = mal_side

    cache: dict = {}

    def _worst(cx):
        if cx is None:
            return (np.nan, np.nan)
        if cx not in cache:
            subs = [s for s in str(cx).split("_") if s in g.index]
            if not subs:
                cache[cx] = (np.nan, np.nan)
            else:
                sub = g.loc[subs]
                # a complex is only as good as its worst subunit
                i = sub["fdr"].fillna(1.0).idxmax()
                cache[cx] = (float(sub.loc[i, "lfc"]), float(sub.loc[i, "fdr"]))
        return cache[cx]

    vals = [_worst(cx) for cx in edges["malignant_side_complex"]]
    edges["contrast_lfc"] = [v[0] for v in vals]
    edges["contrast_fdr"] = [v[1] for v in vals]
    edges["contrast_ok"] = (edges["contrast_fdr"] < FDR_ALPHA) & (edges["contrast_lfc"] > 0)
    edges.attrs["n_donors"] = len(donors)
    if verbose:
        print(f"{int(edges['contrast_ok'].sum()):,}/{len(edges):,} edges: malignant side up at "
              f"FDR<{FDR_ALPHA}")
    return genes, edges


# ================================================================ edge landing (C5c)


def skillings_mack(mat: pd.DataFrame):
    """Skillings-Mack statistic for an incomplete randomised block design -> (stat, df, p).

    Blocks are rows (donors), treatments are columns (levels); NaN = the level was not
    claimable in that donor. Reduces to Friedman when nothing is missing. Lower rank = better,
    so the input must already be within-block ranks or raw scores where small is good.
    """
    m = mat.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if m.shape[1] < 2 or m.shape[0] < 2:
        return (np.nan, 0, np.nan)
    if not m.isna().any().any():
        stat, p = st.friedmanchisquare(*[m[c].to_numpy() for c in m.columns])
        return (float(stat), m.shape[1] - 1, float(p))

    cols = list(m.columns)
    A = np.zeros(len(cols))
    Sigma = np.zeros((len(cols), len(cols)))
    for _b, row in m.iterrows():
        present = [c for c in cols if pd.notna(row[c])]
        k = len(present)
        if k < 2:
            continue
        r = st.rankdata(row[present].to_numpy())
        w = np.sqrt(12.0 / (k + 1))
        for c, rank in zip(present, r):
            A[cols.index(c)] += w * (rank - (k + 1) / 2.0)
        for c in present:
            i = cols.index(c)
            Sigma[i, i] += k - 1
            for d in present:
                if d != c:
                    Sigma[i, cols.index(d)] -= 1
    stat = float(A @ np.linalg.pinv(Sigma) @ A)
    df = len(cols) - 1
    return (stat, df, float(st.chi2.sf(stat, df)))


def landing_test(adata, resource, tests=None, groupby=None, donor_key=None, min_cells=None,
                 seed=None, verbose=True):
    """Which level carries each named edge, power-equalised within donor -> (ranks, summary).

    Per donor: every candidate level that clears `min_cells` is downsampled to the SMALLEST
    such level in that donor, one consensus run over the candidate grid, then the candidate
    levels are ranked against each other on specificity_rank. The comparison is therefore
    always within one run and at equal n, which is what `magnitude_rank` across separate runs
    never was. Returns the per-(test, donor, level) rank frame and a summary carrying the
    median rank, the Skillings-Mack p across donors and a leave-one-donor-out jackknife win %.
    """
    tests = cfg.RESOLUTION_TESTS if tests is None else tests
    groupby = cfg.GROUPBY if groupby is None else groupby
    donor_key = cfg.DONOR_KEY if donor_key is None else donor_key
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    spec = {}
    for name, t in tests.items():
        focal = t.get("sender", t.get("receiver"))
        as_sender = "sender" in t
        resolved = resolve_pairs(t["pairs"], resource)
        resolved = resolved[resolved["orientation"] != "absent"]
        spec[name] = {"focal": focal, "as_sender": as_sender, "candidates": t["candidates"],
                      "expected": t["expected"],
                      "lr": list(zip(resolved["ligand_complex"], resolved["receptor_complex"])),
                      "flipped": set(resolved.loc[resolved.orientation == "flipped", "ligand_in"])}

    all_cand = sorted({c for s in spec.values() for c in s["candidates"]})
    all_focal = sorted({s["focal"] for s in spec.values()})
    lab = adata.obs[groupby].astype(str)

    rows = []
    for donor, didx in adata.obs.groupby(adata.obs[donor_key].astype(str), observed=True).groups.items():
        sub_obs = adata.obs.loc[pd.Index(didx)]
        n_by = sub_obs[groupby].astype(str).value_counts()
        cands = [c for c in all_cand if n_by.get(c, 0) >= min_cells]
        focals = [f for f in all_focal if n_by.get(f, 0) >= min_cells]
        if len(cands) < 2 or not focals:
            continue
        n_eq = int(min(n_by[c] for c in cands + focals))
        picked = []
        for lv in cands + focals:
            ii = sub_obs.index[sub_obs[groupby].astype(str) == lv].to_numpy()
            picked.append(rng.choice(ii, n_eq, replace=False) if len(ii) > n_eq else ii)
        d = adata[pd.Index(np.concatenate(picked))].copy()
        d.obs[groupby] = d.obs[groupby].astype(str).astype("category")
        d.layers[cfg.LAYER] = d.X
        pairs = build_groupby_pairs(cands + focals)
        try:
            res = run_spec_consensus(d, resource, pairs, groupby=groupby,
                                     key_added=f"land_{donor}", verbose=False)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  donor {donor}: {type(exc).__name__}: {exc}")
            continue

        for name, s in spec.items():
            for lv in s["candidates"]:
                if lv not in cands or s["focal"] not in focals:
                    continue
                src, tgt = (s["focal"], lv) if s["as_sender"] else (lv, s["focal"])
                hit = res[(res.source == src) & (res.target == tgt)
                          & res[["ligand_complex", "receptor_complex"]].apply(tuple, axis=1).isin(s["lr"])]
                hit = hit[hit["passed_expr_prop"]]
                rows.append({"test": name, "donor": donor, "level": lv, "n_cells_equalised": n_eq,
                             "score": float(hit[RANK_COL].min()) if len(hit) else np.nan})
        if verbose:
            print(f"  donor {donor}: {len(cands)} candidates at n={n_eq}")

    summary_cols = ["test", "level", "median_within_donor_rank", "n_donors", "is_winner",
                    "expected", "jackknife_win_pct", "skillings_mack_stat",
                    "skillings_mack_df", "skillings_mack_p"]
    ranks = pd.DataFrame(rows, columns=["test", "donor", "level", "n_cells_equalised", "score"])
    if not len(ranks):
        return ranks, pd.DataFrame(columns=summary_cols)
    ranks["within_donor_rank"] = ranks.groupby(["test", "donor"])["score"].rank(method="min")

    summary = []
    for name, grp in ranks.groupby("test"):
        mat = grp.pivot_table(index="donor", columns="level", values="within_donor_rank")
        if mat.empty:                      # the test's pairs were never scored in any donor
            continue
        stat, df, p = skillings_mack(mat)
        med = mat.median(axis=0)
        winner = med.idxmin() if med.notna().any() else None
        wins = []
        for d in mat.index:
            sub = mat.drop(index=d).median(axis=0)
            wins.append(sub.idxmin() if sub.notna().any() else None)
        jack = float(np.mean([w == winner for w in wins])) if wins else np.nan
        for lv in mat.columns:
            summary.append({
                "test": name, "level": lv, "median_within_donor_rank": float(med[lv]),
                "n_donors": int(mat[lv].notna().sum()), "is_winner": lv == winner,
                "expected": lv in spec[name]["expected"],
                "jackknife_win_pct": jack if lv == winner else np.nan,
                "skillings_mack_stat": stat, "skillings_mack_df": df, "skillings_mack_p": p,
            })
    out = pd.DataFrame(summary, columns=summary_cols)
    return ranks, out.sort_values(["test", "median_within_donor_rank"]) if len(out) else out


# ================================================================ controls


def control_report(res, positives=None, negatives=None, resource=None, top_n=None):
    """Per control edge: found, rank position on specificity_rank, percentile, top-n membership.

    Pass `resource` so CSF1->CSF1R and IL7->IL7R resolve onto the resource's spelling instead of
    reporting as absent. No p-value and no magnitude appear here by design.
    """
    positives = cfg.POSITIVE_CONTROLS if positives is None else positives
    negatives = cfg.NEGATIVE_CONTROLS if negatives is None else negatives
    top_n = cfg.TOP_N if top_n is None else top_n
    if resource is not None:
        positives = resolve_controls(positives, resource)
        negatives = resolve_controls(negatives, resource)

    r = res[res["passed_expr_prop"]].sort_values(RANK_COL).reset_index(drop=True)
    r["_pos"] = np.arange(1, len(r) + 1)
    rows = []
    for kind, controls in (("positive", positives), ("negative", negatives)):
        for src, tgt, lig, rec in controls:
            hit = r[(r.source == src) & (r.target == tgt)
                    & ((lig == "*") | (r.ligand_complex == lig))
                    & ((rec == "*") | (r.receptor_complex == rec))]
            row = {"kind": kind, "source": src, "target": tgt, "ligand_complex": lig,
                   "receptor_complex": rec, "found": len(hit) > 0}
            if len(hit):
                best = hit.iloc[0]
                row.update(rank_position=int(best["_pos"]),
                           specificity_rank=float(best[RANK_COL]),
                           pct_of_scored=int(best["_pos"]) / len(r),
                           in_top_n=int(best["_pos"]) <= top_n)
            rows.append(row)
    return pd.DataFrame(rows)


def ambient_regression(res, cube, meta, edges=None, min_donors=MIN_DONORS_AMBIENT,
                       malignant_level=None, verbose=True) -> pd.DataFrame:
    """Per edge: donor score ~ malignant fraction + log median library size -> ambient_flag.

    Replaces the LR_CAVEATS blacklist, which dropped exactly the IL4/IL13 pairs the forced
    panel exists to report. Ambient contamination scales with the local abundance of the source
    cell type, so an edge whose per-donor score is explained by that donor's malignant fraction
    is flagged rather than trusted. Donor score is the pseudobulk analogue of an LR mean:
    the average of the sender's ligand and receiver's receptor mean log1p-CPM, worst subunit.
    BH across edges; `ambient_flag` = positive malignant-fraction coefficient at FDR<0.05.
    """
    malignant_level = cfg.CD4_MALIGNANT if malignant_level is None else malignant_level
    edges = res[res["passed_expr_prop"]][KEY_COLS].drop_duplicates() if edges is None else edges

    C = _Cube(cube, meta, value="cpm")
    n_by_donor = meta.groupby("donor")["n_cells"].sum().reindex(C.donors).to_numpy()
    mal_by_donor = (meta[meta.level == malignant_level].set_index("donor")["n_cells"]
                    .reindex(C.donors).fillna(0).to_numpy())
    malig_frac = mal_by_donor / np.maximum(n_by_donor, 1)
    med_lib = np.log(meta.groupby("donor")["median_lib"].median()
                     .reindex(C.donors).fillna(1).clip(lower=1).to_numpy())
    elig = {lv: C.eligible(lv, cfg.MIN_CELLS) for lv in C.levels}

    rows = []
    for src, tgt, lig, rec in edges.itertuples(index=False):
        score = 0.5 * (C.side(lig, src) + C.side(rec, tgt))
        ok = elig[src] & elig[tgt] & np.isfinite(score) & np.isfinite(malig_frac) & np.isfinite(med_lib)
        if int(ok.sum()) < min_donors:
            rows.append({"source": src, "target": tgt, "ligand_complex": lig,
                         "receptor_complex": rec, "n_donors": int(ok.sum()),
                         "beta_malig_frac": np.nan, "p_malig_frac": np.nan})
            continue
        Xd = np.column_stack([np.ones(int(ok.sum())), malig_frac[ok], med_lib[ok]])
        y = score[ok].astype(float)
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta
        dof = len(y) - Xd.shape[1]
        se = np.sqrt(np.diag(np.linalg.pinv(Xd.T @ Xd)) * (resid @ resid) / max(dof, 1))
        tstat = beta[1] / se[1] if se[1] > 0 else 0.0
        rows.append({"source": src, "target": tgt, "ligand_complex": lig, "receptor_complex": rec,
                     "n_donors": int(ok.sum()), "beta_malig_frac": float(beta[1]),
                     "p_malig_frac": float(2 * st.t.sf(abs(tstat), max(dof, 1)))})
    out = pd.DataFrame(rows)
    out["fdr_malig_frac"] = bh_fdr(out["p_malig_frac"])
    out["ambient_flag"] = (out["fdr_malig_frac"] < FDR_ALPHA) & (out["beta_malig_frac"] > 0)
    if verbose:
        print(f"{int(out['ambient_flag'].sum()):,}/{len(out):,} edges explained by donor "
              f"malignant fraction (FDR<{FDR_ALPHA}, positive slope)")
    return out


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN in, NaN out."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(ranked)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def forced_panel_expression(adata, panel=None, groupby=None, layer=None, min_cells=None,
                            resource=None) -> pd.DataFrame:
    """Per (level, gene): detection proportion and mean lognorm, read straight off the matrix.

    So a curated pair that failed expr_prop is reported as "ligand detected in 6% of the
    sender" rather than as absent. Resolved subunits are included, so a pair that fails only
    on an obligate extra chain (IL2RG on IL7->IL7R) is diagnosable.
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    layer = cfg.LAYER if layer is None else layer
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    frame = resolve_panel(panel, resource)
    cols = ["ligand_in", "receptor_in", "ligand_complex", "receptor_complex"]
    genes = sorted({sub for entry in pd.concat([frame[c] for c in cols]).dropna()
                    for sub in str(entry).split("_") if sub != "*"})
    genes = [g for g in genes if g in adata.var_names]

    X = adata[:, genes].layers[layer] if layer in adata.layers else adata[:, genes].X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    lab = adata.obs[groupby].astype(str).to_numpy()
    rows = []
    for lv in pd.unique(lab):
        idx = np.nonzero(lab == lv)[0]
        if not len(idx):
            continue
        sub = X[idx]
        rows.append(pd.DataFrame({
            "level": lv, "gene": genes, "n_cells": len(idx),
            "expr_prop": np.asarray((sub > 0).sum(axis=0)).ravel() / len(idx),
            "mean_lognorm": np.asarray(sub.mean(axis=0)).ravel(),
            "passes_min_cells": len(idx) >= min_cells,
        }))
    return pd.concat(rows, ignore_index=True)


# ================================================================ the reportable set


def reportable_set(res, repro, contrast, ambient, claimable, shuffle_recurrent=None,
                   top_decile=TOP_DECILE, verbose=True):
    """The six C7 gates ANDed -> (reportable frame, per-gate drop counts).

    1 both levels claimable | 2 donor_frac and >=2 studies | 3 paired-contrast FDR and sign |
    4 not ambient-flagged | 5 not recoverable under the within-donor label shuffle |
    6 specificity_rank in the run's top decile.
    Assumes every input frame is keyed on (source, target, ligand_complex, receptor_complex).
    `shuffle_recurrent` is the second return of `shuffle_control`; None disables gate 5.
    """
    r = res[res["passed_expr_prop"]].copy()
    r = r.merge(repro, on=KEY_COLS, how="left")
    r = r.merge(contrast[KEY_COLS + ["contrast_lfc", "contrast_fdr", "contrast_ok"]],
                on=KEY_COLS, how="left")
    r = r.merge(ambient[KEY_COLS + ["beta_malig_frac", "fdr_malig_frac", "ambient_flag"]],
                on=KEY_COLS, how="left")

    if shuffle_recurrent is None or not len(shuffle_recurrent):
        r["n_shuffles_recovered"], r["shuffle_recurrent"] = 0, False
    else:
        sr = shuffle_recurrent[KEY_COLS + ["n_shuffles_recovered", "shuffle_recurrent"]]
        r = r.merge(sr, on=KEY_COLS, how="left")
        r["n_shuffles_recovered"] = r["n_shuffles_recovered"].fillna(0).astype(int)
        r["shuffle_recurrent"] = r["shuffle_recurrent"].fillna(False).astype(bool)

    cut = r[RANK_COL].quantile(top_decile)
    gates = {
        "1_levels_claimable": r["source"].isin(claimable) & r["target"].isin(claimable),
        "2_donor_reproducible": r["repro_ok"].fillna(False),
        "3_paired_contrast": r["contrast_ok"].fillna(False),
        "4_not_ambient": ~r["ambient_flag"].fillna(False),
        "5_not_shuffle_recurrent": ~r["shuffle_recurrent"],
        "6_top_decile": r[RANK_COL] <= cut,
    }
    for name, g in gates.items():
        r[name] = g
    keep = np.logical_and.reduce([g.to_numpy() for g in gates.values()])

    surviving = pd.Series(True, index=r.index)
    drops = []
    for name, g in gates.items():
        before = int(surviving.sum())
        surviving &= g
        drops.append({"gate": name, "n_in": before, "n_out": int(surviving.sum()),
                      "n_dropped": before - int(surviving.sum()),
                      "n_failing_alone": int((~g).sum())})
    if verbose:
        print(f"reportable set: {int(keep.sum()):,} of {len(r):,} scored edges "
              f"(top-decile cut specificity_rank <= {cut:.4g})")
    return r[keep].sort_values(RANK_COL).reset_index(drop=True), pd.DataFrame(drops)


# ================================================================ figures


def set_style() -> None:
    """White background, one teal accent, Calibri/Carlito where installed (DejaVu here)."""
    import logging

    import matplotlib as mpl

    # Neither font is installed on this cluster; the preference list is kept so the SVG is
    # right wherever it is opened, and the per-glyph fallback warning is silenced.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": ["Calibri", "Carlito", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "svg.fonttype": "none", "figure.dpi": 110,
        "savefig.dpi": 200,
        # every figure below is built with layout="constrained"; these are its paddings
        "figure.constrained_layout.h_pad": 0.10, "figure.constrained_layout.w_pad": 0.10,
        "figure.constrained_layout.hspace": 0.10, "figure.constrained_layout.wspace": 0.06,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _interaction(frame):
    return frame["ligand_complex"] + " -> " + frame["receptor_complex"]


def _label_inches(labels, per_char=0.075, floor=1.2):
    """Horizontal inches a column of tick labels needs, so constrained_layout never clips it."""
    labels = [str(x) for x in labels]
    return max(floor, per_char * max((len(x) for x in labels), default=8))


def _size_handles(fracs, base, scale, color="0.45"):
    """Proxy markers for a dot-size legend; `s = base + scale * frac` must match the scatter."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker="o", linestyle="none", color=color,
                   markersize=np.sqrt(base + scale * f), label=f"{f:.0%}") for f in fracs]


def _text_on(cmap, value, vmin, vmax):
    """Black or white annotation, whichever survives on that cell."""
    import matplotlib.colors as mcolors

    if vmax <= vmin or not np.isfinite(value):
        return "black"
    r, g, b, _ = cmap(mcolors.Normalize(vmin, vmax)(value))
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


def plot_dotplot(res, repro, focal=None, partners=None, top_n=None, out=None, title=None):
    """Dotplot: x = partner level, y = interaction, colour = inverted rank, size = donor frac.

    One panel per direction (focal as sender / as receiver), stacked vertically so the long
    `ligand -> receptor` labels never run into the neighbouring panel. Assumes `repro` carries
    donor_frac on the same keys as `res`. Returns the Figure.
    """
    import matplotlib.pyplot as plt

    focal = cfg.CD4_MALIGNANT if focal is None else focal
    top_n = cfg.DOTPLOT_TOP_N if top_n is None else top_n
    r = res[res["passed_expr_prop"]].merge(repro[KEY_COLS + ["donor_frac"]], on=KEY_COLS,
                                           how="left", suffixes=("", "_y"))
    r = r.assign(interaction=_interaction(r), score=-np.log10(r[RANK_COL].clip(lower=1e-12)))
    if partners is not None:
        r = r[r["source"].isin([*partners, focal]) & r["target"].isin([*partners, focal])]

    # Both panels are built before the figure so the canvas is sized from the content and the
    # colour scale is shared; sizing after the fact is what made the labels collide.
    panels = {}
    for direction in ("out", "in"):
        d = (r[r["source"] == focal].assign(partner=lambda x: x["target"]) if direction == "out"
             else r[r["target"] == focal].assign(partner=lambda x: x["source"]))
        panels[direction] = d[d["interaction"].isin(d.nsmallest(top_n, RANK_COL)["interaction"])]

    drawn = [d for d in panels.values() if len(d)]
    vmin, vmax = ((min(d["score"].min() for d in drawn), max(d["score"].max() for d in drawn))
                  if drawn else (0.0, 1.0))
    # One partner axis for both panels, so a column means the same level in each.
    parts = ([p for p in partners if p != focal] if partners is not None
             else sorted({p for d in drawn for p in d["partner"].unique()}))
    n_rows = max((d["interaction"].nunique() for d in drawn), default=1)
    lab_w = _label_inches([i for d in drawn for i in d["interaction"].unique()])
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(0.55 * len(parts) + lab_w + 3.0, 2 * (0.34 * n_rows + 1.3)),
        layout="constrained")
    sc = None
    for ax, direction in zip(axes, ["out", "in"]):
        d = panels[direction]
        if not len(d):
            ax.set_axis_off()
            continue
        order = d.groupby("interaction")[RANK_COL].min().sort_values().index.tolist()
        sc = ax.scatter([parts.index(p) for p in d["partner"]],
                        [order.index(i) for i in d["interaction"]],
                        c=d["score"], s=30 + 200 * d["donor_frac"].fillna(0),
                        cmap="viridis", vmin=vmin, vmax=vmax, edgecolor="none")
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels(parts, rotation=45, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_title(f"{focal} -> partner" if direction == "out" else f"partner -> {focal}",
                     color=TEAL, loc="left")
        ax.set_xlim(-0.7, len(parts) - 0.3)
        ax.set_ylim(len(order) - 0.3, -0.7)
        ax.grid(color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    if sc is not None:
        fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.015, label="-log10 specificity_rank")
        fig.legend(handles=_size_handles([0.25, 0.50, 1.00], 30, 200),
                   title="donors recovering", loc="outside lower right", frameon=False,
                   ncols=3, handletextpad=0.6)
    fig.suptitle(title or f"{focal}: top {top_n} edges per direction", color=TEAL)
    if out is not None:
        fig.savefig(out)
    return fig


def plot_dotplot_programs(res, repro, programs=None, focal=None, partners=None, out=None,
                          title=None, drop_empty=True):
    """`plot_dotplot`, but rows are the curated edges grouped into biological programs.

    Same encoding as figure 1 (x = partner, colour = -log10 specificity_rank, size = donor
    frac) so the two read against each other; no top-N truncation, because the point here is
    which curated edges are present and which are not. Rows are blocked by program in
    `CCC_PROGRAMS` order, shaded in alternating bands, and the program name is written to the
    right of its block. Returns the Figure, with the curated edges that were never scored on
    its `missing_` attribute so the negatives can be logged rather than silently dropped.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    programs = CCC_PROGRAMS if programs is None else programs
    focal = cfg.CD4_MALIGNANT if focal is None else focal
    lookup: dict[str, str] = {}
    for name, lrs in programs.items():
        for lr in lrs:
            lookup.setdefault(lr, name)          # first program wins if an edge is listed twice

    r = res[res["passed_expr_prop"]].merge(repro[KEY_COLS + ["donor_frac"]], on=KEY_COLS,
                                           how="left", suffixes=("", "_y"))
    r = r.assign(interaction=_interaction(r),
                 score=-np.log10(r[RANK_COL].clip(lower=1e-12)),
                 program=(r["ligand_complex"] + "-" + r["receptor_complex"]).map(lookup))
    r = r[r["program"].notna()]
    if partners is not None:
        r = r[r["source"].isin([*partners, focal]) & r["target"].isin([*partners, focal])]

    panels = {"out": r[r["source"] == focal].assign(partner=lambda x: x["target"]),
              "in": r[r["target"] == focal].assign(partner=lambda x: x["source"])}
    drawn = [d for d in panels.values() if len(d)]
    if not drawn:
        raise ValueError(f"no curated program edge involving {focal} cleared expr_prop")
    vmin, vmax = min(d["score"].min() for d in drawn), max(d["score"].max() for d in drawn)
    parts = ([p for p in partners if p != focal] if partners is not None
             else sorted({p for d in drawn for p in d["partner"].unique()}))

    # Row order and block extents are fixed before anything is drawn, so both panels can be
    # sized from their true row counts and a program never straddles a band boundary.
    orders, blocks = {}, {}
    for direction, d in panels.items():
        order, blk = [], []
        for name in programs:
            rows = (d[d["program"] == name].groupby("interaction")[RANK_COL]
                    .min().sort_values().index.tolist())
            if not rows and drop_empty:
                continue
            blk.append((name, len(order), len(order) + len(rows)))
            order += rows
        orders[direction], blocks[direction] = order, blk

    n_out, n_in = max(len(orders["out"]), 1), max(len(orders["in"]), 1)
    lab_w = _label_inches([i for o in orders.values() for i in o])
    prog_w = _label_inches([n.replace("_", " ") for n in programs], floor=1.5) + 0.4
    fig_w = 0.55 * len(parts) + lab_w + prog_w + 2.0
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(fig_w, 0.34 * (n_out + n_in) + 3.2),
        height_ratios=[n_out, n_in], layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1 - prog_w / fig_w, 1))

    sc = None
    for ax, direction in zip(axes, ["out", "in"]):
        d, order = panels[direction], orders[direction]
        if not len(d):
            ax.set_axis_off()
            continue
        blend = blended_transform_factory(ax.transAxes, ax.transData)
        for j, (name, lo, hi) in enumerate(blocks[direction]):
            if j % 2:
                ax.axhspan(lo - 0.5, hi - 0.5, color="0.96", zorder=0)
            if lo:
                ax.axhline(lo - 0.5, color="0.75", lw=0.8)
            ax.plot([1.01, 1.01], [lo - 0.35, hi - 0.65], transform=blend, color=TEAL, lw=1.5,
                    clip_on=False, solid_capstyle="butt")
            ax.text(1.03, (lo + hi - 1) / 2, name.replace("_", " "), transform=blend,
                    va="center", ha="left", fontsize=8, color=TEAL, clip_on=False)
        sc = ax.scatter([parts.index(p) for p in d["partner"]],
                        [order.index(i) for i in d["interaction"]],
                        c=d["score"], s=30 + 200 * d["donor_frac"].fillna(0),
                        cmap="viridis", vmin=vmin, vmax=vmax, edgecolor="none", zorder=3)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels(parts, rotation=45, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_title(f"{focal} -> partner" if direction == "out" else f"partner -> {focal}",
                     color=TEAL, loc="left")
        ax.set_xlim(-0.7, len(parts) - 0.3)
        ax.set_ylim(len(order) - 0.3, -0.7)
        ax.grid(color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    if sc is not None:
        fig.colorbar(sc, ax=axes, location="bottom", fraction=0.03, pad=0.02, shrink=0.5,
                     label="-log10 specificity_rank")
        fig.legend(handles=_size_handles([0.25, 0.50, 1.00], 30, 200),
                   title="donors recovering", loc="outside lower left", frameon=False,
                   ncols=3, handletextpad=0.6)
    fig.suptitle(title or f"{focal}: curated edges by biological program", color=TEAL)
    if out is not None:
        fig.savefig(out)

    scored = set((r["ligand_complex"] + "-" + r["receptor_complex"]).unique())
    fig.missing_ = {name: [lr for lr in lrs if lr not in scored] for name, lrs in programs.items()}
    return fig


def plot_heatmap(res, contrast, focal=None, partners=None, top_n=None, out=None, title=None):
    """Heatmap: interactions x partner, value = inverted rank, `*` at paired-contrast FDR<0.05.

    Shared vmin/vmax across both panels; `set_bad` grey so a structurally absent cell reads as
    missing rather than weak. Rows ordered by best rank across partners. Returns the Figure.
    """
    import matplotlib.pyplot as plt

    focal = cfg.CD4_MALIGNANT if focal is None else focal
    top_n = cfg.HEATMAP_TOP_N if top_n is None else top_n
    r = res[res["passed_expr_prop"]].merge(contrast[KEY_COLS + ["contrast_fdr"]], on=KEY_COLS,
                                           how="left")
    r = r.assign(interaction=_interaction(r), value=-np.log10(r[RANK_COL].clip(lower=1e-12)))
    if partners is None:
        partners = sorted(set(r.loc[r.source == focal, "target"])
                          | set(r.loc[r.target == focal, "source"]))
    partners = [p for p in partners if p != focal]
    r = r[((r.source == focal) & r.target.isin(partners))
          | ((r.target == focal) & r.source.isin(partners))]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(GREY_BAD)
    vmin, vmax = (r["value"].min(), r["value"].max()) if len(r) else (0, 1)
    lab_w = _label_inches(r["interaction"].unique() if len(r) else [])
    fig, axes = plt.subplots(
        2, 1, sharex=True,
        figsize=(0.55 * len(partners) + lab_w + 3.0, 2 * (0.32 * top_n + 1.3)),
        layout="constrained")
    im = None
    for ax, direction in zip(axes, ["out", "in"]):
        d = (r[r.source == focal].assign(partner=lambda x: x["target"]) if direction == "out"
             else r[r.target == focal].assign(partner=lambda x: x["source"]))
        if not len(d):
            ax.set_axis_off()
            continue
        order = d.groupby("interaction")[RANK_COL].min().sort_values().head(top_n).index
        mat = d.pivot_table(index="interaction", columns="partner", values="value").reindex(
            index=order, columns=partners)
        sig = d.pivot_table(index="interaction", columns="partner",
                            values="contrast_fdr").reindex(index=order, columns=partners)
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(partners)))
        ax.set_xticklabels(partners, rotation=45, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_title(f"{focal} -> partner" if direction == "out" else f"partner -> {focal}",
                     color=TEAL, loc="left")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if pd.notna(sig.values[i, j]) and sig.values[i, j] < FDR_ALPHA:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=11,
                            color=_text_on(cmap, mat.values[i, j], vmin, vmax))
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, label="-log10 specificity_rank")
    fig.suptitle(title or f"{focal} <-> TME  (* = paired contrast FDR<{FDR_ALPHA}; "
                          "grey = below min_cells)", color=TEAL)
    if out is not None:
        fig.savefig(out)
    return fig


def plot_landing_heatmap(summary, out=None, title=None):
    """Heatmap: RESOLUTION_TEST x level, median within-donor rank, winner annotated with win %."""
    import matplotlib.pyplot as plt

    if not len(summary):
        fig, ax = plt.subplots(figsize=(7, 2.2), layout="constrained")
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no resolution test was scored in any donor", ha="center")
        return fig
    mat = summary.pivot_table(index="test", columns="level", values="median_within_donor_rank")
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(GREY_BAD)
    vmin, vmax = np.nanmin(mat.values), np.nanmax(mat.values)
    fig, ax = plt.subplots(
        figsize=(0.85 * mat.shape[1] + _label_inches(mat.index) + 2.5, 0.95 * mat.shape[0] + 2.6),
        layout="constrained")
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    win = summary[summary["is_winner"]].set_index("test")
    for i, t in enumerate(mat.index):
        for j, lv in enumerate(mat.columns):
            if pd.isna(mat.values[i, j]):
                continue
            mark = ""
            if t in win.index and win.loc[t, "level"] == lv:
                mark = f"\n{100 * win.loc[t, 'jackknife_win_pct']:.0f}%"
            ax.text(j, i, f"{mat.values[i, j]:.0f}{mark}", ha="center", va="center",
                    fontsize=9, linespacing=1.3,
                    color=_text_on(cmap, mat.values[i, j], vmin, vmax))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015,
                 label="median within-donor rank (1 = best)")
    ax.set_title(title or "Where each named edge lands, power-equalised within donor\n"
                          "(cell = median rank; winner annotated with donor-jackknife win %)",
                 color=TEAL)
    if out is not None:
        fig.savefig(out)
    return fig


def plot_panel_dotplot(fpe, panel_frame, levels=None, out=None, title=None):
    """Forced curated panel: x = level, y = panel gene, size = DETECTION proportion.

    Plotted whether or not the pair cleared expr_prop -- reporting the negatives is the point.
    """
    import matplotlib.pyplot as plt

    genes = sorted({s for entry in pd.concat([panel_frame["ligand_in"],
                                              panel_frame["receptor_in"]]).dropna()
                    for s in str(entry).split("_") if s != "*"})
    d = fpe[fpe["gene"].isin(genes)]
    if levels is not None:
        d = d[d["level"].isin(levels)]
    lv = sorted(d["level"].unique())
    gs = [g for g in genes if g in set(d["gene"])]
    # One tall column: ~0.3 in per gene keeps every symbol legible; a wide layout would only be
    # shrunk to the notebook width and take the tick labels with it.
    fig, ax = plt.subplots(figsize=(0.55 * len(lv) + _label_inches(gs, floor=0.9) + 3.0,
                                    0.30 * len(gs) + 2.6), layout="constrained")
    sc = ax.scatter([lv.index(x) for x in d["level"]], [gs.index(g) for g in d["gene"]],
                    s=10 + 180 * d["expr_prop"], c=d["mean_lognorm"], cmap="viridis",
                    edgecolor="none")
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels(lv, rotation=45, ha="right")
    ax.set_yticks(range(len(gs)))
    ax.set_yticklabels(gs, fontsize=9)
    ax.set_xlim(-0.7, len(lv) - 0.3)
    ax.set_ylim(-0.7, len(gs) - 0.3)
    ax.grid(color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.015, label="mean lognorm")
    fig.legend(handles=_size_handles([0.1, 0.5, 1.0], 10, 180), title="detection",
               loc="outside lower right", frameon=False, ncols=3, handletextpad=0.6)
    ax.set_title(title or "Curated panel, forced: size = detection proportion, colour = mean "
                          "lognorm\n(plotted whether or not the pair cleared expr_prop)",
                 color=TEAL)
    if out is not None:
        fig.savefig(out)
    return fig


# ================================================================ printouts


def printout(title: str, lines=None, tables=None, log=None, width=110) -> None:
    """Print one self-describing fixed-width block and append it to tables/ccc40_run_log.md.

    The block must be readable pasted into a chat with no access to the notebook: what ran,
    n per level, n donors, n tests, n passing, and the top rows as text. No emoji, no ASCII
    art, no progress bars.
    """
    buf = ["=" * width, title, "=" * width]
    for line in (lines or []):
        buf.append(str(line))
    for name, frame in (tables or {}).items():
        buf.append("")
        buf.append(f"--- {name} ---")
        if isinstance(frame, pd.DataFrame):
            buf.append(frame.to_string(index=False, max_rows=25, float_format=lambda v: f"{v:.4g}"))
        elif isinstance(frame, pd.Series):
            buf.append(frame.to_string(float_format=lambda v: f"{v:.4g}"))
        else:
            buf.append(str(frame))
    block = "\n".join(buf)
    print(block)
    log = RUN_LOG if log is None else log      # resolved at call time, not at import
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write("\n```\n" + block + "\n```\n")


def reset_run_log(log=None, header="# ccc40 run log") -> None:
    """Truncate the run log so a clean-kernel run does not append to the previous one."""
    log = RUN_LOG if log is None else log
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log).write_text(f"{header}\n")
