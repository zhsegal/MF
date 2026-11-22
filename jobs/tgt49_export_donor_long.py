#!/usr/bin/env python
"""Donor-level long export of the tgt49 skin CD4 surfaceome screen (notebook 49 / report 49).

Purpose: give the figures a per-donor frame so the shortlist can be drawn as paired per-donor
strips instead of medians, and add the one statistic report 49 is missing -- detection measured
at a common sequencing depth.

What is reused, not re-derived. The screen's per-(arm, donor, gene) aggregation is
`data/targets/skin_target_long.parquet`, written by `jobs/run_target_screen_qc.py`; this script
reads it. The cohort is `target_cohort.skin_donor_table()` + `target_cohort.filter_long()`, the
arms are the builder's, the normalisation is the builder's log1p CP10K on `obs.total_counts`, and
the tests are `target_screen.paired_selectivity` / `.prevalence` and `target_het.lfc_matrix` /
`.eligibility` -- the same calls notebook 49 makes. Nothing published is overwritten; two new
tables are written.

No refactor of the medians was needed. Notebook 49 computes nothing inline: every median in
`tables/tgt49_all_genes.csv` already comes from `TS.prevalence(long, arm)` applied to the
per-donor frame above, and that frame is this export's input.

THE HEALTHY-ARM FIX (repair, not exclusion)
-------------------------------------------
D14__SC320 / SC348 / SC353 / SC354 are zero for every one of the 2,355 screened genes, not only
the 33 detail genes. Cause: 1,083 of the 1,147 gaydosik2023 (D14) cells in
`skin_T_tcr_annotated_v4.h5ad` have an EMPTY count row -- `indptr[i+1] == indptr[i]` -- in both
`X` and `layers/raw_counts`, while their `obs` still carries intact `total_counts` (median 3,774)
and `n_genes`. The counts were lost when the joint atlas was assembled; they are not a
gene-symbol mismatch (the 64 D14 cells that survived carry ordinary symbols) and not biology.
The builder then divided a row of zeros by a plausible library size and produced a donor of pure
zeros that passed the >= 20-cell floor.

The counts are recoverable exactly. All 1,147 cells match `data/D14_gaydosik2023_skin/processed/
standardized.h5ad` by cell name, and `total_counts` and `n_genes` agree there for 1,147 of 1,147
cells, with `nnz == n_genes` per row. So the five affected donor-arms are REPAIRED here: their
`CD4_HC` aggregates are recomputed from the cohort's own object with the builder's normalisation.
A fifth donor is repaired that the brief does not name -- D14__SC405, 85 of its 149 cells are
empty, so its published `pct_expr` is diluted by 57% structural zeros rather than zeroed outright.
Three of the 828 exported genes (ADAM29, CHRNB3, DRD5) are absent from D14's own `var` and are
therefore genuinely unmeasured in those five donors: NaN, not 0.

The repair changes the healthy arm, which is the point of it, so `tgt49_donor_long.csv` and the
`vs healthy` columns of `tgt49_gene_stats.csv` carry the REPAIRED values. Verification 1 runs
both ways: the published (broken) HC arm must reproduce the published tables to 1e-6, and the
repaired delta is reported separately, per table.

DEPTH MATCHING
--------------
Section 1e gives 3,330 median UMI per malignant CD4 cell against 2,040 reactive -- 1.6x. Every
`pct_expr` comparison in report 49, and the shortlist's `malig_frac_donors_pos >= 0.5` gate, is a
detection threshold applied across that gap. `pct_expr_depthmatched` is `pct_expr` recomputed
after binomially thinning every cell to `DS_DEPTH` = 2,000 UMI (`run_target_screen_qc.thin`, the
builder's own function), over the cells that reach that depth. It is computed for ALL 2,355
screened genes, not only the 828 exported, so its BH correction runs over the same gene universe
as the headline `q` and the two are directly comparable.

Determinism: SEED = 0, one `np.random.default_rng(SEED)` created here and consumed in gene-block
order. This reproduces exactly on re-run. It is NOT the same draw as the builder's `mean_all_ds`
column -- that rng walked 2,355 genes over all 539,916 cells in a different order -- so the
published `mean_all_ds` is compared here as a sanity check with a Monte-Carlo tolerance, never at
1e-6.

Outputs (new files only):
  tables/tgt49_donor_long.csv   (donor_key, arm, gene) grid, NaN where not evaluated
  tables/tgt49_gene_stats.csv   one row per exported gene, four tests, suffixed per test
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TARGET_SCREEN_THREADS", "4"))
os.environ.setdefault("MPLBACKEND", "Agg")

import json                                     # noqa: E402
import h5py                                     # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
from scipy import stats                         # noqa: E402
from statsmodels.stats.multitest import multipletests   # noqa: E402


def _nb_dir() -> Path:
    for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if p.name == "MF" and (p / "data").is_dir():
            return p
    raise RuntimeError("MF project root not found")


NB_DIR = _nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))
sys.path.insert(0, str(NB_DIR / "jobs"))

import cluster_malig_helpers as CM              # noqa: E402
import malignancy as MAL                        # noqa: E402
import lineage as LIN
import surface_targets as ST                    # noqa: E402
import target_cohort as TC                      # noqa: E402
import target_het as TH                         # noqa: E402
import target_qc as TQ                          # noqa: E402
import target_screen as TS                      # noqa: E402
import run_target_screen_qc as Q                # noqa: E402   the builder; read_block/thin/consts

# ---- the notebook's constants, restated so this file is readable on its own -------------------
TISSUE = "skin"
TARGETS = NB_DIR / "data" / "targets"
TAB_DIR = NB_DIR / "tables"
MALIG, REACT, HC = "CD4_malignant", "CD4_reactive", "CD4_HC"
CD4_ARMS = [MALIG, REACT, HC]
MIN_CELLS = 20          # applied by the builder; recorded here, not re-applied
MIN_DONORS = 5
DETECT_FLOOR = 0.10
PSEUDO = 0.05
SHORT_Q, SHORT_LFC, SHORT_PREV = 0.05, 0.5, 0.5
SEED = Q.SEED           # 0
DS_DEPTH = Q.DS_DEPTH   # 2,000 UMI
D14_SRC = NB_DIR / "data" / "D14_gaydosik2023_skin" / "processed" / "standardized.h5ad"

TOL = 1e-6
FAILURES: list[str] = []


# --------------------------------------------------------------------------- reporting
def hr(title):
    print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96, flush=True)


def check(label, dev, n):
    ok = np.isfinite(dev) and dev <= TOL
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:<64} max|dev| = {dev:.3e}  over {n:,} values")
    if not ok:
        FAILURES.append(f"{label}: max|dev| = {dev:.3e} over {n:,} values")


def dev_of(new, old, cols, keys, label, record=True):
    """Max absolute deviation between a recomputed frame and a published table."""
    m = new.merge(old, on=keys, how="outer", suffixes=("_new", "_pub"), indicator=True)
    if (m["_merge"] != "both").any():
        bad = m.loc[m["_merge"] != "both", keys].astype(str).agg(" ".join, axis=1).tolist()
        msg = f"{label}: {len(bad)} key(s) present on one side only: {bad[:8]}"
        print(f"  [FAIL] {msg}")
        if record:
            FAILURES.append(msg)
        m = m[m["_merge"] == "both"]
    worst, n = 0.0, 0
    for c in cols:
        a = pd.to_numeric(m[f"{c}_new"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[f"{c}_pub"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.abs(a - b)
        d[both_nan] = 0.0
        if np.isnan(d).any() and record:
            FAILURES.append(f"{label}/{c}: {int(np.isnan(d).sum())} value(s) NaN on one side only")
        worst = max(worst, float(np.nanmax(d)) if len(d) else 0.0)
        n += int((~both_nan).sum())
    if record:
        check(label, worst, n)
    return worst, n


# --------------------------------------------------------------------------- inputs
def load_published():
    """Notebook 49 section 1, verbatim in effect: cohort, CD4 long frame, per-arm meta."""
    man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
    assert man.get("builder") == "run_target_screen_qc.py", man.get("builder")
    assert man.get("ds_depth") == DS_DEPTH, (man.get("ds_depth"), DS_DEPTH)

    CM.use_tissue(TISSUE)
    MAL.use_tissue(TISSUE)
    TQ.use_tissue(TISSUE)
    assert MAL.CLUSTER_PQ.name.startswith("skin_"), MAL.CLUSTER_PQ
    assert "KRT14" in TQ.BAND_GENES

    don = TC.skin_donor_table()
    raw = pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet")
    long = TC.filter_long(raw, don)
    long = long[long["arm"].isin(CD4_ARMS)].copy()
    del raw
    meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)

    don = don.join(TC.burden(long), how="left")
    both = long[long["arm"].isin([MALIG, REACT])].groupby("donor")["arm"].nunique().eq(2)
    paired = sorted(both.index[both])
    print(f"cohort: {int(don['cohort_keep'].sum())} independent donor keys; "
          f"{long['donor'].nunique()} with a CD4 arm; {len(paired)} paired "
          f"(malignant AND reactive); {long.loc[long['arm'].eq(HC), 'donor'].nunique()} CD4_HC")
    return long, meta, don, paired, man


def gene_universe(long):
    """The screened gene list and the surfaceome reference notebook 49 section 2 annotates with."""
    surf_all = ST.load_surfaceome(confident_only=False, strict=True)
    return sorted(long["gene"].unique()), surf_all


def export_genes(all_genes_pub, shortlist_pub, genes_present):
    """The brief's gene set: shortlist + positive controls + loss markers + surface q < 0.05.

    `gene_set` names which of those four a gene is carried for, with that precedence, so a
    positive control that is also q < 0.05 is not lost among the 823.
    """
    short = shortlist_pub["gene"].tolist()
    pos = [g for g in ST.POSITIVE_CONTROLS if g in genes_present]
    loss = [g for g in ST.LOSS_MARKERS if g in genes_present]
    surf_q = all_genes_pub.loc[all_genes_pub["gene_set"].eq("surface")
                               & (all_genes_pub["q"] < SHORT_Q), "gene"].tolist()
    role = {}
    for g in surf_q:
        role[g] = "surface_q05"
    for g in loss:
        role[g] = "loss_marker"
    for g in pos:
        role[g] = "positive_control"
    for g in short:
        role[g] = "shortlist"
    genes = sorted(role)
    print(f"export gene set: {len(short)} shortlist + {len(pos)} positive controls + "
          f"{len(loss)} loss markers + {len(surf_q)} surface genes with q < {SHORT_Q} "
          f"= {len(genes)} distinct genes")
    print("  the brief says 817 surface genes at q < 0.05; the published table has "
          f"{len(surf_q)} (gene_set == 'surface' and q < {SHORT_Q}). The larger set is written.")
    print("  gene_set counts: "
          + ", ".join(f"{k} {v}" for k, v in pd.Series(role).value_counts().items()))
    return genes, role


# --------------------------------------------------------------------------- per-cell arms
def cell_arms(long):
    """The builder's per-cell arm label, rebuilt, and checked against the published cell counts.

    `run_target_screen_qc.main()` does not store a per-cell arm, and both the repair and the
    depth-matched pass need one. It is rebuilt with the same three lines the builder uses, then
    asserted equal to the published `n_cells` of every (arm, donor) group -- which is what makes
    it safe to aggregate new statistics on.
    """
    cfg = Q.TISSUE[TISSUE]
    with h5py.File(cfg["obj"], "r") as h5:
        names = pd.Index(CM._read_elem(h5["obs"][h5["obs"].attrs["_index"]])).astype(str)
        obs = CM.read_obs(h5, cols=[cfg["donor"], "study", cfg["celltype"], "has_tcr",
                                    "total_counts"])
    obs.index = names
    mal = MAL.load_malignancy(obs.index, has_tcr=obs["has_tcr"].astype(bool).to_numpy())
    lvl = pd.Series(np.asarray(MAL.cd4_level(obs[cfg["celltype"]], mal, call=MAL.HEADLINE)
                               ).astype(str), index=obs.index)
    don_tbl = TC.donor_table(TISSUE, obj=cfg["obj"])
    donor = obs[cfg["donor"]].astype(str)
    is_cd4 = obs[cfg["celltype"]].astype(str).isin(LIN.CD4_ANY).to_numpy()
    is_hc = donor.map(don_tbl["is_hc"]).fillna(False).to_numpy()
    lvl[is_cd4 & is_hc] = HC

    cells = pd.DataFrame({"arm": lvl.to_numpy(), "donor": donor.to_numpy(),
                          "study": obs["study"].astype(str).to_numpy(),
                          "lib": pd.to_numeric(obs["total_counts"], errors="coerce")
                          .to_numpy(float)}, index=obs.index)
    cells["row"] = np.arange(len(cells))

    pub = long.groupby(["arm", "donor"])["n_cells"].first()
    got = cells.groupby(["arm", "donor"]).size()
    j = pd.concat([pub.rename("pub"), got.rename("got")], axis=1).dropna(subset=["pub"])
    bad = j[j["pub"] != j["got"]]
    print(f"per-cell arms rebuilt: {len(j):,} published CD4 (arm, donor) groups checked, "
          f"{len(bad)} disagree on cell count")
    assert bad.empty, bad.head(20)
    return cells


# --------------------------------------------------------------------------- the HC repair
def repair_d14(long, cells, genes_all):
    """Recompute the five gaydosik2023 CD4_HC donor-arms from that cohort's own object.

    Returns (replacement rows, diagnosis frame, unmeasured genes). The replacement carries the
    builder's quantities for the columns this export uses: n_cells, pct_expr, mean_all, sd_all.
    """
    cfg = Q.TISSUE[TISSUE]
    with h5py.File(cfg["obj"], "r") as h5:
        ip = h5["layers"][cfg["layer"]]["indptr"][:].astype(np.int64)
    nnz = np.diff(ip)
    cells = cells.assign(nnz=nnz)

    empty = cells[cells["nnz"] == 0]
    print(f"\nempty count rows in {Path(cfg['obj']).name}: {len(empty):,} of {len(cells):,} "
          f"({len(empty) / len(cells):.3%}), in {empty['study'].nunique()} study(s): "
          f"{sorted(empty['study'].unique())}")
    diag = (cells[cells["study"].eq("gaydosik2023")]
            .groupby(["donor", "arm"])
            .agg(n_cells=("nnz", "size"), n_empty_rows=("nnz", lambda s: int((s == 0).sum())),
                 median_total_counts=("lib", "median"))
            .reset_index())
    diag["frac_empty"] = diag["n_empty_rows"] / diag["n_cells"]
    print("\ngaydosik2023 (D14), every arm:")
    print(diag.to_string(index=False))

    d14 = cells[cells["study"].eq("gaydosik2023") & cells["arm"].eq(HC)]
    assert set(d14["donor"]) <= set(long.loc[long["arm"].eq(HC), "donor"]), "a D14 HC donor is " \
        "not in the published HC arm"

    with h5py.File(D14_SRC, "r") as h5:
        src_names = pd.Index(CM._read_elem(h5["obs"][h5["obs"].attrs["_index"]])).astype(str)
        src_var = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        sobs = CM.read_obs(h5, cols=["total_counts", "n_genes"])
        pos = pd.Series(np.arange(len(src_names)), index=src_names)
        take = pos.reindex(d14.index)
        assert take.notna().all(), f"{int(take.isna().sum())} D14 cells absent from {D14_SRC.name}"
        take = take.to_numpy(int)

        # the atlas obs must be the same cells, or the repair is grafting the wrong counts on
        lib_src = pd.to_numeric(sobs["total_counts"], errors="coerce").to_numpy(float)[take]
        ng_src = pd.to_numeric(sobs["n_genes"], errors="coerce").to_numpy(float)[take]
        same_lib = int(np.sum(lib_src == d14["lib"].to_numpy()))
        print(f"\nD14 CD4_HC cells matched into {D14_SRC.name}: {len(take):,}; "
              f"total_counts identical in {same_lib:,} of {len(take):,}")
        assert same_lib == len(take), "total_counts differs between the two objects"

        gi = src_var.get_indexer(genes_all)
        unmeasured = [g for g, i in zip(genes_all, gi) if i < 0]
        g = h5["layers"]["raw_counts"]
        sip = g["indptr"][:].astype(np.int64)
        lookup = -np.ones(len(src_var), dtype=np.int64)
        lookup[gi[gi >= 0]] = np.arange(int((gi >= 0).sum()))
        X = np.zeros((len(take), int((gi >= 0).sum())), dtype=np.float32)
        rowsum = np.zeros(len(take))
        nonint = 0
        for r, i in enumerate(take):
            a, b = int(sip[i]), int(sip[i + 1])
            if b == a:
                continue
            idx = g["indices"][a:b]
            dat = g["data"][a:b]
            rowsum[r] = float(dat.sum())
            nonint += int(np.sum(dat != np.rint(dat)))
            p = lookup[idx]
            keep = p >= 0
            X[r, p[keep]] = dat[keep].astype(np.float32)
        assert nonint == 0, f"{nonint} non-integer entries -- that layer is not raw counts"
        frac = rowsum / np.maximum(d14["lib"].to_numpy(float), 1)
        print(f"recovered counts vs the atlas obs.total_counts: row sum / total_counts in "
              f"[{frac.min():.4f}, {frac.max():.4f}], median {np.median(frac):.4f} "
              "(< 1 only by the var entries the standardisation dropped)")
        assert frac.max() <= 1.0 + 1e-9 and np.median(frac) > 0.95, \
            "recovered counts do not reconcile with obs.total_counts"
        # The source rows must be non-empty counts. `n_genes` was written before the
        # standardisation dropped a few var entries, so it is an upper bound on the row nnz,
        # not an identity -- 19,033 of the 20,630 D14 cells match it exactly and none exceeds it.
        nnz_src = np.diff(sip)[take]
        assert (nnz_src > 0).all(), f"{int((nnz_src == 0).sum())} source rows are also empty"
        assert (nnz_src <= ng_src).all(), "source row nnz exceeds obs.n_genes -- wrong layer"
        print(f"source rows: nnz in [{nnz_src.min()}, {nnz_src.max()}], equal to obs.n_genes in "
              f"{int((nnz_src == ng_src).sum()):,} of {len(take):,} cells "
              f"(max shortfall {int((ng_src - nnz_src).max())} genes, dropped by the "
              "standardisation after n_genes was written)")

    kept = [g for g, i in zip(genes_all, gi) if i >= 0]
    lib = d14["lib"].to_numpy(float)
    lib[lib <= 0] = 1.0
    L = np.log1p(X / lib[:, None] * 1e4)

    rows = []
    for donor, idx in d14.reset_index(drop=True).groupby("donor").groups.items():
        i = np.asarray(idx)
        blk, lblk = X[i], L[i]
        rec = pd.DataFrame({
            "arm": HC, "donor": donor, "gene": kept, "n_cells": len(i),
            "pct_expr": (blk > 0).mean(0), "mean_all": lblk.mean(0),
            "sd_all": lblk.std(0, ddof=1) if len(i) > 1 else np.zeros(blk.shape[1]),
        })
        if unmeasured:
            rec = pd.concat([rec, pd.DataFrame({
                "arm": HC, "donor": donor, "gene": unmeasured, "n_cells": len(i),
                "pct_expr": np.nan, "mean_all": np.nan, "sd_all": np.nan})], ignore_index=True)
        rows.append(rec)
    rep = pd.concat(rows, ignore_index=True)
    print(f"repaired {rep['donor'].nunique()} CD4_HC donor-arms x {len(genes_all):,} screened "
          f"genes. {len(unmeasured)} of those genes are absent from D14's own var (mostly "
          "olfactory receptors) and\n  are genuinely unmeasured in these five donors -> NaN, "
          "never 0. They were structural zeros before the repair.")
    return rep, diag, unmeasured


def apply_repair(long, rep):
    """Published HC rows of the repaired donors replaced; every other row untouched."""
    out = long.copy()
    key = pd.MultiIndex.from_frame(out[["arm", "donor", "gene"]])
    rkey = pd.MultiIndex.from_frame(rep[["arm", "donor", "gene"]])
    hit = key.isin(rkey)
    assert int(hit.sum()) == len(rep), (int(hit.sum()), len(rep))
    r = rep.set_index(["arm", "donor", "gene"])
    for c in ("pct_expr", "mean_all", "sd_all"):
        out.loc[hit, c] = r[c].reindex(key[hit]).to_numpy()
    # the builder's derived columns are not repaired; drop them so nothing downstream reads a
    # stale value for these five donors.
    for c in ("pct_expr_clean", "mean_all_clean", "mean_all_ds", "mean_all_hvg", "mean_pos"):
        if c in out.columns:
            out.loc[hit, c] = np.nan
    return out


# --------------------------------------------------------------------------- depth matching
def depth_matched(cells, genes_all, cohort_donors):
    """`pct_expr` recomputed after thinning every cell to DS_DEPTH, malignant and reactive arms.

    One pass per 128-gene block over the skin T object, the builder's `read_block` and `thin`,
    one rng seeded at SEED and consumed in block order.
    """
    cfg = Q.TISSUE[TISSUE]
    rng = np.random.default_rng(SEED)
    lib = cells["lib"].to_numpy(float)
    lib[~np.isfinite(lib) | (lib <= 0)] = 1.0
    ds_ok = lib >= DS_DEPTH
    ds_p = np.where(ds_ok, DS_DEPTH / np.maximum(lib, 1), 0.0)
    print(f"\ndepth matching at {DS_DEPTH:,} UMI, seed {SEED}: "
          f"{ds_ok.mean():.1%} of all {len(cells):,} cells reach it")

    sel = cells[cells["arm"].isin([MALIG, REACT]) & cells["donor"].isin(cohort_donors)]
    groups = {k: v["row"].to_numpy() for k, v in sel.groupby(["arm", "donor"])}
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_CELLS}
    print(f"{len(groups):,} (arm, donor) groups in the two CD4 arms at >= {MIN_CELLS} cells")

    acc = []
    with h5py.File(cfg["obj"], "r") as h5:
        var = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        gi = var.get_indexer(genes_all)
        assert (gi >= 0).all(), [g for g, i in zip(genes_all, gi) if i < 0][:10]
        g = h5["layers"][cfg["layer"]]
        n_obs = len(cells)
        nblk = (len(gi) + Q.BLOCK - 1) // Q.BLOCK
        for bi, start in enumerate(range(0, len(gi), Q.BLOCK), 1):
            cols = gi[start:start + Q.BLOCK]
            names = genes_all[start:start + Q.BLOCK]
            X = Q.read_block(g, cols, n_obs)
            D = Q.thin(X, ds_p, rng)
            Lds = np.log1p(D / DS_DEPTH * 1e4)
            for (arm, donor), idx in groups.items():
                d = idx[ds_ok[idx]]
                acc.append(pd.DataFrame({
                    "arm": arm, "donor": donor, "gene": names, "n_cells_ds": len(d),
                    "pct_expr_ds": ((D[d] > 0).mean(0) if len(d)
                                    else np.full(len(names), np.nan)),
                    "mean_all_ds_check": (Lds[d].mean(0) if len(d)
                                          else np.full(len(names), np.nan))}))
            del X, D, Lds
            print(f"  block {bi}/{nblk}  ({start + len(cols):,}/{len(gi):,} genes)", flush=True)
    out = pd.concat(acc, ignore_index=True)
    print(f"depth-matched frame: {out.shape}; "
          f"{int(out['pct_expr_ds'].isna().sum()):,} NaN (no cell of the donor-arm reached "
          f"{DS_DEPTH:,} UMI)")
    return out


# --------------------------------------------------------------------------- the four tests
def paired_counts(long, value):
    """n_up / n_down / n_tie of the per-donor difference, beside the test that uses them."""
    w = (long[long["arm"].isin([MALIG, REACT])]
         .pivot_table(index=["gene", "donor"], columns="arm", values=value))
    w = w.dropna(subset=[MALIG, REACT])
    d = w[MALIG] - w[REACT]
    return (pd.DataFrame({"n_up": (d > 0).astype(int), "n_down": (d < 0).astype(int),
                          "n_tie": (d == 0).astype(int)})
            .groupby(level="gene").sum().reset_index())


def vs_reference(long, arm_pos, arm_ref, value="mean_all", min_donors=MIN_DONORS):
    """Unpaired donor-level Mann-Whitney, malignant vs healthy.

    Copied from notebook 49 section 3 (`vs_reference`) because a notebook exposes no importable
    function; it is byte-for-byte the same estimator, with `n_donors_hc` added.
    """
    d = long[long["arm"].isin([arm_pos, arm_ref])]
    rows = []
    for gene, g in d.groupby("gene"):
        x = g.loc[g["arm"].eq(arm_pos), value].dropna().to_numpy()
        y = g.loc[g["arm"].eq(arm_ref), value].dropna().to_numpy()
        if len(x) < min_donors or len(y) < min_donors:
            continue
        p = (stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
             if np.ptp(np.r_[x, y]) > 0 else 1.0)
        rows.append({"gene": gene,
                     "log2fc_vs_hc": np.log2((np.median(x) + PSEUDO) / (np.median(y) + PSEUDO)),
                     "p_vs_hc": p, "n_donors_hc": len(y)})
    out = pd.DataFrame(rows)
    out["q_vs_hc"] = multipletests(out["p_vs_hc"], method="fdr_bh")[1]
    return out


def all_tests(long, ds):
    """The four per-gene tests, each over the full 2,355-gene universe so every BH matches."""
    mean = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS,
                                 pseudo=PSEUDO).merge(paired_counts(long, "mean_all"), on="gene")
    det = TS.paired_selectivity(long, MALIG, REACT, value="pct_expr", min_donors=MIN_DONORS,
                                pseudo=PSEUDO)

    dsl = long.merge(ds[["arm", "donor", "gene", "pct_expr_ds"]], on=["arm", "donor", "gene"],
                     how="left")
    dsl = dsl.dropna(subset=["pct_expr_ds"]).assign(pct_expr=lambda d: d["pct_expr_ds"])
    dm = TS.paired_selectivity(dsl, MALIG, REACT, value="pct_expr", min_donors=MIN_DONORS,
                               pseudo=PSEUDO)

    hc = vs_reference(long, MALIG, HC)
    return mean, det, dm, hc


def build_gene_stats(genes, role, mean, det, dm, hc, surf_all):
    out = pd.DataFrame({"gene": genes})
    out["gene_set"] = out["gene"].map(role)
    out = out.merge(surf_all[["gene", "surface_confident"]], on="gene", how="left")
    out["surface_confident"] = out["surface_confident"].fillna(False).astype(bool)
    out["disputed_in_42"] = out["gene"].isin(TQ.SUSPECTS)

    out = out.merge(mean[["gene", "log2fc_paired", "p", "q", "n_up", "n_down", "n_tie",
                          "n_donors_paired"]]
                    .rename(columns={"log2fc_paired": "log2fc_paired_mean", "p": "p_mean",
                                     "q": "q_mean", "n_up": "n_up_mean", "n_down": "n_down_mean",
                                     "n_tie": "n_tie_mean",
                                     "n_donors_paired": "n_donors_paired_mean"}),
                    on="gene", how="left")
    out = out.merge(det[["gene", "log2fc_paired", "p", "q", "n_donors_paired"]]
                    .rename(columns={"log2fc_paired": "log2fc_detect", "p": "p_detect",
                                     "q": "q_detect",
                                     "n_donors_paired": "n_donors_paired_detect"}),
                    on="gene", how="left")
    out = out.merge(dm[["gene", "log2fc_paired", "p", "q", "n_donors_paired"]]
                    .rename(columns={"log2fc_paired": "log2fc_detect_depthmatched",
                                     "p": "p_detect_depthmatched",
                                     "q": "q_detect_depthmatched",
                                     "n_donors_paired": "n_donors_paired_depthmatched"}),
                    on="gene", how="left")
    out = out.merge(hc[["gene", "log2fc_vs_hc", "p_vs_hc", "q_vs_hc", "n_donors_hc"]],
                    on="gene", how="left")
    for c in out.columns:
        if c.startswith(("n_up", "n_down", "n_tie", "n_donors")):
            out[c] = out[c].astype("Int64")
    return out.sort_values(["gene_set", "log2fc_paired_mean"], ascending=[True, False],
                           ignore_index=True)


# --------------------------------------------------------------------------- the long export
def build_donor_long(long, meta, ds, genes, role):
    """One row per (donor_key, arm, gene) the analysis could have evaluated. Absent -> NaN."""
    donors = sorted(set(long["donor"]))
    idx = pd.MultiIndex.from_product([donors, CD4_ARMS, genes],
                                     names=["donor", "arm", "gene"])
    base = long[long["gene"].isin(set(genes))].set_index(["donor", "arm", "gene"])
    assert base.index.is_unique, "duplicate (donor, arm, gene) in the long frame"
    out = base[["n_cells", "pct_expr", "mean_all"]].reindex(idx).reset_index()

    m = meta.set_index(["donor", "arm"])
    out["study"] = out["donor"].map(m.groupby(level="donor")["study"].first())
    assert out["study"].notna().all(), "a donor key has no study"
    out["median_total_counts"] = m["median_total_counts"].reindex(
        pd.MultiIndex.from_frame(out[["donor", "arm"]])).to_numpy()
    out["gene_set"] = out["gene"].map(role)

    d = ds.set_index(["donor", "arm", "gene"])["pct_expr_ds"]
    out["pct_expr_depthmatched"] = d.reindex(
        pd.MultiIndex.from_frame(out[["donor", "arm", "gene"]])).to_numpy()
    out.loc[~out["arm"].isin([MALIG, REACT]), "pct_expr_depthmatched"] = np.nan

    out = out.rename(columns={"donor": "donor_key", "mean_all": "mean_all"})
    out["n_cells"] = out["n_cells"].astype("Int64")
    out["passes_min_cells"] = out["n_cells"] >= MIN_CELLS      # NA where the arm is absent

    cols = ["study", "donor_key", "arm", "gene", "gene_set", "n_cells", "median_total_counts",
            "mean_all", "pct_expr", "pct_expr_depthmatched", "passes_min_cells"]
    return out[cols].sort_values(["arm", "donor_key", "gene"], ignore_index=True)


# --------------------------------------------------------------------------- verification
def _prev(long, arm, tag):
    return (TS.prevalence(long, arm, detect_floor=DETECT_FLOOR)
            .rename(columns={"median_pct_expr": f"{tag}_median_pct",
                             "mean_pct_expr": f"{tag}_mean_pct",
                             "median_mean_all": f"{tag}_median_mean",
                             "frac_donors_positive": f"{tag}_frac_donors_pos",
                             "n_donors": f"{tag}_n_donors"}))


def _res_frame(long, mean, det):
    """The §3 `res` table, rebuilt from a long frame: everything §3b and §5b print."""
    hc = vs_reference(long, MALIG, HC)
    return (mean.rename(columns={"p": "p", "q": "q"})
            .merge(det[["gene", "log2fc_paired", "frac_donors_up", "p", "q"]]
                   .rename(columns={"log2fc_paired": "log2fc_detect",
                                    "frac_donors_up": "frac_donors_up_detect",
                                    "p": "p_detect", "q": "q_detect"}), on="gene", how="outer")
            .merge(_prev(long, MALIG, "malig"), on="gene", how="outer")
            .merge(_prev(long, REACT, "react"), on="gene", how="outer")
            .merge(_prev(long, HC, "hc"), on="gene", how="outer")
            .merge(hc, on="gene", how="outer"))


def _detail_genes(short_pub, genes_present):
    return list(dict.fromkeys(short_pub["gene"].head(30).tolist()
                              + [g for g in ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS
                                 if g in genes_present]))


def _summ_6e(long, detail):
    lfc = TH.lfc_matrix(long, MALIG, REACT, genes=detail, pseudo=PSEUDO,
                        min_donors=MIN_DONORS).T
    L = lfc.reindex(columns=[g for g in detail if g in lfc.columns])
    n = L.notna().sum()
    s = pd.DataFrame({"n_donors": n, "median": L.median(), "q25": L.quantile(.25),
                      "q75": L.quantile(.75), "min": L.min(), "max": L.max(),
                      "frac_up": (L > 0).sum() / n, "frac_up_1": (L > 1).sum() / n,
                      "frac_down_1": (L < -1).sum() / n}).reindex(L.columns)
    s.index.name = "gene"
    return s.reset_index()


def _elig_6f(long, detail, paired):
    thr = (0.10, 0.25, 0.50, 0.75)
    e = pd.concat([
        TH.eligibility(long, MALIG, thresholds=thr, genes=detail, donors=paired).assign(arm=MALIG),
        TH.eligibility(long, REACT, thresholds=thr, genes=detail, donors=paired).assign(arm=REACT),
        TH.eligibility(long, HC, thresholds=thr, genes=detail).assign(arm=HC)])
    return e[["gene", "arm", "n_donors", "median_pct_expr"]
             + [f"frac_donors_ge{int(x * 100)}" for x in thr]]


def verify_published(long_pub, long_fix, mean, det, paired):
    """Every section 3b / 5b / 6e / 6f value, recomputed from the exported long frame."""
    hr("VERIFICATION 1 -- sections 3b, 5b, 6e, 6f recomputed from the long frame")
    all_pub = pd.read_csv(TAB_DIR / "tgt49_all_genes.csv")
    short_pub = pd.read_csv(TAB_DIR / "tgt49_shortlist.csv")
    det_pub = pd.read_csv(TAB_DIR / "tgt49_detail_summary.csv")
    elig_pub = pd.read_csv(TAB_DIR / "tgt49_eligibility.csv")
    detail = _detail_genes(short_pub, set(all_pub["gene"]))

    res_pub = _res_frame(long_pub, mean, det)       # HC arm exactly as published (broken)
    res_fix = _res_frame(long_fix, mean, det)       # HC arm repaired

    # --- 3b: surface genes up at q < 0.05
    cols_3b = ["log2fc_paired", "frac_donors_up", "p", "q", "log2fc_detect", "q_detect",
               "malig_median_pct", "react_median_pct", "malig_frac_donors_pos"]
    cols_hc = ["hc_median_pct", "log2fc_vs_hc", "p_vs_hc", "q_vs_hc"]
    def sub(res, pub, cols):
        """`res` restricted to the genes the published table prints, in its column set."""
        return res[["gene"] + cols].merge(pub[["gene"]], on="gene")

    up = all_pub[all_pub["gene_set"].eq("surface") & (all_pub["q"] < SHORT_Q)
                 & (all_pub["log2fc_paired"] > 0)]
    print(f"\nsection 3b -- {len(up)} surface genes up at q < {SHORT_Q}")
    dev_of(sub(res_pub, up, cols_3b), up[["gene"] + cols_3b], cols_3b, ["gene"],
           "3b, columns that do not involve the healthy arm")
    dev_of(sub(res_pub, up, cols_hc), up[["gene"] + cols_hc], cols_hc, ["gene"],
           "3b, healthy-arm columns, HC as published")

    # --- 5b: the shortlist
    cols_5b = cols_3b + ["react_frac_donors_pos"]
    cols_5bhc = cols_hc + ["hc_frac_donors_pos"]
    print(f"\nsection 5b -- the {len(short_pub)} shortlist genes")
    dev_of(sub(res_pub, short_pub, cols_5b), short_pub[["gene"] + cols_5b], cols_5b, ["gene"],
           "5b, columns that do not involve the healthy arm")
    dev_of(sub(res_pub, short_pub, cols_5bhc), short_pub[["gene"] + cols_5bhc], cols_5bhc,
           ["gene"], "5b, healthy-arm columns, HC as published")

    # --- 6e / 6f
    print(f"\nsections 6e and 6f -- the {len(detail)} detail genes")
    s_cols = ["n_donors", "median", "q25", "q75", "min", "max", "frac_up", "frac_up_1",
              "frac_down_1"]
    dev_of(_summ_6e(long_pub, detail), det_pub[["gene"] + s_cols], s_cols, ["gene"],
           "6e, per-gene summary of the donor effects")
    e_cols = ["n_donors", "median_pct_expr", "frac_donors_ge10", "frac_donors_ge25",
              "frac_donors_ge50", "frac_donors_ge75"]
    e_pub = elig_pub[["gene", "arm"] + e_cols]
    e_new = _elig_6f(long_pub, detail, paired)
    dev_of(e_new[e_new["arm"].isin([MALIG, REACT])], e_pub[e_pub["arm"].isin([MALIG, REACT])],
           e_cols, ["gene", "arm"], "6f, eligibility, malignant and reactive rows")
    dev_of(e_new[e_new["arm"].eq(HC)], e_pub[e_pub["arm"].eq(HC)], e_cols, ["gene", "arm"],
           "6f, eligibility, healthy rows, HC as published")

    # --- what the repair moves, per table. Not a failure: it is the fix.
    hr("VERIFICATION 1b -- what the healthy-donor repair changes, per table")
    for label, pub_rows, cols in (("3b healthy columns", up, cols_hc),
                                  ("5b healthy columns", short_pub, cols_5bhc)):
        m = sub(res_fix, pub_rows, cols).merge(sub(res_pub, pub_rows, cols), on="gene",
                                               suffixes=("_fix", "_pub"))
        print(f"\n{label} ({len(m)} genes):")
        for c in cols:
            d = (pd.to_numeric(m[f"{c}_fix"], errors="coerce")
                 - pd.to_numeric(m[f"{c}_pub"], errors="coerce")).abs()
            print(f"  {c:<18} changed in {int((d > TOL).sum()):>3} of {len(m)} genes; "
                  f"max |delta| = {d.max():.4g}")
    hcn = {"published": int(res_pub["hc_n_donors"].max()),
           "repaired": int(res_fix["hc_n_donors"].max())}
    print(f"\nCD4_HC donors: {hcn['published']} before the fix, {hcn['repaired']} after -- "
          "the four zero donors are repaired, not dropped, so the count is unchanged; "
          "what changes is that five of them now carry real values.")
    sb = (short_pub[["gene", "hc_median_pct"]]
          .merge(res_fix[["gene", "hc_median_pct"]], on="gene", suffixes=("_pub", "_fix")))
    sb["delta"] = sb["hc_median_pct_fix"] - sb["hc_median_pct_pub"]
    print("\nsection 5b `hc_median_pct`, published vs repaired, all 23 shortlist genes:")
    print(sb.to_string(index=False))

    e_fix = _elig_6f(long_fix, detail, paired)
    e_m = (e_fix[e_fix["arm"].eq(HC)][["gene", "median_pct_expr", "frac_donors_ge10"]]
           .merge(e_pub[e_pub["arm"].eq(HC)][["gene", "median_pct_expr", "frac_donors_ge10"]],
                  on="gene", suffixes=("_fix", "_pub")))
    moved = e_m[(e_m["median_pct_expr_fix"] - e_m["median_pct_expr_pub"]).abs() > TOL]
    print(f"\nsection 6f healthy rows: {len(moved)} of {len(e_m)} detail genes move "
          f"(median_pct_expr); max |delta| = "
          f"{(e_m['median_pct_expr_fix'] - e_m['median_pct_expr_pub']).abs().max():.4g}")
    return res_pub, res_fix, detail, short_pub, all_pub


def verify_pairing(dl, genes):
    hr("VERIFICATION 2 -- donors with a non-NaN value in BOTH CD4 arms, per gene")
    w = (dl[dl["arm"].isin([MALIG, REACT])]
         .pivot_table(index=["gene", "donor_key"], columns="arm", values="mean_all"))
    both = (w.dropna(subset=[MALIG, REACT]).groupby(level="gene").size()
            .rename("n_donors_both_cd4_arms").reset_index())
    vc = both["n_donors_both_cd4_arms"].value_counts().sort_index()
    print("  " + "; ".join(f"{n} donors -> {k} gene(s)" for n, k in vc.items()))
    if both["n_donors_both_cd4_arms"].nunique() == 1:
        n = int(both["n_donors_both_cd4_arms"].iloc[0])
        print(f"\n  All {len(genes):,} exported genes have exactly {n} paired donors. That is "
              "uniform because the integrated object's `var` is a 42,347-gene UNION over 14\n"
              "  cohorts whose own objects carry 15,790-32,496 genes each: a gene a cohort never "
              "measured is read back as a column of zeros and counted as a real zero.\n"
              "  `jobs/run_payload_screen.py` builds the (study, gene) coverage mask that fixes "
              "this and finds 342 genes unmeasured in li2024 alone, which is 31 of these\n"
              f"  {n} donors. No mask is applied here, by decision: it would change the screen, "
              "and this export has to reproduce the published tables to 1e-6.")
    w2 = (dl[dl["arm"].isin([MALIG, REACT])]
          .pivot_table(index=["gene", "donor_key"], columns="arm", values="pct_expr_depthmatched"))
    b2 = (w2.dropna(subset=[MALIG, REACT]).groupby(level="gene").size()
          .rename("n").reset_index())
    print(f"\n  under depth matching: min {int(b2['n'].min())}, median "
          f"{int(b2['n'].median())}, max {int(b2['n'].max())} paired donors per gene "
          f"(a donor-arm drops out when no cell of it reaches {DS_DEPTH:,} UMI).")
    return both


def verify_nan(dl, genes, unmeasured):
    hr("VERIFICATION 3 -- NaN versus 0 in tgt49_donor_long.csv, by reason")
    n = len(dl)
    arm_absent = dl["n_cells"].isna()
    unm = (~arm_absent) & dl["pct_expr"].isna()
    zero = (~arm_absent) & (~unm) & (dl["pct_expr"] == 0)
    pos = (~arm_absent) & (~unm) & (dl["pct_expr"] > 0)
    print(f"{n:,} rows = {dl['donor_key'].nunique()} donors x {dl['arm'].nunique()} arms x "
          f"{dl['gene'].nunique():,} genes")
    hit = sorted(set(unmeasured) & set(genes))
    for lab, m in (("NaN  - arm absent in that donor, or below the 20-cell floor", arm_absent),
                   (f"NaN  - gene not in D14's own var, repaired donors: {hit}", unm),
                   ("0    - measured, genuinely zero in every cell of the arm", zero),
                   ("> 0  - measured and detected", pos)):
        print(f"  {lab:<66} {int(m.sum()):>9,}  ({m.mean():6.2%})")
    assert int((arm_absent | unm | zero | pos).sum()) == n, "row classes do not partition"
    print(f"\npasses_min_cells: {int(dl['passes_min_cells'].eq(False).sum()):,} rows False. "
          "The builder applied the floor before writing the long table, so a donor-arm below 20 "
          "cells\n  is absent from the published frame and cannot be recovered without re-running "
          "it; the column is carried as the record of the rule, not as a live filter.")
    dm = dl[dl["arm"].isin([MALIG, REACT])]
    print(f"\npct_expr_depthmatched: non-NaN on {int(dm['pct_expr_depthmatched'].notna().sum()):,}"
          f" of {len(dm):,} CD4_malignant/CD4_reactive rows; NaN by construction on all "
          f"{len(dl) - len(dm):,} CD4_HC rows.")


def verify_ds_sanity(ds, long_pub):
    hr("VERIFICATION 4 -- the depth-matched pass against the builder's own mean_all_ds")
    m = (long_pub[["arm", "donor", "gene", "n_cells_ds", "mean_all_ds"]]
         .merge(ds, on=["arm", "donor", "gene"], how="inner",
                suffixes=("_builder", "_here")))
    nc = m.drop_duplicates(["arm", "donor"])
    same = int((nc["n_cells_ds_builder"] == nc["n_cells_ds_here"]).sum())
    print(f"  cells above the {DS_DEPTH:,} UMI floor: identical in {same} of {len(nc)} "
          "donor-arms (this is deterministic and must be exact)")
    if same != len(nc):
        FAILURES.append(f"n_cells_ds differs in {len(nc) - same} donor-arms")
    m = m.dropna(subset=["mean_all_ds", "mean_all_ds_check"]).copy()
    m["signed"] = m["mean_all_ds_check"] - m["mean_all_ds"]
    d = m["signed"].abs()
    r = float(np.corrcoef(m["mean_all_ds"], m["mean_all_ds_check"])[0, 1])
    bias = float(m["signed"].mean())
    print(f"  mean_all_ds: Pearson r = {r:.6f}, mean SIGNED delta = {bias:+.3e}, "
          f"median |delta| = {d.median():.4g}, max |delta| = {d.max():.4g}, "
          f"over {len(m):,} values.")
    print("\n  These are two INDEPENDENT binomial draws of the same thinning -- the builder's rng "
          "walked 2,355 genes over all 539,916 cells in a different order -- so they\n"
          "  agree to Monte-Carlo error, not to 1e-6. The test is therefore (a) no BIAS: a wrong "
          "depth, a wrong library size or a wrong gene index shifts the mean; and\n"
          "  (b) the scatter must shrink with the number of cells averaged, the way sampling "
          "noise does and a coding error does not.")
    q = pd.qcut(m["n_cells_ds_here"], 5, duplicates="drop")
    tab = m.groupby(q, observed=True).agg(n_values=("signed", "size"),
                                          median_cells=("n_cells_ds_here", "median"),
                                          mean_signed=("signed", "mean"),
                                          rms=("signed", lambda s: float(np.sqrt((s ** 2).mean()))))
    print("\n  by donor-arm size (cells above the depth floor):")
    print("    " + tab.to_string().replace("\n", "\n    "))
    if r < 0.99:
        FAILURES.append(f"depth-matched pass disagrees with the builder: r = {r:.6f}")
    if abs(bias) > 2e-3:
        FAILURES.append(f"depth-matched pass is biased against the builder: {bias:+.3e}")


def verify_kir3dl2(dl, stats_tab, don):
    hr("VERIFICATION 5 -- KIR3DL2: log2fc_paired = 0 with q = 9.5e-4 and a zero malignant median")
    print(stats_tab[stats_tab["gene"].eq("KIR3DL2")].T.to_string())
    w = (dl[dl["gene"].eq("KIR3DL2") & dl["arm"].isin([MALIG, REACT])]
         .pivot_table(index="donor_key", columns="arm",
                      values=["mean_all", "pct_expr", "pct_expr_depthmatched"]))
    w.columns = [f"{a}_{b.replace('CD4_', '')}" for a, b in w.columns]
    w = w.dropna(subset=["mean_all_malignant", "mean_all_reactive"])
    w["d_mean"] = w["mean_all_malignant"] - w["mean_all_reactive"]
    w["d_pct"] = w["pct_expr_malignant"] - w["pct_expr_reactive"]
    w = w.join(don[["study", "dx2", "burden"]]).sort_values("pct_expr_malignant",
                                                            ascending=False)
    print(f"\nall {len(w)} paired donors, sorted by malignant detection:")
    print(w.to_string())
    p = w["pct_expr_malignant"]
    print(f"\nmalignant pct_expr: {int((p >= 0.10).sum())}/{len(p)} donors >= 10% "
          f"({(p >= 0.10).mean():.0%}), {int((p >= 0.50).sum())}/{len(p)} >= 50% "
          f"({(p >= 0.50).mean():.0%}), median {p.median():.4g}, "
          f"{int((p == 0).sum())} donors at exactly 0.")
    print(f"paired differences on mean_all: {int((w['d_mean'] > 0).sum())} up, "
          f"{int((w['d_mean'] < 0).sum())} down, {int((w['d_mean'] == 0).sum())} tied at 0.")
    print("\n  The signed-rank test ranks |difference| and is driven by the many small negative "
          "differences in the KIR3DL2-negative majority; the median log2 ratio is 0 because\n"
          "  more than half the donors sit at or near zero in both arms. Neither number "
          "describes a bimodal antigen that is absent in two thirds of patients and dominant in\n"
          "  a third. The quantity to quote for KIR3DL2 is the per-donor prevalence above -- a "
          "biomarker-selected fraction, which is also how lacutamab is given.")


def verify_depthmatched_shortlist(stats_tab, short_pub):
    hr("VERIFICATION 6 -- do the 23 shortlist genes survive the depth-matched detection test?")
    s = stats_tab[stats_tab["gene"].isin(short_pub["gene"])].copy()
    s["raw_sig"] = (s["q_detect"] < SHORT_Q) & (s["log2fc_detect"] > 0)
    s["dm_sig"] = (s["q_detect_depthmatched"] < SHORT_Q) & (s["log2fc_detect_depthmatched"] > 0)
    s["same_sign"] = (np.sign(s["log2fc_detect"]) == np.sign(s["log2fc_detect_depthmatched"]))
    cols = ["gene", "log2fc_detect", "q_detect", "log2fc_detect_depthmatched",
            "q_detect_depthmatched", "n_donors_paired_depthmatched", "raw_sig", "dm_sig",
            "same_sign"]
    print(s[cols].sort_values("log2fc_detect_depthmatched", ascending=False).to_string(index=False))
    both = int((s["raw_sig"] & s["dm_sig"] & s["same_sign"]).sum())
    print(f"\n  {int(s['raw_sig'].sum())} of {len(s)} shortlist genes are significant and up on "
          f"the RAW detection test;\n  {both} of {len(s)} stay significant with the same sign "
          f"under the depth-matched test ({both / len(s):.0%}).")
    lost = s.loc[s["raw_sig"] & ~(s["dm_sig"] & s["same_sign"]), "gene"].tolist()
    print(f"  lost under depth matching: {lost or 'none'}")
    m = stats_tab["gene_set"].eq("shortlist")
    print(f"\n  for reference, the mean-expression test (the gate the shortlist actually uses) "
          f"holds for {int((stats_tab.loc[m, 'q_mean'] < SHORT_Q).sum())} of {int(m.sum())}.")


# --------------------------------------------------------------------------- main
def main():
    hr(f"tgt49 donor-level export -- seed {SEED}, depth {DS_DEPTH:,} UMI, "
       f"cell floor {MIN_CELLS} recorded not applied")
    long_pub, meta, don, paired, man = load_published()
    genes_all, surf_all = gene_universe(long_pub)
    all_pub = pd.read_csv(TAB_DIR / "tgt49_all_genes.csv")
    short_pub = pd.read_csv(TAB_DIR / "tgt49_shortlist.csv")
    genes, role = export_genes(all_pub, short_pub, set(genes_all))

    hr("THE HEALTHY-DONOR DIAGNOSIS")
    cells = cell_arms(long_pub)
    rep, diag, unmeasured = repair_d14(long_pub, cells, genes_all)
    long_fix = apply_repair(long_pub, rep)

    hr("DEPTH-MATCHED DETECTION")
    ds = depth_matched(cells, genes_all, set(long_pub["donor"]))

    hr("TESTS")
    mean, det, dm, hc = all_tests(long_fix, ds)
    print(f"mean test: {len(mean):,} genes; raw detection: {len(det):,}; "
          f"depth-matched detection: {len(dm):,}; vs healthy: {len(hc):,}")
    stats_tab = build_gene_stats(genes, role, mean, det, dm, hc, surf_all)
    dl = build_donor_long(long_fix, meta, ds, genes, role)

    f_long = TAB_DIR / "tgt49_donor_long.csv"
    f_stat = TAB_DIR / "tgt49_gene_stats.csv"
    dl.to_csv(f_long, index=False)
    stats_tab.to_csv(f_stat, index=False)
    print(f"\nwrote {f_long}  {dl.shape}")
    print(f"wrote {f_stat}  {stats_tab.shape}")

    # the mean test's n_up must be the frac_donors_up the published table prints
    fu = mean["n_up"] / mean["n_donors_paired"]
    check("n_up_mean / n_donors_paired_mean == frac_donors_up",
          float((fu - mean["frac_donors_up"]).abs().max()), len(mean))

    verify_published(long_pub, long_fix, mean, det, paired)
    verify_pairing(dl, genes)
    verify_nan(dl, genes, unmeasured)
    verify_ds_sanity(ds, long_pub)
    verify_kir3dl2(dl, stats_tab, don)
    verify_depthmatched_shortlist(stats_tab, short_pub)

    hr("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("every recomputation agrees with the published tables to within 1e-6.")


if __name__ == "__main__":
    main()
