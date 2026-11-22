#!/usr/bin/env python
"""The (arm, donor, gene) table for the surface-target screen, with the statistics that tell a
real antigen from differential contamination.

    python run_target_screen_qc.py --tissue blood   [--out-dir ...] [--max-cells N]
    python run_target_screen_qc.py --tissue skin

Supersedes `run_target_screen.py` in both compartments. Same single-pass CSR gene-block read,
same arms,
same groups -- four additions, each one a test the first ranking could not be subjected to:

1. ROBUST LOCATION. `mean_all` is a mean of log1p over every cell, so one doublet carrying 200
   LYZ counts moves it as much as 200 T cells carrying one. Added: `mean_pos`, `trim1_mean`,
   `trim10_mean`, `wins99_mean`, `wins90_mean`, `sd_all`, and `top1_share` / `top5_share` -- the fraction
   of the group's total counts for that gene held by its top 1% / 5% of cells. A signal carried
   by <1% of cells collapses under all of them; a broadly expressed antigen does not move.

2. A CLEAN CELL SET. Cells with any strict lineage marker of the compartment -- blood: LYZ,
   S100A8, S100A9, CD14, VCAN, FCN1, CD68, MNDA, PPBP, PF4, HBB, MS4A1, CD79A; skin: the
   keratins, KRTDAP, DMKN, SFN, the collagens, DCN, LUM, VWF, CDH5, CLDN5, PMEL, MLANA, TYRP1,
   DCT, LYZ, CD14, CD68, C1QA/B, CD163, the tryptases, CPA3, MS4A1, CD79A, JCHAIN, MZB1, HBB,
   HBA2 (`target_qc.LINEAGE_MODULES_BY_TISSUE`) -- or a doublet score in the top decile of their
   study are dropped, and the core three statistics recomputed on what is left. The drop rate is
   recorded per group, because under the contamination model it is itself higher in the
   malignant arm.

3. DEPTH MATCHING. Counts are binomially thinned to a common depth before normalisation, so a
   gene cannot score simply because the malignant arm is sequenced deeper. The same device as
   nb28 Part III.

4. PER-CELL COUPLINGS. Inside the malignant arm, per donor, the Spearman correlation of each
   gene with per-cell library size, doublet score and one module score per abundant contaminant
   (`target_qc.COUPLING_MODULES`: myeloid in blood; keratinocyte, fibroblast and myeloid in
   skin), plus the odds of detection in the top-UMI decile. Fisher-z averaged over donors.

Two corrections to the v1 job:

  - the CP10K denominator is `obs.total_counts` (the true 42k-gene library, median 4,127), not
    the sum over the 10,000 HVG columns (median 2,186). The HVG-capture ratio differs between a
    clean T cell and a myeloid doublet, which is exactly the confound under test. The old
    quantity is kept as `mean_all_hvg` so the change is auditable.
  - the meta table no longer carries groups below `--min-cells`, which the v1 job wrote.

The gene set is the screen panel UNION the lineage controls UNION the trafficking modules: the
controls are what makes the contamination null empirical rather than chosen, and the trafficking
genes are what nb45 regresses surface protein on.

CPU only, no GPU. Memory is one gene-block of cells x BLOCK genes, a few times over.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TARGET_SCREEN_THREADS", "4"))

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import h5py                              # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy import stats                  # noqa: E402

import cluster_malig_helpers as CM       # noqa: E402
import malignancy as MAL                 # noqa: E402
import surface_targets as ST             # noqa: E402
import target_cohort as TC               # noqa: E402
import target_qc as TQ                   # noqa: E402

OUT_DIR = NB_DIR / "data" / "targets"
MIN_CELLS = 20
BLOCK = 128                 # genes per pass; smaller than v1 because more arrays are held
DS_DEPTH = 2000             # thinning target, under the median total_counts of 4,127
DBL_Q = 0.90                # per-study doublet-score quantile above which a cell is not "clean"
SEED = 0

TISSUE = {
    "skin": dict(obj=CM.CONFIG["skin"]["TCR_OBJ"], layer="raw_counts",
                 celltype="cell_type_T", donor="donor"),
    "blood": dict(obj=CM.CONFIG["blood"]["TCR_OBJ"], layer="counts",
                  celltype="cell_type_T", donor="donor"),
}
QC_OBS = ["total_counts", "n_genes", "doublet_score", "pct_mito"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissue", choices=sorted(TISSUE), required=True)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--min-cells", type=int, default=MIN_CELLS)
    p.add_argument("--ds-depth", type=int, default=DS_DEPTH)
    p.add_argument("--strict-min-counts", type=int, default=2,
                   help="counts of ONE strict lineage marker that make a cell contaminated")
    p.add_argument("--strict-min-genes", type=int, default=2,
                   help="distinct strict lineage markers at >=1 count that do the same")
    p.add_argument("--genes", nargs="*", default=None, help="override the gene set (smoke test)")
    p.add_argument("--max-cells", type=int, default=None,
                   help="use only the first N cells (smoke test; never for a real run)")
    p.add_argument("--no-cluster", action="store_true", help="skip the per-cluster table")
    p.add_argument("--call", default=None, choices=[None, *MAL.CALLS],
                   help="malignancy call for the arms; default is the project headline "
                        "(mal_cluster). `mal_tcr` builds the arms from the clone, which is "
                        "independent of the transcriptome -- the control for the headline "
                        "call being a cluster label and the comparison being between those "
                        "same clusters.")
    return p.parse_args()


def read_block(g, cols, n_obs):
    """Columns `cols` of a CSR h5ad matrix group, as a dense (cells x len(cols)) float32 array.

    CSR is row-major, so a column slice means walking every row. The matrix is read once in row
    chunks and the wanted columns picked out of each chunk -- one pass over the data per block.
    """
    want = np.asarray(cols)
    lookup = -np.ones(int(g.attrs["shape"][1]), dtype=np.int64)
    lookup[want] = np.arange(len(want))
    out = np.zeros((n_obs, len(want)), dtype=np.float32)
    indptr = g["indptr"]
    step = 50_000
    for a in range(0, n_obs, step):
        b = min(a + step, n_obs)
        ip = indptr[a:b + 1].astype(np.int64)
        lo, hi = int(ip[0]), int(ip[-1])
        if hi == lo:
            continue
        idx = g["indices"][lo:hi]
        dat = g["data"][lo:hi]
        pos = lookup[idx]
        keep = pos >= 0
        if not keep.any():
            continue
        rows = np.repeat(np.arange(a, b), np.diff(ip))
        out[rows[keep], pos[keep]] = dat[keep].astype(np.float32)
    return out


def row_sums(g, n_obs):
    """Per-cell sum over every column of a CSR matrix group, without materialising it."""
    out = np.zeros(n_obs, dtype=np.float64)
    indptr = g["indptr"]
    for a in range(0, n_obs, 50_000):
        b = min(a + 50_000, n_obs)
        ip = indptr[a:b + 1].astype(np.int64)
        lo, hi = int(ip[0]), int(ip[-1])
        if hi == lo:
            continue
        dat = g["data"][lo:hi].astype(np.float64)
        out[a:b] = np.add.reduceat(dat, ip[:-1] - lo) * (np.diff(ip) > 0)
    return out


def thin(X, p, rng):
    """Binomial thinning of a dense count block to a common depth. Only non-zeros are drawn."""
    out = np.zeros_like(X)
    nz = np.nonzero(X)
    if len(nz[0]):
        out[nz] = rng.binomial(X[nz].astype(np.int64), np.clip(p[nz[0]], 0, 1))
    return out


# ------------------------------------------------------------------ per-group statistics
def group_stats(blk, lblk, names, trim_fracs=(0.01, 0.10), wins_q=(0.99, 0.90)):
    """Detection, robust location and concentration for one (arm, donor) x gene block.

    `blk` is raw counts, `lblk` the log1p CP10K of the same cells. Everything is computed down
    the cell axis, one value per gene.
    """
    n = blk.shape[0]
    pos = blk > 0
    npos = pos.sum(0)
    tot = blk.sum(0)
    srt = np.sort(lblk, axis=0)
    out = {
        "gene": names,
        "n_cells": n,
        "pct_expr": pos.mean(0),
        "pct_expr_ge2": (blk >= 2).mean(0),
        "mean_all": lblk.mean(0),
        "mean_pos": np.divide(np.where(pos, lblk, 0).sum(0), npos,
                              out=np.zeros(blk.shape[1], float), where=npos > 0),
        # sd over cells, so a donor's mean carries a standard error and the per-donor effects
        # can go into a random-effects meta-analysis with the sampling term separated out.
        "sd_all": lblk.std(0, ddof=1) if n > 1 else np.zeros(blk.shape[1]),
    }
    for f in trim_fracs:
        k = int(np.floor(n * f))                  # cells removed from the TOP
        out[f"trim{int(f * 100)}_mean"] = srt[:n - k].mean(0) if n - k > 0 else np.zeros(
            blk.shape[1])
    for q in wins_q:
        cap = srt[min(int(np.ceil(q * n)) - 1, n - 1)]
        out[f"wins{int(q * 100)}_mean"] = np.minimum(lblk, cap).mean(0)
    srt_c = np.sort(blk, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        for f in (0.01, 0.05):
            k = max(1, int(np.ceil(n * f)))
            out[f"top{int(f * 100)}_share"] = np.where(tot > 0, srt_c[n - k:].sum(0) / tot, np.nan)
    return out


def couplings(lblk, cov, names):
    """Spearman of each gene with each per-cell covariate, inside one donor's malignant arm.

    Ranked once per block rather than per gene -- 1,250 genes x 50 donors of
    `scipy.stats.spearmanr` would dominate the run.
    """
    n = lblk.shape[0]
    rg = stats.rankdata(lblk, axis=0)
    rg = rg - rg.mean(0)
    sd = np.sqrt((rg ** 2).sum(0))
    out = {"gene": names, "n_cells_cpl": n}
    for key, v in cov.items():
        rv = stats.rankdata(v)
        rv = rv - rv.mean()
        den = sd * np.sqrt((rv ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"r_{key}"] = np.where(den > 0, (rg * rv[:, None]).sum(0) / den, np.nan)
    return out


def _fisher_mean(df, col):
    z = np.arctanh(np.clip(df[col].to_numpy(float), -0.999999, 0.999999))
    z = z[np.isfinite(z)]
    return float(np.tanh(z.mean())) if len(z) else np.nan


def main():
    args = parse_args()
    cfg = TISSUE[args.tissue]
    CM.use_tissue(args.tissue)
    MAL.use_tissue(args.tissue)      # or blood would silently assemble the SKIN parquets
    TQ.use_tissue(args.tissue)       # which lineages contaminate this compartment
    call = args.call or MAL.HEADLINE
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    with h5py.File(cfg["obj"], "r") as h5:
        genes_all = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        obs = CM.read_obs(h5, cols=[cfg["donor"], "sample_id", "study", "disease",
                                    cfg["celltype"], "has_tcr"] + QC_OBS)
        if args.max_cells:
            obs = obs.iloc[:args.max_cells]
            log(f"SMOKE TEST: first {len(obs):,} cells only -- not a cohort result")
        n_obs = len(obs)
        log(f"{cfg['obj'].name}: {n_obs:,} cells x {len(genes_all):,} genes")
        if len(genes_all) < 20_000:
            log(f"WARNING: only {len(genes_all):,} genes -- an HVG subset. A gene absent from it "
                "is absent from the screen, which is not the same as unexpressed.")

        # ---- arms: the headline malignancy call, healthy CD4 kept separate
        mal = MAL.load_malignancy(obs.index, has_tcr=obs["has_tcr"].astype(bool).to_numpy(),
                                  verbose=True)
        lvl = pd.Series(np.asarray(MAL.cd4_level(obs[cfg["celltype"]], mal,
                                                 call=call)).astype(str), index=obs.index)
        log(f"malignancy call: {call} "
            f"({mal[call].notna().mean():.1%} of cells defined)"
            if call in mal else f"malignancy call: {call}")
        is_cd4 = obs[cfg["celltype"]].astype(str).eq("CD4").to_numpy()
        # The atlas `disease` column calls the five geskin healthy donors SS, so CD4_HC is set
        # here from the curated diagnosis instead. Without this their cells sit in CD4_reactive
        # and are read as the benign compartment beside a tumour.
        try:
            don_tbl = TC.donor_table(args.tissue, obj=cfg["obj"])
        except Exception as exc:                                   # noqa: BLE001
            log(f"target_cohort unavailable ({exc}); falling back to the atlas disease column")
            don_tbl = None
        donor = obs[cfg["donor"]].astype(str)
        if don_tbl is not None:
            is_hc = donor.map(don_tbl["is_hc"]).fillna(False).to_numpy()
            log(f"healthy donors from clin_dx: {int(don_tbl['is_hc'].sum())} keys "
                f"({int(don_tbl['cohort_keep'].sum())} of {len(don_tbl)} keys are independent "
                "donors -- the long table carries them all, the notebooks filter)")
        else:
            is_hc = obs["disease"].astype(str).eq("HC").to_numpy()
        lvl[is_cd4 & is_hc] = "CD4_HC"
        obs["arm"] = lvl.to_numpy()
        obs["donor"] = donor
        log("arms:\n" + obs.groupby("arm").size().sort_values(ascending=False).to_string())

        # ---- genes: panel + surfaceome, plus the lineage controls and the trafficking modules
        if args.genes:
            genes = [g for g in args.genes if g in set(genes_all)]
            report = {"override": True, "n_screened": len(genes)}
        else:
            genes, report = ST.screen_genes(genes_all)
            have = set(genes_all)
            extra_lin = [g for g in TQ.lineage_genes(include_nk=True)
                         if g in have and g not in set(genes)]
            extra_tr = [g for g in ST.TRAFFIC_GENES if g in have and g not in set(genes)]
            genes = list(genes) + extra_lin + extra_tr
            report.update(n_lineage_added=len(extra_lin), n_traffic_added=len(extra_tr),
                          n_total=len(genes),
                          lineage_absent=[g for g in TQ.lineage_genes(True) if g not in have])
        gi = pd.Index(genes_all).get_indexer(genes)
        assert (gi >= 0).all()
        log(f"screening {len(genes):,} genes "
            f"(+{report.get('n_lineage_added', 0)} lineage, "
            f"+{report.get('n_traffic_added', 0)} trafficking)")

        g = h5["layers"][cfg["layer"]]

        # ---- library sizes. `total_counts` is the real one; the HVG sum is kept only so the
        #      v1 quantity stays reproducible from this table.
        lib = pd.to_numeric(obs["total_counts"], errors="coerce").to_numpy(float)
        if not np.isfinite(lib).all() or (lib <= 0).any():
            log("total_counts missing or non-positive for some cells -- falling back to the "
                "matrix row sums for those")
            rs = row_sums(g, n_obs)
            lib = np.where(np.isfinite(lib) & (lib > 0), lib, rs)
        lib_hvg = row_sums(g, n_obs)
        lib[lib <= 0] = 1.0
        lib_hvg[lib_hvg <= 0] = 1.0
        log(f"library sizes: total_counts median {np.median(lib):,.0f} | "
            f"HVG-sum median {np.median(lib_hvg):,.0f}")

        # ---- per-cell lineage load, from one small block read before the main loop
        lin_all = [x for x in TQ.lineage_genes(include_nk=True) if x in set(genes_all)]
        li = pd.Index(genes_all).get_indexer(lin_all)
        LIN = read_block(g, li, n_obs)
        lin_pos = pd.Index(lin_all)
        strict = [x for x in TQ.strict_genes() if x in set(lin_all)]
        # A single stray count of one lineage gene is ambient and would drop a fifth of the
        # cohort. A doublet looks different: it carries its partner's whole programme, so either
        # one marker at depth or several markers at once. Both forms are counted.
        S1 = LIN[:, lin_pos.get_indexer(strict)] >= 1
        Sn = LIN[:, lin_pos.get_indexer(strict)] >= args.strict_min_counts
        lineage_pos = Sn.any(1) | (S1.sum(1) >= args.strict_min_genes)
        log("strict lineage markers, share of cells carrying >= 1 count:\n"
            + pd.Series(S1.mean(0), index=strict).sort_values(ascending=False
                                                              ).round(4).to_string())
        log(f"  any marker at >= {args.strict_min_counts} counts : {Sn.any(1).mean():.3%}")
        log(f"  >= {args.strict_min_genes} distinct markers at >= 1: "
            f"{(S1.sum(1) >= args.strict_min_genes).mean():.3%}")
        mod_pos = {}
        for mod, spec in TQ.LINEAGE_MODULES.items():
            cols = [x for x in spec["strict"] if x in set(lin_all)]
            if cols:
                sub = LIN[:, lin_pos.get_indexer(cols)]
                mod_pos[mod] = ((sub >= args.strict_min_counts).any(1)
                                | ((sub >= 1).sum(1) >= args.strict_min_genes))
        scores = {}
        for mod, spec in TQ.LINEAGE_MODULES.items():
            cols = [x for x in spec["genes"] if x in set(lin_all)]
            if cols:
                scores[mod] = np.log1p(LIN[:, lin_pos.get_indexer(cols)]
                                       / lib[:, None] * 1e4).mean(1)
        nk_cols = [x for x in TQ.NK_MARKERS if x in set(lin_all)]
        if nk_cols:
            scores["nk"] = np.log1p(LIN[:, lin_pos.get_indexer(nk_cols)] / lib[:, None] * 1e4
                                    ).mean(1)
        cpl_mods = [m for m in TQ.COUPLING_MODULES[args.tissue] if m in scores]
        log(f"per-cell coupling modules: {cpl_mods or 'none available'}")
        del LIN
        log(f"lineage-positive cells (any strict marker): {lineage_pos.mean():.1%} "
            f"({int(lineage_pos.sum()):,})")

        # ---- clean mask: not lineage-positive, and not in the study's top doublet decile
        dbl = pd.to_numeric(obs["doublet_score"], errors="coerce").fillna(0).to_numpy(float)
        study = obs["study"].astype(str).to_numpy()
        cut = pd.Series(dbl).groupby(study).transform(lambda s: s.quantile(DBL_Q)).to_numpy()
        clean = (~lineage_pos) & (dbl <= cut)
        log(f"clean cells: {clean.mean():.1%} "
            f"(lineage drop {lineage_pos.mean():.1%}, doublet drop {(dbl > cut).mean():.1%})")

        # ---- depth matching: only cells deep enough to be thinned to the target
        ds_ok = lib >= args.ds_depth
        ds_p = np.where(ds_ok, args.ds_depth / np.maximum(lib, 1), 0.0)
        log(f"depth-matched subset at {args.ds_depth:,} UMI: {ds_ok.mean():.1%} of cells")

        # ---- groups
        key = pd.MultiIndex.from_arrays([obs["arm"].astype(str), obs["donor"]],
                                        names=["arm", "donor"])
        groups = {k: v.to_numpy() for k, v in
                  pd.Series(np.arange(n_obs), index=key).groupby(level=[0, 1])}
        groups = {k: v for k, v in groups.items() if len(v) >= args.min_cells}
        log(f"{len(groups):,} (arm, donor) groups with >= {args.min_cells} cells")

        clusters = {}
        if not args.no_cluster and CM.CLUSTERS_PQ.exists():
            cl = pd.read_parquet(CM.CLUSTERS_PQ, columns=["cell_id", "cluster"])
            cmap = cl.set_index("cell_id")["cluster"].reindex(obs.index)
            ckey = pd.MultiIndex.from_arrays(
                [obs["arm"].astype(str), obs["donor"], cmap.fillna("").astype(str)],
                names=["arm", "donor", "cluster"])
            clusters = {k: v.to_numpy() for k, v in
                        pd.Series(np.arange(n_obs), index=ckey).groupby(level=[0, 1, 2])
                        if k[2] != ""}
            clusters = {k: v for k, v in clusters.items() if len(v) >= args.min_cells}
            log(f"{len(clusters):,} (arm, donor, cluster) groups for the escape axis")

        mal_groups = {k: v for k, v in groups.items() if k[0] == "CD4_malignant"}
        log(f"{len(mal_groups):,} malignant groups enter the per-cell coupling pass")

        # ---- one pass per gene block
        acc_long, acc_cl, acc_cpl = [], [], []
        nblk = (len(gi) + BLOCK - 1) // BLOCK
        for bi, start in enumerate(range(0, len(gi), BLOCK), 1):
            cols = gi[start:start + BLOCK]
            names = genes[start:start + BLOCK]
            X = read_block(g, cols, n_obs)
            L = np.log1p(X / lib[:, None] * 1e4)
            Lh = np.log1p(X / lib_hvg[:, None] * 1e4)
            Lds = np.log1p(thin(X, ds_p, rng) / args.ds_depth * 1e4)

            for store, grps in ((acc_long, groups), (acc_cl, clusters)):
                for k, idx in grps.items():
                    rec = group_stats(X[idx], L[idx], names)
                    rec["arm"], rec["donor"] = k[0], k[1]
                    rec["mean_all_hvg"] = Lh[idx].mean(0)
                    d = idx[ds_ok[idx]]
                    rec["n_cells_ds"] = len(d)
                    rec["mean_all_ds"] = (Lds[d].mean(0) if len(d)
                                          else np.full(len(names), np.nan))
                    c = idx[clean[idx]]
                    rec["n_cells_clean"] = len(c)
                    if len(c):
                        rec["pct_expr_clean"] = (X[c] > 0).mean(0)
                        rec["mean_all_clean"] = L[c].mean(0)
                    else:
                        rec["pct_expr_clean"] = np.full(len(names), np.nan)
                        rec["mean_all_clean"] = np.full(len(names), np.nan)
                    if len(k) == 3:
                        rec["cluster"] = k[2]
                    store.append(pd.DataFrame(rec))

            for k, idx in mal_groups.items():
                cov = {"umi": np.log1p(lib[idx]), "dbl": dbl[idx],
                       **{m: scores[m][idx] for m in cpl_mods}}
                rec = couplings(L[idx], cov, names)
                rec["donor"] = k[1]
                top = lib[idx] >= np.quantile(lib[idx], 0.9)
                det_hi = (X[idx][top] > 0).mean(0)
                det_lo = (X[idx][~top] > 0).mean(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rec["log_or_topdecile"] = np.log(
                        (det_hi + 1e-3) / (1 - det_hi + 1e-3)
                        * (1 - det_lo + 1e-3) / (det_lo + 1e-3))
                acc_cpl.append(pd.DataFrame(rec))

            del X, L, Lh, Lds
            log(f"  block {bi}/{nblk}  ({start + len(cols):,}/{len(gi):,} genes)")

    # ---------------------------------------------------------------- write
    long = pd.concat(acc_long, ignore_index=True)
    front = ["arm", "donor", "gene", "n_cells"]
    long = long[front + [c for c in long.columns if c not in front]]
    out_long = args.out_dir / f"{args.tissue}_target_long.parquet"
    long.to_parquet(out_long, index=False)
    log(f"wrote {out_long.name}  {long.shape}")

    out_cl = None
    if acc_cl:
        cl_long = pd.concat(acc_cl, ignore_index=True)
        out_cl = args.out_dir / f"{args.tissue}_target_cluster.parquet"
        cl_long.to_parquet(out_cl, index=False)
        log(f"wrote {out_cl.name}  {cl_long.shape}")

    cpl = pd.concat(acc_cpl, ignore_index=True)
    agg = cpl.groupby("gene").apply(
        lambda d: pd.Series({
            "r_umi": _fisher_mean(d, "r_umi"),
            "r_dbl": _fisher_mean(d, "r_dbl"),
            **{f"r_{m}": _fisher_mean(d, f"r_{m}") for m in cpl_mods},
            "or_topdecile": float(np.exp(np.nanmean(d["log_or_topdecile"]))),
            "n_donors_cpl": int(d["donor"].nunique()),
        }), include_groups=False).reset_index()
    out_cpl = args.out_dir / f"{args.tissue}_target_qc_cell.parquet"
    agg.to_parquet(out_cpl, index=False)
    cpl.to_parquet(args.out_dir / f"{args.tissue}_target_qc_cell_donor.parquet", index=False)
    log(f"wrote {out_cpl.name}  {agg.shape}")

    keep_groups = set(groups)
    aug = obs.assign(_lin=lineage_pos, _clean=clean, _dbl=dbl, _lib=lib,
                     _myeloid=scores.get("myeloid", np.zeros(n_obs)),
                     _ng=pd.to_numeric(obs["n_genes"], errors="coerce"),
                     **{f"_pos_{m}": v for m, v in mod_pos.items()},
                     **{f"_score_{m}": v for m, v in scores.items()})
    meta = (aug.groupby(["arm", "donor"])
            .agg(study=("study", "first"), disease=("disease", "first"),
                 n_cells=("study", "size"), n_clean=("_clean", "sum"),
                 frac_lineage_pos=("_lin", "mean"), median_total_counts=("_lib", "median"),
                 median_n_genes=("_ng", "median"), median_doublet_score=("_dbl", "median"),
                 mean_myeloid_score=("_myeloid", "mean"),
                 **{f"frac_pos_{m}": (f"_pos_{m}", "mean") for m in mod_pos},
                 **{f"score_{m}": (f"_score_{m}", "mean") for m in scores})
            .reset_index())
    meta = meta[[tuple(r) in keep_groups for r in meta[["arm", "donor"]].to_numpy()]]
    meta.to_parquet(args.out_dir / f"{args.tissue}_target_meta.parquet", index=False)
    log(f"wrote {args.tissue}_target_meta.parquet  {meta.shape}")

    (args.out_dir / f"{args.tissue}_target_manifest.json").write_text(json.dumps({
        "tissue": args.tissue, "source": str(cfg["obj"]), "layer": cfg["layer"],
        "builder": "run_target_screen_qc.py",
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "malignancy_call": call,
        "n_cells": int(n_obs), "n_genes_in_object": int(len(genes_all)),
        "gene_selection": report,
        "min_cells": args.min_cells,
        "library": "obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg",
        "hc_arm_from": "clin_dx" if don_tbl is not None else "disease",
        "lineage_modules": {m: spec["genes"] for m, spec in TQ.LINEAGE_MODULES.items()},
        "coupling_modules": cpl_mods,
        "band_genes": TQ.BAND_GENES,
        "clean_rule": {"strict_lineage_markers": TQ.strict_genes(),
                       "strict_min_counts": args.strict_min_counts,
                       "strict_min_genes": args.strict_min_genes,
                       "frac_lineage_pos": float(lineage_pos.mean()),
                       "doublet_quantile_per_study": DBL_Q,
                       "frac_cells_clean": float(clean.mean())},
        "ds_depth": args.ds_depth, "frac_cells_ds": float(ds_ok.mean()),
        "arms": {k: int(v) for k, v in obs.groupby("arm").size().items()},
        "has_cluster_table": out_cl is not None,
    }, indent=1) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
