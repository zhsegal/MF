#!/usr/bin/env python
"""Per-(arm, donor, gene) histograms of log1p CP10K, for the pooled screens (nb46, nb48).

    python run_target_pool_hist.py --tissue blood   [--out-dir ...]
    python run_target_pool_hist.py --tissue skin

The (arm, donor, gene) table from `run_target_screen_qc.py` carries means, SDs and detection
rates, which is enough to pool any set of donors exactly for those three quantities -- but not
for a rank statistic. A pooled AUROC ("how well does this gene separate malignant from reactive
cells across the cohort") needs the distribution. A histogram per group is the distribution in a
form that can be re-pooled with any donor weights, any stratum and any bootstrap draw without
touching the matrix again.

Bins: 0 is exactly zero; then BIN_W-wide bins on (0, BIN_MAX] of log1p CP10K, overflow folded
into the top bin. Ties inside a bin count as ties, so the AUROC is exact up to BIN_W.

Same gene set as the long table (read from it), same library (`obs.total_counts`), same arms
(the block below is `run_target_screen_qc.py:253-279`, CD4 arms only). The job fails unless
every group's cell count and detection rate reproduce the long table, so the two tables cannot
silently describe different cells.

CPU only, no GPU.
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

JOB_DIR = Path(__file__).resolve().parent
NB_DIR = JOB_DIR.parent
sys.path.insert(0, str(NB_DIR / "helpers"))
sys.path.insert(0, str(JOB_DIR))

import h5py                              # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import cluster_malig_helpers as CM       # noqa: E402
import malignancy as MAL                 # noqa: E402
import target_cohort as TC               # noqa: E402
from run_target_screen_qc import TISSUE, read_block, row_sums, log   # noqa: E402

OUT_DIR = NB_DIR / "data" / "targets"
ARMS = ("CD4_malignant", "CD4_reactive", "CD4_HC")
MIN_CELLS = 20
BLOCK = 128
BIN_W = 0.05
BIN_MAX = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissue", choices=sorted(TISSUE), default="blood")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--long", type=Path, default=None,
                   help="long table whose genes and groups this must reproduce")
    return p.parse_args()


def to_bins(L):
    """log1p CP10K -> bin index: 0 for exact zero, 1..NB for (0, BIN_MAX]."""
    nb = int(round(BIN_MAX / BIN_W))
    b = np.ceil(L / BIN_W).astype(np.int64)
    return np.clip(b, 0, nb), nb + 1


def main():
    args = parse_args()
    cfg = TISSUE[args.tissue]
    CM.use_tissue(args.tissue)
    MAL.use_tissue(args.tissue)
    assert MAL.CLUSTER_PQ.name.startswith(f"{args.tissue}_"), MAL.CLUSTER_PQ
    long_path = args.long or OUT_DIR / f"{args.tissue}_target_long.parquet"
    man = json.loads(long_path.with_name(f"{args.tissue}_target_manifest.json").read_text())
    call = man["malignancy_call"]
    ref = pd.read_parquet(long_path, columns=["arm", "donor", "gene", "n_cells", "pct_expr",
                                              "mean_all"])
    ref = ref[ref["arm"].isin(ARMS)]
    genes = sorted(ref["gene"].unique())
    log(f"reference {long_path.name}: call {call}, {len(genes):,} genes, "
        f"{ref.groupby(['arm', 'donor']).ngroups} CD4 groups")
    t0 = time.time()

    with h5py.File(cfg["obj"], "r") as h5:
        genes_all = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        obs = CM.read_obs(h5, cols=[cfg["donor"], "study", "disease", cfg["celltype"], "has_tcr",
                                    "total_counts"])
        n_obs = len(obs)

        # ---- arms, as run_target_screen_qc.py builds them
        mal = MAL.load_malignancy(obs.index, has_tcr=obs["has_tcr"].astype(bool).to_numpy(),
                                  verbose=False)
        lvl = pd.Series(np.asarray(MAL.cd4_level(obs[cfg["celltype"]], mal,
                                                 call=call)).astype(str), index=obs.index)
        is_cd4 = obs[cfg["celltype"]].astype(str).eq("CD4").to_numpy()
        don_tbl = TC.donor_table(args.tissue, obj=cfg["obj"])
        donor = obs[cfg["donor"]].astype(str)
        is_hc = donor.map(don_tbl["is_hc"]).fillna(False).to_numpy()
        lvl[is_cd4 & is_hc] = "CD4_HC"
        obs["arm"] = lvl.to_numpy()
        obs["donor"] = donor

        key = pd.MultiIndex.from_arrays([obs["arm"].astype(str), obs["donor"]],
                                        names=["arm", "donor"])
        groups = {k: v.to_numpy() for k, v in
                  pd.Series(np.arange(n_obs), index=key).groupby(level=[0, 1])
                  if k[0] in ARMS}
        groups = {k: v for k, v in groups.items() if len(v) >= MIN_CELLS}
        log(f"{len(groups):,} CD4 (arm, donor) groups with >= {MIN_CELLS} cells")

        # ---- guard (a): the same cells as the long table
        n_ref = ref.groupby(["arm", "donor"])["n_cells"].first()
        n_here = pd.Series({k: len(v) for k, v in groups.items()})
        n_here.index.names = ["arm", "donor"]
        both = pd.concat([n_ref.rename("long"), n_here.rename("hist")], axis=1)
        bad = both[both["long"].ne(both["hist"])]
        if len(bad):
            raise SystemExit(f"group cell counts differ from {long_path.name}:\n{bad}")
        log("guard (a) passed: every group's cell count matches the long table")

        gi = genes_all.get_indexer(genes)
        assert (gi >= 0).all()
        g = h5["layers"][cfg["layer"]]
        lib = pd.to_numeric(obs["total_counts"], errors="coerce").to_numpy(float)
        if not np.isfinite(lib).all() or (lib <= 0).any():
            rs = row_sums(g, n_obs)
            lib = np.where(np.isfinite(lib) & (lib > 0), lib, rs)
        lib[lib <= 0] = 1.0

        acc = []
        nblk = (len(gi) + BLOCK - 1) // BLOCK
        for bi, start in enumerate(range(0, len(gi), BLOCK), 1):
            cols = gi[start:start + BLOCK]
            names = np.asarray(genes[start:start + BLOCK])
            X = read_block(g, cols, n_obs)
            B, nbin = to_bins(np.log1p(X / lib[:, None] * 1e4))
            off = np.arange(len(cols)) * nbin
            for (arm, dn), idx in groups.items():
                h = np.bincount((B[idx] + off).ravel(), minlength=len(cols) * nbin)
                h = h.reshape(len(cols), nbin)
                gg, bb = np.nonzero(h)
                acc.append(pd.DataFrame({"arm": arm, "donor": dn, "gene": names[gg],
                                         "bin": bb.astype(np.int16),
                                         "count": h[gg, bb].astype(np.int32)}))
            del X, B
            log(f"  block {bi}/{nblk}")

    hist = pd.concat(acc, ignore_index=True)
    for c in ("arm", "donor", "gene"):
        hist[c] = hist[c].astype("category")

    # ---- guard (b): detection reproduces exactly; the mean to within half a bin
    centers = np.r_[0.0, (np.arange(1, nbin) - 0.5) * BIN_W]
    chk = hist.assign(_c=lambda d: d["count"] * centers[d["bin"].to_numpy()],
                      _nz=lambda d: d["count"].where(d["bin"] > 0, 0))
    chk = chk.groupby(["arm", "donor", "gene"], observed=True)[["count", "_c", "_nz"]].sum()
    chk["pct_hist"] = chk["_nz"] / chk["count"]
    chk["mean_hist"] = chk["_c"] / chk["count"]
    chk = chk.join(ref.set_index(["arm", "donor", "gene"])[["pct_expr", "mean_all"]],
                   how="inner")
    d_pct = float((chk["pct_hist"] - chk["pct_expr"]).abs().max())
    d_mean = float((chk["mean_hist"] - chk["mean_all"]).abs().max())
    log(f"max |pct_hist - pct_expr| = {d_pct:.3g}; max |mean_hist - mean_all| = {d_mean:.4f} "
        f"(half a bin = {BIN_W / 2})")
    if d_pct > 1e-9:
        raise SystemExit("guard (b) failed: detection rates do not reproduce the long table")
    if d_mean > BIN_W / 2 + 1e-9:
        raise SystemExit("guard (b) failed: means differ by more than half a bin")
    log("guard (b) passed")

    out = args.out_dir / f"{args.tissue}_target_hist.parquet"
    hist.to_parquet(out, index=False)
    log(f"wrote {out.name}  {hist.shape}")
    (args.out_dir / f"{args.tissue}_target_hist_manifest.json").write_text(json.dumps({
        "tissue": args.tissue, "source": str(cfg["obj"]), "layer": cfg["layer"],
        "builder": "run_target_pool_hist.py", "written": time.strftime("%Y-%m-%d %H:%M"),
        "reference_long": str(long_path), "malignancy_call": call,
        "arms": list(ARMS), "min_cells": MIN_CELLS, "n_genes": len(genes),
        "n_groups": len(groups), "bin_width": BIN_W, "bin_max": BIN_MAX, "n_bins": int(nbin),
        "bin0": "exact zero", "library": "obs.total_counts",
        "max_abs_pct_diff": d_pct, "max_abs_mean_diff": d_mean,
    }, indent=1) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
