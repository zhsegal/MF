#!/usr/bin/env python
"""Per-sample expression sex call (XIST vs chrY genes) for every atlas sample.

Two uses, both in jobs/build_clinical_metadata.py:
  * fill clin_sex where no paper reports it (sex is unknown for ~50 samples);
  * check the paper-derived sex. A mismatch means the row -> patient mapping is wrong, which is
    the open question for harro2023 (MF1-3 / SS1-4 order), herrera2021 (SS5/SS6) and
    gaydosik2019 (MF2/MF5).

Streams layers/raw_counts of joint_mrvi_input.h5ad (all six genes are in its 10k HVGs) in row
blocks, so memory stays at one block. XIST is the primary signal: chrY loss is common in male
CTCL clones, so a low Y signal alone does not make a sample female.

Writes tables/sample_sex_inferred.csv.

Usage:
  python run_sex_infer.py
  python run_sex_infer.py --chunk 100000
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
SRC = NB_DIR / "data" / "atlas_joint" / "joint_mrvi_input.h5ad"
OUT = NB_DIR / "tables" / "sample_sex_inferred.csv"
Y_GENES = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY"]


def _str(g, col):
    o = g[col]
    if isinstance(o, h5py.Group):
        cats = np.asarray(o["categories"][:]).astype(str)
        return cats[o["codes"][:]]
    return np.asarray(o[:]).astype(str)


def call(r):
    """Fractions of cells with any XIST / any chrY read. 10x 5' detects XIST in only ~10-25% of
    female cells, so the thresholds are relative to the other signal, not absolute."""
    if r.n_cells < 50:
        return "low_cells"
    if r.xist_frac < 0.01 and r.y_frac < 0.01:
        return "no_signal"          # sex genes absent, e.g. filtered out of the deposited object
    if r.xist_frac >= 0.05 and r.y_frac < 0.02:
        return "F"
    if r.y_frac >= 0.05 and r.xist_frac < 0.02:
        return "M"
    if r.xist_frac < 0.02:
        return "M_lowY"             # no XIST, weak chrY, e.g. clonal chrY loss
    if r.xist_frac >= 0.05 and r.y_frac >= 0.05:
        return "mixed"              # pooled lane, or heavy ambient/doublets
    return "ambiguous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--layer", default="raw_counts")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    with h5py.File(args.src, "r") as f:
        var = f["var"]
        genes = np.asarray(var[var.attrs["_index"]][:]).astype(str)
        col = {g: int(np.flatnonzero(genes == g)[0]) for g in ["XIST"] + Y_GENES
               if (genes == g).any()}
        assert "XIST" in col and len(col) > 1, f"sex genes missing: {sorted(col)}"
        sid = _str(f["obs"], "sample_id")
        M = f["layers"][args.layer]
        indptr = M["indptr"][:]
        n = len(indptr) - 1
        xist = np.zeros(n, np.float32)
        ysum = np.zeros(n, np.float32)
        lib = np.zeros(n, np.float32)
        for s in range(0, n, args.chunk):
            e = min(s + args.chunk, n)
            a, b = indptr[s], indptr[e]
            ind = M["indices"][a:b]
            dat = M["data"][a:b].astype(np.float32)
            row = np.repeat(np.arange(e - s), np.diff(indptr[s:e + 1]))
            lib[s:e] = np.bincount(row, weights=dat, minlength=e - s)
            m = ind == col["XIST"]
            xist[s:e] = np.bincount(row[m], weights=dat[m], minlength=e - s)
            m = np.isin(ind, [col[g] for g in Y_GENES if g in col])
            ysum[s:e] = np.bincount(row[m], weights=dat[m], minlength=e - s)
            if (s // args.chunk) % 10 == 0:
                print(f"  {e:,}/{n:,} rows  {time.time() - t0:.0f}s", flush=True)

    d = pd.DataFrame({"sample_id": sid, "xist": xist, "y": ysum, "lib": lib})
    g = d.groupby("sample_id")
    out = pd.DataFrame({
        "n_cells": g.size(),
        "xist_frac": g.xist.apply(lambda v: float((v > 0).mean())),
        "y_frac": g.y.apply(lambda v: float((v > 0).mean())),
        "xist_cp10k": g.xist.sum() / g.lib.sum() * 1e4,
        "y_cp10k": g.y.sum() / g.lib.sum() * 1e4,
    }).reset_index()
    out["sex_expr"] = out.apply(call, axis=1)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} samples) in {time.time() - t0:.0f}s")
    print(out.sex_expr.value_counts().to_string())


if __name__ == "__main__":
    sys.exit(main())
