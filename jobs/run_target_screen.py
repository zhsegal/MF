#!/usr/bin/env python
"""Build the (arm, donor, gene) table the surface-target screen reads, for one compartment.

    python run_target_screen.py --tissue skin    [--out-dir ...]
    python run_target_screen.py --tissue blood

Reads the compartment's full-gene T object one gene-block at a time -- never the whole matrix --
and writes, per arm and donor:

    data/targets/<tissue>_target_long.parquet     arm, donor, gene, pct_expr, mean_all, n_cells
    data/targets/<tissue>_target_cluster.parquet  the same per 26b/32b cluster, for escape risk
    data/targets/<tissue>_target_manifest.json

The arms come from `helpers/malignancy.py` at the project headline call (`mal_cluster`):
CD4_malignant / CD4_reactive inside the disease arm, CD4_HC for healthy CD4, and every other T
type carried as its own arm for the off-tumour axis. Non-T compartments are NOT here -- in skin
they come from the degradome pseudobulk cube, which already has them at sample level.

Genes = `surface_targets.screen_genes()`, the a-priori panel UNION the confident surfaceome,
about 2,300 of the 42,347. `mean_all` is computed on log1p CP10K with the library size summed
over ALL genes before the subset, which is the same gate the degradome build asserts: take the
size factor after subsetting and every value is scaled by a cell-type-dependent factor.

CPU only, no GPU. Memory is one gene-block of cells x BLOCK genes.
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
from scipy import sparse                 # noqa: E402

import cluster_malig_helpers as CM       # noqa: E402
import malignancy as MAL                 # noqa: E402
import surface_targets as ST             # noqa: E402

OUT_DIR = NB_DIR / "data" / "targets"
MIN_CELLS = 20
BLOCK = 256                              # genes read per pass

TISSUE = {
    "skin": dict(obj=CM.CONFIG["skin"]["TCR_OBJ"], layer="raw_counts",
                 celltype="cell_type_T", donor="donor"),
    "blood": dict(obj=CM.CONFIG["blood"]["TCR_OBJ"], layer="counts",
                  celltype="cell_type_T", donor="donor"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissue", choices=sorted(TISSUE), required=True)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--min-cells", type=int, default=MIN_CELLS)
    p.add_argument("--genes", nargs="*", default=None, help="override the gene set (smoke test)")
    p.add_argument("--max-cells", type=int, default=None,
                   help="use only the first N cells (smoke test; never for a real run)")
    return p.parse_args()


def read_block(g, cols, n_obs):
    """Columns `cols` of a CSR h5ad matrix group, as a dense (cells x len(cols)) float32 array.

    CSR is row-major, so a column slice means walking every row. The matrix is read once in row
    chunks and the wanted columns picked out of each chunk -- one pass over the data per block,
    which is why BLOCK is as large as memory allows rather than one gene at a time.
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


def main():
    args = parse_args()
    cfg = TISSUE[args.tissue]
    CM.use_tissue(args.tissue)
    MAL.use_tissue(args.tissue)      # or blood would silently assemble the SKIN parquets
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with h5py.File(cfg["obj"], "r") as h5:
        genes_all = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        obs = CM.read_obs(h5, cols=[cfg["donor"], "sample_id", "study", "disease",
                                    cfg["celltype"], "has_tcr"])
        if args.max_cells:
            obs = obs.iloc[:args.max_cells]
            log(f"SMOKE TEST: first {len(obs):,} cells only -- results are not a cohort result")
        n_obs = len(obs)
        log(f"{cfg['obj'].name}: {n_obs:,} cells x {len(genes_all):,} genes")

        if len(genes_all) < 20_000:
            log(f"WARNING: only {len(genes_all):,} genes -- this is an HVG subset, not the "
                "full-gene object. Any gene absent from it is absent from the screen, which is "
                "not the same as being unexpressed.")

        # ---- arms: the headline malignancy call, healthy CD4 kept separate
        mal = MAL.load_malignancy(obs.index, has_tcr=obs["has_tcr"].astype(bool).to_numpy(),
                                  verbose=True)
        lvl = pd.Series(np.asarray(MAL.cd4_level(obs[cfg["celltype"]], mal,
                                                 call=MAL.HEADLINE)).astype(str), index=obs.index)
        is_hc = obs["disease"].astype(str).eq("HC").to_numpy()
        is_cd4 = obs[cfg["celltype"]].astype(str).eq("CD4").to_numpy()
        # HC CD4 is its own arm. Folding it into "reactive" would make that arm mean two things:
        # the benign cells beside a tumour, and the T cells of a person who has no tumour.
        lvl[is_cd4 & is_hc] = "CD4_HC"
        obs["arm"] = lvl.to_numpy()
        obs["donor"] = obs[cfg["donor"]].astype(str)

        counts = obs.groupby("arm").size().sort_values(ascending=False)
        log("arms:\n" + counts.to_string())

        # ---- genes
        if args.genes:
            genes = [g for g in args.genes if g in set(genes_all)]
            report = {"override": True, "n_screened": len(genes)}
        else:
            genes, report = ST.screen_genes(genes_all)
        gi = pd.Index(genes_all).get_indexer(genes)
        assert (gi >= 0).all()
        log(f"screening {len(genes):,} genes  ({report.get('by_source', {})})")

        # ---- library size over ALL genes, before any subset. The gate the degradome build
        #      asserts: a size factor computed after subsetting scales every value by a
        #      cell-type-dependent factor and nothing downstream means anything.
        g = h5["layers"][cfg["layer"]]
        lib = np.zeros(n_obs, dtype=np.float64)
        indptr = g["indptr"]
        for a in range(0, n_obs, 50_000):
            b = min(a + 50_000, n_obs)
            ip = indptr[a:b + 1].astype(np.int64)
            lo, hi = int(ip[0]), int(ip[-1])
            if hi == lo:
                continue
            dat = g["data"][lo:hi].astype(np.float64)
            lib[a:b] = np.add.reduceat(dat, ip[:-1] - lo) * (np.diff(ip) > 0)
        lib[lib == 0] = 1.0
        log(f"library sizes: median {np.median(lib):,.0f}")

        # ---- group indices: (arm, donor) and (arm, donor, cluster)
        key = pd.MultiIndex.from_arrays([obs["arm"].astype(str), obs["donor"]],
                                        names=["arm", "donor"])
        groups = {k: v.to_numpy() for k, v in
                  pd.Series(np.arange(n_obs), index=key).groupby(level=[0, 1])}
        groups = {k: v for k, v in groups.items() if len(v) >= args.min_cells}
        log(f"{len(groups):,} (arm, donor) groups with >= {args.min_cells} cells")

        clusters = None
        if CM.CLUSTERS_PQ.exists():
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
        else:
            log(f"no {CM.CLUSTERS_PQ.name} -- escape risk will have no within-donor term")

        # ---- one pass per gene block
        acc_long, acc_cl = [], []
        for start in range(0, len(gi), BLOCK):
            cols = gi[start:start + BLOCK]
            names = [genes[i] for i in range(start, min(start + BLOCK, len(genes)))]
            X = read_block(g, cols, n_obs)
            L = np.log1p(X / lib[:, None] * 1e4)          # CP10K on the FULL library size
            for store, grps in ((acc_long, groups), (acc_cl, clusters or {})):
                for k, idx in grps.items():
                    blk, lblk = X[idx], L[idx]
                    rec = {"arm": k[0], "donor": k[1], "gene": names,
                           "pct_expr": (blk > 0).mean(0), "mean_all": lblk.mean(0),
                           "n_cells": len(idx)}
                    if len(k) == 3:
                        rec["cluster"] = k[2]
                    store.append(pd.DataFrame(rec))
            del X, L
            log(f"  block {start // BLOCK + 1}/{(len(gi) + BLOCK - 1) // BLOCK} "
                f"({start + len(cols):,}/{len(gi):,} genes)")

    long = pd.concat(acc_long, ignore_index=True)
    out_long = args.out_dir / f"{args.tissue}_target_long.parquet"
    long.to_parquet(out_long, index=False)
    log(f"wrote {out_long.name}  {long.shape}")

    out_cl = None
    if acc_cl:
        cl_long = pd.concat(acc_cl, ignore_index=True)
        out_cl = args.out_dir / f"{args.tissue}_target_cluster.parquet"
        cl_long.to_parquet(out_cl, index=False)
        log(f"wrote {out_cl.name}  {cl_long.shape}")

    meta = (obs.groupby(["arm", "donor"])
            .agg(study=("study", "first"), disease=("disease", "first"), n_cells=("study", "size"))
            .reset_index())
    meta.to_parquet(args.out_dir / f"{args.tissue}_target_meta.parquet", index=False)

    (args.out_dir / f"{args.tissue}_target_manifest.json").write_text(json.dumps({
        "tissue": args.tissue, "source": str(cfg["obj"]), "layer": cfg["layer"],
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "malignancy_call": MAL.HEADLINE,
        "n_cells": int(len(obs)), "n_genes_in_object": int(len(genes_all)),
        "gene_selection": report,
        "min_cells": args.min_cells,
        "arms": {k: int(v) for k, v in obs.groupby("arm").size().items()},
        "has_cluster_table": out_cl is not None,
    }, indent=1) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
