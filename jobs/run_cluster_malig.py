#!/usr/bin/env python
"""Per-sample Leiden clustering of all T cells, for the cluster-level malignancy call.

  --tissue skin   nb26b. Reads skin_T_tcr_annotated_v4.h5ad (539,916 T cells, layers/raw_counts,
                  li2024 V(D)J folded in), grouped by `sample_id`.
  --tissue blood  nb32b. Reads blood_T_annotated.h5ad (layers/counts), grouped by `donor` --
                  geskin `sample_id` is an HTO lane pooling 5-6 donors, so clustering per sample
                  would mix donors inside one cluster and the ALICE family share would be
                  meaningless. The written column is still called `sample_id`.

Every path comes from `cluster_malig_helpers.CONFIG[tissue]`. For skin it writes:

  data/atlas_joint/skin_T_clusters_v1.parquet          per cell: cell_id, sample_id, cluster,
                                                       umap_1/2, top_trbv, top_trav, cell_type_T,
                                                       tcr_ok, family
  data/atlas_joint/skin_T_cluster_pseudobulk_v1.h5ad   per cluster: summed raw counts (X),
                                                       layers/n_detected, obs = composition + TCR
  data/atlas_joint/skin_T_clusters_v1_parts/           one pickle per sample (resume cache)
  data/atlas_joint/skin_T_clusters_v1_manifest.json

and the `blood_T_*` equivalents for blood.

Per sample, all T types together: HVGs without TR/IG/MT/RP genes -> PCA -> kNN -> Leiden res 1.0
-> clusters < 10 cells merged into the nearest centroid -> UMAP. Samples with <= 30 expressed T
cells are skipped (listed in the manifest). CPU only.

Smoke test (login node, seconds):
  python run_cluster_malig.py --tissue blood --samples <donor> --out-dir <scratch dir>
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# BLAS / numba default to one thread per core: on a 256-core node a 253 x 2000 PCA then takes 36 s
# instead of < 1 s. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("CLUSTER_MALIG_THREADS", "4"))

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import anndata as ad                   # noqa: E402
import h5py                            # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
from scipy import sparse               # noqa: E402

import cluster_malig_helpers as CM     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissue", choices=sorted(CM.CONFIG), default="skin")
    p.add_argument("--samples", nargs="*", default=None, help="only these group keys (smoke test)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="recompute cached per-sample parts")
    args = p.parse_args()
    CM.use_tissue(args.tissue)          # before any CM path is read
    if args.out_dir is None:
        args.out_dir = CM.ATLAS_JOINT
    return args


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def one_sample(sid, rows, g, obs, genes, allowed, tcr_ok, fam):
    counts = CM.read_csr_rows(g, rows)
    labels, umap = CM.cluster_sample(counts, genes.to_numpy(), allowed)
    sums, det = CM.cluster_sums(counts, labels)
    o = obs.iloc[rows]
    keys = np.array([f"{sid}__{k}" for k in labels])
    cells = pd.DataFrame({
        "cell_id": o.index.to_numpy(), "sample_id": sid, "cluster": keys,
        "umap_1": umap[:, 0], "umap_2": umap[:, 1],
        "top_trbv": CM.v_calls(counts, genes, "TRBV"),
        "top_trav": CM.v_calls(counts, genes, "TRAV"),
        "cell_type_T": o["cell_type_T"].astype(str).to_numpy(),
        "tcr_ok": tcr_ok[rows], "family": fam[rows]})

    order = [f"{sid}__{k}" for k in range(sums.shape[0])]
    cl = (cells.groupby("cluster")
          .agg(n_cells=("cell_id", "size"), n_tcr=("tcr_ok", "sum"), n_family=("family", "sum"))
          .reindex(order))
    comp = (pd.crosstab(cells["cluster"], cells["cell_type_T"])
            .reindex(index=order, columns=CM.CELL_TYPES, fill_value=0))
    for c in CM.CELL_TYPES:
        cl[f"frac_{c}"] = comp[c] / cl["n_cells"]
    cl.insert(0, "sample_id", sid)
    cl.insert(1, "cluster_idx", np.arange(len(order)))
    meta = o[CM.META_COLS].astype(str).iloc[0]
    for c in CM.META_COLS:
        cl[c] = meta[c]
    cl.index.name = "cluster"
    return {"cells": cells, "obs": cl, "sums": sums.astype(np.float32),
            "det": det.astype(np.float32)}


def main():
    args = parse_args()
    parts = args.out_dir / f"{CM.CLUSTERS_PQ.stem}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with h5py.File(CM.TCR_OBJ, "r") as h5:
        # lineage=False on purpose: this job FEEDS mal_cluster, and the re-annotated labels were
        # themselves derived from mal_cluster. Overlaying them here would move the call that the
        # labels depend on. See helpers/lineage.py.
        obs = CM.read_obs(h5, lineage=False)
        genes = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
        g = h5["layers"][CM.COUNTS]
        expr = np.diff(g["indptr"][:]) > 0
        log(f"{CM.TCR_OBJ.name}: {len(obs):,} cells x {len(genes):,} genes | "
            f"empty rows {int((~expr).sum()):,}")

        alice = pd.read_parquet(CM.ALICE_PQ).set_index("cell_id")[CM.ALICE_COL]
        fam = CM.family_mask(obs, alice)
        tcr_ok = obs["has_tcr"].astype(bool).to_numpy() & (CM.trb_key(obs).to_numpy() != "")
        # on the CD4 cells ALICE called, the family must reproduce tcr_clonal & has_tcr exactly
        cd4 = obs["cell_type_T"].astype(str).eq("CD4").to_numpy() & obs.index.isin(alice.index)
        ref = (alice.reindex(obs.index).fillna(False).astype(bool).to_numpy()
               & obs["has_tcr"].astype(bool).to_numpy())
        assert (fam[cd4] == ref[cd4]).all(), f"{int((fam[cd4] != ref[cd4]).sum())} CD4 mismatches"
        log(f"family: {int(fam.sum()):,} cells = {int(fam[cd4].sum()):,} ALICE CD4 "
            f"+ {int(fam[~cd4].sum()):,} Treg/CD8/non-ALICE | TCR+ {int(tcr_ok.sum()):,}")

        allowed = ~CM.cluster_excluded(genes)
        # The grouping unit: sample_id in skin, donor in blood (geskin sample_id is an HTO lane
        # pooling 5-6 donors). Downstream everything still calls this column `sample_id`.
        sid_all = obs[CM.GROUP_KEY].astype(str).to_numpy()
        samples = sorted(set(sid_all)) if not args.samples else list(args.samples)
        unknown = set(samples) - set(sid_all)
        assert not unknown, f"unknown {CM.GROUP_KEY}s: {sorted(unknown)}"

        skipped, failed, done = {}, {}, []
        for i, sid in enumerate(samples, 1):
            part = parts / (re.sub(r"[^A-Za-z0-9_.-]", "_", sid) + ".pkl")
            rows = np.flatnonzero((sid_all == sid) & expr)
            if len(rows) < CM.MIN_SAMPLE_CELLS:
                skipped[sid] = int(len(rows))
                continue
            done.append(part)
            if part.exists() and not args.force:
                continue
            t = time.time()
            try:
                res = one_sample(sid, rows, g, obs, genes, allowed, tcr_ok, fam)
            except Exception as e:                       # one bad sample must not kill the run
                failed[sid] = f"{type(e).__name__}: {e}"
                done.pop()
                log(f"[{i}/{len(samples)}] {sid}: FAILED {failed[sid]}")
                continue
            with open(part, "wb") as fh:
                pickle.dump(res, fh)
            log(f"[{i}/{len(samples)}] {sid}: {len(rows):,} cells -> {len(res['obs'])} clusters "
                f"({time.time() - t:.0f}s)")

    log(f"assembling {len(done)} samples | skipped (<{CM.MIN_SAMPLE_CELLS} cells): {len(skipped)}"
        f" | failed: {len(failed)}")
    P = []
    for f in done:
        with open(f, "rb") as fh:
            P.append(pickle.load(fh))
    cells = pd.concat([d["cells"] for d in P], ignore_index=True)
    cells.to_parquet(args.out_dir / CM.CLUSTERS_PQ.name, index=False)
    pb = ad.AnnData(X=sparse.vstack([d["sums"] for d in P], format="csr"),
                    obs=pd.concat([d["obs"] for d in P]),
                    var=pd.DataFrame(index=genes),
                    layers={"n_detected": sparse.vstack([d["det"] for d in P], format="csr")})
    pb.write_h5ad(args.out_dir / CM.PSEUDOBULK_H5AD.name)

    manifest = {"source": str(CM.TCR_OBJ), "written": time.strftime("%Y-%m-%d %H:%M"),
                "n_samples": len(P), "n_clusters": int(pb.n_obs), "n_cells": int(len(cells)),
                "params": {"res": CM.RES, "n_hvg": CM.N_HVG, "n_pcs": CM.N_PCS,
                           "n_neighbors": CM.N_NEIGHBORS, "min_sample_cells": CM.MIN_SAMPLE_CELLS,
                           "min_cluster_cells": CM.MIN_CLUSTER_CELLS, "seed": CM.SEED},
                "skipped": skipped, "failed": failed}
    manifest["tissue"] = CM.TISSUE
    manifest["group_key"] = CM.GROUP_KEY
    (args.out_dir / CM.MANIFEST_JSON.name).write_text(json.dumps(manifest, indent=1))
    log(f"wrote {CM.CLUSTERS_PQ.name} ({len(cells):,} cells) + {CM.PSEUDOBULK_H5AD.name} "
        f"({pb.n_obs:,} clusters) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
