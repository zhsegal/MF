#!/usr/bin/env python
"""Standalone MrVI donor-level stage_group DE on T-cells.

Mirrors cells `9564f223` + `8e51d2a7` of notebooks/MF/02_mrvi.ipynb.
Reads the cached HVG adata + saved MrVI model; writes a netcdf.

Submit via notebooks/MF/jobs/run_mrvi_de.sh.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

NB_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = NB_DIR / "models" / "mrvi_ctcl"
CACHE_H5AD = NB_DIR / "data" / "cache" / "mrvi_ctcl_cache.h5ad"
TCELL_CNV = NB_DIR / "data" / "cache" / "tcells_cnv.h5ad"
DEFAULT_OUT = NB_DIR / "figures" / "mrvi_de_stage_tcells.nc"
ALLCELLS_OUT = NB_DIR / "figures" / "mrvi_de_stage_allcells.nc"

SAMPLE_KEY = "donor"
BATCH_KEY = "study"
COUNTS_LAYER = "raw_counts"
EARLY = {"IA", "IB", "IIA"}
TCELL_REGEX = r"T[ _]?cell|^Tc|^Th|Treg"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-per-ct", type=int, default=1500,
                   help="per-cell-type cap; ignored when --malignant-only")
    p.add_argument("--min-per-ct", type=int, default=50)
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--malignant-only", action="store_true",
                   help="restrict to inferCNV-malignant T cells; no per-CT cap")
    p.add_argument("--all-cells", action="store_true",
                   help="all cell types (per-CT capped), not just T cells")
    p.add_argument("--out-nc", type=str, default=str(DEFAULT_OUT),
                   help="output netcdf path")
    p.add_argument("--force", action="store_true",
                   help="overwrite output if present")
    return p.parse_args()


def preflight(malignant_only: bool):
    missing = []
    if not CACHE_H5AD.exists():
        missing.append(f"  cache h5ad: {CACHE_H5AD}")
    if not (MODEL_DIR / "model.pt").exists():
        missing.append(f"  model.pt:   {MODEL_DIR/'model.pt'}")
    if malignant_only and not TCELL_CNV.exists():
        missing.append(f"  tcells_cnv: {TCELL_CNV}")
    if missing:
        sys.exit(
            "missing required inputs:\n" + "\n".join(missing) +
            "\n\nrun notebooks/MF/02_mrvi.ipynb through the re-cache cell "
            "(just before the DE section) once interactively to materialize "
            "the cache, then resubmit.")


def main():
    args = parse_args()
    preflight(args.malignant_only)
    de_nc = Path(args.out_nc)
    if args.all_cells and args.out_nc == str(DEFAULT_OUT):
        de_nc = ALLCELLS_OUT

    if de_nc.exists() and not args.force:
        sys.exit(f"output already exists ({de_nc}); pass --force to overwrite")

    t0 = time.time()
    import jax
    import torch
    print(f"jax devices: {jax.devices()}  backend: {jax.default_backend()}")
    print(f"torch cuda: {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scvi
    from scvi.external import MRVI

    print(f"scvi {scvi.__version__}")
    print(f"loading cache: {CACHE_H5AD}")
    adata = sc.read_h5ad(CACHE_H5AD)
    if COUNTS_LAYER not in adata.layers:
        adata.layers[COUNTS_LAYER] = adata.X.copy()
    print(f"  adata: {adata.shape}; obsm: {list(adata.obsm)}")

    print(f"loading model: {MODEL_DIR}")
    model = MRVI.load(str(MODEL_DIR), adata=adata)

    adata.obs["stage_group"] = (
        adata.obs["stage"].astype(str)
        .map(lambda s: "early" if s in EARLY else "advanced")
        .astype("category").cat.reorder_categories(["early", "advanced"])
    )
    donor_stage = adata.obs.groupby("donor", observed=True)["stage_group"].nunique()
    assert (donor_stage == 1).all(), "stage_group must be constant per donor"
    model.update_sample_info(adata)
    print(adata.obs.drop_duplicates("donor")
          .groupby("stage_group", observed=True)["donor"].nunique())

    if args.malignant_only:
        print(f"loading inferCNV malignant labels: {TCELL_CNV}")
        ad_t_cnv = sc.read_h5ad(TCELL_CNV, backed="r")
        mal_bc = set(ad_t_cnv.obs_names[ad_t_cnv.obs["is_malignant"].values])
        del ad_t_cnv
        keep_mask = adata.obs_names.isin(mal_bc)
        adata_t = adata[keep_mask].copy()
        print(f"DE subset (malignant only, no cap): {adata_t.shape}")
        print(adata_t.obs["cell_type"].value_counts())
    elif args.all_cells:
        rng = np.random.default_rng(args.seed)
        ct = adata.obs["cell_type"].astype(str).values
        keep = []
        for _, idx in pd.Series(np.arange(adata.n_obs)).groupby(ct):
            idx = idx.values
            if len(idx) < args.min_per_ct:
                continue
            keep.append(idx if len(idx) <= args.max_per_ct
                        else rng.choice(idx, size=args.max_per_ct, replace=False))
        keep = np.sort(np.concatenate(keep))
        adata_t = adata[keep].copy()
        print(f"DE subset (all cell types, capped at {args.max_per_ct}): {adata_t.shape}")
        print(adata_t.obs["cell_type"].value_counts())
    else:
        rng = np.random.default_rng(args.seed)
        t_mask = adata.obs["cell_type"].astype(str).str.contains(TCELL_REGEX, regex=True)
        t_idx = np.where(t_mask.values)[0]
        ct = adata.obs["cell_type"].astype(str).values
        keep = []
        for _, idx in pd.Series(t_idx).groupby(ct[t_idx]):
            idx = idx.values
            if len(idx) < args.min_per_ct:
                continue
            keep.append(idx if len(idx) <= args.max_per_ct
                        else rng.choice(idx, size=args.max_per_ct, replace=False))
        keep = np.sort(np.concatenate(keep))
        adata_t = adata[keep].copy()
        print(f"DE subset: {adata_t.shape}")
        print(adata_t.obs["cell_type"].value_counts())

    print(f"running DE  mc_samples={args.mc_samples}  batch_size={args.batch_size}")
    de_res = model.differential_expression(
        adata=adata_t,
        sample_cov_keys=["stage_group"],
        add_batch_specific_offsets=True,
        store_lfc=True,
        mc_samples=args.mc_samples,
        batch_size=args.batch_size,
        use_vmap=False,  # loop instead of vmap → far lower peak GPU mem
    )
    de_nc.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in de_res.data_vars}
    de_res.to_netcdf(de_nc, engine="h5netcdf", encoding=enc)
    print(f"saved DE -> {de_nc}  ({de_nc.stat().st_size/1e6:.1f} MB)")
    print(de_res)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
