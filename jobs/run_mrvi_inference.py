#!/usr/bin/env python
"""MrVI sample-level diagnostics: donor distances + differential abundance.

Mirrors cells `donor-dist` + `diff-abund` of notebooks/MF/03_mrvi_replication.ipynb.
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
DIST_NC = NB_DIR / "figures" / "mrvi_donor_distances.nc"
DA_NC = NB_DIR / "figures" / "mrvi_da_stage.nc"
EARLY = {"IA", "IB", "IIA"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def preflight():
    miss = []
    if not CACHE_H5AD.exists():
        miss.append(f"  cache:    {CACHE_H5AD}")
    if not (MODEL_DIR / "model.pt").exists():
        miss.append(f"  model.pt: {MODEL_DIR/'model.pt'}")
    if miss:
        sys.exit("missing inputs:\n" + "\n".join(miss))


def main():
    args = parse_args()
    preflight()
    if DIST_NC.exists() and DA_NC.exists() and not args.force:
        sys.exit(f"outputs exist; pass --force\n  {DIST_NC}\n  {DA_NC}")

    t0 = time.time()
    import jax
    import torch
    print(f"jax devices: {jax.devices()}  backend: {jax.default_backend()}")
    print(f"torch cuda: {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    import scanpy as sc
    from scvi.external import MRVI

    def _stringify_cat_coords(ds):
        for c in list(ds.coords):
            if str(ds[c].dtype) == "category":
                ds = ds.assign_coords({c: ds[c].astype(str)})
        return ds

    print(f"loading cache: {CACHE_H5AD}")
    adata = sc.read_h5ad(CACHE_H5AD)
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
    print(f"  adata: {adata.shape}")

    print(f"loading model: {MODEL_DIR}")
    model = MRVI.load(str(MODEL_DIR), adata=adata)

    adata.obs["stage_group"] = (
        adata.obs["stage"].astype(str)
        .map(lambda s: "early" if s in EARLY else "advanced")
        .astype("category").cat.reorder_categories(["early", "advanced"])
    )
    model.update_sample_info(adata)

    DIST_NC.parent.mkdir(parents=True, exist_ok=True)

    print("computing donor distances (groupby=cell_type)...")
    dist = model.get_local_sample_distances(
        keep_cell=False, groupby="cell_type", batch_size=args.batch_size,
    )
    dist = _stringify_cat_coords(dist)
    enc = {v: {"zlib": True, "complevel": 4} for v in dist.data_vars}
    dist.to_netcdf(DIST_NC, engine="h5netcdf", encoding=enc)
    print(f"  -> {DIST_NC}  ({DIST_NC.stat().st_size/1e6:.1f} MB)")

    print("computing differential abundance (stage_group)...")
    da = model.differential_abundance(
        sample_cov_keys=["stage_group"],
        compute_log_enrichment=True,
        batch_size=args.batch_size,
    )
    da = _stringify_cat_coords(da)
    enc = {v: {"zlib": True, "complevel": 4} for v in da.data_vars}
    da.to_netcdf(DA_NC, engine="h5netcdf", encoding=enc)
    print(f"  -> {DA_NC}  ({DA_NC.stat().st_size/1e6:.1f} MB)")

    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
