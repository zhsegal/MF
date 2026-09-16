#!/usr/bin/env python
"""MrVI differential abundance + donor distances for the descriptive atlas notebook (nb32).

Runs on the retrained skin MrVI (models/mrvi_joint_skin) over the extended atlas
(data/atlas_joint/joint_mrvi_input_skin.h5ad, 749510 skin cells, 99 samples, 7 studies).
Extends the covariate set beyond the training-time diagnostics (disease + stage_class) to the
clinical axes used in nb32, and groups donor distances by the *new* reannotation label
`cell_type_final` (merged from skin_cell_type_final.csv) rather than the Li2024 `cell_type`.

Covariate confound caveat (same as run_mrvi_joint.py): study is partly confounded with
disease/compartment/treatment; read comparative DA within strata where both groups co-occur.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

NB_DIR = Path(__file__).resolve().parent.parent
MRVI_INPUT = NB_DIR / "data" / "atlas_joint" / "joint_mrvi_input_skin.h5ad"
MODEL_DIR = NB_DIR / "models" / "mrvi_joint_skin"
LABEL_CSV = NB_DIR / "data" / "atlas_joint" / "skin_cell_type_final.csv"
DIST_NC = NB_DIR / "figures" / "mrvi_dist_descriptive.nc"
DA_NC = NB_DIR / "figures" / "mrvi_da_descriptive.nc"

SAMPLE_KEY = "sample_id"
BATCH_KEY = "study"
COUNTS_LAYER = "raw_counts"
GROUPBY = "cell_type_final"
EARLY = {"IA", "IB", "IIA"}
# candidate sample-level covariates for DA; filtered to those with >=2 levels present
DA_COV_KEYS = [
    "disease", "stage_class", "stage_group", "blood_involvement",
    "treatment_context", "cohort_region", "tissue", "sex",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def preflight():
    miss = [str(p) for p in (MRVI_INPUT, MODEL_DIR / "model.pt", LABEL_CSV) if not p.exists()]
    if miss:
        sys.exit("missing inputs:\n  " + "\n  ".join(miss))


def _stringify_cat_coords(ds):
    for c in list(ds.coords):
        if str(ds[c].dtype) == "category":
            ds = ds.assign_coords({c: ds[c].astype(str)})
    return ds


def main():
    args = parse_args()
    preflight()
    if DIST_NC.exists() and DA_NC.exists() and not args.force:
        sys.exit(f"outputs exist; pass --force\n  {DIST_NC}\n  {DA_NC}")

    t0 = time.time()
    import jax
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scvi.external import MRVI

    print(f"jax devices: {jax.devices()}  backend: {jax.default_backend()}")
    print(f"loading MrVI input: {MRVI_INPUT}")
    adata = sc.read_h5ad(MRVI_INPUT)
    if COUNTS_LAYER not in adata.layers:
        adata.layers[COUNTS_LAYER] = adata.X.copy()
    else:
        # X and layers[raw_counts] are byte-identical float64 CSRs (345M nnz each, ~5.5 GB
        # apiece). Only the layer is registered with MRVI -> alias X to it and drop the
        # duplicate so the reservation can stay small.
        adata.X = adata.layers[COUNTS_LAYER]
    # obsm carried from earlier notebooks is unused here and adds a few GB
    for _k in ("X_adt", "X_scVI", "X_scVI_MDE"):
        adata.obsm.pop(_k, None)
    print(f"  adata: {adata.shape}  ({adata.layers[COUNTS_LAYER].nnz:,} nnz)")

    # merge new reannotation label for the distance groupby
    lab = pd.read_csv(LABEL_CSV).set_index("cell_id")
    adata.obs[GROUPBY] = (
        lab["cell_type_final"].reindex(adata.obs["cell_id"]).astype("category").values
    )
    n_lab = adata.obs[GROUPBY].notna().sum()
    print(f"  merged {GROUPBY}: {n_lab}/{adata.n_obs} labeled")

    # binary stage_group (early vs advanced) from disease_stage
    stg = adata.obs["disease_stage"].astype(str)
    adata.obs["stage_group"] = np.where(
        stg.isin(EARLY), "early",
        np.where(stg.isin({"IIB", "IIIB", "IV", "IVA2", "IVB"}), "advanced", "other"),
    )

    print(f"loading model: {MODEL_DIR}")
    MRVI.setup_anndata(adata, layer=COUNTS_LAYER, sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)
    model = MRVI.load(str(MODEL_DIR), adata=adata)
    model.update_sample_info(adata)

    DIST_NC.parent.mkdir(parents=True, exist_ok=True)

    print(f"computing donor distances (groupby={GROUPBY})...")
    dist = model.get_local_sample_distances(
        keep_cell=False, groupby=GROUPBY, batch_size=args.batch_size,
    )
    dist = _stringify_cat_coords(dist)
    enc = {v: {"zlib": True, "complevel": 4} for v in dist.data_vars}
    dist.to_netcdf(DIST_NC, engine="h5netcdf", encoding=enc)
    print(f"  -> {DIST_NC}  ({DIST_NC.stat().st_size/1e6:.1f} MB)")

    # keep only covariates that are present and have >=2 observed levels
    cov_keys = []
    for k in DA_COV_KEYS:
        if k in adata.obs and adata.obs[k].astype(str).nunique() >= 2:
            cov_keys.append(k)
    print(f"computing differential abundance ({cov_keys})...")
    da = model.differential_abundance(
        sample_cov_keys=cov_keys,
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
