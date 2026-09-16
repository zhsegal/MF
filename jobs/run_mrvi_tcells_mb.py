#!/usr/bin/env python
"""Train MrVI on T-cells with malignant/benign pseudo-samples, then DE.

Mirrors cells `mrvi-mb-prep` + `mrvi-mb-fit` + `mrvi-mb-de` of
notebooks/old/03_mrvi_replication.ipynb.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

NB_DIR = Path(__file__).resolve().parent.parent
CACHE_H5AD = NB_DIR / "data" / "cache" / "mrvi_ctcl_cache.h5ad"
MODEL_MB_DIR = NB_DIR / "models" / "mrvi_tumor_vs_benignT"
MB_LATENT_NPY = NB_DIR / "data" / "cache" / "tumor_benignT_X_mrvi_mb.npy"
MB_HISTORY = NB_DIR / "figures" / "mrvi_mb_history.json"
DE_MB_NC = NB_DIR / "figures" / "mrvi_de_tumor_vs_benignT.nc"
TUMOR = "tumor_cell"
BENIGN_T = ["Th", "Tc", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]
MIN_CELLS_PER_PSEUDO = 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--de-batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing model in MODEL_MB_DIR")
    p.add_argument("--skip-de", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def preflight():
    if not CACHE_H5AD.exists():
        sys.exit(f"missing cache: {CACHE_H5AD}\n"
                 "run notebook 02 through the re-cache cell first")


def main():
    args = parse_args()
    preflight()
    if DE_MB_NC.exists() and not args.force and not args.skip_de:
        sys.exit(f"DE_MB_NC exists; pass --force or --skip-de\n  {DE_MB_NC}")

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

    print(f"loading cache: {CACHE_H5AD}")
    adata = sc.read_h5ad(CACHE_H5AD)
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
    ct = adata.obs["cell_type"].astype(str)
    ad_t = adata[ct.isin([TUMOR, *BENIGN_T])].copy()
    del adata
    ad_t.obs["is_malignant"] = (ad_t.obs["cell_type"].astype(str) == TUMOR).values
    print(f"  subset (tumor_cell + benign T): {ad_t.shape}  "
          f"malignant={int(ad_t.obs['is_malignant'].sum())}")

    ad_t.obs["sample_mb"] = (
        ad_t.obs["donor"].astype(str) + "_" +
        np.where(ad_t.obs["is_malignant"], "mal", "ben")
    ).astype("category")
    counts = ad_t.obs["sample_mb"].value_counts()
    keep_samples = counts[counts >= MIN_CELLS_PER_PSEUDO].index
    print(f"keeping {len(keep_samples)}/{len(counts)} pseudo-samples "
          f"(>={MIN_CELLS_PER_PSEUDO} cells)")
    ad_t_mb = ad_t[ad_t.obs["sample_mb"].isin(keep_samples)].copy()
    ad_t_mb.obs["sample_mb"] = ad_t_mb.obs["sample_mb"].cat.remove_unused_categories()
    ad_t_mb.obs["target_status"] = (
        np.where(ad_t_mb.obs["is_malignant"], "malignant", "benign")
    )
    ad_t_mb.obs["target_status"] = (
        ad_t_mb.obs["target_status"].astype("category")
        .cat.reorder_categories(["benign", "malignant"])
    )
    print(f"  ad_t_mb: {ad_t_mb.shape}")

    scvi.settings.seed = args.seed

    if not args.skip_train:
        MRVI.setup_anndata(ad_t_mb, layer="raw_counts",
                           sample_key="sample_mb", batch_key="study")
        model_mb = MRVI(ad_t_mb)
        print("training MrVI pseudo-sample model...")
        model_mb.train(
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            early_stopping=True, early_stopping_patience=args.patience,
            check_val_every_n_epoch=1, train_size=0.9,
        )
        MODEL_MB_DIR.parent.mkdir(parents=True, exist_ok=True)
        model_mb.save(str(MODEL_MB_DIR), overwrite=True, save_anndata=False)

        hist = {}
        for k, v in (model_mb.history or {}).items():
            if isinstance(v, pd.DataFrame):
                hist[k] = v.iloc[:, 0].astype(float).tolist()
            else:
                try:
                    hist[k] = [float(x) for x in v]
                except (TypeError, ValueError):
                    pass
        MB_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with open(MB_HISTORY, "w") as f:
            json.dump(hist, f)
        print(f"  saved model -> {MODEL_MB_DIR}")
        print(f"  saved history -> {MB_HISTORY}")
    else:
        print("skip-train: reloading model_mb")
        MRVI.setup_anndata(ad_t_mb, layer="raw_counts",
                           sample_key="sample_mb", batch_key="study")
        model_mb = MRVI.load(str(MODEL_MB_DIR), adata=ad_t_mb)

    ad_t_mb.obsm["X_mrvi_mb"] = model_mb.get_latent_representation()
    MB_LATENT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(MB_LATENT_NPY, ad_t_mb.obsm["X_mrvi_mb"])
    print(f"  saved latent -> {MB_LATENT_NPY}  shape={ad_t_mb.obsm['X_mrvi_mb'].shape}")

    if not args.skip_de:
        model_mb.update_sample_info(ad_t_mb)
        print("running DE (target_status)...")
        de_mb = model_mb.differential_expression(
            sample_cov_keys=["target_status"],
            add_batch_specific_offsets=True,
            store_lfc=True,
            mc_samples=args.mc_samples,
            batch_size=args.de_batch_size,
            use_vmap=False,  # loop instead of vmap → far lower peak GPU mem
        )
        DE_MB_NC.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 4} for v in de_mb.data_vars}
        de_mb.to_netcdf(DE_MB_NC, engine="h5netcdf", encoding=enc)
        print(f"  saved DE -> {DE_MB_NC}  ({DE_MB_NC.stat().st_size/1e6:.1f} MB)")
        print(de_mb)

    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
