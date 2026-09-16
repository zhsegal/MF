#!/usr/bin/env python
"""Train a pseudo-sample MrVI for Method 2 (malignant-vs-benign T cells).

Pseudo-sample = donor split by TCR anchor status (malignant / benign / unknown).
Trains MrVI with sample_key="pseudo_sample", batch_key="study" on ALL T cells
(unknown included), then persists the u (sample-unaware) and z latents per cell.
The notebook builds per-cluster anchor centroids from z and scores every cell.

Input h5ad is built by 17_mrvi_pseudosample_malignancy.ipynb (anchor logic lives
there). Submit via notebooks/MF/jobs/run_mrvi_pseudosample_m2.sh.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

NB_DIR = Path(__file__).resolve().parent.parent
OUT = NB_DIR / "data" / "atlas_joint" / "mrvi_malignancy"
INPUT_H5AD = OUT / "m2_pseudosample_input.h5ad"
MODEL_DIR = NB_DIR / "models" / "mrvi_pseudosample_m2"
U_NPY = OUT / "m2_u.npy"
Z_NPY = OUT / "m2_z.npy"
BC_NPY = OUT / "m2_barcodes.npy"
HISTORY = NB_DIR / "figures" / "mrvi_psm2_history.json"
DE_NC = NB_DIR / "figures" / "mrvi_de_pseudosample_m2.nc"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing model in MODEL_DIR")
    p.add_argument("--run-de", action="store_true",
                   help="also run differential_expression on malignant_status "
                        "(interpretability only; not used by the classifier)")
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--de-batch-size", type=int, default=128)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def preflight():
    if not INPUT_H5AD.exists():
        sys.exit(f"missing input: {INPUT_H5AD}\n"
                 "run 17_mrvi_pseudosample_malignancy.ipynb through the "
                 "'build pseudo-samples + job input' cell first")


def main():
    args = parse_args()
    preflight()

    t0 = time.time()
    import jax
    import torch
    print(f"jax devices: {jax.devices()}  backend: {jax.default_backend()}")
    print(f"torch cuda: {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    import numpy as np
    import scanpy as sc
    import scvi
    from scvi.external import MRVI

    print(f"scvi {scvi.__version__}")
    print(f"loading input: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
    print(f"  adata: {adata.shape}")
    print(adata.obs["malignant_status"].value_counts().to_string())

    scvi.settings.seed = args.seed
    MRVI.setup_anndata(adata, layer="raw_counts",
                       sample_key="pseudo_sample", batch_key="study")

    if not args.skip_train:
        model = MRVI(adata)
        print("training pseudo-sample MrVI...")
        model.train(
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            early_stopping=True, early_stopping_patience=args.patience,
            check_val_every_n_epoch=1, train_size=0.9,
        )
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_DIR), overwrite=True, save_anndata=False)
        hist = {}
        for k, v in (model.history or {}).items():
            try:
                hist[k] = v.iloc[:, 0].astype(float).tolist()
            except AttributeError:
                try:
                    hist[k] = [float(x) for x in v]
                except (TypeError, ValueError):
                    pass
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY, "w") as f:
            json.dump(hist, f)
        print(f"  saved model -> {MODEL_DIR}")
        print(f"  saved history -> {HISTORY}")
    else:
        print("skip-train: reloading model")
        model = MRVI.load(str(MODEL_DIR), adata=adata)

    OUT.mkdir(parents=True, exist_ok=True)
    u = model.get_latent_representation(give_z=False)
    z = model.get_latent_representation(give_z=True)
    np.save(U_NPY, np.asarray(u))
    np.save(Z_NPY, np.asarray(z))
    np.save(BC_NPY, adata.obs_names.to_numpy().astype(str))
    print(f"  saved u -> {U_NPY}  {u.shape}")
    print(f"  saved z -> {Z_NPY}  {z.shape}")
    print(f"  saved barcodes -> {BC_NPY}  ({adata.n_obs})")

    if args.run_de:
        if DE_NC.exists() and not args.force:
            print(f"DE exists; pass --force to overwrite ({DE_NC}); skipping DE")
        else:
            import numpy as np  # noqa: F811
            contrast = (
                adata.obs.loc[adata.obs["malignant_status"] != "unknown",
                              "pseudo_sample"].astype(str).unique().tolist()
            )
            print(f"running DE (malignant_status) over {len(contrast)} contrast pseudo-samples...")
            model.update_sample_info(adata)
            de = model.differential_expression(
                sample_cov_keys=["malignant_status"],
                sample_subset=contrast,
                add_batch_specific_offsets=True,
                store_lfc=True,
                mc_samples=args.mc_samples,
                batch_size=args.de_batch_size,
                use_vmap=False,
            )
            DE_NC.parent.mkdir(parents=True, exist_ok=True)
            enc = {v: {"zlib": True, "complevel": 4} for v in de.data_vars}
            de.to_netcdf(DE_NC, engine="h5netcdf", encoding=enc)
            print(f"  saved DE -> {DE_NC}  ({DE_NC.stat().st_size/1e6:.1f} MB)")

    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
