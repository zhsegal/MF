#!/usr/bin/env python
"""Train MrVI on the full joint CTCL/MF/SS atlas, cache latent + diagnostics.

Sample-aware model over the joint atlas (data/atlas_joint/joint_annotated.h5ad,
~1.34M cells, 109 samples, 9 studies). sample_key=sample_id, batch_key=study.

CAVEAT: this fork's MRVI takes a single batch_key only (study). In this atlas study
is partly confounded with disease/compartment (blood-only Sezary vs skin-only MF), so
comparative DE/DA across disease must be read within strata where both groups co-occur;
the confound-free contrasts are within-donor (see run_mrvi_tcells_mb.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# PREALLOCATE THE GPU (2026-09-06). With on-demand allocation (false, the previous
# setting) MrVI training dies with a non-deterministic CUDA_ERROR_ILLEGAL_ADDRESS partway
# through: blood failed at epoch 4 twice, the full atlas at epoch 8 and 46, on three hosts
# and on BOTH GPU classes (A40 and L40S). It is not the data -- a 200k-cell subsample of
# the same blood input spanning all 89 samples ran 8 epochs clean, and the inputs carry no
# NaN, no zero-count cells, no invalid categorical codes, and smaller library-size extremes
# than the skin run that completes. It is allocator fragmentation: with preallocation on,
# full-size blood cleared 6 epochs on the first try. Peak GPU use is ~17 GB of 48 GB, so
# preallocating costs nothing under -gpu j_exclusive=yes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

NB_DIR = Path(__file__).resolve().parent.parent
ANNOT = NB_DIR / "data" / "atlas_joint" / "joint_annotated.h5ad"
HVG_INPUT = NB_DIR / "data" / "atlas_joint" / "joint_mrvi_input.h5ad"
MODEL_DIR = NB_DIR / "models" / "mrvi_joint"
LATENT_Z_NPY = NB_DIR / "data" / "atlas_joint" / "joint_X_mrvi.npy"      # z (sample-aware)
LATENT_U_NPY = NB_DIR / "data" / "atlas_joint" / "joint_X_mrvi_u.npy"    # u (sample-unaware)
HISTORY = NB_DIR / "figures" / "mrvi_joint_history.json"
DIST_NC = NB_DIR / "figures" / "mrvi_joint_donor_distances.nc"
DA_NC = NB_DIR / "figures" / "mrvi_joint_da.nc"

SAMPLE_KEY = "sample_id"
BATCH_KEY = "study"
COUNTS_LAYER = "raw_counts"
DA_COV_KEYS = ["disease", "stage_class"]


def parse_args():
    p = argparse.ArgumentParser()
    # defaults follow the MrVI paper (s41592-025-02808-x) + v1.3.3 tutorial / DEFAULT_TRAIN_KWARGS
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-top-genes", type=int, default=10000)
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--n-latent-u", type=int, default=10)
    p.add_argument("--diag-batch-size", type=int, default=512)
    p.add_argument("--subsample", type=int, default=0,
                   help="0 = train on all cells; >0 = fit on a random subsample, transform all")
    p.add_argument("--subset-compartment", default=None,
                   help="restrict to adata.obs['compartment'] == this value (e.g. Skin)")
    p.add_argument("--tag", default="",
                   help="suffix for all outputs (e.g. skin) so subset runs don't collide")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing model in MODEL_DIR")
    p.add_argument("--skip-diag", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rebuild the HVG input cache even if it exists")
    return p.parse_args()


def _tag_path(p: Path, tag: str) -> Path:
    if not tag:
        return p
    if p.suffix:                                  # file: insert before the extension
        return p.with_name(f"{p.stem}_{tag}{p.suffix}")
    return p.with_name(f"{p.name}_{tag}")         # dir (MODEL_DIR): append


def apply_tag(tag: str):
    """Rebind all output paths with a `_{tag}` suffix (ANNOT source is never tagged)."""
    global HVG_INPUT, MODEL_DIR, LATENT_Z_NPY, LATENT_U_NPY, HISTORY, DIST_NC, DA_NC
    HVG_INPUT = _tag_path(HVG_INPUT, tag)
    MODEL_DIR = _tag_path(MODEL_DIR, tag)
    LATENT_Z_NPY = _tag_path(LATENT_Z_NPY, tag)
    LATENT_U_NPY = _tag_path(LATENT_U_NPY, tag)
    HISTORY = _tag_path(HISTORY, tag)
    DIST_NC = _tag_path(DIST_NC, tag)
    DA_NC = _tag_path(DA_NC, tag)


def preflight():
    if not ANNOT.exists() and not HVG_INPUT.exists():
        sys.exit(f"missing inputs: neither\n  {ANNOT}\n  {HVG_INPUT}\nexists; "
                 "run notebook 10 through Step 3b first")


def build_hvg_input(args):
    """Step A - HVG-subset cache (seurat_v3, batched on study)."""
    import scanpy as sc

    if HVG_INPUT.exists() and not args.force:
        print(f"reusing HVG input cache: {HVG_INPUT}")
        return sc.read_h5ad(HVG_INPUT)

    print(f"loading annotated atlas: {ANNOT}")
    adata = sc.read_h5ad(ANNOT)
    if COUNTS_LAYER not in adata.layers:
        adata.layers[COUNTS_LAYER] = adata.X.copy()

    if args.subset_compartment:
        n0 = adata.n_obs
        adata = adata[adata.obs["compartment"] == args.subset_compartment].copy()
        # drop now-empty study/sample levels so MrVI sees only the present batches/samples
        for c in (BATCH_KEY, SAMPLE_KEY):
            if str(adata.obs[c].dtype) == "category":
                adata.obs[c] = adata.obs[c].cat.remove_unused_categories()
        print(f"  subset compartment=={args.subset_compartment!r}: {n0} -> {adata.n_obs} cells, "
              f"{adata.obs[SAMPLE_KEY].nunique()} samples, {adata.obs[BATCH_KEY].nunique()} studies")

    print(f"  atlas: {adata.shape}  -> HVG {args.n_top_genes} (seurat_v3, batch={BATCH_KEY})")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_top_genes, flavor="seurat_v3",
        batch_key=BATCH_KEY, layer=COUNTS_LAYER, subset=True,
    )
    HVG_INPUT.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(HVG_INPUT)
    print(f"  wrote HVG input -> {HVG_INPUT}  {adata.shape}")
    return adata


def save_history(model):
    import pandas as pd

    hist = {}
    for k, v in (model.history or {}).items():
        if isinstance(v, pd.DataFrame):
            hist[k] = v.iloc[:, 0].astype(float).tolist()
        else:
            try:
                hist[k] = [float(x) for x in v]
            except (TypeError, ValueError):
                pass
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "w") as f:
        json.dump(hist, f)
    print(f"  saved history -> {HISTORY}")


def main():
    args = parse_args()
    apply_tag(args.tag)
    preflight()

    t0 = time.time()
    import jax
    import torch
    print(f"jax devices: {jax.devices()}  backend: {jax.default_backend()}")
    print(f"torch cuda: {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    import numpy as np
    import scvi
    from scvi.external import MRVI

    adata = build_hvg_input(args)
    scvi.settings.seed = args.seed

    if not args.skip_train:
        # escape hatch: fit on a random subsample, then transform all cells below
        adata_fit = adata
        if args.subsample and args.subsample < adata.n_obs:
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(adata.n_obs, args.subsample, replace=False))
            adata_fit = adata[idx].copy()
            print(f"subsample: fitting on {adata_fit.n_obs}/{adata.n_obs} cells")
        MRVI.setup_anndata(adata_fit, layer=COUNTS_LAYER,
                           sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)
        model = MRVI(adata_fit, n_latent=args.n_latent, n_latent_u=args.n_latent_u)
        print("training MrVI on the joint atlas...")
        model.train(
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            early_stopping=True, early_stopping_patience=args.patience,
            check_val_every_n_epoch=1, train_size=0.9,
        )
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_DIR), overwrite=True, save_anndata=False)
        save_history(model)
        print(f"  saved model -> {MODEL_DIR}")

    # (re)load against the full atlas so latent/diagnostics cover every cell
    MRVI.setup_anndata(adata, layer=COUNTS_LAYER,
                       sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)
    model = MRVI.load(str(MODEL_DIR), adata=adata)

    # Step C - latent (u = sample-unaware, z = sample-aware) over all cells
    u = model.get_latent_representation(batch_size=args.diag_batch_size, give_z=False)
    LATENT_U_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(LATENT_U_NPY, u)
    print(f"  saved u -> {LATENT_U_NPY}  shape={u.shape}")
    z = model.get_latent_representation(batch_size=args.diag_batch_size, give_z=True)
    np.save(LATENT_Z_NPY, z)
    print(f"  saved z -> {LATENT_Z_NPY}  shape={z.shape}")

    if not args.skip_diag:
        run_diagnostics(model, adata, args)

    print(f"done in {(time.time()-t0)/60:.1f} min")


def _stringify_cat_coords(ds):
    for c in list(ds.coords):
        if str(ds[c].dtype) == "category":
            ds = ds.assign_coords({c: ds[c].astype(str)})
    return ds


def run_diagnostics(model, adata, args):
    """Step D - per-cell-type sample distances + differential abundance."""
    model.update_sample_info(adata)

    DIST_NC.parent.mkdir(parents=True, exist_ok=True)
    print("computing sample distances (groupby=cell_type)...")
    dist = model.get_local_sample_distances(
        keep_cell=False, groupby="cell_type", batch_size=args.diag_batch_size,
    )
    dist = _stringify_cat_coords(dist)
    enc = {v: {"zlib": True, "complevel": 4} for v in dist.data_vars}
    dist.to_netcdf(DIST_NC, engine="h5netcdf", encoding=enc)
    print(f"  -> {DIST_NC}  ({DIST_NC.stat().st_size/1e6:.1f} MB)")

    cov_keys = [k for k in DA_COV_KEYS if k in adata.obs]
    print(f"computing differential abundance ({cov_keys})...")
    da = model.differential_abundance(
        sample_cov_keys=cov_keys,
        compute_log_enrichment=True,
        batch_size=args.diag_batch_size,
    )
    da = _stringify_cat_coords(da)
    enc = {v: {"zlib": True, "complevel": 4} for v in da.data_vars}
    da.to_netcdf(DA_NC, engine="h5netcdf", encoding=enc)
    print(f"  -> {DA_NC}  ({DA_NC.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
