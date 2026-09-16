#!/usr/bin/env python
"""Train MrVI on the CD4 subset of the v2 skin-T atlas, restricted to one gene list.

Used by nb43 (transcriptome malignancy with a held-out study). Two runs of this script
produce the two matched latents the notebook compares:

    --gene-list tables/cd4_hvg_10000.csv --tag hvg10k     # the full-transcriptome arm
    --gene-list tables/cd4_panel_1000.csv --tag panel1k    # the Xenium-sized panel arm

Same cells, same `sample_key`/`batch_key`, same hyper-parameters -- only the gene space
differs, so the 10k-vs-1k gap is the panel effect and nothing else.

The input `data/atlas_joint/cd4_mrvi_input.h5ad` is written by nb43 Step 2 and already
carries the union of both gene lists, so this script only ever subsets columns.

MrVI is unsupervised: training on every cell (including the held-out study's) leaks no
malignancy label. That is what lets the notebook train once and evaluate leave-one-study-out.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# PREALLOCATE THE GPU. With on-demand allocation MrVI training dies with a non-deterministic
# CUDA_ERROR_ILLEGAL_ADDRESS partway through (see the note in run_mrvi_joint.py -- it is
# allocator fragmentation, not the data). Peak GPU use is ~17 GB of 48 GB, so preallocating
# costs nothing under -gpu j_exclusive=yes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

NB_DIR = Path(__file__).resolve().parent.parent
INPUT = NB_DIR / "data" / "atlas_joint" / "cd4_mrvi_input.h5ad"
OUT_DIR = NB_DIR / "data" / "atlas_joint"

SAMPLE_KEY = "sample_id"
BATCH_KEY = "study"
COUNTS_LAYER = "counts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gene-list", required=True,
                   help="CSV with a `gene` column; the model is trained on these genes only")
    p.add_argument("--tag", required=True,
                   help="output suffix, e.g. hvg10k / panel1k")
    # defaults follow run_mrvi_joint.py / the MrVI paper
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--n-latent-u", type=int, default=10)
    p.add_argument("--transform-batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse the existing model directory, only re-emit the latents")
    return p.parse_args()


def main():
    args = parse_args()
    gene_csv = Path(args.gene_list)
    if not gene_csv.is_absolute():
        gene_csv = NB_DIR / gene_csv
    if not INPUT.exists():
        sys.exit(f"missing input: {INPUT}\nrun nb43 Step 2 first")
    if not gene_csv.exists():
        sys.exit(f"missing gene list: {gene_csv}\nrun nb43 Step 2 first")

    model_dir = NB_DIR / "models" / f"mrvi_cd4_{args.tag}"
    latent_u = OUT_DIR / f"cd4_X_mrvi_u_{args.tag}.npy"
    latent_z = OUT_DIR / f"cd4_X_mrvi_z_{args.tag}.npy"
    barcodes = OUT_DIR / f"cd4_mrvi_barcodes_{args.tag}.npy"
    history = NB_DIR / "figures" / f"mrvi_cd4_{args.tag}_history.json"

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

    genes = pd.read_csv(gene_csv)["gene"].astype(str).tolist()
    print(f"loading {INPUT}")
    adata = sc.read_h5ad(INPUT)
    keep = adata.var_names.isin(genes)
    missing = sorted(set(genes) - set(adata.var_names))
    assert not missing, (
        f"{len(missing)} genes of {gene_csv.name} are absent from the input "
        f"(first few: {missing[:5]}) -- nb43 Step 2 must write the union of every gene list")
    adata = adata[:, keep].copy()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes  |  "
          f"{adata.obs[SAMPLE_KEY].nunique()} samples, {adata.obs[BATCH_KEY].nunique()} studies")

    scvi.settings.seed = args.seed
    # nb43 writes the raw counts straight into X to keep the input file half the size; fall back
    # to a `counts` layer if one is present, so this also works on a conventionally-layered input.
    layer = COUNTS_LAYER if COUNTS_LAYER in adata.layers else None
    print(f"  counts read from {'layers[' + COUNTS_LAYER + ']' if layer else 'X'}")
    MRVI.setup_anndata(adata, layer=layer, sample_key=SAMPLE_KEY, batch_key=BATCH_KEY)

    if args.skip_train:
        model = MRVI.load(str(model_dir), adata=adata)
        print(f"  reused model {model_dir}")
    else:
        model = MRVI(adata, n_latent=args.n_latent, n_latent_u=args.n_latent_u)
        print(f"training MrVI [{args.tag}] ...")
        model.train(
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            early_stopping=True, early_stopping_patience=args.patience,
            check_val_every_n_epoch=1, train_size=0.9,
        )
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir), overwrite=True, save_anndata=False)
        hist = {}
        for k, v in (model.history or {}).items():
            if isinstance(v, pd.DataFrame):
                hist[k] = v.iloc[:, 0].astype(float).tolist()
            else:
                try:
                    hist[k] = [float(x) for x in v]
                except (TypeError, ValueError):
                    pass
        history.parent.mkdir(parents=True, exist_ok=True)
        with open(history, "w") as f:
            json.dump(hist, f)
        print(f"  saved model -> {model_dir}\n  saved history -> {history}")

    u = model.get_latent_representation(batch_size=args.transform_batch_size, give_z=False)
    z = model.get_latent_representation(batch_size=args.transform_batch_size, give_z=True)
    np.save(latent_u, u)
    np.save(latent_z, z)
    # barcodes make the latents self-describing: nb43 reindexes on them and asserts the match,
    # so a stale .npy from a different cell subset fails loudly instead of silently mis-aligning.
    np.save(barcodes, adata.obs_names.to_numpy().astype(str))
    print(f"  saved u -> {latent_u}  {u.shape}\n"
          f"  saved z -> {latent_z}  {z.shape}\n"
          f"  saved barcodes -> {barcodes}  {adata.n_obs}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
