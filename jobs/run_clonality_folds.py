#!/usr/bin/env python
"""Per-fold training of the malignancy-axis arms (nb38 Part A writes the config).

Leave-3-samples-out over the TCR/ALICE-labelled skin donors. For each fold the held-out
donors' cells are removed from training entirely (the `perturbation_trial` protocol), the model
is trained on everything else, and then *all* cells — training, held-out and the TCR-less "dark"
donors — are encoded with that fold's model. The notebook consumes the saved latents.

Arms share the protocol exactly (folds, cells, HVGs, epochs, KL schedule, batch/labels keys, seed)
and differ only in what the notebook writes into `models`:

    structural   SemanticSCVI, decoder_mode='structural'      (loadings ARE a function of the map)
    structural_shuffled  same, `shuffle_map` set              (matched no-prior control for it)
    semantic     SemanticSCVI, coherence_weight=1000          (penalty decoder, geometric prior)
    ldvae        SemanticSCVI, coherence_weight=0             (no-prior control for the PENALTY arm
                                                               only -- inert under 'structural')
    scvi         SCVI, use_batch_norm="encoder", likelihood nb (nonlinear decoder)

Config + slim input (adata + semantic_map) come from the "write job input" cell of
38_delta_gene_axis.ipynb Part A. Submit via notebooks/MF/jobs/run_clonality_folds.sh.

Usage:
    python run_clonality_folds.py                        # every run in the config
    python run_clonality_folds.py --folds scvi_fold3      # one run by name
    python run_clonality_folds.py --folds 3               # fold 3 of every arm
    python run_clonality_folds.py --force                # retrain, ignore caches
"""
import argparse
import json
import sys
import time
from pathlib import Path

NB_MF = Path(__file__).resolve().parent.parent      # notebooks/MF
NB = NB_MF.parent                                   # notebooks
for _p in (str(NB_MF), str(NB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scanpy as sc
import torch

from benchmark_helpers import (
    plot_training_curves,
    train_or_load_scvi,
    train_or_load_semantic_scvi,
)

CONFIG = NB_MF / "jobs" / "clonality_folds_config.json"


def _batch_key(cfg, mcfg):
    """Batch key for one arm: an arm may override the cohort-wide one.

    `structural_donor` uses `donor` (70 categories) so donor effects land in the decoder's batch
    offset rather than in the 10-d latent. That arm is all-donor only: holding a donor out would
    delete its batch category from training and leave its cells unencodable.
    """
    return mcfg.get("batch_key", cfg["batch_key"])


def _fit(model_type, adata_tr, semantic_map, cfg, mcfg, cache_dir, force):
    """Dispatch to the trainer for one arm. Shared knobs come from cfg, arm knobs from mcfg."""
    shared = dict(
        cache_dir=cache_dir,
        force_train=force,
        max_epochs=cfg["max_epochs"],
        n_epochs_kl_warmup=cfg["n_epochs_kl_warmup"],
        batch_key=_batch_key(cfg, mcfg),
        labels_key=cfg["labels_key"],
    )
    if model_type == "scvi":
        return train_or_load_scvi(adata_tr, **shared, **mcfg["kwargs"])
    # `shuffle_map` permutes the embedding ROWS among the vocabulary-covered genes: the geometry
    # (cosine distribution, anisotropy, cluster structure) is untouched and only the gene-to-vector
    # assignment is destroyed. This is the matched no-prior control for decoder_mode='structural',
    # where coherence_weight is inert -- the semantic map IS the decoder, so switching the penalty
    # off changes nothing at all.
    smap = semantic_map
    if mcfg.get("shuffle_map") is not None:
        from benchmark_helpers import shuffled_semantic_map

        smap = shuffled_semantic_map(semantic_map, seed=int(mcfg["shuffle_map"]))
    return train_or_load_semantic_scvi(
        adata_tr, smap,
        warmup_epochs=mcfg.get("warmup_epochs", cfg["warmup_epochs"]),
        **shared, **{k: v for k, v in mcfg["kwargs"].items() if k != "shuffle_map"},
    )


def _train_one(adata, semantic_map, cfg, run, out_dir, force):
    """Train one run (an arm x fold, or an arm's all-donor reference); save its latent for ALL cells."""
    name = run["name"]
    model_type = run.get("model", "semantic")
    mcfg = cfg["models"][model_type]
    z_path = out_dir / f"z_{name}.npy"
    cache_dir = Path(cfg["model_cache_dir"]) / model_type / run["cache_key"]
    held = set(run["held_out"])

    if z_path.exists() and not force:
        print(f"[{name}] latent cached -> {z_path.name}", flush=True)
        return

    batch_key = _batch_key(cfg, mcfg)
    keep = ~adata.obs[cfg["donor_key"]].astype(str).isin(held).to_numpy()
    pool_key = cfg.get("train_pool_key")
    if pool_key:                       # per-donor cell cap, decided once in the notebook
        keep &= adata.obs[pool_key].to_numpy(dtype=bool)
    print(f"\n=== {name} === model={model_type} batch={batch_key} "
          f"held_out={sorted(held) or '(none)'} "
          f"| train {int(keep.sum())}/{adata.n_obs} cells | cache={cache_dir}", flush=True)

    adata_tr = adata[keep].copy()                    # scvi UUID collision -> own copy
    # every batch category must survive in the training set, else the held-out cells cannot
    # be encoded (unseen category has no trained embedding)
    miss = set(adata.obs[batch_key].astype(str)) - set(adata_tr.obs[batch_key].astype(str))
    if miss:
        raise SystemExit(f"[{name}] fold empties batch category/ies {sorted(miss)} — fix the folds")

    model = _fit(model_type, adata_tr, semantic_map, cfg, mcfg, cache_dir, force)
    plot_training_curves(model, name, out_dir / f"training_curves_{name}.png",
                         semantic=(model_type != "scvi"))

    # encode EVERY cell with this fold's model (held-out donors included, dark donors included)
    adata_all = adata.copy()
    z = np.asarray(model.get_latent_representation(adata_all), dtype=np.float32)
    assert z.shape[0] == adata.n_obs, f"{z.shape=} vs {adata.n_obs=}"
    np.save(z_path, z)
    print(f"[{name}] saved {z_path.name} {z.shape}", flush=True)
    del adata_tr, adata_all, model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None,
                    help="run names (e.g. scvi_fold3), or a bare fold id / 'all' to take that "
                         "fold across every arm; default = every run in the config")
    ap.add_argument("--force", action="store_true", help="retrain even if a cache/latent exists")
    args = ap.parse_args()

    if not CONFIG.exists():
        sys.exit(f"missing {CONFIG}\nrun the 'write job input' cell of "
                 "38_delta_gene_axis.ipynb (Part A) first")
    cfg = json.loads(CONFIG.read_text())

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(cfg["input_h5ad"])
    semantic_map = torch.load(cfg["semantic_map"], weights_only=False)
    print(f"input {adata.shape} | semantic_map {tuple(semantic_map.shape)} "
          f"| runs {len(cfg['runs'])}", flush=True)
    assert semantic_map.shape[0] == adata.n_vars, (semantic_map.shape, adata.shape)

    runs = cfg["runs"]
    if args.folds:
        # accept a full run name (scvi_fold3), or a bare fold shorthand (3 / fold3 / all) that
        # expands across every arm
        want = set(args.folds) | {f"fold{f}" for f in args.folds}
        runs = [r for r in runs
                if r["name"] in want or r.get("cache_key") in want]
        if not runs:
            sys.exit(f"no run matched {args.folds}; available: {[r['name'] for r in cfg['runs']]}")

    for run in runs:
        _train_one(adata, semantic_map, cfg, run, out_dir, args.force)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s", flush=True)
