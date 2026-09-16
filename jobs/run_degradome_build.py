#!/usr/bin/env python
"""Build the MF/CTCL skin degradome object for nb42.

Streams data/atlas_joint/joint_annotated.h5ad (48 GB, 1,173,694 x 40,821 raw counts) one
row-chunk at a time, keeps the 720,125 skin cells that carry a degradome level, and writes:

  data/degradome/skin_degradome.h5ad                     720,125 x 156 panel genes
  data/degradome/skin_degradome_pseudobulk_full.parquet  (level x sample) x 40,821 counts
  data/degradome/skin_degradome_pseudobulk_meta.parquet  design metadata per pseudo-sample
  data/degradome/degradome_equivalence_testset.h5ad      3 donors at FULL gene width
  data/degradome/degradome_build_manifest.json

Why a job and not a notebook cell: the panel needs ADAM10 and ADAM17, which are in nb31's
protease panel and are NOT in joint_mrvi_input_skin.h5ad's 10k HVGs (both are ubiquitous, so
they are never highly variable). Comparing against nb31 while silently dropping two of its
twelve genes is not acceptable, so the object is rebuilt from the full-gene source.

Correctness note the whole notebook rests on: the library-size factor is summed over all
40,821 genes and only THEN are the columns subset to the panel. Normalising after subsetting
would rescale every cell by (panel counts / total counts) -- roughly 40x here, and varying by
cell type, which is precisely the axis nb42 measures. The equivalence test set exists to gate
exactly this; nb42 section 0 asserts it before anything else runs.

Pure I/O, no GPU. Runs in mrvi_env (ccc_helpers imports liana; neural_nmf_env has no liana).
"""
import argparse
import sys
import time
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR))

import degradome_data as cfg      # noqa: E402
import degradome_helpers as D     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=50_000, help="rows per streaming chunk")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-testset", action="store_true",
                   help="skip the full-gene equivalence test set (nb42 section 0 will fail)")
    return p.parse_args()


def preflight():
    miss = [str(p) for p in (cfg.SOURCE_H5AD, cfg.OBS_PARQUET, cfg.SUBTYPE_CSV)
            if not p.exists()]
    if miss:
        sys.exit("missing inputs:\n  " + "\n  ".join(miss))


def main():
    args = parse_args()
    preflight()

    outputs = [cfg.DEG_ADATA, cfg.PSEUDOBULK_PQ, cfg.PSEUDOBULK_META, cfg.MANIFEST]
    if all(p.exists() for p in outputs) and not args.force:
        sys.exit("outputs exist; pass --force\n  " + "\n  ".join(str(p) for p in outputs))

    cfg.DEG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] source={cfg.SOURCE_H5AD}")
    print(f"[{time.strftime('%H:%M:%S')}] panel={len(cfg.panel_genes())} genes "
          f"({len(cfg.panel_genes(False))} degradome + {len(cfg.CONTEXT_GENES)} identity)")
    print(f"[{time.strftime('%H:%M:%S')}] chunk={args.chunk}")

    adata, manifest = D.build_degradome_object(
        source=cfg.SOURCE_H5AD,
        obs_parquet=cfg.OBS_PARQUET,
        out=cfg.DEG_ADATA,
        pseudobulk_out=cfg.PSEUDOBULK_PQ,
        pseudobulk_meta_out=cfg.PSEUDOBULK_META,
        testset_out=None if args.skip_testset else cfg.TESTSET_H5AD,
        manifest_out=cfg.MANIFEST,
        chunk=args.chunk,
        verbose=True,
    )

    print(f"\n[{time.strftime('%H:%M:%S')}] done in {(time.time() - t0) / 60:.1f} min")
    print(f"  {cfg.DEG_ADATA}  {adata.shape}")
    print(f"  donors={manifest['n_donors']}  samples={manifest['n_samples']}")
    print(f"  dropped {manifest['n_dropped_no_identity']:,} of "
          f"{manifest['n_skin_cells']:,} skin cells with no identity")
    print(f"  pseudobulk {manifest['pseudobulk_shape']}")


if __name__ == "__main__":
    main()
