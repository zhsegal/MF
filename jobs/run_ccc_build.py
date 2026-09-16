#!/usr/bin/env python
"""Build the MF/CTCL skin CCC object for the LIANA analysis (nb34-36).

Streams data/atlas_joint/joint_annotated.h5ad (48 GB, 1,173,694 x 40,821 raw
counts) one row-chunk at a time, keeps the ~749k skin cells that carry a CCC
label, and writes:

  data/ccc/ccc_skin.h5ad                    ~727k x ~1,832 LR-resource genes
  data/ccc/ccc_skin_pseudobulk_full.parquet (ccc_celltype x sample) x 40,821 counts
  data/ccc/ccc_skin_pseudobulk_meta.parquet design metadata + n_cells per pseudo-sample
  data/ccc/ccc_equivalence_testset.h5ad     3 donors at FULL gene width
  data/ccc/ccc_build_manifest.json

Correctness note that the whole pipeline rests on: the library-size factor is
summed over all 40,821 genes and only THEN are the columns subset to the resource
genes. Normalising after subsetting would rescale every cell by the ratio of
resource-gene counts to total counts (~16x) and produce plausible, wrong lr_means.
nb34 section 5 gates on this by re-running rank_aggregate over the full-gene test
set and asserting the statistics are identical.

Pure I/O (~15 min) -- no GPU. Runs in mrvi_env; neural_nmf_env has no liana.
"""
import argparse
import sys
import time
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR))

import ccc_data as cd            # noqa: E402
import ccc_helpers as C          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=50_000, help="rows per streaming chunk")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-testset", action="store_true", help="skip the equivalence test set")
    return p.parse_args()


def preflight():
    miss = [str(p) for p in (cd.SOURCE_H5AD, cd.OBS_PARQUET) if not p.exists()]
    if miss:
        sys.exit("missing inputs:\n  " + "\n  ".join(miss))


def main():
    args = parse_args()
    preflight()

    outputs = [cd.CCC_ADATA, cd.PSEUDOBULK_PQ, cd.PSEUDOBULK_META, cd.MANIFEST]
    if all(p.exists() for p in outputs) and not args.force:
        sys.exit("outputs exist; pass --force\n  " + "\n  ".join(str(p) for p in outputs))

    cd.CCC_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] source={cd.SOURCE_H5AD}")
    print(f"[{time.strftime('%H:%M:%S')}] chunk={args.chunk}")

    adata, manifest = C.build_ccc_object(
        source=cd.SOURCE_H5AD,
        obs_parquet=cd.OBS_PARQUET,
        out=cd.CCC_ADATA,
        pseudobulk_out=cd.PSEUDOBULK_PQ,
        pseudobulk_meta_out=cd.PSEUDOBULK_META,
        testset_out=None if args.skip_testset else cd.TESTSET_H5AD,
        manifest_out=cd.MANIFEST,
        chunk=args.chunk,
        verbose=True,
    )

    print(f"\n[{time.strftime('%H:%M:%S')}] done in {(time.time() - t0) / 60:.1f} min")
    print(f"  {cd.CCC_ADATA}  {adata.shape}")
    print(f"  donors={manifest['n_donors']}  samples={manifest['n_samples']}")
    print(f"  genes kept {manifest['n_vars']}/{manifest['n_genes_source']}")


if __name__ == "__main__":
    main()
