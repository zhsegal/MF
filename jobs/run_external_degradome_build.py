#!/usr/bin/env python
"""Build the external healthy-skin degradome object for nb28 §11.

Reads data/external_ref/reynolds2021_healthy_skin.h5ad (Reynolds 2021 / E-MTAB-8142,
195,739 healthy adult skin cells over 5 donors) and writes:

  data/degradome/external_reynolds_degradome.h5ad   kept cells x ~288 panel genes
  tables/deg28_external_panel_coverage.csv          per-family gene presence
  tables/deg28_external_donor_median.csv            per (target, donor) family medians

The normalisation is the same order as the main build -- library size over the FULL gene width
first, panel subset second -- because the whole point of the object is to be comparable with
ours. helpers/external_ref.build_external enforces that; this script is the job wrapper.

Not the login node: the source CSR is streamed in row blocks, but the two output matrices and
the symbol-collapse operator are resident, and the scoring pass touches every cell.

Usage:
  python run_external_degradome_build.py
  python run_external_degradome_build.py --chunk 40000 --force
"""
import argparse
import sys
import time
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import degradome_data as cfg       # noqa: E402
import degradome_cells as DC       # noqa: E402
import external_ref as X           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=20_000, help="rows per streaming chunk")
    p.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if not cfg.EXT_H5AD.exists():
        sys.exit(f"missing {cfg.EXT_H5AD}\n  run ./jobs/download_external_skin.sh first")
    if cfg.EXT_DEG_ADATA.exists() and not args.force:
        sys.exit(f"{cfg.EXT_DEG_ADATA} exists -- pass --force to rebuild")

    print(f"[{time.strftime('%H:%M:%S')}] describing the source")
    desc = X.describe(cfg.EXT_H5AD)
    assert desc["_shape"][0] == cfg.EXT_N_OBS, (desc["_shape"][0], cfg.EXT_N_OBS)
    X.assert_label_map(desc[X.LABEL_COL])

    print(f"[{time.strftime('%H:%M:%S')}] building the panel object")
    adata = X.build_external(cfg.EXT_H5AD, out=cfg.EXT_DEG_ADATA, chunk=args.chunk)

    print(f"[{time.strftime('%H:%M:%S')}] scoring families")
    scored = DC.family_score_cells(adata)
    DC.balance_cells(adata)
    score_cols = ([f"fam_{f}" for f in scored]
                  + [c for c in adata.obs.columns if c.startswith("bal_")])

    dmed = X.external_donor_median(adata, score_cols)
    out_csv = cfg.tab("external_donor_median")
    dmed.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  {dmed.shape}")

    print("\nper-target donor coverage (eligible = >= "
          f"{cfg.MIN_CELLS} cells):")
    cov = (dmed.groupby("target", observed=True)
           .agg(n_donors=("eligible", "size"), n_eligible=("eligible", "sum"),
                n_cells=("n_cells", "sum"))
           .sort_values("n_cells", ascending=False))
    print(cov.to_string())
    thin = cov.index[cov["n_eligible"] < cfg.MIN_DONORS_PER_ARM].tolist()
    if thin:
        print(f"\nbelow MIN_DONORS_PER_ARM={cfg.MIN_DONORS_PER_ARM}, will abstain in §11: {thin}")
    print(f"\nour levels with no external counterpart: {X.NO_EXTERNAL}")
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
