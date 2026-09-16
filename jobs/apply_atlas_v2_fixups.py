#!/usr/bin/env python
"""Propagate the v2 metadata corrections into every built atlas object.

The corrections and the manifest live in ``atlas_v2_fixups.py``; this only drives them and
verifies the result. Obs-only: a few MB of categorical codes per file, no matrix is opened, so it
runs on the login node in under a minute even against the 112 GB object.

    python jobs/apply_atlas_v2_fixups.py                 # dry run, prints the per-file diff
    python jobs/apply_atlas_v2_fixups.py --apply         # write, then verify
    python jobs/apply_atlas_v2_fixups.py --apply         # again: reports nothing to do

Idempotent. A rollback snapshot of every touched cell is written before anything else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

NB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB_DIR))
import atlas_v2_fixups as F  # noqa: E402

PARQUET = NB_DIR / "data" / "atlas_joint" / "atlas_obs_full_v2.parquet"
ROLLBACK = NB_DIR / "tables" / "atlas_v2_fixup_rollback.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = ap.parse_args()
    dry = not args.apply

    print(f"{'DRY RUN' if dry else 'APPLYING'} — {len(F.MANIFEST)} objects + the obs cache\n")

    if not dry and not ROLLBACK.exists():
        F.rollback_table([NB_DIR / p for p in F.MANIFEST], ROLLBACK)
        print()

    total = {}
    print("h5ad objects")
    for rel in F.MANIFEST:
        p = NB_DIR / rel
        if not p.exists():
            print(f"  {p.name}: MISSING — skipped")
            continue
        for col, n in F.patch_h5ad_obs(p, dry_run=dry).items():
            total[col] = total.get(col, 0) + n

    print("\nobs cache")
    if PARQUET.exists():
        for col, n in F.patch_parquet(PARQUET, dry_run=dry).items():
            total[col] = total.get(col, 0) + n
    else:
        print(f"  {PARQUET.name}: MISSING — skipped")

    print("\nlegacy objects on pre-v2 cell sets, deliberately not patched:")
    for rel in F.LEGACY:
        print(f"  {Path(rel).name}")

    print("\ntotal cell-values " + ("to change" if dry else "changed") + ":")
    for c, n in sorted(total.items()):
        print(f"  {c:<22} {n:>9,}")
    if not total:
        print("  nothing — every object already agrees with the fix-ups")

    if not dry:
        print("\nverifying…")
        bad = 0
        for rel in F.MANIFEST:
            p = NB_DIR / rel
            if p.exists() and F.patch_h5ad_obs(p, dry_run=True, verbose=False):
                print(f"  FAIL {p.name} still differs")
                bad += 1
        if PARQUET.exists():
            A = pd.read_parquet(PARQUET)
            left = F.apply_fixups(A, verbose=False)
            for c in ("sample_id", "organ", "entity", "stage_clean", "stage_class"):
                if c in A and (A[c].astype(str).to_numpy() != left[c].astype(str).to_numpy()).any():
                    print(f"  FAIL parquet {c} still differs")
                    bad += 1
        print("  all objects agree with the fix-ups" if not bad else f"  {bad} object(s) FAILED")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
