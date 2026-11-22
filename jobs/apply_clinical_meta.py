#!/usr/bin/env python
"""Write the curated clin_* columns into every built v2 object and the obs cache.

The columns come from data/clinical_metadata_v1.csv (jobs/build_clinical_metadata.py). Only
clin_* columns are created or replaced; no existing obs column is touched, so no rollback snapshot
is needed. Obs-only: no matrix is opened, so it runs on the login node.

    python jobs/apply_clinical_meta.py              # dry run, prints the per-file diff
    python jobs/apply_clinical_meta.py --apply      # write
    python jobs/apply_clinical_meta.py --apply      # again: reports nothing to do

Idempotent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB_DIR / "helpers"))
import atlas_v2_fixups as F  # noqa: E402
import clinical_meta as CM  # noqa: E402

# the fixups manifest plus the degradome object that 28_tme_degradome_mc / 28b read
MANIFEST = F.MANIFEST + ["data/degradome/skin_degradome.h5ad"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = ap.parse_args()
    dry = not args.apply

    clin = CM.load_clinical()
    rd = CM.real_donor_by_cell()              # for objects whose obs dropped real_donor
    print(f"{'DRY RUN' if dry else 'APPLYING'} — {len(clin)} clinical rows, "
          f"{len(CM.OBS_COLS) + len(CM.OBS_NUM_COLS)} clin_* columns, {len(MANIFEST)} objects\n")

    n, locked = 0, []
    for rel in MANIFEST:
        p = NB_DIR / rel
        if not p.exists():
            print(f"  {p.name}: MISSING — skipped")
            continue
        try:
            n += bool(CM.patch_h5ad_obs_clinical(p, clin, rd_by_cell=rd, dry_run=dry))
        except BlockingIOError:              # open in a kernel or a job; never force the lock
            print(f"  {p.name}: LOCKED by another process — skipped")
            locked.append(p.name)
    if CM.OBS_PARQUET.exists():
        n += bool(CM.patch_parquet_clinical(CM.OBS_PARQUET, clin, dry_run=dry))
    print(f"\n{n} objects {'would change' if dry else 'changed'}")
    if locked:
        print(f"{len(locked)} locked, rerun once they are closed: {', '.join(locked)}")
    return 1 if locked else 0


if __name__ == "__main__":
    sys.exit(main())
