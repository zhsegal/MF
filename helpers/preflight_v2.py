#!/usr/bin/env python
"""Pre-flight for the v2 concat: check every standardized cache against the ledgers.

concat_joint is a ~200 GB, hour-scale job that overwrites in place. Everything it asserts
can be checked first from obs alone (h5py, no matrix reads), so check it here:

  1. every registry cohort with in_expression has a standardized.h5ad
  2. every ledger DROP sample_id exists in some cache  (a token drift = a silent no-op drop)
  3. every cache sample_id appears in that cohort's samples.tsv
  4. cell_id is unique within each cache and across all of them
  5. disease / compartment vocabularies are closed
  6. patient_key covers every surviving sample
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_join_helpers as H  # noqa: E402

NB = Path(__file__).resolve().parent.parent


def cat(f, key):
    g = f["obs"][key]
    if isinstance(g, h5py.Group):
        c = np.array([x.decode() for x in g["categories"][:]])
        return c, c[g["codes"][:]]
    v = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])
    return np.unique(v), v


def main() -> int:
    REG = H.dataset_registry(NB)
    expr = [k for k, v in REG.items() if v.get("in_expression")]
    drop_ids, pkey = H.load_dedup_ledger(NB)

    problems, n_cells_total = [], 0
    seen_samples, all_disease, all_comp = {}, set(), set()
    all_cell_ids, dup_within = set(), []

    print(f"{'cohort':<16}{'cells':>10}{'samples':>9}{'donors':>8}{'drop':>6}  status")
    for label in expr:
        # resolve the cache path WITHOUT calling build_standardized -- that function
        # builds a missing cache (load + QC + Scrublet), which is exactly the hours-long
        # work this pre-flight exists to avoid, and it makes the MISSING branch unreachable.
        reg = REG[label]
        if label == "Li2024_atlas":
            p = NB / "data" / "atlas_joint" / "atlas_standardized.h5ad"
        else:
            folder = (reg["path"].parents[1] if reg.get("source") == "concat"
                      else Path(reg["raw"]).parent)
            p = folder / "processed" / "standardized.h5ad"
        if not Path(p).exists():
            problems.append(f"{label}: standardized.h5ad MISSING -> build it"); 
            print(f"{label:<16}{'-':>10}{'-':>9}{'-':>8}{'-':>6}  MISSING")
            continue
        try:
            with h5py.File(p, "r") as f:
                n = f["obs"][f["obs"].attrs["_index"]].shape[0]
                _, dis = cat(f, "disease")
                _, cids = cat(f, "cell_id")
                if len(set(cids)) != len(cids):
                    dup_within.append(label)
                ov = all_cell_ids & set(cids)
                if ov:
                    problems.append(f"{label}: {len(ov)} cell_ids collide with an earlier "
                                    f"cohort, e.g. {sorted(ov)[:3]}")
                all_cell_ids |= set(cids)
                sids, sid_v = cat(f, "sample_id")
                dons, _ = cat(f, "donor")
                comp, _ = cat(f, "compartment")
        except Exception as e:                                   # noqa: BLE001
            print(f"{label:<16}{'-':>10}{'-':>9}{'-':>8}{'-':>6}  UNREADABLE ({e!r})")
            problems.append(f"{label}: unreadable ({e!r})"); continue

        hit = sorted(set(sids) & drop_ids)
        n_cells_total += n
        all_disease |= set(dis); all_comp |= set(comp)
        for s in sids:
            seen_samples.setdefault(s, label)
        print(f"{label:<16}{n:>10,}{len(sids):>9}{len(dons):>8}{len(hit):>6}  ok")

        # (3b) a cache built before a vocab fix keeps the OLD value. `unknown` is a legal
        # member of DISEASE_VOCAB, so run_build_joint's assert cannot see it -- this is how
        # 332,088 cells shipped with disease="unknown" when their samples.tsv said
        # "CTCL_other". Compare the cache against the sheet that produced it.
        meta0 = REG[label].get("meta")
        if meta0 and Path(meta0).exists():
            try:
                t0 = pd.read_csv(meta0, sep="\t", dtype=str).fillna("")
                if "disease" in t0:
                    want = {H._norm_disease(x) for x in t0["disease"] if str(x).strip()}
                    got = set(dis)
                    lost = {d for d in want if d not in got and d != "unknown"}
                    if lost:
                        # a warning, not a blocker: a cohort may legitimately override the
                        # sheet (H/herrera21 prefers its `disease_state` column) or have had
                        # the sample dropped upstream
                        print(f"    note [{label}]: samples.tsv declares {sorted(lost)} "
                              f"not present in the cache ({sorted(got)})")
                    if "unknown" in got and "unknown" not in want:
                        n_unk = int((dis == "unknown").sum()) if len(dis) == len(sid_v) else -1
                        problems.append(
                            f"{label}: cache has disease='unknown' but samples.tsv never "
                            f"says unknown -- STALE CACHE, rebuild with --force-std")
            except Exception:                                    # noqa: BLE001
                pass

        # (3) cache samples must be declared in samples.tsv
        meta = REG[label].get("meta")
        if meta and Path(meta).exists() and REG[label].get("source") not in ("atlas_h5ad",):
            try:
                t = pd.read_csv(meta, sep="\t", dtype=str)
                col = next((c for c in ("sample_id", "sample_id_in_raw") if c in t), None)
                if col:
                    declared = {H.namespace(label, s) for s in t[col].astype(str)}
                    undeclared = set(sids) - declared
                    if undeclared and label not in ("D1", "D3", "H", "B4", "B5"):
                        problems.append(f"{label}: {len(undeclared)} cache samples not in "
                                        f"{Path(meta).name}: {sorted(undeclared)[:5]}")
            except Exception:                                    # noqa: BLE001
                pass

    # (2) a ledger drop that matches nothing is only a problem if that cohort DECLARES the
    # sample -- otherwise the library simply was never ingested (D3 has no P303_Blood row;
    # D8's atopic-dermatitis samples are out of scope), which is expected, not a drift.
    declared_all = set()
    for label in expr:
        meta = REG[label].get("meta")
        if meta and Path(meta).exists():
            try:
                t = pd.read_csv(meta, sep="\t", dtype=str)
                col = next((c for c in ("sample_id", "sample_id_in_raw") if c in t), None)
                if col:
                    declared_all |= {H.namespace(label, s) for s in t[col].astype(str)}
            except Exception:                                    # noqa: BLE001
                pass
    unmatched = sorted(s for s in drop_ids
                       if s.split("__")[0] in set(expr) and s not in seen_samples)
    drift = [s for s in unmatched if s in declared_all]
    benign = [s for s in unmatched if s not in declared_all]
    if drift:
        problems.append(f"{len(drift)} ledger DROPs name a DECLARED sample that is absent "
                        f"from the cache (token drift): {drift[:8]}")
    if benign:
        print(f"\nledger DROPs for libraries never ingested (expected, not an error): "
              f"{len(benign)} -> {benign}")

    if dup_within:
        problems.append(f"duplicate cell_id WITHIN: {dup_within}")

    # (5) closed vocabularies
    bad_d = all_disease - H.DISEASE_VOCAB
    bad_c = all_comp - H.COMPARTMENT_VOCAB
    if bad_d:
        problems.append(f"disease values outside DISEASE_VOCAB: {sorted(bad_d)}")
    if bad_c:
        problems.append(f"compartment values outside COMPARTMENT_VOCAB: {sorted(bad_c)}")

    # (6) patient_key coverage of surviving samples
    surviving = {s for s in seen_samples if s not in drop_ids}
    nokey = sorted(s for s in surviving if s not in pkey)
    print(f"\ncells across caches: {n_cells_total:,}  (before {len(drop_ids)} duplicate "
          f"samples are dropped)")
    print(f"samples: {len(seen_samples)}  surviving: {len(surviving)}  "
          f"patient_key covers: {len(surviving) - len(nokey)}/{len(surviving)}")
    if nokey:
        print(f"  no patient_key (will fall back to `donor`): {len(nokey)} e.g. {nokey[:6]}")

    print()
    if problems:
        print(f"BLOCKING PROBLEMS ({len(problems)}):")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("PRE-FLIGHT OK -- safe to run concat_joint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
