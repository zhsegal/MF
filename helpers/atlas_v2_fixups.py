"""Corrections to v2 atlas metadata found by the release QC (45_atlas_v2_qc.ipynb).

These are obs-level fixes, not data fixes. Each one is expressed twice: as a patch to
``data/sample_metadata_final.csv``, which is what ``run_build_joint.py`` broadcasts from, so the
next atlas rebuild picks it up; and as ``apply_fixups(obs)``, so an analysis running against the
*current* built objects can get the corrected columns without waiting for a rebuild.

Until the atlas is rebuilt, ``joint_annotated.h5ad``, the ``joint_mrvi_input*`` objects and
``atlas_obs_full_v2.parquet`` all still carry the pre-fix values. That is deliberate: silently
patching some caches and not others is the divergence that made nb30 report v1 numbers on a v2
input. Call ``apply_fixups`` explicitly, or rebuild.

Both fixes are idempotent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --- fix 1: stage lost between deposits ------------------------------------------------------
# Six Vienna patients were deposited in two GEO series each. One deposit records the stage, the
# other leaves it NA, so a stage-stratified analysis silently drops those cells. No patient has
# two *different* known stages, so propagating the single known value is unambiguous.
STAGE_BY_SAMPLE = {
    "D1__P76":         ("IVA1", "advanced"),   # vienna:MF309, known from D7__MF309_*
    "D1__P311_thick":  ("IVA1", "advanced"),   # vienna:MF311, known from D7__MF311_*
    "D9__P72":         ("IVA1", "advanced"),   # vienna:MF311, same patient
    "D1__P312_thick":  ("IIB",  "advanced"),   # vienna:MF312, known from D7__MF312_*
    "D1__P303_skin":   ("IVB",  "advanced"),   # vienna:P303,  known from D5__MFIVB_*
    "D8__P65_LPD3_MF": ("IB",   "early"),      # vienna:P65,   known from D7__P65_nonlesional
    "D8__P90_LPD4_MF": ("IB",   "early"),      # vienna:P90,   known from D7__P90_nonlesional
}

# --- fix 2: one hashtag lane holding two specimens -------------------------------------------
# gaydosik2022's `SZ29` lane is HTO-multiplexed over two specimens of one patient -- see
# data/D13_gaydosik2022_skin_blood/meta/hto_map.tsv:
#     SZ29  HTO1  donor SZ29   Blood  PBMC        SS
#     SZ29  HTO2  donor MF25   Skin   Skin tumor  MF
# The demux is correct: per-cell donor, compartment and tissue all match the map. What is wrong
# is that both specimens carry sample_id D13__SZ29, so the sample-level broadcast (organ, entity)
# describes only the blood half. Splitting the sample_id costs no cells and makes D13__SZ29 stop
# being the one sample_id in the atlas that spans two compartments.
# Both halves need correcting, because one sheet row cannot describe two specimens: the skin half
# inherits organ=blood from the lane, and the blood half inherits entity=MF from it. The blood is
# the only SS-diseased block in the atlas labelled MF_classic; its sibling Sezary samples in the
# same cohort (D13__SZ16, D13__SZ22) carry entity=SS / entity_h=Sezary, which is what it gets.
SPLIT_SAMPLE = {
    "sample_id": "D13__SZ29",
    "skin": {"when": {"donor": "D13__MF25"},
             "set": {"sample_id": "D13__MF25_skin", "organ": "skin",
                     "entity": "MF", "entity_h": "MF_classic"}},
    "blood": {"when": {"donor": "D13__SZ29"},
              "set": {"entity": "SS", "entity_h": "Sezary"}},
}

# --- known, deliberately unfixed -------------------------------------------------------------
# The dedup gate scored the whole SZ29 lane as one unit and dropped D15__MF25 (9,275 barcodes of
# the same patient's skin, il4ra2026) in its favour, because the lane "has usable V(D)J". So the
# atlas keeps a 631-cell version of that skin specimen rather than a 9,275-cell one. Reversing it
# means re-running the ledger and rebuilding, so it is recorded rather than corrected.
DEDUP_NOTE = (
    "G033: D15__MF25 (9,275 barcodes) dropped in favour of D13__SZ29, which contributes only "
    "631 cells of that specimen. Winner chosen on V(D)J availability, not cell count."
)


def apply_fixups(obs: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return a copy of an atlas obs frame with both corrections applied."""
    obs = obs.copy()
    for col in ("sample_id", "organ", "entity", "entity_h", "stage_clean", "stage_class", "donor"):
        if col in obs and isinstance(obs[col].dtype, pd.CategoricalDtype):
            obs[col] = obs[col].astype(str)

    n = 0
    for sid, (clean, cls) in STAGE_BY_SAMPLE.items():
        m = (obs["sample_id"] == sid).to_numpy()
        if not m.any():
            continue
        obs.loc[m, "stage_clean"] = clean
        if "stage_class" in obs:
            obs.loc[m, "stage_class"] = cls
        n += int(m.sum())
    if verbose:
        print(f"stage propagated within patient_key: {n:,} cells across "
              f"{len(STAGE_BY_SAMPLE)} samples")

    lane = (obs["sample_id"] == SPLIT_SAMPLE["sample_id"]).to_numpy()
    for half in ("skin", "blood"):
        spec = SPLIT_SAMPLE[half]
        m = lane.copy()
        for k, v in spec["when"].items():
            m &= (obs[k] == v).to_numpy()
        for k, v in spec["set"].items():
            if k in obs:
                obs.loc[m, k] = v
        if verbose:
            print(f"split {SPLIT_SAMPLE['sample_id']} / {half}: {int(m.sum()):,} cells -> "
                  + ", ".join(f"{k}={v}" for k, v in spec["set"].items()))
    return obs


def patch_sample_sheet(path, verbose: bool = True) -> pd.DataFrame:
    """Apply the same corrections to data/sample_metadata_final.csv and return the frame.

    Writes nothing -- the caller decides, because this file is the input to every future atlas
    build and should not be rewritten as a side effect of importing a module.
    """
    s = pd.read_csv(path)
    for sid, (clean, cls) in STAGE_BY_SAMPLE.items():
        m = s.sample_id == sid
        if not m.any():
            continue
        if verbose:
            print(f"  {sid:<18} stage_clean {s.loc[m, 'stage_clean'].iat[0]!r} -> {clean!r}, "
                  f"stage_class {s.loc[m, 'stage_class'].iat[0]!r} -> {cls!r}")
        s.loc[m, "stage_clean"] = clean
        s.loc[m, "stage_class"] = cls

    lane = s.sample_id == SPLIT_SAMPLE["sample_id"]
    for k, v in SPLIT_SAMPLE["blood"]["set"].items():          # the lane row keeps the blood
        if k in s.columns:
            if verbose and lane.any():
                print(f"  {SPLIT_SAMPLE['sample_id']:<18} {k} {s.loc[lane, k].iat[0]!r} -> {v!r}")
            s.loc[lane, k] = v

    new_id = SPLIT_SAMPLE["skin"]["set"]["sample_id"]
    if (s.sample_id == new_id).any():
        if verbose:
            print(f"  {new_id} row already present")
        return s
    if lane.any():
        row = s[lane].iloc[0].to_dict()
        row.update(SPLIT_SAMPLE["skin"]["set"])
        row.update(donor=SPLIT_SAMPLE["skin"]["when"]["donor"], compartment="Skin",
                   tissue="Skin", disease="MF")
        for c in ("n_cells", "n_tcr", "n_clones", "n_malignant"):
            row.pop(c, None)                 # recomputed at build time, not carried over
        s = pd.concat([s, pd.DataFrame([row], columns=s.columns)], ignore_index=True)
        if verbose:
            print(f"  added row {new_id} (skin half of the SZ29 lane)")
    return s


# ---------------------------------------------------------------------------------------------
# Propagation into the built objects
# ---------------------------------------------------------------------------------------------
# Every v2 object replicates the same obs block, so a correction has to be applied to all of them
# or they drift apart -- the failure mode that once had nb30 reporting v1 numbers on a v2 input.
# These are obs-only writes: a few MB of codes per file, no matrix is opened.

LANE_OF = {"D13__MF25_skin": "D13__SZ29"}     # split sample -> the sequencing library it came from

# Objects on the v2 cell set. Columns a file does not carry are skipped, so one manifest covers
# the 57-column joint objects, the 27-column joint_raw and the 5-column cd4 input alike.
MANIFEST = [
    "data/atlas_joint/joint_annotated.h5ad",
    "data/atlas_joint/joint_raw.h5ad",
    "data/atlas_joint/joint_mrvi_input.h5ad",
    "data/atlas_joint/joint_mrvi_input_skin.h5ad",
    "data/atlas_joint/joint_mrvi_input_blood.h5ad",
    "data/atlas_joint/skin_T_annotated.h5ad",
    "data/atlas_joint/skin_T_tcr_annotated_v4.h5ad",
    "data/atlas_joint/blood_T_annotated.h5ad",
    "data/atlas_joint/blood_T_fullgene_cnv_input.h5ad",
    "data/atlas_joint/skin_mye_fib_subset.h5ad",
    "data/atlas_joint/cd4_mrvi_input.h5ad",
    "data/share/ctcl_cd4_atlas_v1.h5ad",
]

# Pre-v2 cell sets. Left alone deliberately; listed so "not patched" is a decision, not an
# oversight. Deleted from the live tree in the 2026-09-16 reorg -- the byte-identical copies
# they were verified against are in data/_archive_v1/atlas_joint/.
LEGACY = [
    "data/atlas_joint/skin_T_annotated_v2.h5ad",
    "data/atlas_joint/skin_T_tcr_annotated.h5ad",
    "data/atlas_joint/skin_T_tcr_annotated_v3.h5ad",
    "data/atlas_joint/skin_T_tcr_malig_v2.h5ad",
    "data/atlas_joint/atlas_standardized.h5ad",
    "data/atlas_joint/atlas_descriptive_slim.h5ad",
    "data/atlas_joint/atlas_obs_full.parquet",
]


def _read_cat(g, col):
    """(values as a str array, categories list) for an obs column, categorical or plain."""
    o = g[col]
    if isinstance(o, h5py.Group):
        cats = np.asarray(o["categories"][:]).astype(str)
        return cats[o["codes"][:]], list(cats)
    return np.asarray(o[:]).astype(str), None


def _write_cat(g, col, rows, values):
    """Set obs[col] to `values` at `rows`, appending categories if new ones appear."""
    o = g[col]
    if not isinstance(o, h5py.Group):                      # plain string array
        d = np.asarray(o[:]).astype(str)
        d[rows] = values
        del g[col]
        ds = g.create_dataset(col, data=d.astype(object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
        ds.attrs["encoding-type"] = "string-array"
        ds.attrs["encoding-version"] = "0.2.0"
        return

    cats = list(np.asarray(o["categories"][:]).astype(str))
    new = [v for v in pd.unique(np.asarray(values, dtype=object)) if v not in cats]
    if new:
        cats = cats + list(new)
        attrs = dict(o["categories"].attrs)
        del o["categories"]
        ds = o.create_dataset("categories", data=np.asarray(cats, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
        for k, v in attrs.items():
            ds.attrs[k] = v
    idx = {c: i for i, c in enumerate(cats)}
    codes = o["codes"][:]
    if len(cats) > np.iinfo(codes.dtype).max:              # widen before it silently wraps
        codes = codes.astype(np.int32)
        del o["codes"]
        o.create_dataset("codes", data=codes)
    o["codes"][rows] = np.asarray([idx[v] for v in values], dtype=codes.dtype)


def _targets(sample_id, donor):
    """Row masks for the three corrections, given obs sample_id and donor as str arrays."""
    lane = SPLIT_SAMPLE["sample_id"]
    new_id = SPLIT_SAMPLE["skin"]["set"]["sample_id"]
    in_lane = np.isin(sample_id, [lane, new_id])           # already-split rows count too
    skin = in_lane & (donor == SPLIT_SAMPLE["skin"]["when"]["donor"])
    blood = in_lane & (donor == SPLIT_SAMPLE["blood"]["when"]["donor"])
    stage = np.isin(sample_id, list(STAGE_BY_SAMPLE))
    return skin, blood, stage


def patch_h5ad_obs(path, dry_run: bool = True, verbose: bool = True) -> dict:
    """Apply all three corrections to one h5ad's obs. Idempotent; returns a per-column diff."""
    path = Path(path)
    diff = {}
    with h5py.File(path, "r" if dry_run else "a") as f:
        g = f["obs"]
        if "sample_id" not in g or "donor" not in g:
            if verbose:
                print(f"  {path.name}: no sample_id/donor — skipped")
            return diff
        sid, _ = _read_cat(g, "sample_id")
        don, _ = _read_cat(g, "donor")
        skin, blood, stage = _targets(sid, don)

        plan = []
        if skin.any():
            plan.append(("sample_id", skin, SPLIT_SAMPLE["skin"]["set"]["sample_id"]))
            plan.append(("organ", skin, SPLIT_SAMPLE["skin"]["set"]["organ"]))
            plan.append(("entity", skin, SPLIT_SAMPLE["skin"]["set"]["entity"]))
        if blood.any():
            plan.append(("entity", blood, SPLIT_SAMPLE["blood"]["set"]["entity"]))
        for s, (clean, cls) in STAGE_BY_SAMPLE.items():
            m = stage & (sid == s)
            if m.any():
                plan.append(("stage_clean", m, clean))
                plan.append(("stage_class", m, cls))

        for col, mask, val in plan:
            if col not in g:
                continue
            cur, _ = _read_cat(g, col)
            rows = np.flatnonzero(mask & (cur != val))
            if not len(rows):
                continue
            diff[col] = diff.get(col, 0) + len(rows)
            if not dry_run:
                _write_cat(g, col, rows, [val] * len(rows))

    if verbose:
        tag = "would change" if dry_run else "changed"
        print(f"  {path.name}: {tag} " +
              (", ".join(f"{c} x{n:,}" for c, n in sorted(diff.items())) if diff else "nothing"))
    return diff


# --- derived columns, mirrored from 32_atlas_descriptive.ipynb cell 3 -------------------------
# Kept in sync by hand; nb45 re-derives them and asserts they match, so a drift is caught.
EARLY = {"IA", "IB", "IIA"}
ADVANCED = {"IIB", "IIIA", "IIIB", "IV", "IVA", "IVA1", "IVA2", "IVB"}
ENTITY_H = {"MF": "MF_classic", "SS": "Sezary", "healthy": "healthy_control"}
LEUKEMIC_ENT = {"Sezary", "SS", "MF/SS_leukemic", "erythrodermic_CTCL(eMF/SS)"}


def rederive(A: pd.DataFrame) -> pd.DataFrame:
    """Recompute the four derived columns that depend on stage_clean / entity / organ."""
    A = A.copy()
    stg = A["stage_clean"].astype(str)
    A["entity_h"] = A["entity"].astype(str).replace(ENTITY_H)
    A["stage_group"] = np.select(
        [A["disease"].astype(str).eq("HC"), stg.isin(EARLY), stg.isin(ADVANCED)],
        ["HC", "early", "advanced"], default="unknown")
    org, tis = A["organ"].astype(str), A["tissue"].astype(str).str.lower()
    A["skin_layer"] = np.where(
        org.ne("skin"), "not_skin",
        np.select([tis.str.contains("epiderm"), tis.str.contains("derm")],
                  ["epidermis", "dermis"], default="whole"))
    cur = A["blood_involvement"].astype(str)
    A["blood_involvement_eff"] = np.select(
        [cur.isin(["yes", "B2_yes_by_def"]), cur.eq("no"),
         A["entity_h"].isin(LEUKEMIC_ENT), stg.isin(["IV", "IVA", "IVA1"]),
         stg.eq("IIIB"), stg.isin(["IVA2", "IVB"])],
        ["yes(curated)", "no(HC)", "yes(leukemic entity)", "yes(stage IV/B2)",
         "likely(stage IIIB/B1)", "stage IV non-blood (IVA2/IVB)"], default="unknown")
    return A


def patch_parquet(path, dry_run: bool = True, verbose: bool = True) -> dict:
    """Apply the corrections to the obs cache, recompute what depends on them, add lane_id."""
    path = Path(path)
    A = pd.read_parquet(path)
    before = A.copy()
    A = apply_fixups(A, verbose=False)
    A = rederive(A)
    # the sequencing library each cell came off, which is what the cell_id prefix has always been
    A["lane_id"] = A["sample_id"].astype(str).replace(LANE_OF)

    diff = {}
    for c in ["sample_id", "organ", "entity", "entity_h", "stage_clean", "stage_class",
              "stage_group", "skin_layer", "blood_involvement_eff"]:
        if c in before:
            n = int((before[c].astype(str).to_numpy() != A[c].astype(str).to_numpy()).sum())
            if n:
                diff[c] = n
    if "lane_id" not in before.columns:
        diff["lane_id"] = int(len(A))
    if verbose:
        tag = "would change" if dry_run else "changed"
        print(f"  {path.name}: {tag} " +
              (", ".join(f"{c} x{n:,}" for c, n in sorted(diff.items())) if diff else "nothing"))
    if not dry_run and diff:
        tmp = path.with_suffix(".tmp.parquet")
        A.to_parquet(tmp)
        tmp.replace(path)
    return diff


def rollback_table(paths, out_csv, verbose: bool = True) -> pd.DataFrame:
    """Record the pre-fix values of every cell the patch will touch, before anything is written."""
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists() or p.suffix != ".h5ad":
            continue
        with h5py.File(p, "r") as f:
            g = f["obs"]
            if "sample_id" not in g or "donor" not in g:
                continue
            sid, _ = _read_cat(g, "sample_id")
            don, _ = _read_cat(g, "donor")
            skin, blood, stage = _targets(sid, don)
            m = np.flatnonzero(skin | blood | stage)
            if not len(m):
                continue
            rec = {"file": p.name, "row": m,
                   "cell_id": np.asarray(g[g.attrs["_index"]][:]).astype(str)[m]}
            for c in ("sample_id", "organ", "entity", "stage_clean", "stage_class"):
                if c in g:
                    rec[c] = _read_cat(g, c)[0][m]
            rows.append(pd.DataFrame(rec))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out.to_csv(out_csv, index=False)
    if verbose:
        print(f"rollback snapshot: {len(out):,} rows -> {out_csv}")
    return out
