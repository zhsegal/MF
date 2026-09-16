#!/usr/bin/env python
"""Promote meta/samples_draft.tsv -> meta/samples.tsv for the v2 cohorts.

Three things the draft cannot do on its own:

1. **sample_id must be the RAW-FILE token, not the series-matrix title.** The dedup ledger
   (tables/atlas_dedup_v2_decisions.csv) is keyed on `<CODE>__<raw token>`, and concat_joint
   drops on that key. If samples.tsv used the title (`MF17_GEX`) instead of the file token
   (`MF17`), every duplicate drop would silently miss. Derived here with the exact same
   regex check_overlap.py used.
2. **Merge V(D)J-only GSMs into their GEX row.** GSE207679/GSE290557/GSE182861 deposit the
   TCR library as a separate GSM; load_persample_dataset iterates rows and needs one row
   per library carrying both files.
3. **Apply the scope decisions** (2026-09-05): healthy controls IN; atopic dermatitis and
   the PCFCL lesion OUT; ex-vivo culture arms OUT (B5 precedent); per-row `loader` for the
   cohorts that mix plain and hashtag-multiplexed lanes.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("co", DATA.parent / "check_overlap.py")
co = importlib.util.module_from_spec(spec)
spec.loader.exec_module(co)

CODE_OF = {lab: (lab.split("_")[0]) for lab in co.COHORTS}
LABEL_OF = {v: k for k, v in CODE_OF.items()}

DROP_DISEASE = {"AD", "Pso"}                       # out of CTCL scope
DROP_TITLE_RE = re.compile(r"culture", re.I)       # ex-vivo arms (B5 precedent)
DROP_COHORT = {"D12_pacritinib_skin"}              # culture-only cohort
# cohorts that mix loaders; value = (default_loader, {sample_token: loader})
# Per-hashtag facts a lane's single metadata row cannot express. Only needed where the
# hashtags differ in something other than donor identity.
#   GSE182861 GSM5687770: "SZ29 and MF25 are hashtagged together. SZ29 uses HTO1 and MF25
#   uses HTO2. Both samples are from the same patient" -- one blood sample and one skin
#   tumour in ONE lane, so the lane-level Blood/MF would mislabel MF25's skin cells.
HTO_OVERRIDE = {
    ("D13_gaydosik2022_skin_blood", "SZ29", "HTO1"):
        dict(donor="SZ29", compartment="Blood", tissue="PBMC", disease="SS"),
    ("D13_gaydosik2022_skin_blood", "SZ29", "HTO2"):
        dict(donor="MF25", compartment="Skin", tissue="Skin tumor", disease="MF"),
}
# hashtag labels that mark an ex-vivo treatment arm rather than a donor. The vehicle
# control is kept as the baseline; the drug arm is excluded, consistent with B5's and
# D12's cultured arms (decision 2026-09-05).
ARM_KEEP = {"ctr", "control", "vehicle", "dmso", "untreated"}
ARM_DROP = {"regn", "treated", "drug"}

LOADERS = {
    "D13_gaydosik2022_skin_blood": ("cellranger_h5_raw", {
        "SC327": "cellranger_h5_hto_raw", "SC374": "cellranger_h5_hto_raw",
        "SZ29": "cellranger_h5_hto_raw", "HB": "cellranger_h5_hto_raw"}),
    "D14_gaydosik2023_skin": ("cellranger_h5_raw", {}),
    "D15_il4ra_blockade_skin": ("cellranger_h5", {"PF2_3": "cellranger_h5_hto"}),
}
VDJ_SUFFIX = re.compile(r"[-_](GEX|VDJ|TCR|RNA)\d*$", re.I)


# The sample token MUST equal the one check_overlap.py derived from the *barcodes* file,
# because the dedup ledger is keyed on it. These patterns are the GEX-file equivalents of
# check_overlap.COHORTS[...]["re"], and _assert_tokens_match() below proves they agree.
GEX_RE = [
    r"^GSM\d+_(.+?)_(?:matrix|counts)\.mtx\.gz$",          # 10x mtx triplet
    r"^GSM\d+_filtered_feature_bc_matrix(.+?)\.h5$",        # GSE293752: token AFTER
    r"^GSM\d+_(.+?)_?raw_feature_bc_matrix\.h5$",           # GSE182861/206123: token BEFORE
    r"^GSM\d+_(.+?)_scRNA\.zip$",                           # GSE272005
]


def token_of(label: str, fname: str) -> str:
    for pat in GEX_RE:
        m = re.match(pat, fname or "")
        if m:
            return m.group(1)
    # a deposit can ship a library with no sample token at all (GSE264636's GSM9038965
    # is `GSM9038965_matrix.mtx.gz`). Fall back to the GSM accession, which is unique,
    # rather than to a meaningless suffix like "matrix".
    m = re.match(r"^(GSM\d+)_((?:matrix|counts)\.mtx\.gz)$", fname or "")
    if m:
        return m.group(1)
    t = re.sub(r"^GSM\d+_", "", fname or "")
    return VDJ_SUFFIX.sub("", re.sub(r"\.(mtx|csv|tsv|h5|zip|tar)(\.gz)?$", "", t))


def _assert_tokens_match(label: str, tokens) -> None:
    """The ledger is keyed on check_overlap's barcode-file tokens. If these drift, every
    duplicate drop silently misses -- so fail loudly instead."""
    code = CODE_OF[label]
    known = set(co.units_for(label)) if label in co.COHORTS else set()
    if not known:
        return
    unseen = set(tokens) - known
    if unseen:
        print(f"    ! [{label}] {len(unseen)} sample tokens not seen by check_overlap "
              f"(below its 100-cell floor, or a regex drift): {sorted(unseen)[:6]}")


def pair_key(title: str) -> str:
    """One key per library, so a GEX title and its VDJ title collapse together.

    Covers `MF1-GEX`/`MF1-VDJ`, `WU1084_RNA2`/`WU1084_TCR2`,
    `PT56_Tumor_scRNA`/`PT56_Tumor_scVDJ`, and `SZ29_MF25_GEX_HTO`/`SZ29_MF25_TCR`
    (the trailing `_HTO` marks the chemistry, not the library).
    """
    t = re.sub(r"[-_]HTO$", "", title.strip(), flags=re.I)
    m = re.match(r"^(.*?)[-_](?:sc)?(?:GEX|VDJ|TCR|RNA)(\d*)$", t, re.I)
    return f"{m.group(1)}#{m.group(2)}" if m else t


def finalize(label: str) -> None:
    d = DATA / label / "meta" / "samples_draft.tsv"
    if not d.exists():
        print(f"[{label}] no draft -> skip"); return
    try:
        df = pd.read_csv(d, sep="\t", dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        print(f"[{label}] empty draft -> skip (expected for the CosMx spatial track)")
        return

    # --- 2. merge VDJ-only rows into their GEX row -----------------------------------
    df["_pk"] = df["title"].map(pair_key)
    gex = df[df["gex_file"] != ""].copy()
    vdj = df[(df["gex_file"] == "") & (df["contig_file"] != "")]
    vmap = dict(zip(vdj["_pk"], vdj["contig_file"]))
    merged = 0
    for i, r in gex.iterrows():
        if not r["contig_file"] and r["_pk"] in vmap:
            gex.at[i, "contig_file"] = vmap[r["_pk"]]
            merged += 1

    # --- 1. sample_id / donor from the RAW FILE token --------------------------------
    gex["sample_id"] = [token_of(label, f) for f in gex["gex_file"]]
    _assert_tokens_match(label, gex["sample_id"])
    code = CODE_OF[label]
    gex["donor"] = [co.donor_of(code, s) for s in gex["sample_id"]]
    # a library with no sample token keeps the GSM as sample_id (stable, and what the
    # dedup ledger is keyed on) but takes its DONOR from the series-matrix title --
    # GSE264636's GSM9038965 is patient P220.
    bare = gex["sample_id"].str.fullmatch(r"GSM\d+")
    if bare.any():
        gex.loc[bare, "donor"] = (gex.loc[bare, "title"].str.strip()
                                  .str.split(r"[ _/]").str[0])
    # explicit, so standardize_obs does not fall through to the namespaced `donor`
    # (atlas_join_helpers.py:269). Bare, matching li24/D1/D3/B4. The reliable
    # cross-deposit identity is `patient_key`, attached in concat_joint.
    gex["real_donor"] = gex["donor"]

    # --- 3. scope decisions ------------------------------------------------------------
    n0 = len(gex)
    reasons = []
    keep = pd.Series(True, index=gex.index)
    if label in DROP_COHORT:
        keep[:] = False
        reasons.append(f"whole cohort excluded (culture-only, B5 precedent)")
    m = gex["disease"].isin(DROP_DISEASE)
    if m.any():
        keep &= ~m
        reasons.append(f"{int(m.sum())} rows out of CTCL scope ({sorted(set(gex.loc[m,'disease']))})")
    m = gex["title"].str.contains(DROP_TITLE_RE) | gex["sample_id"].str.contains(DROP_TITLE_RE)
    if m.any():
        keep &= ~m
        reasons.append(f"{int(m.sum())} ex-vivo culture arms")
    # upstream deposit defects: GSE290557's GSM8816535 barcodes.tsv.gz is 0 bytes in GEO
    # itself (its filelist.txt records size 0), so that triplet cannot be read at all.
    def _bad(f):
        if not f:
            return True
        q = DATA / label / "raw" / f
        return (not q.exists()) or q.stat().st_size == 0
    def _triplet_bad(f):
        if _bad(f):
            return True
        for part in ("barcodes.tsv.gz",):
            sib = re.sub(r"_(matrix|counts)\.mtx\.gz$", f"_{part}", f)
            if sib != f and _bad(sib):
                return True
        return False
    m = gex["gex_file"].map(_triplet_bad)
    if m.any():
        keep &= ~m
        reasons.append(f"{int(m.sum())} rows with a missing/0-byte file in the deposit "
                       f"({sorted(gex.loc[m, 'sample_id'])})")
    m = gex["disease"].isin(["unknown"])
    if m.any():
        keep &= ~m
        reasons.append(f"{int(m.sum())} unmapped disease ({sorted(set(gex.loc[m,'title']))[:3]})")
    excluded = gex[~keep].copy()
    gex = gex[keep].copy()

    # per-row loader
    if label in LOADERS:
        default, over = LOADERS[label]
        gex["loader"] = [over.get(s, default) for s in gex["sample_id"]]

    # --- per-hashtag map ------------------------------------------------------------
    draft_map = cdir_map = DATA / label / "meta" / "hto_map_draft.tsv"
    if draft_map.exists():
        hm = pd.read_csv(draft_map, sep="\t", dtype=str).fillna("")
        # draft keys lanes by series-matrix TITLE; samples.tsv keys by raw-file token
        t2s = dict(zip(gex["title"], gex["sample_id"]))
        rows_hm = []
        for _, h in hm.iterrows():
            sid = t2s.get(h["sample_id"])
            if sid is None:
                continue
            base = gex[gex["sample_id"] == sid]
            if not len(base):
                continue
            b = base.iloc[0]
            lab = str(h["label"]).strip()
            is_arm = lab.lower() in ARM_KEEP | ARM_DROP
            rec = dict(sample_id=sid, hto=h["hto"].upper(),
                       donor=(b["donor"] if is_arm else lab),
                       compartment=b["compartment"], tissue=b["tissue"],
                       disease=b["disease"],
                       arm=(lab.lower() if is_arm else ""),
                       keep=("no" if lab.lower() in ARM_DROP else "yes"))
            rec.update(HTO_OVERRIDE.get((label, sid, h["hto"].upper()), {}))
            rows_hm.append(rec)
        if rows_hm:
            hmo = pd.DataFrame(rows_hm)
            hmo.to_csv(DATA / label / "meta" / "hto_map.tsv", sep="\t", index=False)
            hashed = sorted(set(hmo["sample_id"]))
            gex.loc[gex["sample_id"].isin(hashed), "loader"] = (
                "cellranger_h5_hto_raw" if "raw" in LOADERS.get(label, ("", {}))[0]
                else "cellranger_h5_hto")
            ndrop = int((hmo["keep"] == "no").sum())
            print(f"    [{label}] hto_map: {len(hmo)} hashtags over {len(hashed)} lanes"
                  f"{f', {ndrop} treatment arms excluded' if ndrop else ''}")

    cols = ["sample_id", "donor", "real_donor", "disease", "disease_stage", "sex", "compartment",
            "tissue", "tech", "lesion_status", "age", "treatment", "timepoint",
            "gex_file", "contig_file", "gsm", "title", "notes"]
    if "loader" in gex:
        cols.insert(8, "loader")
    gex = gex[[c for c in cols if c in gex.columns]]
    gex.to_csv(DATA / label / "meta" / "samples.tsv", sep="\t", index=False)
    if len(excluded):
        excluded.to_csv(DATA / label / "meta" / "samples_excluded.tsv", sep="\t", index=False)
    print(f"[{label}] {n0} GEX rows -> {len(gex)} kept, {len(excluded)} excluded"
          f"{'; ' + '; '.join(reasons) if reasons else ''}"
          f" | VDJ merged into {merged} rows"
          f" | donors={gex['donor'].nunique() if len(gex) else 0}")


def main() -> None:
    man = pd.read_csv(DATA / "_new_cohorts.tsv", sep="\t")
    want = sys.argv[1:] or ["--all"]
    for lab in man["label"]:
        if "--all" in want or lab in want:
            finalize(lab)


if __name__ == "__main__":
    main()
