#!/usr/bin/env python
"""Draft meta/samples.tsv for a v2 cohort from its GEO series matrix + raw files.

Built from the series matrix, never from filename guessing: GSE182861 hides two patients
in one lane (`SZ29_MF25_GEX_HTO`, `HC1and2_GEX`) and GSE293752 pools `PF2/PF3`, neither of
which is visible in the supplementary file names.

Emits a DRAFT plus meta/samples_review.txt listing everything a human has to resolve
(pooled/hashed lanes, culture arms, unmapped disease strings). Nothing here is
authoritative until that review file is empty or explicitly signed off.

  python build_samples_tsv_generic.py <LABEL>          # one cohort
  python build_samples_tsv_generic.py --all
"""
from __future__ import annotations

import csv
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
MANIFEST = DATA / "_new_cohorts.tsv"

# GEO characteristic value -> atlas DISEASE_VOCAB. Anything unmapped lands in the review file.
DISEASE_MAP = {
    "cutaneous t cell lymphoma": "CTCL_other", "ctcl": "CTCL_other",
    "mycosis fungoides": "MF", "mf": "MF", "large plaque parapsoriasis": "MF",
    "sezary syndrome": "SS", "sézary syndrome": "SS", "sezary": "SS", "ss": "SS",
    "lymphomatoid papulosis": "CTCL_other", "lyp": "CTCL_other",
    "small plaque parapsoriasis": "CTCL_other", "parapsoriasis": "CTCL_other",
    "healthy": "HC", "healthy control": "HC", "normal": "HC", "normal skin": "HC",
    "healthy donor": "HC", "none": "HC", "control": "HC", "non-lesional": "HC",
    "atopic dermatitis": "AD", "psoriasis": "Pso",
}
# entities the atlas deliberately does not carry (decision 2026-09-05):
# healthy controls are IN; atopic dermatitis and the B-cell lymphoma lesion are OUT.
OUT_OF_SCOPE = {"AD", "Pso"}
CULTURE_RE = re.compile(r"culture", re.I)
POOLED_RE = re.compile(r"(\d\s*and\s*\d)|(/)|(_HTO$)|(and\d)", re.I)


def parse_series_matrix(path: Path) -> pd.DataFrame:
    rows: dict[str, list[str]] = {}
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, *vals = line.rstrip("\n").split("\t")
            key = key[1:]
            vals = [v.strip().strip('"') for v in vals]
            rows.setdefault(key, []).append(vals)
    if "Sample_geo_accession" not in rows:
        raise ValueError(f"no !Sample_geo_accession in {path}")
    gsms = rows["Sample_geo_accession"][0]
    out = pd.DataFrame({"gsm": gsms})
    for key, blocks in rows.items():
        for i, vals in enumerate(blocks):
            if len(vals) != len(gsms):
                continue
            col = key.replace("Sample_", "").lower()
            out[col if len(blocks) == 1 else f"{col}_{i}"] = vals
    return out


def _norm_sex(v) -> str:
    """CANON_OBS documents M | F | unknown. The old `.lower()[:1]` turned the "unknown"
    default into "u", splitting the same category two ways."""
    v = str(v or "").strip().lower()
    if v.startswith("m") and not v.startswith("mixed"):
        return "M"
    if v.startswith("f"):
        return "F"
    return "unknown"


def _norm_tech(chem, title) -> str:
    """3' vs 5' matters (only 5' libraries can carry V(D)J). GSE206123 states `3PV2`;
    the previous expression had identical branches and asserted 5' for every cohort."""
    t = f"{chem} {title}"
    if re.search(r"\b3'|3p|3PV", t, re.I):
        return "10x_3p"
    if re.search(r"\b5'|5p|5PV", t, re.I):
        return "10x_5p"
    return "10x_5p"


_STAGE_RE = re.compile(r"\b(IV\s?A1|IV\s?A2|IV\s?A|IV\s?B|III\s?A|III\s?B|III|"
                       r"II\s?A|II\s?B|I\s?A|I\s?B)", re.I)  # no trailing \b: GSE206123
# writes the stage fused to the TNMB code (`IVAT4NxM0B0`). Alternation is longest-first, so
# III/II are matched before I.


def _norm_stage(v) -> str:
    """Extract the ISCL/EORTC stage from free text.

    GSE206123 writes `F IIB T3NxB0M0` and `IVAT4NxM0B0`; leaving those raw makes every
    stage-stratified group a singleton and defeats stage_class in append_v2_metadata.py.
    """
    t = str(v or "").strip()
    if not t or t.upper() == "NA":
        return "NA"
    m = _STAGE_RE.search(t.replace("IVA", "IV A").replace("IVB", "IV B")
                          .replace("IIIA", "III A").replace("IIIB", "III B"))
    return m.group(1).replace(" ", "").upper() if m else t


def _char_cols(df):
    return [c for c in df.columns if c.startswith("characteristics_ch1")]


def kv(df, gsm_row) -> dict[str, str]:
    """Flatten a GSM's 'key: value' characteristics into a dict."""
    d = {}
    for c in _char_cols(df):
        v = str(gsm_row.get(c, ""))
        if ":" in v:
            k, _, val = v.partition(":")
            d[k.strip().lower()] = val.strip()
    return d


def map_disease(*candidates) -> tuple[str, str]:
    for c in candidates:
        c = (c or "").strip().lower()
        if not c:
            continue
        if c in DISEASE_MAP:
            return DISEASE_MAP[c], ""
        for k, v in DISEASE_MAP.items():
            if k in c:
                return v, ""
    return "unknown", f"unmapped disease string: {candidates!r}"


_GAP = None


def gap_meta(gse: str) -> dict[str, dict[str, str]]:
    """Per-donor sex/age/stage/LCT/treatment from docs/wilcox_gap_donors.csv.

    Wilcox Table S1 carries clinical fields several deposits omit entirely (GSE197619 states
    no stage at all). Keyed on (accession, donor) -- never on donor alone, since the same
    bare id means different patients across deposits.
    """
    global _GAP
    if _GAP is None:
        _GAP = {}
        f = DATA.parent / "docs" / "wilcox_gap_donors.csv"
        if f.exists():
            for r in csv.DictReader(f.open()):
                _GAP.setdefault(r["accession"], {})[r["donor"].strip()] = r
    return _GAP.get(gse, {})


# Deposits state their hashtag assignment only in free-text !Sample_description:
#   GSE182861 "SZ29 and MF25 are hashtagged together. SZ29 uses HTO1 and MF25 uses HTO2."
#   GSE182861 "HB1, HB2, and HB3 are hashtagged together using HTO1, HTO2, and HTO3 respectively"
#   GSE293752 "Treatments hashed together, HTO1-ctr,HTO2-regn"
# Nothing in the file names carries it, so a lane loaded without this silently pools
# several patients -- or a treated and an untreated arm -- under one donor.
_HTO_USES = re.compile(r"([A-Za-z0-9_]+)\s+uses\s+(HTO\d+)", re.I)
_HTO_DASH = re.compile(r"(HTO\d+)\s*-\s*([A-Za-z0-9_]+)", re.I)
_HTO_RESP = re.compile(r"([A-Za-z0-9_,\s]+?)\s+are hashtagged together using\s+"
                       r"([HTO0-9,\s and]+?)\s+respectively", re.I)


def parse_hto(descs) -> list[tuple[str, str]]:
    """-> [(hashtag, label)] for one lane, from its free-text descriptions."""
    # drop the boilerplate filename description, or it is swallowed into the label list
    # ("Hashtags_feature_reference.csv HB1, HB2, and HB3 ...")
    txt = " ".join(d for d in descs if not d.strip().lower().endswith(".csv"))
    out = [(h.upper(), lab) for lab, h in _HTO_USES.findall(txt)]
    if out:
        return out
    out = [(h.upper(), lab) for h, lab in _HTO_DASH.findall(txt)]
    if out:
        return out
    m = _HTO_RESP.search(txt)
    if m:
        labs = [x.strip() for x in re.split(r",|\band\b", m.group(1)) if x.strip()]
        htos = [x.strip().upper() for x in re.split(r",|\band\b", m.group(2)) if x.strip()]
        if len(labs) == len(htos):
            return list(zip(htos, labs))
    return []


def build(label: str, gse: str, compartment_default: str,
          disease_default: str = "") -> None:
    cdir = DATA / label
    sm = sorted((cdir / "meta").glob("*series_matrix.txt.gz"))
    if not sm:
        print(f"[{label}] NO series matrix -> skip"); return
    df = parse_series_matrix(sm[0])

    raw = cdir / "raw"
    files = sorted(p.name for p in raw.iterdir() if p.is_file())
    by_gsm: dict[str, list[str]] = {}
    for f in files:
        m = re.match(r"(GSM\d+)_", f)
        if m:
            by_gsm.setdefault(m.group(1), []).append(f)

    gap = gap_meta(gse)
    rows, review, hto_rows = [], [], []
    for _, r in df.iterrows():
        gsm = r["gsm"]
        fs = by_gsm.get(gsm, [])
        # note the .h5 test is a substring, not endswith: GSE293752 appends the sample id
        # AFTER the suffix (`..._filtered_feature_bc_matrixMF21.h5`).
        gex = next((f for f in fs if f.endswith(("matrix.mtx.gz", "counts.mtx.gz", "scRNA.zip"))
                    or ("feature_bc_matrix" in f and f.endswith(".h5"))), "")
        contig = next((f for f in fs if "contig_annotation" in f
                       or f.endswith(("_TCR.tar.gz", "scVDJ.zip"))), "")
        if not gex and not contig:
            continue
        ch = kv(df, r)
        descs = [str(r.get(c, "")) for c in df.columns if c.startswith("description")]
        title = str(r.get("title", gsm))
        src = str(r.get("source_name_ch1", ""))
        disease, warn = map_disease(ch.get("disease"), ch.get("disease state"),
                                    ch.get("diagnosis"), ch.get("clinical diagnosis"),
                                    ch.get("molecular condition"), ch.get("health status"),
                                    ch.get("group"), title, src)
        if disease == "unknown" and disease_default:
            # deposits that state no diagnosis field at all (GSE290557, GSE303446,
            # GSE309807). The default comes from the paper, recorded in _new_cohorts.tsv.
            disease, warn = disease_default, ""
        donor = re.split(r"[ _/]", title.strip())[0] or gsm
        tissue = ch.get("tissue") or src or compartment_default
        comp = ("Blood" if re.search(r"blood|pbmc|pb\b", f"{tissue} {src}", re.I)
                else "LN" if re.search(r"lymph", f"{tissue} {src}", re.I) else "Skin")
        note = []
        if warn:
            note.append(warn); review.append(f"{gsm} {title!r}: {warn}")
        if CULTURE_RE.search(title):
            note.append("EX-VIVO CULTURE ARM -> exclude (B5 precedent)")
            review.append(f"{gsm} {title!r}: culture arm -- confirm exclusion")
        hto = parse_hto(descs)
        if hto:
            note.append(f"hashed lane: {', '.join(f'{h}={l}' for h, l in hto)}")
        if POOLED_RE.search(title) or hto:
            if not hto:
                review.append(f"{gsm} {title!r}: looks pooled but the deposit states no "
                              "hashtag assignment -- donor is a GUESS until demuxed")
            note.append("HASHED LANE -> demux")
        if disease in OUT_OF_SCOPE:
            note.append(f"{disease} -> OUT OF SCOPE, exclude")
            review.append(f"{gsm} {title!r}: {disease} is out of CTCL scope")
        g = gap.get(donor, {})
        if g:
            note.append("clinical fields from wilcox_gap_donors.csv")
        rows.append(dict(
            sample_id=re.sub(r"[^A-Za-z0-9_.-]", "_", title.strip()) or gsm,
            donor=donor, gsm=gsm, title=title,
            disease=disease,
            disease_stage=_norm_stage(ch.get("stage") or ch.get("disease stage")
                                      or ch.get("tumor stage") or g.get("stage")),
            disease_stage_raw=(ch.get("stage") or ch.get("disease stage")
                               or ch.get("tumor stage") or g.get("stage") or ""),
            sex=_norm_sex(ch.get("Sex") or ch.get("sex") or ch.get("gender")
                          or g.get("sex")),
            compartment=comp, tissue=tissue,
            tech=_norm_tech(ch.get("chemistry", ""), title),
            age=ch.get("age", "") or g.get("age", ""),
            lct=g.get("lct", ""),
            treatment=ch.get("treatment", "") or g.get("treatment", ""),
            timepoint=ch.get("batch") or ch.get("time") or "",
            lesion_status=ch.get("lesion type") or ch.get("lesion") or "",
            gex_file=gex, contig_file=contig, notes="; ".join(note), _hto=hto))

    if not rows:
        print(f"[{label}] no GSM matched a raw file -- skip "
              f"(expected for the CosMx spatial track)")
        return
    for r0 in rows:
        for h, lab in (r0.pop("_hto", None) or []):
            hto_rows.append(dict(sample_id=r0["sample_id"], hto=h, label=lab))
    out = pd.DataFrame(rows)
    (cdir / "meta").mkdir(parents=True, exist_ok=True)
    out.to_csv(cdir / "meta" / "samples_draft.tsv", sep="\t", index=False)
    if hto_rows:
        pd.DataFrame(hto_rows).to_csv(cdir / "meta" / "hto_map_draft.tsv",
                                      sep="\t", index=False)
        print(f"    [{label}] hashtag map: {len(hto_rows)} rows over "
              f"{len({h['sample_id'] for h in hto_rows})} lanes")
    (cdir / "meta" / "samples_review.txt").write_text(
        "\n".join(review) + ("\n" if review else ""))
    n_gex = int((out["gex_file"] != "").sum())
    print(f"[{label}] {len(out)} GSM rows ({n_gex} with GEX, "
          f"{int((out['contig_file'] != '').sum())} with VDJ) "
          f"| disease={dict(out['disease'].value_counts())} "
          f"| REVIEW: {len(review)}")


def main() -> None:
    man = pd.read_csv(MANIFEST, sep="\t")
    want = sys.argv[1:] or ["--all"]
    for _, r in man.iterrows():
        if "--all" in want or r["label"] in want:
            build(r["label"], r["gse"], r["compartment"],
                  "" if pd.isna(r.get("disease_default")) else str(r["disease_default"]))


if __name__ == "__main__":
    main()
