#!/usr/bin/env python
"""Append the v2 cohorts' rows to data/sample_metadata_final.csv.

`run_build_joint.py` asserts every obs `sample_id` is present in that CSV, and it does so
*after* the concat + annotate + resolve_lineage -- so a missing row aborts a multi-hour,
380 GB job at the very end and never writes joint_annotated.h5ad. The CSV holds 149 v1 rows
and v1 only ever appended B5 (`_append_b5_metadata`), one hardcoded block per cohort.

This is the generic version: it derives each row from the cohort's own meta/samples.tsv plus
data/_new_cohorts.tsv, so onboarding a cohort needs no new code. Idempotent -- rows for a
sample_id already present are replaced, not duplicated.

  python append_v2_metadata.py --dry-run
  python append_v2_metadata.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
FINAL = DATA / "sample_metadata_final.csv"
MANIFEST = DATA / "_new_cohorts.tsv"
sys.path.insert(0, str(DATA.parent))
import atlas_join_helpers as H  # noqa: E402

# per-cohort constants that samples.tsv cannot carry
COHORT_META = {
    "B6": dict(repo="GEO", accession="GSE197619", cohort_country="USA", cohort_region="North American"),
    "B7": dict(repo="GEO", accession="GSE207679", cohort_country="USA", cohort_region="North American"),
    "B8": dict(repo="GEO", accession="GSE290557", cohort_country="USA", cohort_region="North American"),
    "D7": dict(repo="GEO", accession="GSE173205", cohort_country="Austria", cohort_region="European"),
    "D8": dict(repo="GEO", accession="GSE247047", cohort_country="Austria", cohort_region="European"),
    "D9": dict(repo="GEO", accession="GSE264636", cohort_country="Austria", cohort_region="European"),
    "D10": dict(repo="GEO", accession="GSE173820", cohort_country="Austria", cohort_region="European"),
    "D11": dict(repo="GEO", accession="GSE303446", cohort_country="unknown", cohort_region="unknown"),
    "D13": dict(repo="GEO", accession="GSE182861", cohort_country="USA", cohort_region="North American"),
    "D14": dict(repo="GEO", accession="GSE206123", cohort_country="USA", cohort_region="North American"),
    "D15": dict(repo="GEO", accession="GSE293752", cohort_country="USA", cohort_region="North American"),
}
STAGE_ADV = {"IIB", "III", "IIIA", "IIIB", "IVA", "IVA1", "IVA2", "IVB"}


def stage_class(stage: str, disease: str) -> str:
    s = (stage or "").upper().replace(" ", "")
    if disease == "HC":
        return "HC"
    if disease == "SS":
        return "advanced"
    for k in sorted(STAGE_ADV, key=len, reverse=True):
        if s.startswith(k):
            return "advanced"
    if s.startswith(("IA", "IB", "IIA")):
        return "early"
    return "other" if disease == "CTCL_other" else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cur = pd.read_csv(FINAL, dtype=str).fillna("")
    cols = list(cur.columns)
    man = pd.read_csv(MANIFEST, sep="\t").fillna("")
    REG = H.dataset_registry(DATA.parent)
    drop_ids, pkey = H.load_dedup_ledger(DATA.parent)

    rows = []
    for _, mrow in man.iterrows():
        label = mrow["label"]
        code = label.split("_")[0]
        if code not in COHORT_META or not REG.get(code, {}).get("in_expression"):
            continue
        f = DATA / label / "meta" / "samples.tsv"
        if not f.exists():
            continue
        t = pd.read_csv(f, sep="\t", dtype=str).fillna("")
        for _, r in t.iterrows():
            sid = H.namespace(code, r["sample_id"])
            if sid in drop_ids:
                continue                       # duplicate; concat_joint drops these cells
            disease = r.get("disease", "") or "unknown"
            stage = r.get("disease_stage", "") or ""
            row = dict.fromkeys(cols, "")
            row.update(COHORT_META[code])
            comp = r.get("compartment", "") or "Skin"
            row.update(
                sample_id=sid,
                dataset=H.DATASET_NAME[code],
                study=REG[code]["study"],
                donor=H.namespace(code, r.get("donor", r["sample_id"])),
                n_donors="1",
                disease=disease,
                disease_stage=stage if stage != "NA" else "",
                compartment=comp,
                tissue=H._norm_tissue(r.get("tissue", "") or comp),
                tech=r.get("tech", "10x_5p") or "10x_5p",
                sex=r.get("sex", "unknown") or "unknown",
                organ="blood" if comp == "Blood" else ("lymph_node" if comp == "LN" else "skin"),
                tissue_detail=r.get("tissue", "") or comp,
                malignant_call_method=("TCR+CNV" if r.get("contig_file") else "CNV_only"),
                tcr_available="yes" if r.get("contig_file") else "no",
                malignant_labeled="no",
                stage_class=stage_class(stage, disease),
                stage_clean=stage if stage != "NA" else "",
                lineage="unresolved_provisional",
                entity=("healthy" if disease == "HC" else disease),
                fmf_status="not_applicable",
                lineage_resolve="no",
                treatment_context=(r.get("timepoint", "") or r.get("treatment", "")
                                   or "unknown"),
                blood_involvement=("yes" if comp == "Blood" else "unknown"),
                lesion_type=r.get("lesion_status", "") or "unknown",
                origin_resolve="no",
                east_asian="False",
            )
            rows.append(row)

    new = pd.DataFrame(rows, columns=cols)
    dupes = set(new["sample_id"]) & set(cur["sample_id"])
    kept = cur[~cur["sample_id"].isin(new["sample_id"])]
    out = pd.concat([kept, new], ignore_index=True)

    print(f"existing rows : {len(cur)}")
    print(f"v2 rows built : {len(new)}  ({len(dupes)} replaced existing)")
    print(f"total after   : {len(out)}")
    print(f"\nper cohort: {dict(new['dataset'].value_counts())}")
    print(f"disease    : {dict(new['disease'].value_counts())}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        print(new.head(3).to_string())
        return 0
    out.to_csv(FINAL, index=False)
    print(f"\nwrote {FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
