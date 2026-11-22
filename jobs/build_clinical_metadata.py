#!/usr/bin/env python
"""Build data/clinical_metadata_v1.csv from the paper extractions, plus three report tables.

Reads only small tables (data/clinical/raw/*.tsv, sample_metadata_final.csv, and two columns of the
obs cache for patient_key), so it is safe on the login node.

    python jobs/build_clinical_metadata.py

Writes
    data/clinical_metadata_v1.csv        one row per atlas sample_id: clin_* + raw_* text
    tables/clinical_v1_coverage.csv      per study: persons with a known value, per field
    tables/clinical_v1_contrasts.csv     persons (and samples) per level of each comparison axis
    tables/clinical_v1_conflicts.csv     every disagreement with the current atlas columns
    tables/clinical_v1_tcr_identity.csv  dominant TRB clones shared across patient_keys
    tables/clinical_v1_stage_changes.csv persons whose samples carry different stages (timepoints)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

NB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB_DIR / "helpers"))
import clinical_meta as CM  # noqa: E402

SHEET = NB_DIR / "data" / "sample_metadata_final.csv"
SEX_EXPR = NB_DIR / "tables" / "sample_sex_inferred.csv"    # jobs/run_sex_infer.sh
TCR = NB_DIR / "data" / "atlas_joint" / "tcr_clones.parquet"
TAB = NB_DIR / "tables"
UNKNOWN = {"", "unknown", "nan", "NA", "none/unspecified", "not_applicable",
           "unresolved_provisional"}


def sample_sheet() -> pd.DataFrame:
    s = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
    pk = (pd.read_parquet(CM.OBS_PARQUET, columns=["sample_id", "patient_key"])
          .astype(str).drop_duplicates())
    assert not pk.sample_id.duplicated().any(), "sample_id spans >1 patient_key"
    s = s.merge(pk, on="sample_id", how="left", validate="1:1")
    # sheet rows with no cells in v2 (D5__MFIVB_skin lost the dedup to D1__P303_skin) take
    # their donor's key from that donor's other samples
    by_donor = s.dropna(subset=["patient_key"]).groupby("donor").patient_key.first()
    s["patient_key"] = s.patient_key.fillna(s.donor.map(by_donor))
    assert s.patient_key.notna().all(), s[s.patient_key.isna()].sample_id.tolist()
    return s


def add_expr_sex(clin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill clin_sex from the expression call where no paper reports it; list disagreements.

    M_lowY (no XIST, weak chrY) counts as M: chrY loss is common in male CTCL clones.
    """
    if not SEX_EXPR.exists():
        print(f"(no {SEX_EXPR.name} yet -- run jobs/run_sex_infer.sh; sex stays paper-only)")
        return clin, pd.DataFrame()
    e = pd.read_csv(SEX_EXPR).set_index("sample_id")
    call = clin.sample_id.map(e.sex_expr.replace({"M_lowY": "M"})).fillna("")
    clin = clin.assign(clin_sex_expr=clin.sample_id.map(e.sex_expr).fillna(""))
    fill = (clin.clin_sex == "") & call.isin(["M", "F"])
    clin.loc[fill, "clin_sex"] = call[fill]
    clin.loc[fill, "clin_sex_source"] = "expr"
    bad = clin[(clin.clin_sex_source == "paper") & call.isin(["M", "F"]) & (call != clin.clin_sex)]
    mm = pd.DataFrame(dict(sample_id=bad.sample_id, study=bad.study, field="sex_expr",
                           current=call[bad.index], curated=bad.clin_sex,
                           source=bad.clin_source, confidence=bad.clin_confidence))
    print(f"expression sex: filled {int(fill.sum())} samples, {len(mm)} paper/expr mismatches")
    return clin, mm


def tcr_identity(clin: pd.DataFrame, min_frac=0.05, min_cells=20) -> pd.DataFrame:
    """Dominant TRB CDR3s (>= min_frac of a sample's TRB cells) seen under >1 patient_key.

    A malignant clone shared by two patient_keys means one person. That is how SS1/SZ22/MF19 =
    Herrera SS2 and SZ32 = MF26 were found; those are in PERSON_MERGE, so `explained` is True.
    The rest are flagged in clin_qc_flag: contamination, a label swap, or an unknown duplicate.
    """
    t = pd.read_parquet(TCR, columns=["sample_id", "trb_cdr3", "has_tcr"])
    t = t[t.has_tcr & t.trb_cdr3.notna() & t.trb_cdr3.ne("")]
    n = t.groupby("sample_id").size()
    c = t.groupby(["sample_id", "trb_cdr3"]).size().rename("n_cells").reset_index()
    c["frac"] = (c.n_cells / c.sample_id.map(n)).round(4)
    top = c[(c.frac >= min_frac) & (c.n_cells >= min_cells)].copy()
    k = clin.set_index("sample_id")
    top["patient_key"] = top.sample_id.map(k.patient_key)
    top["clin_person_key"] = top.sample_id.map(k.clin_person_key)
    top = top.dropna(subset=["patient_key"])
    shared = top.groupby("trb_cdr3").filter(lambda g: g.patient_key.nunique() > 1).copy()
    shared["explained"] = shared.groupby("trb_cdr3").clin_person_key.transform("nunique").eq(1)
    return shared.sort_values(["trb_cdr3", "frac"], ascending=[True, False])


def person_level(clin: pd.DataFrame, col: str) -> pd.Series:
    """One value per person: the most frequent known value across that person's samples."""
    k = clin[~clin[col].isin(UNKNOWN)]
    return k.groupby("clin_person_key")[col].agg(lambda s: s.mode().iat[0])


def coverage(clin: pd.DataFrame) -> pd.DataFrame:
    fields = ["clin_dx", "clin_phenotype", "clin_cd30", "clin_lct", "clin_stage", "clin_B",
              "clin_lesion", "clin_sex", "clin_age", "clin_race", "clin_duration_y",
              "clin_blood_tumor_pct", "clin_prior_tx", "clin_current_tx", "clin_prior_systemic",
              "clin_current_systemic", "clin_ever_systemic", "clin_tx_naive", "clin_response",
              "clin_drug_sensitivity", "clin_course", "clin_outcome"]
    c = clin.astype({f: str for f in CM.OBS_NUM_COLS})
    rows = []
    for study, g in c.groupby("study"):
        r = {"study": study, "n_samples": len(g), "n_persons": g.clin_person_key.nunique()}
        for f in fields:
            r[f] = g.loc[~g[f].isin(UNKNOWN), "clin_person_key"].nunique()
        rows.append(r)
    out = pd.DataFrame(rows)
    tot = out.drop(columns="study").sum().to_dict()
    tot["n_persons"] = c.clin_person_key.nunique()
    for f in fields:
        tot[f] = c.loc[~c[f].isin(UNKNOWN), "clin_person_key"].nunique()
    return pd.concat([out, pd.DataFrame([{"study": "ALL", **tot}])], ignore_index=True)


def contrasts(clin: pd.DataFrame) -> pd.DataFrame:
    ctcl = clin[clin.clin_group == "CTCL"]
    axes = ["clin_dx", "clin_phenotype", "clin_cd30", "clin_lct", "clin_fmf", "clin_stage_class",
            "clin_tx_naive", "clin_prior_systemic", "clin_current_systemic",
            "clin_ever_systemic", "clin_course",
            "clin_outcome", "clin_response", "clin_overall_response", "clin_drug_sensitivity",
            "clin_timepoint", "clin_ex_vivo", "clin_input"]
    rows = []
    for comp in ["Skin", "Blood", "ALL"]:
        g = ctcl if comp == "ALL" else ctcl[ctcl.compartment == comp]
        for ax in axes:
            for lvl, h in g[~g[ax].isin(UNKNOWN)].groupby(ax):
                rows.append(dict(axis=ax, level=lvl, compartment=comp,
                                 n_persons=h.clin_person_key.nunique(), n_samples=len(h)))
        for col, axis in [("clin_prior_tx_classes", "prior_tx_class"),
                          ("clin_current_tx_classes", "current_tx_class"),
                          ("clin_prior_tx", "prior_tx_agent"),
                          ("clin_current_tx", "current_tx_agent")]:
            expo = g.assign(c=g[col].str.split(";")).explode("c")
            for lvl, h in expo[expo.c.ne("") & expo.c.notna()].groupby("c"):
                rows.append(dict(axis=axis, level=lvl, compartment=comp,
                                 n_persons=h.clin_person_key.nunique(), n_samples=len(h)))
    return pd.DataFrame(rows)


DX_COARSE = {"MF": "MF", "SS": "SS", "HC": "HC"}


def conflicts(clin: pd.DataFrame, sheet: pd.DataFrame) -> pd.DataFrame:
    m = clin.merge(sheet, on="sample_id", suffixes=("", "_cur"))
    rows = []

    def add(mask, field, cur, new):
        for _, r in m[mask].iterrows():
            rows.append(dict(sample_id=r.sample_id, study=r.study, field=field,
                             current=r[cur], curated=r[new], source=r.clin_source,
                             confidence=r.clin_confidence))

    dxc = m.clin_dx.map(DX_COARSE).fillna("CTCL_other")
    m["_dx_coarse"] = dxc
    add(m.disease != dxc, "disease", "disease", "_dx_coarse")
    cur_st, new_st = m.stage_clean.replace({"nan": "", "NA": ""}), m.clin_stage
    add((cur_st != "") & (new_st != "") & (cur_st != new_st), "stage_clean", "stage_clean",
        "clin_stage")
    lin = m.lineage.map({"CD4": "CD4", "CD8": "CD8", "gamma_delta": "gd"}).fillna("")
    add((lin != "") & m.clin_phenotype.isin(["CD4", "CD8", "gd", "DN"]) & (lin != m.clin_phenotype),
        "lineage", "lineage", "clin_phenotype")
    sx = m.sex.map({"M": "M", "F": "F", "male": "M", "female": "F"}).fillna("")
    m["_sx"] = sx
    add((sx != "") & (m.clin_sex != "") & (sx != m.clin_sex), "sex", "_sx", "clin_sex")
    add(m.patient_key != m.clin_person_key, "patient_key", "patient_key", "clin_person_key")
    tc = m.treatment_context.isin(["dupilumab_course", "photopheresis/vorinostat_course",
                                   "ex_vivo_baseline"])
    m["_tx"] = m.clin_current_tx + " | ex_vivo=" + m.clin_ex_vivo
    add(tc, "treatment_context", "treatment_context", "_tx")
    return pd.DataFrame(rows)


def main() -> int:
    sheet = sample_sheet()
    raw = CM.load_raw()
    clin = CM.harmonize(raw, sheet[["sample_id", "study", "compartment", "patient_key"]])
    missing = set(sheet.sample_id) - set(clin.sample_id)
    assert not missing, f"no clinical row for {sorted(missing)}"
    stage_changes = CM.validate(clin)               # paper-derived values only
    clin, sex_mm = add_expr_sex(clin)
    tcr = tcr_identity(clin)
    flag = (tcr[~tcr.explained].groupby("sample_id").trb_cdr3.first().map(
        lambda c: f"dominant TRB {c} shared with: "
                  + ", ".join(tcr[(tcr.trb_cdr3 == c)].sample_id)))
    clin["clin_qc_flag"] = clin.sample_id.map(flag).fillna("")
    clin.to_csv(CM.CLIN_CSV, index=False)
    print(f"{CM.CLIN_CSV.relative_to(NB_DIR)}: {len(clin)} samples, "
          f"{clin.clin_person_key.nunique()} persons, {clin.shape[1]} columns")

    cov, con, cf = coverage(clin), contrasts(clin), conflicts(clin, sheet)
    cf = pd.concat([cf, sex_mm], ignore_index=True)
    cov.to_csv(TAB / "clinical_v1_coverage.csv", index=False)
    con.to_csv(TAB / "clinical_v1_contrasts.csv", index=False)
    cf.to_csv(TAB / "clinical_v1_conflicts.csv", index=False)
    tcr.to_csv(TAB / "clinical_v1_tcr_identity.csv", index=False)
    stage_changes.to_csv(TAB / "clinical_v1_stage_changes.csv", index=False)

    s = sheet.set_index("sample_id").loc[clin.sample_id]
    known = lambda v: (~pd.Series(v).astype(str).isin(UNKNOWN)).sum()   # noqa: E731
    print("\ncoverage, samples with a known value (current -> curated):")
    for name, cur, new in [("stage", s.stage_clean, clin.clin_stage),
                           ("lineage/phenotype", s.lineage, clin.clin_phenotype),
                           ("sex", s.sex, clin.clin_sex),
                           ("treatment (any)", s.treatment_context,
                            clin.clin_prior_tx.where(clin.clin_prior_tx != "",
                                                     clin.clin_current_tx)),
                           ("prior treatment", [""] * len(clin), clin.clin_prior_tx),
                           ("current treatment", [""] * len(clin), clin.clin_current_tx)]:
        print(f"  {name:18s} {known(cur):4d} -> {known(new):4d}  / {len(clin)}")
    print(f"\nconflicts: {len(cf)} rows  " + str(cf.field.value_counts().to_dict()))
    print(f"TCR identity: {tcr.trb_cdr3.nunique()} shared dominant clones, "
          f"{(~tcr.explained).sum()} sample rows unexplained -> clin_qc_flag")
    print(f"stage differs across a person's samples: "
          f"{stage_changes.clin_person_key.nunique()} persons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
