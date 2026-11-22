"""Table 1 for atlas v2: cohort composition, split skin / blood.

Reads the obs cache (`data/atlas_joint/atlas_obs_full_v2.parquet`) only -- no expression, so it
runs on a login node in seconds -- and writes one wide CSV, `tables/table1_atlas_v2.csv`.

Rows are `section / characteristic / level`. Every level carries four counting units per
compartment, because they answer different questions:

  n_persons   unique `clin_person_key` -- the patient, after the identity merges of
              docs/CLINICAL_METADATA.md section 4. The denominator for any clinical claim.
  n_donor_units  unique `donor` -- the donor unit the atlas was built on, which is what
              `helpers/target_cohort.py` starts from (146 skin / 131 blood) and what
              `tables/atlas_v2_headline.csv` calls donor-units. Larger than n_samples in blood,
              because geskin2026's 11 `sample_id`s are HTO lanes pooling 65 donor wells.
  n_samples   unique `sample_id` -- the library.
  n_cells     cells.

Continuous variables (age, disease duration, prior lines, blood tumour burden) are summarized
once per person and reported in the `value_*` columns instead of the count columns.

Treatment is reported twice over, as the clinical layer stores it: `clin_prior_tx` is what the
patient had **before** the sample, `clin_current_tx` what they were on **at** sampling. Every
agent in `clinical_meta.AGENTS` gets a row in each block even when it is never used, so the
absent agents are visible. `ever` is the union of the two.

Usage:  python jobs/build_table1.py [--out tables/table1_atlas_v2.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import clinical_meta as CM  # noqa: E402

OBS = NB_DIR / "data" / "atlas_joint" / "atlas_obs_full_v2.parquet"

COMPARTMENTS = ["Skin", "Blood", "LN"]
SUFFIX = {"Skin": "skin", "Blood": "blood", "LN": "LN"}

# Per-cell columns kept in the compact unit table. Everything else is per sample_id.
UNIT_COLS = ["compartment", "study", "sample_id", "donor", "clin_person_key",
             "clin_ex_vivo", "clin_arm", "tech", "tissue", "skin_layer", "clin_input"]

SAMPLE_COLS = ["clin_group", "clin_dx", "clin_phenotype", "clin_cd30", "clin_lct", "clin_fmf",
               "clin_stage", "clin_stage_class", "clin_stage_timing", "clin_T", "clin_N",
               "clin_M", "clin_B", "clin_lesion", "clin_sex", "clin_sex_source", "clin_race",
               "clin_timepoint", "clin_response", "clin_response_scope", "clin_overall_response",
               "clin_drug_sensitivity", "clin_course", "clin_outcome", "clin_tx_naive",
               "clin_prior_systemic", "clin_current_systemic", "clin_ever_systemic",
               "clin_confidence", "clin_qc_flag", "clin_prior_tx", "clin_current_tx",
               "clin_prior_tx_classes", "clin_current_tx_classes"]

NUM_COLS = ["clin_age", "clin_duration_y", "clin_n_prior_lines", "clin_blood_tumor_pct"]

CELL_COLS = ["cell_type_h", "cell_type_broad", "lineage", "has_tcr", "is_malignant",
             "malignant_call_method"]

MISSING = {"", "nan", "None", "<NA>", "NaN"}

# Level orders that are not frequency orders.
ORDERS = {
    "clin_dx": CM.DX,
    "clin_phenotype": CM.PHENOTYPE,
    "clin_stage": CM.STAGES + ["not_staged"],
    "clin_stage_class": ["early", "advanced", "not_staged"],
    "clin_lesion": CM.LESION,
    "clin_timepoint": CM.TIMEPOINT,
    "clin_outcome": CM.OUTCOME,
    "clin_ex_vivo": CM.EX_VIVO,
    "clin_input": CM.INPUT,
    "clin_cd30": ["pos", "neg", "unknown", "not_applicable"],
    "clin_lct": ["yes", "no", "unknown", "not_applicable"],
    "clin_fmf": ["yes", "no", "unknown", "not_applicable"],
    "clin_tx_naive": ["yes", "no", "washout_6mo", "unknown"],
    "clin_prior_systemic": CM.YNU,
    "clin_current_systemic": CM.YNU,
    "clin_ever_systemic": CM.YNU,
    "clin_sex": ["M", "F", "unknown"],
}


def _txt(s: pd.Series, blank: str) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(~s.isin(MISSING), blank)


def load_units() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compact unit table (one row per donor x sample x ex-vivo arm), cell-type table, numerics."""
    cols = sorted(set(UNIT_COLS + SAMPLE_COLS + NUM_COLS + CELL_COLS))
    obs = pq.read_table(OBS, columns=cols).to_pandas()

    for c in UNIT_COLS + SAMPLE_COLS:
        obs[c] = _txt(obs[c], "unknown" if c not in ("clin_prior_tx", "clin_current_tx",
                                                     "clin_prior_tx_classes",
                                                     "clin_current_tx_classes") else "")
    for c in ("clin_stage", "clin_stage_class", "clin_stage_timing",
              "clin_T", "clin_N", "clin_M", "clin_B"):
        obs[c] = obs[c].replace("unknown", "not_staged")

    units = (obs.groupby(UNIT_COLS, observed=True, dropna=False)
                .size().rename("n_cells").reset_index())
    per_sample = obs.drop_duplicates("sample_id").set_index("sample_id")[SAMPLE_COLS]
    units = units.join(per_sample, on="sample_id")

    cells = {}
    for c in CELL_COLS:
        g = (obs.groupby(["compartment", c], observed=True, dropna=False)
                .agg(n_cells=("sample_id", "size"), n_samples=("sample_id", "nunique"),
                     n_persons=("clin_person_key", "nunique"),
                     n_donor_units=("donor", "nunique")).reset_index())
        cells[c] = g.rename(columns={c: "level"}).assign(level=lambda d: d.level.astype(str))

    nums_long = (obs[["clin_person_key", "compartment"] + NUM_COLS]
                 .drop_duplicates(["clin_person_key", "compartment"]))
    return units, cells, nums_long


class Table1:
    def __init__(self, units: pd.DataFrame):
        self.units = units
        self.rows: list[dict] = []
        self.tot = {}
        for comp in COMPARTMENTS + ["all"]:
            u = units if comp == "all" else units[units.compartment == comp]
            self.tot[comp] = dict(n_persons=u.clin_person_key.nunique(),
                                  n_donor_units=u.donor.nunique(),
                                  n_samples=u.sample_id.nunique(),
                                  n_cells=int(u.n_cells.sum()))

    @staticmethod
    def _count(u: pd.DataFrame) -> dict:
        return dict(n_persons=u.clin_person_key.nunique(), n_donor_units=u.donor.nunique(),
                    n_samples=u.sample_id.nunique(), n_cells=int(u.n_cells.sum()))

    def add(self, section: str, characteristic: str, level: str, mask: pd.Series | None = None,
            note: str = "", values: dict | None = None) -> None:
        row = {"section": section, "characteristic": characteristic, "level": level}
        for comp in COMPARTMENTS + ["all"]:
            sfx, tot = SUFFIX.get(comp, comp), self.tot[comp]
            if values is not None:
                for k in ("n_persons", "n_donor_units", "n_samples", "n_cells",
                          "pct_persons", "pct_samples", "pct_cells"):
                    row[f"{k}_{sfx}"] = ""
                row[f"value_{sfx}"] = values.get(comp, "")
                continue
            sub = self.units if mask is None else self.units[mask]
            sub = sub if comp == "all" else sub[sub.compartment == comp]
            c = self._count(sub)
            row |= {f"{k}_{sfx}": v for k, v in c.items()}
            for unit in ("persons", "samples", "cells"):
                d = tot[f"n_{unit}"]
                row[f"pct_{unit}_{sfx}"] = round(100 * c[f"n_{unit}"] / d, 1) if d else ""
            row[f"value_{sfx}"] = ""
        row["note"] = note
        self.rows.append(row)

    def add_cat(self, section: str, characteristic: str, col: str, note: str = "",
                order: list | None = None) -> None:
        s = self.units[col]
        order = order or ORDERS.get(col)
        levels = ([x for x in order if (s == x).any()] +
                  sorted(set(s.unique()) - set(order or []))) if order else \
            list(self.units.groupby(col, observed=True).n_cells.sum()
                 .sort_values(ascending=False).index)
        for lev in levels:
            self.add(section, characteristic, lev, s == lev, note)

    def add_cont(self, section: str, characteristic: str, nums: pd.DataFrame, col: str,
                 note: str = "", fmt: str = "{:.0f}") -> None:
        for stat in ("n persons with a value", "median (IQR)", "range"):
            vals = {}
            for comp in COMPARTMENTS + ["all"]:
                d = nums if comp == "all" else nums[nums.compartment == comp]
                v = pd.to_numeric(d.drop_duplicates("clin_person_key")[col], errors="coerce").dropna()
                if stat.startswith("n persons"):
                    vals[comp] = int(v.size)
                elif v.empty:
                    vals[comp] = ""
                elif stat.startswith("median"):
                    vals[comp] = (f"{fmt.format(v.median())} "
                                  f"({fmt.format(v.quantile(.25))}-{fmt.format(v.quantile(.75))})")
                else:
                    vals[comp] = f"{fmt.format(v.min())}-{fmt.format(v.max())}"
            self.add(section, characteristic, stat, note=note, values=vals)

    def frame(self) -> pd.DataFrame:
        cols = ["section", "characteristic", "level"]
        for comp in COMPARTMENTS + ["all"]:
            sfx = SUFFIX.get(comp, comp)
            cols += [f"n_persons_{sfx}", f"n_donor_units_{sfx}", f"n_samples_{sfx}", f"n_cells_{sfx}",
                     f"pct_persons_{sfx}", f"pct_samples_{sfx}", f"pct_cells_{sfx}",
                     f"value_{sfx}"]
        return pd.DataFrame(self.rows)[cols + ["note"]]


def tx_set(s: pd.Series) -> pd.Series:
    return s.map(lambda x: {a.strip() for a in str(x).split(";") if a.strip()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(NB_DIR / "tables" / "table1_atlas_v2.csv"))
    args = ap.parse_args()

    units, cells, nums = load_units()
    T = Table1(units)
    U = T.units

    # -- overview -----------------------------------------------------------------------------
    T.add("Overview", "atlas", "total",
          note="n_persons = clin_person_key after the identity merges; n_donor_units = the "
               "obs `donor` key (blood lanes are HTO-demultiplexed); n_samples = sample_id")
    T.add("Overview", "studies", "distinct studies", values={
        c: (U if c == "all" else U[U.compartment == c]).study.nunique()
        for c in COMPARTMENTS + ["all"]})
    pk = U.groupby("clin_person_key", observed=True).compartment.agg(lambda s: tuple(sorted(set(s))))
    for combo, n in pk.value_counts().items():
        T.add("Overview", "persons by compartment", " + ".join(combo),
              U.clin_person_key.isin(pk.index[pk == combo]),
              note="persons sampled in these compartments; the n_* columns count that "
                   "person's units inside each compartment column")
    T.add("Overview", "cells per sample", "median (IQR)", values={
        c: (lambda v: f"{v.median():,.0f} ({v.quantile(.25):,.0f}-{v.quantile(.75):,.0f})")(
            (U if c == "all" else U[U.compartment == c]).groupby("sample_id").n_cells.sum())
        for c in COMPARTMENTS + ["all"]})

    # -- provenance ---------------------------------------------------------------------------
    T.add_cat("Study", "study", "study")
    T.add_cat("Assay", "chemistry (tech)", "tech")
    T.add_cat("Assay", "input material (clin_input)", "clin_input")
    T.add_cat("Tissue", "tissue", "tissue")
    T.add_cat("Tissue", "skin layer", "skin_layer")
    T.add_cat("Provenance", "metadata confidence (clin_confidence)", "clin_confidence")
    T.add("Provenance", "QC flag (clin_qc_flag)", "flagged: shared dominant TRB clone",
          U.clin_qc_flag.ne("") & U.clin_qc_flag.ne("unknown"),
          note="docs/CLINICAL_METADATA.md section 4: buus2025 PT05/PT06/PT18")

    # -- disease ------------------------------------------------------------------------------
    T.add_cat("Disease", "disease group (clin_group)", "clin_group")
    T.add_cat("Disease", "diagnosis (clin_dx)", "clin_dx",
              note="erythrodermic MF is MF with clin_lesion = erythroderma")
    T.add_cat("Disease", "malignant clone phenotype (clin_phenotype)", "clin_phenotype",
              note="unknown means not reported, not CD4")
    T.add_cat("Disease", "CD30 (clin_cd30)", "clin_cd30", note="unknown is not negative")
    T.add_cat("Disease", "large-cell transformation (clin_lct)", "clin_lct",
              note="unknown is not negative")
    T.add_cat("Disease", "folliculotropic MF (clin_fmf)", "clin_fmf", note="unknown is not negative")
    T.add_cat("Disease", "course (clin_course)", "clin_course")
    T.add_cat("Disease", "outcome (clin_outcome)", "clin_outcome")

    # -- stage --------------------------------------------------------------------------------
    T.add_cat("Stage", "stage class (clin_stage_class)", "clin_stage_class",
              note="early = IA-IIA; not_staged = HC, parapsoriasis, LyP or unpublished")
    T.add_cat("Stage", "ISCL/EORTC stage (clin_stage)", "clin_stage")
    T.add_cat("Stage", "what the stage refers to (clin_stage_timing)", "clin_stage_timing")
    for ax in ("T", "N", "M", "B"):
        T.add_cat("Stage", f"TNMB: {ax} (clin_{ax})", f"clin_{ax}")
    T.add_cat("Stage", "lesion biopsied (clin_lesion)", "clin_lesion")

    # -- demographics -------------------------------------------------------------------------
    T.add_cat("Demographics", "sex (clin_sex)", "clin_sex")
    T.add_cat("Demographics", "sex source (clin_sex_source)", "clin_sex_source",
              note="expr = inferred from XIST / chrY where no paper reports it")
    T.add_cat("Demographics", "race (clin_race)", "clin_race")
    T.add_cont("Demographics", "age at sampling, years (clin_age)", nums, "clin_age")
    T.add_cont("Demographics", "disease duration, years (clin_duration_y)", nums,
               "clin_duration_y", fmt="{:.1f}")
    T.add_cont("Disease", "blood tumour burden, % of CD4 T (clin_blood_tumor_pct)", nums,
               "clin_blood_tumor_pct", fmt="{:.1f}")

    # -- sampling design ----------------------------------------------------------------------
    T.add_cat("Design", "timepoint (clin_timepoint)", "clin_timepoint")
    T.add_cat("Design", "ex vivo handling (clin_ex_vivo)", "clin_ex_vivo",
              note="geskin2026 is a 2-day ex vivo culture; culture_drug cells are drug-treated")
    T.add_cat("Design", "ex vivo arm (clin_arm)", "clin_arm")
    T.add_cat("Design", "response (clin_response)", "clin_response")
    T.add_cat("Design", "overall response (clin_overall_response)", "clin_overall_response")
    T.add_cat("Design", "ex vivo drug sensitivity (clin_drug_sensitivity)", "clin_drug_sensitivity")

    # -- treatment ----------------------------------------------------------------------------
    tx_note = ("an agent row counts explicit mentions only; a blank treatment field is not "
               "evidence of no treatment -- read it against clin_tx_naive")
    T.add_cat("Treatment status", "treatment naive (clin_tx_naive)", "clin_tx_naive",
              note="washout_6mo = the PKU cohort")
    T.add_cat("Treatment status", "systemic therapy before sampling (clin_prior_systemic)",
              "clin_prior_systemic", note="ECP counts as systemic; a bare 'steroid' is unknown")
    T.add_cat("Treatment status", "systemic therapy at sampling (clin_current_systemic)",
              "clin_current_systemic")
    T.add_cat("Treatment status", "systemic therapy ever (clin_ever_systemic)", "clin_ever_systemic")
    T.add_cont("Treatment status", "prior lines of therapy (clin_n_prior_lines)", nums,
               "clin_n_prior_lines", note="dorando2026 only")
    for lab, col in [("any treatment recorded before sampling", "clin_prior_tx"),
                     ("any treatment recorded at sampling", "clin_current_tx")]:
        T.add("Treatment status", lab, "yes", U[col].ne(""), note=tx_note)
        T.add("Treatment status", lab, "none recorded", U[col].eq(""), note=tx_note)

    prior, current = tx_set(U.clin_prior_tx), tx_set(U.clin_current_tx)
    ever = pd.Series([a | b for a, b in zip(prior, current)], index=U.index)
    pri_c, cur_c = tx_set(U.clin_prior_tx_classes), tx_set(U.clin_current_tx_classes)
    eve_c = pd.Series([a | b for a, b in zip(pri_c, cur_c)], index=U.index)

    classes = sorted({c for _, c, _, _ in CM.AGENTS})
    for block, sets in [("prior (before the sample)", pri_c), ("current (at sampling)", cur_c),
                        ("ever (prior or current)", eve_c)]:
        for cl in classes:
            T.add(f"Treatment classes - {block}", "treatment class", cl,
                  sets.map(lambda s, c=cl: c in s),
                  note=("skin-directed" if cl in CM.SKIN_DIRECTED else
                        "unspecified" if cl in CM.UNSPECIFIED else "systemic") + "; " + tx_note)
        T.add(f"Treatment classes - {block}", "treatment class",
              "any skin-directed", sets.map(lambda s: bool(s & CM.SKIN_DIRECTED)), note=tx_note)
        T.add(f"Treatment classes - {block}", "treatment class", "any systemic",
              sets.map(lambda s: bool(s - CM.SKIN_DIRECTED - CM.UNSPECIFIED)), note=tx_note)

    agents = [a for a, _, _, _ in CM.AGENTS]
    for block, sets in [("prior (before the sample)", prior), ("current (at sampling)", current),
                        ("ever (prior or current)", ever)]:
        for ag in agents:
            T.add(f"Treatment agents - {block}", CM.CLASS_OF.get(ag, "other"), ag,
                  sets.map(lambda s, a=ag: a in s), note=tx_note)
    unseen = sorted({a for s in ever for a in s} - set(agents))
    for ag in unseen:
        T.add("Treatment agents - ever (prior or current)", "unmapped", ag,
              ever.map(lambda s, a=ag: a in s), note="agent not in clinical_meta.AGENTS")

    # -- cells --------------------------------------------------------------------------------
    CELL_NOTES = {
        "lineage": "T-cell lineage call",
        "has_tcr": "cell carries a TCR contig",
        "is_malignant": "atlas obs column, assembled per study (see malignant_call_method); "
                        "NOT the headline mal_cluster call",
        "malignant_call_method": "how is_malignant was set in that study",
    }
    for col in CELL_COLS:
        g = cells[col]
        levels = list(g.groupby("level").n_cells.sum().sort_values(ascending=False).index)
        for lev in levels:
            row = {"section": "Cell composition", "characteristic": col, "level": lev}
            agg = {k: 0 for k in ("n_persons", "n_donor_units", "n_samples", "n_cells")}
            for comp in COMPARTMENTS:
                sfx = SUFFIX[comp]
                sub = g[(g.compartment == comp) & (g.level == lev)]
                c = {k: int(sub[k].sum()) for k in agg}
                for k in agg:
                    agg[k] += c[k]
                row |= {f"{k}_{sfx}": v for k, v in c.items()}
                for unit in ("persons", "samples", "cells"):
                    d = T.tot[comp][f"n_{unit}"]
                    row[f"pct_{unit}_{sfx}"] = round(100 * c[f"n_{unit}"] / d, 1) if d else ""
                row[f"value_{sfx}"] = ""
            row |= {f"{k}_all": v for k, v in agg.items()}
            for unit in ("persons", "samples", "cells"):
                row[f"pct_{unit}_all"] = round(100 * agg[f"n_{unit}"] / T.tot["all"][f"n_{unit}"], 1)
            row["value_all"] = ""
            row["note"] = ("counts of persons/donors/samples are those with at least one such "
                           "cell, so they sum across compartments and do not add to the total; "
                           + CELL_NOTES.get(col, "")).strip("; ")
            T.rows.append(row)

    out = T.frame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} rows x {out.shape[1]} cols)")
    for comp in COMPARTMENTS + ["all"]:
        print(" ", comp, T.tot[comp])


if __name__ == "__main__":
    main()
