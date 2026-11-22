"""Donor truth for the surface-target screens (nb42/47/48 skin, nb43/44/45/46 blood).

The target tables are keyed on obs `donor`, and that key is not a patient in either compartment.

**Blood.** 54 of the 131 keys are geskin2026 lane x HTO wells -- ex vivo 2-day cultures with five
drug arms and one control arm per lane -- so a test that treats donors as independent counts one
patient six times. Five more are healthy donors carried under `disease == "SS"`, which puts their
CD4 cells in the reactive arm of a tumour-vs-reactive comparison. Two patients appear under two
keys.

**Skin.** No ex vivo drug arms, and `disease` agrees with `clin_dx` on who is healthy. Three other
things are wrong with the raw key list: ten patients were biopsied twice or more and appear under
up to four keys (vienna:MF318 has four), nineteen keys carry a malignant clone that is **not CD4**
(CD8, gd or DN by `clin_phenotype`, plus the CD8 aggressive cytotoxic and pcAECTCL cases), and
fifteen carry a diagnosis outside the MF/SS spectrum (LyP, SPP, LPP). A CD8 clone has no malignant
CD4 arm to speak of, so its cells enter `CD4_reactive` and are read as the benign compartment.

Pseudo-replication matters here beyond the usual: `frac_donors_up` is the statistic the screens'
artifact diagnosis rests on, and correlated samples inflate it directly.

Everything below is read from the `clin_*` columns the clinical build already wrote per cell
(`jobs/apply_clinical_meta.py`), so this is an obs-only read -- no matrix, safe on a login node.
The curation behind those columns is `docs/CLINICAL_METADATA.md`; the lines cited in the
constants are that file.

    import target_cohort as TC
    don = TC.donor_table("skin")          # one row per donor key, index = donor
    keep = don.index[don["cohort_keep"]]  # the independent cohort
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- blood cohort rules
# Ex vivo arms. CLINICAL_METADATA.md:138-140 -- "213,181 cells are drug-treated. Filter
# clin_ex_vivo != 'culture_drug'". `culture_ctr` is the vehicle well of the same 2-day culture:
# kept by default, because dropping it would remove the geskin donors entirely, but flagged as
# `is_cultured` so a surface result can be shown with and without it.
EX_VIVO_DROP = "culture_drug"
CULTURED = ("culture_drug", "culture_ctr")

# Same person under two donor keys. CLINICAL_METADATA.md:124-132. `clin_person_key` already
# merges both pairs, but it also merges D13__HB1-3, which are three *different* demultiplexed
# healthy donors sharing one key -- so the collapse is listed explicitly rather than derived
# from the person key.
PERSON_DUPES = {
    "B1__P5": "B1__P4",      # Borcherding 2019: P5 is the non-malignant sort of P4
    "H__SS2": "D13__SZ22",   # pittsburgh:MF19, one patient at two centres (IIIB vs IVA1)
}

# buus2025 PT05/PT06/PT18 share PT06's dominant TRB clone (96% / 22% / 7%). Contamination or a
# label swap, not one person -- flagged, not dropped. CLINICAL_METADATA.md:134. Read off
# `clin_qc_flag` rather than hardcoded, because the flag is keyed on sample_id and the target
# tables are keyed on donor.

# ---------------------------------------------------------------- skin cohort rules
# A donor key whose clone is not CD4 has no malignant CD4 arm: `mal_cluster` still labels CD4
# clusters in that sample, so its cells land in CD4_reactive and are read as benign tissue beside
# a tumour that is somewhere else entirely. CLINICAL_METADATA.md:145-147 (D6 MF5 is a CD8
# aggressive cytotoxic CTCL; D1 carries 4 Berti CD8 and 5 gd-MF).
NON_CD4_PHENOTYPES = ("CD8", "gd", "DN")
NON_CD4_DX = ("CD8_aggr_CTCL", "pcAECTCL")

# The screen is a comparison inside the MF/SS spectrum. LyP is a CD30+ lymphoproliferative
# disorder and SPP/LPP are parapsoriasis -- related, not the same disease, and 15 keys of them
# would enter the malignant arm of an MF/SS screen.
OFF_SPECTRUM_DX = ("LyP", "SPP", "LPP")

# Which biopsy of a patient sampled more than once is the one to keep: a lesion over a
# non-lesion, then the deeper sample. Healthy skin ranks with the lesions because an HC donor has
# nothing else.
LESION_RANK = {"tumor": 0, "erythroderma": 0, "plaque": 0, "patch": 0, "papule": 0, "mixed": 0,
               "LyP_lesion": 0, "healthy_skin": 0, "skin_unspecified": 1, "cleared_lesion": 2,
               "nonlesional": 3}
LESIONAL = ("tumor", "erythroderma", "plaque", "patch", "papule", "mixed", "LyP_lesion")

# Keys that differ only by an HTO index are demultiplexed *different* donors that share one
# patient key -- the D13__HB1-3 case in blood (CLINICAL_METADATA.md:124-134). They are never
# collapsed, however much their person key suggests it.
_HTO_SUFFIX = re.compile(r"_HTO\d+$")

DX_ORDER = ["SS", "MF", "MF/SS", "MF_or_SS_unresolved", "HC"]

CLIN_COLS = [
    "clin_person_key", "clin_dx", "clin_ex_vivo", "clin_arm", "clin_qc_flag",
    "clin_sex", "clin_age", "clin_race", "clin_duration_y",
    "clin_stage", "clin_stage_class", "clin_stage_timing", "clin_B", "clin_T",
    "clin_blood_tumor_pct",
    "clin_current_tx", "clin_current_tx_classes", "clin_current_systemic",
    "clin_prior_tx_classes", "clin_ever_systemic", "clin_tx_naive", "clin_n_prior_lines",
    "clin_timepoint", "clin_days_from_first", "clin_response", "clin_overall_response",
    "clin_drug_sensitivity", "clin_outcome", "clin_input",
]
#: fields that only one compartment carries, or only one compartment has a use for
TISSUE_COLS = {
    "blood": [],
    "skin": ["clin_lesion", "clin_phenotype", "clin_group", "clin_cd30", "clin_lct", "clin_fmf",
             "clin_course"],
}
BASE_COLS = ["donor", "study", "disease"]

# Treatment classes, as `clinical_meta.AGENTS` spells them. Membership is tested against the
# comma-separated `clin_current_tx_classes` / `clin_prior_tx_classes` strings.
TX_CLASSES = ["ecp", "retinoid", "interferon", "methotrexate", "hdaci", "mogamulizumab",
              "brentuximab", "alemtuzumab", "chemotherapy", "checkpoint", "other_targeted",
              "dupilumab", "alloSCT", "other_immunomod", "systemic_steroid"]
#: skin-directed agents, which exist only in the skin cohort and are never systemic exposure
TX_SKIN_CLASSES = ["topical_steroid", "mechlorethamine", "topical_retinoid", "topical_immunomod",
                   "nbuvb", "puva", "phototherapy", "tseb", "radiotherapy"]

_MISSING = {"", "nan", "NaN", "None", "unknown", "NA", "<NA>"}


def _clean(s: pd.Series) -> pd.Series:
    """Strings with the project's several spellings of 'missing' collapsed to NA."""
    out = s.astype(str).str.strip()
    return out.mask(out.isin(_MISSING))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_clean(s), errors="coerce")


def _read_donor_obs(tissue: str, obj=None, extra_cols=()) -> pd.DataFrame:
    """One row per `donor` key, every `clin_*` field this module reads, cleaned."""
    import cluster_malig_helpers as CM
    import h5py

    CM.use_tissue(tissue)
    obj = Path(obj) if obj is not None else Path(CM.CONFIG[tissue]["TCR_OBJ"])
    cols = BASE_COLS + CLIN_COLS + TISSUE_COLS[tissue]
    cols += [c for c in extra_cols if c not in cols]
    with h5py.File(obj, "r") as h5:
        obs = CM.read_obs(h5, cols=cols)

    for c in cols:
        obs[c] = obs[c].astype(str)

    don = (obs.groupby("donor", observed=True)
              .agg(n_cells=("study", "size"),
                   **{c: (c, "first") for c in cols if c != "donor"}))
    for c in don.columns:
        if c != "n_cells":
            don[c] = _clean(don[c])
    return don


def _common_fields(don: pd.DataFrame, tx_classes) -> pd.DataFrame:
    """The derived columns both compartments share: diagnosis, demographics, treatment flags."""
    don["person"] = don["clin_person_key"].fillna(don.index.to_series())
    don["dx"] = don["clin_dx"].fillna("unknown")
    don["is_hc"] = don["dx"].eq("HC")
    # MF/SS and MF_or_SS_unresolved cannot enter an MF-vs-SS contrast on either side.
    don["dx2"] = don["dx"].where(don["dx"].isin(["MF", "SS", "HC"]), "unresolved")
    don["is_cultured"] = don["clin_ex_vivo"].isin(CULTURED)
    don["clone_shared"] = don["clin_qc_flag"].notna()
    don["age"] = _num(don["clin_age"])
    don["sex"] = don["clin_sex"].where(don["clin_sex"].isin(["F", "M"]))
    don["blood_tumor_pct"] = _num(don["clin_blood_tumor_pct"])
    don["n_prior_lines"] = _num(don["clin_n_prior_lines"])

    for field, prefix in (("clin_current_tx_classes", "tx"), ("clin_prior_tx_classes", "prior")):
        txt = don[field].fillna("")
        for k in tx_classes:
            don[f"{prefix}_{k}"] = txt.str.contains(rf"\b{k}\b", case=False, regex=True)
        # "known to be on nothing" is not "unknown". `clin_current_tx_classes` is empty for
        # both, so the systemic flag is what separates them: it carries an explicit "no".
        sysc = "clin_current_systemic" if prefix == "tx" else "clin_ever_systemic"
        don[f"{prefix}_known"] = don[field].notna() | don.get(
            sysc, pd.Series(pd.NA, index=don.index)).notna()
    # "ever on it" = at sampling or before.
    for k in tx_classes:
        don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
    return don.copy()               # one block, instead of ~90 inserted columns


def blood_donor_table(obj=None, extra_cols=()) -> pd.DataFrame:
    """One row per blood `donor` key, with the curated clinical fields and the cohort flags.

    Added on top of the raw `clin_*` values:

      cohort_keep   keep this key as an independent donor
      drop_reason   why not, when `cohort_keep` is False
      dx            curated diagnosis, from `clin_dx`, never from `disease`
      person        `clin_person_key`
      is_cultured   the cells spent 2 days in culture (every geskin key)
      is_hc         a healthy donor, whatever `disease` says
      dx2           MF / SS / HC / unresolved -- the two-arm form for a disease contrast
      clone_shared  buus PT05/PT06/PT18, from `clin_qc_flag`
      tx_*          one boolean per treatment class, at sampling and ever
    """
    don = _common_fields(_read_donor_obs("blood", obj, extra_cols), TX_CLASSES)

    drop = pd.Series(pd.NA, index=don.index, dtype="object")
    drop[don["clin_ex_vivo"].eq(EX_VIVO_DROP)] = "ex vivo drug-culture arm"
    for dupe, kept in PERSON_DUPES.items():
        if dupe in drop.index:
            drop[dupe] = f"same person as {kept}"
    don = don.assign(drop_reason=drop, cohort_keep=drop.isna(), tissue="blood")
    return don.sort_index()


def _skin_person_duplicates(don: pd.DataFrame) -> dict[str, str]:
    """{dropped key: kept key} for patients biopsied more than once.

    The kept key is the lesional sample, and among lesional samples the deepest. Keys that differ
    only by an HTO index are left alone: an HTO index demultiplexes different donors that share a
    patient key, which is the D13__HB1-3 case in blood.
    """
    out: dict[str, str] = {}
    rank = don["clin_lesion"].map(LESION_RANK).fillna(1)
    for person, g in don.groupby("person"):
        if len(g) < 2:
            continue
        bases = {_HTO_SUFFIX.sub("", k) for k in g.index}
        if len(bases) == 1:
            continue                      # one sample, several HTO wells -> different donors
        order = sorted(g.index, key=lambda k: (rank[k], -don.loc[k, "n_cells"], k))
        for k in order[1:]:
            out[k] = order[0]
    return out


def skin_donor_table(obj=None, extra_cols=()) -> pd.DataFrame:
    """One row per skin `donor` key, with the curated clinical fields and the cohort flags.

    Same columns as `blood_donor_table`, plus the skin-only ones:

      is_lesional    the biopsy is a lesion, not healthy or uninvolved skin
      lesion         `clin_lesion`
      lesion_class   lesion / nonlesional / healthy / unspecified
      phenotype      `clin_phenotype` -- the lineage of the malignant clone
      is_cd4_clone   the clone is CD4 (or the donor is healthy)
      tx_skin_*      one boolean per skin-directed agent class

    Three drop rules, in the order they are applied, each with its own `drop_reason`:

      1. the malignant clone is not CD4 -- a CD8, gd or DN clone has no malignant CD4 arm, so
         its cells sit in CD4_reactive and are read as the benign compartment beside a tumour;
      2. the diagnosis is outside the MF/SS spectrum (LyP, SPP, LPP);
      3. the patient is already in the cohort under another key.
    """
    don = _common_fields(_read_donor_obs("skin", obj, extra_cols),
                         TX_CLASSES + TX_SKIN_CLASSES)

    don = don.assign(
        lesion=don["clin_lesion"],
        is_lesional=don["clin_lesion"].isin(LESIONAL) & ~don["is_hc"],
        lesion_class=np.select(
            [don["is_hc"], don["clin_lesion"].isin(LESIONAL),
             don["clin_lesion"].isin(["nonlesional", "cleared_lesion"])],
            ["healthy", "lesion", "nonlesional"], "unspecified"),
        phenotype=don["clin_phenotype"],
        is_cd4_clone=(don["is_hc"] | (~don["clin_phenotype"].isin(NON_CD4_PHENOTYPES)
                                      & ~don["dx"].isin(NON_CD4_DX))),
    )

    drop = pd.Series(pd.NA, index=don.index, dtype="object")
    drop[~don["is_cd4_clone"]] = ("malignant clone is not CD4 ("
                                  + don["phenotype"].fillna(don["dx"]) + ")")
    drop[don["dx"].isin(OFF_SPECTRUM_DX)] = "diagnosis outside the MF/SS spectrum (" + don["dx"] + ")"
    # The duplicate rule runs last and only over keys still standing, so a patient is never
    # dropped as "already in the cohort" against a key that is itself out.
    live = don[drop.isna()]
    for dupe, kept in _skin_person_duplicates(live).items():
        drop[dupe] = f"same person as {kept}"
    don = don.assign(drop_reason=drop, cohort_keep=drop.isna(), tissue="skin")
    return don.sort_index()


def donor_table(tissue: str, obj=None, extra_cols=()) -> pd.DataFrame:
    """`blood_donor_table` or `skin_donor_table`, chosen by name."""
    fn = {"blood": blood_donor_table, "skin": skin_donor_table}[tissue]
    return fn(obj=obj, extra_cols=extra_cols)


def filter_long(long: pd.DataFrame, don: pd.DataFrame, hc_arm="CD4_HC",
                reactive_arm="CD4_reactive") -> pd.DataFrame:
    """Drop the non-independent donor keys, and move healthy donors out of the reactive arm.

    In blood, `run_target_screen.py` assigns `CD4_HC` on the atlas `disease` column, which calls
    the five geskin healthy donors SS. Their CD4 cells therefore land in `CD4_reactive`, where
    they are read as the benign compartment beside a tumour. Reassigned here on `clin_dx`. In
    skin the two columns agree on who is healthy, so this moves nothing and is left in place as
    the guarantee rather than removed.
    """
    keep = set(don.index[don["cohort_keep"]])
    out = long[long["donor"].isin(keep)].copy()
    hc = set(don.index[don["is_hc"]])
    moved = out["arm"].eq(reactive_arm) & out["donor"].isin(hc)
    out.loc[moved, "arm"] = hc_arm
    return out


def cohort_report(don: pd.DataFrame, long: pd.DataFrame | None = None,
                  malig="CD4_malignant", react="CD4_reactive", tissue=None) -> str:
    """The text block nb42/nb43 print: what was removed, what is left, the paired-donor count."""
    if tissue is None:
        tissue = don["tissue"].iloc[0] if "tissue" in don else "blood"
    L = []
    P = L.append
    P(f"{tissue} donor keys in the object: {len(don):,}")
    rs = don.loc[~don["cohort_keep"], "drop_reason"].value_counts()
    for reason, n in rs.items():
        P(f"  - {n:3d} dropped: {reason}")
    keep = don[don["cohort_keep"]]
    P(f"  = {len(keep):3d} independent donor keys ({keep['person'].nunique()} people)")
    P("")
    P("dx x study, independent keys:")
    P(pd.crosstab(keep["dx"], keep["study"]).to_string())
    P("")
    if "lesion_class" in keep:
        P("lesion class, independent keys: "
          + ", ".join(f"{k} {v}" for k, v in keep["lesion_class"].value_counts().items()))
    P(f"ex vivo 2-day culture (vehicle arm kept, flagged): {int(keep['is_cultured'].sum())} keys")
    P(f"shared dominant TRB clone (buus PT05/06/18):       {int(keep['clone_shared'].sum())} keys")

    if long is not None:
        arms = long[long["arm"].isin([malig, react])]
        both = arms.groupby("donor")["arm"].nunique().eq(2)
        paired = don.loc[don.index.intersection(both.index[both])]
        P("")
        P(f"donors with BOTH a malignant and a reactive arm: {len(paired)} "
          f"({paired['person'].nunique()} people)")
        P(paired["dx"].value_counts().to_string())
    return "\n".join(L)


def burden(long: pd.DataFrame, malig="CD4_malignant", react="CD4_reactive") -> pd.DataFrame:
    """Malignant share of a donor's CD4 compartment, from the arm cell counts themselves.

    The one donor-level covariate with full coverage -- `clin_blood_tumor_pct` is curated for 21
    blood donors and fewer in skin. It is also the mechanistic test for differential
    contamination: a signal carried by doublets with a minor population scales with the
    malignant:reactive abundance ratio, and real biology does not.
    """
    n = (long[long["arm"].isin([malig, react])]
         .groupby(["donor", "arm"])["n_cells"].first().unstack("arm"))
    n = n.reindex(columns=[malig, react])
    out = pd.DataFrame({
        "n_malignant": n[malig], "n_reactive": n[react],
        "burden": n[malig] / (n[malig] + n[react]),
        "log_ratio": np.log2((n[malig] + 1) / (n[react] + 1)),
    })
    return out
