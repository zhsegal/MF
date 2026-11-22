"""Curated clinical metadata layer (`clin_*`) re-mined from the source papers.

``data/clinical/raw/*.tsv`` hold the per-sample extractions verbatim: one row per atlas
``sample_id``, every field with its ``source`` and a ``confidence`` of exact / inferred / geo.
``supplements.tsv`` overlays them field by field with the tables that were blocked online and
supplied by hand (``data/clinical/parse_user_supplied.py``). ``harmonize`` turns the result into
controlled-vocabulary ``clin_*`` columns, written by ``jobs/build_clinical_metadata.py`` to
``data/clinical_metadata_v1.csv``. ``attach_clinical(obs)`` joins them onto any atlas obs.
``jobs/apply_clinical_meta.py`` writes them into the built objects.

Treatment is split by timing. ``clin_prior_tx`` lists what the patient had before the sample.
``clin_current_tx`` lists what they were on when it was taken. Both hold normalized agent names
(``AGENTS``), each with a class (``clin_*_tx_classes``) and a systemic flag.

The existing columns (``disease``, ``stage_clean``, ``entity``, ``lineage``, ``sex``,
``treatment_context``, ``patient_key``) are left alone. Where the papers disagree with them, the
disagreement goes to ``tables/clinical_v1_conflicts.csv``. Promoting a reviewed correction is a
separate ``atlas_v2_fixups`` step. Sources, blocked papers and decisions are recorded in
``docs/CLINICAL_METADATA.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = NB_DIR / "data" / "clinical" / "raw"
RAW_FILES = ["vienna.tsv", "li_pittsburgh.tsv", "blood.tsv"]
SUPPLEMENTS = RAW_DIR / "supplements.tsv"
B4_ARMS = RAW_DIR / "b4_hto_arms.tsv"
CLIN_CSV = NB_DIR / "data" / "clinical_metadata_v1.csv"
OBS_PARQUET = NB_DIR / "data" / "atlas_joint" / "atlas_obs_full_v2.parquet"

# ---------------------------------------------------------------------------------------------
# Curation decisions the raw text cannot carry by itself (see docs/CLINICAL_METADATA.md §3)
# ---------------------------------------------------------------------------------------------

# Same person under two patient_keys.
#  * Vienna P-numbers are biopsy numbers, not patients: ChennareddyBJD25 Table 1 footnote puts
#    P97, P124 and P318A/B on one patient, and P195/P218 on another.
#  * Borcherding 2019's four GSMs are one patient: P4 is the malignant sort, P5 the non-malignant.
#  * Shared dominant TRB clone (tables/clinical_v1_tcr_identity.csv), with matching age/sex/history:
#    Gaydosik22 SS1 = MF19 skin + SZ22 blood = Herrera21 SS2 (62/M SS, dx 2019, seen at both
#    centres); Geskin26 SS32 = il4ra26 MF26 (30-32/M, erythrodermic IVB, 2021).
PERSON_MERGE = {
    "vienna:P97": "vienna:MF318",
    "vienna:P124": "vienna:MF318",
    "vienna:P218": "vienna:P195",
    "B1__P5": "B1__P4",
    "pittsburgh:SZ22": "pittsburgh:MF19",
    "H__SS2": "pittsburgh:MF19",
    "pittsburgh:SZ32": "pittsburgh:MF26",
}

# Raw stage strings that carry two sources' values. One is kept, and the reason is recorded.
STAGE_OVERRIDE = {
    "Li2024_atlas__MF18": ("IVA2", "Liu22 TNMB T3N3M0B1 (N3) -> IVA2; Li24 lists IIB"),
    "D9__P72": ("IVA1", "MolCancer21 IVA1 (MF311 follow-up); BloodAdv26 IVA"),
    "B1__P4_a": ("IVA1", "Borcherding23 S1 IVA1; 2019 text IVA"),
    "B1__P4_b": ("IVA1", "Borcherding23 S1 IVA1; 2019 text IVA"),
    "B1__P5_a": ("IVA1", "Borcherding23 S1 IVA1; 2019 text IVA"),
    "B1__P5_b": ("IVA1", "Borcherding23 S1 IVA1; 2019 text IVA"),
}

# "RTX" in ChennaBJD25 / ChennaJACI25 is rituximab by the table footnote. For P303 the
# FrontImm21 case report lists radiotherapy and no rituximab, so P303's RTX is radiotherapy.
AGENT_SWAP = {"D1__P303_skin": {"rituximab": "radiotherapy (local)"}}

# D13__MF21: LCT was CD30+ clinically, but this biopsy was mostly CD30-negative.
CD30_OVERRIDE = {"D13__MF21": "neg"}

# ---------------------------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------------------------

DX = ["MF", "SS", "MF/SS", "MF_or_SS_unresolved", "eCTCL_unresolved", "pcAECTCL",
      "CD8_aggr_CTCL", "LyP", "SPP", "LPP", "HC"]
GROUP_OF_DX = {"HC": "HC", "SPP": "parapsoriasis", "LPP": "parapsoriasis", "LyP": "CD30_LPD"}
PHENOTYPE = ["CD4", "CD8", "gd", "DN", "unknown", "not_applicable"]
YNU = ["yes", "no", "unknown"]
STAGES = ["I", "IA", "IB", "IIA", "IIB", "III", "IIIA", "IIIB", "IV", "IVA", "IVA1", "IVA2", "IVB"]
EARLY = {"I", "IA", "IB", "IIA"}
LESION = ["patch", "plaque", "papule", "tumor", "erythroderma", "mixed", "nonlesional",
          "cleared_lesion", "LyP_lesion", "healthy_skin", "skin_unspecified", "blood", "LN"]
TIMEPOINT = ["single", "baseline", "pre", "on_treatment", "post_progression", "follow_up",
             "serial"]
OUTCOME = ["DOD", "died_other", "died_unspecified", "alive", "unknown"]
EX_VIVO = ["none", "culture_ctr", "culture_drug"]
INPUT = ["tissue", "PBMC", "CD45_sort", "CD4_sort", "T_sort", "malignant_sort",
         "nonmalignant_sort", "sorted_other"]

# Treatment agents: (agent, class, pattern, consume). Matched case-insensitively in this order.
# A `consume` match blanks its span, so a later, more generic pattern cannot count it again
# ("topical bexarotene gel" is not also systemic bexarotene; "oral corticosteroids" is not also
# "steroid (unspecified)").
AGENTS = [
    ("bexarotene (topical)", "topical_other",
     r"bexarotene gel|topical[- ]*(?:targretin|bexarotene)", True),
    ("systemic steroid", "systemic_steroid",
     r"prednis\w*|methylpred\w*|(?:oral|systemic) (?:gcs|cortico\w*|steroids?)"
     r"|intramuscular kenalog", True),
    ("topical steroid", "topical_steroid",
     r"\btcs\b|topical[^;,]*?(?:gcs|cortico\w*|steroids?)|topical\s*(?:and|\+)\s*"
     r"|clobetasol|dermovate|betamethasone|fluocinolone|triamcinolone|\bgcs\b", True),
    ("steroid (unspecified)", "steroid_unspecified", r"\bsteroids?\b|corticosteroids?", True),
    ("mechlorethamine", "topical_chemo",
     r"nitrogen mustard|mechlor\w*|meclor\w*|chlormethine|valchlor", False),
    ("topical immunomodulator", "topical_other",
     r"\bcni\b|\btci\b|tacrolimus|calcineurin|imiquimod|tazarotene", False),
    ("NB-UVB", "phototherapy", r"nb-?uvb|nv-uvb|narrow ?band|\buvb\b", False),
    ("PUVA", "phototherapy", r"puva|\buva\b", False),
    ("phototherapy (other)", "phototherapy",
     r"light therapy|phototherap\w*\b(?!\s*\()|tanning|natural uv", False),   # not "phototherapy (NB-UVB)"
    ("TSEB", "radiotherapy", r"tseb\w*|tset\w*|\btse\b|electron beam|\bleb\b", True),
    ("radiotherapy (local)", "radiotherapy",
     r"radiotherap\w*|radiation|\brt\b|\bisrt\b|x ?ray|\bxrt\b|\bebrt\b|chemorad", False),
    ("bexarotene", "retinoid", r"bexarot\w*|targretin|\bbex\b|bexarone", False),
    ("acitretin", "retinoid", r"acitretin|soriatane|\bacit\b", False),
    ("retinoid (unspecified)", "retinoid", r"\bretinoids?\b", False),
    ("interferon", "interferon", r"\bifn|interferon", False),
    ("methotrexate", "methotrexate", r"methotrexate|\bmtx\b", False),
    ("vorinostat", "hdaci", r"vorinostat|\bvrs\b", False),
    ("romidepsin", "hdaci", r"romi?d[ep]+sin|romedepsin|romidpesin", False),
    ("resminostat", "hdaci", r"resminostat", False),
    ("ECP", "ecp", r"\becp\b|photophe?ore?sis|photopheresis", False),
    ("mogamulizumab", "mogamulizumab", r"mogamuli\w*|mogalizumab|mogamolzumab|\bmoga\b|kw-?0761", False),
    ("brentuximab", "brentuximab", r"brentuximab|bretuximab|\bbv\b|a\+avd", False),
    ("alemtuzumab", "alemtuzumab", r"alemtu\w*", False),
    ("pralatrexate", "chemotherapy", r"pralatrexate", False),
    ("gemcitabine", "chemotherapy", r"gemcitabine", False),
    ("doxorubicin", "chemotherapy", r"doxorubicin|doxil|adriamycin|a\+avd", False),
    ("chlorambucil", "chemotherapy", r"chlorambucil", False),
    ("chemotherapy (other)", "chemotherapy",
     r"\bchop\b|r-chop|cisplatin|cytarabine|decitabine|azacit\w*|taxotere|docetaxel"
     r"|cyclophosphamide|vinblastine|dacarbazine|\bice\b|chemotherap\w*|chemorad", False),
    ("checkpoint inhibitor", "checkpoint", r"pembrolizumab|nivolumab", False),
    ("anti-CD47", "other_targeted", r"tti-?62|cd47|trillium|magrolimab", False),
    ("targeted/trial (other)", "other_targeted",
     r"bispecific|afm13|ags67e|duvelisib|bortezomib|ontak|thalidomide|lenalidomide|iph4102"
     r"|lacutamab", False),
    ("dupilumab", "dupilumab", r"dupilumab|dupixent", False),
    ("alloSCT", "alloSCT", r"transplant|allosct", False),
    ("rituximab", "other_immunomod", r"rituximab|\brtx\b", False),
    ("immunomodulator (other)", "other_immunomod",
     r"infliximab|etanercept|adalimumab|ustekinumab|cyclospor\w*|otezla|apremilast", False),
]
CLASS_OF = {a: c for a, c, _, _ in AGENTS} | {"radiotherapy (local)": "radiotherapy"}
SKIN_DIRECTED = {"topical_steroid", "topical_chemo", "topical_other", "phototherapy",
                 "radiotherapy"}
UNSPECIFIED = {"steroid_unspecified"}          # neither skin-directed nor systemic: "Steroid"

# Columns written into the built objects. The raw free text stays in the CSV only.
OBS_COLS = [
    "clin_person_key", "clin_group", "clin_dx", "clin_phenotype", "clin_cd30", "clin_lct",
    "clin_fmf", "clin_stage", "clin_stage_class", "clin_stage_timing", "clin_T", "clin_N",
    "clin_M", "clin_B", "clin_lesion", "clin_sex", "clin_sex_source", "clin_race",
    "clin_prior_tx", "clin_prior_tx_classes", "clin_current_tx", "clin_current_tx_classes",
    "clin_prior_systemic", "clin_current_systemic", "clin_ever_systemic", "clin_tx_naive",
    "clin_timepoint", "clin_response", "clin_response_scope", "clin_overall_response",
    "clin_drug_sensitivity", "clin_course", "clin_outcome", "clin_ex_vivo", "clin_arm",
    "clin_input", "clin_qc_flag", "clin_confidence",
]
OBS_NUM_COLS = ["clin_age", "clin_duration_y", "clin_n_prior_lines", "clin_days_from_first",
                "clin_blood_tumor_pct"]


# ---------------------------------------------------------------------------------------------
# Field parsers: raw text -> vocabulary
# ---------------------------------------------------------------------------------------------

def _has(s, pat):
    return re.search(pat, s, flags=re.I) is not None


def _dx(raw, sub):
    d, s = raw.strip().lower(), sub.lower()
    if d in ("healthy control", "healthy", "hc"):
        return "HC"
    if "parapsoriasis" in d:
        return "SPP" if "small" in d else "LPP"
    if d.startswith("lyp"):
        return "LyP"
    if "pcaectcl" in d or "berti" in d:
        return "pcAECTCL"
    if d.startswith("erythrodermic ctcl"):
        return "eCTCL_unresolved"
    if d.startswith("mf/ss (not resolved"):
        return "MF_or_SS_unresolved"
    if d == "mf/ss":
        return "MF/SS"
    if d == "ctcl" and "cd8" in s:
        return "CD8_aggr_CTCL"
    if d == "ss":
        return "SS"
    if d.startswith("mf") or "folliculotropic mf" in d:
        return "MF"
    raise ValueError(f"unmapped diagnosis {raw!r}")


def _phenotype(raw, dx):
    if dx in ("HC", "SPP", "LPP"):
        return "not_applicable"
    p = raw.strip()
    if not p or p.startswith("CD30- (CD4/CD8 NR)"):
        return "unknown"
    if p.startswith(("γδ", "TCR-γδ")):
        return "gd"
    if p.startswith("CD4-CD8-"):
        return "DN"
    if p.startswith("CD8"):
        return "CD8"
    if _has(p, r"CD4\+|^CD4\b|CD3\S*\+? ?CD4"):    # CD4-led, incl. "with a DN subset"
        return "CD4"
    raise ValueError(f"unmapped phenotype {raw!r}")


def _cd30(sid, pheno, sub):
    if sid in CD30_OVERRIDE:
        return CD30_OVERRIDE[sid]
    t = f"{pheno} {sub}"
    if _has(t, r"CD30\+|CD30 positiv|focal CD30\+"):
        return "pos"
    if _has(t, r"CD30-|CD30-neg|CD30 <\d|CD30 negative"):
        return "neg"
    return "unknown"


def _lct(raw, dx_raw):
    r = raw.strip().lower()
    if "with lct" in dx_raw.lower() or r.startswith(("+", "yes", "present", "impending")):
        return "yes"
    if r.startswith(("absent", "no")):
        return "no"
    return "unknown"                                   # incl. "possible"


def _fmf(dx_raw, sub):
    t = f"{dx_raw} {sub}".lower()
    return "yes" if "folliculotropic" in t and "?" not in t else "unknown"


_STAGE_RE = re.compile(r"\b(IV[AB][12]?|IV|III[AB]?|II[AB]|I[AB]?)\b")


def _stage(sid, raw):
    if sid in STAGE_OVERRIDE:
        return STAGE_OVERRIDE[sid][0]
    toks = _STAGE_RE.findall(raw)
    if not toks:
        return ""
    if len(toks) > 1 and all(t.startswith(toks[0]) for t in toks):   # "IVA (IVA1 in S1)"
        return max(toks, key=len)
    return toks[0]


def _stage_timing(raw):
    r = raw.lower()
    if "at dx" in r:
        return "dx"
    if "worst" in r:
        return "worst"
    return "sampling" if raw.strip() else ""


def _tnmb(tnmb, blood):
    seg = re.split(r"\s[/(]|;", tnmb)[0]
    if seg.lower().startswith("dx "):                   # harro: "dx T2N0M0B2; coll T0N0M0B1"
        m = re.search(r"coll (\S+)", tnmb)
        seg = m.group(1) if m else seg
    out = {}
    for k, pat in (("T", r"T([0-4][abc]?|x)"), ("N", r"N([0-3][ab]?|x)"),
                   ("M", r"M([01]|x)"), ("B", r"B([0-2][ab]?|x)")):
        m = re.search(pat, seg, flags=re.I)
        out[f"clin_{k}"] = f"{k}{m.group(1).lower()}" if m else ""
    if not out["clin_B"]:
        m = re.match(r"\s*(B[0-2][ab]?)\b", blood)
        out["clin_B"] = m.group(1) if m else ""
    return out


def _lesion(raw, compartment, dx):
    s = raw.lower()
    if dx == "HC":
        return "blood" if compartment == "Blood" else "healthy_skin"
    if s.startswith(("pb", "peripheral blood", "blood")) or compartment == "Blood":
        return "blood"
    if "lymph node" in s or compartment == "LN":
        return "LN"
    if "nonlesional" in s:
        return "nonlesional"
    if "cleared" in s:
        return "cleared_lesion"
    if "lyp lesion" in s:
        return "LyP_lesion"
    if "or tumor" in s or "pp + tt" in s or s.startswith("patch/plaque/tumor"):
        return "mixed"
    hits = [(m.start(), k) for k, pat in (("erythroderma", r"erythrod"), ("tumor", r"tumou?r"),
                                          ("plaque", r"plaque"), ("patch", r"patch"),
                                          ("papule", r"papule"))
            for m in [re.search(pat, s)] if m]
    return min(hits)[1] if hits else "skin_unspecified"


def _num(raw):
    m = re.search(r"\d+(?:\.\d+)?", raw)
    return float(m.group()) if m else np.nan


def _duration(raw):
    s = raw.lower()
    if re.search(r"<\s*1 y", s):
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:y\b|years?)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*mo\b", s)
    return float(m.group(1)) / 12 if m else np.nan


def _blood_tumor_pct(raw):
    """% malignant (or Sezary-phenotype) of blood CD4 T cells, where the source states it."""
    for pat in (r"(\d+(?:\.\d+)?)% of CD4", r"% Sezary of CD4: (\d+(?:\.\d+)?)",
                r"CD4\+CD26- (\d+(?:\.\d+)?)%"):
        m = re.search(pat, raw)
        if m:
            return float(m.group(1))
    return np.nan


def _tx_text(raw):
    """Regimen text with cohort-level statements, bracketed notes and ex vivo arms removed."""
    s = re.sub(r"\[[^\]]*\]", "", raw)
    if _has(s, r"^cohort|cohort-level\)?$|ex vivo|explant|culture|^not extracted"):
        return ""
    if _has(s.strip(), r"^none\b"):                   # "none yet (pre-BV)", "none (...)"
        return ""
    return s


def _agents(sid, raw):
    s = _tx_text(raw)
    found = []
    for name, _, pat, consume in AGENTS:
        if re.search(pat, s, flags=re.I):
            found.append(name)
            if consume:
                s = re.sub(pat, " ", s, flags=re.I)
    swap = AGENT_SWAP.get(sid, {})
    found = [swap.get(a, a) for a in found]
    return sorted(set(found))


def _classes(agents):
    return sorted({CLASS_OF[a] for a in agents})


def _is_none(raw):
    return _has(raw.strip(), r"^(none|treatment-naive)\b")


def _timepoint(arm):
    a = arm.lower()
    if _has(a, r"pre-treatment|pre-bv|^bv: pre"):
        return "pre"
    if "after progression" in a:
        return "post_progression"
    if _has(a, r"response after|non-response|on-treatment|\(on |on vorinostat"):
        return "on_treatment"
    if _has(a, r"follow-?up"):
        return "follow_up"
    if "baseline" in a:
        return "baseline"
    if _has(a, r"draw \d+/\d+|timepoint \(|first timepoint|second timepoint"):
        return "serial"
    return "single"


def _days(arm):
    m = re.search(r"\+(\d+) d", arm)
    if m:
        return float(m.group(1))
    return 0.0 if "day 0" in arm.lower() else np.nan


def _response(resp, arm):
    t = f"{resp} {arm}"
    if _has(resp, r"mixed response"):
        return "mixed", "patient"
    if _has(t, r"lesion NR|non-?respons"):
        return "NR", "lesion"
    if _has(t, r"lesion R\b|response after"):
        return "R", "lesion"
    if _has(resp, r"clear"):
        return "cleared", "lesion"
    if _has(resp, r"stable disease"):
        return "SD", "patient"
    if _has(resp, r"\bPD\b|progress"):
        return "PD", "patient"
    return "", ""


def _overall_response(resp):
    m = re.search(r"overall (CR|PR|SD|PD)", resp)
    return m.group(1) if m else ""


def _sensitivity(resp):
    m = re.search(r"(romidepsin) (SENSITIVE|RESISTANT)", resp)
    return f"{m.group(1)}_{m.group(2).lower()}" if m else ""


def _course(sub, resp):
    t = f"{sub} {resp}".lower()
    if "aggressive" in t:
        return "aggressive"
    if "indolent" in t:
        return "indolent"
    return ""


def _outcome(follow, resp):
    t = f"{follow} {resp}".lower()
    if _has(t, r"died of (disease|ctcl)|disease-specific death"):
        return "DOD"
    if "died of unrelated" in t:
        return "died_other"
    if _has(t, r"deceased|died \(cause"):
        return "died_unspecified"
    if "alive" in t:
        return "alive"
    return "unknown"


def _input(lesion, pheno, arm, compartment):
    t = f"{lesion} {pheno} {arm}"
    if "NONMALIGNANT sort" in t:
        return "nonmalignant_sort"
    if "malignant sort" in t:
        return "malignant_sort"
    if _has(t, r"purified CD4|sorted CD4\+CD26|live CD4\+ sort"):
        return "CD4_sort"
    if "CD45+" in t:
        return "CD45_sort"
    if "T-cell-sorted" in t:
        return "T_sort"
    if "FACS-sorted" in t:
        return "sorted_other"
    return "PBMC" if compartment == "Blood" else "tissue"


def _race(raw):
    r = raw.strip()
    return {"Caucasian": "White", "African-American": "Black"}.get(r, r)


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Base extraction rows, then the hand-supplied overlay field by field."""
    parts = [pd.read_csv(RAW_DIR / f, sep="\t", dtype=str, keep_default_na=False)
             .assign(raw_file=f) for f in RAW_FILES]
    raw = pd.concat(parts, ignore_index=True)
    assert not raw.sample_id.duplicated().any(), raw.sample_id[raw.sample_id.duplicated()]
    raw = raw.set_index("sample_id")
    if SUPPLEMENTS.exists():
        sup = pd.read_csv(SUPPLEMENTS, sep="\t", dtype=str, keep_default_na=False)
        missing = set(sup.sample_id) - set(raw.index)
        assert not missing, f"supplement rows with no base row: {sorted(missing)}"
        for c in sup.columns.difference(raw.columns).drop("sample_id"):
            raw[c] = ""
        for _, s in sup.iterrows():
            for c, v in s.items():
                if c in ("sample_id", "source", "confidence") or v == "":
                    continue
                raw.at[s.sample_id, c] = v
            raw.at[s.sample_id, "source"] = f"{raw.at[s.sample_id, 'source']}; {s.source}"
            if s.confidence == "exact" or raw.at[s.sample_id, "confidence"] == "geo":
                raw.at[s.sample_id, "confidence"] = s.confidence
    return raw.reset_index()


def harmonize(raw: pd.DataFrame, sample_sheet: pd.DataFrame) -> pd.DataFrame:
    """raw extraction rows + atlas sample sheet (sample_id, study, compartment, patient_key)."""
    sheet = sample_sheet.set_index("sample_id")
    raw = raw[raw.sample_id.isin(sheet.index)].copy()      # D13__SZ30 lost the dedup to B4
    rows = []
    for _, r in raw.iterrows():
        sid = r.sample_id
        comp = sheet.at[sid, "compartment"]
        dx = _dx(r.diagnosis, r.subtype_detail)
        pk = sheet.at[sid, "patient_key"]
        prior, now = _agents(sid, r.prior_tx), _agents(sid, r.tx_at_sampling)
        prior_cls, now_cls = _classes(prior), _classes(now)
        prior_none = _is_none(r.prior_tx)
        now_none = _is_none(r.tx_at_sampling)
        lines = re.search(r"(\d+) tx lines", r.prior_tx)
        failed = _has(r.prior_tx, r"failed prior systemic")
        systemic = lambda cls: set(cls) - SKIN_DIRECTED - UNSPECIFIED       # noqa: E731
        if systemic(prior_cls) or failed or (lines and int(lines.group(1))):
            prior_sys = "yes"
        elif (prior and not set(prior_cls) <= UNSPECIFIED) or prior_none:
            prior_sys = "no"
        else:
            prior_sys = "unknown"
        if systemic(now_cls):
            cur_sys = "yes"
        elif (now and not set(now_cls) <= UNSPECIFIED) or now_none:
            cur_sys = "no"
        else:
            cur_sys = "unknown"
        ever_sys = ("yes" if "yes" in (prior_sys, cur_sys) else
                    "no" if (prior_sys, cur_sys) == ("no", "no") else "unknown")
        if _has(r.prior_tx, r"no anti-lymphoma therapy >=6 mo"):
            naive = "washout_6mo"
        elif prior or now or failed or prior_sys == "yes":
            naive = "no"
        elif prior_none and (now_none or not r.tx_at_sampling.strip()):
            naive = "yes"
        else:
            naive = "unknown"
        resp, scope = _response(r.response_outcome, r.arm_timepoint)
        ph = _phenotype(r.malignant_phenotype, dx)
        na = ph == "not_applicable"
        no_stage = dx in ("HC", "SPP", "LPP", "LyP")
        rec = dict(
            sample_id=sid, study=sheet.at[sid, "study"], compartment=comp,
            patient_key=pk, clin_person_key=PERSON_MERGE.get(pk, pk),
            clin_group=GROUP_OF_DX.get(dx, "CTCL"), clin_dx=dx,
            clin_phenotype=ph,
            clin_cd30="not_applicable" if na else _cd30(sid, r.malignant_phenotype, r.subtype_detail),
            clin_lct="not_applicable" if na else _lct(r.lct, r.diagnosis),
            clin_fmf="not_applicable" if na else _fmf(r.diagnosis, r.subtype_detail),
            clin_stage="" if no_stage else _stage(sid, r.stage),
            clin_stage_timing="" if no_stage else _stage_timing(r.stage),
            **_tnmb(r.tnmb, r.blood_inv),
            clin_lesion=_lesion(r.lesion, comp, dx),
            clin_age=_num(r.age), clin_age_at_dx="(at dx)" in r.age,
            clin_sex=r.sex.strip() if r.sex.strip() in ("M", "F") else "",
            clin_sex_source="paper" if r.sex.strip() in ("M", "F") else "",
            clin_race=_race(r.race),
            clin_duration_y=_duration(r.disease_duration),
            clin_blood_tumor_pct=_blood_tumor_pct(r.blood_inv),
            clin_prior_tx=";".join(prior), clin_prior_tx_classes=";".join(prior_cls),
            clin_current_tx=";".join(now), clin_current_tx_classes=";".join(now_cls),
            clin_n_prior_lines=float(lines.group(1)) if lines else np.nan,
            clin_prior_systemic=prior_sys, clin_current_systemic=cur_sys,
            clin_ever_systemic=ever_sys, clin_tx_naive=naive,
            clin_timepoint=_timepoint(r.arm_timepoint),
            clin_days_from_first=_days(r.arm_timepoint),
            clin_response=resp, clin_response_scope=scope,
            clin_overall_response=_overall_response(r.response_outcome),
            clin_drug_sensitivity=_sensitivity(r.response_outcome),
            clin_course=_course(r.subtype_detail, r.response_outcome),
            clin_outcome=_outcome(r.followup, r.response_outcome),
            clin_ex_vivo="culture_ctr" if _has(r.arm_timepoint + r.tx_at_sampling,
                                               r"explant|2-day culture") else "none",
            clin_arm="ctr" if _has(r.arm_timepoint + r.tx_at_sampling,
                                   r"explant|2-day culture") else "none",
            clin_input=_input(r.lesion, r.malignant_phenotype, r.arm_timepoint, comp),
            clin_qc_flag="",
            clin_confidence=r.confidence, clin_source=r.source,
            clin_stage_note=STAGE_OVERRIDE.get(sid, ("", ""))[1],
        )
        rec["clin_stage_class"] = ("" if not rec["clin_stage"] else
                                   "early" if rec["clin_stage"] in EARLY else "advanced")
        rows.append(rec)
    out = pd.DataFrame(rows)
    raw_cols = [c for c in raw.columns if c not in ("sample_id", "donor", "source", "confidence")]
    return out.merge(raw[["sample_id"] + raw_cols].add_prefix("raw_")
                     .rename(columns={"raw_sample_id": "sample_id"}), on="sample_id")


def validate(clin: pd.DataFrame) -> pd.DataFrame:
    """Assert vocabularies and per-person sex; return per-person stage changes (allowed)."""
    vocab = {"clin_dx": DX, "clin_phenotype": PHENOTYPE, "clin_stage": STAGES + [""],
             "clin_lesion": LESION, "clin_timepoint": TIMEPOINT, "clin_outcome": OUTCOME,
             "clin_ex_vivo": EX_VIVO, "clin_input": INPUT,
             "clin_lct": YNU + ["not_applicable"], "clin_fmf": YNU + ["not_applicable"]}
    for c, v in vocab.items():
        bad = set(clin[c]) - set(v)
        assert not bad, (c, bad)
    for c in ("clin_prior_tx", "clin_current_tx"):
        bad = {a for s in clin[c] for a in s.split(";") if a} - set(CLASS_OF)
        assert not bad, (c, bad)
    hc = clin[clin.clin_dx == "HC"]
    assert (hc.clin_stage == "").all() and (hc.clin_phenotype == "not_applicable").all()
    k = clin[clin.clin_sex != ""].groupby("clin_person_key").clin_sex.nunique()
    assert (k <= 1).all(), ("clin_sex", k[k > 1].index.tolist())
    st = clin[clin.clin_stage != ""].groupby("clin_person_key")
    multi = st.clin_stage.nunique()
    return (clin[clin.clin_person_key.isin(multi[multi > 1].index) & (clin.clin_stage != "")]
            [["clin_person_key", "sample_id", "clin_stage", "clin_timepoint", "clin_source"]]
            .sort_values(["clin_person_key", "sample_id"]))


# ---------------------------------------------------------------------------------------------
# Use
# ---------------------------------------------------------------------------------------------

def load_clinical(path=CLIN_CSV) -> pd.DataFrame:
    c = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in OBS_NUM_COLS:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    return c


def load_b4_arms() -> pd.DataFrame:
    return pd.read_csv(B4_ARMS, sep="\t", dtype=str)


def attach_clinical(obs: pd.DataFrame, cols=None, clin=None) -> pd.DataFrame:
    """obs + clin_* columns, joined on sample_id.

    geskin2026 (B4) lanes are hashtag-multiplexed over ex vivo drug arms of one donor, so its
    `clin_arm` / `clin_ex_vivo` are set per cell from the HTO in `real_donor`
    (``PF46_HTO5`` -> lane PF46, HTO5 -> ctr).
    """
    cols = cols or OBS_COLS + OBS_NUM_COLS
    clin = load_clinical() if clin is None else clin
    c = clin.set_index("sample_id")[cols]
    sid = obs["sample_id"].astype(str)
    out = obs.copy()
    for col in cols:
        out[col] = sid.map(c[col]).to_numpy()
    if "real_donor" in obs:
        b4 = sid.str.startswith("B4__").to_numpy()
        if b4.any():
            arms = load_b4_arms()
            key = arms.sample_id + "|" + arms.hto
            hto = obs["real_donor"].astype(str).str.rsplit("_", n=1).str[-1]
            k = (sid + "|" + hto)[b4]
            out.loc[b4, "clin_arm"] = k.map(dict(zip(key, arms.arm))).fillna("unknown").to_numpy()
            out.loc[b4, "clin_ex_vivo"] = k.map(dict(zip(key, arms.ex_vivo))).fillna(
                "unknown").to_numpy()
    return out


# ---------------------------------------------------------------------------------------------
# Propagation into the built objects (obs-only; same mechanics as atlas_v2_fixups)
# ---------------------------------------------------------------------------------------------

def _put_cat(g, col, values):
    """Create or replace a categorical obs column. '' is written as missing (code -1)."""
    values = np.asarray(values, dtype=object)
    cats = sorted({v for v in values if v != ""})
    idx = {c: i for i, c in enumerate(cats)}
    dtype = np.int8 if len(cats) < 127 else np.int16
    codes = np.fromiter((idx.get(v, -1) for v in values), dtype=dtype, count=len(values))
    if col in g:
        del g[col]
    grp = g.create_group(col)
    grp.attrs.update({"encoding-type": "categorical", "encoding-version": "0.2.0",
                      "ordered": False})
    ds = grp.create_dataset("categories", data=np.asarray(cats, dtype=object),
                            dtype=h5py.string_dtype(encoding="utf-8"))
    ds.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})
    grp.create_dataset("codes", data=codes)


def _put_num(g, col, values):
    if col in g:
        del g[col]
    ds = g.create_dataset(col, data=np.asarray(values, dtype=np.float64))
    ds.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})


def _register(g, cols):
    """Append new columns to column-order and drop clin_* columns that are no longer written."""
    keep = set(cols)
    order = [str(c) for c in g.attrs["column-order"]]
    new = [c for c in order if not c.startswith("clin_") or c in keep]
    new += [c for c in cols if c not in new]
    if new != order:
        g.attrs["column-order"] = np.asarray(new, dtype=object)


def _stale(g):
    keep = set(OBS_COLS + OBS_NUM_COLS)
    return [c for c in g.keys() if c.startswith("clin_") and c not in keep]


def _read_str(g, col):
    o = g[col]
    if isinstance(o, h5py.Group):
        cats = np.asarray(o["categories"][:]).astype(str)
        codes = o["codes"][:]
        if not len(cats):                                  # column entirely missing
            return np.full(len(codes), "", dtype=object).astype(str)
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
    return np.asarray(o[:]).astype(str)


def real_donor_by_cell(parquet=OBS_PARQUET) -> pd.Series:
    """cell_id -> real_donor for B4 cells, for objects whose obs dropped real_donor."""
    t = pd.read_parquet(parquet, columns=["cell_id", "sample_id", "real_donor"],
                        filters=[("study", "=", "geskin2026")])
    return t.set_index("cell_id")["real_donor"].astype(str)


def patch_h5ad_obs_clinical(path, clin, rd_by_cell=None, dry_run=True, verbose=True) -> dict:
    """Write OBS_COLS + OBS_NUM_COLS into one h5ad's obs, dropping retired clin_* columns.

    Idempotent; returns a per-column diff (a dropped column counts as one entry).
    """
    path = Path(path)
    diff = {}
    with h5py.File(path, "r" if dry_run else "a") as f:
        g = f["obs"]
        if "sample_id" not in g:
            if verbose:
                print(f"  {path.name}: no sample_id — skipped")
            return diff
        obs = pd.DataFrame({"sample_id": _read_str(g, "sample_id")})
        if "real_donor" in g:
            obs["real_donor"] = _read_str(g, "real_donor")
        elif rd_by_cell is not None:
            cid = pd.Series(_read_str(g, g.attrs["_index"]))
            obs["real_donor"] = cid.map(rd_by_cell).fillna("").to_numpy()
        new = attach_clinical(obs, clin=clin)
        for col in OBS_COLS + OBS_NUM_COLS:
            v = new[col]
            if col in OBS_NUM_COLS:
                v = pd.to_numeric(v, errors="coerce").to_numpy(dtype=np.float64)
                same = col in g and np.array_equal(np.asarray(g[col][:]), v, equal_nan=True)
            else:
                v = v.fillna("").astype(str).to_numpy()
                same = col in g and np.array_equal(_read_str(g, col), v)
            if same:
                continue
            diff[col] = int(len(v))
            if not dry_run:
                (_put_num if col in OBS_NUM_COLS else _put_cat)(g, col, v)
        for col in _stale(g):
            diff[f"-{col}"] = 1
            if not dry_run:
                del g[col]
        if not dry_run and diff:
            _register(g, OBS_COLS + OBS_NUM_COLS)
    if verbose:
        tag = "would write" if dry_run else "wrote"
        drops = [c for c in diff if c.startswith("-")]
        print(f"  {path.name}: {tag} " + (f"{len(diff) - len(drops)} clin_* columns"
                                          + (f", dropped {len(drops)} retired" if drops else "")
                                          if diff else "nothing"))
    return diff


def patch_parquet_clinical(path, clin, dry_run=True, verbose=True) -> dict:
    path = Path(path)
    A = pd.read_parquet(path)
    new = attach_clinical(A[["sample_id", "real_donor"]], clin=clin)
    diff = {}
    for col in [c for c in A.columns if c.startswith("clin_")
                and c not in OBS_COLS + OBS_NUM_COLS]:
        diff[f"-{col}"] = 1
        A = A.drop(columns=col)
    for col in OBS_COLS + OBS_NUM_COLS:
        v = new[col]
        v = (pd.to_numeric(v, errors="coerce") if col in OBS_NUM_COLS
             else v.fillna("").astype(str).astype("category"))
        if col in A and A[col].astype(str).equals(v.astype(str)):
            continue
        diff[col] = len(v)
        A[col] = v.to_numpy() if col in OBS_NUM_COLS else v.values
    if verbose:
        tag = "would write" if dry_run else "wrote"
        print(f"  {path.name}: {tag} " + (f"{len(diff)} column changes" if diff else "nothing"))
    if not dry_run and diff:
        tmp = path.with_suffix(".tmp.parquet")
        A.to_parquet(tmp)
        tmp.replace(path)
    return diff
