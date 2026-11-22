# 10_atlas/12_atlas_descriptive.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/10_atlas/12_atlas_descriptive.ipynb` and is never run._

# 12 · CTCL / MF atlas **v2** — descriptive overview

What the atlas is, and what it looks like. Two questions only:

1. **What is in it** — cells, samples, patients, clinical metadata, origin (§2).
2. **What does it look like** — five UMAPs on the MrVI *u* embedding (cell type, study,
   stage, disease, compartment), drawn from a random 300k subsample because 2.16 M points do
   not plot (§3).

**Atlas v2** (rebuilt 2026-09-05, `docs/ATLAS_V2_DEDUP_GATE.md` §4c):
**2,157,693 cells × 42,347 genes · 263 samples · 268 donor-units · 201 patients · 21 studies**,
Skin 1,345,527 · Blood 811,024 · LN 1,142. Roughly double v1 (1,173,694 / 149 / 116 / 10) after
290,112 duplicate cells were dropped across 45 samples.

- **Working object**: obs only. `joint_mrvi_input.h5ad` (2,157,693 × 10,000 HVG) is read once for
  its `obs` group and cached to `atlas_obs_full_v2.parquet`; `X` is never touched here.
- **Embedding**: `joint_X_mrvi_u.npy` — the 10-d sample-**unaware** MrVI latent, so structure is
  biology rather than cohort.
- **Labels**: `skin_cell_type_final.csv` (`21_reannotation`, v2, 12 levels) and, once `31_reannotation` has been re-run on
  v2 blood, `blood_cell_type_final.csv`. Blood is **currently unannotated** — the v1 sidecar
  covers 423,042 of 811,024 blood cells and is rejected by the shape gate in §1.

Malignancy calls, composition statistics, the marker dotplot, the T-cell zoom and MRVI
differential abundance all lived here in v1 and have been removed — they belong to `23_malignancy_tcr_cnv` / `24_subclone_tcr_signaling` /
`26_transcriptome_malignancy_holdout`.

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl

SEED = 0
np.random.seed(SEED); sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
OUT = NB_DIR / "data" / "atlas_joint"
FIG = NB_DIR / "figures" / "atlas_descriptive"; FIG.mkdir(parents=True, exist_ok=True)
TAB = NB_DIR / "tables"; TAB.mkdir(exist_ok=True)

# --- v2 inputs (all 2026-09-05/06; the v1 latents live in atlas_joint/_stale_v1_latents/) ---
JOINT_FULL   = OUT / "joint_mrvi_input.h5ad"        # 2,157,693 x 10,000 HVG — obs read only
U_NPY        = OUT / "joint_X_mrvi_u.npy"           # 2,157,693 x 10, sample-unaware
SKIN_LABELS  = OUT / "skin_cell_type_final.csv"     # nb10b v2, 1,345,527 rows
BLOOD_LABELS = OUT / "blood_cell_type_final.csv"    # nb33 — still v1 (423,042); gated in §1
SAMPLE_META  = NB_DIR / "data" / "sample_metadata_final.csv"   # curated v2 sheet, 264 x 37

# New parquet name on purpose: nb33/nb35 still read the v1 `atlas_obs_full.parquet`, and swapping
# v2 content under that name is exactly the cache-versioned-by-filename trap that bit nb30 v5.
OBS_FULL = OUT / "atlas_obs_full_v2.parquet"

N_TOTAL, N_SKIN, N_BLOOD, N_LN = 2_157_693, 1_345_527, 811_024, 1_142
# 264 samples, not the 263 the build reported: the release QC split gaydosik2022's HTO-multiplexed
# `D13__SZ29` lane into its blood and skin specimens, because one sample_id spanning two
# compartments makes every sample-level metadata column wrong for one of its halves. See
# atlas_v2_fixups.py; the split is applied to the built objects by jobs/apply_atlas_v2_fixups.py.
N_SAMPLES, N_DONOR_UNITS, N_REAL_DONOR, N_PATIENTS, N_STUDIES = 264, 268, 253, 201, 21
PLOT_N = 300_000                                    # UMAP subsample; 14% of the atlas

for p in [JOINT_FULL, U_NPY, SKIN_LABELS, SAMPLE_META]:
    assert p.exists(), f"missing v2 input: {p}"
print("paths ok")
```

## §1 · Obs-only cache of the v2 atlas

Reads **only** the `obs` group from the 33.8 GB h5ad (header region — `X` and `layers` are never
touched), joins the cell-type sidecars, derives the clinical columns, and caches to parquet so
every table below reloads in seconds. Delete `atlas_obs_full_v2.parquet` to rebuild.

⚠️ **Run this on the GPU kernel, not the login node.** It is 2.16 M × 58 even without expression.

### The v2 metadata is not harmonized — read this before any breakdown table

The 11 cohorts added in v2 wrote their own vocabularies into shared columns, so the raw levels
double-count:

| column | the problem |
|---|---|
| `sex` | `u` and `unknown` are the same thing |
| `cohort_region` | `North American` **and** `North_American` |
| `entity` (13) | `MF`/`MF_classic`, `SS`/`Sezary`, `healthy`/`healthy_control` are three pairs |
| `disease_stage` (15) | mixes stages with `HC`, `NA`, `nan`, `advanced`, `other`. **`stage_clean` (12) is the clean axis** and adds `IIIA`/`IVA`/`IVA1` that v1's stage logic never handled |
| `treatment_context` (28) | 25 levels are raw free-text regimens (`"ECP; BEX; IFN alpha"`, `"Non-response after 6 cycles"`, `"Pre-treatent"` [sic]) |
| `lesion_type` (9) | `lesional/nonlesional/tumor/patch` alongside v1's `patch(thin)/plaque/tumor(thick)` |
| `tissue_detail` (19) | free text; **`organ` (3) is the clean axis** |
| `blood_involvement` | gained `B2_yes_by_def` |
| label sidecars | skin writes `Unk`, blood writes `UNK` |

The cell below keeps every raw column and adds harmonized `*_h` / `*_group` companions. **Every
table in §2 reports the harmonized column.** There is **no `age`** anywhere in v2 — not in obs,
not in the curated sample sheet — so "clinical" here means disease / stage / entity / lesion /
treatment / blood involvement / sex / country.

```python  # cell 3
# ============================================================================
# §1 · obs-only cache of the FULL v2 atlas (skin + blood + LN)
# ============================================================================
import time, h5py

try:                                       # anndata >= 0.10
    from anndata.io import read_elem
except ImportError:                        # older anndata
    from anndata.experimental import read_elem

# --- cell-type vocabularies -------------------------------------------------
# nb10b v2 skin (12 levels, `CD4_Treg` not `Tregs`, `Unk` not `UNK`) ...
CT_SKIN = ["CD4", "CD8", "CD4_Treg", "B", "Plasma", "Myeloid", "Mast",
           "Keratinocyte", "Fibroblast", "Vascular", "Melanocyte", "Unk"]
# ... plus the levels nb33 will contribute once it is re-run on v2 (nb33 §17). Listing them now
# means this notebook needs no edit when the blood sidecar lands.
# nb33 uses 10b's spellings (CD4_Treg, Unk, Myeloid) so every shared level is already in
# CT_SKIN; these are the blood-only levels it can additionally emit.
CT_BLOOD = ["gdT", "MAIT", "NK", "Mono_CD14", "Mono_CD16", "cDC", "pDC",
            "Platelet", "Erythroid", "HSPC", "Prolif", "LowQC", "Doublet"]

# li2024's own 50-level taxonomy -> the harmonized vocabulary, used only where a reannotation
# label is absent. `tumor_cell` and NK/ILC stay separate: folding them into CD4/Unk would
# silently inflate those levels.
CT50_TO_H = {
    "Th": "CD4", "Tc": "CD8", "Tc17_Th17": "CD8", "Tc_IL13_IL22": "CD8", "Treg": "CD4_Treg",
    "tumor_cell": "tumor_cell(atlas)",
    "NK": "NK_ILC", "ILC1_NK": "NK_ILC", "ILC1_3": "NK_ILC", "ILC2": "NK_ILC",
    "B_cell": "B", "Plasma": "Plasma", "Mast_cell": "Mast", "Melanocyte": "Melanocyte",
    "DC1": "Myeloid", "DC2": "Myeloid", "pDC": "Myeloid", "MigDC": "Myeloid",
    "Inf_mac": "Myeloid", "Macro_1": "Myeloid", "Macro_2": "Myeloid", "Mono_mac": "Myeloid",
    "LC_1": "Myeloid", "LC_2": "Myeloid", "LC_3": "Myeloid", "LC_4": "Myeloid",
    "moDC_1": "Myeloid", "moDC_2": "Myeloid", "moDC_3": "Myeloid",
    "Basal": "Keratinocyte", "basal2": "Keratinocyte", "Differentiated_KC": "Keratinocyte",
    "Differentiated_KC*": "Keratinocyte", "Proliferating_KC": "Keratinocyte",
    "Undifferentiated_KC": "Keratinocyte", "Sebaceous": "Keratinocyte",
    "F1": "Fibroblast", "F2": "Fibroblast", "F3": "Fibroblast",
    "VE1": "Vascular", "VE2": "Vascular", "VE3": "Vascular", "LE1": "Vascular",
    "LE2": "Vascular", "Pericyte_1": "Vascular", "Pericyte_2": "Vascular",
    "Schwann_1": "other", "channel": "other", "immune": "Unk", "Unknown": "unannotated",
}
CT_H_ORDER = CT_SKIN + CT_BLOOD + ["tumor_cell(atlas)", "NK_ILC", "other", "unannotated"]


# --- harmonization maps for the v2 vocabulary drift -------------------------
EARLY = {"IA", "IB", "IIA"}
ADVANCED = {"IIB", "IIIA", "IIIB", "IV", "IVA", "IVA1", "IVA2", "IVB"}
ENTITY_H = {"MF": "MF_classic", "SS": "Sezary", "healthy": "healthy_control"}
LEUKEMIC_ENT = {"Sezary", "SS", "MF/SS_leukemic", "erythrodermic_CTCL(eMF/SS)"}


def _treatment_group(v: str) -> str:
    """Collapse the 28 free-text `treatment_context` levels to 8. Order of tests matters:
    a regimen naming both a systemic drug and PUVA is scored as systemic."""
    s = str(v).strip().lower()
    if s in {"none", "none/unspecified", ""} or s == "nan":
        return "none/unspecified"
    if s == "unknown":
        return "unknown"
    if "ex_vivo" in s:
        return "ex_vivo"
    if "cycles" in s or s.startswith("pre-treat"):          # trial timepoints, not regimens
        return "trial_course"
    if s.startswith("ecp") or "photopheresis" in s:
        return "ECP/photopheresis"
    if any(k in s for k in ["bexarotene", "bex", "brentuximab", "moga", "dupilumab",
                            "alemtuzumab", "pembrolizumab", "vorinostat", "acitretin",
                            "acit", "ifn", "interferon"]):
        return "systemic/targeted"
    if "puva" in s or "uvb" in s:
        return "phototherapy"
    if "gcs" in s:
        return "topical"
    return "other"


LESION_GROUP = {
    "patch": "lesional_thin(patch)", "patch(thin)": "lesional_thin(patch)",
    "plaque/tumor(thick)": "lesional_thick(plaque/tumor)", "tumor": "lesional_thick(plaque/tumor)",
    "lesional": "lesional_unspec", "nonlesional": "nonlesional",
    "follow up": "follow_up", "skin_unspecified": "unspecified", "unknown": "unknown",
}


def _bool(s):
    """h5ad/parquet booleans survive reindex+merge as objects or 'True'/'False' strings."""
    return pd.array(pd.Series(np.asarray(s)).astype(str)
                    .isin(["True", "true", "1"]).to_numpy(), dtype="boolean")


def _build_obs_full() -> pd.DataFrame:
    t0 = time.time()
    with h5py.File(JOINT_FULL, "r") as h:
        A = read_elem(h["obs"])                     # obs only — no X, no layers
    print(f"  obs read: {A.shape[0]:,} rows x {A.shape[1]} cols in {time.time() - t0:.0f}s")
    A.index = pd.Index(A["cell_id"].astype(str), name="cell_id")
    comp = A["compartment"].astype(str)

    # --- cell-type sidecars; blood only joins if it actually covers v2 blood ---
    parts = [pd.read_csv(SKIN_LABELS).set_index("cell_id")[["cell_type_broad", "cell_type_final"]]]
    src = pd.Series("reannotated (nb10b)", index=parts[0].index)
    if BLOOD_LABELS.exists():
        bl = pd.read_csv(BLOOD_LABELS).set_index("cell_id")
        cov = A.index[comp == "Blood"].isin(bl.index).mean()
        if cov > 0.99:
            parts.append(bl[["cell_type_broad", "cell_type_final"]])
            src = pd.concat([src, pd.Series("reannotated (nb33)", index=bl.index)])
            print(f"  blood labels joined: {len(bl):,} rows, {cov:.1%} of v2 blood")
        else:
            print(f"  SKIP {BLOOD_LABELS.name}: {len(bl):,} rows cover only {cov:.1%} of v2's "
                  f"{N_BLOOD:,} blood cells -> blood stays unannotated. "
                  "Re-run 33_blood_reannotation.ipynb against v2.")
    lab = pd.concat(parts)
    assert not lab.index.duplicated().any(), "skin and blood label sidecars overlap"
    for c in ["cell_type_broad", "cell_type_final"]:
        A[c] = (lab[c].reindex(A.index).astype(str)
                .replace({"UNK": "Unk", "nan": np.nan}).to_numpy())   # blood writes UNK

    # --- harmonized cell type + where each label came from ---
    ct50 = A["cell_type"].astype(str).map(CT50_TO_H).fillna("other")
    A["cell_type_h"] = pd.Categorical(A["cell_type_final"].fillna(ct50), categories=CT_H_ORDER)
    assert A["cell_type_h"].isna().sum() == 0, (
        "cell_type_h has levels missing from CT_H_ORDER: "
        f"{sorted(set(A['cell_type_final'].dropna()) | set(ct50)) }")
    A["label_source"] = np.where(
        A["cell_type_final"].notna(), src.reindex(A.index).fillna("reannotated (nb10b)"),
        np.where(A["cell_type"].astype(str).ne("Unknown"),
                 "li2024 taxonomy", "NONE (unannotated)"))

    # --- harmonized clinical columns (raw columns are kept untouched) ---
    A["sex_h"] = A["sex"].astype(str).replace({"u": "unknown"})
    A["region_h"] = A["cohort_region"].astype(str).replace({"North_American": "North American"})
    A["entity_h"] = A["entity"].astype(str).replace(ENTITY_H)
    A["treatment_group"] = A["treatment_context"].map(_treatment_group)
    A["lesion_group"] = A["lesion_type"].astype(str).map(LESION_GROUP).fillna("unknown")

    stg = A["stage_clean"].astype(str)               # stage_clean, NOT the dirty disease_stage
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

    A["has_tcr_b"] = _bool(A["has_tcr"])             # coverage only; malignancy lives in nb30/43

    # --- v2 shape gates ---
    assert len(A) == N_TOTAL, f"expected {N_TOTAL:,} cells, got {len(A):,}"
    assert (comp == "Skin").sum() == N_SKIN, "skin row count changed"
    assert (comp == "Blood").sum() == N_BLOOD, "blood row count changed"
    assert (comp == "LN").sum() == N_LN, "LN row count changed"
    for col, n in [("sample_id", N_SAMPLES), ("donor", N_DONOR_UNITS),
                   ("real_donor", N_REAL_DONOR), ("patient_key", N_PATIENTS),
                   ("study", N_STUDIES)]:
        assert A[col].nunique() == n, f"{col}: expected {n}, got {A[col].nunique()}"
    return A


if OBS_FULL.exists():
    A, _src = pd.read_parquet(OBS_FULL), "cache"
else:
    A = _build_obs_full(); A.to_parquet(OBS_FULL); _src = "built"

print(f"atlas obs [{_src}] -> {OBS_FULL.name} | {len(A):,} cells x {A.shape[1]} cols")
```

```
atlas obs [cache] -> atlas_obs_full_v2.parquet | 2,157,693 cells x 71 cols
```

```python  # cell 4
# --- reading order for every categorical, + the table primitives used by §2 ---
def _order(col, cats):
    """Impose a reading order on a metadata column; unlisted levels go last, sorted."""
    seen = set(A[col].astype(str))
    present = [c for c in cats if c in seen]
    extra = sorted(seen - set(present))
    A[col] = pd.Categorical(A[col].astype(str), categories=present + extra)


_order("compartment", ["Skin", "Blood", "LN"])
_order("organ", ["skin", "blood", "lymph_node"])
_order("disease", ["HC", "MF", "SS", "CTCL_other"])
_order("stage_class", ["HC", "early", "advanced", "other", "unknown"])
_order("stage_group", ["HC", "early", "advanced", "unknown"])
_order("stage_clean", ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV", "IVA", "IVA1",
                       "IVA2", "IVB", "NA"])
_order("entity_h", ["healthy_control", "MF_classic", "MF_CD8", "MF_gamma_delta", "MF_unresolved",
                    "MF/SS_leukemic", "Sezary", "erythrodermic_CTCL(eMF/SS)",
                    "CD8_aggressive_epidermotropic_CTCL", "CTCL_other"])
_order("sex_h", ["F", "M", "unknown"])
_order("region_h", ["European", "North American", "East_Asian", "unknown"])
_order("skin_layer", ["epidermis", "dermis", "whole", "not_skin"])
_order("lesion_group", ["lesional_thin(patch)", "lesional_thick(plaque/tumor)", "lesional_unspec",
                        "nonlesional", "follow_up", "unspecified", "unknown"])
_order("treatment_group", ["none/unspecified", "topical", "phototherapy", "ECP/photopheresis",
                           "systemic/targeted", "trial_course", "ex_vivo", "other", "unknown"])
_order("blood_involvement_eff", ["yes(curated)", "yes(leukemic entity)", "yes(stage IV/B2)",
                                 "likely(stage IIIB/B1)", "stage IV non-blood (IVA2/IVB)",
                                 "no(HC)", "unknown"])
_order("label_source", ["reannotated (nb10b)", "reannotated (nb33)", "li2024 taxonomy",
                        "NONE (unannotated)"])

# short names for plot axes and single-study warnings
SH = {"borcherding2019": "borch19", "borcherding2023": "borch23", "brunner2024": "brunner",
      "buus2025": "buus", "chennareddy2025": "chennareddy", "gaydosik2019": "gaydosik19",
      "gaydosik2022": "gaydosik22", "gaydosik2023": "gaydosik23", "geskin2026": "geskin",
      "herrera2021": "herrera", "li2024": "li", "rindler2021_fi": "rindler_fi",
      "rindler2021_mc": "rindler_mc", "alkon2024": "alkon", "brentuximab2026": "brentux",
      "dorando2026": "dorando", "harro2023": "harro", "il4ra2026": "il4ra",
      "jonak2021": "jonak", "lyp2026": "lyp", "ren2023": "ren",
      "CTCL_other": "other CTCL", "yes(curated)": "yes (curated)",
      "yes(leukemic entity)": "yes (leukemic)", "yes(stage IV/B2)": "yes (st.IV)",
      "likely(stage IIIB/B1)": "likely (IIIB)",
      "stage IV non-blood (IVA2/IVB)": "IVA2/IVB", "no(HC)": "no", "unknown": "unk"}


def show(df, title=None, rows=300):
    if title:
        print(f"\n=== {title} ===")
    with pd.option_context("display.max_rows", rows, "display.max_columns", 80,
                           "display.width", 240, "display.float_format", lambda v: f"{v:,.1f}"):
        print(df.to_string())


def mat(d, index, columns="compartment", what="cells"):
    """index x columns matrix of cells / patients / donors / samples, with TOTAL margins."""
    g = d.groupby([index, columns], observed=True)
    s = {"cells": g.size(), "patients": g["patient_key"].nunique(),
         "donors": g["donor"].nunique(), "samples": g["sample_id"].nunique()}[what]
    m = s.unstack(columns).fillna(0).astype(int)
    m["TOTAL"] = m.sum(axis=1)
    # every nunique measure is recomputed for the margins, never summed: a patient (and in
    # principle a donor or sample) can appear under more than one row level, and summing
    # would count them twice.
    key = {"patients": "patient_key", "donors": "donor", "samples": "sample_id"}.get(what)
    if key:
        tot = d.groupby(columns, observed=True)[key].nunique().reindex(m.columns[:-1]).fillna(0)
        m.loc["TOTAL"] = list(tot.astype(int)) + [d[key].nunique()]
        m["TOTAL"] = list(d.groupby(index, observed=True)[key].nunique()
                          .reindex(m.index[:-1]).fillna(0).astype(int)) + [d[key].nunique()]
    else:
        m.loc["TOTAL"] = m.sum(axis=0)
    return m


def tri(d, index, columns="compartment"):
    """cells + patients + samples for one index x columns pair, side by side."""
    return pd.concat({w: mat(d, index, columns, w) for w in ["cells", "patients", "samples"]},
                     axis=1)


print(A["compartment"].value_counts().to_string())
lab_pct = A["cell_type_final"].notna().mean()
print(f"\ncell-type labelled: {A['cell_type_final'].notna().sum():,} / {len(A):,} ({lab_pct:.1%})")
print(A.groupby("compartment", observed=True)["cell_type_final"]
       .apply(lambda s: f"{s.notna().mean():.1%}").to_string())
```

```
compartment
Skin     1345527
Blood     811024
LN          1142

cell-type labelled: 2,156,551 / 2,157,693 (99.9%)
compartment
Skin     100.0%
Blood    100.0%
LN         0.0%
```

## §2 · What is in the atlas

Order: headline → origin (one row per study) → one row per sample → clinical breakdowns →
cell types and label coverage. Everything is written to `tables/atlas_v2_*.csv`.

Three denominators to keep straight, because v2 has three different "patient" columns:

- **`sample_id` (263)** — one library / biopsy.
- **`donor` (268)** — a donor *unit* as deposited, i.e. `<dataset>__<name>`. Two deposits of the
  same person are two donor-units.
- **`patient_key` (201)** — the cross-deposit patient identity resolved by the v2 dedup gate.
  **This is the real patient count** and is what the tables below report as *patients*.
  `real_donor` (253) sits in between and is not used here.

```python  # cell 6
# --- 2a · headline, one column per compartment -----------------------------
def headline(d):
    return pd.Series({
        "studies": d["study"].nunique(),
        "patients (patient_key)": d["patient_key"].nunique(),
        "donor-units": d["donor"].nunique(),
        "samples": d["sample_id"].nunique(),
        "cells": len(d),
        "median cells/sample": int(d.groupby("sample_id", observed=True).size().median()),
        "% TCR+": 100 * d["has_tcr_b"].fillna(False).mean(),
        "% cell-type labelled": 100 * d["cell_type_final"].notna().mean(),
    })


H = pd.concat({str(k): headline(v) for k, v in A.groupby("compartment", observed=True)}, axis=1)
H["ALL"] = headline(A)          # computed from A, not summed: patients span compartments
show(H, "headline — atlas v2 by compartment")
H.to_csv(TAB / "atlas_v2_headline.csv")

print("\nCross-check vs docs/ATLAS_V2_DEDUP_GATE.md §4c: "
      f"{N_TOTAL:,} cells / {N_SAMPLES} samples / {N_DONOR_UNITS} donor-units / "
      f"{N_PATIENTS} patients / {N_STUDIES} studies "
      f"| Skin {N_SKIN:,} · Blood {N_BLOOD:,} · LN {N_LN:,}")
print("LN is a single sample from rindler2021_fi — reported, never used as a comparison group.")
```

```
=== headline — atlas v2 by compartment ===
                              Skin     Blood      LN         ALL
studies                       14.0      10.0     1.0        21.0
patients (patient_key)       137.0      75.0     1.0       201.0
donor-units                  146.0     131.0     1.0       268.0
samples                      174.0      89.0     1.0       263.0
cells                  1,345,527.0 811,024.0 1,142.0 2,157,693.0
median cells/sample        5,561.0   6,279.0 1,142.0     5,848.0
% TCR+                         9.9      58.8    21.9        28.3
% cell-type labelled         100.0     100.0     0.0        99.9

Cross-check vs docs/ATLAS_V2_DEDUP_GATE.md §4c: 2,157,693 cells / 263 samples / 268 donor-units / 201 patients / 21 studies | Skin 1,345,527 · Blood 811,024 · LN 1,142
LN is a single sample from rindler2021_fi — reported, never used as a comparison group.
```

```python  # cell 7
# --- 2b · origin: one row per study ----------------------------------------
for what in ["samples", "patients", "cells"]:
    show(mat(A, "study", "compartment", what), f"study x compartment — {what}")


def _join(s):
    return ", ".join(sorted(set(s.dropna().astype(str)) - {"nan", "NA", ""}))


roster = (A.groupby("study", observed=True)
          .agg(accession=("accession", _join), repo=("repo", _join), tech=("tech", _join),
               country=("cohort_country", _join), region=("region_h", _join),
               compartments=("compartment", _join), diseases=("disease", _join),
               stages=("stage_clean", _join), entities=("entity_h", _join),
               patients=("patient_key", "nunique"), donor_units=("donor", "nunique"),
               samples=("sample_id", "nunique"), cells=("cell_id", "size"),
               pct_tcr=("has_tcr_b", lambda s: round(100 * s.fillna(False).mean(), 1)),
               labels=("label_source", _join)))
roster = roster.sort_values("cells", ascending=False)
show(roster, f"one row per study — {len(roster)} studies (v1 had 10)")
roster.to_csv(TAB / "atlas_v2_study_roster.csv")

br = A.groupby("study", observed=True)["compartment"].nunique()
print("\nCompartment bridges (same study contributes blood AND skin) — the only within-study "
      "blood/skin comparisons available:\n  " + ", ".join(sorted(br[br > 1].index.astype(str))))
```

```
=== study x compartment — samples ===
compartment      Skin  Blood  LN  TOTAL
study                                  
alkon2024          14      0   0     14
borcherding2019     0      4   0      4
borcherding2023     0      5   0      5
brentuximab2026    13      0   0     13
brunner2024         5      0   0      5
buus2025           20     20   0     40
chennareddy2025    22      0   0     22
dorando2026         0     15   0     15
gaydosik2019        9      0   0      9
gaydosik2022        7      4   0     10
gaydosik2023        5      0   0      5
geskin2026          0     11   0     11
harro2023           0      7   0      7
herrera2021         6      8   0     14
il4ra2026          15      0   0     15
jonak2021           1      0   0      1
li2024             36      0   0     36
lyp2026             8      0   0      8
ren2023             0     14   0     14
rindler2021_fi      0      1   1      2
rindler2021_mc     13      0   0     13
TOTAL             174     89   1    263

=== study x compartment — patients ===
compartment      Skin  Blood  LN  TOTAL
study                                  
alkon2024          14      0   0     14
borcherding2019     0      2   0      2
borcherding2023     0      5   0      5
brentuximab2026     6      0   0      6
brunner2024         5      0   0      5
buus2025            3     17   0     17
chennareddy2025    21      0   0     21
dorando2026         0      6   0      6
gaydosik2019        9      0   0      9
gaydosik2022        7      4   0     10
gaydosik2023        5      0   0      5
geskin2026          0     11   0     11
harro2023           0      7   0      7
herrera2021         6      8   0      8
il4ra2026          15      0   0     15
jonak2021           1      0   0      1
li2024             36      0   0     36
lyp2026             8      0   0      8
ren2023             0     14   0     14
rindler2021_fi      0      1   1      1
rindler2021_mc      7      0   0      7
TOTAL             137     75   1    201

=== study x compartment — cells ===
compartment         Skin   Blood    LN    TOTAL
study                                          
alkon2024         113307       0     0   113307
borcherding2019        0   14253     0    14253
borcherding2023        0   42642     0    42642
brentuximab2026   133677       0     0   133677
brunner2024        37883       0     0    37883
buus2025           55483   72378     0   127861
chennareddy2025   191147       0     0   191147
dorando2026            0   85847     0    85847
gaydosik2019       39260       0     0    39260
gaydosik2022       60483   31141     0    91624
gaydosik2023       20630       0     0    20630
geskin2026             0  260455     0   260455
harro2023              0  130829     0   130829
herrera2021         1870   30132     0    32002
il4ra2026          75834       0     0    75834
jonak2021           4353       0     0     4353
li2024            419579       0     0   419579
lyp2026            87300       0     0    87300
ren2023                0  140165     0   140165
rindler2021_fi         0    3182  1142     4324
rindler2021_mc    104721       0     0   104721
TOTAL            1345527  811024  1142  2157693

=== one row per study — 21 studies (v1 had 10) ===
                    accession          repo              tech         country                                region compartments            diseases                            stages                                                                         entities  patients  donor_units  samples   cells  pct_tcr                                   labels
study                                                                                                                                                                                                                                                                                                                                                            
li2024           E-MTAB-14559  ArrayExpress  10x_5p, 
... [7553 more chars truncated]
```

```python  # cell 8
# --- 2c · one row per sample, joined to the curated v2 sample sheet ---------
roster_s = (A.groupby("sample_id", observed=True)
            .agg(study=("study", "first"), patient_key=("patient_key", "first"),
                 donor=("donor", "first"), compartment=("compartment", "first"),
                 organ=("organ", "first"), tissue=("tissue", "first"),
                 skin_layer=("skin_layer", "first"), disease=("disease", "first"),
                 stage=("stage_clean", "first"), stage_group=("stage_group", "first"),
                 entity=("entity_h", "first"), lesion=("lesion_group", "first"),
                 treatment=("treatment_group", "first"),
                 blood_inv=("blood_involvement_eff", "first"), sex=("sex_h", "first"),
                 tech=("tech", "first"), n_cells=("cell_id", "size"),
                 n_tcr=("has_tcr_b", lambda s: int(s.fillna(False).sum())),
                 n_clones=("clone_id", lambda s: s[s.astype(str).ne("")].nunique()),
                 pct_labelled=("cell_type_final", lambda s: round(100 * s.notna().mean(), 1))))
roster_s.insert(roster_s.columns.get_loc("n_clones"), "pct_tcr",
                (100 * roster_s["n_tcr"] / roster_s["n_cells"]).round(1))

# provenance columns come from the curated sheet rather than being re-derived
SM_COLS = ["accession", "repo", "malignant_call_method", "tcr_available", "tcr_recovery",
           "lineage", "lineage_resolve", "origin_resolve"]
sm = pd.read_csv(SAMPLE_META).set_index("sample_id")
miss = set(roster_s.index.astype(str)) - set(sm.index.astype(str))
assert not miss, f"{len(miss)} atlas samples absent from {SAMPLE_META.name}: {sorted(miss)[:5]}"
roster_s = roster_s.join(sm[[c for c in SM_COLS if c in sm.columns]])

show(roster_s.sort_values(["compartment", "study", "patient_key"]),
     f"per-sample roster — {len(roster_s)} samples", rows=300)
roster_s.to_csv(TAB / "atlas_v2_sample_roster.csv")
print(f"\nwrote {TAB / 'atlas_v2_sample_roster.csv'}")

# patients whose libraries span more than one compartment or study
pk = A.groupby("patient_key", observed=True).agg(
    compartments=("compartment", "nunique"), studies=("study", "nunique"),
    samples=("sample_id", "nunique"), cells=("cell_id", "size"))
print(f"\npatients with paired blood AND skin: {(pk['compartments'] > 1).sum()}")
print(f"patients deposited in >1 study:      {(pk['studies'] > 1).sum()}")
show(pk[pk["compartments"] > 1].sort_values("cells", ascending=False),
     "paired blood/skin patients — the substrate for within-patient compartment contrasts")
```

```
=== per-sample roster — 263 samples ===
                                                    study        patient_key                 donor compartment       organ     tissue skin_layer     disease stage stage_group                              entity                        lesion          treatment                      blood_inv      sex      tech  n_cells  n_tcr  pct_tcr  n_clones  pct_labelled     accession          repo                         malignant_call_method tcr_available  tcr_recovery                     lineage               lineage_resolve                          origin_resolve
sample_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
D8__P102_SPD4_PP                                alkon2024        vienna:P102              D8__P102        Skin        skin       Skin      whole  CTCL_other    NA     unknown                          CTCL_other                       unknown            unknown                        unknown  unknown    10x_5p     9306      0      0.0         0         100.0     GSE247047           GEO                                      CNV_only            no           NaN      unresolved_provisional                            no                                      no
D8__P107_LPD5_MF                                alkon2024        vienna:P107              D8__P107        Skin        skin       Skin      whole          MF    NA     unknown                          MF_classic                       unknown            unknown                        unknown  unknown    10x_5p    11072      0      0.0         0         100.0     GSE247047           GEO                                      CNV_only            no           NaN      unresolved_provisional                            no                                      no
D8__P138_SPD5_MF                                alkon2024        vienna:P138              D8__P138        Skin        skin       Skin      whole          MF    NA     unknown                          MF_classic                       unknown            unknown                        unknown  unknown    10x_5p     5005      0      0.0         0         100.0     GSE247047           GEO                                      CNV_only            no           NaN      unresolved_provisional                            no                                      no
D8__P143_LPD2_PP                                alkon2024        vienna:P143              D8__P143        Skin        skin       Skin      whole  CTCL_other    NA     unknown                          CTCL_other                       unknown            unknown                        unknown  unknown    10x_5p    14554      0      0.0         0         100.0     GSE247047           GEO                                      CNV_only            no           NaN      unresolved_provisional                            no                                      no
D8__P144_SPD6_PP                                alkon2024        vienna:P144              D8__P144        Skin        skin       Skin      whole  CTCL_other    NA     unknown                          CTCL_other                       unknown            unknown                        unknown  unknown    10x_5p    13447      0      0.0         0         100.0     GSE247047           GEO                                      CNV_only            no           NaN      unresolved_provisional                            no                                      no
D8__P149_SPD7_PP                               
... [145166 more chars truncated]
```

```python  # cell 9
# --- 2d · clinical breakdowns (harmonized columns only) --------------------
BREAKDOWN = ["disease", "stage_class", "stage_group", "stage_clean", "entity_h", "sex_h",
             "treatment_group", "lesion_group", "organ", "skin_layer",
             "blood_involvement_eff", "cohort_country", "region_h", "tech"]
for col in BREAKDOWN:
    t = tri(A, col)
    show(t, f"{col} x compartment  (cells | patients | samples)")
    t.to_csv(TAB / f"atlas_v2_breakdown_{col}.csv")

print("\nRead every row above against the study table in 2b: a level carried by a single study "
      "is a cohort effect, not a disease effect.")
pat1 = A.drop_duplicates("patient_key")
for col in ["disease", "stage_group", "entity_h", "treatment_group", "lesion_group"]:
    n_stud = pat1.groupby(col, observed=True)["study"].nunique()
    single = n_stud[n_stud == 1]
    if len(single):
        who = (pat1[pat1[col].astype(str).isin(single.index.astype(str))]
               .groupby(col, observed=True)["study"].agg(lambda s: sorted(set(s.astype(str)))[0]))
        print(f"  {col}: single-study levels -> " +
              ", ".join(f"{k}={SH.get(v, v)}" for k, v in who.items()))
```

```
=== disease x compartment  (cells | patients | samples) ===
               cells                        patients                samples               
compartment     Skin   Blood    LN    TOTAL     Skin Blood LN TOTAL    Skin Blood LN TOTAL
disease                                                                                   
HC             83643   47679     0   131322       20     5  0    24      20     5  0    25
MF            918838  106867  1142  1026847       84    11  1    89     103    11  1   115
SS             55483  465364     0   520847        3    42  0    42      20    47  0    67
CTCL_other    287563  191114     0   478677       30    17  0    47      31    26  0    57
TOTAL        1345527  811024  1142  2157693      137    75  1   201     174    89  1   263

=== stage_class x compartment  (cells | patients | samples) ===
               cells                        patients                samples               
compartment     Skin   Blood    LN    TOTAL     Skin Blood LN TOTAL    Skin Blood LN TOTAL
stage_class                                                                               
HC             83643   47679     0   131322       20     5  0    24      20     5  0    25
early         257099   12091     0   269190       19     1  0    20      19     1  0    20
advanced      602100  579565  1142  1182807       55    59  1   105      88    64  1   153
other         140974   85847     0   226821       15     6  0    21      15    15  0    30
unknown       261711   85842     0   347553       31     4  0    34      32     4  0    35
TOTAL        1345527  811024  1142  2157693      137    75  1   201     174    89  1   263

=== stage_group x compartment  (cells | patients | samples) ===
               cells                        patients                samples               
compartment     Skin   Blood    LN    TOTAL     Skin Blood LN TOTAL    Skin Blood LN TOTAL
stage_group                                                                               
HC             83643   47679     0   131322       20     5  0    24      20     5  0    25
early         257099   12091     0   269190       19     1  0    20      19     1  0    20
advanced      412697   96358  1142   510197       33    11  1    44      46    11  1    58
unknown       592088  654896     0  1246984       70    58  0   119      89    72  0   160
TOTAL        1345527  811024  1142  2157693      137    75  1   201     174    89  1   263

=== stage_clean x compartment  (cells | patients | samples) ===
               cells                        patients                samples               
compartment     Skin   Blood    LN    TOTAL     Skin Blood LN TOTAL    Skin Blood LN TOTAL
stage_clean                                                                               
IA            138572   12091     0   150663        6     1  0     7       6     1  0     7
IB            111089       0     0   111089       11     0  0    11      11     0  0    11
IIA             7438       0     0     7438        2     0  0     2       2     0  0     2
IIB           243247   28974     0   272221       22     3  0    25      27     3  0    30
IIIA           56271       0     0    56271        3     0  0     3       6     0  0     6
IIIB           19639       0     0    19639        1     0  0     1       1     0  0     1
IV             17508       0     0    17508        2     0  0     2       2     0  0     2
IVA                0   64202     0    64202        0     7  0     7       0     7  0     7
IVA1           32995       0     0    32995        2     0  0     2       5     0  0     5
IVA2           43037       0     0    43037        3     0  0     3       5     0  0     5
IVB                0    3182  1142     4324        0     1  1     1       0     1  1     2
NA            675731  702575     0  1378306       90    63  0   143     109    77  0   185
TOTAL        1345527  811024  1142  2157693      137    75  1   201     174    89  1   263

=== entity_h 
... [10928 more chars truncated]
```

```python  # cell 10
# --- 2e · cell types and label coverage ------------------------------------
show(tri(A, "label_source"), "where does each cell's label come from?")

ct = pd.concat({
    "cells": A.groupby("cell_type_h", observed=True).size(),
    "% of atlas": 100 * A.groupby("cell_type_h", observed=True).size() / len(A),
    "patients": A.groupby("cell_type_h", observed=True)["patient_key"].nunique(),
    "studies": A.groupby("cell_type_h", observed=True)["study"].nunique(),
}, axis=1).sort_values("cells", ascending=False)
show(ct, "harmonized cell types (cell_type_h)")
ct.to_csv(TAB / "atlas_v2_celltype.csv")

show(mat(A, "cell_type_h", "compartment", "cells"), "cell_type_h x compartment — cells")

unl = A.loc[A["label_source"].astype(str).eq("NONE (unannotated)"), "compartment"]
if len(unl):
    print(f"\n{len(unl):,} cells carry no cell-type label at all:")
    print(unl.value_counts().to_string())
    print("Blood is unannotated until 33_blood_reannotation.ipynb is re-run on the v2 blood "
          f"object ({N_BLOOD:,} cells; the sidecar on disk covers the v1 423,042).")
```

```
=== where does each cell's label come from? ===
                       cells                        patients                samples               
compartment             Skin   Blood    LN    TOTAL     Skin Blood LN TOTAL    Skin Blood LN TOTAL
label_source                                                                                      
reannotated (nb10b)  1345527       0     0  1345527      137     0  0   137     174     0  0   174
reannotated (nb33)         0  811024     0   811024        0    75  0    75       0    89  0    89
NONE (unannotated)         0       0  1142     1142        0     0  1     1       0     0  1     1
TOTAL                1345527  811024  1142  2157693      137    75  1   201     174    89  1   263

=== harmonized cell types (cell_type_h) ===
               cells  % of atlas  patients  studies
cell_type_h                                        
CD4           997351        46.2       201       21
Keratinocyte  272566        12.6       136       14
Fibroblast    184452         8.5       136       14
CD8           168532         7.8       198       21
Myeloid       144009         6.7       137       14
Vascular      104873         4.9       136       14
Mono_CD14      69303         3.2        69       10
B              47114         2.2       181       20
CD4_Treg       41011         1.9       131       14
Unk            38280         1.8       128       14
cDC            25103         1.2        53        9
Plasma         18408         0.9       187       20
Erythroid      16323         0.8        53        9
Melanocyte     15422         0.7       122       13
Mono_CD16       5073         0.2        60       10
Platelet        4430         0.2        71       10
Mast            3695         0.2       114       13
unannotated     1142         0.1         1        1
NK               606         0.0        15        6

=== cell_type_h x compartment — cells ===
compartment      Skin   Blood    LN    TOTAL
cell_type_h                                 
CD4            442982  554369     0   997351
CD8             55923  112609     0   168532
CD4_Treg        41011       0     0    41011
B               28610   18504     0    47114
Plasma          13704    4704     0    18408
Myeloid        144009       0     0   144009
Mast             3695       0     0     3695
Keratinocyte   272566       0     0   272566
Fibroblast     184452       0     0   184452
Vascular       104873       0     0   104873
Melanocyte      15422       0     0    15422
Unk             38280       0     0    38280
NK                  0     606     0      606
Mono_CD14           0   69303     0    69303
Mono_CD16           0    5073     0     5073
cDC                 0   25103     0    25103
Platelet            0    4430     0     4430
Erythroid           0   16323     0    16323
unannotated         0       0  1142     1142
TOTAL         1345527  811024  1142  2157693

1,142 cells carry no cell-type label at all:
compartment
LN       1142
Skin        0
Blood       0
Blood is unannotated until 33_blood_reannotation.ipynb is re-run on the v2 blood object (811,024 cells; the sidecar on disk covers the v1 423,042).
```

## §3 · UMAP on the MrVI *u* embedding

2.16 M points do not plot — matplotlib chokes and the result is a black blob. `plot_view()`
(ported from `20_skin/21_reannotation.ipynb` §5) draws a **uniform random 300,000-cell subsample**
(14 % of the atlas) and embeds *that*.

Two properties worth keeping:

- **Uniform, not stratified.** At 300k/2.16M a population holding 0.1 % of cells still keeps
  ~300 points — enough to see. Stratifying would distort the relative sizes the figure is for.
- **The parent row count is in the cache filename** (`joint_umap_u_2157693_300000.npy`), so a
  rebuilt atlas can never silently reuse an embedding computed on a different set of cells. This
  is the failure mode that made `23_malignancy_tcr_cnv` report v1 numbers on a v2 input.

**Five figures, one per variable** — `cell type · study · stage · disease · compartment` — each
written to its own `figures/atlas_descriptive/umap_v2_<column>.png` with a right-margin legend.
Titles carry the variable name only; the subsample size is printed to stdout, so a figure read out
of context can never be mistaken for the whole atlas.

The subsample is for **plotting only** — never compute a label, a proportion or a statistic from
it. Every number in §2 comes from all 2,157,693 cells.

⚠️ The `sc.pp.neighbors` + `sc.tl.umap` call is the only heavy step in this notebook. Run it on
the GPU kernel; it caches to `.npy` and every later run is instant.

```python  # cell 12
# --- light embedding object: obs + the u latent, no expression -------------
u = np.load(U_NPY)
assert u.shape == (N_TOTAL, 10), f"latent is {u.shape}, expected ({N_TOTAL}, 10)"
emb = ad.AnnData(X=np.zeros((N_TOTAL, 1), np.float32), obs=A.copy(),
                 obsm={"X_mrvi_u": np.ascontiguousarray(u, dtype=np.float32)})
print("emb:", emb.shape, "| latent:", u.shape)


def plot_view(a, key, n=PLOT_N, seed=SEED):
    """A subsampled copy of `a` carrying a UMAP, for plotting only.

    Never use the result to compute labels — it is a random subset. The embedding caches as
    joint_umap_<key>_<parent_n>_<sub_n>.npy; the parent row count is in the name so a rebuilt
    atlas cannot silently reuse an embedding computed on different cells.
    """
    if a.n_obs <= n:
        sub, idx = a.copy(), np.arange(a.n_obs)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(a.n_obs, n, replace=False))
        sub = a[idx].copy()
    cache = OUT / f"joint_umap_{key}_{a.n_obs}_{sub.n_obs}.npy"
    if cache.exists():
        sub.obsm["X_umap"] = np.load(cache)
        print(f"plot_view[{key}]: {sub.n_obs:,}/{a.n_obs:,} cells, cached UMAP")
    else:
        sc.pp.neighbors(sub, use_rep="X_mrvi_u", random_state=seed)
        sc.tl.umap(sub, random_state=seed)
        np.save(cache, sub.obsm["X_umap"])
        print(f"plot_view[{key}]: {sub.n_obs:,}/{a.n_obs:,} cells, computed UMAP ->", cache.name)
    sub.uns["plot_idx"] = idx
    return sub


v = plot_view(emb, "u")
```

```
emb: (2157693, 1) | latent: (2157693, 10)
plot_view[u]: 300,000/2,157,693 cells, computed UMAP -> joint_umap_u_2157693_300000.npy
```

```python  # cell 13
# --- the five overview UMAPs, one figure each ------------------------------
def save_umap(a, colors, fname, size=2, legend_loc="right margin", legend_fontsize=None,
              figsize=None, **kw):
    """One figure per colour — never a grid. Saves to FIG/<fname stem>_<colour>.png.

    Every categorical gets a right-margin legend. Rather than dropping it on a high-cardinality
    column (21 studies, 12 cell types), the font is scaled to the level count and the canvas is
    widened; `set_box_aspect(1)` keeps the scatter square so the extra width goes to the legend
    and not to a stretched UMAP. Titles are the column label alone — the subsample size is
    printed to stdout, not written on the plot.
    """
    stem = Path(fname).stem
    written = []
    for c in [c for c in colors if c in a.obs]:
        s = a.obs[c]
        col, cat, ncat = c, False, 0
        if s.dtype == bool:                            # -> legend, not a 0/1 colourbar
            col = f"_pl_{c}"
            a.obs[col] = pd.Categorical(np.where(s.to_numpy(), "True", "False"),
                                        categories=["False", "True"])
            cat, ncat = True, 2
        elif isinstance(s.dtype, pd.CategoricalDtype):
            cat, ncat = True, int(s.cat.categories.size)
        fs = legend_fontsize or (10 if ncat <= 8 else 9 if ncat <= 15 else 7 if ncat <= 40 else 5)
        fw = figsize or (6 + min(6.0, 0.8 + 0.05 * ncat), 6)
        with mpl.rc_context({"figure.figsize": fw}):
            fig = sc.pl.umap(a, color=col, frameon=False, size=size, show=False, return_fig=True,
                             legend_loc=(legend_loc if cat else None),
                             legend_fontsize=fs, title=LABEL.get(c, c), **kw)
        fig.axes[0].set_box_aspect(1)
        for ax in fig.axes:
            for coll in ax.collections:
                coll.set_rasterized(True)
        out = FIG / f"{stem}_{c}.png"
        fig.savefig(out, dpi=200)
        plt.show(); plt.close(fig)
        if col != c:
            del a.obs[col]
        written.append(out)
        print(f"saved {out.name}  ({ncat} categories)" if cat else f"saved {out.name}")
    return written


LABEL = {"cell_type_h": "cell type", "compartment": "compartment", "study": "study",
         "disease": "disease", "stage_group": "stage"}

print(f"plotting {v.n_obs:,} of {emb.n_obs:,} cells (uniform random subsample)")
save_umap(v, ["cell_type_h", "study", "stage_group", "disease", "compartment"],
          "umap_v2.png", size=3)
```

```
plotting 300,000 of 2,157,693 cells (uniform random subsample)

<Figure size 852.5x660 with 1 Axes>
saved umap_v2_cell_type_h.png  (19 categories)

<Figure size 863.5x660 with 1 Axes>
saved umap_v2_study.png  (21 categories)

<Figure size 770x660 with 1 Axes>
saved umap_v2_stage_group.png  (4 categories)

<Figure size 770x660 with 1 Axes>
saved umap_v2_disease.png  (4 categories)

<Figure size 764.5x660 with 1 Axes>
saved umap_v2_compartment.png  (3 categories)

[PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/atlas_descriptive/umap_v2_cell_type_h.png'),
 PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/atlas_descriptive/umap_v2_study.png'),
 PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/atlas_descriptive/umap_v2_stage_group.png'),
 PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/atlas_descriptive/umap_v2_disease.png'),
 PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/atlas_descriptive/umap_v2_compartment.png')]
```

```python  # cell 14
# --- what the plotted subsample actually contains, so the figures read honestly ---
chk = pd.DataFrame({
    "subsample %": 100 * v.obs["compartment"].value_counts(normalize=True),
    "atlas %": 100 * A["compartment"].value_counts(normalize=True),
}).round(2)
show(chk, "the subsample is uniform: compartment shares match the full atlas")
```

```
=== the subsample is uniform: compartment shares match the full atlas ===
             subsample %  atlas %
compartment                      
Skin                62.2     62.4
Blood               37.7     37.6
LN                   0.1      0.1
```
