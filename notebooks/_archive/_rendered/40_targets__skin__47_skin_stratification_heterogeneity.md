# 40_targets/skin/47_skin_stratification_heterogeneity.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/skin/47_skin_stratification_heterogeneity.ipynb` and is never run._

# 47 — Skin: stratification and heterogeneity of the malignant-vs-reactive CD4 axis

42 asks whether an antigen separates tumour from reactive CD4 **on average**. That average is made
of ~90 patients with different diseases, different lesions, different stages and very different
tumour burdens, and MF is heterogeneous enough that an average can describe nobody. This notebook
asks the two questions that follow from it:

1. **Does the effect depend on who the patient is, and on which lesion was cut?** — MF vs Sézary,
   patch/plaque vs tumour/erythroderma, early vs advanced stage, CD30 status, large-cell
   transformation, folliculotropism, treatment at sampling, sex, age, tumour burden.
2. **Is the effect the same size in everyone, or large in a subset?** — random-effects
   heterogeneity, the subgroup statistics built for exactly this (COPA, the outlier sum),
   bimodality, donor clustering, and the eligibility curve a biomarker-selected trial is designed
   against.

**Where skin differs from the blood twin (44).** Blood could not test stage at all — 2 early-stage
donors against 35 advanced — and its disease split was 8 MF against 37 SS. Skin inverts both: the
cohort is MF-dominant, **22 early-stage against 62 advanced**, and it carries a variable the blood
cohort does not have at all — **which lesion was biopsied**. A patch, a plaque and a tumour from
the same disease are different tissue, and the malignant fraction rises across them, so the lesion
axis is simultaneously the most interesting stratification here and the one most confounded with
`burden`. §2a tests it first and prints that confounding rather than averaging over it.

**Power first.** §1 prints the counts before anything is tested, and every contrast is placed in
one of three tiers:

| tier | meaning |
|---|---|
| **1** | enough donors to test, and the test is run |
| **2** | enough to describe, with effect sizes and no claim |
| **3** | not testable — reported as a count, so the gap is visible rather than silently absent |

Nothing here re-derives the screen. It reads the cleaned cohort and the rebuilt tables that 42
produces, so **run 42 first** (or at least `jobs/run_target_screen_qc.sh --tissue skin`).

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))

import surface_targets as ST
import target_screen as TS
import target_cohort as TC
import target_qc as TQ
import target_figs as TF
import target_het as TH
import cluster_malig_helpers as CM
import malignancy as MAL

CM.use_tissue("skin")
MAL.use_tissue("skin")
TQ.use_tissue("skin")

TISSUE = "skin"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt47_"
MALIG, REACT, HC_CD4 = "CD4_malignant", "CD4_reactive", "CD4_HC"
MIN_CELLS, MIN_DONORS, DETECT_FLOOR = 20, 5, 0.10
N_PERM = 1000      # sign-flip permutations for the COPA / outlier-sum null

ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX
TF.rc()
pd.set_option("display.width", 240)

don = TC.skin_donor_table()
long = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet"), don)
meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)
man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
assert man.get("builder") == "run_target_screen_qc.py", (
    "run jobs/run_target_screen_qc.sh --tissue skin first -- this notebook needs sd_all for "
    "the random-effects model and mean_all on the true library")
bur = TC.burden(long)
don = don.join(bur, how="left")

paired = meta[meta["arm"].isin([MALIG, REACT])].groupby("donor")["arm"].nunique().eq(2)
PAIRED = sorted(paired.index[paired])
DON = don.loc[PAIRED]
DONOR_STUDY = dict(zip(meta["donor"], meta["study"]))

sel = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
prev = TS.prevalence(long, MALIG, detect_floor=DETECT_FLOOR)

# The gene set everything below runs on. 42's shortlist is the headline, but restricting the
# heterogeneity analysis to it would answer only "are my winners consistent" -- a gene that is
# flat on average because it is strongly up in a third of patients is invisible in that list
# and is exactly what this notebook exists to find. So: every gene that reaches the paired
# test, with the shortlist marked.
SL = ST.TAB_DIR / "tgt42_shortlist_skin.csv"
SHORT = pd.read_csv(SL)["gene"].tolist() if SL.exists() else []
if not SHORT:
    print(f"*** {SL.name} missing -- run 42 first; every 'on_shortlist' column below is False ***")
GENES = sel["gene"].tolist()
print(f"NB_DIR = {NB_DIR}")
print(f"{len(PAIRED)} paired donors | {len(GENES):,} genes tested | "
      f"{len(SHORT)} on 42's shortlist")
print("\nthe cohort, in one line per axis this notebook will try to split on:")
for col, label in (("dx2", "disease"), ("lesion", "lesion"), ("lesion_class", "lesion class"),
                   ("clin_stage_class", "stage class"), ("clin_B", "B stage"),
                   ("clin_cd30", "CD30"), ("clin_lct", "large-cell transformation"),
                   ("clin_fmf", "folliculotropic"), ("sex", "sex")):
    if col in DON:
        print(f"  {label:26s} " + ", ".join(f"{k}:{v}" for k, v in
                                            DON[col].value_counts(dropna=False).items()))
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
75 paired donors | 2,355 genes tested | 28 on 42's shortlist

the cohort, in one line per axis this notebook will try to split on:
  disease                    MF:67, SS:6, unresolved:2
  lesion                     tumor:19, mixed:17, erythroderma:11, plaque:10, patch:9, skin_unspecified:8, papule:1
  lesion class               lesion:67, unspecified:8
  stage class                advanced:55, early:18, nan:2
  B stage                    B0:40, nan:18, B2:9, B1:5, B2b:3
  CD30                       nan:58, pos:9, neg:8
  large-cell transformation  nan:43, yes:23, no:9
  folliculotropic            nan:70, yes:5
  sex                        M:44, F:31
```

## §1 — What this cohort can and cannot answer

Read the table before the sections. Every row is a candidate stratification; the counts decide
which tier it lands in, and the tier decides whether the result below is a test, a description or
an absence.

Three structural facts to keep in view:

- **The lesion is confounded with the burden.** A tumour lesion holds a larger malignant fraction
  than a patch, so "the antigen behaves differently in tumours" and "the antigen's effect scales
  with the malignant share of the compartment" are the same observation seen twice. §2a prints the
  burden distribution inside each lesion class, and §2d carries both terms in one model so they
  compete.
- **Treatment is confounded with study, and mostly absent.** Treatment at sampling is known for
  about half the cohort and no agent reaches the double digits; every treatment row here is tier 2
  or 3. The brentuximab2026 cohort is on BV by construction (`CLINICAL_METADATA.md:150`).
- **Burden is the only covariate with complete coverage.** `clin_blood_tumor_pct` is a blood
  measurement and is curated for a handful of these patients; the malignant share of the CD4
  compartment is computable for all of them. It is also the quantity 42 used as the contamination
  test, so a marker whose effect rides on burden should be read twice.

```python  # cell 3
# ============================================================================
# §1  The power table
# ============================================================================
FIELDS = {
    "dx2 (MF vs SS)": "dx2", "lesion": "lesion", "lesion class": "lesion_class",
    "sex": "sex", "age": "age",
    "stage": "clin_stage", "stage class": "clin_stage_class", "B stage": "clin_B",
    "CD30": "clin_cd30", "large-cell transformation": "clin_lct", "folliculotropic": "clin_fmf",
    "timepoint": "clin_timepoint", "treatment at sampling known": "tx_known",
    "treatment-naive": "clin_tx_naive", "prior lines": "n_prior_lines",
    "response": "clin_response", "outcome": "clin_outcome", "malignant burden": "burden",
}
rows = []
for label, col in FIELDS.items():
    if col not in DON.columns:
        rows.append({"field": label, "n": 0, "note": "column absent"}); continue
    v = DON[col]
    # A boolean column is never NA, so counting `notna` would report every donor as covered.
    have = v.astype(bool) if v.dtype == bool else v.notna()
    rec = {"field": label, "n": int(have.sum()),
           "MF": int(have[DON["dx2"].eq("MF")].sum()),
           "SS": int(have[DON["dx2"].eq("SS")].sum())}
    n = int(have.sum())
    if v.dtype == object and n:
        rec["levels"] = ", ".join(f"{k}:{c}" for k, c in v.value_counts().head(6).items())
    elif pd.api.types.is_numeric_dtype(v) and n:
        rec["levels"] = f"median {v.median():.3g}, range {v.min():.3g}-{v.max():.3g}"
    rows.append(rec)
power = pd.DataFrame(rows)
print(f"coverage over the {len(DON)} paired donors:")
print(power.to_string(index=False))

print("\ntreatment at sampling (a donor can be on several):")
tx = pd.DataFrame({k[3:]: DON[k] for k in DON.columns if k.startswith("tx_")
                   and k != "tx_known"}).sum().sort_values(ascending=False)
tx = tx[tx > 0].rename("donors").to_frame()
tx["MF"] = [int((DON[f"tx_{k}"] & DON["dx2"].eq("MF")).sum()) for k in tx.index]
tx["SS"] = [int((DON[f"tx_{k}"] & DON["dx2"].eq("SS")).sum()) for k in tx.index]
tx["studies"] = [", ".join(sorted(DON.loc[DON[f"tx_{k}"], "study"].unique())) for k in tx.index]
print(tx.to_string() if len(tx) else "  no treatment class reaches one donor at sampling")

print("\nlesion against study -- the confounding, as a table rather than a sentence:")
print(pd.crosstab(DON["study"], DON["lesion"], dropna=False).to_string())
print("\nlesion against stage class:")
print(pd.crosstab(DON["lesion"], DON["clin_stage_class"], dropna=False).to_string())

TIERS = {}
for label, col in FIELDS.items():
    if col not in DON.columns:
        TIERS.setdefault(3, []).append(f"{label} (column absent)"); continue
    v = DON[col]
    if pd.api.types.is_numeric_dtype(v) and v.dtype != bool:
        n = int(v.notna().sum())
        TIERS.setdefault(1 if n >= 20 else 2 if n >= 8 else 3, []).append(f"{label} (n={n})")
    else:
        vc = (v.astype(bool) if v.dtype == bool else v).value_counts()
        small = int(vc.iloc[1]) if len(vc) > 1 else 0
        TIERS.setdefault(1 if small >= 10 else 2 if small >= 4 else 3,
                         []).append(f"{label} (smaller side n={small})")
print("\n" + "=" * 78)
for t in sorted(TIERS):
    print(f"TIER {t}: " + "; ".join(TIERS[t]))
print("=" * 78)
print("tier 1 = tested below; 2 = effect sizes only; 3 = counted, so the gap is explicit.")
power.to_csv(ST.tab("power_table"), index=False)
tx.to_csv(ST.tab("treatment_counts"))
print("wrote", ST.tab("power_table"), "and", ST.tab("treatment_counts"))
```

```
coverage over the 75 paired donors:
                      field  n  MF  SS                                                                      levels
             dx2 (MF vs SS) 75  67   6                                                   MF:67, SS:6, unresolved:2
                     lesion 75  67   6 tumor:19, mixed:17, erythroderma:11, plaque:10, patch:9, skin_unspecified:8
               lesion class 75  67   6                                                    lesion:67, unspecified:8
                        sex 75  67   6                                                                  M:44, F:31
                        age 75  67   6                                                      median 64, range 30-87
                      stage 73  67   4                                  IIB:30, IB:11, IVA1:8, IA:7, IVB:4, IVA2:4
                stage class 73  67   4                                                       advanced:55, early:18
                    B stage 57  51   4                                                    B0:40, B2:9, B1:5, B2b:3
                       CD30 17  17   0                                                                pos:9, neg:8
  large-cell transformation 32  32   0                                                                yes:23, no:9
            folliculotropic  5   5   0                                                                       yes:5
                  timepoint 75  67   6                      single:60, pre:5, baseline:5, serial:4, on_treatment:1
treatment at sampling known 51  43   6                                                         median 1, range 0-1
            treatment-naive 58  50   6                                                 no:46, yes:6, washout_6mo:6
                prior lines  0   0   0                                                                         NaN
                   response  7   6   1                                                 PD:3, cleared:2, NR:1, SD:1
                    outcome 19  19   0                                         alive:13, DOD:5, died_unspecified:1
           malignant burden 75  67   6                                           median 0.658, range 0.00837-0.985

treatment at sampling (a donor can be on several):
                 donors  MF  SS                                                 studies
ecp                   7   5   2  brunner2024, buus2025, chennareddy2025, rindler2021_mc
retinoid              4   3   1                chennareddy2025, gaydosik2019, il4ra2026
interferon            2   2   0                            brunner2024, chennareddy2025
methotrexate          2   2   0                                               il4ra2026
topical_steroid       2   1   1                                 gaydosik2019, il4ra2026
hdaci                 1   0   1                                            gaydosik2019
brentuximab           1   1   0                                         brentuximab2026
mogamulizumab         1   1   0                                            gaydosik2022
alemtuzumab           1   0   1                                                buus2025
other_immunomod       1   1   0                                               jonak2021
phototherapy          1   1   0                                               il4ra2026

lesion against study -- the confounding, as a table rather than a sentence:
lesion           erythroderma  mixed  papule  patch  plaque  skin_unspecified  tumor
study                                                                               
alkon2024                   0      0       0      5       0                 0      0
brentuximab2026             0      0       0      0       3                 0      3
brunner2024                 3      0       0      0       0                 0      0
buus2025                    2      0       0      0       0                 0      0
chennareddy2025             1      0       0      0       1                 0      3
... [1917 more chars truncated]
```

## §2 — Tier 1: the contrasts with enough donors to test

Each contrast is the same paired within-donor test as 42, run inside each stratum, plus a
two-sample comparison of the per-donor effects between strata (`p_interaction`). That interaction
is the question — not "is the antigen up in tumour lesions" but "is it *more* up in tumour lesions
than in patches" — and the n for each side is printed on every row so it cannot be read as more
than that.

§2a takes the lesion first, because it is the axis blood does not have.

```python  # cell 5
# ============================================================================
# §2a  The lesion: patch/plaque vs tumour/erythroderma
# ============================================================================
# The axis the blood cohort cannot offer. A patch, a plaque and a tumour are different tissue,
# and the clinical question behind the split is direct: does a candidate antigen survive the
# progression that makes a patient need systemic therapy in the first place.
LES_ADV = ("tumor", "erythroderma")
LES_EARLY = ("patch", "plaque")
les = DON["lesion"].map(lambda v: "tumour/eryth" if v in LES_ADV
                        else "patch/plaque" if v in LES_EARLY else np.nan).dropna()
print(f"patch/plaque {int(les.eq('patch/plaque').sum())} donors vs "
      f"tumour/erythroderma {int(les.eq('tumour/eryth').sum())} "
      f"({len(DON) - len(les)} mixed, unspecified or healthy, excluded from this contrast)")

print("\nthe confounding, before the test: malignant burden inside each lesion class")
print(DON.groupby(les)["burden"].describe(percentiles=[.25, .5, .75]).round(3).to_string())
_b = DON.loc[les.index].groupby(les)["burden"].apply(list)
if len(_b) == 2:
    _u = stats.mannwhitneyu(*_b.tolist())
    print(f"  Mann-Whitney on burden between the two classes: p = {_u.pvalue:.3g}")
    print("  A small p here means the lesion contrast and the burden contrast are largely the "
          "same comparison. §2d carries both terms at once so they compete for the same "
          "variance.")

lls = TS.strat_selectivity(long, MALIG, REACT, les, value="mean_all", min_donors=3)
wide = lls.pivot_table(index="gene", columns="stratum",
                       values=["log2fc_paired", "n_donors_paired", "frac_donors_up"])
wide.columns = [f"{a}_{b}".replace("/", "_") for a, b in wide.columns]
inter = (lls[["gene", "delta_lfc", "p_interaction", "q_interaction"]]
         .drop_duplicates("gene").set_index("gene"))
les_tab = wide.join(inter).join(sel.set_index("gene")[["log2fc_paired", "q"]]
                                .rename(columns={"log2fc_paired": "log2fc_all", "q": "q_all"}))
les_tab = les_tab.reset_index()
les_tab["on_shortlist"] = les_tab["gene"].isin(SHORT)

LES_COLS = [c for c in les_tab.columns
            if c.startswith(("log2fc_paired", "n_donors_paired"))]
LES_COLS = (["gene"] + sorted(LES_COLS)
            + ["delta_lfc", "q_interaction", "log2fc_all", "on_shortlist"])
sig = les_tab[les_tab["q_interaction"] < 0.05].sort_values("delta_lfc")
print(f"\ngenes whose malignant-vs-reactive effect differs by lesion class at q < 0.05: "
      f"{len(sig)}")
print(sig[LES_COLS].head(40).to_string(index=False, float_format=lambda v: f"{v:.3g}")
      if len(sig) else "  none")

print("\n42's shortlist, split by lesion class (read the two n columns):")
print(les_tab[les_tab["on_shortlist"]].sort_values("log2fc_all", ascending=False)[LES_COLS]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
les_tab.to_csv(ST.tab("strat_lesion"), index=False)
print("\nwrote", ST.tab("strat_lesion"))
```

```
patch/plaque 19 donors vs tumour/erythroderma 30 (26 mixed, unspecified or healthy, excluded from this contrast)

the confounding, before the test: malignant burden inside each lesion class
              count   mean    std    min    25%    50%    75%    max
lesion                                                              
patch/plaque   19.0  0.544  0.234  0.153  0.329  0.566  0.752  0.883
tumour/eryth   30.0  0.589  0.291  0.008  0.359  0.624  0.827  0.985
  Mann-Whitney on burden between the two classes: p = 0.394
  A small p here means the lesion contrast and the burden contrast are largely the same comparison. §2d carries both terms at once so they compete for the same variance.


genes whose malignant-vs-reactive effect differs by lesion class at q < 0.05: 0
  none

42's shortlist, split by lesion class (read the two n columns):
    gene  log2fc_paired_patch_plaque  log2fc_paired_tumour_eryth  n_donors_paired_patch_plaque  n_donors_paired_tumour_eryth  delta_lfc  q_interaction  log2fc_all  on_shortlist
    CD70                        1.09                         1.1                            19                            30     0.0132              1         1.1          True
     CD9                        1.08                       0.946                            19                            30     -0.139              1        1.08          True
     CPM                        1.39                        1.39                            19                            30    0.00716              1        1.02          True
  COL6A3                        1.17                       0.842                            19                            30     -0.328          0.898       0.811          True
   CD151                       0.988                       0.103                            19                            30     -0.885           0.74       0.794          True
     GRN                       0.953                       0.623                            19                            30      -0.33           0.88       0.778          True
    TFRC                        1.07                       0.693                            19                            30      -0.38          0.704       0.731          True
 SLC39A6                        0.84                       0.512                            19                            30     -0.328          0.678       0.707          True
  SLC4A2                       0.763                       0.537                            19                            30     -0.226          0.876       0.698          True
TNFRSF25                       0.947                       0.722                            19                            30     -0.224          0.999       0.698          True
  SLC1A4                       0.307                       0.724                            19                            30      0.416              1        0.69          True
  LGALS3                        1.06                      0.0216                            19                            30      -1.04              1       0.663          True
   PDIA4                       0.919                        0.48                            19                            30     -0.439           0.65       0.637          True
    FUT8                       0.659                       0.676                            19                            30     0.0166              1       0.635          True
    CD82                       0.609                       0.593                            19                            30    -0.0161              1       0.632          True
  SLC5A3                       0.846                       0.514                            19                            30     -0.332          0.785       0.617          True
    IL6R                       0.687                       0.844                            19                            30      0.157      
... [2050 more chars truncated]
```

```python  # cell 6
# ============================================================================
# §2b  Disease: MF vs Sezary, and stage: early vs advanced
# ============================================================================
# Two contrasts in one cell because they ask the same shape of question and neither needs its
# own figure. Skin is the compartment where the stage contrast is possible at all: blood had 2
# early-stage donors, here there are enough to test.
_strat2b = {}
for name, lab_map, fname in (
        ("disease (MF vs SS)", DON.loc[DON["dx2"].isin(["MF", "SS"]), "dx2"], "strat_disease"),
        ("stage class", DON["clin_stage_class"].dropna(), "strat_stage")):
    lab = lab_map.dropna()
    counts = lab.value_counts().to_dict()
    print("\n" + "=" * 78)
    print(f"{name}: {counts}")
    print("=" * 78)
    if len(counts) < 2 or min(counts.values()) < 3:
        print("  fewer than 3 donors on one side -- tier 3, counted only")
        continue
    s = TS.strat_selectivity(long, MALIG, REACT, lab, value="mean_all", min_donors=3)
    w = s.pivot_table(index="gene", columns="stratum",
                      values=["log2fc_paired", "n_donors_paired"])
    w.columns = [f"{a}_{b}".replace(" ", "_") for a, b in w.columns]
    w = w.join(s[["gene", "delta_lfc", "p_interaction", "q_interaction"]]
               .drop_duplicates("gene").set_index("gene")).reset_index()
    w["on_shortlist"] = w["gene"].isin(SHORT)
    cc = (["gene"] + sorted(c for c in w.columns if c.startswith(("log2fc_paired",
                                                                  "n_donors_paired")))
          + ["delta_lfc", "q_interaction", "on_shortlist"])
    sig = w[w["q_interaction"] < 0.05].sort_values("delta_lfc")
    print(f"genes differing at q < 0.05: {len(sig)}")
    print(sig[cc].head(30).to_string(index=False, float_format=lambda v: f"{v:.3g}")
          if len(sig) else "  none -- with these counts that is the expected result, and it is "
                           "not evidence that the two strata behave the same")
    print("\n42's shortlist on this split:")
    print(w[w["on_shortlist"]][cc].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    _strat2b[fname] = (name, w[cc])
    w.to_csv(ST.tab(fname), index=False)
    print("wrote", ST.tab(fname))

# Within-study, where the strata actually sit side by side. Everywhere else, the stratum and the
# study are partly the same variable.
print("\n" + "=" * 78)
print("within-study MF vs SS, the only places the contrast is not confounded with cohort:")
for study in sorted(DON["study"].unique()):
    sub = DON[DON["study"].eq(study) & DON["dx2"].isin(["MF", "SS"])]
    if sub["dx2"].nunique() < 2 or sub["dx2"].value_counts().min() < 2:
        continue
    print(f"\n  {study}: {sub['dx2'].value_counts().to_dict()}")
    s = TS.strat_selectivity(
        long[long["donor"].isin(sub.index) & long["gene"].isin(SHORT[:12])],
        MALIG, REACT, sub["dx2"], value="mean_all", min_donors=2, interaction=False)
    print(s.pivot_table(index="gene", columns="stratum", values="log2fc_paired")
          .round(2).to_string())
```

```
==============================================================================
disease (MF vs SS): {'MF': 67, 'SS': 6}
==============================================================================

genes differing at q < 0.05: 66
    gene  log2fc_paired_MF  log2fc_paired_SS  n_donors_paired_MF  n_donors_paired_SS  delta_lfc  q_interaction  on_shortlist
    CCR7            -0.278              2.35                  67                   6      -2.63         0.0149         False
   S1PR1             -1.29              1.15                  67                   6      -2.44         0.0399         False
  GPR155            -0.371              1.12                  67                   6       -1.5         0.0409         False
   PELI1            -0.328              1.01                  67                   6      -1.34         0.0149         False
   CTLA4             0.086              1.15                  67                   6      -1.07          0.046         False
    CD55            -0.734             0.193                  67                   6     -0.927         0.0409         False
   MFSD8            0.0673             0.552                  67                   6     -0.484         0.0409         False
   GALR2                 0             0.141                  67                   6     -0.141         0.0453         False
KIAA0922                 0             0.044                  67                   6     -0.044         0.0227         False
  OR2A14                 0                 0                  67                   6          0         0.0409         False
   GFRA4                 0                 0                  67                   6          0         0.0409         False
   OR2C1                 0                 0                  67                   6          0         0.0409         False
    MC2R                 0                 0                  67                   6          0        0.00182         False
   HTR3B                 0                 0                  67                   6          0         0.0409         False
   OR2B2                 0                 0                  67                   6          0         0.0409         False
    APOB                 0                 0                  67                   6          0         0.0409         False
   OR2H1                 0                 0                  67                   6          0         0.0409         False
   OR2C3                 0                 0                  67                   6          0         0.0409         False
   OR2L5                 0                 0                  67                   6          0         0.0409         False
   OR1L3                 0                 0                  67                   6          0         0.0409         False
   OR6K3                 0                 0                  67                   6          0         0.0409         False
   OR6F1                 0                 0                  67                   6          0         0.0409         False
   OR5B3                 0                 0                  67                   6          0         0.0409         False
  OR56A3                 0                 0                  67                   6          0         0.0409         False
  OR51I2                 0                 0                  67                   6          0         0.0409         False
   OR4B1                 0                 0                  67                   6          0         0.0409         False
  OR52N1                 0                 0                  67                   6          0         0.0409         False
  OR52B6                 0                 0                  67                   6          0         0.0409         False
  OR2T10                 0                 0                  67                   6          0         0.0409         False
   OR2L8           
... [12966 more chars truncated]
```

```python  # cell 7
# ============================================================================
# §2c  Everything at once: per-gene regression of the donor effect on who the donor is
# ============================================================================
# One model per gene, over donors: does the within-donor effect depend on lesion class,
# diagnosis, stage, sex, age or tumour burden. Lesion and burden are in the same design on
# purpose -- §2a showed they are confounded, and this is where they compete for the variance.
# Study is left out (with this many studies it would absorb the lesion term it is confounded
# with) and is reported separately as the variance it explains on its own.
M = TH.lfc_matrix(long, MALIG, REACT, min_donors=MIN_DONORS)
X = pd.DataFrame(index=M.columns)
X["is_tumour_lesion"] = DON["lesion"].reindex(X.index).isin(["tumor", "erythroderma"]).astype(float)
X["is_MF"] = DON["dx2"].reindex(X.index).eq("MF").astype(float)
X["is_advanced"] = DON["clin_stage_class"].reindex(X.index).eq("advanced").astype(float)
X["is_F"] = DON["sex"].reindex(X.index).eq("F").astype(float)
X["age_z"] = ((DON["age"].reindex(X.index) - DON["age"].mean()) / DON["age"].std())
X["burden_z"] = ((DON["burden"].reindex(X.index) - DON["burden"].mean())
                 / DON["burden"].std())
X = X.dropna()
print(f"{len(X)} donors with every covariate non-missing (of {M.shape[1]}); "
      f"terms: {list(X.columns)}")
print(X.describe().round(3).to_string())
print("\ncorrelation between the terms -- where this is large, the coefficients split one "
      "signal between them and neither is interpretable alone:")
print(X.corr().round(2).to_string())

Xd = sm.add_constant(X)
rows = []
for gene, y in M.loc[:, X.index].iterrows():
    y = y.dropna()
    if len(y) < 20:
        continue
    fit = sm.OLS(y.to_numpy(float), Xd.loc[y.index]).fit()
    rec = {"gene": gene, "n": len(y), "r2": fit.rsquared}
    for t in X.columns:
        rec[f"b_{t}"] = fit.params[t]
        rec[f"p_{t}"] = fit.pvalues[t]
    rows.append(rec)
reg = pd.DataFrame(rows)
for t in X.columns:
    reg[f"q_{t}"] = multipletests(reg[f"p_{t}"], method="fdr_bh")[1]
reg["on_shortlist"] = reg["gene"].isin(SHORT)

print(f"\n{len(reg):,} genes modelled. Terms surviving BH at q < 0.05:")
print(pd.Series({t: int((reg[f"q_{t}"] < 0.05).sum()) for t in X.columns}).to_string())

for t in X.columns:
    s = reg[reg[f"q_{t}"] < 0.05].reindex(reg[f"b_{t}"].abs().sort_values(ascending=False).index)
    s = s.dropna(subset=["gene"]).head(15)
    print(f"\n--- {t}: {len(reg[reg[f'q_{t}'] < 0.05])} genes, strongest 15")
    if len(s):
        print(s[["gene", f"b_{t}", f"q_{t}", "r2", "n", "on_shortlist"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        print("  none")

print("\nstudy on its own, as a one-way ANOVA per gene (not in the model above):")
st = DON["study"].reindex(M.columns)
rows = []
for gene, y in M.iterrows():
    y = y.dropna()
    g = st.reindex(y.index)
    groups = [y[g.eq(s)].to_numpy() for s in g.dropna().unique() if (g.eq(s)).sum() >= 3]
    if len(groups) < 3:
        continue
    rows.append({"gene": gene, "p_study": float(stats.f_oneway(*groups).pvalue)})
anv = pd.DataFrame(rows)
anv["q_study"] = multipletests(anv["p_study"], method="fdr_bh")[1]
print(f"  {int((anv['q_study'] < 0.05).sum()):,} of {len(anv):,} genes differ by study at "
      "q < 0.05 -- the scale of the batch term the model above does not carry.")

reg.merge(anv, on="gene", how="left").to_csv(ST.tab("strat_regression"), index=False)
print("\nwrote", ST.tab("strat_regression"))
```

```
75 donors with every covariate non-missing (of 75); terms: ['is_tumour_lesion', 'is_MF', 'is_advanced', 'is_F', 'age_z', 'burden_z']
       is_tumour_lesion   is_MF  is_advanced    is_F   age_z  burden_z
count            75.000  75.000       75.000  75.000  75.000    75.000
mean              0.400   0.893        0.733   0.413  -0.000     0.000
std               0.493   0.311        0.445   0.496   1.000     1.000
min               0.000   0.000        0.000   0.000  -2.382    -2.155
25%               0.000   1.000        0.000   0.000  -0.555    -0.928
50%               0.000   1.000        1.000   0.000   0.102     0.230
75%               1.000   1.000        1.000   1.000   0.833     0.848
max               1.000   1.000        1.000   1.000   1.783     1.428

correlation between the terms -- where this is large, the coefficients split one signal between them and neither is interpretable alone:
                  is_tumour_lesion  is_MF  is_advanced  is_F  age_z  burden_z
is_tumour_lesion              1.00  -0.34         0.37  0.09   0.21     -0.02
is_MF                        -0.34   1.00        -0.01 -0.06  -0.08     -0.01
is_advanced                   0.37  -0.01         1.00 -0.11   0.03      0.17
is_F                          0.09  -0.06        -0.11  1.00  -0.17      0.11
age_z                         0.21  -0.08         0.03 -0.17   1.00      0.02
burden_z                     -0.02  -0.01         0.17  0.11   0.02      1.00


2,355 genes modelled. Terms surviving BH at q < 0.05:
is_tumour_lesion    0
is_MF               0
is_advanced         0
is_F                0
age_z               0
burden_z            0

--- is_tumour_lesion: 0 genes, strongest 15
  none

--- is_MF: 0 genes, strongest 15
  none

--- is_advanced: 0 genes, strongest 15
  none

--- is_F: 0 genes, strongest 15
  none

--- age_z: 0 genes, strongest 15
  none

--- burden_z: 0 genes, strongest 15
  none

study on its own, as a one-way ANOVA per gene (not in the model above):

  0 of 2,355 genes differ by study at q < 0.05 -- the scale of the batch term the model above does not carry.

wrote /home/projects/nyosef/zvise/MF/tables/tgt47_strat_regression.csv
```

## §3 — Tiers 2 and 3

Tier 2 is described, not tested: six donors on a retinoid can produce an effect size and a
confidence interval, and nothing that should survive a multiple-testing correction. Tier 3 is
printed as counts only. A stratification that is absent from a notebook reads as a stratification
that was not interesting; one that is printed with `n = 1` reads as what it is.

The skin-only clinical flags live here: **CD30 status**, **large-cell transformation** and
**folliculotropism** are recorded for part of the cohort and are exactly the kind of thing a
target programme would want stratified — CD30 because brentuximab is approved against it, LCT
because it marks the patients who progress. None of them reaches tier 1.

```python  # cell 9
# ============================================================================
# §3  Tier 2 described, tier 3 counted
# ============================================================================
def describe_split(mask, label, genes, min_side=3):
    """Median per-donor effect on each side of a binary split, with a CI and no test."""
    m = mask.dropna().astype(bool)
    a, b = m.index[m], m.index[~m]
    if min(len(a), len(b)) < min_side:
        return pd.DataFrame()
    rows = []
    for gene in genes:
        if gene not in M.index:
            continue
        ya, yb = M.loc[gene, M.columns.intersection(a)], M.loc[gene, M.columns.intersection(b)]
        ya, yb = ya.dropna(), yb.dropna()
        if min(len(ya), len(yb)) < min_side:
            continue
        boot = [np.median(np.random.default_rng(i).choice(ya, len(ya)))
                - np.median(np.random.default_rng(i + 7).choice(yb, len(yb)))
                for i in range(400)]
        rows.append({"split": label, "gene": gene, "n_yes": len(ya), "n_no": len(yb),
                     "median_yes": float(np.median(ya)), "median_no": float(np.median(yb)),
                     "delta": float(np.median(ya) - np.median(yb)),
                     "ci_lo": float(np.percentile(boot, 2.5)),
                     "ci_hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows)


def _yes(col, truthy=("yes", "pos", "positive", "True", True)):
    if col not in DON:
        return pd.Series(np.nan, index=DON.index)
    v = DON[col]
    return v.map(lambda x: x in truthy if pd.notna(x) else np.nan)


TIER2 = {
    "CD30 positive": _yes("clin_cd30"),
    "large-cell transformation": _yes("clin_lct"),
    "folliculotropic MF": _yes("clin_fmf"),
    "B2 (vs B0/B1)": DON["clin_B"].map(lambda v: v == "B2" if pd.notna(v) else np.nan),
    "any systemic treatment at sampling": DON["clin_current_systemic"].map(
        lambda v: v == "yes" if pd.notna(v) else np.nan),
    "retinoid at sampling": DON["tx_retinoid"].where(DON["tx_known"]),
    "ECP at sampling": DON["tx_ecp"].where(DON["tx_known"]),
    "brentuximab at sampling": DON["tx_brentuximab"].where(DON["tx_known"]),
    "treatment-naive": DON["clin_tx_naive"].map(
        lambda v: v == "yes" if pd.notna(v) else np.nan),
    "cultured ex vivo": DON["is_cultured"],
    "erythrodermic lesion": DON["lesion"].map(
        lambda v: v == "erythroderma" if pd.notna(v) else np.nan),
}
t2 = pd.concat([describe_split(v, k, SHORT[:15]) for k, v in TIER2.items()],
               ignore_index=True)
print("TIER 2 -- effect sizes with bootstrap CIs, on 42's leading antigens. No test, no "
      "correction, and no row here is evidence of a stratum effect:\n")
if len(t2):
    for split, d in t2.groupby("split"):
        print(f"--- {split}  (n = {d['n_yes'].iloc[0]} vs {d['n_no'].iloc[0]})")
        print(d[["gene", "median_yes", "median_no", "delta", "ci_lo", "ci_hi"]]
              .sort_values("delta").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print()
    t2.to_csv(ST.tab("tier2_descriptive"), index=False)
    print("wrote", ST.tab("tier2_descriptive"))
else:
    print("  no tier-2 split has >= 3 donors on both sides")

print("=" * 78)
print("TIER 3 -- not testable in this cohort. The counts, so the gap is explicit:")
print("=" * 78)
t3 = []
for col, name, note in (
        ("clin_response", "response", "recorded for a handful of donors"),
        ("clin_outcome", "outcome", "essentially absent"),
        ("clin_drug_sensitivity", "drug sensitivity", "romidepsin arm only, elsewhere empty"),
        ("clin_timepoint", "timepoint", "almost every sample is a single timepoint"),
        ("clin_tx_naive", "treatment-naive", "'unknown' dominates; a bare steroid counts as "
                                             "unknown by construction"),
        ("n_prior_lines", "prior lines", "dorando26 only, and that is a blood cohort")):
    if col in DON:
        t3.append((name, DON[col].value_counts(dropna=False).head(6).to_dict(), note))
for agent in ("mogamulizumab", "alemtuzumab", "chemotherapy", "checkpoint", "alloSCT",
              "methotrexate", "interferon", "hdaci"):
    c = f"tx_{agent}"
    if c in DON:
        n = int(DON[c].sum())
        if n <= 4:
            t3.append((f"treatment: {agent}", {"on": n, "off": int(len(DON) - n)},
                       "single-donor case note at most"))
for name, counts, note in t3:
    print(f"  {name:34s} {counts}")
    print(f"  {'':34s} -> {note}")
```

```
TIER 2 -- effect sizes with bootstrap CIs, on 42's leading antigens. No test, no correction, and no row here is evidence of a stratum effect:

--- B2 (vs B0/B1)  (n = 9 vs 48)
    gene  median_yes  median_no  delta  ci_lo  ci_hi
  LGALS3      -0.974       1.06  -2.04  -3.61 -0.605
     CD9      -0.163       1.29  -1.46  -2.94   2.15
   CD151      -0.113       1.04  -1.15  -1.29  0.416
  SLC1A4      -0.073       0.71 -0.783  -1.66 -0.124
    CD70       0.485       1.07 -0.589 -0.827  0.819
  COL6A3       0.771       1.17 -0.395 -0.797   0.55
   PDIA4       0.408      0.757 -0.349 -0.778  0.233
    CD82       0.359      0.662 -0.303  -1.07 0.0825
     GRN       0.515      0.799 -0.285 -0.673  0.657
    TFRC       0.631      0.898 -0.267 -0.668  0.122
     CPM        1.14       1.41 -0.267  -1.46  0.827
  SLC4A2       0.566       0.76 -0.194  -0.66  0.668
 SLC39A6       0.601      0.716 -0.115 -0.896  0.457
    FUT8       0.568      0.676 -0.108 -0.512  0.207
TNFRSF25       0.812      0.751 0.0606  -1.52  0.636

--- CD30 positive  (n = 9 vs 8)
    gene  median_yes  median_no   delta   ci_lo  ci_hi
  LGALS3      -0.256       1.08   -1.34   -2.49    2.2
     CD9        1.08       1.76  -0.675   -2.44   2.38
  SLC1A4       0.307      0.604  -0.297   -2.27   0.63
   PDIA4       0.938       1.23  -0.289  -0.808  0.359
 SLC39A6       0.725      0.919  -0.194  -0.538   0.45
    TFRC        1.01       1.07 -0.0586  -0.884  0.884
TNFRSF25       0.947      0.917  0.0294  -0.686    2.1
     GRN        1.02      0.872    0.15  -0.802  0.646
  SLC4A2           1      0.724   0.276   -0.65  0.971
    FUT8        1.01      0.728   0.281  -0.574   1.39
   CD151        1.33       1.01   0.318  0.0425   1.28
    CD82        1.06      0.678    0.38   -1.12  0.851
     CPM        1.72       1.25   0.474   -2.34   2.57
    CD70        1.12      0.377   0.739 -0.0416   2.25
  COL6A3        1.49      0.468    1.02 -0.0731   2.41

--- ECP at sampling  (n = 7 vs 44)
    gene  median_yes  median_no    delta  ci_lo   ci_hi
   CD151      -0.152      0.974    -1.13  -1.45    0.54
     CD9       0.378       1.08   -0.701  -1.92    2.43
   PDIA4      0.0666      0.751   -0.685 -0.915   0.101
  SLC1A4        0.16      0.761   -0.601  -1.61 0.00603
  LGALS3     -0.0758      0.378   -0.454   -1.7    1.52
TNFRSF25       0.599      0.836   -0.237 -0.753    1.01
    TFRC       0.692      0.855   -0.163 -0.632   0.122
 SLC39A6       0.576      0.705   -0.129 -0.749   0.221
  SLC4A2       0.439      0.556   -0.116 -0.627   0.829
    CD82       0.566      0.679   -0.113  -1.26   0.186
    FUT8       0.645      0.709  -0.0639 -0.424   0.373
  COL6A3       0.887      0.895 -0.00796 -0.672   0.659
     GRN       0.933      0.754    0.179 -0.477   0.658
     CPM        1.82       1.35    0.464 -0.826    1.56
    CD70        1.99       1.07    0.912 -0.608    1.65

--- any systemic treatment at sampling  (n = 15 vs 36)
    gene  median_yes  median_no   delta  ci_lo  ci_hi
  LGALS3      -0.459       1.04    -1.5  -1.89  0.159
  SLC1A4       0.135      0.836    -0.7  -1.55 -0.184
     CD9       0.471       1.08  -0.607  -1.92   1.32
   CD151       0.524      0.951  -0.427  -1.19  0.914
     GRN       0.499      0.828  -0.328 -0.518  0.352
   PDIA4       0.418      0.709  -0.291 -0.771  0.416
    TFRC       0.691      0.895  -0.203 -0.707  0.104
 SLC39A6       0.576      0.705  -0.129  -0.47  0.174
  SLC4A2       0.439      0.568  -0.129 -0.621  0.398
TNFRSF25       0.754      0.836 -0.0816 -0.478   0.62
    CD82       0.619      0.679 -0.0602 -0.699  0.233
    CD70        1.16       1.09  0.0647 -0.693  0.931
    FUT8       0.866      0.659   0.207 -0.275  0.623
  COL6A3       0.994      0.776   0.218 -0.528  0.813
     CPM        1.59       1.11    0.48 -0.336   1.68

--- cultured ex vivo  (n = 4 vs 71)
    gene  median_yes  median_no    delta  ci_lo  ci_hi
     CPM       0.117       1.08   -0.968  -1.85  0.519
TNFRSF25     -0.0173      0.747   -0.765  -1.71 -0.148
     GRN 
... [6649 more chars truncated]
```

## §4 — Heterogeneity: is the effect the same size in everyone?

Four views of the same donor × gene matrix, because they disagree and the disagreement is the
information.

1. **Random-effects meta-analysis.** Each donor contributes an effect and a standard error (delta
   method, from the per-arm cell SDs the QC build stores). `tau2` is the spread of the *true*
   effect once each donor's own sampling noise is removed, `I2` the share of observed spread that
   is not noise, and the prediction interval what the next patient would plausibly show. A large
   pooled effect with `I2` near 1 is a subgroup antigen with a high average.
2. **COPA and the outlier sum.** Built for genes activated in a fraction of tumours, which a mean
   test averages away. Both standardise by median and MAD and score the upper tail; the null is a
   sign flip of the paired differences, the same exchangeability the Wilcoxon assumes.
3. **Bimodality.** One- versus two-component Gaussian mixture BIC and the bimodality coefficient.
   (`diptest` is not installed in `neural_nmf_env`, so Hartigan's dip is not available.) With this
   many donors these are suggestive, not conclusive, and are reported beside `I2` rather than
   instead of it.
4. **Eligibility.** In what fraction of patients would at least X% of the tumour carry the
   antigen. This is the form a biomarker-selected trial is designed against, and the one that can
   be read without any of the above.

Then: do the donors fall into groups, and do those groups correspond to anything clinical — in
skin, that question has a candidate answer the blood twin lacks, because the lesion is recorded.

```python  # cell 11
# ============================================================================
# §4a  Random-effects meta-analysis across donors
# ============================================================================
SE = TH.lfc_se(long, MALIG, REACT)
re_tab = TH.meta_random_effects(M, SE, min_donors=MIN_DONORS)
re_tab["on_shortlist"] = re_tab["gene"].isin(SHORT)
print("sampling variance source:", re_tab["se_source"].iloc[0])
print(f"{len(re_tab):,} genes pooled over up to {int(re_tab['n_donors'].max())} donors\n")

print(f"median I2 = {re_tab['I2'].median():.2f}. Each donor contributes hundreds to thousands "
      "of cells, so the sampling term in the model is almost zero and nearly all the observed "
      "spread is counted as real: I2 therefore sits near 1 for almost every gene and does NOT "
      "separate them. `tau` -- the absolute between-patient SD of the effect, in log2FC units "
      "-- keeps its meaning, and the ranking below uses it.\n")

MET_COLS = ["gene", "n_donors", "mu", "se", "tau", "tau_over_mu", "I2", "pi_lo",
            "pi_hi", "frac_donors_up", "q_Q", "on_shortlist"]
het = re_tab[(re_tab["mu"] > 0.3)].sort_values("tau", ascending=False)
print("the most heterogeneous antigens with a positive pooled effect -- large on average, and "
      "not the same size in everyone:")
print(het[MET_COLS].head(25).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

con = re_tab[(re_tab["mu"] > 0.3) & (re_tab["pi_lo"] > 0)].sort_values("mu", ascending=False)
print("\nthe most CONSISTENT antigens with a positive pooled effect -- the ones whose "
      "prediction interval stays above zero, which is what 'works in every patient' means:")
print(con[MET_COLS].head(25).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(f"\n{len(con)} of {int((re_tab['mu'] > 0.3).sum())} genes with a pooled effect above 0.3 "
      "have a prediction interval that excludes zero.")

print("\n42's shortlist, ranked by how consistent it is (smallest tau first):")
print(re_tab[re_tab["on_shortlist"]].sort_values("tau")[MET_COLS]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
re_tab.to_csv(ST.tab("heterogeneity_meta"), index=False)
print("\nwrote", ST.tab("heterogeneity_meta"))
```

```
sampling variance source: delta-method from per-arm cell SD
2,355 genes pooled over up to 75 donors

median I2 = 0.99. Each donor contributes hundreds to thousands of cells, so the sampling term in the model is almost zero and nearly all the observed spread is counted as real: I2 therefore sits near 1 for almost every gene and does NOT separate them. `tau` -- the absolute between-patient SD of the effect, in log2FC units -- keeps its meaning, and the ranking below uses it.

the most heterogeneous antigens with a positive pooled effect -- large on average, and not the same size in everyone:
   gene  n_donors    mu    se   tau  tau_over_mu    I2  pi_lo  pi_hi  frac_donors_up  q_Q  on_shortlist
    CD9        75  1.19 0.182  1.57         1.32 0.987  -1.96   4.34           0.733    0          True
KIR3DL2        75 0.744 0.176  1.51         2.03 0.987  -2.29   3.78            0.36    0         False
  NTRK2        75 0.737 0.161  1.39         1.88 0.987  -2.05   3.52             0.6    0         False
 ENTPD1        75 0.335 0.161  1.37         4.09 0.992  -2.41   3.08             0.6    0         False
  EPHB6        75  0.42 0.158  1.34         3.19 0.989  -2.27   3.11           0.627    0         False
  TIMP1        75 0.308 0.148  1.27         4.13 0.995  -2.24   2.86            0.56    0         False
    APP        75 0.382 0.146  1.24         3.24  0.99   -2.1   2.87           0.587    0         False
  FGFR1        75 0.414 0.143  1.23         2.97 0.987  -2.05   2.87            0.52    0         False
  PTH2R        75 0.999 0.143  1.23         1.23 0.987  -1.46   3.46           0.667    0         False
 LGALS3        75 0.386 0.141  1.21         3.14 0.997  -2.04   2.82           0.587    0          True
   IL9R        75  0.53 0.142   1.2         2.27 0.988  -1.89   2.95            0.64    0         False
    CPM        75  1.04  0.14  1.18         1.13 0.986  -1.33   3.41           0.747    0          True
 ADAM12        75 0.415 0.135  1.16          2.8 0.987  -1.92   2.75           0.493    0         False
 PRSS21        75 0.872 0.133  1.14         1.31 0.987  -1.42   3.16           0.587    0         False
   THY1        75 0.649 0.131  1.13         1.74 0.987  -1.62   2.91           0.587    0         False
 SLC1A4        75 0.608 0.131   1.1         1.81 0.989  -1.61   2.82           0.747    0          True
 SEMA4A        75 0.328 0.125  1.05          3.2 0.983  -1.78   2.44             0.6    0         False
    MME        75 0.435 0.121  1.04          2.4 0.987  -1.66   2.53           0.453    0         False
   MRC2        75  1.07 0.124  1.04        0.969 0.985  -1.01   3.15            0.88    0         False
 SLC1A5        75 0.402 0.122  1.03         2.56  0.99  -1.66   2.47           0.667    0         False
   SGCE        75 0.811 0.119  1.02         1.26 0.987  -1.24   2.86           0.627    0         False
TSPAN13        75 0.446 0.118  1.02         2.28 0.987  -1.59   2.49           0.627    0         False
   FGL2        75 0.421 0.119 0.995         2.36 0.979  -1.58   2.42           0.627    0         False
 GALNT2        75 0.565 0.118 0.989         1.75 0.982  -1.42   2.55            0.72    0         False
 TSPAN2        75  0.51 0.117 0.983         1.93 0.978  -1.46   2.48           0.747    0          True

the most CONSISTENT antigens with a positive pooled effect -- the ones whose prediction interval stays above zero, which is what 'works in every patient' means:
Empty DataFrame
Columns: [gene, n_donors, mu, se, tau, tau_over_mu, I2, pi_lo, pi_hi, frac_donors_up, q_Q, on_shortlist]
Index: []

0 of 215 genes with a pooled effect above 0.3 have a prediction interval that excludes zero.

42's shortlist, ranked by how consistent it is (smallest tau first):
    gene  n_donors    mu     se   tau  tau_over_mu    I2   pi_lo  pi_hi  frac_donors_up       q_Q  on_shortlist
    STX6        75 0.499 0.0442 0.316        0.635  0.84  -0.138   1.14           0.893  1.17e-57          True
   VDAC1        75
... [3079 more chars truncated]
```

```python  # cell 12
# ============================================================================
# §4b  COPA and the outlier sum: up in a subset, by design
# ============================================================================
cp = TH.copa(M, pct=75, n_perm=N_PERM, min_donors=MIN_DONORS)
cp = cp.merge(re_tab[["gene", "mu", "I2"]], on="gene", how="left")
cp = cp.merge(sel[["gene", "log2fc_paired", "q"]], on="gene", how="left")
cp["on_shortlist"] = cp["gene"].isin(SHORT)

COPA_COLS = ["gene", "copa_p75", "outlier_sum", "n_outlier_donors", "frac_outlier",
             "q_copa", "q_outlier_sum", "mu", "I2", "log2fc_paired", "q",
             "on_shortlist"]
COPA_COLS = [c for c in COPA_COLS if c in cp.columns]
print("highest COPA score -- extreme in a minority of patients:")
print(cp.sort_values("copa_p75", ascending=False).head(25)[COPA_COLS]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\nthe interesting cell of the table: a high outlier score and a cohort effect that is "
      "NOT significant. These are antigens a mean test discards and a biomarker-selected trial "
      "would not:")
hid = cp[(cp["q"] >= 0.05) & (cp.get("q_copa", pd.Series(1, index=cp.index)) < 0.1)]
hid = hid.sort_values("copa_p75", ascending=False).head(25)
print(hid[COPA_COLS].to_string(index=False, float_format=lambda v: f"{v:.3g}")
      if len(hid) else "  none at these thresholds")

bm = TH.bimodality(M, min_donors=20)
bm = bm.merge(cp[["gene", "copa_p75", "frac_outlier"]], on="gene", how="left")
print("\nstrongest two-component structure in the per-donor effects "
      "(delta_bic > 0 favours two groups):")
print(bm.head(20).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(f"\n{int(bm['two_component'].sum()):,} of {len(bm):,} genes prefer two components. With "
      f"{M.shape[1]} donors that is a weak instrument -- read it with I2 and the COPA column, "
      "not alone.")

cp.to_csv(ST.tab("heterogeneity_copa"), index=False)
bm.to_csv(ST.tab("heterogeneity_bimodality"), index=False)
print("\nwrote", ST.tab("heterogeneity_copa"), "and", ST.tab("heterogeneity_bimodality"))
```

```
highest COPA score -- extreme in a minority of patients:
    gene  copa_p75  outlier_sum  n_outlier_donors  frac_outlier  q_copa  q_outlier_sum      mu    I2  log2fc_paired        q  on_shortlist
   GRIK2      86.7     5.11e+03                14         0.187  0.0294         0.0471   0.406 0.987              0 1.22e-05         False
   OPCML      27.8     3.53e+03                15           0.2  0.0294         0.0277   0.367 0.987              0  8.1e-05         False
    CD1B      22.7     1.78e+03                13         0.173   0.292              1 0.00877 0.987              0    0.247         False
   GPR37      21.3     4.71e+03                14         0.187   0.453              1   0.013 0.987              0    0.303         False
   GFRA3      16.6     1.52e+03                13         0.173   0.242           0.78  0.0144 0.987              0   0.0977         False
  LILRA6      13.3     1.36e+03                13         0.173   0.437          0.547  0.0194 0.987              0    0.182         False
  GALNT9      12.2          480                 9          0.12  0.0294          0.617   0.115 0.987              0  0.00012         False
  SORCS1      10.4          258                 9          0.12  0.0294         0.0383    0.37 0.987         0.0281 2.35e-06         False
    LIPG      10.2     1.31e+03                11         0.147   0.364          0.885  0.0326 0.987              0    0.154         False
  GRIN2D      9.77          690                10         0.133  0.0452          0.205  0.0571 0.987              0  0.00244         False
PCDHGA11      9.29          641                10         0.133  0.0294          0.694  0.0531 0.987              0 0.000891         False
  PTCHD1      8.53          629                14         0.187  0.0294         0.0471    0.41 0.987         0.0235  4.9e-07         False
    GLDN      8.16          758                15           0.2   0.216          0.162  0.0421 0.987              0   0.0179         False
 SLC26A4      6.02          927                14         0.187   0.367          0.851  0.0159 0.987              0    0.242         False
 SLC22A1      5.53          378                14         0.187   0.162          0.179  0.0736 0.987              0   0.0093         False
   WNT5A      5.43          361                11         0.147  0.0294          0.104   0.143 0.987              0 0.000518         False
  SUCNR1      5.32          652                13         0.173   0.368              1 0.00627 0.987              0    0.276         False
     SMO      5.13     1.17e+03                12          0.16  0.0852          0.291  0.0548 0.987              0   0.0203         False
  THSD7A       5.1          659                14         0.187   0.154          0.286  0.0644 0.987              0   0.0237         False
  PTGER1      5.04          485                14         0.187  0.0294         0.0383   0.123 0.987              0 0.000134         False
    CDH4      4.89          234                12          0.16  0.0294         0.0823   0.451 0.987              0 3.44e-05         False
   ENPEP      4.87          943                14         0.187  0.0294          0.065   0.112 0.987              0    0.005         False
  IGLON5      4.73     1.12e+04                18          0.24       1          0.213   0.111 0.987              0    0.517         False
 KIR3DL2       4.6            0                 0             0  0.0294              1   0.744 0.987              0 0.000947         False
   FRAS1       4.5     2.84e+03                15           0.2   0.711          0.179   0.123 0.987              0    0.182         False

the interesting cell of the table: a high outlier score and a cohort effect that is NOT significant. These are antigens a mean test discards and a biomarker-selected trial would not:
     gene  copa_p75  outlier_sum  n_outlier_donors  frac_outlier  q_copa  q_outlier_sum     mu    I2  log2fc_paired      q  on_shortlist
   PTGER
... [5215 more chars truncated]
```

```python  # cell 13
# ============================================================================
# §4c  Do the donors fall into groups, and is any of it clinical
# ============================================================================
HG = [g for g in (SHORT + het["gene"].head(40).tolist()) if g in M.index]
HG = list(dict.fromkeys(HG))
Mh = M.loc[HG].dropna(axis=1, thresh=int(0.6 * len(HG)))
labels, Z, order = TH.donor_clusters(Mh, min_size=3)
print(f"{Mh.shape[0]} genes x {Mh.shape[1]} donors -> {labels.nunique()} donor clusters "
      f"{labels.value_counts().sort_index().to_dict()}")
if labels.nunique() == 1:
    print("\n*** NO DONOR GROUPING. No cut of the dendrogram into clusters of at least 3 "
          "donors beat the silhouette criterion, so on these genes the patients do not fall "
          "into subtypes: the heterogeneity measured above is a continuum, not a set of "
          "groups. The association tests below are then trivially one-sided, and F1 still "
          "shows the per-donor structure. A split into one outlier and everyone else, which "
          "an unconstrained silhouette would have returned, is not a subtype. ***")

ann = DON.reindex(labels.index)[["dx2", "study", "sex", "age", "burden", "lesion",
                                 "lesion_class", "clin_stage_class", "is_cultured"]].copy()
ann["cluster"] = labels
print("\ndonor clusters against the clinical fields:")
for col in ["dx2", "study", "sex", "lesion", "lesion_class", "clin_stage_class"]:
    ct = pd.crosstab(ann["cluster"], ann[col])
    if ct.shape[1] < 2:
        continue
    p = stats.chi2_contingency(ct)[1] if ct.values.min() >= 0 else np.nan
    print(f"\n  {col}  (chi2 p = {p:.3g})")
    print(ct.to_string())
for col in ["age", "burden"]:
    g = [ann.loc[ann["cluster"].eq(k), col].dropna() for k in sorted(ann["cluster"].unique())]
    g = [x for x in g if len(x) >= 3]
    if len(g) >= 2:
        print(f"\n  {col}: medians {[round(float(x.median()), 3) for x in g]}  "
              f"Kruskal p = {stats.kruskal(*g).pvalue:.3g}")

print("\nper-cluster mean effect for the leading antigens -- what distinguishes the groups:")
prof = Mh.T.groupby(labels).mean().T
prof["spread"] = prof.max(1) - prof.min(1)
print(prof.sort_values("spread", ascending=False).head(20).round(3).to_string())

ann.to_csv(ST.tab("donor_clusters"))
prof.to_csv(ST.tab("donor_cluster_profiles"))
print("\nwrote", ST.tab("donor_clusters"), "and", ST.tab("donor_cluster_profiles"))
```

```
60 genes x 75 donors -> 2 donor clusters {1: 38, 2: 37}

donor clusters against the clinical fields:

  dx2  (chi2 p = 0.0128)
dx2      MF  SS  unresolved
cluster                    
1        30   6           2
2        37   0           0

  study  (chi2 p = 0.0187)
study    alkon2024  brentuximab2026  brunner2024  buus2025  chennareddy2025  gaydosik2019  gaydosik2022  herrera2021  il4ra2026  jonak2021  li2024  rindler2021_mc
cluster                                                                                                                                                           
1                0                5            3         2                4             2             3            3          3          1      10               2
2                5                1            0         0                1             1             2            0          5          0      20               2

  sex  (chi2 p = 0.0752)
sex       F   M
cluster        
1        20  18
2        11  26

  lesion  (chi2 p = 0.0109)
lesion   erythroderma  mixed  papule  patch  plaque  skin_unspecified  tumor
cluster                                                                     
1                  10      5       0      2       7                 3     11
2                   1     12       1      7       3                 5      8

  lesion_class  (chi2 p = 0.679)
lesion_class  lesion  unspecified
cluster                          
1                 35            3
2                 32            5

  clin_stage_class  (chi2 p = 0.0035)
clin_stage_class  advanced  early
cluster                          
1                       33      3
2                       22     15

  age: medians [60.5, 66.0]  Kruskal p = 0.119


  burden: medians [0.731, 0.584]  Kruskal p = 0.112

per-cluster mean effect for the leading antigens -- what distinguishes the groups:
donor_cluster      1      2  spread
gene                               
LGALS3        -0.559  1.329   1.888
APP            1.289 -0.559   1.849
NTRK2         -0.008  1.502   1.510
FGFR1          1.109 -0.301   1.410
CPM            1.689  0.349   1.340
ADAM12        -0.199  1.046   1.245
SLC1A4        -0.003  1.220   1.223
KIR3DL2        1.323  0.149   1.174
CHRNA6         1.197  0.041   1.156
SGCE           1.347  0.261   1.086
TNFRSF25       0.991 -0.083   1.075
IL9R          -0.015  1.049   1.065
SLC1A5        -0.101  0.896   0.997
ALCAM          0.884 -0.109   0.992
SEMA4A        -0.144  0.800   0.944
EPHB6          0.875 -0.050   0.925
PTH2R          1.429  0.556   0.873
ADGRE1         1.149  0.289   0.860
CD9            0.767  1.620   0.853
TSPAN13        0.025  0.877   0.852

wrote /home/projects/nyosef/zvise/MF/tables/tgt47_donor_clusters.csv and /home/projects/nyosef/zvise/MF/tables/tgt47_donor_cluster_profiles.csv
```

```python  # cell 14
# ============================================================================
# §4d F1  The donor x gene matrix with the clinical annotation beside it
# ============================================================================
# Three orderings of the same matrix. F1: donors in effect-profile order. F1b: the same, with
# the donor dendrogram drawn, so the eye can see how deep each split is. F1c: genes clustered
# into programs that rise and fall together across patients, donors by burden.
from scipy.cluster.hierarchy import dendrogram

AN = [("cluster", ann["cluster"].astype(str)), ("dx", ann["dx2"].astype(str)),
      ("lesion", ann["lesion_class"].astype(str)),
      ("stage", ann["clin_stage_class"].astype(str)), ("sex", ann["sex"].astype(str))]
NUM = [("burden", ann["burden"]), ("age", ann["age"])]
STARS = dict(zip(re_tab["gene"], re_tab["q_Q"]))
DENDRO = dict(no_labels=True, color_threshold=0, above_threshold_color="#888888")


def _strip(ax, name, vv, categorical):
    if categorical:
        cats = [c for c in pd.unique(vv.dropna()) if c != "nan"]
        cmap = dict(zip(cats, plt.get_cmap("tab20")(np.linspace(0, 1, max(len(cats), 2)))))
        ax.imshow([[1]], visible=False)
        for x, val in enumerate(vv):
            ax.add_patch(plt.Rectangle((x - .5, -.5), 1, 1,
                                       facecolor=cmap.get(val, "#eeeeee"), edgecolor="none"))
        # key, each word in its own colour, so the strip can be read without a lookup
        _x = 1.005
        for c in cats:
            ax.text(_x, 0, str(c), transform=ax.transAxes, fontsize=4.5, va="center",
                    color=cmap.get(c, "#888888"))
            _x += 0.012 + 0.009 * len(str(c))
    else:
        z = (vv - vv.min()) / (vv.max() - vv.min() or 1)
        for x, val in enumerate(z):
            ax.add_patch(plt.Rectangle((x - .5, -.5), 1, 1,
                                       facecolor=plt.get_cmap("Greys")(0.15 + 0.8 * val)
                                       if np.isfinite(val) else "#ffffff", edgecolor="none"))
        ax.text(1.005, 0, f"{vv.min():.3g}-{vv.max():.3g}", transform=ax.transAxes,
                fontsize=4.5, va="center", color="#555555")
    ax.set_xlim(-.5, len(vv) - .5); ax.set_ylim(-.5, .5)
    ax.set_yticks([0]); ax.set_yticklabels([name], fontsize=5.5)
    ax.set_xticks([]); [s.set_visible(False) for s in ax.spines.values()]


def draw_heat(Md, title, xlabel, col_Z=None, row_Z=None, row_groups=None):
    """Annotation strips over a gene x donor heatmap, with optional dendrograms. `col_Z` /
    `row_Z` must be linkages whose leaf order is already the column / row order of `Md`."""
    ordc = list(Md.columns)
    W = 0.13 * Md.shape[1] + 3.4 + (1.0 if row_Z is not None else 0)
    H = 0.17 * len(Md) + 2.6 + (0.5 if col_Z is not None else 0)
    ratios = ([2.0] if col_Z is not None else []) + [0.5] * (len(AN) + len(NUM)) \
        + [len(Md) * 0.5]
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.12, top=0.955,
                          bottom=0.10, left=(1.4 + (1.0 if row_Z is not None else 0)) / W,
                          right=0.88)
    k = 0
    if col_Z is not None:
        axd = fig.add_subplot(gs[0]); k = 1
        dendrogram(col_Z, ax=axd, **DENDRO)
        axd.set_xlim(0, 10 * len(ordc)); axd.set_axis_off()
    for i, (name, v) in enumerate(AN + NUM):
        _strip(fig.add_subplot(gs[k + i]), name, v.reindex(ordc),
               categorical=name in [a for a, _ in AN])

    ax = fig.add_subplot(gs[-1])
    TF.donor_gene_heat(ax, Md, stars_by=STARS, cbar_label="log2FC malignant / reactive")
    ax.set_xlabel(xlabel)
    if row_Z is not None:
        fig.canvas.draw()
        p = ax.get_position()
        lab = 0.75 / W                                  # room for the gene names
        if row_groups is not None:
            axs = fig.add_axes([p.x0 - lab - 0.12 / W, p.y0, 0.10 / W, p.height])
            g = row_groups.reindex(Md.index)
            cols = plt.get_cmap("tab10").colors
            for y, gi in enumerate(g):
                axs.add_patch(plt.Rectangle((0, y - .5), 1, 1, edgecolor="none",
                                            facecolor=cols[(int(gi) - 1) % 10]))
            axs.set_xlim(0, 1); axs.set_ylim(len(Md) - .5, -.5); axs.set_axis_off()
        axr = fig.add_axes([p.x0 - lab - 0.95 / W, p.y0, 0.80 / W, p.height])
        dendrogram(row_Z, ax=axr, orientation="right", **DENDRO)
        axr.set_ylim(10 * len(Md), 0); axr.set_axis_off()
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=9)
    return fig


CAP = "Shortlist + 40 most heterogeneous genes. Stars: BH heterogeneity Q."

# F1 -- donors in effect-profile order
ordc = [d for d in order if d in Mh.columns]
Md = Mh[ordc]
fig = draw_heat(Md, "Per-donor log2FC, donors clustered", "donor, effect-profile order")
TF.save(fig, "F1_donor_clustermap", caption_text=CAP)
TF.save_source(Md.reset_index(), "F1_donor_clustermap_source")
TF.logfig("F1_donor_clustermap", "donor x gene effects with clinical annotation", Md.size)

# F1b -- same order, donor dendrogram shown
fig = draw_heat(Md, "Per-donor log2FC, by sample similarity",
                "donor (average linkage, correlation distance)", col_Z=Z)
TF.save(fig, "F1b_donor_dendrogram", caption_text=CAP)
TF.logfig("F1b_donor_dendrogram", "donor x gene effects, donor dendrogram", Md.size)

# F1c -- genes clustered into co-varying programs, donors by burden
glab, gZ, gorder = TH.donor_clusters(Mh.T, min_size=3)
glab = glab.rename("program")
_b = ann["burden"].reindex(Mh.columns).sort_values()
Mg = Mh.loc[gorder, list(_b.index)]
_np = glab.nunique()
fig = draw_heat(Mg, f"Per-donor log2FC, {_np} gene programs" if _np > 1
                else "Per-donor log2FC, genes clustered (no distinct programs)",
                f"donor, ordered by malignant share of CD4 ({_b.min():.0%} → {_b.max():.0%})",
                row_Z=gZ, row_groups=glab if _np > 1 else None)
TF.save(fig, "F1c_gene_programs", caption_text=CAP + (" Left strip: gene program." if _np > 1
                                                      else " No silhouette cut with ≥3 genes "
                                                           "per program."))
TF.save_source(Mg.reset_index(), "F1c_gene_programs_source")
TF.logfig("F1c_gene_programs", "donor x gene effects, genes clustered into programs", Mg.size)

if glab.nunique() == 1:
    print("no gene grouping beat the silhouette criterion (min 3 genes per program): the "
          "dendrogram is shown, but the genes do not split into programs")
prog = glab.reindex(gorder).reset_index()
prog.columns = ["gene", "program"]
print("gene programs:")
for k_, g_ in prog.groupby("program")["gene"]:
    print(f"  {k_}: {', '.join(g_)}")
_pm = Mh.groupby(glab).mean()
print("\nper-program effect across donors:")
print(pd.DataFrame({"n_genes": glab.value_counts().sort_index(),
                    "median_lfc": _pm.median(1), "frac_donors_up_0.5": (_pm > 0.5).mean(1),
                    "rho_burden": _pm.T.corrwith(ann["burden"].reindex(_pm.columns),
                                                 method="spearman")})
      .to_string(float_format=lambda v: f"{v:.3g}"))
prog.to_csv(ST.tab("gene_programs"), index=False)
print("wrote", ST.tab("gene_programs"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F1_donor_clustermap.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F1_donor_clustermap.pdf

<Figure size 1315x1280 with 9 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt47_F1_donor_clustermap_source.csv

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F1b_donor_dendrogram.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F1b_donor_dendrogram.pdf

<Figure size 1315x1330 with 10 Axes>
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F1c_gene_programs.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F1c_gene_programs.pdf

<Figure size 1415x1280 with 11 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt47_F1c_gene_programs_source.csv
gene programs:
  1: CD70, TMEM30A, PLD1, TSPAN13, OPCML, TNFSF4, VDAC1, GPI, SYPL1, PDIA4, SLC4A2, CALU, SLC39A6, SLC5A3, P2RY11, CD151, QSOX1, CD9, IL9R, SEMA4A, NTRK2, ADAM12, SLC1A4, LGALS3, SURF4, SLC1A5, AP2S1, TIMP1, GALNT2
  2: FLT1, EPCAM, MME, EPHB6, COL6A3, MRC2, GRN, STX6, TFRC, TNFRSF25, IL6R, TNFRSF4, ADGRE1, FGL2, FGFR1, PTH2R, APP, CPM, CHRNA6, SGCE, ALCAM, FUT8, KIR3DL2, CD82, ENTPD1, GPR55, IGHM, TSPAN2, THY1, PRSS21, L1CAM

per-program effect across donors:
         n_genes  median_lfc  frac_donors_up_0.5  rho_burden
program                                                     
1             29       0.581                 0.6     -0.0935
2             31       0.638                 0.6        0.11
wrote /home/projects/nyosef/zvise/MF/tables/tgt47_gene_programs.csv
```

```python  # cell 15
# ============================================================================
# §4e F2  Eligibility: how many patients, at what tumour-positivity cut
# ============================================================================
THRESH = (0.05, 0.10, 0.25, 0.50, 0.75)
elig = TH.eligibility(long, MALIG, thresholds=THRESH, donors=PAIRED)
elig = elig.merge(re_tab[["gene", "mu", "I2"]], on="gene", how="left")
elig["on_shortlist"] = elig["gene"].isin(SHORT)
_e = ["gene", "n_donors", "median_pct_expr"] + \
     [f"frac_donors_ge{int(t * 100)}" for t in THRESH] + ["mu", "I2"]

print("42's shortlist, as a recruitment statement:")
print(elig[elig["on_shortlist"]].sort_values("frac_donors_ge25", ascending=False)[_e]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\nbest coverage at a 25% tumour-cell cut, over every screened gene:")
print(elig.sort_values("frac_donors_ge25", ascending=False).head(25)[_e]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

TOPE = elig[elig["on_shortlist"]].sort_values("frac_donors_ge25",
                                              ascending=False)["gene"].head(10).tolist()
fig, ax = plt.subplots(figsize=(5.0, 3.2))
cmap = plt.get_cmap("viridis")
for i, gene in enumerate(TOPE):
    r_ = elig[elig["gene"].eq(gene)].iloc[0]
    y = [r_[f"frac_donors_ge{int(t * 100)}"] for t in THRESH]
    ax.step(THRESH, y, where="post", lw=1.2, color=cmap(i / max(len(TOPE) - 1, 1)), label=gene)
    ax.scatter(THRESH, y, s=8, color=cmap(i / max(len(TOPE) - 1, 1)), zorder=3)
ax.set_xlabel("required fraction of a patient's malignant CD4 cells positive")
ax.set_ylabel("fraction of patients eligible")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=5.5, frameon=False, ncol=2)
ax.set_title("Patients eligible by tumour-positivity cut", loc="left", fontsize=7.5)
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F2_eligibility", caption_text=(
    f"Transcript detection, {len(PAIRED)} paired skin donors."))
TF.save_source(elig, "F2_eligibility_source")
TF.logfig("F2_eligibility", "eligibility curves for the shortlist", len(elig))
```

```
42's shortlist, as a recruitment statement:
    gene  n_donors  median_pct_expr  frac_donors_ge5  frac_donors_ge10  frac_donors_ge25  frac_donors_ge50  frac_donors_ge75    mu    I2
   SURF4        75            0.439                1             0.987             0.907             0.373              0.12 0.449 0.975
   VDAC1        75            0.494                1             0.987              0.88             0.493            0.0933 0.648 0.956
    CD82        75            0.495                1                 1             0.853             0.493            0.0667 0.543 0.984
     GPI        75            0.482            0.987             0.973             0.827             0.453              0.08  0.44 0.981
  LGALS3        75             0.48             0.96              0.88             0.773              0.48             0.253 0.386 0.997
   AP2S1        75             0.37            0.987             0.933             0.733              0.28            0.0133  0.57 0.958
   SYPL1        75              0.4            0.987              0.96             0.693             0.253                 0 0.617 0.961
TNFRSF25        75            0.328             0.92              0.88             0.613              0.28              0.04 0.462 0.987
    TFRC        75            0.294            0.987              0.96               0.6               0.2              0.04 0.758 0.954
 TNFRSF4        75             0.39            0.907             0.893               0.6             0.253            0.0533 0.203 0.994
   PDIA4        75            0.261            0.973             0.893             0.533             0.133            0.0133 0.615  0.97
     CD9        75            0.262             0.84             0.733             0.507             0.227            0.0267  1.19 0.987
    IL6R        75            0.234            0.867             0.773             0.467            0.0667                 0 0.442 0.982
  SLC5A3        75            0.209             0.92             0.773             0.453            0.0933                 0 0.619 0.954
     GRN        75            0.212            0.987             0.867             0.453             0.107                 0 0.761  0.96
    CD70        75            0.215             0.92             0.813             0.427            0.0533                 0  1.16 0.976
   CD151        75              0.2            0.893             0.827             0.427             0.107            0.0133 0.712 0.975
    FUT8        75            0.183            0.907             0.747               0.4              0.12            0.0267 0.647 0.946
 TMEM30A        75            0.193                1              0.88             0.347            0.0133                 0 0.433  0.91
    CALU        75            0.193             0.96             0.867             0.347            0.0533                 0 0.614  0.92
  SLC1A4        75            0.143            0.813             0.627             0.307            0.0533            0.0133 0.608 0.989
 SLC39A6        75            0.187            0.973              0.92             0.293            0.0667                 0  0.79 0.981
     CPM        75            0.133            0.667             0.573              0.24            0.0533            0.0133  1.04 0.986
   QSOX1        75            0.138             0.92             0.693             0.213            0.0267                 0 0.581 0.952
  SLC4A2        75            0.119            0.813              0.64               0.2            0.0267                 0 0.796 0.906
    STX6        75            0.143            0.907              0.68             0.187            0.0133                 0 0.499  0.84
  TSPAN2        75            0.133            0.787             0.653             0.187            0.0133                 0  0.51 0.978
  COL6A3        75            0.111            0.693              0.56             0.173             0.107            0.
... [3963 more chars truncated]
```

```python  # cell 16
# ============================================================================
# §4f F3  Consistency against size: which antigens are cohort-wide and which are subgroup
# ============================================================================
d = re_tab.merge(cp[["gene", "frac_outlier"]], on="gene", how="left")
d = d[d["n_donors"] >= MIN_DONORS]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axvline(0, lw=.5, c="k")
ax.plot([0, d["mu"].max()], [0, d["mu"].max()], lw=.5, c="#bbbbbb", ls="--")
ax.scatter(d["mu"], d["tau"], s=6, c="#d5d8dc", linewidths=0)
k = d["on_shortlist"]
sc = ax.scatter(d.loc[k, "mu"], d.loc[k, "tau"], s=24, c=d.loc[k, "frac_outlier"],
                cmap="magma_r", vmin=0, vmax=0.4, linewidths=0)
ax.margins(x=0.12)
_txt = [ax.text(r_["mu"], r_["tau"], r_["gene"], fontsize=5.5)
        for _, r_ in d[k].sort_values("mu", ascending=False).head(14).iterrows()]
try:
    from adjustText import adjust_text
    adjust_text(_txt, ax=ax, expand=(1.15, 1.3),
                arrowprops=dict(arrowstyle="-", lw=0.3, color="#999999"))
except ImportError:
    pass
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("fraction of donors in the outlier tail", fontsize=6)
cb.ax.tick_params(labelsize=5.5)
ax.set_xlabel("pooled effect (random-effects mu), log2FC")
ax.set_ylabel(r"$\tau$ — between-patient SD of the effect (log2FC)")
ax.set_title("Pooled effect vs between-patient spread", loc="left", fontsize=7.5)
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F3_effect_vs_heterogeneity", caption_text=(
    "Grey: all genes; coloured: 42's shortlist. Dashed: τ = μ."))
TF.save_source(d, "F3_effect_vs_heterogeneity_source")
TF.logfig("F3_effect_vs_heterogeneity", "pooled effect vs between-patient spread", len(d))

print("the quadrant that matters for a single-agent therapy -- large and consistent:")
q1 = d[(d["mu"] > 0.5) & (d["tau"] < d["tau"].median()) & (d["pi_lo"] > 0)]
print(q1.sort_values("mu", ascending=False)
      [["gene", "mu", "tau", "pi_lo", "pi_hi", "frac_donors_up", "on_shortlist"]]
      .head(25).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\nand the quadrant that argues for patient selection -- large but not in everyone:")
q2 = d[(d["mu"] > 0.5) & (d["tau"] >= d["tau"].median())]
print(q2.sort_values("mu", ascending=False)
      [["gene", "mu", "tau", "pi_lo", "pi_hi", "frac_donors_up", "frac_outlier",
        "on_shortlist"]].head(25)
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F3_effect_vs_heterogeneity.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt47_F3_effect_vs_heterogeneity.pdf

<Figure size 520x360 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt47_F3_effect_vs_heterogeneity_source.csv
the quadrant that matters for a single-agent therapy -- large and consistent:
Empty DataFrame
Columns: [gene, mu, tau, pi_lo, pi_hi, frac_donors_up, on_shortlist]
Index: []

and the quadrant that argues for patient selection -- large but not in everyone:
    gene    mu   tau   pi_lo  pi_hi  frac_donors_up  frac_outlier  on_shortlist
     CD9  1.19  1.57   -1.96   4.34           0.733        0.0267          True
    CD70  1.16 0.902  -0.647   2.97            0.92        0.0133          True
    MRC2  1.07  1.04   -1.01   3.15            0.88          0.04         False
     CPM  1.04  1.18   -1.33   3.41           0.747             0          True
   PTH2R 0.999  1.23   -1.46   3.46           0.667        0.0267         False
  COL6A3 0.966 0.829  -0.698   2.63            0.88          0.04          True
  PRSS21 0.872  1.14   -1.42   3.16           0.587             0         False
    NRN1 0.867 0.866  -0.871   2.61           0.773        0.0133         False
 SLC29A1 0.861 0.528  -0.199   1.92           0.933         0.107         False
    SGCE 0.811  1.02   -1.24   2.86           0.627             0         False
  SLC4A2 0.796 0.443 -0.0948   1.69           0.933        0.0533          True
 SLC39A6  0.79 0.864  -0.943   2.52           0.947          0.08          True
    PLD1 0.783 0.933   -1.09   2.66           0.747        0.0133         False
     GRN 0.761 0.594  -0.431   1.95           0.907        0.0267          True
    TFRC 0.758 0.499  -0.244   1.76           0.893        0.0267          True
 KIR3DL2 0.744  1.51   -2.29   3.78            0.36             0         False
   NTRK2 0.737  1.39   -2.05   3.52             0.6        0.0667         False
  ADGRE1 0.725 0.926   -1.13   2.58           0.707        0.0133         False
   CD151 0.712 0.794  -0.883   2.31            0.76        0.0133          True
SLC39A14 0.658 0.736  -0.819   2.13           0.853        0.0267         False
   QSOX2 0.653 0.715  -0.783   2.09           0.853        0.0933         False
    THY1 0.649  1.13   -1.62   2.91           0.587         0.147         False
   VDAC1 0.648 0.341 -0.0378   1.33           0.947        0.0533          True
    FUT8 0.647 0.498  -0.353   1.65           0.867        0.0133          True
    EVC2  0.63 0.628  -0.631   1.89           0.853        0.0133         False
```

```python  # cell 17
# ============================================================================
# §5  Dossier
# ============================================================================
def SEC(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def kv(k, v): print(f"  {k:36s} {v}")

SEC("1 - COHORT")
kv("paired donors", len(DON))
kv("people", DON["person"].nunique())
print()
print(DON["dx"].value_counts().to_string())
kv("\n  studies", DON["study"].nunique())
kv("donors with every model covariate", len(X))
print()
print(DON["lesion"].value_counts().to_string())

SEC("2 - WHAT WAS TESTED (TIER 1)")
for t, n in (("lesion class",
              f"{int(les.eq('patch/plaque').sum())} patch/plaque vs "
              f"{int(les.eq('tumour/eryth').sum())} tumour/erythroderma"),
             ("MF vs SS", f"{int(DON['dx2'].eq('MF').sum())} vs "
                          f"{int(DON['dx2'].eq('SS').sum())}"),
             ("stage class", DON["clin_stage_class"].value_counts().to_dict()),
             ("sex", f"{int(DON['sex'].eq('F').sum())} F / "
                     f"{int(DON['sex'].eq('M').sum())} M"),
             ("age", f"{int(DON['age'].notna().sum())} donors, median "
                     f"{DON['age'].median():.0f}"),
             ("burden", f"{int(DON['burden'].notna().sum())} donors, "
                        f"{DON['burden'].min():.0%}-{DON['burden'].max():.0%}")):
    kv(t, n)
print()
print(pd.Series({t: int((reg[f"q_{t}"] < 0.05).sum()) for t in X.columns},
                name="genes at q<0.05").to_string())

SEC("3 - HETEROGENEITY")
kv("genes pooled", len(re_tab))
kv("median I2 (saturated)", f"{re_tab['I2'].median():.2f}")
kv("median tau", f"{re_tab['tau'].median():.2f}")
kv("pooled effect > 0.3", int((re_tab["mu"] > 0.3).sum()))
kv("  of those, PI excludes zero", int(((re_tab["mu"] > 0.3) & (re_tab["pi_lo"] > 0)).sum()))
kv("donor clusters", f"{labels.nunique()} {labels.value_counts().sort_index().to_dict()}")

SEC("4 - WHAT THIS IS NOT")
for line in [
    "A treatment-effect study. Treatment at sampling is known for about half the cohort and no "
    "agent reaches tier 1; brentuximab2026 is on BV by construction.",
    "A clean lesion-stage result: the lesion is confounded with the malignant burden, and §2c "
    "is the only place the two terms compete.",
    "A response or survival analysis: those fields are near-empty in this cohort.",
    "A protein result. Every number is transcript, and skin carries no CITE-seq at all.",
    "A statement about LyP, parapsoriasis or non-CD4 CTCL: those patients are not in the "
    "cohort (42 §0b).",
]:
    print("  - " + line)

SEC("5 - OUTPUTS")
TF.write_index("figure_index")
for p in sorted(ST.TAB_DIR.glob(f"{PREFIX}*")) + sorted(ST.FIG_DIR.glob(f"{PREFIX}*")):
    print("  ", p.relative_to(NB_DIR))
```

```
==============================================================================
1 - COHORT
==============================================================================
  paired donors                        75
  people                               75

dx
MF       67
SS        6
MF/SS     2
  
  studies                           12
  donors with every model covariate    75

lesion
tumor               19
mixed               17
erythroderma        11
plaque              10
patch                9
skin_unspecified     8
papule               1

==============================================================================
2 - WHAT WAS TESTED (TIER 1)
==============================================================================
  lesion class                         19 patch/plaque vs 30 tumour/erythroderma
  MF vs SS                             67 vs 6
  stage class                          {'advanced': 55, 'early': 18}
  sex                                  31 F / 44 M
  age                                  75 donors, median 64
  burden                               75 donors, 1%-99%

is_tumour_lesion    0
is_MF               0
is_advanced         0
is_F                0
age_z               0
burden_z            0

==============================================================================
3 - HETEROGENEITY
==============================================================================
  genes pooled                         2355
  median I2 (saturated)                0.99
  median tau                           0.26
  pooled effect > 0.3                  215
    of those, PI excludes zero         0
  donor clusters                       2 {1: 38, 2: 37}

==============================================================================
4 - WHAT THIS IS NOT
==============================================================================
  - A treatment-effect study. Treatment at sampling is known for about half the cohort and no agent reaches tier 1; brentuximab2026 is on BV by construction.
  - A clean lesion-stage result: the lesion is confounded with the malignant burden, and §2c is the only place the two terms compete.
  - A response or survival analysis: those fields are near-empty in this cohort.
  - A protein result. Every number is transcript, and skin carries no CITE-seq at all.
  - A statement about LyP, parapsoriasis or non-CD4 CTCL: those patients are not in the cohort (42 §0b).

==============================================================================
5 - OUTPUTS
==============================================================================
wrote /home/projects/nyosef/zvise/MF/tables/tgt47_figure_index.csv
   tables/tgt47_F1_donor_clustermap_source.csv
   tables/tgt47_F1c_gene_programs_source.csv
   tables/tgt47_F2_eligibility_source.csv
   tables/tgt47_F3_effect_vs_heterogeneity_source.csv
   tables/tgt47_donor_cluster_profiles.csv
   tables/tgt47_donor_clusters.csv
   tables/tgt47_figure_index.csv
   tables/tgt47_gene_programs.csv
   tables/tgt47_heterogeneity_bimodality.csv
   tables/tgt47_heterogeneity_copa.csv
   tables/tgt47_heterogeneity_meta.csv
   tables/tgt47_llm_export.md
   tables/tgt47_power_table.csv
   tables/tgt47_strat_disease.csv
   tables/tgt47_strat_lesion.csv
   tables/tgt47_strat_regression.csv
   tables/tgt47_strat_stage.csv
   tables/tgt47_tier2_descriptive.csv
   tables/tgt47_treatment_counts.csv
   figures/targets/tgt47_F1_donor_clustermap.pdf
   figures/targets/tgt47_F1_donor_clustermap.png
   figures/targets/tgt47_F1b_donor_dendrogram.pdf
   figures/targets/tgt47_F1b_donor_dendrogram.png
   figures/targets/tgt47_F1c_gene_programs.pdf
   figures/targets/tgt47_F1c_gene_programs.png
   figures/targets/tgt47_F2_eligibility.pdf
   figures/targets/tgt47_F2_eligibility.png
   figures/targets/tgt47_F3_effect_vs_heterogeneity.pdf
   figures/targets/tgt47_F3_effect_vs_heterogeneity.png
```

```python  # cell 18
# ============================================================================
# §6  The complete numerical result, as one block to hand to another model
# ============================================================================
_ctx = f"""
Compartment: MF/CTCL **skin**. This notebook stratifies the malignant-vs-reactive CD4 antigen
axis that notebook 42 established, and measures how much it varies between patients. It derives
no new screen: it reads 42's cleaned cohort and tables.

Cohort: {len(DON)} donors ({DON['person'].nunique()} people, {DON['study'].nunique()} studies)
carrying BOTH a malignant and a reactive CD4 arm of >= 20 cells. Non-CD4 clones, diagnoses
outside the MF/SS spectrum and repeat biopsies of one patient were already removed in 42 §0b.
Diagnosis from the curated `clin_dx`.

Per-donor effect: for each gene and donor, log2((mean malignant + 0.05)/(mean reactive + 0.05))
on log1p CP10K. Everything below is a function of that donor x gene matrix.

Stratified tests: the paired Wilcoxon is rerun inside each stratum, and `p_interaction` compares
the per-donor effects between strata (BH-adjusted as `q_interaction`). Per-gene OLS regresses the
donor effect on lesion class, diagnosis, stage, sex, age and burden at once.

Heterogeneity: DerSimonian-Laird random effects over donors (`mu` pooled effect, `tau`
between-patient SD, `I2`, `pi_lo`/`pi_hi` prediction interval, `q_Q` BH-adjusted Q test); COPA and
the outlier sum with a {N_PERM}-draw sign-flip null; two-component Gaussian mixture BIC;
hierarchical donor clustering with a silhouette cut and a minimum of 3 donors per cluster.

What is special about the skin cohort relative to the blood one (notebook 44): the lesion that was
biopsied is recorded (patch / plaque / tumour / erythroderma), and stage is testable here
(early vs advanced) where blood had 2 early-stage donors. The lesion is confounded with the
malignant burden -- a tumour holds a larger clone than a patch -- and that confounding is measured
rather than assumed.
"""

_s2b = [
    {"name": f"\u00a72b {_lab}",
     "what": "each leading antigen's malignant-vs-reactive effect inside each level of "
             f"{_strat2b[_k][0]}, and whether the levels differ.",
     "how": "paired Wilcoxon within each stratum; delta_lfc and q_interaction compare the "
            "per-donor effects between strata.",
     "table": _strat2b[_k][1][_strat2b[_k][1]["on_shortlist"]],
     "file": str(ST.tab(_k).relative_to(NB_DIR))}
    for _k, _lab in (("strat_disease", "Disease, MF vs SS"),
                     ("strat_stage", "Stage class, early vs advanced"))
    if _k in _strat2b
]

_sections = [
    {"name": "§1 Power table",
     "what": "coverage of every candidate stratification over the paired donors.",
     "how": "count of non-missing values per field, split by diagnosis; `levels` gives the "
            "value counts or the numeric range.",
     "table": power, "file": str(ST.tab("power_table").relative_to(NB_DIR))},
    {"name": "§1 Treatment at sampling",
     "what": "donors on each treatment class at the time of sampling, and in which studies.",
     "how": "membership of the curated clin_current_tx_classes string. A donor can be on "
            "several.", "table": tx},
    {"name": "§2a Lesion class, burden confounding",
     "what": "the malignant share of the CD4 compartment inside each lesion class -- the "
             "confounder of the lesion contrast.",
     "how": "burden = malignant cells / (malignant + reactive) per donor.",
     "table": DON.groupby(les)["burden"].describe(percentiles=[.25, .5, .75])},
    {"name": "§2a Lesion class, 42's shortlist",
     "what": "each leading antigen's malignant-vs-reactive effect inside patch/plaque and "
             "inside tumour/erythroderma, and whether the two differ.",
     "how": "paired Wilcoxon within each stratum; delta_lfc and q_interaction compare the "
            "per-donor effects between strata.",
     "table": (les_tab[les_tab["on_shortlist"]][LES_COLS] if len(SHORT)
               else les_tab.head(0)),
     "file": str(ST.tab("strat_lesion").relative_to(NB_DIR))},
    *_s2b,
    {"name": "§2c Donor-level regression",
     "what": "which donor properties move each gene's effect, with all terms in one model.",
     "how": "OLS per gene over donors, terms " + ", ".join(X.columns) + "; BH per term. The "
            "term correlation matrix is given because lesion and burden share variance.",
     "table": pd.Series({t: int((reg[f"q_{t}"] < 0.05).sum()) for t in X.columns},
                        name="genes_at_q_lt_0.05"),
     "file": str(ST.tab("strat_regression").relative_to(NB_DIR))},
    {"name": "§2c Term correlations", "what": "how far the model's terms overlap.",
     "how": "Pearson between the donor-level covariates.", "table": X.corr()},
    {"name": "§3 Tier 2 descriptive splits",
     "what": "effect sizes on each side of the splits too small to test -- CD30, large-cell "
             "transformation, folliculotropism, B stage, individual treatments.",
     "how": "median per-donor effect per side with a 400-draw bootstrap CI. No p-value, no "
            "correction.",
     "table": t2, "file": str(ST.tab("tier2_descriptive").relative_to(NB_DIR)),
     "max_rows": 60},
    {"name": "§4a Random-effects meta-analysis, 42's shortlist",
     "what": "pooled effect and between-patient spread for each leading antigen.",
     "how": "DerSimonian-Laird; `pi_lo > 0` is what 'works in every patient' means. I2 "
            "saturates near 1 because each donor contributes many cells, so rank on tau.",
     "table": (re_tab[re_tab["on_shortlist"]].sort_values("tau")[MET_COLS]
               if len(SHORT) else re_tab.head(0)),
     "file": str(ST.tab("heterogeneity_meta").relative_to(NB_DIR))},
    {"name": "§4a Most consistent antigens cohort-wide",
     "what": "genes with a pooled effect above 0.3 whose prediction interval excludes zero.",
     "how": "same model, ranked by mu.", "table": con[MET_COLS], "max_rows": 40},
    {"name": "§4b COPA, subgroup antigens",
     "what": "genes extreme in a minority of patients, which a mean test averages away.",
     "how": "median/MAD-standardised 75th percentile, sign-flip null, BH.",
     "table": cp.sort_values("copa_p75", ascending=False).head(40)[
         [c for c in ("gene", "copa_p75", "frac_outlier", "q_copa", "mu", "log2fc_paired", "q",
                      "on_shortlist") if c in cp.columns]],
     "file": str(ST.tab("heterogeneity_copa").relative_to(NB_DIR))},
    {"name": "§4b Bimodality of the per-donor effect",
     "what": "whether a gene's donor effects fall into two groups rather than one spread.",
     "how": "one- versus two-component Gaussian mixture BIC; delta_bic > 0 favours two "
            "components. A weak instrument at this donor count -- read it beside I2 and COPA.",
     "table": bm.head(25), "file": 'tables/tgt47_heterogeneity_bimodality.csv'},
    {"name": "§4c Donor clusters against the clinical fields",
     "what": "whether the patients fall into groups on these antigens, and whether the groups "
             "mean anything clinically.",
     "how": "hierarchical clustering of the donor x gene effect matrix, silhouette cut, "
            "minimum 3 donors per cluster; chi-square against each categorical field.",
     "table": ann, "file": str(ST.tab("donor_clusters").relative_to(NB_DIR)), "max_rows": 100},
    {"name": "§4c Per-cluster mean effect for the leading antigens",
     "what": "what separates the donor clusters, where there are any.",
     "how": "mean per-donor effect inside each cluster; `spread` is max minus min across "
            "clusters.",
     "table": prof.sort_values("spread", ascending=False).head(25), "file": 'tables/tgt47_donor_cluster_profiles.csv'},
    {"name": "§4e Eligibility",
     "what": "the fraction of patients in whom at least X% of malignant CD4 cells carry the "
             "antigen.", "how": "transcript detection per donor, thresholded at 5/10/25/50/75%.",
     "table": elig[elig["on_shortlist"]][_e] if len(SHORT) else elig.head(0),
     "file": str(ST.tab("F2_eligibility_source").relative_to(NB_DIR))},
    {"name": "§4f Large and consistent vs large and selective",
     "what": "the two quadrants a development decision turns on.",
     "how": "mu > 0.5 split on whether tau is below the cohort median and the prediction "
            "interval excludes zero.",
     "table": pd.concat([q1.assign(quadrant="large_and_consistent"),
                         q2.assign(quadrant="large_but_variable")])
                [["gene", "quadrant", "mu", "tau", "pi_lo", "pi_hi", "frac_donors_up",
                  "on_shortlist"]], "max_rows": 50},
]
TF.llm_export("47", "Skin stratification and heterogeneity", _ctx, _sections)
```

```
======================== COPY FROM HERE ========================

# 47 — Skin stratification and heterogeneity

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**. This notebook stratifies the malignant-vs-reactive CD4 antigen
axis that notebook 42 established, and measures how much it varies between patients. It derives
no new screen: it reads 42's cleaned cohort and tables.

Cohort: 75 donors (75 people, 12 studies)
carrying BOTH a malignant and a reactive CD4 arm of >= 20 cells. Non-CD4 clones, diagnoses
outside the MF/SS spectrum and repeat biopsies of one patient were already removed in 42 §0b.
Diagnosis from the curated `clin_dx`.

Per-donor effect: for each gene and donor, log2((mean malignant + 0.05)/(mean reactive + 0.05))
on log1p CP10K. Everything below is a function of that donor x gene matrix.

Stratified tests: the paired Wilcoxon is rerun inside each stratum, and `p_interaction` compares
the per-donor effects between strata (BH-adjusted as `q_interaction`). Per-gene OLS regresses the
donor effect on lesion class, diagnosis, stage, sex, age and burden at once.

Heterogeneity: DerSimonian-Laird random effects over donors (`mu` pooled effect, `tau`
between-patient SD, `I2`, `pi_lo`/`pi_hi` prediction interval, `q_Q` BH-adjusted Q test); COPA and
the outlier sum with a 1000-draw sign-flip null; two-component Gaussian mixture BIC;
hierarchical donor clustering with a silhouette cut and a minimum of 3 donors per cluster.

What is special about the skin cohort relative to the blood one (notebook 44): the lesion that was
biopsied is recorded (patch / plaque / tumour / erythroderma), and stage is testable here
(early vs advanced) where blood had 2 early-stage donors. The lesion is confounded with the
malignant burden -- a tumour holds a larger clone than a patch -- and that confounding is measured
rather than assumed.

## §1 Power table

**What:** coverage of every candidate stratification over the paired donors.

**How:** count of non-missing values per field, split by diagnosis; `levels` gives the value counts or the numeric range.

**Full table on disk:** `tables/tgt47_power_table.csv`

```csv
field,n,MF,SS,levels
dx2 (MF vs SS),75,67,6,"MF:67, SS:6, unresolved:2"
lesion,75,67,6,"tumor:19, mixed:17, erythroderma:11, plaque:10, patch:9, skin_unspecified:8"
lesion class,75,67,6,"lesion:67, unspecified:8"
sex,75,67,6,"M:44, F:31"
age,75,67,6,"median 64, range 30-87"
stage,73,67,4,"IIB:30, IB:11, IVA1:8, IA:7, IVB:4, IVA2:4"
stage class,73,67,4,"advanced:55, early:18"
B stage,57,51,4,"B0:40, B2:9, B1:5, B2b:3"
CD30,17,17,0,"pos:9, neg:8"
large-cell transformation,32,32,0,"yes:23, no:9"
folliculotropic,5,5,0,yes:5
timepoint,75,67,6,"single:60, pre:5, baseline:5, serial:4, on_treatment:1"
treatment at sampling known,51,43,6,"median 1, range 0-1"
treatment-naive,58,50,6,"no:46, yes:6, washout_6mo:6"
prior lines,0,0,0,
response,7,6,1,"PD:3, cleared:2, NR:1, SD:1"
outcome,19,19,0,"alive:13, DOD:5, died_unspecified:1"
malignant burden,75,67,6,"median 0.658, range 0.00837-0.985"
```

## §1 Treatment at sampling

**What:** donors on each treatment class at the time of sampling, and in which studies.

**How:** membership of the curated clin_current_tx_classes string. A donor can be on several.

```csv
index,donors,MF,SS,studies
ecp,7,5,2,"brunner2024, buus2025, chennareddy2025, rindler2021_mc"
retinoid,4,3,1,"chennareddy2025, gaydosik2019, il4ra2026"
interferon,2,2,0,"brunner2024, chennareddy2025"
methotrexate,2,2,0,il4ra2026
topical_steroid,2,1,
... [29993 more chars truncated]
```
