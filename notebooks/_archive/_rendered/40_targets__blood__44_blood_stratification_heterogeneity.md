# 40_targets/blood/44_blood_stratification_heterogeneity.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/blood/44_blood_stratification_heterogeneity.ipynb` and is never run._

# 44 — Blood: stratification and heterogeneity of the malignant-vs-reactive CD4 axis

43 asks whether an antigen separates tumour from reactive CD4 **on average**. That average is
made of 52 patients with different diseases, different treatments, different ages and very
different tumour burdens, and CTCL is heterogeneous enough that an average can describe nobody.
This notebook asks the two questions that follow from it:

1. **Does the effect depend on who the patient is?** — MF vs Sézary, treatment at sampling,
   sex, age, tumour burden.
2. **Is the effect the same size in everyone, or large in a subset?** — random-effects
   heterogeneity, the subgroup statistics built for exactly this (COPA, the outlier sum),
   bimodality, donor clustering, and the eligibility curve a biomarker-selected trial is
   designed against.

**Power first.** The cohort is 52 independent paired donors: 37 SS, 8 MF, 5 unresolved, 2
MF/SS. Treatment is almost entirely confounded with study, there are 2 early-stage donors
against 35 advanced, and response and survival are essentially absent. §1 prints the counts
before anything is tested, and every contrast is placed in one of three tiers:

| tier | meaning |
|---|---|
| **1** | enough donors to test, and the test is run |
| **2** | enough to describe, with effect sizes and no claim |
| **3** | not testable — reported as a count, so the gap is visible rather than silently absent |

Nothing here re-derives the screen. It reads the cleaned cohort and the rebuilt tables that 43
produces, so **run 43 first** (or at least `jobs/run_target_screen_qc.sh --tissue blood`).

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

CM.use_tissue("blood")
MAL.use_tissue("blood")
TQ.use_tissue("blood")           # 42/47/48 set it to skin, and the module keeps the last value

TISSUE = "blood"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt44_"
MALIG, REACT, HC_CD4 = "CD4_malignant", "CD4_reactive", "CD4_HC"
MIN_CELLS, MIN_DONORS, DETECT_FLOOR = 20, 5, 0.10
N_PERM = 1000      # sign-flip permutations for the COPA / outlier-sum null

ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX
TF.rc()
pd.set_option("display.width", 240)

don = TC.blood_donor_table()
long = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet"), don)
meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)
man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
assert man.get("builder") == "run_target_screen_qc.py", (
    "run jobs/run_target_screen_qc.sh --tissue blood first -- this notebook needs sd_all for "
    "the random-effects model and mean_all on the true library")
bur = TC.burden(long)
don = don.join(bur, how="left")

paired = meta[meta["arm"].isin([MALIG, REACT])].groupby("donor")["arm"].nunique().eq(2)
PAIRED = sorted(paired.index[paired])
DON = don.loc[PAIRED]
DONOR_STUDY = dict(zip(meta["donor"], meta["study"]))

sel = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
prev = TS.prevalence(long, MALIG, detect_floor=DETECT_FLOOR)

# The gene set everything below runs on. 43's shortlist is the headline, but restricting the
# heterogeneity analysis to it would answer only "are my winners consistent" -- a gene that is
# flat on average because it is strongly up in a third of patients is invisible in that list
# and is exactly what this notebook exists to find. So: every gene that reaches the paired
# test, with the shortlist marked.
SHORT = pd.read_csv(ST.TAB_DIR / "tgt43_shortlist_blood.csv")["gene"].tolist() \
    if (ST.TAB_DIR / "tgt43_shortlist_blood.csv").exists() else []
GENES = sel["gene"].tolist()
print(f"NB_DIR = {NB_DIR}")
print(f"{len(PAIRED)} paired donors | {len(GENES):,} genes tested | "
      f"{len(SHORT)} on 43's shortlist")
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
52 paired donors | 1,098 genes tested | 30 on 43's shortlist
```

## §1 — What this cohort can and cannot answer

Read the table before the sections. Every row is a candidate stratification; the counts decide
which tier it lands in, and the tier decides whether the result below is a test, a description
or an absence.

Two structural facts to keep in view:

- **Treatment is confounded with study.** ECP is 25 donors, but they are concentrated in buus
  and ren; bexarotene in ren is never given without ECP. The ECP contrast is therefore run
  study-blocked (van Elteren), and the within-study comparison is printed separately so the
  confounding is visible rather than averaged over.
- **Burden is the only covariate with complete coverage.** `clin_blood_tumor_pct` is curated for
  ~20 donors; the malignant share of the CD4 compartment is computable for all of them and is
  the better continuous axis. It is also the one 43 used as the contamination test, so a marker
  whose effect rides on burden should be read twice.

```python  # cell 3
# ============================================================================
# §1  The power table
# ============================================================================
FIELDS = {
    "dx2 (MF vs SS)": "dx2", "sex": "sex", "age": "age",
    "stage": "clin_stage", "stage class": "clin_stage_class", "B stage": "clin_B",
    "blood tumour %": "blood_tumor_pct", "treatment at sampling known": "tx_known",
    "treatment-naive": "clin_tx_naive", "prior lines": "n_prior_lines",
    "response": "clin_response", "outcome": "clin_outcome",
    "drug sensitivity": "clin_drug_sensitivity", "malignant burden": "burden",
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
        rec["levels"] = ", ".join(f"{k}:{c}" for k, c in v.value_counts().head(5).items())
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
print(tx.to_string())

print("\nECP against study -- the confounding, as a table rather than a sentence:")
print(pd.crosstab(DON["study"], DON["tx_ecp"], dropna=False).to_string())

TIERS = {
    1: ["dx2 (MF vs SS)", "treatment: ECP", "sex", "age", "malignant burden"],
    2: ["treatment: HDACi", "treatment: interferon", "treatment: retinoid", "B stage",
        "prior lines"],
    3: ["stage (2 early vs 35 advanced)", "treatment-naive", "treatment: mogamulizumab",
        "bexarotene (never separable from ECP)", "response", "outcome", "survival"],
}
print("\n" + "=" * 78)
for t, items in TIERS.items():
    print(f"TIER {t}: " + "; ".join(items))
print("=" * 78)
power.to_csv(ST.tab("power_table"), index=False)
tx.to_csv(ST.tab("treatment_counts"))
print("wrote", ST.tab("power_table"), "and", ST.tab("treatment_counts"))
```

```
coverage over the 52 paired donors:
                      field  n  MF  SS                                         levels
             dx2 (MF vs SS) 52   8  37                      SS:37, MF:8, unresolved:7
                        sex 52   8  37                                     M:30, F:22
                        age 44   7  31                         median 70, range 32-85
                      stage 36   6  24           IVA1:13, IVA:9, IVB:4, IIB:3, IVA2:3
                stage class 36   6  24                           advanced:34, early:2
                    B stage 25   6  17                             B2:13, B1:11, B0:1
             blood tumour % 20   5  11                         median 59, range 13-98
treatment at sampling known 44   8  34                            median 1, range 0-1
            treatment-naive 44   8  31                                   no:40, yes:4
                prior lines  4   0   0                            median 2, range 0-4
                   response  3   1   0                                  mixed:2, PD:1
                    outcome  4   1   1        DOD:2, died_unspecified:1, died_other:1
           drug sensitivity  4   0   0 romidepsin_resistant:3, romidepsin_sensitive:1
           malignant burden 52   8  37                 median 0.858, range 0.05-0.994

treatment at sampling (a donor can be on several):
                  donors  MF  SS                                                           studies
ecp                   25   5  20  borcherding2019, borcherding2023, buus2025, herrera2021, ren2023
retinoid              14   4  10                                borcherding2023, buus2025, ren2023
interferon             6   1   5                                                 buus2025, ren2023
hdaci                  6   1   4   borcherding2019, borcherding2023, buus2025, geskin2026, ren2023
systemic_steroid       2   0   2                                                          buus2025
mogamulizumab          1   0   1                                                          buus2025
methotrexate           1   0   1                                                          buus2025
other_targeted         1   0   1                                                      gaydosik2022
alemtuzumab            1   0   1                                                          buus2025

ECP against study -- the confounding, as a table rather than a sentence:
tx_ecp           False  True 
study                        
borcherding2019      0      1
borcherding2023      1      2
buus2025             4     11
dorando2026          5      0
gaydosik2022         3      0
geskin2026           5      0
harro2023            3      0
herrera2021          5      1
ren2023              0     10
rindler2021_fi       1      0

==============================================================================
TIER 1: dx2 (MF vs SS); treatment: ECP; sex; age; malignant burden
TIER 2: treatment: HDACi; treatment: interferon; treatment: retinoid; B stage; prior lines
TIER 3: stage (2 early vs 35 advanced); treatment-naive; treatment: mogamulizumab; bexarotene (never separable from ECP); response; outcome; survival
==============================================================================
wrote /home/projects/nyosef/zvise/MF/tables/tgt44_power_table.csv and /home/projects/nyosef/zvise/MF/tables/tgt44_treatment_counts.csv
```

## §2 — Tier 1: the contrasts with enough donors to test

Each contrast is the same paired within-donor test as 43, run inside each stratum, plus a
two-sample comparison of the per-donor effects between strata (`p_interaction`). That
interaction is the question — not "is the antigen up in MF" but "is it *more* up in MF than in
SS" — and with 8 MF donors it is a consistency check, not a finding. The n for each side is
printed on every row so it cannot be read as more than that.

```python  # cell 5
# ============================================================================
# §2a  Disease: MF vs Sezary
# ============================================================================
dx_map = DON.loc[DON["dx2"].isin(["MF", "SS"]), "dx2"]
print(f"MF {int(dx_map.eq('MF').sum())} donors vs SS {int(dx_map.eq('SS').sum())} donors "
      f"({len(DON) - len(dx_map)} unresolved or MF/SS, excluded from this contrast)")

dxs = TS.strat_selectivity(long, MALIG, REACT, dx_map, value="mean_all", min_donors=3)
wide = dxs.pivot_table(index="gene", columns="stratum",
                       values=["log2fc_paired", "n_donors_paired", "frac_donors_up"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
inter = (dxs[["gene", "delta_lfc", "p_interaction", "q_interaction"]]
         .drop_duplicates("gene").set_index("gene"))
dx_tab = wide.join(inter).join(sel.set_index("gene")[["log2fc_paired", "q"]]
                               .rename(columns={"log2fc_paired": "log2fc_all",
                                                "q": "q_all"}))
dx_tab = dx_tab.reset_index()
dx_tab["on_shortlist"] = dx_tab["gene"].isin(SHORT)

sig = dx_tab[dx_tab["q_interaction"] < 0.05].sort_values("delta_lfc")
print(f"\ngenes whose effect differs between MF and SS at q < 0.05: {len(sig)}")
_c = ["gene", "log2fc_paired_MF", "n_donors_paired_MF", "log2fc_paired_SS",
      "n_donors_paired_SS", "delta_lfc", "q_interaction", "log2fc_all", "on_shortlist"]
print(sig[_c].to_string(index=False, float_format=lambda v: f"{v:.3g}")
      if len(sig) else "  none -- with 8 MF donors this is the expected result, and it is not "
      "evidence that the two diseases behave the same")

print("\nthe shortlist, split by disease (no claim; read the two n columns):")
print(dx_tab[dx_tab["on_shortlist"]].sort_values("log2fc_all", ascending=False)[_c]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# Within-study, where the two diseases actually sit side by side. Everywhere else, disease and
# study are the same variable.
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

dx_tab.to_csv(ST.tab("strat_disease"), index=False)
print("\nwrote", ST.tab("strat_disease"))
```

```
MF 8 donors vs SS 37 donors (7 unresolved or MF/SS, excluded from this contrast)


genes whose effect differs between MF and SS at q < 0.05: 0
  none -- with 8 MF donors this is the expected result, and it is not evidence that the two diseases behave the same

the shortlist, split by disease (no claim; read the two n columns):
    gene  log2fc_paired_MF  n_donors_paired_MF  log2fc_paired_SS  n_donors_paired_SS  delta_lfc  q_interaction  log2fc_all  on_shortlist
 KIR3DL2              3.44                   8              2.64                  37     -0.794              1        2.79          True
    CD70              1.83                   8              1.93                  37     0.0976              1        1.93          True
     TPO              1.19                   8              1.91                  37      0.716          0.992        1.85          True
   TIGIT              1.67                   8              1.83                  37      0.162              1        1.65          True
    STAM              1.27                   8              1.69                  37      0.423          0.992        1.64          True
   FCRL3              1.66                   8              1.43                  37     -0.236              1        1.54          True
   GPR15             0.814                   8              1.77                  37       0.96              1        1.44          True
   PDCD1              1.28                   8              1.14                  37     -0.146              1        1.22          True
   NRROS                 1                   8              1.18                  37      0.179          0.992         1.2          True
    PI16             0.898                   8              1.11                  37      0.211          0.992        1.17          True
   ITGB1              1.08                   8              1.16                  37     0.0835              1        1.16          True
    CCR4              1.63                   8               1.1                  37     -0.536          0.992        1.13          True
    CD84              0.91                   8             0.843                  37    -0.0671              1        1.07          True
   EPHA4              0.89                   8                 1                  37      0.115              1       0.989          True
  TSPAN2             0.938                   8              1.02                  37     0.0815              1       0.938          True
   ABCC1              0.77                   8              1.01                  37      0.242              1       0.907          True
     MBP             0.685                   8             0.908                  37      0.222              1       0.843          True
  GPR146             0.489                   8             0.855                  37      0.366          0.975       0.824          True
    FUT8             0.788                   8             0.753                  37    -0.0358              1       0.793          True
  SLAMF6             0.629                   8             0.811                  37      0.182              1       0.771          True
    CD82              1.02                   8             0.684                  37     -0.332              1       0.765          True
    TFRC             0.546                   8              0.79                  37      0.244          0.975       0.746          True
    IL6R             0.421                   8             0.785                  37      0.364          0.975       0.738          True
   ANXA2             0.722                   8             0.727                  37    0.00511              1       0.706          True
TNFRSF1B             0.821                   8             0.344                  37     -0.476          0.995       0.654          True
   SIRPG             0.695                   8             0.588                  37     -0.108              
... [1432 more chars truncated]
```

```python  # cell 6
# ============================================================================
# §2b  Treatment at sampling: ECP, the only agent with two usable arms
# ============================================================================
# Extracorporeal photopheresis. 25 of the paired donors are on it, 20 are known not to be, and
# the split is almost a study split -- so the headline test is the study-blocked one
# (van Elteren), not the pooled one. Both are printed.
ecp_known = DON[DON["tx_known"]]
ecp_map = ecp_known["tx_ecp"].map({True: "ECP", False: "no ECP"})
print(f"{int((ecp_map == 'ECP').sum())} on ECP vs {int((ecp_map == 'no ECP').sum())} not, "
      f"{len(DON) - len(ecp_map)} with treatment unknown")

ecp = TS.strat_selectivity(long, MALIG, REACT, ecp_map, value="mean_all", min_donors=3)
ew = ecp.pivot_table(index="gene", columns="stratum",
                     values=["log2fc_paired", "n_donors_paired"])
ew.columns = [f"{a}_{b}".replace(" ", "_") for a, b in ew.columns]
ew = ew.join(ecp[["gene", "delta_lfc", "p_interaction", "q_interaction"]]
             .drop_duplicates("gene").set_index("gene")).reset_index()
ew["on_shortlist"] = ew["gene"].isin(SHORT)

sig = ew[ew["q_interaction"] < 0.05].sort_values("delta_lfc")
print(f"\ngenes whose malignant-vs-reactive effect differs by ECP status at q < 0.05: "
      f"{len(sig)}")
_c = ["gene", "log2fc_paired_ECP", "n_donors_paired_ECP", "log2fc_paired_no_ECP",
      "n_donors_paired_no_ECP", "delta_lfc", "q_interaction", "on_shortlist"]
print(sig[_c].head(40).to_string(index=False, float_format=lambda v: f"{v:.3g}")
      if len(sig) else "  none")

print("\nthe shortlist by ECP status:")
print(ew[ew["on_shortlist"]][_c].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\n" + "=" * 78)
print("study-blocked version of the same contrast (van Elteren), which is the one to quote:")
blk = TS.blocked_selectivity(
    long[long["donor"].isin(ecp_known.index)].assign(
        arm=lambda d: d["arm"].where(~d["arm"].eq(MALIG),
                                     d["donor"].map(ecp_map).radd("MALIG_"))),
    "MALIG_ECP", "MALIG_no ECP", block=DONOR_STUDY, value="mean_all", min_per_arm=3)
print(f"{len(blk):,} genes testable with >=3 donors per arm per study block, "
      f"{int((blk['q_blocked'] < 0.05).sum())} at q < 0.05")
print(blk.sort_values("z_blocked").head(15).to_string(index=False,
                                                      float_format=lambda v: f"{v:.3g}"))
print("...")
print(blk.sort_values("z_blocked", ascending=False).head(15)
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\n*** This compares the malignant arm of ECP-treated donors against the malignant arm "
      "of untreated donors, which is a between-patient comparison and loses the within-donor "
      "pairing. Bexarotene cannot be separated from ECP anywhere in this cohort (every ren "
      "donor on bexarotene is also on ECP), so 'ECP' here means 'ECP, usually with a "
      "retinoid'. ***")
ew.to_csv(ST.tab("strat_ecp"), index=False)
blk.to_csv(ST.tab("strat_ecp_blocked"), index=False)
print("\nwrote", ST.tab("strat_ecp"), "and", ST.tab("strat_ecp_blocked"))
```

```
25 on ECP vs 19 not, 8 with treatment unknown


genes whose malignant-vs-reactive effect differs by ECP status at q < 0.05: 0
  none

the shortlist by ECP status:
    gene  log2fc_paired_ECP  n_donors_paired_ECP  log2fc_paired_no_ECP  n_donors_paired_no_ECP  delta_lfc  q_interaction  on_shortlist
   ABCC1              0.946                   25                  1.01                      19    -0.0657          0.946          True
   ANXA2              0.756                   25                 0.711                      19     0.0447          0.961          True
    CCR4               1.19                   25                  1.25                      19    -0.0693          0.949          True
    CD28              0.706                   25                 0.607                      19     0.0989          0.987          True
    CD70               1.84                   25                  2.06                      19     -0.216          0.956          True
    CD82              0.811                   25                 0.626                      19      0.185          0.949          True
    CD84               1.05                   25                  0.84                      19       0.21          0.949          True
   CTLA4              0.733                   25                 0.587                      19      0.146          0.957          True
   EPHA4              0.975                   25                     1                      19    -0.0281              1          True
   FCRL3               1.88                   25                 0.976                      19      0.903          0.788          True
    FUT8              0.629                   25                 0.968                      19     -0.339          0.778          True
  GPR146              0.791                   25                 0.794                      19   -0.00297              1          True
   GPR15               2.26                   25                  1.08                      19       1.18          0.841          True
     GRN              0.627                   25                 0.651                      19    -0.0235          0.988          True
    IL6R              0.488                   25                 0.785                      19     -0.297          0.861          True
   ITGB1               1.16                   25                  1.16                      19   0.000303              1          True
 KIR3DL2               2.85                   25                  2.62                      19      0.233          0.999          True
     MBP              0.802                   25                 0.877                      19    -0.0758          0.952          True
   NRROS               1.08                   25                  1.25                      19     -0.179          0.895          True
   PDCD1               1.13                   25                  1.42                      19     -0.287          0.865          True
    PI16               1.09                   25                  1.25                      19     -0.159          0.961          True
   SIRPG              0.739                   25                 0.369                      19       0.37          0.788          True
  SLAMF6              0.933                   25                  0.56                      19      0.374          0.713          True
    STAM               1.61                   25                  1.52                      19      0.084          0.947          True
    TFRC               0.79                   25                 0.762                      19     0.0279          0.956          True
   TIGIT               1.91                   25                  1.41                      19      0.492          0.938          True
TNFRSF1B              0.847                   25                -0.207                      19       1.05          0.713          True
TNFRSF25              0.745                   25         
... [2708 more chars truncated]
```

```python  # cell 7
# ============================================================================
# §2c  Everything at once: per-gene regression of the donor effect on who the donor is
# ============================================================================
# One model per gene, over donors: does the within-donor effect depend on diagnosis, ECP,
# sex, age or tumour burden. Study is left out of the design -- with 10 studies and 52 donors
# it would absorb the treatment term it is confounded with -- and is reported separately as
# the variance it explains on its own.
M = TH.lfc_matrix(long, MALIG, REACT, min_donors=MIN_DONORS)
X = pd.DataFrame(index=M.columns)
X["is_MF"] = DON["dx2"].reindex(X.index).eq("MF").astype(float)
X["ecp"] = DON["tx_ecp"].reindex(X.index).where(
    DON["tx_known"].reindex(X.index)).astype(float)
X["is_F"] = DON["sex"].reindex(X.index).eq("F").astype(float)
X["age_z"] = ((DON["age"].reindex(X.index) - DON["age"].mean()) / DON["age"].std())
X["burden_z"] = ((DON["burden"].reindex(X.index) - DON["burden"].mean())
                 / DON["burden"].std())
X = X.dropna()
print(f"{len(X)} donors with every covariate non-missing "
      f"(of {M.shape[1]}); terms: {list(X.columns)}")
print(X.describe().round(3).to_string())

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
40 donors with every covariate non-missing (of 52); terms: ['is_MF', 'ecp', 'is_F', 'age_z', 'burden_z']
        is_MF     ecp    is_F   age_z  burden_z
count  40.000  40.000  40.000  40.000    40.000
mean    0.175   0.525   0.375   0.065    -0.116
std     0.385   0.506   0.490   0.998     0.990
min     0.000   0.000   0.000  -3.098    -2.516
25%     0.000   0.000   0.000  -0.349    -0.661
50%     0.000   1.000   0.000   0.184     0.071
75%     0.000   1.000   1.000   0.823     0.767
max     1.000   1.000   1.000   1.420     0.893


1,098 genes modelled. Terms surviving BH at q < 0.05:
is_MF       0
ecp         0
is_F        0
age_z       0
burden_z    0

--- is_MF: 0 genes, strongest 15
  none

--- ecp: 0 genes, strongest 15
  none

--- is_F: 0 genes, strongest 15
  none

--- age_z: 0 genes, strongest 15
  none

--- burden_z: 0 genes, strongest 15
  none

study on its own, as a one-way ANOVA per gene (not in the model above):

  0 of 1,098 genes differ by study at q < 0.05 -- the scale of the batch term the model above does not carry.

wrote /home/projects/nyosef/zvise/MF/tables/tgt44_strat_regression.csv
```

## §3 — Tiers 2 and 3

Tier 2 is described, not tested: six donors on an HDAC inhibitor can produce an effect size and
a confidence interval, and nothing that should survive a multiple-testing correction. Tier 3 is
printed as counts only. A stratification that is absent from a notebook reads as a
stratification that was not interesting; one that is printed with `n = 1` reads as what it is.

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


TIER2 = {
    "HDACi at sampling": DON["tx_hdaci"].where(DON["tx_known"]),
    "interferon at sampling": DON["tx_interferon"].where(DON["tx_known"]),
    "retinoid at sampling": DON["tx_retinoid"].where(DON["tx_known"]),
    "ever on ECP": DON["ever_ecp"],
    "B2 (vs B0/B1)": DON["clin_B"].map(lambda v: v == "B2" if pd.notna(v) else np.nan),
    ">=3 prior lines": DON["n_prior_lines"].map(
        lambda v: v >= 3 if pd.notna(v) else np.nan),
    "cultured 2 days ex vivo": DON["is_cultured"],
    "shared TRB clone (buus PT05/06/18)": DON["clone_shared"],
}
t2 = pd.concat([describe_split(v, k, SHORT[:15]) for k, v in TIER2.items()],
               ignore_index=True)
print("TIER 2 -- effect sizes with bootstrap CIs, on 43's leading antigens. No test, no "
      "correction, and no row here is evidence of a treatment effect:\n")
for split, d in t2.groupby("split"):
    print(f"--- {split}  (n = {d['n_yes'].iloc[0]} vs {d['n_no'].iloc[0]})")
    print(d[["gene", "median_yes", "median_no", "delta", "ci_lo", "ci_hi"]]
          .sort_values("delta").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print()
t2.to_csv(ST.tab("tier2_descriptive"), index=False)

print("=" * 78)
print("TIER 3 -- not testable in this cohort. The counts, so the gap is explicit:")
print("=" * 78)
t3 = []
t3.append(("stage, early vs advanced",
           DON["clin_stage_class"].value_counts().to_dict(),
           "2 early donors; the blood cohort is advanced disease by construction"))
t3.append(("treatment-naive", DON["clin_tx_naive"].value_counts().to_dict(),
           "harro's 'none' comes from an aggregator, so naive stays unknown there"))
for agent in ("mogamulizumab", "brentuximab", "alemtuzumab", "chemotherapy", "checkpoint",
              "alloSCT", "methotrexate"):
    n = int(DON[f"tx_{agent}"].sum())
    if n <= 4:
        t3.append((f"treatment: {agent}", {"on": n, "off": int(len(DON) - n)},
                   "single-donor case note at most"))
t3.append(("bexarotene separable from ECP",
           pd.crosstab(DON["tx_retinoid"], DON["tx_ecp"]).to_dict(),
           "every retinoid donor in ren is also on ECP"))
for col, name in (("clin_response", "response"), ("clin_outcome", "outcome"),
                  ("clin_drug_sensitivity", "drug sensitivity")):
    t3.append((name, DON[col].value_counts().to_dict(), "essentially absent"))
for name, counts, note in t3:
    print(f"  {name:38s} {counts}")
    print(f"  {'':38s} -> {note}")

print("\nmogamulizumab, as a single-donor case note on its own target:")
_mog = DON.index[DON["tx_mogamulizumab"]]
if len(_mog):
    _c = M.loc[M.index.intersection(["CCR4"]), M.columns.intersection(_mog)]
    print(_c.round(3).to_string())
    print(f"  cohort median CCR4 effect: {M.loc['CCR4'].median():.3f} "
          f"over {int(M.loc['CCR4'].notna().sum())} donors")
    print("  Loss of CCR4 is the documented resistance mechanism to mogamulizumab, so a low "
          "value in a treated donor is the expected direction. One donor cannot distinguish "
          "that from noise.")
else:
    print("  no paired donor is on mogamulizumab at sampling")
print("\nwrote", ST.tab("tier2_descriptive"))
```

```
TIER 2 -- effect sizes with bootstrap CIs, on 43's leading antigens. No test, no correction, and no row here is evidence of a treatment effect:

--- B2 (vs B0/B1)  (n = 13 vs 12)
   gene  median_yes  median_no   delta  ci_lo  ci_hi
   CCR4         1.1       1.33  -0.238  -1.42   0.88
   PI16        1.42        1.6  -0.183  -1.38   1.15
KIR3DL2        2.62       2.77  -0.155  -1.29    2.2
  NRROS        1.08       1.15 -0.0677 -0.567  0.738
   CD84        0.84      0.789  0.0514 -0.794  0.589
  GPR15        1.03      0.957   0.076  -1.28   1.83
   STAM        1.52       1.42   0.104 -0.612   1.06
   CD70        1.84       1.73   0.108 -0.511   1.03
  ITGB1        1.16      0.987   0.175 -0.681  0.871
  PDCD1        1.42       1.15   0.262 -0.283   1.34
 TSPAN2        1.41       1.14   0.276  -1.27   1.44
  FCRL3        2.04       1.54   0.497  -1.34    1.9
  EPHA4        1.45      0.917   0.531 -0.359  0.813
  TIGIT        1.83       1.27   0.557 -0.805   1.38
    TPO        1.91        1.3   0.608  -1.49   1.78

--- HDACi at sampling  (n = 6 vs 38)
   gene  median_yes  median_no  delta   ci_lo  ci_hi
  NRROS        0.86       1.23 -0.367  -0.874  0.592
  EPHA4       0.763          1 -0.241  -0.767  0.577
  ITGB1       0.984       1.16 -0.177  -0.923  0.606
  PDCD1        1.18       1.24 -0.062  -0.543   1.18
   PI16        1.26        1.2 0.0602   -1.02  0.658
   STAM        1.71        1.6  0.115  -0.693  0.572
    TPO        2.06       1.88  0.185    -1.7   2.28
 TSPAN2        1.36       1.09   0.27   -1.23   1.33
  TIGIT        2.15       1.75  0.397  -0.544   1.04
   CD84        1.24      0.842  0.399  -0.337  0.618
  FCRL3         2.1       1.54   0.56   -1.02   2.29
   CCR4        1.82       1.11  0.704  0.0434   1.69
   CD70        2.49       1.78  0.709 -0.0265   2.03
  GPR15        2.74       1.44    1.3   0.168    2.8
KIR3DL2        4.13       2.69   1.45   -2.91   1.88

--- cultured 2 days ex vivo  (n = 5 vs 47)
   gene  median_yes  median_no   delta  ci_lo   ci_hi
  FCRL3       -0.27        1.8   -2.07  -2.57  -0.792
KIR3DL2       0.875        2.9   -2.02  -3.09   0.776
    TPO       0.125       1.97   -1.85  -2.82  -0.561
  GPR15       0.447        1.7   -1.26  -2.25   0.641
  ITGB1     -0.0421       1.17   -1.21  -3.47 -0.0088
  TIGIT       0.738       1.83   -1.09  -2.66  -0.636
  EPHA4       0.398          1  -0.606  -1.14   0.558
  PDCD1       0.748       1.24  -0.494  -1.88  0.0309
   CD84       0.637        1.1  -0.467 -0.988   0.521
   STAM        1.25       1.68  -0.429  -1.88   0.575
 TSPAN2       0.593       1.02  -0.427  -1.35   0.691
   CD70        1.53       1.93  -0.394 -0.915    1.12
   PI16           1       1.25  -0.251 -0.916   0.337
  NRROS        1.17       1.22 -0.0485  -1.39   0.545
   CCR4        1.47       1.13   0.342  -2.26   0.803

--- ever on ECP  (n = 29 vs 23)
   gene  median_yes  median_no    delta  ci_lo  ci_hi
 TSPAN2       0.869       1.21   -0.337  -1.28  0.385
   CD84        1.05       1.33   -0.279 -0.619  0.493
  NRROS        1.08        1.3    -0.22 -0.825  0.208
   PI16        1.09       1.25   -0.159  -1.15  0.797
  PDCD1        1.14       1.26   -0.125 -0.721  0.512
   CD70        1.84       1.95   -0.113 -0.546  0.515
  EPHA4       0.975          1  -0.0281 -0.581  0.344
  ITGB1        1.15       1.16 -0.00705 -0.523  0.364
   CCR4        1.19        1.1   0.0883 -0.678      1
KIR3DL2        2.85       2.64    0.211  -1.03   1.04
   STAM        1.82       1.52    0.294 -0.261  0.647
  TIGIT        1.91       1.41    0.498 -0.275   1.32
  FCRL3        1.88      0.878        1 -0.356   2.13
  GPR15        2.17       1.06     1.11 -0.286   2.23
    TPO        2.67        1.3     1.37  0.214   2.08

--- interferon at sampling  (n = 6 vs 38)
   gene  median_yes  median_no   delta   ci_lo  ci_hi
  GPR15       0.519       1.97   -1.45   -3.18  0.396
  FCRL3       0.882       1.76  -0.875   -2.02   1.22
   CD70        1.23          2  -0.773   -1.58  0.124
    TPO        1.72    
... [5073 more chars truncated]
```

## §4 — Heterogeneity: is the effect the same size in everyone?

Four views of the same donor × gene matrix, because they disagree and the disagreement is the
information.

1. **Random-effects meta-analysis.** Each donor contributes an effect and a standard error
   (delta method, from the per-arm cell SDs the QC build stores). `tau2` is the spread of the
   *true* effect once each donor's own sampling noise is removed, `I2` the share of observed
   spread that is not noise, and the prediction interval what the next patient would plausibly
   show. A large pooled effect with `I2` near 1 is a subgroup antigen with a high average.
2. **COPA and the outlier sum.** Built for genes activated in a fraction of tumours, which a
   mean test averages away. Both standardise by median and MAD and score the upper tail; the
   null is a sign flip of the paired differences, the same exchangeability the Wilcoxon assumes.
3. **Bimodality.** One- versus two-component Gaussian mixture BIC and the bimodality
   coefficient. (`diptest` is not installed in `neural_nmf_env`, so Hartigan's dip is not
   available.) With ~50 donors these are suggestive, not conclusive, and are reported beside
   `I2` rather than instead of it.
4. **Eligibility.** In what fraction of patients would at least X% of the tumour carry the
   antigen. This is the form a biomarker-selected trial is designed against, and the one that
   can be read without any of the above.

Then: do the donors fall into groups, and do those groups correspond to anything clinical.

```python  # cell 11
# ============================================================================
# §4a  Random-effects meta-analysis across donors
# ============================================================================
SE = TH.lfc_se(long, MALIG, REACT)
re_tab = TH.meta_random_effects(M, SE, min_donors=MIN_DONORS)
re_tab["on_shortlist"] = re_tab["gene"].isin(SHORT)
print("sampling variance source:", re_tab["se_source"].iloc[0])
print(f"{len(re_tab):,} genes pooled over up to {int(re_tab['n_donors'].max())} donors\n")

print(f"median I2 = {re_tab['I2'].median():.2f}. Each donor contributes thousands of cells, "
      "so the sampling term in the model is almost zero and nearly all the observed spread is "
      "counted as real: I2 therefore sits near 1 for almost every gene and does NOT separate "
      "them. `tau` -- the absolute between-patient SD of the effect, in log2FC units -- keeps "
      "its meaning, and the ranking below uses it.\n")

print("the most heterogeneous antigens with a positive pooled effect -- large on average, "
      "and not the same size in everyone:")
_c = ["gene", "n_donors", "mu", "se", "tau", "tau_over_mu", "I2", "pi_lo", "pi_hi",
      "frac_donors_up", "q_Q", "on_shortlist"]
het = re_tab[(re_tab["mu"] > 0.3)].sort_values("tau", ascending=False)
print(het[_c].head(25).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\nthe most CONSISTENT antigens with a positive pooled effect -- the ones whose "
      "prediction interval stays above zero, which is what 'works in every patient' means:")
con = re_tab[(re_tab["mu"] > 0.3) & (re_tab["pi_lo"] > 0)].sort_values("mu", ascending=False)
print(con[_c].head(25).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(f"\n{len(con)} of {int((re_tab['mu'] > 0.3).sum())} genes with a pooled effect above "
      "0.3 have a prediction interval that excludes zero.")

print("\n43's shortlist, ranked by how consistent it is (smallest tau first):")
print(re_tab[re_tab["on_shortlist"]].sort_values("tau")[_c]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
re_tab.to_csv(ST.tab("heterogeneity_meta"), index=False)
print("\nwrote", ST.tab("heterogeneity_meta"))
```

```
sampling variance source: delta-method from per-arm cell SD
1,098 genes pooled over up to 52 donors

median I2 = 0.98. Each donor contributes thousands of cells, so the sampling term in the model is almost zero and nearly all the observed spread is counted as real: I2 therefore sits near 1 for almost every gene and does NOT separate them. `tau` -- the absolute between-patient SD of the effect, in log2FC units -- keeps its meaning, and the ranking below uses it.

the most heterogeneous antigens with a positive pooled effect -- large on average, and not the same size in everyone:
      gene  n_donors    mu    se  tau  tau_over_mu    I2  pi_lo  pi_hi  frac_donors_up  q_Q  on_shortlist
   KIR3DL2        52  2.55   0.3 2.15        0.845 0.999  -1.82   6.91           0.904    0          True
     GPR15        52  1.54 0.258 1.84          1.2 0.998   -2.2   5.28           0.827    0          True
       TPO        52  2.01 0.223  1.6        0.795 0.998  -1.23   5.26           0.962    0          True
     FCRL3        52  1.44 0.208 1.49         1.03 0.981  -1.58   4.46           0.712    0          True
     RAB25        52  1.22 0.204 1.46         1.19 0.981  -1.74   4.18           0.769    0         False
      CCR4        52 0.885 0.201 1.43         1.62 0.993  -2.02   3.79           0.769    0          True
     PDCD1        52  1.03 0.195 1.39         1.34 0.991  -1.78   3.84           0.865    0          True
    GALNT2        52 0.378 0.191 1.35         3.57 0.987  -2.36   3.12           0.635    0         False
     CTLA4        52 0.649 0.179 1.27         1.95  0.99  -1.92   3.22           0.731    0          True
      PI16        52  1.18 0.177 1.26         1.07 0.994  -1.37   3.73           0.846    0          True
     EPCAM        52 0.737 0.168  1.2         1.63 0.981   -1.7   3.17           0.596    0         False
   TNFRSF4        52 0.405 0.166 1.18          2.9 0.991  -1.98   2.79           0.673    0         False
      LAG3        52  0.69 0.164 1.17          1.7 0.981  -1.69   3.07            0.75    0         False
     CCR10        52 0.374 0.162 1.16          3.1 0.981  -1.98   2.72           0.692    0         False
     TIGIT        52  1.47 0.164 1.16        0.788 0.991 -0.881   3.82           0.827    0          True
    CX3CR1        52 0.428 0.156 1.11          2.6 0.981  -1.83   2.68           0.577    0         False
     PTH2R        52  1.16 0.154  1.1        0.946 0.981  -1.07    3.4           0.865    0         False
ST6GALNAC1        52 0.587 0.153 1.09         1.86 0.981  -1.62    2.8           0.654    0         False
        F5        52 0.565 0.155 1.09         1.92 0.981  -1.64   2.77           0.692    0         False
   KIR3DL1        52 0.597 0.149 1.07         1.78 0.981  -1.56   2.76           0.615    0         False
      CCR8        52 0.373 0.149 1.06         2.85 0.981  -1.78   2.53           0.615    0         False
     CD79B        52 0.475  0.15 1.05         2.22 0.986  -1.67   2.62           0.788    0         False
  TNFRSF1B        52 0.387 0.148 1.05          2.7 0.992  -1.74   2.51           0.577    0          True
      CD70        52  1.77 0.146 1.04        0.589 0.981 -0.344   3.89           0.885    0          True
       CD9        52 0.484 0.144 1.03         2.12 0.981   -1.6   2.56           0.615    0         False

the most CONSISTENT antigens with a positive pooled effect -- the ones whose prediction interval stays above zero, which is what 'works in every patient' means:
 gene  n_donors    mu     se   tau  tau_over_mu    I2  pi_lo  pi_hi  frac_donors_up       q_Q  on_shortlist
  MBP        52 0.793 0.0472 0.321        0.405 0.965  0.141   1.45           0.942 2.66e-270          True
IGSF8        52 0.582 0.0463 0.275        0.473 0.831  0.021   1.14           0.962  2.79e-37         False

2 of 133 genes with a pooled effect above 0.3 have a prediction interval that excludes zero.

43's shortlist, ranked by how consistent it is (smallest tau first):
    gene 
... [3536 more chars truncated]
```

```python  # cell 12
# ============================================================================
# §4b  COPA and the outlier sum: up in a subset, by design
# ============================================================================
cp = TH.copa(M, pct=75, n_perm=N_PERM, min_donors=MIN_DONORS)
cp = cp.merge(re_tab[["gene", "mu", "I2"]], on="gene", how="left")
cp = cp.merge(sel[["gene", "log2fc_paired", "q"]], on="gene", how="left")
cp["on_shortlist"] = cp["gene"].isin(SHORT)

_c = ["gene", "copa_p75", "outlier_sum", "n_outlier_donors", "frac_outlier", "q_copa",
      "q_outlier_sum", "mu", "I2", "log2fc_paired", "q", "on_shortlist"]
_c = [c for c in _c if c in cp.columns]
print("highest COPA score -- extreme in a minority of patients:")
print(cp.sort_values("copa_p75", ascending=False).head(25)[_c]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\nthe interesting cell of the table: a high outlier score and a cohort effect that is "
      "NOT significant. These are antigens a mean test discards and a biomarker-selected "
      "trial would not:")
hid = cp[(cp["q"] >= 0.05) & (cp.get("q_copa", pd.Series(1, index=cp.index)) < 0.1)]
hid = hid.sort_values("copa_p75", ascending=False).head(25)
print(hid[_c].to_string(index=False, float_format=lambda v: f"{v:.3g}")
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
  GRIK2      35.4     2.91e+03                 9         0.173   0.051         0.0535   0.107 0.981              0 0.000234         False
CNTNAP2      29.2          767                 7         0.135  0.0658          0.952  0.0124 0.981              0   0.0197         False
  EPCAM      19.9          535                 9         0.173  0.0392          0.693   0.737 0.981         0.0183 2.08e-05         False
   MSR1      17.8          919                10         0.192  0.0392         0.0392   0.295 0.981        0.00603 4.62e-05         False
  RAMP2      16.3          332                 4        0.0769  0.0392              1  0.0646 0.981              0  0.00319         False
  FCRL2      15.1          457                 7         0.135   0.127          0.561  0.0122 0.981              0   0.0174         False
 ADAM29      11.6          251                 5        0.0962  0.0392           0.65   0.437 0.981         0.0332 5.57e-06         False
 CHRNB3      9.65     1.88e+03                10         0.192  0.0392         0.0392   0.129 0.981        0.00235 0.000831         False
CLEC12A      8.85     1.06e+03                11         0.212   0.848          0.695 0.00847 0.981              0    0.326         False
   VNN3       5.6          283                 8         0.154   0.203              1 0.00416 0.981              0    0.104         False
 LILRA6      5.48          165                 7         0.135  0.0844          0.638  0.0314 0.981              0  0.00385         False
   CYBB      4.86          253                10         0.192   0.242          0.933  0.0167 0.981              0   0.0298         False
  S1PR3      4.77          359                10         0.192   0.697              1 0.00142 0.981              0    0.443         False
  SCN2A      4.57     1.59e+03                12         0.231  0.0392         0.0392   0.221 0.981              0 0.000851         False
  BTNL9      4.53          127                 8         0.154  0.0392          0.439   0.122 0.981         0.0219 5.57e-06         False
 CHRNA7       4.3         74.3                 5        0.0962  0.0658              1   0.468 0.981         0.0981 2.23e-06         False
   AOC3      3.87         93.1                 6         0.115  0.0392         0.0392  0.0133 0.981        0.00196  0.00231         False
  LAMA3       3.8          120                 7         0.135  0.0392          0.168    0.15 0.981         0.0155 0.000155         False
 GALNT9      3.79          155                10         0.192  0.0392          0.626   0.443 0.981         0.0504 2.31e-05         False
 SLC6A3       3.6          288                11         0.212   0.344          0.795  0.0129 0.981              0   0.0711         False
 SLC8A1      3.59          639                10         0.192   0.829         0.0672  0.0539 0.981              0    0.151         False
KIR2DL1      3.52          125                 8         0.154  0.0392         0.0795   0.323 0.981         0.0143  0.00206         False
  GPNMB      3.47         93.6                 7         0.135    0.15          0.739  0.0118 0.981        0.00389  0.00551         False
  FFAR3      3.44          146                10         0.192  0.0658          0.135   0.157 0.981          0.017 0.000603         False
  PTPRF       3.4          254                 8         0.154   0.548          0.751  0.0068 0.981              0    0.114         False

the interesting cell of the table: a high outlier score and a cohort effect that is NOT significant. These are antigens a mean test discards and a biomarker-selected trial would not:
  gene  copa_p75  outlier_sum  n_outlier_donors  frac_outlier  q_copa  q_outlier_sum    mu    I2  log2fc_paired      q  on_shortlist
 CLDN7      2.62          125         
... [3228 more chars truncated]
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
    print("\n*** NO DONOR GROUPING. No cut of the dendrogram into clusters of at least "
          "3 donors beat the silhouette criterion, so on these genes the 52 patients do "
          "not fall into subtypes: the heterogeneity measured above is a continuum, not "
          "a set of groups. The association tests below are then trivially one-sided, "
          "and F1 still shows the per-donor structure. A split into one outlier and "
          "everyone else, which an unconstrained silhouette would have returned, is not "
          "a subtype. ***")

ann = DON.reindex(labels.index)[["dx2", "study", "sex", "age", "burden", "tx_ecp",
                                 "tx_hdaci", "is_cultured", "n_prior_lines"]].copy()
ann["cluster"] = labels
print("\ndonor clusters against the clinical fields:")
for col in ["dx2", "study", "sex", "tx_ecp", "tx_hdaci", "is_cultured"]:
    ct = pd.crosstab(ann["cluster"], ann[col])
    if ct.shape[1] < 2:
        continue
    p = stats.chi2_contingency(ct)[1] if ct.values.min() >= 0 else np.nan
    print(f"\n  {col}  (chi2 p = {p:.3g})")
    print(ct.to_string())
for col in ["age", "burden", "n_prior_lines"]:
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
56 genes x 52 donors -> 1 donor clusters {1: 52}

*** NO DONOR GROUPING. No cut of the dendrogram into clusters of at least 3 donors beat the silhouette criterion, so on these genes the 52 patients do not fall into subtypes: the heterogeneity measured above is a continuum, not a set of groups. The association tests below are then trivially one-sided, and F1 still shows the per-donor structure. A split into one outlier and everyone else, which an unconstrained silhouette would have returned, is not a subtype. ***

donor clusters against the clinical fields:

  dx2  (chi2 p = 1)
dx2      MF  SS  unresolved
cluster                    
1         8  37           7

  study  (chi2 p = 1)
study    borcherding2019  borcherding2023  buus2025  dorando2026  gaydosik2022  geskin2026  harro2023  herrera2021  ren2023  rindler2021_fi
cluster                                                                                                                                    
1                      1                3        15            5             3           5          3            6       10               1

  sex  (chi2 p = 1)
sex       F   M
cluster        
1        22  30

  tx_ecp  (chi2 p = 1)
tx_ecp   False  True 
cluster              
1           27     25

  tx_hdaci  (chi2 p = 1)
tx_hdaci  False  True 
cluster               
1            46      6

  is_cultured  (chi2 p = 1)
is_cultured  False  True 
cluster                  
1               47      5

per-cluster mean effect for the leading antigens -- what distinguishes the groups:
donor_cluster      1  spread
gene                        
KIR3DL2        2.546     0.0
CD70           1.773     0.0
TPO            2.013     0.0
TIGIT          1.471     0.0
STAM           1.444     0.0
FCRL3          1.441     0.0
GPR15          1.533     0.0
PDCD1          1.025     0.0
NRROS          1.111     0.0
PI16           1.172     0.0
ITGB1          1.080     0.0
CCR4           0.880     0.0
CD84           1.134     0.0
EPHA4          1.070     0.0
TSPAN2         1.139     0.0
ABCC1          0.871     0.0
MBP            0.785     0.0
GPR146         0.855     0.0
FUT8           0.695     0.0
SLAMF6         0.831     0.0

wrote /home/projects/nyosef/zvise/MF/tables/tgt44_donor_clusters.csv and /home/projects/nyosef/zvise/MF/tables/tgt44_donor_cluster_profiles.csv
```

```python  # cell 14
# ============================================================================
# §4d F1  The donor x gene matrix with the clinical annotation beside it
# ============================================================================
# Three orderings of the same matrix. F1: donors in effect-profile order (as before). F1b: the
# same, with the donor dendrogram drawn, so the eye can see how deep each split is. F1c: genes
# clustered into programs that rise and fall together across patients, donors by burden.
from scipy.cluster.hierarchy import dendrogram

AN = [("cluster", ann["cluster"].astype(str)), ("dx", ann["dx2"].astype(str)),
      ("ECP", ann["tx_ecp"].map({True: "yes", False: "no"}).astype(str)),
      ("sex", ann["sex"].astype(str))]
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
prog = glab.reindex(gorder).reset_index().rename(columns={"index": "gene"})
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
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F1_donor_clustermap.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F1_donor_clustermap.pdf

<Figure size 1016x1212 with 8 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt44_F1_donor_clustermap_source.csv

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F1b_donor_dendrogram.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F1b_donor_dendrogram.pdf

<Figure size 1016x1262 with 9 Axes>
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F1c_gene_programs.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F1c_gene_programs.pdf

<Figure size 1116x1212 with 9 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt44_F1c_gene_programs_source.csv
no gene grouping beat the silhouette criterion (min 3 genes per program): the dendrogram is shown, but the genes do not split into programs
gene programs:
  1: TNFRSF4, ACVR2A, LRRC32, CD28, CD84, SLAMF6, GRN, CD79B, FCGRT, PDCD1, ITGB1, FUT8, ST6GALNAC1, CD70, LAG3, CX3CR1, FZD3, STAM, FGFR1, KIR3DL2, TNFRSF25, MBP, GPR146, CD82, NRROS, TSPAN2, KIR3DL1, KIR2DL3, EPHA4, TRGC2, GALNT2, NXPH4, CDH1, RAB25, EPCAM, CLDN7, TPO, ABCC1, CPM, F5, FCRL3, CTLA4, TIGIT, TNFRSF1B, IL6R, GPR15, SIRPG, CCR4, CD9, ANXA2, CCR8, PI16, CCR10, PTH2R, TFRC, ENTPD1

per-program effect across donors:
         n_genes  median_lfc  frac_donors_up_0.5  rho_burden
program                                                     
1             56       0.846               0.788      -0.228
wrote /home/projects/nyosef/zvise/MF/tables/tgt44_gene_programs.csv
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

print("43's shortlist, as a recruitment statement:")
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
    ax.step(THRESH, y, where="post", lw=1.2, color=cmap(i / max(len(TOPE) - 1, 1)),
            label=gene)
    ax.scatter(THRESH, y, s=8, color=cmap(i / max(len(TOPE) - 1, 1)), zorder=3)
ax.set_xlabel("required fraction of a patient's malignant CD4 cells positive")
ax.set_ylabel("fraction of patients eligible")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=5.5, frameon=False, ncol=2)
ax.set_title("Patients eligible by tumour-positivity cut", loc="left", fontsize=7.5)
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F2_eligibility", caption_text=(
    f"Transcript detection, 10k-HVG object. {len(PAIRED)} donors."))
TF.save_source(elig, "F2_eligibility_source")
TF.logfig("F2_eligibility", "eligibility curves for the shortlist", len(elig))
```

```
43's shortlist, as a recruitment statement:
    gene  n_donors  median_pct_expr  frac_donors_ge5  frac_donors_ge10  frac_donors_ge25  frac_donors_ge50  frac_donors_ge75    mu    I2
     MBP        52            0.719                1                 1                 1             0.885             0.423 0.793 0.965
   ITGB1        52            0.828            0.981             0.981             0.981             0.923             0.673  1.08 0.995
   ANXA2        52            0.524                1             0.981             0.885             0.538             0.269 0.648  0.99
    CD28        52            0.412                1                 1             0.846             0.212            0.0192 0.578  0.96
    CD82        52             0.44            0.962             0.923             0.808             0.385            0.0385 0.763 0.982
    STAM        52            0.445            0.962             0.885             0.769             0.404            0.0192  1.45 0.979
   TIGIT        52            0.424            0.962             0.923              0.75             0.327             0.154  1.47 0.991
TNFRSF25        52            0.395                1                 1             0.712             0.327            0.0385 0.422 0.984
TNFRSF1B        52            0.291            0.942             0.885             0.654             0.288            0.0962 0.387 0.992
 KIR3DL2        52            0.367            0.808             0.769             0.615             0.423             0.115  2.55 0.999
    PI16        52            0.254            0.808             0.712               0.5              0.25            0.0192  1.18 0.994
  SLAMF6        52             0.25                1             0.885               0.5            0.0769                 0 0.827 0.959
    IL6R        52            0.243                1             0.981             0.481             0.173            0.0192 0.754 0.972
    CD84        52            0.243                1             0.942             0.481            0.0577            0.0192  1.13 0.976
    CD70        52            0.241            0.904             0.808             0.481            0.0577                 0  1.77 0.981
    CCR4        52            0.231            0.827             0.731             0.462             0.115                 0 0.885 0.993
   ABCC1        52            0.221                1             0.942             0.442            0.0577                 0 0.893 0.928
   SIRPG        52            0.245            0.962             0.827             0.442             0.154                 0 0.584 0.974
    FUT8        52            0.227                1             0.885             0.404            0.0385                 0 0.712 0.955
   GPR15        52            0.182            0.635             0.558             0.385             0.115            0.0192  1.54 0.998
   FCRL3        52            0.198            0.673             0.596             0.385             0.115                 0  1.44 0.981
     TPO        52            0.121            0.673             0.558             0.365             0.173            0.0192  2.01 0.998
   EPHA4        52            0.189            0.981             0.846             0.308            0.0192                 0  1.08 0.977
   NRROS        52            0.149            0.865             0.615             0.308                 0                 0  1.11 0.976
   CTLA4        52            0.159            0.923              0.75             0.288            0.0769            0.0192 0.649  0.99
  TSPAN2        52            0.132            0.904             0.615             0.288            0.0962                 0  1.18 0.978
   PDCD1        52            0.113            0.788             0.519             0.192            0.0192                 0  1.03 0.991
     GRN        52            0.159            0.904             0.673             0.192                 0              
... [4289 more chars truncated]
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
    "Grey: all genes; coloured: shortlist. Dashed: τ = μ."))
TF.save_source(d, "F3_effect_vs_heterogeneity_source")
TF.logfig("F3_effect_vs_heterogeneity", "pooled effect vs I2", len(d))

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
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F3_effect_vs_heterogeneity.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt44_F3_effect_vs_heterogeneity.pdf

<Figure size 520x360 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt44_F3_effect_vs_heterogeneity_source.csv
the quadrant that matters for a single-agent therapy -- large and consistent:
Empty DataFrame
Columns: [gene, mu, tau, pi_lo, pi_hi, frac_donors_up, on_shortlist]
Index: []

and the quadrant that argues for patient selection -- large but not in everyone:
   gene    mu   tau   pi_lo  pi_hi  frac_donors_up  frac_outlier  on_shortlist
KIR3DL2  2.55  2.15   -1.82   6.91           0.904             0          True
    TPO  2.01   1.6   -1.23   5.26           0.962             0          True
   CD70  1.77  1.04  -0.344   3.89           0.885        0.0192          True
  GPR15  1.54  1.84    -2.2   5.28           0.827             0          True
  TIGIT  1.47  1.16  -0.881   3.82           0.827        0.0192          True
   STAM  1.45 0.836  -0.249   3.14           0.865        0.0192          True
  FCRL3  1.44  1.49   -1.58   4.46           0.712             0          True
  RAB25  1.22  1.46   -1.74   4.18           0.769             0         False
    CPM  1.22 0.999  -0.804   3.25           0.904         0.115         False
   PI16  1.18  1.26   -1.37   3.73           0.846             0          True
 TSPAN2  1.18 0.859  -0.567   2.92           0.923             0          True
  PTH2R  1.16   1.1   -1.07    3.4           0.865        0.0385         False
   CD84  1.13 0.751  -0.394   2.65           0.981        0.0192          True
  NRROS  1.11 0.823  -0.561   2.78           0.904             0          True
  FGFR1  1.09 0.841  -0.622   2.79           0.885             0         False
  EPHA4  1.08 0.905  -0.756   2.92           0.942        0.0192          True
  ITGB1  1.08 0.859  -0.663   2.82           0.904        0.0192          True
 ADGRE1  1.05 0.839  -0.656   2.75           0.865             0         False
  PDCD1  1.03  1.39   -1.78   3.84           0.865        0.0577          True
  GPR55  1.01 0.828  -0.669   2.69           0.904             0         False
 ACVR2A 0.951 0.942  -0.961   2.86           0.923        0.0385         False
  ABCC1 0.893 0.471 -0.0633   1.85           0.904        0.0385          True
   CCR4 0.885  1.43   -2.02   3.79           0.769        0.0192          True
 GPR146 0.862 0.679  -0.516   2.24           0.904        0.0385          True
   EVC2 0.857 0.437 -0.0304   1.75           0.962        0.0192         False
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

SEC("2 - WHAT WAS TESTED (TIER 1)")
for t, n in (("MF vs SS", f"{int(dx_map.eq('MF').sum())} vs {int(dx_map.eq('SS').sum())}"),
             ("ECP vs not", f"{int((ecp_map == 'ECP').sum())} vs "
                            f"{int((ecp_map == 'no ECP').sum())}"),
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
    "A treatment-effect study. Treatment is confounded with study; ECP is the only agent with "
    "two arms and even that is mostly buus and ren.",
    "A disease comparison with power: 8 MF donors against 37 SS, MF spread over 4 studies.",
    "A stage analysis: 2 early-stage donors.",
    "A response or survival analysis: those fields are empty in this cohort.",
    "A protein result. Every number is transcript, in a 10,000-gene HVG subset.",
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
  paired donors                        52
  people                               52

dx
SS                     37
MF                      8
MF_or_SS_unresolved     5
MF/SS                   2
  
  studies                           10
  donors with every model covariate    40

==============================================================================
2 - WHAT WAS TESTED (TIER 1)
==============================================================================
  MF vs SS                             8 vs 37
  ECP vs not                           25 vs 19
  sex                                  22 F / 30 M
  age                                  44 donors, median 70
  burden                               52 donors, 5%-99%

is_MF       0
ecp         0
is_F        0
age_z       0
burden_z    0

==============================================================================
3 - HETEROGENEITY
==============================================================================
  genes pooled                         1098
  median I2 (saturated)                0.98
  median tau                           0.19
  pooled effect > 0.3                  133
    of those, PI excludes zero         2
  donor clusters                       1 {1: 52}

==============================================================================
4 - WHAT THIS IS NOT
==============================================================================
  - A treatment-effect study. Treatment is confounded with study; ECP is the only agent with two arms and even that is mostly buus and ren.
  - A disease comparison with power: 8 MF donors against 37 SS, MF spread over 4 studies.
  - A stage analysis: 2 early-stage donors.
  - A response or survival analysis: those fields are empty in this cohort.
  - A protein result. Every number is transcript, in a 10,000-gene HVG subset.

==============================================================================
5 - OUTPUTS
==============================================================================
wrote /home/projects/nyosef/zvise/MF/tables/tgt44_figure_index.csv
   tables/tgt44_F1_donor_clustermap_source.csv
   tables/tgt44_F1c_gene_programs_source.csv
   tables/tgt44_F2_eligibility_source.csv
   tables/tgt44_F3_effect_vs_heterogeneity_source.csv
   tables/tgt44_donor_cluster_profiles.csv
   tables/tgt44_donor_clusters.csv
   tables/tgt44_figure_index.csv
   tables/tgt44_gene_programs.csv
   tables/tgt44_heterogeneity_bimodality.csv
   tables/tgt44_heterogeneity_copa.csv
   tables/tgt44_heterogeneity_meta.csv
   tables/tgt44_llm_export.md
   tables/tgt44_power_table.csv
   tables/tgt44_strat_disease.csv
   tables/tgt44_strat_ecp.csv
   tables/tgt44_strat_ecp_blocked.csv
   tables/tgt44_strat_regression.csv
   tables/tgt44_tier2_descriptive.csv
   tables/tgt44_treatment_counts.csv
   figures/targets/tgt44_F1_donor_clustermap.pdf
   figures/targets/tgt44_F1_donor_clustermap.png
   figures/targets/tgt44_F1b_donor_dendrogram.pdf
   figures/targets/tgt44_F1b_donor_dendrogram.png
   figures/targets/tgt44_F1c_gene_programs.pdf
   figures/targets/tgt44_F1c_gene_programs.png
   figures/targets/tgt44_F2_eligibility.pdf
   figures/targets/tgt44_F2_eligibility.png
   figures/targets/tgt44_F3_effect_vs_heterogeneity.pdf
   figures/targets/tgt44_F3_effect_vs_heterogeneity.png
```

```python  # cell 18
# ============================================================================
# §6  The complete numerical result, as one block to hand to another model
# ============================================================================
_ctx = f"""
Compartment: MF/CTCL **blood**. This notebook stratifies the malignant-vs-reactive CD4 antigen
axis that notebook 43 established, and measures how much it varies between patients. It derives
no new screen: it reads 43's cleaned cohort and tables.

Cohort: {len(DON)} donors ({DON['person'].nunique()} people, {DON['study'].nunique()} studies)
carrying BOTH a malignant and a reactive CD4 arm of >= 20 cells -- 37 Sezary, 8 MF, 5 unresolved,
2 MF/SS. The geskin2026 ex vivo drug wells and duplicate persons were removed in 43 section 0b,
and five healthy donors the atlas labelled SS were moved out of the reactive arm.

Per-donor effect: for each gene and donor, log2((mean malignant + 0.05)/(mean reactive + 0.05))
on log1p CP10K. Everything below is a function of that donor x gene matrix.

Stratified tests: the paired Wilcoxon is rerun inside each stratum, and `p_interaction` compares
the per-donor effects between strata (BH as `q_interaction`). Per-gene OLS regresses the donor
effect on diagnosis, ECP, sex, age and burden at once.

Heterogeneity: DerSimonian-Laird random effects (`mu` pooled effect, `tau` between-patient SD,
`I2`, `pi_lo`/`pi_hi` prediction interval, `q_Q`); COPA and the outlier sum with a
{N_PERM}-draw sign-flip null; two-component Gaussian mixture BIC; hierarchical donor clustering
with a silhouette cut and at least 3 donors per cluster.

Power limits that apply to every result: treatment is almost entirely confounded with study (ECP
is the only agent with two usable arms, and it is concentrated in buus and ren); there are 2
early-stage donors against 35 advanced; response and survival are essentially absent. I2
saturates near 1 because each donor contributes thousands of cells, so rank consistency on `tau`,
not on I2. Everything is transcript, in a 10,000-gene HVG subset.
"""
_sec = [
    {"name": "§1 Power table",
     "what": "coverage of every candidate stratification over the paired donors.",
     "how": "count of non-missing values per field, split by diagnosis.",
     "table": power, "file": "tables/tgt44_power_table.csv"},
    {"name": "§1 Treatment at sampling",
     "what": "donors on each treatment class, and in which studies.",
     "how": "membership of the curated clin_current_tx_classes string; a donor can be on "
            "several.", "table": tx, "file": "tables/tgt44_treatment_counts.csv"},
    {"name": "§2a Disease, MF vs SS, on 43's shortlist",
     "what": "each leading antigen's effect inside MF and inside SS, and whether they differ.",
     "how": "paired Wilcoxon within stratum; delta_lfc and q_interaction compare the per-donor "
            "effects. 8 MF donors: a consistency check, not a finding.",
     "table": dx_tab[dx_tab["on_shortlist"]].sort_values("log2fc_all", ascending=False),
     "file": "tables/tgt44_strat_disease.csv", "max_rows": 40},
    {"name": "§2b ECP, on 43's shortlist",
     "what": "the same split by extracorporeal photopheresis at sampling.",
     "how": "as above; the study-blocked (van Elteren) version is the one to quote and is in "
            "the file named below.",
     "table": ew[ew["on_shortlist"]], "file": "tables/tgt44_strat_ecp_blocked.csv",
     "max_rows": 40},
    {"name": "§2c Donor-level regression",
     "what": "which donor properties move each gene's effect, all terms in one model.",
     "how": "OLS per gene over donors, terms " + ", ".join(X.columns) + "; BH per term. Study "
            "is deliberately out of the design and is reported as its own ANOVA.",
     "table": pd.Series({t: int((reg[f"q_{t}"] < 0.05).sum()) for t in X.columns},
                        name="genes_at_q_lt_0.05"),
     "file": "tables/tgt44_strat_regression.csv"},
    {"name": "§3 Tier 2 descriptive splits",
     "what": "effect sizes on each side of the splits too small to test.",
     "how": "median per-donor effect per side with a 400-draw bootstrap CI. No p-value.",
     "table": t2, "file": "tables/tgt44_tier2_descriptive.csv", "max_rows": 60},
    {"name": "§4a Random-effects meta-analysis, 43's shortlist",
     "what": "pooled effect and between-patient spread for each leading antigen.",
     "how": "DerSimonian-Laird; `pi_lo > 0` is what 'works in every patient' means.",
     "table": re_tab[re_tab["on_shortlist"]].sort_values("tau"),
     "file": "tables/tgt44_heterogeneity_meta.csv", "max_rows": 40},
    {"name": "§4a Most consistent antigens cohort-wide",
     "what": "pooled effect above 0.3 with a prediction interval excluding zero.",
     "how": "same model, ranked by mu.",
     "table": con[["gene", "n_donors", "mu", "tau", "pi_lo", "pi_hi", "frac_donors_up",
                   "on_shortlist"]], "max_rows": 40},
    {"name": "§4b COPA, subgroup antigens",
     "what": "genes extreme in a minority of patients, which a mean test averages away.",
     "how": "median/MAD-standardised 75th percentile, sign-flip null, BH.",
     "table": cp.sort_values("copa_p75", ascending=False).head(40)[
         [c for c in ("gene", "copa_p75", "frac_outlier", "q_copa", "mu", "log2fc_paired", "q",
                      "on_shortlist") if c in cp.columns]],
     "file": "tables/tgt44_heterogeneity_copa.csv"},
    {"name": "§4b Bimodality of the per-donor effect",
     "what": "whether a gene's donor effects fall into two groups rather than one spread.",
     "how": "one- versus two-component Gaussian mixture BIC; delta_bic > 0 favours two "
            "components. A weak instrument at this donor count -- read it beside I2 and COPA.",
     "table": bm.head(25), "file": 'tables/tgt44_heterogeneity_bimodality.csv'},
    {"name": "§4c Donor clusters against the clinical fields",
     "what": "whether the patients fall into groups on these antigens.",
     "how": "hierarchical clustering of the donor x gene matrix, silhouette cut, min 3 donors.",
     "table": ann, "file": "tables/tgt44_donor_clusters.csv", "max_rows": 80},
    {"name": "§4c Per-cluster mean effect for the leading antigens",
     "what": "what separates the donor clusters, where there are any.",
     "how": "mean per-donor effect inside each cluster; `spread` is max minus min across "
            "clusters.",
     "table": prof.sort_values("spread", ascending=False).head(25), "file": 'tables/tgt44_donor_cluster_profiles.csv'},
    {"name": "§4e Eligibility",
     "what": "the fraction of patients in whom at least X% of malignant CD4 cells are positive.",
     "how": "transcript detection per donor, thresholded at 5/10/25/50/75%.",
     "table": elig[elig["on_shortlist"]], "max_rows": 40},
    {"name": "§4f Large and consistent vs large and selective",
     "what": "the two quadrants a development decision turns on.",
     "how": "mu > 0.5 split on tau against its cohort median and on whether the prediction "
            "interval excludes zero.",
     "table": pd.concat([q1.assign(quadrant="large_and_consistent"),
                         q2.assign(quadrant="large_but_variable")])
                [["gene", "quadrant", "mu", "tau", "pi_lo", "pi_hi", "frac_donors_up",
                  "on_shortlist"]], "max_rows": 50},
]
TF.llm_export("44", "Blood stratification and heterogeneity", _ctx, _sec)
```

```
======================== COPY FROM HERE ========================

# 44 — Blood stratification and heterogeneity

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**. This notebook stratifies the malignant-vs-reactive CD4 antigen
axis that notebook 43 established, and measures how much it varies between patients. It derives
no new screen: it reads 43's cleaned cohort and tables.

Cohort: 52 donors (52 people, 10 studies)
carrying BOTH a malignant and a reactive CD4 arm of >= 20 cells -- 37 Sezary, 8 MF, 5 unresolved,
2 MF/SS. The geskin2026 ex vivo drug wells and duplicate persons were removed in 43 section 0b,
and five healthy donors the atlas labelled SS were moved out of the reactive arm.

Per-donor effect: for each gene and donor, log2((mean malignant + 0.05)/(mean reactive + 0.05))
on log1p CP10K. Everything below is a function of that donor x gene matrix.

Stratified tests: the paired Wilcoxon is rerun inside each stratum, and `p_interaction` compares
the per-donor effects between strata (BH as `q_interaction`). Per-gene OLS regresses the donor
effect on diagnosis, ECP, sex, age and burden at once.

Heterogeneity: DerSimonian-Laird random effects (`mu` pooled effect, `tau` between-patient SD,
`I2`, `pi_lo`/`pi_hi` prediction interval, `q_Q`); COPA and the outlier sum with a
1000-draw sign-flip null; two-component Gaussian mixture BIC; hierarchical donor clustering
with a silhouette cut and at least 3 donors per cluster.

Power limits that apply to every result: treatment is almost entirely confounded with study (ECP
is the only agent with two usable arms, and it is concentrated in buus and ren); there are 2
early-stage donors against 35 advanced; response and survival are essentially absent. I2
saturates near 1 because each donor contributes thousands of cells, so rank consistency on `tau`,
not on I2. Everything is transcript, in a 10,000-gene HVG subset.

## §1 Power table

**What:** coverage of every candidate stratification over the paired donors.

**How:** count of non-missing values per field, split by diagnosis.

**Full table on disk:** `tables/tgt44_power_table.csv`

```csv
field,n,MF,SS,levels
dx2 (MF vs SS),52,8,37,"SS:37, MF:8, unresolved:7"
sex,52,8,37,"M:30, F:22"
age,44,7,31,"median 70, range 32-85"
stage,36,6,24,"IVA1:13, IVA:9, IVB:4, IIB:3, IVA2:3"
stage class,36,6,24,"advanced:34, early:2"
B stage,25,6,17,"B2:13, B1:11, B0:1"
blood tumour %,20,5,11,"median 59, range 13-98"
treatment at sampling known,44,8,34,"median 1, range 0-1"
treatment-naive,44,8,31,"no:40, yes:4"
prior lines,4,0,0,"median 2, range 0-4"
response,3,1,0,"mixed:2, PD:1"
outcome,4,1,1,"DOD:2, died_unspecified:1, died_other:1"
drug sensitivity,4,0,0,"romidepsin_resistant:3, romidepsin_sensitive:1"
malignant burden,52,8,37,"median 0.858, range 0.05-0.994"
```

## §1 Treatment at sampling

**What:** donors on each treatment class, and in which studies.

**How:** membership of the curated clin_current_tx_classes string; a donor can be on several.

**Full table on disk:** `tables/tgt44_treatment_counts.csv`

```csv
index,donors,MF,SS,studies
ecp,25,5,20,"borcherding2019, borcherding2023, buus2025, herrera2021, ren2023"
retinoid,14,4,10,"borcherding2023, buus2025, ren2023"
interferon,6,1,5,"buus2025, ren2023"
hdaci,6,1,4,"borcherding2019, borcherding2023, buus2025, geskin2026, ren2023"
systemic_steroid,2,0,2,buus2025
mogamulizumab,1,0,1,buus2025
methotrexate,1,0,1,buus2025
other_targeted,1,0,1,gaydosik2022
alemtuzumab,1,0,1,buus2025
```

## §2a Disease, MF vs 
... [27025 more chars truncated]
```
