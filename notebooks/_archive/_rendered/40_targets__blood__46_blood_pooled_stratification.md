# 40_targets/blood/46_blood_pooled_stratification.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/blood/46_blood_pooled_stratification.ipynb` and is never run._

# 46 — Blood, pooled: malignant vs reactive CD4 across all cells, and by who the patient is

43 asks the malignant-vs-reactive question **inside each donor**. That is the right test for the
screen and the wrong shape for 44's question: pairing spends the cohort on the 52 donors that
carry both arms, and every clinical stratum then has to be cut out of those 52 — 8 MF, 2 early
stage. This notebook **pools** instead. All malignant CD4 cells are compared with all reactive CD4
cells, and then the same comparison is made inside, and between, clinical strata: disease, stage,
B stage, treatment at sampling, sex, age and burden.

Pooling gains two things, and there is one thing it cannot do.

- **Donors with one arm count.** 6 malignant-only and 5 reactive-only donors join the 52, and 12
  healthy donors form a third pool.
- **Strata become comparisons of cell populations.** "Malignant cells of stage IV patients vs
  malignant cells of stage I–III patients" is a well-defined contrast whether or not those
  patients have a reactive arm.
- It does **not** create metadata. Early stage is still 2 donors with a malignant arm, and
  treatment is still confounded with study. Pooling adds cells, not patients, and cells are not
  the unit of inference.

Two design choices, set in §0 and stated on every table:

| choice | headline | beside it |
|---|---|---|
| **weight** | donor-balanced: every donor contributes the same total weight to its arm | cell-weighted: every cell once |
| **inference** | donor cluster bootstrap for CIs; permutation of donor labels **within study** for between-stratum p | free permutation; van Elteren (study-blocked, donor-level) for the headline |

Why donor-balanced: in the cell pool, three harro2023 early-MF donors with no malignant arm hold
half of all reactive cells. "All malignant vs all reactive" would then mostly be "SS tumour vs
harro early MF". §1 shows this before anything is tested. No p-value is computed over cells
(`degradome_data.CAVEATS["cell_level_pvalues"]`).

| § | what it does |
|---|---|
| §0b | the cohort, and what pooling adds to the paired set |
| §1 | who each pool is made of, under each weighting |
| **§2** | **the headline pooled contrast**: log2FC, detection, Cohen's d and AUROC, against reactive and healthy CD4 |
| §3 | the power table for the pooled cohort, with tiers |
| **§4** | **stratified**: within each stratum, tumour cells across strata, reactive cells across strata, and the difference-in-differences |
| §5 | a joint model on donor pseudobulk, malignant and reactive arms |
| §6 | dossier |

**Inputs.**
- `jobs/run_target_screen_qc.sh --tissue blood` writes the long table.
- `jobs/run_target_pool_hist.sh --tissue blood` writes the per-group histograms behind the AUROC.
  It refuses to write unless it reproduces the long table's cell counts and detection rates.
- 43 supplies the shortlist and the artifact evidence.

> Everything here is transcript, in the 10,000-gene HVG object. `CD4` and `TNFRSF8` (CD30) are
> not in it.

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
import target_pool as P
import cluster_malig_helpers as CM
import malignancy as MAL

CM.use_tissue("blood")
MAL.use_tissue("blood")
TQ.use_tissue("blood")           # 42/47/48 set it to skin, and the module keeps the last value
assert MAL.CLUSTER_PQ.name.startswith("blood_"), MAL.CLUSTER_PQ
assert "LYZ" in TQ.BAND_GENES and "KRT14" not in TQ.BAND_GENES, TQ.BAND_GENES

TISSUE = "blood"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt46_"
MALIG, REACT, HC_CD4 = "CD4_malignant", "CD4_reactive", "CD4_HC"
WEIGHT = "donor"          # the headline pool; "cells" is printed beside it
# 50,000 permutations, not 2,000: BH over 1,098 genes puts a single hit at p < 4.6e-5, and
# 2,000 permutations cannot go below 1/2001 = 5e-4 -- every stratum would read as null by
# construction.
N_BOOT, N_PERM = 1000, 50000
Q = 0.05

ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX
TF.rc()
pd.set_option("display.width", 240)

man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
hman = json.loads((TARGETS / f"{TISSUE}_target_hist_manifest.json").read_text())
assert man.get("builder") == "run_target_screen_qc.py", (
    "run jobs/run_target_screen_qc.sh --tissue blood first")
# The histograms are valid only for the long table they were checked against: a rebuilt long
# table (another call, another cohort rule) would make them describe different cells.
assert hman["malignancy_call"] == man["malignancy_call"], (hman["malignancy_call"],
                                                         man["malignancy_call"])
assert hman["written"] >= man["written"], (
    f"histograms ({hman['written']}) are older than the long table ({man['written']}) -- "
    "rerun jobs/run_target_pool_hist.sh --tissue blood")

print("NB_DIR =", NB_DIR)
print(f"long: {man['written']} | call {man['malignancy_call']} | "
      f"{man['n_genes_in_object']:,} genes in object (HVG subset)")
print(f"hist: {hman['written']} | {hman['n_bins']} bins of {hman['bin_width']} log1p CP10K, "
      f"bin 0 = exact zero | max |mean_hist - mean_all| = {hman['max_abs_mean_diff']:.4f}")
print(f"headline weight: {WEIGHT} | {N_BOOT} donor-bootstrap draws | {N_PERM:,} permutations "
      f"(smallest attainable p {1 / (N_PERM + 1):.1e})")
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
long: 2026-10-25 10:53 | call mal_cluster | 10,000 genes in object (HVG subset)
hist: 2026-10-25 16:47 | 201 bins of 0.05 log1p CP10K, bin 0 = exact zero | max |mean_hist - mean_all| = 0.0091
headline weight: donor | 1000 donor-bootstrap draws | 50,000 permutations (smallest attainable p 2.0e-05)
```

## §0b — The cohort, and what pooling adds

The cohort rule is the same as 43's (`target_cohort`): ex vivo drug wells and duplicate persons
are dropped, diagnosis comes from `clin_dx`, and healthy donors are moved out of the reactive arm.
The difference is who enters: every independent donor with **any** CD4 arm of ≥ 20 cells.

`burden` is the malignant share of a donor's CD4 compartment. A donor with only one arm has the
other arm below 20 cells, and that arm is counted as 0. Malignant-only donors therefore sit at 1
and reactive-only donors at 0. That is a bound, not a measurement, and it is the only place a
missing arm is filled in.

```python  # cell 3
# ============================================================================
# §0b  Cohort: every independent donor with any CD4 arm
# ============================================================================
don = TC.blood_donor_table()
long = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet"), don)
meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)
hist = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_hist.parquet")
                      .astype({"arm": str, "donor": str, "gene": str}), don)

mal, rea, hc = (P.panel(long, a, hist=hist) for a in (MALIG, REACT, HC_CD4))
assert mal.genes.equals(rea.genes) and mal.genes.equals(hc.genes)
GENES = mal.genes
STUDY = don["study"]
del hist

n_arm = meta.pivot_table(index="donor", columns="arm", values="n_cells", aggfunc="sum")
DON = don[don["cohort_keep"]].join(
    n_arm.reindex(columns=[MALIG, REACT, HC_CD4])
         .rename(columns={MALIG: "n_mal", REACT: "n_rea", HC_CD4: "n_hc"}))
has_m, has_r = DON["n_mal"].notna(), DON["n_rea"].notna()
DON["pattern"] = np.select([has_m & has_r, has_m, has_r, DON["n_hc"].notna()],
                           ["paired", "malignant only", "reactive only", "healthy"],
                           "no CD4 arm")
_m, _r = DON["n_mal"].fillna(0), DON["n_rea"].fillna(0)
DON["burden"] = (_m / (_m + _r)).where(has_m | has_r)
PT = DON[DON["pattern"].isin(["paired", "malignant only", "reactive only"])]

print(f"{len(GENES):,} genes | malignant {len(mal.donors)} donors / {int(mal.n.sum()):,} cells, "
      f"reactive {len(rea.donors)} / {int(rea.n.sum()):,}, "
      f"healthy CD4 {len(hc.donors)} / {int(hc.n.sum()):,}")
print("\nindependent donor keys by arm pattern:")
print(pd.crosstab(DON["pattern"], DON["dx"], margins=True).to_string())
print(f"\npaired: {int(DON['pattern'].eq('paired').sum())} -- the whole of 43/44. Pooling adds "
      f"{int(DON['pattern'].isin(['malignant only', 'reactive only']).sum())} patients with one "
      f"arm and {int(DON['pattern'].eq('healthy').sum())} healthy donors.")
print("\nthe donors only pooling can use:")
_c = ["study", "dx", "clin_stage", "clin_stage_class", "clin_current_tx_classes", "n_mal",
      "n_rea", "pattern"]
print(DON[DON["pattern"].isin(["malignant only", "reactive only"])][_c]
      .sort_values(["pattern", "study"]).to_string())
DON.drop(columns=[c for c in DON.columns if c.startswith(("prior_", "ever_"))]) \
   .to_csv(ST.tab("cohort"))
print("\nwrote", ST.tab("cohort"))
```

```
1,098 genes | malignant 58 donors / 295,656 cells, reactive 57 / 111,125, healthy CD4 12 / 40,400

independent donor keys by arm pattern:
dx              HC  MF  MF/SS  MF_or_SS_unresolved  SS  All
pattern                                                    
healthy         12   0      0                    0   0   12
malignant only   0   0      0                    0   6    6
paired           0   8      2                    5  37   52
reactive only    0   3      0                    1   1    5
All             12  11      2                    6  44   75

paired: 52 -- the whole of 43/44. Pooling adds 11 patients with one arm and 12 healthy donors.

the donors only pooling can use:
                         study                   dx clin_stage clin_stage_class  clin_current_tx_classes    n_mal    n_rea         pattern
donor                                                                                                                                     
B2__CTCL2      borcherding2023                   SS       IVA1         advanced                 retinoid  30525.0      NaN  malignant only
B5__PT01              buus2025                   SS        NaN              NaN  ecp;interferon;retinoid   3294.0      NaN  malignant only
B5__PT04              buus2025                   SS        NaN              NaN                    hdaci    434.0      NaN  malignant only
B4__SZ33_HTO5       geskin2026                   SS        IVA         advanced           ecp;interferon    257.0      NaN  malignant only
B7__SS3              harro2023                   SS       IVA1         advanced                      NaN  19592.0      NaN  malignant only
B6__P8                 ren2023                   SS        IVA         advanced            mogamulizumab   8160.0      NaN  malignant only
B2__CTCL3      borcherding2023                   SS       IVA1         advanced           ecp;interferon      NaN    815.0   reactive only
B8__1137           dorando2026  MF_or_SS_unresolved        IIB         advanced                      NaN      NaN   5640.0   reactive only
B7__MF1              harro2023                   MF          I            early                      NaN      NaN  33036.0   reactive only
B7__MF2              harro2023                   MF          I            early                      NaN      NaN  16871.0   reactive only
B7__MF3              harro2023                   MF          I            early                      NaN      NaN   7785.0   reactive only

wrote /home/projects/nyosef/zvise/MF/tables/tgt46_cohort.csv
```

## §1 — Who the pool is made of

A pool is a weighted sum over donors, and the weights decide whose cells the comparison is about.
Under cell weighting a donor counts in proportion to the number of cells they contributed. In
this atlas that number is set by sorting strategy and sequencing depth, not by biology.

The table gives the effective number of donors for each arm under each weighting (Kish,
(Σw)²/Σw²); the figure gives the study mix. Donor balance does not fix the study mix — a study
with ten donors still holds ten shares — but it stops a single deep sample from standing in for
an arm.

```python  # cell 5
# ============================================================================
# §1 F1  Composition of each pool, cell-weighted vs donor-balanced
# ============================================================================
comp, summ = {}, []
for pan in (mal, rea, hc):
    by_study, s = P.composition(pan, STUDY)
    comp[pan.arm] = by_study
    summ.append(s)
summ = pd.DataFrame(summ).set_index("arm")
print(summ.to_string(float_format=lambda v: f"{v:.3g}"))
for arm, t in comp.items():
    print(f"\n{arm}: share of the pool by study")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))

_r = comp[REACT]
_top = _r["share_cells"].idxmax()
studies = sorted(set().union(*[set(t.index) for t in comp.values()]))
scol = dict(zip(studies, plt.get_cmap("tab10").colors))
bars = [(f"{TF.ARM_SHORT.get(a, a)}, {w}", comp[a][f"share_{w}"])
        for a in (MALIG, REACT, HC_CD4) for w in ("cells", "donor")]
fig, ax = plt.subplots(figsize=(5, 3))
for i, (_, sh) in enumerate(bars):
    left = 0.0
    for st in studies:
        v = float(sh.get(st, 0.0))
        if v:
            ax.barh(i, v, left=left, color=scol[st], edgecolor="white", lw=0.4)
            left += v
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=6)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of the pool")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=scol[s]) for s in studies], studies, fontsize=5,
          frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title(f"Cell-weighted reactive pool is {_r.loc[_top, 'share_cells']:.0%} {_top}; "
             f"donor-balanced, {_r.loc[_top, 'share_donor']:.0%}", loc="left", fontsize=7.5)
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F1_pool_composition", caption_text=(
    f"Kish effective donors, cell-weighted: malignant {summ.loc[MALIG, 'kish_n_eff_cells']:.1f} "
    f"of {summ.loc[MALIG, 'n_donors']}, reactive {summ.loc[REACT, 'kish_n_eff_cells']:.1f} of "
    f"{summ.loc[REACT, 'n_donors']}, healthy {summ.loc[HC_CD4, 'kish_n_eff_cells']:.1f} of "
    f"{summ.loc[HC_CD4, 'n_donors']}."))
TF.save_source(pd.concat(comp, names=["arm"]).reset_index(), "F1_pool_composition_source")
TF.logfig("F1_pool_composition", "study share of each pool under each weighting", None)
summ.to_csv(ST.tab("pool_composition"))
print("\nwrote", ST.tab("pool_composition"))
```

```
n_donors  n_cells  kish_n_eff_cells  kish_n_eff_donor  top_donor  top_donor_share_cells  top_study_share_cells  top_study
arm                                                                                                                                     
CD4_malignant        58   295656              24.2                58  B2__CTCL2                  0.103                  0.273    ren2023
CD4_reactive         57   111125              7.75                57    B7__MF1                  0.297                   0.54  harro2023
CD4_HC               12    40400              6.02                12     B6__N2                  0.245                  0.672    ren2023

CD4_malignant: share of the pool by study
                  n_cells  share_cells  share_donor  n_donors
study                                                        
ren2023         80697.000        0.273        0.190        11
dorando2026     57462.000        0.194        0.086         5
harro2023       49878.000        0.169        0.069         4
buus2025        45993.000        0.156        0.293        17
borcherding2023 31719.000        0.107        0.069         4
geskin2026       9601.000        0.032        0.103         6
gaydosik2022     7356.000        0.025        0.052         3
herrera2021      6375.000        0.022        0.103         6
borcherding2019  6128.000        0.021        0.017         1
rindler2021_fi    447.000        0.002        0.017         1

CD4_reactive: share of the pool by study
                  n_cells  share_cells  share_donor  n_donors
study                                                        
harro2023       59972.000        0.540        0.105         6
dorando2026     15180.000        0.137        0.105         6
ren2023         14767.000        0.133        0.175        10
gaydosik2022     5991.000        0.054        0.053         3
buus2025         5647.000        0.051        0.263        15
herrera2021      4766.000        0.043        0.105         6
geskin2026       2422.000        0.022        0.088         5
borcherding2023  1651.000        0.015        0.070         4
rindler2021_fi    619.000        0.006        0.018         1
borcherding2019   110.000        0.001        0.018         1

CD4_HC: share of the pool by study
               n_cells  share_cells  share_donor  n_donors
study                                                     
ren2023      27151.000        0.672        0.250         3
geskin2026    7595.000        0.188        0.417         5
gaydosik2022  4118.000        0.102        0.250         3
herrera2021   1536.000        0.038        0.083         1

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F1_pool_composition.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F1_pool_composition.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F1_pool_composition_source.csv

wrote /home/projects/nyosef/zvise/MF/tables/tgt46_pool_composition.csv
```

## §2 — Headline: pooled malignant vs reactive CD4

Every non-healthy donor that has an arm enters that arm's pool. Four effect sizes are reported,
because they answer different questions:

- `log2fc` — the log2 ratio of pooled mean log1p CP10K, on 43's scale (pseudocount 0.05).
- `d_det` — the difference in the fraction of cells detected. This is the quantity a CAR is
  dosed against.
- `cohen_d` — the mean difference over the pooled SD. It stays small for a gene detected in a
  minority of cells, however large the ratio.
- `auroc` — the chance that a random malignant cell expresses more than a random reactive cell.
  0.5 is no separation. For a sparse gene it is bounded by detection: a gene detected in 12% of
  tumour cells cannot go much above 0.56.

The CIs come from a donor bootstrap: donors are resampled and bring both arms, so paired donors
stay paired. The p-value is van Elteren: donor-level means ranked within study, the test
`target_screen.blocked_selectivity` already provides for donors with one arm. Beside the headline
sit the cell-weighted pool, 43's paired effect, and the same pooled contrast against **healthy**
CD4.

The pooled shortlist gates mirror 43's:
- `artifact_flag == clean`, read from 43 and not re-derived;
- log2FC > 0.5, with the CI above 0;
- van Elteren q < 0.05;
- not a loss marker, and not a lineage control.

```python  # cell 7
# ============================================================================
# §2a  The pooled contrast, every gene
# ============================================================================
head = P.contrast(mal, rea, weight=WEIGHT, n_boot=N_BOOT, seed=0)
cells = P.contrast(mal, rea, weight="cells", n_boot=0)
vs_hc = P.contrast(mal, hc, weight=WEIGHT, n_boot=N_BOOT, seed=1)
paired = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=5)
vel = TS.blocked_selectivity(long, MALIG, REACT, block=STUDY.to_dict(), value="mean_all")

pool = (head
        .merge(cells[["gene", "log2fc", "d_det", "auroc"]]
               .rename(columns=lambda c: c if c == "gene" else f"{c}_cells"), on="gene")
        .merge(vs_hc[["gene", "log2fc", "log2fc_lo", "d_det", "pct_b", "auroc", "auroc_lo",
                      "auroc_hi"]]
               .rename(columns=lambda c: c if c == "gene" else f"{c}_vs_hc"), on="gene")
        .merge(vel[["gene", "z_blocked", "n_blocks", "q_blocked"]], on="gene", how="left")
        .merge(paired[["gene", "log2fc_paired", "n_donors_paired", "frac_donors_up", "q"]]
               .rename(columns={"q": "q_paired"}), on="gene", how="left")
        .merge(ST.annotate(head["gene"])[["gene", "cd_name", "stage"]], on="gene", how="left"))

AE = ST.TAB_DIR / "tgt43_artifact_evidence.csv"
if AE.exists():
    pool = pool.merge(pd.read_csv(AE)[["gene", "artifact_flag"]], on="gene", how="left")
else:
    pool["artifact_flag"] = np.nan
    print(f"*** {AE.name} missing -- run 43; the artifact gate is NOT applied below ***")
SL = ST.TAB_DIR / "tgt43_shortlist_blood.csv"
SHORT = pd.read_csv(SL)["gene"].tolist() if SL.exists() else []
pool["on_43_shortlist"] = pool["gene"].isin(SHORT)

gate = ((pool["artifact_flag"].eq("clean") | pool["artifact_flag"].isna())
        & (pool["log2fc"] > 0.5) & (pool["log2fc_lo"] > 0) & (pool["q_blocked"] < Q)
        & ~pool["gene"].isin(ST.LOSS_MARKERS) & ~pool["gene"].isin(TQ.BAND_GENES))
pool["pooled_shortlist"] = gate
PSHORT = pool[gate].sort_values("log2fc", ascending=False)["gene"].tolist()

RHO = stats.spearmanr(pool["log2fc"], pool["log2fc_paired"], nan_policy="omit").statistic
_rho_c = stats.spearmanr(pool["log2fc"], pool["log2fc_cells"], nan_policy="omit").statistic
print(f"{len(pool):,} genes | van Elteren testable for {int(pool['q_blocked'].notna().sum()):,}, "
      f"{int((pool['q_blocked'] < Q).sum()):,} at q < {Q}")
print(f"pooled (donor-balanced) vs paired log2FC:        Spearman rho = {RHO:.3f}")
print(f"pooled donor-balanced vs cell-weighted log2FC:   Spearman rho = {_rho_c:.3f}")

print(f"\npooled shortlist: {len(PSHORT)} genes pass every gate "
      f"({int((gate & pool['artifact_flag'].isna()).sum())} without an artifact verdict in 43); "
      f"{len(set(PSHORT) & set(SHORT))} of 43's {len(SHORT)} are among them")
_c = ["gene", "cd_name", "log2fc", "log2fc_lo", "log2fc_hi", "pct_a", "pct_b", "d_det", "auroc",
      "auroc_lo", "cohen_d", "log2fc_cells", "log2fc_paired", "q_blocked", "log2fc_vs_hc",
      "auroc_vs_hc", "on_43_shortlist"]
print(pool[gate].sort_values("log2fc", ascending=False)[_c]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\non 43's shortlist but NOT the pooled one, with the column that keeps it out:")
_miss = pool[pool["gene"].isin(SHORT) & ~gate]
print(_miss[["gene", "log2fc", "log2fc_lo", "q_blocked", "artifact_flag", "log2fc_paired",
             "n_donors_paired"]].to_string(index=False, float_format=lambda v: f"{v:.3g}")
      if len(_miss) else "  none")

print("\ncontrols -- CCR4 / KIR3DL2 / IL2RA up; CD7 and DPP4 down, with AUROC below 0.5:")
_ctl = [g for g in ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS if g in set(pool["gene"])]
print(pool.set_index("gene").loc[_ctl, ["log2fc", "log2fc_lo", "log2fc_hi", "d_det", "auroc",
                                        "auroc_lo", "auroc_hi", "log2fc_cells",
                                        "log2fc_paired", "q_blocked"]]
      .to_string(float_format=lambda v: f"{v:.3g}"))
pool.to_csv(ST.tab("pooled_all_genes"), index=False)
print("\nwrote", ST.tab("pooled_all_genes"))
```

```
1,098 genes | van Elteren testable for 1,098, 400 at q < 0.05
pooled (donor-balanced) vs paired log2FC:        Spearman rho = 0.834
pooled donor-balanced vs cell-weighted log2FC:   Spearman rho = 0.887

pooled shortlist: 78 genes pass every gate (0 without an artifact verdict in 43); 28 of 43's 30 are among them
      gene       cd_name  log2fc  log2fc_lo  log2fc_hi  pct_a    pct_b  d_det  auroc  auroc_lo  cohen_d  log2fc_cells  log2fc_paired  q_blocked  log2fc_vs_hc  auroc_vs_hc  on_43_shortlist
   KIR3DL2        CD158k    3.04       2.78        3.3  0.396   0.0214  0.374  0.688     0.653    0.977          3.37           2.79   1.04e-09          3.46        0.695             True
       TPO                  2.76       2.38        3.1  0.236  0.00888  0.227  0.614     0.583    0.702          2.96           1.85   1.81e-12          2.98        0.617             True
     RAB25                  1.98       1.53       2.36  0.142  0.00782  0.135  0.567     0.543    0.506          1.79          0.375   0.000295             2        0.568            False
     GPR15                  1.94       1.58        2.3  0.205   0.0338  0.172  0.587     0.562    0.535          2.08           1.44   1.28e-05          2.44        0.597             True
      CD70          CD70    1.94       1.68       2.22  0.251   0.0443  0.206  0.604     0.584    0.583          2.29           1.93   2.71e-09          2.38        0.615             True
     FCRL3                  1.91       1.59       2.24  0.214   0.0353  0.179   0.59     0.568    0.539          2.01           1.54   4.03e-06          2.27        0.598             True
     PTH2R                  1.74       1.33       2.13 0.0908  0.00311 0.0877  0.544      0.53    0.386          1.29          0.971   4.48e-08          1.82        0.545            False
      STAM                  1.64       1.39       1.86  0.396    0.124  0.272  0.642     0.612    0.665           1.8           1.64   1.25e-07          1.85        0.652             True
       CPM                   1.5       1.14       1.84  0.103   0.0144 0.0887  0.544      0.53    0.371          1.33            1.1   5.19e-09          1.61        0.547            False
    TSPAN2                  1.47       1.19       1.75  0.214    0.058  0.156  0.579      0.56    0.458          1.69          0.938   9.26e-09          1.96        0.593             True
     TIGIT                  1.43       1.15       1.76  0.437    0.147   0.29  0.647     0.621    0.633          1.56           1.65   4.05e-09          1.46        0.649             True
     EPCAM                  1.38      0.815       1.82 0.0647  0.00126 0.0635  0.532     0.515    0.344         0.982         0.0183     0.0014           1.4        0.532            False
      PI16                  1.35       1.01       1.69   0.29    0.105  0.185  0.595     0.569    0.487          1.61           1.17   2.58e-06          1.92        0.617             True
     FGFR1                  1.32       1.08       1.53   0.11   0.0225 0.0872  0.544     0.533    0.347          1.29           1.15   5.19e-09          1.55        0.549            False
     NRROS                  1.28       1.08       1.46  0.192   0.0612  0.131  0.566     0.551    0.398           1.3            1.2   1.63e-08          1.35        0.569             True
     EPHA4                  1.19          1       1.38  0.217   0.0761  0.141  0.571     0.557    0.388          1.42          0.989   5.24e-11          1.53        0.583             True
      CD84 CD84 / SLAMF5    1.17       1.01       1.33  0.284    0.118  0.166  0.586      0.57    0.434           1.3           1.07   8.24e-12          1.43        0.599             True
ST6GALNAC1                  1.15       0.81       1.46  0.119   0.0367  0.082  0.542     0.525    0.319          1.34          0.361    0.00562          1.42        0.549            False
      CDH1                  1.14      0.822        1.4  0.055  0.00276 0.0522  0.526     0.517    0.295          1
... [12905 more chars truncated]
```

```python  # cell 8
# ============================================================================
# §2b F2  Does pooling reproduce the paired ranking
# ============================================================================
# The two are not the same estimator: paired is the median of per-donor ratios, pooled the ratio
# of the mean donor means. They part company for genes detected in a few donors at high level
# (a mean picks those up, a median does not), which is what the disagreement table lists.
d = pool.dropna(subset=["log2fc", "log2fc_paired"]).copy()
fig, ax = plt.subplots(figsize=(5, 3))
ax.axhline(0, lw=.5, c="k"); ax.axvline(0, lw=.5, c="k")
lim = [min(d["log2fc_paired"].min(), d["log2fc"].min()),
       max(d["log2fc_paired"].max(), d["log2fc"].max())]
ax.plot(lim, lim, ls="--", lw=.5, c="#bbbbbb")
ax.scatter(d["log2fc_paired"], d["log2fc"], s=4, c="#d5d8dc", linewidths=0)
k = d["gene"].isin(PSHORT)
ax.scatter(d.loc[k, "log2fc_paired"], d.loc[k, "log2fc"], s=10, c=TF.MALIG, linewidths=0,
           label=f"pooled shortlist ({int(k.sum())})")
k2 = d["gene"].isin(SHORT) & ~k
ax.scatter(d.loc[k2, "log2fc_paired"], d.loc[k2, "log2fc"], s=12, facecolors="none",
           edgecolors=TF.MALIG, linewidths=.6, label=f"43 shortlist only ({int(k2.sum())})")
kc = d["gene"].isin(ST.POSITIVE_CONTROLS + ["CD7", "DPP4"])
ax.scatter(d.loc[kc, "log2fc_paired"], d.loc[kc, "log2fc"], s=18, c=TF.DOWN, linewidths=0)
for _, r_ in d[kc].iterrows():
    ax.annotate(r_["gene"], (r_["log2fc_paired"], r_["log2fc"]), fontsize=5.5, xytext=(3, 2),
                textcoords="offset points")
ax.set_xlabel("paired: median per-donor log2FC (43)")
ax.set_ylabel("pooled: donor-balanced log2FC")
ax.set_title(f"Pooled ranking {'reproduces' if RHO >= 0.8 else 'departs from'} the paired one "
             f"(ρ = {RHO:.2f}, {len(d):,} genes)", loc="left", fontsize=7.5)
ax.legend(fontsize=5.5, frameon=False, loc="upper left")
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F2_pooled_vs_paired", caption_text=(
    "Red: CCR4, KIR3DL2, IL2RA, CD7, DPP4. Dashed: identity."))
TF.save_source(d[["gene", "log2fc", "log2fc_paired", "log2fc_cells", "pooled_shortlist",
                  "on_43_shortlist"]], "F2_pooled_vs_paired_source")
TF.logfig("F2_pooled_vs_paired", "pooled vs paired log2FC, every gene", len(d))

d["delta"] = d["log2fc"] - d["log2fc_paired"]
print("largest disagreements, pooled minus paired:")
print(d.reindex(d["delta"].abs().sort_values(ascending=False).index).head(15)
      [["gene", "log2fc", "log2fc_paired", "delta", "log2fc_cells", "pct_a", "pct_b",
        "n_donors_paired", "frac_donors_up", "pooled_shortlist"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F2_pooled_vs_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F2_pooled_vs_paired.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F2_pooled_vs_paired_source.csv
largest disagreements, pooled minus paired:
   gene  log2fc  log2fc_paired  delta  log2fc_cells  pct_a    pct_b  n_donors_paired  frac_donors_up  pooled_shortlist
  RAB25    1.98          0.375    1.6          1.79  0.142  0.00782               52           0.769              True
  EPCAM    1.38         0.0183   1.36         0.982 0.0647  0.00126               52           0.596              True
CYSLTR1   0.482         -0.836   1.32         0.395  0.163    0.118               52           0.385             False
  KLRB1      -1          -2.23   1.23         -1.28  0.104    0.224               52           0.154             False
   GNLY    0.73         -0.369    1.1          0.61  0.115   0.0723               52           0.404             False
KIR3DL1    1.13         0.0811   1.05          1.36  0.056  0.00478               52           0.615              True
  ITGA4   -1.78          -2.73  0.949         -1.59  0.096    0.379               52          0.0577             False
 COL6A2   0.959         0.0133  0.945          1.31 0.0769   0.0235               52             0.5             False
    TPO    2.76           1.85  0.905          2.96  0.236  0.00888               52           0.962              True
 GALNT2    1.02           0.14  0.877          1.44  0.137   0.0653               52           0.635             False
   NT5E    1.24          0.378  0.867           1.3 0.0967   0.0227               52           0.615             False
   CCR6  -0.316          -1.14  0.823       -0.0847  0.074    0.114               52           0.192             False
    CD9   0.962          0.162  0.799         0.775 0.0663    0.017               52           0.615             False
 ADAM29   0.828         0.0332  0.795         0.836 0.0294 0.000575               52           0.692              True
   CDH1    1.14          0.343  0.793          1.02  0.055  0.00276               52           0.731              True
```

```python  # cell 9
# ============================================================================
# §2c F3  Cell-level separation for the shortlist, and leave-one-study-out
# ============================================================================
SHOW = (PSHORT or SHORT)[:15]
r = pool.set_index("gene").loc[SHOW].sort_values("auroc")
y = np.arange(len(r))
fig, ax = plt.subplots(figsize=(5, 3))
ax.axvline(0.5, lw=.5, c="k")
ax.errorbar(r["auroc"], y + .15, xerr=[r["auroc"] - r["auroc_lo"], r["auroc_hi"] - r["auroc"]],
            fmt="o", ms=3, lw=.8, color=TF.MALIG, label="vs reactive CD4")
ax.errorbar(r["auroc_vs_hc"], y - .15,
            xerr=[r["auroc_vs_hc"] - r["auroc_lo_vs_hc"], r["auroc_hi_vs_hc"] - r["auroc_vs_hc"]],
            fmt="o", ms=3, lw=.8, color=TF.COMPARATOR, label="vs healthy CD4")
ax.set_yticks(y)
ax.set_yticklabels([TF.it(g) for g in r.index], fontsize=6)
ax.set_xlabel("pooled AUROC, malignant cells vs comparator")
n_sep = int((r["auroc_lo"] > 0.5).sum())
ax.set_title(f"{n_sep} of {len(r)} leading antigens separate tumour from reactive cells "
             "(CI > 0.5)", loc="left", fontsize=7.5)
ax.legend(fontsize=5.5, frameon=False, loc="lower right")
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F3_shortlist_auroc", caption_text=(
    "Donor-balanced pools; bars: 95% donor-bootstrap CI. AUROC is capped by detection for "
    "sparse genes."))
TF.save_source(r.reset_index(), "F3_shortlist_auroc_source")
TF.logfig("F3_shortlist_auroc", "cell-level separation of the leading antigens", len(r))

print("AUROC beside detection -- a sparse antigen cannot separate many cells, however selective:")
print(r[["pct_a", "pct_b", "pct_b_vs_hc", "d_det", "auroc", "auroc_lo", "auroc_hi",
         "auroc_vs_hc", "cohen_d"]].iloc[::-1].to_string(float_format=lambda v: f"{v:.3g}"))

LO_GENES = list(dict.fromkeys(SHOW + [g for g in ST.POSITIVE_CONTROLS + ["CD7", "DPP4"]
                                      if g in set(GENES)]))
LO = P.loso(mal, rea, STUDY, genes=LO_GENES, weight=WEIGHT)
_base = pool.set_index("gene").loc[LO.index, "log2fc"]
lo_sum = pd.DataFrame({"pooled": _base, "loso_min": LO.min(1), "loso_max": LO.max(1),
                       "study_moving_most": LO.sub(_base, axis=0).abs().idxmax(1),
                       "sign_kept_every_study": (np.sign(LO).eq(np.sign(_base), axis=0)).all(1)})
print("\nleave-one-study-out pooled log2FC:")
print(lo_sum.to_string(float_format=lambda v: f"{v:.3g}"))
LO.to_csv(ST.tab("loso"))
print("\nwrote", ST.tab("loso"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F3_shortlist_auroc.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F3_shortlist_auroc.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F3_shortlist_auroc_source.csv
AUROC beside detection -- a sparse antigen cannot separate many cells, however selective:
         pct_a   pct_b  pct_b_vs_hc  d_det  auroc  auroc_lo  auroc_hi  auroc_vs_hc  cohen_d
gene                                                                                       
KIR3DL2  0.396  0.0214        0.007  0.374  0.688     0.653     0.723        0.695    0.977
TIGIT    0.437   0.147        0.142   0.29  0.647     0.621     0.678        0.649    0.633
STAM     0.396   0.124        0.101  0.272  0.642     0.612      0.67        0.652    0.665
TPO      0.236 0.00888      0.00171  0.227  0.614     0.583     0.647        0.617    0.702
CD70     0.251  0.0443       0.0224  0.206  0.604     0.584     0.623        0.615    0.583
PI16      0.29   0.105       0.0587  0.185  0.595     0.569     0.623        0.617    0.487
FCRL3    0.214  0.0353       0.0177  0.179   0.59     0.568     0.615        0.598    0.539
GPR15    0.205  0.0338       0.0124  0.172  0.587     0.562     0.614        0.597    0.535
TSPAN2   0.214   0.058       0.0297  0.156  0.579      0.56       0.6        0.593    0.458
RAB25    0.142 0.00782      0.00648  0.135  0.567     0.543     0.594        0.568    0.506
NRROS    0.192  0.0612       0.0543  0.131  0.566     0.551     0.582        0.569    0.398
CPM      0.103  0.0144      0.00943 0.0887  0.544      0.53     0.561        0.547    0.371
PTH2R   0.0908 0.00311     0.000568 0.0877  0.544      0.53     0.561        0.545    0.386
FGFR1     0.11  0.0225        0.013 0.0872  0.544     0.533     0.556        0.549    0.347
EPCAM   0.0647 0.00126     0.000683 0.0635  0.532     0.515      0.55        0.532    0.344

leave-one-study-out pooled log2FC:
         pooled  loso_min  loso_max study_moving_most  sign_kept_every_study
gene                                                                        
KIR3DL2    3.04      2.99      3.17        geskin2026                   True
TPO        2.76      2.61      2.87   borcherding2023                   True
RAB25      1.98      1.74      2.09          buus2025                   True
GPR15      1.94      1.68      2.01          buus2025                   True
CD70       1.94      1.88      2.02        geskin2026                   True
FCRL3      1.91      1.74      2.06          buus2025                   True
PTH2R      1.74      1.07      1.84          buus2025                   True
STAM       1.64      1.57      1.69          buus2025                   True
CPM         1.5      1.38      1.61          buus2025                   True
TSPAN2     1.47      1.35      1.63          buus2025                   True
TIGIT      1.43      1.36      1.61        geskin2026                   True
EPCAM      1.38      1.16      1.47   borcherding2023                   True
PI16       1.35      1.13      1.45           ren2023                   True
FGFR1      1.32      1.18      1.37           ren2023                   True
NRROS      1.28      1.22      1.33           ren2023                   True
CCR4       1.03     0.991      1.08    rindler2021_fi                   True
IL2RA    -0.215    -0.345   -0.0985          buus2025                   True
CD7       -1.55     -1.61     -1.48      gaydosik2022                   True
DPP4      -1.85     -1.89     -1.83         harro2023                   True

wrote /home/projects/nyosef/zvise/MF/tables/tgt46_loso.csv
```

## §3 — What the pooled cohort can answer

The tiers are 44's — **1** test, **2** describe (an effect with a CI, no p), **3** count only — but
they are now set per arm, because a stratum can be well populated on one side and empty on the
other. Early stage is the clear case: 5 donors with a reactive arm, and 2 with a malignant one.

Every field is a two-level split, `x` vs `y`, and log2FC is always x over y. The study mix of
each level is printed beside the counts. A split that is also a study split can only be tested by
the within-study permutation, and that permutation then has few arrangements to work with
(`log10_configs_study` in §4).

```python  # cell 11
# ============================================================================
# §3  Strata, counts per arm, tiers
# ============================================================================
def _yn(flag, known=None):
    s = flag.map({True: "yes", False: "no"})
    return s.where(known) if known is not None else s


def _split(v, hi="high", lo="low"):
    med = v.median()
    return v.map(lambda a: np.nan if pd.isna(a) else hi if a >= med else lo), med


_stage = PT["clin_stage"].map(
    lambda s: np.nan if pd.isna(s) else ("IV" if str(s).startswith("IV") else "I-III"))
_age, AGE_MED = _split(PT["age"], "older", "younger")
_bur, BUR_MED = _split(PT["burden"])
_btp, BTP_MED = _split(PT["blood_tumor_pct"])
FIELDS = {
    "disease": (PT["dx2"], "MF", "SS"),
    "stage IV vs I-III": (_stage, "IV", "I-III"),
    "stage class": (PT["clin_stage_class"], "advanced", "early"),
    "B stage": (PT["clin_B"].map({"B2": "B2", "B1": "B0-B1", "B0": "B0-B1"}), "B2", "B0-B1"),
    "ECP": (_yn(PT["tx_ecp"], PT["tx_known"]), "yes", "no"),
    "retinoid": (_yn(PT["tx_retinoid"], PT["tx_known"]), "yes", "no"),
    "interferon": (_yn(PT["tx_interferon"], PT["tx_known"]), "yes", "no"),
    "HDACi": (_yn(PT["tx_hdaci"], PT["tx_known"]), "yes", "no"),
    "treatment-naive": (PT["clin_tx_naive"], "yes", "no"),
    "sex": (PT["sex"], "F", "M"),
    f"age (median {AGE_MED:.0f})": (_age, "older", "younger"),
    f"burden (median {BUR_MED:.2f})": (_bur, "high", "low"),
    f"blood tumour % (median {BTP_MED:.0f})": (_btp, "high", "low"),
    "cultured 2 days ex vivo": (_yn(PT["is_cultured"]), "yes", "no"),
}

rows = []
for name, (lab, x, y_) in FIELDS.items():
    lab = lab.dropna()
    rec = {"field": name, "x": x, "y": y_}
    for tag, pan in (("mal", mal), ("rea", rea)):
        for s, side in ((x, "x"), (y_, "y")):
            rec[f"n_{tag}_{side}"] = int(pan.donors.isin(lab.index[lab.eq(s)]).sum())
        rec[f"tier_{tag}"] = P.tier(rec[f"n_{tag}_x"], rec[f"n_{tag}_y"])
    rec["tier_did"] = max(rec["tier_mal"], rec["tier_rea"])
    for s, side in ((x, "x"), (y_, "y")):
        rec[f"studies_{side}"] = ", ".join(
            f"{k}:{v}" for k, v in STUDY.reindex(lab.index[lab.eq(s)]).value_counts().items())
    rows.append(rec)
POWER = pd.DataFrame(rows).set_index("field")
print("donors per arm and level (mal = with a malignant arm, rea = with a reactive arm):")
print(POWER[[c for c in POWER.columns if not c.startswith("studies")]].to_string())
print("\nstudy mix per level:")
for f, r_ in POWER.iterrows():
    print(f"  {f:34s} {r_['x']:>9s}: {r_['studies_x']}")
    print(f"  {'':34s} {r_['y']:>9s}: {r_['studies_y']}")
print("\ntier 1 = test, 2 = describe, 3 = count only. Treatment fields use only donors whose "
      "treatment at sampling is known.")
POWER.to_csv(ST.tab("power_table"))
print("wrote", ST.tab("power_table"))
```

```
donors per arm and level (mal = with a malignant arm, rea = with a reactive arm):
                                   x        y  n_mal_x  n_mal_y  tier_mal  n_rea_x  n_rea_y  tier_rea  tier_did
field                                                                                                          
disease                           MF       SS        8       43         1       11       38         1         1
stage IV vs I-III                 IV    I-III       33        7         1       30       11         1         1
stage class                 advanced    early       38        2         3       36        5         1         3
B stage                           B2    B0-B1       15       12         1       13       15         1         1
ECP                              yes       no       27       23         1       26       22         1         1
retinoid                         yes       no       16       34         1       14       34         1         1
interferon                       yes       no        8       42         1        7       41         1         1
HDACi                            yes       no        7       43         1        6       42         1         1
treatment-naive                  yes       no        4       45         2        4       42         2         2
sex                                F        M       24       34         1       24       33         1         1
age (median 70)                older  younger       29       20         1       28       21         1         1
burden (median 0.90)            high      low       32       26         1       26       31         1         1
blood tumour % (median 59)      high      low       11       10         1       10       11         1         1
cultured 2 days ex vivo          yes       no        6       52         1        5       52         1         1

study mix per level:
  disease                                   MF: ren2023:4, harro2023:3, buus2025:2, rindler2021_fi:1, herrera2021:1
                                            SS: buus2025:15, ren2023:7, geskin2026:5, borcherding2023:5, herrera2021:5, harro2023:4, gaydosik2022:2, borcherding2019:1
  stage IV vs I-III                         IV: ren2023:7, herrera2021:6, geskin2026:5, borcherding2023:5, dorando2026:4, harro2023:3, gaydosik2022:2, borcherding2019:1, rindler2021_fi:1
                                         I-III: ren2023:4, harro2023:4, geskin2026:1, dorando2026:1, gaydosik2022:1
  stage class                         advanced: ren2023:10, herrera2021:6, geskin2026:6, dorando2026:5, borcherding2023:5, gaydosik2022:3, harro2023:3, borcherding2019:1, rindler2021_fi:1
                                         early: harro2023:4, ren2023:1
  B stage                                   B2: ren2023:6, herrera2021:4, harro2023:2, gaydosik2022:2, borcherding2019:1
                                         B0-B1: ren2023:5, harro2023:5, herrera2021:2, geskin2026:1, gaydosik2022:1, rindler2021_fi:1
  ECP                                      yes: buus2025:12, ren2023:10, borcherding2023:3, borcherding2019:1, geskin2026:1, herrera2021:1
                                            no: harro2023:7, buus2025:5, herrera2021:5, gaydosik2022:3, borcherding2023:2, geskin2026:2, ren2023:1, rindler2021_fi:1
  retinoid                                 yes: ren2023:10, buus2025:4, borcherding2023:2
                                            no: buus2025:13, harro2023:7, herrera2021:6, gaydosik2022:3, borcherding2023:3, geskin2026:3, borcherding2019:1, ren2023:1, rindler2021_fi:1
  interferon                               yes: buus2025:4, ren2023:3, geskin2026:1, borcherding2023:1
                                            no: buus2025:13, ren2023:8, harro2023:7, herrera2021:6, borcherding2023:4, gaydosik2022:3, geskin2026:2, borcherding2019:1, rindler2021_fi:1
  HDACi                                    yes: buus2025:3, borcherding2019:1, borcherding2023:1, geskin2026:1, ren2023:1
               
... [2062 more chars truncated]
```

## §4 — Stratified: does the antigen depend on who the patient is

Four contrasts per field, each answering a different question:

| contrast | compares | question |
|---|---|---|
| **(a) within** | malignant vs reactive, inside stratum x and inside stratum y | is the antigen tumour-selective in this kind of patient |
| **(b) tumour** | malignant cells of x vs malignant cells of y | does the tumour itself differ between these patients |
| **(c) reactive** | reactive cells of x vs reactive cells of y | does every T cell differ — treatment and stage can act on the whole compartment |
| **(d) DiD** | (a) in x minus (a) in y | does the tumour-vs-reactive effect differ: the interaction 44 could test on paired donors only |

How the contrasts combine:
- (b) without (c) is a tumour-specific difference.
- (b) and (c) in the same direction is a compartment-wide difference, and (d) is then near zero.

The p-values come from permuting donors.
- `q_study` permutes only within study. It is the one to read, because disease and treatment are
  concentrated in particular studies.
- `q_free` is printed beside it, to show how much of a free-permutation signal is study.
- When a split cannot be rearranged within study in at least 20 ways, `q_study` is left empty
  rather than computed from a handful of permutations.

Tier-2 rows carry effects only, and tier-3 rows counts only.

```python  # cell 13
# ============================================================================
# §4a  Every field, four contrasts
# ============================================================================
tum, rct, dd, wit = [], [], [], []
for i, (name, (lab, x, y_)) in enumerate(FIELDS.items()):
    lab = lab.dropna()
    pw = POWER.loc[name]
    tag = dict(field=name, x=x, y=y_)
    tum.append(P.between(mal, lab, x, y_, block=STUDY, seed=i,
                         n_perm=N_PERM if pw["tier_mal"] == 1 else 0)
               .assign(**tag, tier=pw["tier_mal"]))
    rct.append(P.between(rea, lab, x, y_, block=STUDY, seed=100 + i,
                         n_perm=N_PERM if pw["tier_rea"] == 1 else 0)
               .assign(**tag, tier=pw["tier_rea"]))
    dd.append(P.did(mal, rea, lab, x, y_, block=STUDY, seed=200 + i,
                    n_perm=N_PERM if pw["tier_did"] == 1 else 0)
              .assign(**tag, tier=pw["tier_did"]))
    for s in (x, y_):
        dn = lab.index[lab.eq(s)]
        A, B = mal.sub(dn), rea.sub(dn)
        if min(len(A.donors), len(B.donors)) >= P.DESCRIBE_MIN:
            wit.append(P.contrast(A, B, weight=WEIGHT, n_boot=N_BOOT, seed=300 + i)
                       .assign(field=name, stratum=s))
TUM, RCT, DID, WIT = (pd.concat(t, ignore_index=True) for t in (tum, rct, dd, wit))
for c in ("q_study", "q_free", "p_study", "p_free", "log10_configs_study"):
    for T in (TUM, RCT, DID):
        if c not in T:
            T[c] = np.nan


def _n_sig(T, col):
    return int((T[col] < Q).sum()) if T[col].notna().any() else np.nan


rows = []
for name in FIELDS:
    t, r_, d_ = (T[T["field"].eq(name)] for T in (TUM, RCT, DID))
    pw = POWER.loc[name]
    rows.append({
        "field": name, "x vs y": f"{pw['x']} vs {pw['y']}",
        "mal n": f"{pw['n_mal_x']}/{pw['n_mal_y']}", "rea n": f"{pw['n_rea_x']}/{pw['n_rea_y']}",
        "tier mal/rea/did": f"{pw['tier_mal']}/{pw['tier_rea']}/{pw['tier_did']}",
        "log10 cfg study": t["log10_configs_study"].iloc[0],
        "tumour q_study": _n_sig(t, "q_study"), "tumour q_free": _n_sig(t, "q_free"),
        "reactive q_study": _n_sig(r_, "q_study"), "reactive q_free": _n_sig(r_, "q_free"),
        "DiD q_study": _n_sig(d_, "q_study"), "DiD q_free": _n_sig(d_, "q_free"),
    })
SUMMARY = pd.DataFrame(rows).set_index("field")
print(f"genes at q < {Q}, per field and contrast (of {len(GENES):,}); empty = not tested at "
      "that tier, or fewer than 20 within-study arrangements:")
print(SUMMARY.to_string(float_format=lambda v: f"{v:.3g}"))

TUM.to_csv(ST.tab("strat_tumour"), index=False)
RCT.to_csv(ST.tab("strat_reactive"), index=False)
DID.to_csv(ST.tab("strat_did"), index=False)
WIT.to_csv(ST.tab("strat_within"), index=False)
SUMMARY.to_csv(ST.tab("strat_summary"))
print("\nwrote", *(ST.tab(n) for n in ("strat_tumour", "strat_reactive", "strat_did",
                                      "strat_within", "strat_summary")), sep="\n  ")
```

```
genes at q < 0.05, per field and contrast (of 1,098); empty = not tested at that tier, or fewer than 20 within-study arrangements:
                                       x vs y  mal n  rea n tier mal/rea/did  log10 cfg study  tumour q_study  tumour q_free  reactive q_study  reactive q_free  DiD q_study  DiD q_free
field                                                                                                                                                                                   
disease                              MF vs SS   8/43  11/38            1/1/1             5.43               0              0                 0                0            0           0
stage IV vs I-III                 IV vs I-III   33/7  30/11            1/1/1             4.38               0              0                 0                0            1           9
stage class                 advanced vs early   38/2   36/5            3/1/3              NaN             NaN            NaN                 0                0          NaN         NaN
B stage                           B2 vs B0-B1  15/12  13/15            1/1/1              5.1               0              0                 0                0            0           0
ECP                                 yes vs no  27/23  26/22            1/1/1             6.87               0              0                 0                2            0           2
retinoid                            yes vs no  16/34  14/34            1/1/1              5.2               0              2                 0                0            0           0
interferon                          yes vs no   8/42   7/41            1/1/1             6.07               0              0                 0                0            0           0
HDACi                               yes vs no   7/43   6/42            1/1/1             4.95               0              0                 0                0            0           0
treatment-naive                     yes vs no   4/45   4/42            2/2/2              NaN             NaN            NaN               NaN              NaN          NaN         NaN
sex                                    F vs M  24/34  24/33            1/1/1             11.2               0              0                 0                0            0           0
age (median 70)              older vs younger  29/20  28/21            1/1/1                8               0              1                 0                0            0           0
burden (median 0.90)              high vs low  32/26  26/31            1/1/1             9.12               0              0                 8                9            0           0
blood tumour % (median 59)        high vs low  11/10  10/11            1/1/1             4.44               0              0                 0                0            0           0
cultured 2 days ex vivo             yes vs no   6/52   5/52            1/1/1                0             NaN            152               NaN              212          NaN           0


wrote
  /home/projects/nyosef/zvise/MF/tables/tgt46_strat_tumour.csv
  /home/projects/nyosef/zvise/MF/tables/tgt46_strat_reactive.csv
  /home/projects/nyosef/zvise/MF/tables/tgt46_strat_did.csv
  /home/projects/nyosef/zvise/MF/tables/tgt46_strat_within.csv
  /home/projects/nyosef/zvise/MF/tables/tgt46_strat_summary.csv
```

```python  # cell 14
# ============================================================================
# §4b  Per field: the strongest tumour-cell differences, with the reactive arm beside them
# ============================================================================
key = ["field", "gene"]
J = (TUM.merge(RCT[key + ["log2fc", "q_study", "q_free"]]
               .rename(columns={"log2fc": "log2fc_rea", "q_study": "q_study_rea",
                                "q_free": "q_free_rea"}), on=key, how="left")
        .merge(DID[key + ["log2fc_x", "log2fc_y", "did", "q_study", "q_free"]]
               .rename(columns={"q_study": "q_study_did", "q_free": "q_free_did"}),
               on=key, how="left"))
_sig_t = J["q_study"] < Q
_sig_r = J["q_study_rea"] < Q
_same = np.sign(J["log2fc"]) == np.sign(J["log2fc_rea"])
J["reading"] = np.select([_sig_t & ~_sig_r, _sig_t & _sig_r & _same, _sig_t & _sig_r & ~_same],
                         ["tumour-specific", "compartment-wide", "opposite in reactive"], "")
J["on_shortlist"] = J["gene"].isin(set(PSHORT) | set(SHORT))

_c = ["gene", "log2fc", "pct_x", "pct_y", "auroc", "q_study", "q_free", "log2fc_rea",
      "q_study_rea", "did", "q_study_did", "reading", "on_shortlist"]
for name in FIELDS:
    pw = POWER.loc[name]
    j = J[J["field"].eq(name)]
    print("\n" + "=" * 78)
    print(f"{name}: {pw['x']} vs {pw['y']} | tumour cells from {pw['n_mal_x']} vs "
          f"{pw['n_mal_y']} donors | tier {pw['tier_mal']}")
    print("=" * 78)
    if pw["tier_mal"] == 3:
        print("  count only -- fewer than 3 donors with a malignant arm on one side")
        continue
    if j["q_study"].notna().any():
        top = j.sort_values(["q_study", "p_study"]).head(12)
        print(f"  {int((j['q_study'] < Q).sum())} genes at q_study < {Q} "
              f"({int((j['q_free'] < Q).sum())} at q_free); readings: "
              f"{j.loc[j['q_study'] < Q, 'reading'].value_counts().to_dict()}")
    else:
        top = j.reindex(j["log2fc"].abs().sort_values(ascending=False).index).head(12)
        print("  no within-study p (tier 2, or too few arrangements) -- largest effects only")
    print(top[[c for c in _c if c in top]].to_string(index=False,
                                                    float_format=lambda v: f"{v:.3g}"))

print("\n" + "=" * 78)
print("the leading antigens on tumour cells, every field (log2FC x vs y):")
_g = (PSHORT or SHORT)[:15]
print(TUM[TUM["gene"].isin(_g)].pivot_table(index="gene", columns="field", values="log2fc")
      .reindex(index=_g, columns=list(FIELDS)).round(2).to_string())
J.to_csv(ST.tab("strat_joined"), index=False)
print("\nwrote", ST.tab("strat_joined"))
```

```
==============================================================================
disease: MF vs SS | tumour cells from 8 vs 43 donors | tier 1
==============================================================================
  0 genes at q_study < 0.05 (0 at q_free); readings: {}
   gene  log2fc    pct_x    pct_y  auroc  q_study  q_free  log2fc_rea  q_study_rea     did  q_study_did reading  on_shortlist
  TGFB1   -0.58    0.325    0.471  0.418    0.966       1      0.0249            1  -0.604        0.475                 False
 PTCHD4  0.0071 0.000246 2.93e-06    0.5    0.966       1    0.000784            1 0.00631        0.958                 False
  ATP4B  0.0115 0.000266 1.54e-05    0.5    0.966       1    0.000742            1  0.0108        0.251                 False
   CD33  0.0977  0.00382  0.00166  0.501        1       1      0.0357            1   0.062            1                 False
   GYPA  0.0142  0.00065 0.000126    0.5        1       1    -0.00565            1  0.0199        0.251                 False
   FLT1  -0.479  0.00375    0.022  0.491        1       1      0.0148            1  -0.493        0.504                 False
   CDH9 0.00652 0.000206 2.21e-05    0.5        1       1    0.000479            1 0.00604        0.251                 False
    GP6 -0.0669 0.000346  0.00234  0.499        1       1     -0.0504            1 -0.0164            1                 False
TMEM104   -0.31    0.029   0.0503  0.489        1       1     -0.0587            1  -0.252        0.958                 False
KIR2DL3   -1.17  0.00122   0.0447  0.478        1       1      -0.202            1  -0.963        0.965                 False
  TMED4   0.287    0.355    0.306  0.532        1       1      0.0425            1   0.244        0.251                 False
 SEMA4A   0.426   0.0199   0.0058  0.507        1       1     -0.0721            1   0.498        0.597                 False

==============================================================================
stage IV vs I-III: IV vs I-III | tumour cells from 33 vs 7 donors | tier 1
==============================================================================
  0 genes at q_study < 0.05 (0 at q_free); readings: {}
  gene  log2fc    pct_x    pct_y  auroc  q_study  q_free  log2fc_rea  q_study_rea      did  q_study_did reading  on_shortlist
 PDIA4   0.679    0.297    0.156  0.567    0.642    0.72       0.383         0.88    0.296        0.562                 False
 SEC13   0.466    0.287    0.176   0.55    0.642    0.72       0.226        0.976     0.24        0.716                 False
IL17RA   0.363    0.201    0.132  0.532    0.642    0.72      0.0219            1    0.342       0.0615                 False
TNFSF4  0.0423  0.00187 0.000392  0.501    0.642    0.72      0.0661            1  -0.0237            1                 False
PMEPA1 -0.0482 0.000916  0.00158    0.5    0.642   0.888      -0.264            1    0.215        0.988                 False
  BST1 -0.0431  0.00107  0.00193    0.5    0.642    0.72     0.00199            1  -0.0451        0.301                 False
 OLFM1  -0.138 0.000211  0.00503  0.498    0.642    0.72    -0.00697            1   -0.132       0.0439                 False
 TLR10 -0.0136 0.000194 0.000483    0.5    0.642    0.72     -0.0117            1 -0.00189            1                 False
  VNN3 -0.0209 0.000274 0.000667    0.5    0.642    0.72     -0.0174            1 -0.00348            1                 False
 IGF1R  -0.501   0.0148   0.0324  0.491    0.642    0.72      -0.304            1   -0.196        0.988                 False
MCEMP1 -0.0512  0.00075  0.00226  0.499    0.642    0.72     0.00894            1  -0.0601        0.259                 False
 GFRA2 -0.0325 0.000768  0.00125    0.5    0.642   0.908    -0.00948            1   -0.023        0.931                 False

==============================================================================
stage class: advanced vs early | tumour cells from 38 vs 2 donors | tier 3
===
... [25130 more chars truncated]
```

```python  # cell 15
# ============================================================================
# §4c F4-F6  The antigen inside each stratum: disease, ECP, stage
# ============================================================================
FOREST_GENES = (PSHORT or SHORT)[:12]
FOREST_COL = ("#c0392b", "#2e86c1")


def forest(field, fname):
    _, x, y_ = FIELDS[field]
    pw = POWER.loc[field]
    w = WIT[WIT["field"].eq(field) & WIT["gene"].isin(FOREST_GENES)]
    if w["stratum"].nunique() < 2:
        print(f"{field}: one stratum has < {P.DESCRIBE_MIN} donors in an arm -- no forest")
        return
    piv = w.pivot_table(index="gene", columns="stratum", values="log2fc")
    order = piv.mean(1).sort_values().index.tolist()
    dq = DID[DID["field"].eq(field)].set_index("gene")["q_study"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axvline(0, lw=.5, c="k")
    yy = np.arange(len(order))
    for j, (s, col) in enumerate(zip((x, y_), FOREST_COL)):
        ws = w[w["stratum"].eq(s)].set_index("gene").reindex(order)
        ax.errorbar(ws["log2fc"], yy + (.17 if j == 0 else -.17),
                    xerr=[ws["log2fc"] - ws["log2fc_lo"], ws["log2fc_hi"] - ws["log2fc"]],
                    fmt="o", ms=3, lw=.8, color=col,
                    label=f"{s}: {int(ws['n_donors_a'].iloc[0])} mal / "
                          f"{int(ws['n_donors_b'].iloc[0])} rea donors")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{TF.it(g)} {TF.stars(dq.get(g, np.nan))}".strip() for g in order],
                       fontsize=6)
    ax.set_xlabel("pooled log2FC, malignant vs reactive, within stratum")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    if pw["tier_did"] == 1 and dq.notna().any():
        k = int((dq.reindex(order) < Q).sum())
        title = (f"{field}: tumour selectivity differs for {k} of {len(order)} antigens"
                 if k else f"{field}: no leading antigen changes its tumour selectivity")
    else:
        title = f"{field}: described only (tier {pw['tier_did']}, no within-study test)"
    ax.set_title(title, loc="left", fontsize=7.5)
    TF._clean(ax)
    fig.tight_layout()
    TF.save(fig, fname, caption_text=(
        "Donor-balanced pools; bars: 95% donor-bootstrap CI. Stars: DiD q, within-study "
        "permutation."))
    TF.save_source(w, fname + "_source")
    TF.logfig(fname, f"within-stratum pooled log2FC by {field}", len(order))
    print(f"{field}:")
    print(DID[DID["field"].eq(field) & DID["gene"].isin(order)]
          .set_index("gene").reindex(order[::-1])
          [["log2fc_x", "log2fc_y", "did", "q_study", "q_free"]]
          .to_string(float_format=lambda v: f"{v:.3g}"))


forest("disease", "F4_forest_disease")
forest("ECP", "F5_forest_ecp")
forest("stage IV vs I-III", "F6_forest_stage")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F4_forest_disease.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F4_forest_disease.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F4_forest_disease_source.csv
disease:
         log2fc_x  log2fc_y    did  q_study  q_free
gene                                               
KIR3DL2      3.21      2.94  0.273        1       1
TPO          1.94      2.84 -0.896    0.669       1
GPR15        2.11       1.9  0.207        1       1
FCRL3        2.08      1.91  0.172        1       1
CD70         1.95      1.88 0.0665        1       1
RAB25        1.56      2.02 -0.463        1       1
PTH2R        1.64      1.86 -0.229        1       1
TSPAN2       1.61      1.47  0.133        1       1
TIGIT        1.68      1.37  0.308        1       1
CPM          1.46      1.57 -0.112        1       1
STAM         1.28      1.67 -0.387    0.965       1
EPCAM        1.23      1.47 -0.239        1       1

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F5_forest_ecp.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F5_forest_ecp.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F5_forest_ecp_source.csv
ECP:
         log2fc_x  log2fc_y     did  q_study  q_free
gene                                                
KIR3DL2      2.93      3.28  -0.349    0.827   0.924
TPO          2.87      2.65   0.227        1   0.993
FCRL3        2.04      2.01  0.0304        1       1
CD70         1.86      2.14  -0.285     0.95   0.965
GPR15        2.08      1.84   0.238        1   0.993
RAB25        1.86      1.91 -0.0547        1       1
PTH2R         1.7         2  -0.299        1   0.993
STAM         1.68       1.6  0.0811        1       1
CPM          1.57      1.64 -0.0636        1       1
TIGIT        1.63      1.52   0.104        1       1
TSPAN2       1.34      1.73  -0.393    0.862   0.924
EPCAM        1.24      1.46  -0.213        1       1

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F6_forest_stage.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F6_forest_stage.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F6_forest_stage_source.csv
stage IV vs I-III:
         log2fc_x  log2fc_y    did  q_study  q_free
gene                                               
KIR3DL2      3.02      2.71  0.318    0.931       1
TPO          2.79      2.24  0.549    0.838       1
FCRL3        1.56      2.26 -0.698    0.719   0.987
CD70         1.89      1.83 0.0603        1       1
GPR15        1.64       1.9 -0.256    0.996       1
STAM         1.49      1.78  -0.29    0.889       1
RAB25        1.74      1.51   0.23        1       1
TSPAN2       1.66      1.39  0.269    0.985       1
TIGIT         1.3      1.54 -0.237    0.999       1
CPM          1.39      1.37  0.018        1       1
EPCAM        1.43      1.31  0.113        1       1
PTH2R        1.15     0.607  0.539    0.869   0.801
```

```python  # cell 16
# ============================================================================
# §4d F7  Tumour-cell log2FC between clinical strata, leading antigens
# ============================================================================
HG = (PSHORT or SHORT)[:15]
SHOWN = [f for f in FIELDS if POWER.loc[f, "tier_mal"] < 3]
_t = TUM[TUM["gene"].isin(HG) & TUM["field"].isin(SHOWN)]
H = _t.pivot_table(index="gene", columns="field", values="log2fc").reindex(index=HG,
                                                                          columns=SHOWN)
Qm = (_t.pivot_table(index="gene", columns="field", values="q_study", dropna=False)
      .reindex(index=HG, columns=SHOWN))
untested = Qm.isna().all()        # tier 2, or a split that is also a study split

# Column labels spell out the split and its cut-off, so no number on the axis needs a lookup.
SPLIT_LABEL = {
    "disease": "disease: MF vs SS",
    "stage IV vs I-III": "stage: IV vs I–III",
    "B stage": "blood stage: B2 vs B0–B1",
    "ECP": "ECP at sampling: yes vs no",
    "retinoid": "retinoid at sampling: yes vs no",
    "interferon": "interferon at sampling: yes vs no",
    "HDACi": "HDACi at sampling: yes vs no",
    "treatment-naive": "treatment-naive: yes vs no",
    "sex": "sex: F vs M",
    f"age (median {AGE_MED:.0f})": f"age: ≥{AGE_MED:.0f} vs <{AGE_MED:.0f} y",
    f"burden (median {BUR_MED:.2f})": f"malignant share of CD4: ≥{BUR_MED:.2f} vs <{BUR_MED:.2f}",
    f"blood tumour % (median {BTP_MED:.0f})": f"blood tumour %: ≥{BTP_MED:.0f} vs <{BTP_MED:.0f}",
    "cultured 2 days ex vivo": "cultured 2 days ex vivo: yes vs no",
}
xl = [f"{SPLIT_LABEL.get(f, f'{f}: {FIELDS[f][1]} vs {FIELDS[f][2]}')}"
      f"{' †' if untested[f] else ''}\n"
      f"n = {POWER.loc[f, 'n_mal_x']} vs {POWER.loc[f, 'n_mal_y']} donors" for f in SHOWN]

fig, ax = plt.subplots(figsize=(8, 5.5))
v = float(np.nanmax(np.abs(H.to_numpy()))) or 1.0
im = ax.imshow(H.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if Qm.iloc[i, j] < Q:
            ax.text(j, i, "*", ha="center", va="center", fontsize=10)
ax.set_xticks(range(len(SHOWN)))
ax.set_xticklabels(xl, rotation=50, ha="right", rotation_mode="anchor", fontsize=6.5)
ax.set_yticks(range(len(HG)))
ax.set_yticklabels([TF.it(g) for g in HG], fontsize=7.5)
ax.set_xticks(np.arange(len(SHOWN) + 1) - .5, minor=True)
ax.set_yticks(np.arange(len(HG) + 1) - .5, minor=True)
ax.grid(which="minor", color="white", lw=0.6)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
cb.set_label("log2 fold change, pooled tumour-cell expression\n"
             "first level ÷ second level of each split", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
cb.ax.text(0.5, 1.02, "higher in\nfirst level", transform=cb.ax.transAxes, ha="center",
           va="bottom", fontsize=6)
cb.ax.text(0.5, -0.02, "higher in\nsecond level", transform=cb.ax.transAxes, ha="center",
           va="top", fontsize=6)
ax.set_title(f"Tumour-cell log2FC between clinical strata, {len(HG)} leading antigens",
             loc="left", fontsize=9)
fig.tight_layout()
TF.save(fig, "F7_strata_tumour_heat", caption_text=(
    "Tile: log2 ratio of donor-balanced pooled mean expression in malignant CD4 cells, first "
    f"level over second. *: q < {Q}, within-study permutation ({N_PERM:,}). †: no within-study "
    "test (tier 2, or the split is a single study). n: donors with a malignant arm. Tier-3 "
    "fields omitted."))
TF.save_source(H.reset_index(), "F7_strata_tumour_heat_source")
TF.logfig("F7_strata_tumour_heat", "tumour-cell log2FC across clinical strata, leading antigens",
          int(Qm.notna().sum().sum()))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F7_strata_tumour_heat.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt46_F7_strata_tumour_heat.pdf

<Figure size 800x550 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_F7_strata_tumour_heat_source.csv
```

```python  # cell 17
# ============================================================================
# §4e  Sensitivity: disease and ECP without the cultured and shared-clone donors
# ============================================================================
# The geskin vehicle wells spent two days in culture, and buus PT05/06/18 share a dominant TRB
# clone (contamination or a label swap). Both stay in the cohort, flagged; here they leave it.
EXCL = PT.index[PT["is_cultured"] | PT["clone_shared"]]
print(f"excluded: {len(EXCL)} donor keys ({int(PT['is_cultured'].sum())} cultured, "
      f"{int(PT['clone_shared'].sum())} shared TRB clone)\n")
sens = []
for name in ("disease", "ECP"):
    lab, x, y_ = FIELDS[name]
    lab2 = lab.dropna().drop(EXCL, errors="ignore")
    t2 = P.between(mal, lab2, x, y_, block=STUDY, n_perm=N_PERM, seed=400).set_index("gene")
    d2 = P.did(mal, rea, lab2, x, y_, block=STUDY, n_perm=N_PERM, seed=401).set_index("gene")
    t1 = TUM[TUM["field"].eq(name)].set_index("gene")
    d1 = DID[DID["field"].eq(name)].set_index("gene")
    s_all = set(t1.index[t1["q_study"] < Q])
    s_ex = set(t2.index[t2["q_study"] < Q]) if "q_study" in t2 else set()
    rec = {"field": name, "n_mal_x/y": f"{int(t2['n_x'].iloc[0])}/{int(t2['n_y'].iloc[0])}",
           "rho_tumour_lfc": stats.spearmanr(t1["log2fc"], t2.loc[t1.index, "log2fc"]).statistic,
           "rho_did": stats.spearmanr(d1["did"], d2.loc[d1.index, "did"],
                                      nan_policy="omit").statistic,
           "tumour q<.05 all": len(s_all), "tumour q<.05 excl": len(s_ex),
           "in both": len(s_all & s_ex)}
    sens.append(rec)
    _g = [g for g in (PSHORT or SHORT)[:12]]
    print(f"{name}, leading antigens -- tumour-cell log2FC and DiD, all donors vs excluded:")
    print(pd.DataFrame({"tumour_all": t1.loc[_g, "log2fc"], "tumour_excl": t2.loc[_g, "log2fc"],
                        "did_all": d1.loc[_g, "did"], "did_excl": d2.loc[_g, "did"]})
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print()
sens = pd.DataFrame(sens).set_index("field")
print(sens.to_string(float_format=lambda v: f"{v:.3g}"))
sens.to_csv(ST.tab("sensitivity_excl"))
print("\nwrote", ST.tab("sensitivity_excl"))
```

```
excluded: 9 donor keys (6 cultured, 3 shared TRB clone)


disease, leading antigens -- tumour-cell log2FC and DiD, all donors vs excluded:
         tumour_all  tumour_excl  did_all  did_excl
gene                                               
KIR3DL2       -0.15      -0.0049    0.273     0.326
TPO          -0.998         -1.2   -0.896     -1.07
RAB25        -0.507       -0.728   -0.463    -0.667
GPR15        -0.111       -0.512    0.207    -0.157
CD70          -0.36       -0.372   0.0665   -0.0351
FCRL3         0.263        0.138    0.172    0.0504
PTH2R        -0.311       -0.669   -0.229    -0.593
STAM         -0.469       -0.321   -0.387    -0.191
CPM          -0.151       -0.183   -0.112    -0.165
TSPAN2        0.161       0.0988    0.133     0.158
TIGIT         0.214        0.393    0.308     0.299
EPCAM        -0.272       -0.155   -0.239    -0.145


ECP, leading antigens -- tumour-cell log2FC and DiD, all donors vs excluded:
         tumour_all  tumour_excl  did_all  did_excl
gene                                               
KIR3DL2      -0.216       -0.165   -0.349    -0.327
TPO           0.368        0.438    0.227     0.312
RAB25        -0.112      -0.0987  -0.0547   -0.0457
GPR15         0.239        0.187    0.238     0.245
CD70         -0.394         -0.3   -0.285    -0.218
FCRL3         0.514        0.495   0.0304    0.0206
PTH2R        -0.282       -0.342   -0.299    -0.365
STAM          0.253        0.274   0.0811     0.136
CPM         -0.0539      0.00931  -0.0636   0.00329
TSPAN2       -0.199       -0.112   -0.393    -0.306
TIGIT          0.29        0.348    0.104     0.114
EPCAM        -0.241       -0.194   -0.213    -0.165

        n_mal_x/y  rho_tumour_lfc  rho_did  tumour q<.05 all  tumour q<.05 excl  in both
field                                                                                   
disease      7/36           0.851    0.853                 0                  0        0
ECP         23/21           0.867    0.841                 0                  0        0

wrote /home/projects/nyosef/zvise/MF/tables/tgt46_sensitivity_excl.csv
```

## §5 — Disease, ECP, sex and age together

The §4 contrasts take one field at a time, and the fields are correlated: ECP is mostly given in
SS, and mostly in buus and ren.

The model is one OLS per gene on donor pseudobulk, with each donor's arm summarised as
log2(mean + 0.05):
- terms `is_MF + ECP + female + age_z`;
- HC3 standard errors, BH correction per term.

It is run on the malignant arm and, separately, on the reactive arm. Each term can then be read
as tumour-specific (malignant only), compartment-wide (both arms, same sign) or reactive-only.

Study stays out of the design for the reason 44 gives: with this many studies it would absorb the
treatment term. It is reported instead as its own one-way ANOVA on the malignant arm.

```python  # cell 19
# ============================================================================
# §5  Joint model on donor pseudobulk, malignant and reactive arms
# ============================================================================
X = pd.DataFrame(index=PT.index)
X["is_MF"] = PT["dx2"].map({"MF": 1.0, "SS": 0.0})
X["ecp"] = PT["tx_ecp"].where(PT["tx_known"]).astype(float)
X["is_F"] = PT["sex"].map({"F": 1.0, "M": 0.0})
X["age_z"] = (PT["age"] - PT["age"].mean()) / PT["age"].std()
X = X.dropna()
TERMS = list(X.columns)
MIN_DET_DONORS = 10       # a gene at zero in nearly every donor gives a degenerate HC3 fit


def fit_arm(pan):
    Y = pd.DataFrame(np.log2(pan.M + P.PSEUDO), index=pan.donors, columns=pan.genes)
    Xa = X.reindex(Y.index).dropna()
    Xd = sm.add_constant(Xa)
    det = pd.DataFrame(pan.M > 0, index=pan.donors, columns=pan.genes).loc[Xa.index].sum()
    rows = []
    for g, yv in Y.loc[Xa.index, det.index[det >= MIN_DET_DONORS]].items():
        f = sm.OLS(yv.to_numpy(), Xd).fit(cov_type="HC3")
        rows.append({"gene": g, **{f"b_{t}": f.params[t] for t in TERMS},
                     **{f"p_{t}": f.pvalues[t] for t in TERMS}})
    out = pd.DataFrame(rows)
    for t in TERMS:
        out[f"q_{t}"] = multipletests(out[f"p_{t}"].fillna(1.0), method="fdr_bh")[1]
    return out, Xa


reg_m, Xm = fit_arm(mal)
reg_r, Xr = fit_arm(rea)
print(f"malignant arm: {len(Xm)} donors with every covariate "
      f"(MF {int(Xm['is_MF'].sum())}, ECP {int(Xm['ecp'].sum())}, F {int(Xm['is_F'].sum())}), "
      f"{len(reg_m):,} genes detected in >= {MIN_DET_DONORS} of them; reactive arm: {len(Xr)} "
      f"(MF {int(Xr['is_MF'].sum())}, ECP {int(Xr['ecp'].sum())}, F {int(Xr['is_F'].sum())}), "
      f"{len(reg_r):,} genes")
print("ECP x disease among the malignant-arm donors:")
print(pd.crosstab(Xm["is_MF"].map({1.0: "MF", 0.0: "SS"}),
                  Xm["ecp"].map({1.0: "ECP", 0.0: "no ECP"})).to_string())

reg = reg_m.merge(reg_r, on="gene", how="outer", suffixes=("_mal", "_rea"))
for t in TERMS:
    sm_, sr = reg[f"q_{t}_mal"] < Q, reg[f"q_{t}_rea"] < Q
    same = np.sign(reg[f"b_{t}_mal"]) == np.sign(reg[f"b_{t}_rea"])
    reg[f"reading_{t}"] = np.select([sm_ & ~sr, sm_ & sr & same, sm_ & sr & ~same, ~sm_ & sr],
                                    ["tumour-specific", "compartment-wide", "opposite",
                                     "reactive-only"], "")
reg["on_shortlist"] = reg["gene"].isin(set(PSHORT) | set(SHORT))
print(f"\ngenes at q < {Q} per term, by reading:")
print(pd.DataFrame({t: reg[f"reading_{t}"].value_counts() for t in TERMS})
      .drop(index="", errors="ignore").fillna(0).astype(int).to_string())

for t in TERMS:
    s = reg[reg[f"reading_{t}"].eq("tumour-specific")]
    s = s.reindex(s[f"b_{t}_mal"].abs().sort_values(ascending=False).index).head(10)
    print(f"\n--- {t}: tumour-specific, strongest 10")
    print(s[["gene", f"b_{t}_mal", f"q_{t}_mal", f"b_{t}_rea", f"q_{t}_rea", "on_shortlist"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}") if len(s) else "  none")

print("\nthe leading antigens, malignant-arm coefficients (log2 units per level / per SD of age):")
_g = [g for g in (PSHORT or SHORT)[:15] if g in set(reg["gene"])]
print(reg.set_index("gene").loc[_g, [f"b_{t}_mal" for t in TERMS] + [f"q_{t}_mal" for t in TERMS]]
      .to_string(float_format=lambda v: f"{v:.3g}"))

Ym = pd.DataFrame(np.log2(mal.M + P.PSEUDO), index=mal.donors, columns=mal.genes)
Ym = Ym[reg_m["gene"]]
st = STUDY.reindex(Ym.index)
keep_st = st.value_counts()[lambda v: v >= 3].index
anv = pd.DataFrame({"gene": Ym.columns, "p_study": [
    stats.f_oneway(*[Ym.loc[st.eq(s), g] for s in keep_st]).pvalue for g in Ym.columns]})
anv["q_study"] = multipletests(anv["p_study"].fillna(1.0), method="fdr_bh")[1]
print(f"\nstudy, one-way ANOVA on the malignant arm over {len(keep_st)} studies with >= 3 donors: "
      f"{int((anv['q_study'] < Q).sum()):,} of {len(anv):,} genes at q < {Q} -- the size of the "
      "term the model does not carry.")
reg.merge(anv, on="gene", how="left").to_csv(ST.tab("joint_model"), index=False)
print("wrote", ST.tab("joint_model"))
```

```
malignant arm: 43 donors with every covariate (MF 7, ECP 22, F 17), 808 genes detected in >= 10 of them; reactive arm: 42 (MF 10, ECP 22, F 17), 692 genes
ECP x disease among the malignant-arm donors:
ecp    ECP  no ECP
is_MF             
MF       4       3
SS      18      18

genes at q < 0.05 per term, by reading:
                 is_MF  ecp  is_F  age_z
tumour-specific      1    0     0      0

--- is_MF: tumour-specific, strongest 10
 gene  b_is_MF_mal  q_is_MF_mal  b_is_MF_rea  q_is_MF_rea  on_shortlist
CD274       -0.249       0.0207      -0.0519            1         False

--- ecp: tumour-specific, strongest 10
  none

--- is_F: tumour-specific, strongest 10
  none

--- age_z: tumour-specific, strongest 10
  none

the leading antigens, malignant-arm coefficients (log2 units per level / per SD of age):
         b_is_MF_mal  b_ecp_mal  b_is_F_mal  b_age_z_mal  q_is_MF_mal  q_ecp_mal  q_is_F_mal  q_age_z_mal
gene                                                                                                     
KIR3DL2      -0.0453        0.5      0.0784       -0.611        0.995      0.976       0.995        0.964
TPO           -0.949      0.341       0.531        0.395        0.972      0.976       0.995        0.964
RAB25         -0.677     -0.176       -0.45        0.114        0.995      0.976       0.995        0.978
GPR15          -1.13      0.609      -0.682       -0.101        0.974      0.976       0.995        0.977
CD70          -0.539     -0.204       0.371        -0.21        0.995      0.976       0.995        0.964
FCRL3         -0.174      0.544       0.125        0.254        0.995      0.976       0.995        0.964
PTH2R          -0.41     -0.281    -0.00363       0.0776        0.995      0.976       0.998        0.977
STAM          -0.327       0.61       0.482      -0.0646        0.995      0.976       0.995         0.98
CPM          -0.0224    -0.0303       0.157         0.25        0.995      0.992       0.995        0.964
TSPAN2         0.326     -0.209      0.0869       -0.311        0.995      0.976       0.995        0.964
TIGIT          0.219      0.388      -0.227       0.0802        0.995      0.976       0.995        0.978
EPCAM         -0.061     -0.436     -0.0879        0.324        0.995      0.976       0.995        0.964
PI16          -0.045     -0.636       0.528       -0.086        0.995      0.976       0.995        0.978
FGFR1           0.02      0.446      0.0354       0.0149        0.995      0.976       0.995        0.982
NRROS        -0.0856      -0.11       0.251       -0.286        0.995      0.976       0.995        0.964


study, one-way ANOVA on the malignant arm over 8 studies with >= 3 donors: 212 of 808 genes at q < 0.05 -- the size of the term the model does not carry.
wrote /home/projects/nyosef/zvise/MF/tables/tgt46_joint_model.csv
```

```python  # cell 20
# ============================================================================
# §6  Dossier
# ============================================================================
def SEC(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def kv(k, v): print(f"  {k:40s} {v}")


N_ONE_ARM = int(DON["pattern"].isin(["malignant only", "reactive only"]).sum())
N_HC = int(DON["pattern"].eq("healthy").sum())
N_PAIRED = int(DON["pattern"].eq("paired").sum())

SEC("1 - COHORT")
kv("independent donor keys", len(DON))
print(DON["pattern"].value_counts().to_string())
kv("patients in a pool", f"{len(PT)} ({PT['person'].nunique()} people)")

SEC("2 - POOLS")
kv("headline weighting", WEIGHT)
print(summ[["n_donors", "n_cells", "kish_n_eff_cells", "top_donor_share_cells", "top_study",
            "top_study_share_cells"]].to_string(float_format=lambda v: f"{v:.3g}"))

SEC("3 - HEADLINE: POOLED MALIGNANT VS REACTIVE")
kv("genes", len(pool))
kv("van Elteren q < 0.05", int((pool["q_blocked"] < Q).sum()))
kv("rho vs paired log2FC", f"{RHO:.3f}")
kv("pooled shortlist", len(PSHORT))
kv("  of which on 43's shortlist", len(set(PSHORT) & set(SHORT)))
kv("43's shortlist missing from pooled", sorted(set(SHORT) - set(PSHORT)))
for g in [g for g in ("CCR4", "KIR3DL2", "IL2RA", "CD7", "DPP4") if g in set(pool["gene"])]:
    r_ = pool.set_index("gene").loc[g]
    kv(f"  {g}", f"log2FC {r_['log2fc']:+.2f} [{r_['log2fc_lo']:+.2f}, {r_['log2fc_hi']:+.2f}]"
                 f"  AUROC {r_['auroc']:.3f}  q {r_['q_blocked']:.2g}")

SEC("4 - STRATA")
print(SUMMARY.to_string(float_format=lambda v: f"{v:.3g}"))

SEC("5 - JOINT MODEL")
kv("donors, malignant / reactive arm", f"{len(Xm)} / {len(Xr)}")
print(pd.DataFrame({t: reg[f"reading_{t}"].value_counts() for t in TERMS})
      .drop(index="", errors="ignore").fillna(0).astype(int).to_string())

SEC("6 - WHAT THIS IS NOT")
for line in [
    "A cell-level test. Every p-value here has the donor as its unit; cells only weight the pool.",
    f"More patients. Pooling adds {N_ONE_ARM} one-arm patients and {N_HC} healthy donors to the "
    f"{N_PAIRED} paired; the metadata gaps of 44 are unchanged.",
    f"A stage analysis on tumour cells: {POWER.loc['stage class', 'n_mal_y']} early-stage donors "
    "have a malignant arm.",
    "A treatment-effect study. Treatment is concentrated in particular studies; q_study is the "
    "honest p, and for several splits it has few arrangements to work with.",
    "A full-gene or protein result: transcript, in a 10,000-gene HVG subset.",
]:
    print("  - " + line)

SEC("7 - OUTPUTS")
TF.write_index("figure_index")
for p in sorted(ST.TAB_DIR.glob(f"{PREFIX}*")) + sorted(ST.FIG_DIR.glob(f"{PREFIX}*")):
    print("  ", p.relative_to(NB_DIR))
```

```
==============================================================================
1 - COHORT
==============================================================================
  independent donor keys                   75
pattern
paired            52
healthy           12
malignant only     6
reactive only      5
  patients in a pool                       63 (63 people)

==============================================================================
2 - POOLS
==============================================================================
  headline weighting                       donor
               n_donors  n_cells  kish_n_eff_cells  top_donor_share_cells  top_study  top_study_share_cells
arm                                                                                                        
CD4_malignant        58   295656              24.2                  0.103    ren2023                  0.273
CD4_reactive         57   111125              7.75                  0.297  harro2023                   0.54
CD4_HC               12    40400              6.02                  0.245    ren2023                  0.672

==============================================================================
3 - HEADLINE: POOLED MALIGNANT VS REACTIVE
==============================================================================
  genes                                    1098
  van Elteren q < 0.05                     400
  rho vs paired log2FC                     0.834
  pooled shortlist                         78
    of which on 43's shortlist             28
  43's shortlist missing from pooled       ['TNFRSF1B', 'TNFRSF25']
    CCR4                                   log2FC +1.03 [+0.71, +1.36]  AUROC 0.567  q 0.00081
    KIR3DL2                                log2FC +3.04 [+2.78, +3.30]  AUROC 0.688  q 1e-09
    IL2RA                                  log2FC -0.21 [-0.58, +0.12]  AUROC 0.493  q 0.14
    CD7                                    log2FC -1.55 [-1.94, -1.18]  AUROC 0.276  q 6e-11
    DPP4                                   log2FC -1.85 [-2.02, -1.70]  AUROC 0.433  q 2.3e-13

==============================================================================
4 - STRATA
==============================================================================
                                       x vs y  mal n  rea n tier mal/rea/did  log10 cfg study  tumour q_study  tumour q_free  reactive q_study  reactive q_free  DiD q_study  DiD q_free
field                                                                                                                                                                                   
disease                              MF vs SS   8/43  11/38            1/1/1             5.43               0              0                 0                0            0           0
stage IV vs I-III                 IV vs I-III   33/7  30/11            1/1/1             4.38               0              0                 0                0            1           9
stage class                 advanced vs early   38/2   36/5            3/1/3              NaN             NaN            NaN                 0                0          NaN         NaN
B stage                           B2 vs B0-B1  15/12  13/15            1/1/1              5.1               0              0                 0                0            0           0
ECP                                 yes vs no  27/23  26/22            1/1/1             6.87               0              0                 0                2            0           2
retinoid                            yes vs no  16/34  14/34            1/1/1              5.2               0              2                 0                0            0           0
interferon                          yes vs no   8/42   7/41            1/1/1             6.07               0              0                 0                0            0           0
HDACi                               yes vs no   7/43   6/42         
... [3942 more chars truncated]
```

```python  # cell 21
# ============================================================================
# §7  The complete numerical result, as one block to hand to another model
# ============================================================================
_ctx = f"""
Compartment: MF/CTCL **blood**, pooled rather than paired. Notebook 43 compares malignant and
reactive CD4 cells WITHIN each donor; this one pools all malignant CD4 cells against all reactive
CD4 cells, which lets donors with only one arm contribute and makes clinical strata comparable as
cell populations.

Cohort: {len(DON)} independent donor keys ({N_PAIRED} with both arms, {N_ONE_ARM} with one arm,
{N_HC} healthy). The geskin2026 ex vivo drug wells and duplicate persons were removed in 43
section 0b. Malignancy call `{man['malignancy_call']}`.

Weighting: donor-balanced is the headline (every donor contributes equal weight to its arm);
cell-weighted is printed beside it as `*_cells`. This matters here -- three harro2023 early-MF
donors with no malignant arm hold half of all reactive cells under cell weighting.

Effect sizes: `log2fc` (log2 ratio of pooled mean log1p CP10K, pseudocount 0.05), `d_det`
(difference in the fraction of cells detected), `cohen_d`, and `auroc` (probability a random
malignant cell exceeds a random comparator cell; 0.5 = no separation, capped by detection for
sparse genes). CIs from a {N_BOOT}-draw donor bootstrap. The headline p is van Elteren
(`q_blocked`): donor-level means ranked within study. No p-value is computed over cells.

Stratified contrasts, per field: (a) malignant vs reactive within each stratum, (b) malignant
cells of one stratum vs the other, (c) the same for reactive cells, (d) difference-in-differences.
p-values permute donor labels within study ({N_PERM:,} permutations, `q_study`), with a free
permutation beside them (`q_free`). Fields are tier 1 (tested), 2 (described) or 3 (counted).

Everything is transcript, in a 10,000-gene HVG subset. Pooling adds cells, not patients: the
metadata gaps of notebook 44 are unchanged.
"""
_sec = [
    {"name": "§0b Cohort by arm pattern",
     "what": "how many independent donors have both CD4 arms, one, or are healthy.",
     "how": "an arm counts when it holds >= 20 cells.",
     "table": pd.crosstab(DON["pattern"], DON["dx"], margins=True),
     "file": "tables/tgt46_cohort.csv"},
    {"name": "§1 Pool composition",
     "what": "who each pool is actually made of, under each weighting.",
     "how": "Kish effective donors (sum(w)^2/sum(w^2)).",
     "table": summ, "file": "tables/tgt46_pool_composition.csv"},
    {"name": "§2a Pooled shortlist",
     "what": "the genes passing every pooled gate: clean in 43, log2FC > 0.5 with the bootstrap "
             "CI above zero, van Elteren q < 0.05, not a loss marker or lineage control.",
     "how": "donor-balanced pooled contrast, malignant vs reactive CD4.",
     "table": pool[gate].sort_values("log2fc", ascending=False)[
         ["gene", "cd_name", "log2fc", "log2fc_lo", "log2fc_hi", "pct_a", "pct_b", "d_det",
          "auroc", "auroc_lo", "cohen_d", "log2fc_cells", "log2fc_paired", "q_blocked",
          "log2fc_vs_hc", "auroc_vs_hc", "on_43_shortlist"]],
     "file": "tables/tgt46_pooled_all_genes.csv"},
    {"name": "§2a Controls",
     "what": "the antigens with a known answer, pooled.",
     "how": "CCR4, KIR3DL2 and IL2RA should be up; CD7 and DPP4 down with AUROC below 0.5.",
     "table": pool.set_index("gene").loc[_ctl, ["log2fc", "log2fc_lo", "log2fc_hi", "d_det",
                                                "auroc", "auroc_lo", "auroc_hi",
                                                "log2fc_paired", "q_blocked"]]},
    {"name": "§2b Pooled vs paired agreement",
     "what": "whether pooling reproduces 43's paired ranking.",
     "how": "Spearman over all genes.",
     "table": {"spearman_pooled_vs_paired": round(float(RHO), 3),
               "spearman_donor_balanced_vs_cell_weighted": round(float(_rho_c), 3)}},
    {"name": "§2c Leave-one-study-out",
     "what": "whether any single study carries the pooled effect.",
     "how": "the pooled log2FC recomputed with each study dropped in turn.",
     "table": lo_sum, "file": "tables/tgt46_loso.csv"},
    {"name": "§3 Power table",
     "what": "donors per arm on each side of every split, and the resulting tier.",
     "how": "tier 1 = tested, 2 = described, 3 = counted only.",
     "table": POWER[[c for c in POWER.columns if not c.startswith("studies")]],
     "file": "tables/tgt46_power_table.csv"},
    {"name": "§4a Stratified summary",
     "what": "how many genes differ between the levels of each field, for tumour cells, "
             "reactive cells and the difference-in-differences.",
     "how": "within-study donor permutation (q_study) and free permutation (q_free), BH.",
     "table": SUMMARY, "file": "tables/tgt46_strat_summary.csv"},
    {"name": "§4b Leading antigens across every stratum",
     "what": "each leading antigen's tumour-cell log2FC between the levels of every field.",
     "how": "pooled, donor-balanced, first level over second.",
     "table": TUM[TUM["gene"].isin((PSHORT or SHORT)[:15])].pivot_table(
         index="gene", columns="field", values="log2fc"),
     "file": "tables/tgt46_strat_joined.csv"},
    {"name": "§4e Sensitivity",
     "what": "whether the disease and ECP contrasts survive dropping the cultured and "
             "shared-clone donors.",
     "how": "the same contrast refitted; rho is against the full-cohort per-gene values.",
     "table": sens, "file": "tables/tgt46_sensitivity_excl.csv"},
    {"name": "§5 Joint model readings",
     "what": "for each donor-level term, how many genes are tumour-specific, compartment-wide, "
             "opposite or reactive-only.",
     "how": "OLS per gene on donor pseudobulk log2(mean + 0.05), HC3 errors, BH per term, run "
            "separately on the malignant and reactive arms.",
     "table": pd.DataFrame({t: reg[f"reading_{t}"].value_counts() for t in TERMS})
                .drop(index="", errors="ignore").fillna(0).astype(int),
     "file": "tables/tgt46_joint_model.csv"},
]
TF.llm_export("46", "Blood, pooled and stratified", _ctx, _sec)
```

```
======================== COPY FROM HERE ========================

# 46 — Blood, pooled and stratified

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**, pooled rather than paired. Notebook 43 compares malignant and
reactive CD4 cells WITHIN each donor; this one pools all malignant CD4 cells against all reactive
CD4 cells, which lets donors with only one arm contribute and makes clinical strata comparable as
cell populations.

Cohort: 75 independent donor keys (52 with both arms, 11 with one arm,
12 healthy). The geskin2026 ex vivo drug wells and duplicate persons were removed in 43
section 0b. Malignancy call `mal_cluster`.

Weighting: donor-balanced is the headline (every donor contributes equal weight to its arm);
cell-weighted is printed beside it as `*_cells`. This matters here -- three harro2023 early-MF
donors with no malignant arm hold half of all reactive cells under cell weighting.

Effect sizes: `log2fc` (log2 ratio of pooled mean log1p CP10K, pseudocount 0.05), `d_det`
(difference in the fraction of cells detected), `cohen_d`, and `auroc` (probability a random
malignant cell exceeds a random comparator cell; 0.5 = no separation, capped by detection for
sparse genes). CIs from a 1000-draw donor bootstrap. The headline p is van Elteren
(`q_blocked`): donor-level means ranked within study. No p-value is computed over cells.

Stratified contrasts, per field: (a) malignant vs reactive within each stratum, (b) malignant
cells of one stratum vs the other, (c) the same for reactive cells, (d) difference-in-differences.
p-values permute donor labels within study (50,000 permutations, `q_study`), with a free
permutation beside them (`q_free`). Fields are tier 1 (tested), 2 (described) or 3 (counted).

Everything is transcript, in a 10,000-gene HVG subset. Pooling adds cells, not patients: the
metadata gaps of notebook 44 are unchanged.

## §0b Cohort by arm pattern

**What:** how many independent donors have both CD4 arms, one, or are healthy.

**How:** an arm counts when it holds >= 20 cells.

**Full table on disk:** `tables/tgt46_cohort.csv`

```csv
pattern,HC,MF,MF/SS,MF_or_SS_unresolved,SS,All
healthy,12,0,0,0,0,12
malignant only,0,0,0,0,6,6
paired,0,8,2,5,37,52
reactive only,0,3,0,1,1,5
All,12,11,2,6,44,75
```

## §1 Pool composition

**What:** who each pool is actually made of, under each weighting.

**How:** Kish effective donors (sum(w)^2/sum(w^2)).

**Full table on disk:** `tables/tgt46_pool_composition.csv`

```csv
arm,n_donors,n_cells,kish_n_eff_cells,kish_n_eff_donor,top_donor,top_donor_share_cells,top_study_share_cells,top_study
CD4_malignant,58,295656,24.2,58,B2__CTCL2,0.103,0.273,ren2023
CD4_reactive,57,111125,7.75,57,B7__MF1,0.297,0.54,harro2023
CD4_HC,12,40400,6.02,12,B6__N2,0.245,0.672,ren2023
```

## §2a Pooled shortlist

**What:** the genes passing every pooled gate: clean in 43, log2FC > 0.5 with the bootstrap CI above zero, van Elteren q < 0.05, not a loss marker or lineage control.

**How:** donor-balanced pooled contrast, malignant vs reactive CD4.

**Full table on disk:** `tables/tgt46_pooled_all_genes.csv`

```csv
index,gene,cd_name,log2fc,log2fc_lo,log2fc_hi,pct_a,pct_b,d_det,auroc,auroc_lo,cohen_d,log2fc_cells,log2fc_paired,q_blocked,log2fc_vs_hc,auroc_vs_hc,on_43_shortlist
570,KIR3DL2,CD158k,3.04,2.78,3.3,0.396,0.0214,0.374,0.688,0.653,0.977,3.37,2.79,1.04e-09,3.46,0.695,True
1048,TPO,,2.76,2.38,3.1,0.236,0.00888,0.227,0.614,0.583,0.702,2.96,1.85,1.81e-12,2.98,0.617,True
827,RAB25,,1.98,1.53,2.36,0.142,0.00782
... [14951 more chars truncated]
```
