# 40_targets/skin/48_skin_pooled_stratification.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/skin/48_skin_pooled_stratification.ipynb` and is never run._

# 48 — Skin, pooled: malignant vs reactive CD4 across all cells, and by who the patient is

42 asks the malignant-vs-reactive question **inside each donor**. That is the right test for the
screen and the wrong shape for 47's question: pairing spends the cohort on the donors that carry
both arms, and every clinical stratum then has to be cut out of those. This notebook **pools**
instead. All malignant CD4 cells are compared with all reactive CD4 cells, and then the same
comparison is made inside, and between, clinical strata: disease, lesion, stage, B stage,
treatment at sampling, sex, age and burden.

Pooling gains two things, and there is one thing it cannot do.

- **Donors with one arm count.** The donors whose biopsy held a malignant arm but too few reactive
  cells — or the reverse — join the paired set, and the healthy-skin donors form a third pool.
- **Strata become comparisons of cell populations.** "Malignant cells of tumour lesions vs
  malignant cells of patches" is a well-defined contrast whether or not those patients have a
  reactive arm, which is what makes the lesion axis testable at full strength here.
- It does **not** create metadata. Treatment is still mostly unknown and still confounded with
  study. Pooling adds cells, not patients, and cells are not the unit of inference.

Two design choices, set in §0 and stated on every table:

| choice | headline | beside it |
|---|---|---|
| **weight** | donor-balanced: every donor contributes the same total weight to its arm | cell-weighted: every cell once |
| **inference** | donor cluster bootstrap for CIs; permutation of donor labels **within study** for between-stratum p | free permutation; van Elteren (study-blocked, donor-level) for the headline |

Why donor-balanced: in the cell pool, a handful of deeply sequenced li2024 biopsies hold a large
share of all malignant cells. "All malignant vs all reactive" would then mostly be a statement
about those samples. §1 shows this before anything is tested. No p-value is computed over cells
(`degradome_data.CAVEATS["cell_level_pvalues"]`).

| § | what it does |
|---|---|
| §0b | the cohort, and what pooling adds to the paired set |
| §1 | who each pool is made of, under each weighting |
| **§2** | **the headline pooled contrast**: log2FC, detection, Cohen's d and AUROC, against reactive and healthy CD4 |
| §3 | the power table for the pooled cohort, with tiers |
| **§4** | **stratified**: within each stratum, tumour cells across strata, reactive cells across strata, and the difference-in-differences |
| §5 | a joint model on donor pseudobulk, malignant and reactive arms |
| §6 | dossier, and §7 the numerical hand-off block |

**Inputs.**
- `jobs/run_target_screen_qc.sh --tissue skin` writes the long table.
- `jobs/run_target_pool_hist.sh --tissue skin` writes the per-group histograms behind the AUROC.
  It refuses to write unless it reproduces the long table's cell counts and detection rates.
- 42 supplies the shortlist and the artifact evidence.

> Everything here is transcript. Skin carries no CITE-seq, so nothing in this notebook has been
> seen as protein.

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

CM.use_tissue("skin")
MAL.use_tissue("skin")
TQ.use_tissue("skin")
assert MAL.CLUSTER_PQ.name.startswith("skin_"), MAL.CLUSTER_PQ

TISSUE = "skin"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt48_"
MALIG, REACT, HC_CD4 = "CD4_malignant", "CD4_reactive", "CD4_HC"
WEIGHT = "donor"          # the headline pool; "cells" is printed beside it
# 50,000 permutations, not 2,000: BH over ~2,000 genes puts a single hit near p < 2.5e-5, and
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
    "run jobs/run_target_screen_qc.sh --tissue skin first")
# The histograms are valid only for the long table they were checked against: a rebuilt long
# table (another call, another cohort rule) would make them describe different cells.
assert hman["malignancy_call"] == man["malignancy_call"], (hman["malignancy_call"],
                                                           man["malignancy_call"])
assert hman["written"] >= man["written"], (
    f"histograms ({hman['written']}) are older than the long table ({man['written']}) -- "
    "rerun jobs/run_target_pool_hist.sh --tissue skin")

print("NB_DIR =", NB_DIR)
print(f"long: {man['written']} | call {man['malignancy_call']} | "
      f"{man['n_genes_in_object']:,} genes in object")
print(f"hist: {hman['written']} | {hman['n_bins']} bins of {hman['bin_width']} log1p CP10K, "
      f"bin 0 = exact zero | max |mean_hist - mean_all| = {hman['max_abs_mean_diff']:.4f}")
print(f"headline weight: {WEIGHT} | {N_BOOT} donor-bootstrap draws | {N_PERM:,} permutations "
      f"(smallest attainable p {1 / (N_PERM + 1):.1e})")
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
long: 2026-10-26 13:59 | call mal_cluster | 42,347 genes in object
hist: 2026-10-26 14:09 | 201 bins of 0.05 log1p CP10K, bin 0 = exact zero | max |mean_hist - mean_all| = 0.0080
headline weight: donor | 1000 donor-bootstrap draws | 50,000 permutations (smallest attainable p 2.0e-05)
```

## §0b — The cohort, and what pooling adds

The cohort rule is the same as 42's (`target_cohort.skin_donor_table`): non-CD4 clones,
off-spectrum diagnoses and repeat biopsies of one patient are dropped, and diagnosis comes from
`clin_dx`. The difference is who enters: every independent donor with **any** CD4 arm of ≥ 20
cells.

`burden` is the malignant share of a donor's CD4 compartment. A donor with only one arm has the
other arm below 20 cells, and that arm is counted as 0. Malignant-only donors therefore sit at 1
and reactive-only donors at 0. That is a bound, not a measurement, and it is the only place a
missing arm is filled in — which matters more in skin than in blood, because a patch biopsy with
a small clone and a healthy-skin sample are both legitimately near 0.

```python  # cell 3
# ============================================================================
# §0b  Cohort: every independent donor with any CD4 arm
# ============================================================================
don = TC.skin_donor_table()
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
print(f"\npaired: {int(DON['pattern'].eq('paired').sum())} -- the whole of 42/47. Pooling adds "
      f"{int(DON['pattern'].isin(['malignant only', 'reactive only']).sum())} patients with one "
      f"arm and {int(DON['pattern'].eq('healthy').sum())} healthy donors.")
print("\nthe donors only pooling can use:")
COHORT_COLS = ["study", "dx", "lesion", "clin_stage", "clin_stage_class",
               "clin_current_tx_classes", "n_mal", "n_rea", "pattern"]
print(DON[DON["pattern"].isin(["malignant only", "reactive only"])][COHORT_COLS]
      .sort_values(["pattern", "study"]).to_string())
DON.drop(columns=[c for c in DON.columns if c.startswith(("prior_", "ever_"))]) \
   .to_csv(ST.tab("cohort"))
print("\nwrote", ST.tab("cohort"))
```

```
2,355 genes | malignant 78 donors / 187,748 cells, reactive 82 / 147,233, healthy CD4 21 / 4,724

independent donor keys by arm pattern:
dx              HC  MF  MF/SS  SS  All
pattern                               
healthy         21   0      0   0   21
malignant only   0   2      0   1    3
no CD4 arm       1   1      0   1    3
paired           0  67      2   6   75
reactive only    0   5      0   2    7
All             22  75      2  10  109

paired: 75 -- the whole of 42/47. Pooling adds 10 patients with one arm and 21 healthy donors.

the donors only pooling can use:
                            study  dx            lesion clin_stage clin_stage_class clin_current_tx_classes  n_mal   n_rea         pattern
donor                                                                                                                                     
D3__P87               brunner2024  SS      erythroderma       IVA1         advanced                     NaN  351.0     NaN  malignant only
Li2024_atlas__CTCL11       li2024  MF  skin_unspecified         IB            early                     NaN   90.0     NaN  malignant only
Li2024_atlas__CTCL15       li2024  MF  skin_unspecified         IA            early                     NaN  103.0     NaN  malignant only
D3__P94               brunner2024  MF      erythroderma       IVA1         advanced                     NaN    NaN  2701.0   reactive only
B5__PT22                 buus2025  SS      erythroderma        NaN              NaN                retinoid    NaN  7637.0   reactive only
H__SS3                herrera2021  SS            plaque       IVA1         advanced                     NaN    NaN    52.0   reactive only
D15__MF40               il4ra2026  MF      erythroderma       IIIB         advanced                     NaN    NaN   576.0   reactive only
Li2024_atlas__MF15         li2024  MF            plaque         IB            early                     NaN    NaN  1147.0   reactive only
Li2024_atlas__MF18         li2024  MF             tumor       IVA2         advanced                     NaN    NaN   600.0   reactive only
Li2024_atlas__MF26         li2024  MF            plaque        IIA            early                     NaN    NaN  1999.0   reactive only

wrote /home/projects/nyosef/zvise/MF/tables/tgt48_cohort.csv
```

## §1 — Who the pool is made of

A pool is a weighted sum over donors, and the weights decide whose cells the comparison is about.
Under cell weighting a donor counts in proportion to the number of cells they contributed. In
this atlas that number is set by biopsy size, dissociation yield and sequencing depth, not by
biology — a 9,000-cell rindler patch and a 130-cell one are the same patient's disease.

The table gives the effective number of donors for each arm under each weighting (Kish,
(Σw)²/Σw²); the figure gives the study mix. Donor balance does not fix the study mix — li2024
contributes 35 patients and still holds 35 shares — but it stops a single deep sample from
standing in for an arm.

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

_r = comp[MALIG]
_top = _r["share_cells"].idxmax()
studies = sorted(set().union(*[set(t.index) for t in comp.values()]))
scol = dict(zip(studies, plt.get_cmap("tab20").colors))
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
ax.set_title(f"Cell-weighted malignant pool is {_r.loc[_top, 'share_cells']:.0%} {_top}; "
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
n_donors  n_cells  kish_n_eff_cells  kish_n_eff_donor            top_donor  top_donor_share_cells  top_study_share_cells        top_study
arm                                                                                                                                                     
CD4_malignant        78   187748              33.4                78             B5__PT09                 0.0555                  0.476           li2024
CD4_reactive         82   147233              11.4                82  Li2024_atlas__CTCL7                  0.261                  0.622           li2024
CD4_HC               21     4724              12.4                21     D13__SC374_HTO10                  0.166                  0.256  chennareddy2025

CD4_malignant: share of the pool by study
                  n_cells  share_cells  share_donor  n_donors
study                                                        
li2024          89425.000        0.476        0.410        32
brentuximab2026 29046.000        0.155        0.077         6
buus2025        16798.000        0.089        0.026         2
rindler2021_mc  13169.000        0.070        0.051         4
chennareddy2025 12471.000        0.066        0.064         5
il4ra2026       11009.000        0.059        0.103         8
gaydosik2022     4992.000        0.027        0.064         5
alkon2024        4584.000        0.024        0.064         5
brunner2024      3848.000        0.020        0.051         4
gaydosik2019     1774.000        0.009        0.038         3
jonak2021         348.000        0.002        0.013         1
herrera2021       284.000        0.002        0.038         3

CD4_reactive: share of the pool by study
                  n_cells  share_cells  share_donor  n_donors
study                                                        
li2024          91575.000        0.622        0.402        33
buus2025        14157.000        0.096        0.037         3
il4ra2026        8024.000        0.054        0.110         9
rindler2021_mc   7945.000        0.054        0.049         4
chennareddy2025  7248.000        0.049        0.061         5
brentuximab2026  6000.000        0.041        0.073         6
brunner2024      4011.000        0.027        0.049         4
alkon2024        2801.000        0.019        0.061         5
gaydosik2022     1954.000        0.013        0.061         5
jonak2021        1920.000        0.013        0.012         1
gaydosik2019     1400.000        0.010        0.037         3
herrera2021       198.000        0.001        0.049         4

CD4_HC: share of the pool by study
                 n_cells  share_cells  share_donor  n_donors
study                                                       
chennareddy2025 1207.000        0.256        0.190         4
gaydosik2022    1201.000        0.254        0.190         4
il4ra2026       1122.000        0.238        0.190         4
gaydosik2023     953.000        0.202        0.238         5
gaydosik2019     241.000        0.051        0.190         4

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F1_pool_composition.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F1_pool_composition.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F1_pool_composition_source.csv

wrote /home/projects/nyosef/zvise/MF/tables/tgt48_pool_composition.csv
```

## §2 — Headline: pooled malignant vs reactive CD4

Every non-healthy donor that has an arm enters that arm's pool. Four effect sizes are reported,
because they answer different questions:

- `log2fc` — the log2 ratio of pooled mean log1p CP10K, on 42's scale (pseudocount 0.05).
- `d_det` — the difference in the fraction of cells detected. This is the quantity a CAR is dosed
  against.
- `cohen_d` — the mean difference over the pooled SD. It stays small for a gene detected in a
  minority of cells, however large the ratio.
- `auroc` — the chance that a random malignant cell expresses more than a random reactive cell.
  0.5 is no separation. For a sparse gene it is bounded by detection: a gene detected in 12% of
  tumour cells cannot go much above 0.56.

The CIs come from a donor bootstrap: donors are resampled and bring both arms, so paired donors
stay paired. The p-value is van Elteren: donor-level means ranked within study, the test
`target_screen.blocked_selectivity` already provides for donors with one arm. Beside the headline
sit the cell-weighted pool, 42's paired effect, and the same pooled contrast against **healthy**
CD4 — which in skin means CD4 cells from healthy skin, not from blood.

The pooled shortlist gates mirror 42's:
- `artifact_flag == clean`, read from 42 and not re-derived;
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

AE = ST.TAB_DIR / "tgt42_artifact_evidence.csv"
if AE.exists():
    pool = pool.merge(pd.read_csv(AE)[["gene", "artifact_flag"]], on="gene", how="left")
else:
    pool["artifact_flag"] = np.nan
    print(f"*** {AE.name} missing -- run 42; the artifact gate is NOT applied below ***")
SL = ST.TAB_DIR / "tgt42_shortlist_skin.csv"
SHORT = pd.read_csv(SL)["gene"].tolist() if SL.exists() else []
pool["on_42_shortlist"] = pool["gene"].isin(SHORT)

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
      f"({int((gate & pool['artifact_flag'].isna()).sum())} without an artifact verdict in 42); "
      f"{len(set(PSHORT) & set(SHORT))} of 42's {len(SHORT)} are among them")
POOL_COLS = ["gene", "cd_name", "log2fc", "log2fc_lo", "log2fc_hi", "pct_a", "pct_b",
             "d_det", "auroc", "auroc_lo", "cohen_d", "log2fc_cells", "log2fc_paired",
             "q_blocked", "log2fc_vs_hc", "auroc_vs_hc", "on_42_shortlist"]
print(pool[gate].sort_values("log2fc", ascending=False)[POOL_COLS]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\non 42's shortlist but NOT the pooled one, with the column that keeps it out:")
_miss = pool[pool["gene"].isin(SHORT) & ~gate]
print(_miss[["gene", "log2fc", "log2fc_lo", "q_blocked", "artifact_flag", "log2fc_paired",
             "n_donors_paired"]].to_string(index=False, float_format=lambda v: f"{v:.3g}")
      if len(_miss) else "  none")

print("\ncontrols -- CCR4 / TNFRSF8 / KIR3DL2 / IL2RA up; CD7 and DPP4 down, AUROC below 0.5:")
_ctl = [g for g in ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS if g in set(pool["gene"])]
print(pool.set_index("gene").loc[_ctl, ["log2fc", "log2fc_lo", "log2fc_hi", "d_det", "auroc",
                                        "auroc_lo", "auroc_hi", "log2fc_cells",
                                        "log2fc_paired", "q_blocked"]]
      .to_string(float_format=lambda v: f"{v:.3g}"))
pool.to_csv(ST.tab("pooled_all_genes"), index=False)
print("\nwrote", ST.tab("pooled_all_genes"))
```

```
2,355 genes | van Elteren testable for 2,355, 525 at q < 0.05
pooled (donor-balanced) vs paired log2FC:        Spearman rho = 0.739
pooled donor-balanced vs cell-weighted log2FC:   Spearman rho = 0.791

pooled shortlist: 85 genes pass every gate (0 without an artifact verdict in 42); 24 of 42's 28 are among them
     gene cd_name  log2fc  log2fc_lo  log2fc_hi  pct_a    pct_b  d_det  auroc  auroc_lo  cohen_d  log2fc_cells  log2fc_paired  q_blocked  log2fc_vs_hc  auroc_vs_hc  on_42_shortlist
    PTH2R            1.61       1.16       2.04  0.106   0.0113 0.0946  0.547     0.531    0.364          2.27           0.53      0.016          1.99        0.552            False
      CPM            1.49       1.21       1.75  0.177   0.0387  0.138  0.569     0.551    0.401          1.53           1.02   0.000677          2.23        0.584             True
     MRC2            1.44       1.14       1.71  0.173   0.0386  0.134  0.567     0.549    0.386          1.95          0.892   2.61e-05          2.19        0.582            False
      CD9            1.34      0.929       1.71  0.289   0.0928  0.196  0.597     0.566    0.456          2.11           1.08   4.13e-05          1.17         0.59             True
     CD70    CD70    1.31       1.09       1.56  0.234    0.068  0.166  0.582     0.568      0.4          1.38            1.1   1.15e-08          2.13        0.604             True
     THY1             1.3      0.801        1.7 0.0766   0.0116  0.065  0.532     0.518    0.283          1.23         0.0513    0.00498           1.5        0.535            False
   COL6A3             1.3       1.03       1.59  0.184   0.0558  0.128  0.564     0.547    0.357          1.67          0.811   1.69e-05          2.22        0.584             True
     SGCE            1.24      0.956       1.49 0.0854  0.00923 0.0762  0.538     0.527    0.316          1.68          0.259     0.0253          1.47        0.542            False
     NRN1            1.22      0.904       1.52  0.085   0.0125 0.0725  0.536     0.526     0.29          1.54          0.776   3.89e-06          1.42        0.539            False
     PLD1             1.2      0.943       1.43  0.113   0.0244 0.0887  0.544     0.532    0.299          1.32          0.736   0.000245          1.42        0.548            False
    OPCML            1.14      0.411       1.75 0.0387  0.00269  0.036  0.518     0.507    0.233           1.5              0     0.0316          1.23        0.519            False
    EPCAM            1.03      0.529       1.49 0.0418  0.00289 0.0389  0.519     0.509    0.244           1.2         0.0128     0.0406          1.08         0.52            False
  SLC29A1            1.02      0.878       1.16  0.115   0.0295 0.0859  0.543     0.534    0.275          1.01          0.841   6.99e-10          1.43        0.552            False
   CHRNA6           0.986      0.686       1.26  0.061   0.0108 0.0502  0.525     0.517    0.234         0.791          0.429      0.022          1.28        0.529            False
   ADGRE1           0.967       0.75       1.19 0.0795   0.0139 0.0656  0.533     0.524    0.253         0.927          0.792    0.00177          1.36        0.539            False
    GRIK2           0.909      0.437       1.33 0.0338 0.000999 0.0328  0.516     0.507    0.226         0.801              0     0.0133         0.948        0.517            False
   SLC4A2           0.896       0.75       1.04  0.162   0.0571  0.105  0.551     0.541    0.261          1.01          0.698   1.31e-08          1.69        0.572             True
  FAM174B           0.896      0.634       1.13 0.0718    0.017 0.0548  0.527     0.519     0.22          1.22          0.491     0.0442          1.11        0.532            False
  TNFRSF8    CD30   0.889       0.52       1.22 0.0758   0.0239 0.0519  0.526     0.515    0.209          0.98          0.332    0.00544          1.63        0.537            False
     FLT1           0.876      0.559       1.16 0.0538   0.0114 0.
... [13891 more chars truncated]
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
           edgecolors=TF.MALIG, linewidths=.6, label=f"42 shortlist only ({int(k2.sum())})")
kc = d["gene"].isin(ST.POSITIVE_CONTROLS + ["CD7", "DPP4"])
ax.scatter(d.loc[kc, "log2fc_paired"], d.loc[kc, "log2fc"], s=18, c=TF.DOWN, linewidths=0)
for _, r_ in d[kc].iterrows():
    ax.annotate(r_["gene"], (r_["log2fc_paired"], r_["log2fc"]), fontsize=5.5, xytext=(3, 2),
                textcoords="offset points")
ax.set_xlabel("paired: median per-donor log2FC (42)")
ax.set_ylabel("pooled: donor-balanced log2FC")
ax.set_title(f"Pooled ranking {'reproduces' if RHO >= 0.8 else 'departs from'} the paired one "
             f"(ρ = {RHO:.2f}, {len(d):,} genes)", loc="left", fontsize=7.5)
ax.legend(fontsize=5.5, frameon=False, loc="upper left")
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F2_pooled_vs_paired", caption_text=(
    "Red: the validated antigens, CD7 and DPP4. Dashed: identity."))
TF.save_source(d[["gene", "log2fc", "log2fc_paired", "log2fc_cells", "pooled_shortlist",
                  "on_42_shortlist"]], "F2_pooled_vs_paired_source")
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
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F2_pooled_vs_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F2_pooled_vs_paired.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F2_pooled_vs_paired_source.csv
largest disagreements, pooled minus paired:
   gene  log2fc  log2fc_paired  delta  log2fc_cells  pct_a   pct_b  n_donors_paired  frac_donors_up  pooled_shortlist
KIR3DL2    1.63              0   1.63           2.1  0.143  0.0209               75            0.36             False
   GNLY  0.0509           -1.3   1.35         0.408  0.265   0.233               75           0.293             False
   NKG7  -0.883          -2.17   1.29        -0.585  0.205   0.293               75            0.28             False
    MME    1.27              0   1.27          1.26 0.0521 0.00404               75           0.453             False
   THY1     1.3         0.0513   1.25          1.23 0.0766  0.0116               75           0.587              True
   SELL  -0.441          -1.63   1.19        -0.363  0.239   0.287               75           0.347             False
  OPCML    1.14              0   1.14           1.5 0.0387 0.00269               75           0.453              True
    CD7   -1.73          -2.85   1.12         -1.94  0.155   0.508               75          0.0933             False
  PTH2R    1.61           0.53   1.08          2.27  0.106  0.0113               75           0.667              True
TSPAN13    1.19          0.114   1.08          1.31 0.0673  0.0131               75           0.627             False
 PRSS21    1.42          0.399   1.02           1.8  0.109  0.0116               75           0.587             False
  EPCAM    1.03         0.0128   1.02           1.2 0.0418 0.00289               75            0.56              True
   SGCE    1.24          0.259   0.98          1.68 0.0854 0.00923               75           0.627              True
 ADAM12   0.958              0  0.958          1.04 0.0941  0.0262               75           0.493             False
  CXCR3  -0.308          -1.24  0.932        -0.669  0.169   0.189               75           0.347             False
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
            fmt="o", ms=3, lw=.8, color=TF.COMPARATOR, label="vs healthy-skin CD4")
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
print("\nli2024 contributes about a third of the patients, so the row to read first is the one "
      "where dropping it moves the effect most.")
LO.to_csv(ST.tab("loso"))
print("\nwrote", ST.tab("loso"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F3_shortlist_auroc.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F3_shortlist_auroc.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F3_shortlist_auroc_source.csv
AUROC beside detection -- a sparse antigen cannot separate many cells, however selective:
         pct_a   pct_b  pct_b_vs_hc  d_det  auroc  auroc_lo  auroc_hi  auroc_vs_hc  cohen_d
gene                                                                                       
CD9      0.289  0.0928        0.107  0.196  0.597     0.566     0.624         0.59    0.456
CD70     0.234   0.068       0.0233  0.166  0.582     0.568     0.597        0.604      0.4
CPM      0.177  0.0387      0.00952  0.138  0.569     0.551     0.588        0.584    0.401
MRC2     0.173  0.0386      0.00921  0.134  0.567     0.549     0.585        0.582    0.386
COL6A3   0.184  0.0558        0.015  0.128  0.564     0.547     0.584        0.584    0.357
PTH2R    0.106  0.0113      0.00108 0.0946  0.547     0.531     0.568        0.552    0.364
PLD1     0.113  0.0244       0.0157 0.0887  0.544     0.532     0.558        0.548    0.299
SLC29A1  0.115  0.0295       0.0119 0.0859  0.543     0.534     0.552        0.552    0.275
SGCE    0.0854 0.00923      0.00212 0.0762  0.538     0.527     0.551        0.542    0.316
NRN1     0.085  0.0125      0.00747 0.0725  0.536     0.526     0.547        0.539     0.29
ADGRE1  0.0795  0.0139      0.00215 0.0656  0.533     0.524     0.543        0.539    0.253
THY1    0.0766  0.0116      0.00616  0.065  0.532     0.518     0.551        0.535    0.283
CHRNA6   0.061  0.0108       0.0026 0.0502  0.525     0.517     0.534        0.529    0.234
EPCAM   0.0418 0.00289      0.00203 0.0389  0.519     0.509     0.533         0.52    0.244
OPCML   0.0387 0.00269     0.000108  0.036  0.518     0.507     0.534        0.519    0.233


leave-one-study-out pooled log2FC:
         pooled  loso_min  loso_max study_moving_most  sign_kept_every_study
gene                                                                        
PTH2R      1.61      1.46      1.74   brentuximab2026                   True
CPM        1.49       1.4      1.54   brentuximab2026                   True
MRC2       1.44      1.29      1.48            li2024                   True
CD9        1.34     0.909      1.44            li2024                   True
CD70       1.31      1.23      1.36         alkon2024                   True
THY1        1.3      1.09      1.37            li2024                   True
COL6A3      1.3     0.936      1.42            li2024                   True
SGCE       1.24      1.07      1.33   brentuximab2026                   True
NRN1       1.22      1.08      1.27            li2024                   True
PLD1        1.2       1.1      1.25            li2024                   True
OPCML      1.14      0.56      1.49         il4ra2026                   True
EPCAM      1.03     0.905       1.1            li2024                   True
SLC29A1    1.02     0.978      1.04      gaydosik2022                   True
CHRNA6    0.986      0.91      1.19            li2024                   True
ADGRE1    0.967     0.904      1.02   chennareddy2025                   True
CCR4      0.562     0.475     0.802            li2024                   True
TNFRSF8   0.889       0.5     0.946            li2024                   True
KIR3DL2    1.63      1.35      1.94            li2024                   True
IL2RA   -0.0975    -0.134   -0.0367   brentuximab2026                   True
CD7       -1.73     -1.83     -1.64      gaydosik2019                   True
DPP4     -0.342    -0.475    -0.277            li2024                   True

li2024 contributes about a third of the patients, so the row to read first is the one where dropping it moves the effect most.

wrote /home/projects/nyosef/zvise/MF/tables/tgt48_loso.csv
```

## §3 — What the pooled cohort can answer

The tiers are 47's — **1** test, **2** describe (an effect with a CI, no p), **3** count only —
but they are now set per arm, because a stratum can be well populated on one side and empty on
the other.

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


def _tri(v, yes="yes"):
    """yes / no / NA. `clin_lct` and friends are curated as unknown where nobody
    reported them, and `target_cohort._clean` has already turned that into NA --
    `.eq("yes")` would silently call every one of those donors negative."""
    return v.map(lambda x: (yes if x == yes else "no") if pd.notna(x) else np.nan)


def _split(v, hi="high", lo="low"):
    med = v.median()
    return v.map(lambda a: np.nan if pd.isna(a) else hi if a >= med else lo), med


_les = PT["lesion"].map(lambda s: "tumour/eryth" if s in ("tumor", "erythroderma")
                        else "patch/plaque" if s in ("patch", "plaque") else np.nan)
_stage = PT["clin_stage"].map(
    lambda s: np.nan if pd.isna(s) else ("IV" if str(s).startswith("IV") else "I-III"))
_age, AGE_MED = _split(PT["age"], "older", "younger")
_bur, BUR_MED = _split(PT["burden"])
FIELDS = {
    "lesion": (_les, "tumour/eryth", "patch/plaque"),
    "disease": (PT["dx2"], "MF", "SS"),
    "stage class": (PT["clin_stage_class"], "advanced", "early"),
    "stage IV vs I-III": (_stage, "IV", "I-III"),
    "B stage": (PT["clin_B"].map({"B2": "B2", "B1": "B0-B1", "B0": "B0-B1"}), "B2", "B0-B1"),
    "CD30": (PT["clin_cd30"].map({"pos": "pos", "neg": "neg"}), "pos", "neg"),
    "large-cell transformation": (_tri(PT["clin_lct"]), "yes", "no"),
    "folliculotropic": (_tri(PT["clin_fmf"]), "yes", "no"),
    "any systemic tx at sampling": (PT["clin_current_systemic"].map({"yes": "yes", "no": "no"}),
                                    "yes", "no"),
    "retinoid": (_yn(PT["tx_retinoid"], PT["tx_known"]), "yes", "no"),
    "ECP": (_yn(PT["tx_ecp"], PT["tx_known"]), "yes", "no"),
    "brentuximab": (_yn(PT["tx_brentuximab"], PT["tx_known"]), "yes", "no"),
    "treatment-naive": (PT["clin_tx_naive"].map({"yes": "yes", "no": "no"}), "yes", "no"),
    "sex": (PT["sex"], "F", "M"),
    f"age (median {AGE_MED:.0f})": (_age, "older", "younger"),
    f"burden (median {BUR_MED:.2f})": (_bur, "high", "low"),
    "cultured ex vivo": (_yn(PT["is_cultured"]), "yes", "no"),
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
    print(f"  {f:30s} {r_['x']:>13s}: {r_['studies_x']}")
    print(f"  {'':30s} {r_['y']:>13s}: {r_['studies_y']}")
print("\ntier 1 = test, 2 = describe, 3 = count only. Treatment fields use only donors whose "
      "treatment at sampling is known.")
POWER.to_csv(ST.tab("power_table"))
print("wrote", ST.tab("power_table"))
```

```
donors per arm and level (mal = with a malignant arm, rea = with a reactive arm):
                                        x             y  n_mal_x  n_mal_y  tier_mal  n_rea_x  n_rea_y  tier_rea  tier_did
field                                                                                                                    
lesion                       tumour/eryth  patch/plaque       31       19         1       34       22         1         1
disease                                MF            SS       69        7         1       72        8         1         1
stage class                      advanced         early       56       20         1       59       20         1         1
stage IV vs I-III                      IV         I-III       20       56         1       22       57         1         1
B stage                                B2         B0-B1        9       45         1       10       49         1         1
CD30                                  pos           neg        9        8         1        9        8         1         1
large-cell transformation             yes            no       23        9         1       23        9         1         1
folliculotropic                       yes            no        5        0         3        5        0         3         3
any systemic tx at sampling           yes            no       15       37         1       16       42         1         1
retinoid                              yes            no        4       48         2        5       53         1         2
ECP                                   yes            no        7       45         1        7       51         1         1
brentuximab                           yes            no        1       51         3        1       57         3         3
treatment-naive                       yes            no        6       47         1        7       49         1         1
sex                                     F             M       32       46         1       32       50         1         1
age (median 65)                     older       younger       39       39         1       42       40         1         1
burden (median 0.65)                 high           low       43       35         1       40       42         1         1
cultured ex vivo                      yes            no        4       74         2        5       77         1         2

study mix per level:
  lesion                          tumour/eryth: il4ra2026:6, li2024:6, brunner2024:5, gaydosik2022:5, chennareddy2025:4, buus2025:3, brentuximab2026:3, gaydosik2019:2, herrera2021:1
                                  patch/plaque: alkon2024:5, rindler2021_mc:4, li2024:4, brentuximab2026:3, herrera2021:3, gaydosik2019:1, jonak2021:1, chennareddy2025:1
  disease                                   MF: li2024:35, il4ra2026:9, brentuximab2026:6, alkon2024:5, chennareddy2025:5, rindler2021_mc:4, gaydosik2022:3, brunner2024:3, gaydosik2019:2, jonak2021:1, herrera2021:1
                                            SS: buus2025:3, herrera2021:3, brunner2024:2, gaydosik2019:1
  stage class                         advanced: li2024:19, il4ra2026:9, brentuximab2026:6, chennareddy2025:5, gaydosik2022:5, brunner2024:5, rindler2021_mc:4, herrera2021:4, gaydosik2019:3
                                         early: li2024:16, alkon2024:5, jonak2021:1
  stage IV vs I-III                         IV: brunner2024:5, herrera2021:4, li2024:4, chennareddy2025:3, brentuximab2026:2, rindler2021_mc:2, il4ra2026:1, gaydosik2022:1, gaydosik2019:1
                                         I-III: li2024:31, il4ra2026:8, alkon2024:5, gaydosik2022:4, brentuximab2026:4, rindler2021_mc:2, chennareddy2025:2, gaydosik2019:2, jonak2021:1
  B stage                                   B2: herrera2021:3, rindler2021_mc:2, gaydosik2022:1, brunner2024:1, chennareddy2025:1, gaydosik2019:1, li2024:1
                                         B0-B1: li2024:16, il4ra2026:9, brentuximab2026:6, alkon2024:5, 
... [4171 more chars truncated]
```

## §4 — Stratified: does the antigen depend on who the patient is

Four contrasts per field, each answering a different question:

| contrast | compares | question |
|---|---|---|
| **(a) within** | malignant vs reactive, inside stratum x and inside stratum y | is the antigen tumour-selective in this kind of patient |
| **(b) tumour** | malignant cells of x vs malignant cells of y | does the tumour itself differ between these patients |
| **(c) reactive** | reactive cells of x vs reactive cells of y | does every T cell differ — stage and treatment can act on the whole compartment |
| **(d) DiD** | (a) in x minus (a) in y | does the tumour-vs-reactive effect differ: the interaction 47 could test on paired donors only |

How the contrasts combine:
- (b) without (c) is a tumour-specific difference.
- (b) and (c) in the same direction is a compartment-wide difference, and (d) is then near zero.
  In skin this is the common case for the lesion axis: a tumour nodule's whole T compartment sits
  in different tissue from a patch's.

The p-values come from permuting donors.
- `q_study` permutes only within study. It is the one to read, because lesion type and treatment
  are concentrated in particular studies.
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
genes at q < 0.05, per field and contrast (of 2,355); empty = not tested at that tier, or fewer than 20 within-study arrangements:
                                                   x vs y  mal n  rea n tier mal/rea/did  log10 cfg study  tumour q_study  tumour q_free  reactive q_study  reactive q_free  DiD q_study  DiD q_free
field                                                                                                                                                                                               
lesion                       tumour/eryth vs patch/plaque  31/19  34/22            1/1/1             4.28               0              1                 0                0            0           0
disease                                          MF vs SS   69/7   72/8            1/1/1             1.73               0              0                 0                0            0          34
stage class                             advanced vs early  56/20  59/20            1/1/1             8.67               0              3                 0                0            0           2
stage IV vs I-III                             IV vs I-III  20/56  22/57            1/1/1             8.73               0              0                 1                0            0           0
B stage                                       B2 vs B0-B1   9/45  10/49            1/1/1             4.18               0              1                 0                0            0           0
CD30                                           pos vs neg    9/8    9/8            1/1/1             1.38               0              0                 0                0            0           0
large-cell transformation                       yes vs no   23/9   23/9            1/1/1            0.903             NaN            166               NaN              285          NaN           0
folliculotropic                                 yes vs no    5/0    5/0            3/3/3              NaN             NaN            NaN               NaN              NaN          NaN         NaN
any systemic tx at sampling                     yes vs no  15/37  16/42            1/1/1             6.08               0              0                 0                0            0           0
retinoid                                        yes vs no   4/48   5/53            2/1/2              NaN             NaN            NaN                 0                0          NaN         NaN
ECP                                             yes vs no   7/45   7/51            1/1/1             2.38               0              0                 0                0            0           0
brentuximab                                     yes vs no   1/51   1/57            3/3/3              NaN             NaN            NaN               NaN              NaN          NaN         NaN
treatment-naive                                 yes vs no   6/47   7/49            1/1/1             3.38               0              0                 0                0            0           0
sex                                                F vs M  32/46  32/50            1/1/1             14.4               0              0                 0                0            0           0
age (median 65)                          older vs younger  39/39  42/40            1/1/1             13.5               0              0                 0                0            0           0
burden (median 0.65)                          high vs low  43/35  40/42            1/1/1             15.9               0              0                 0                0            2           0
cultured ex vivo                                yes vs no   4/74   5/77            2/1/2              NaN             NaN            NaN                 0               83          NaN         NaN


wrote
  /home/projects/nyosef/zvise/MF/tables/tgt48_strat_tumour.csv
  /home/projects/nyosef/zvise/MF/tables/tgt48_strat_rea
... [196 more chars truncated]
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

JOIN_COLS = ["gene", "log2fc", "pct_x", "pct_y", "auroc", "q_study", "q_free",
             "log2fc_rea", "q_study_rea", "did", "q_study_did", "reading",
             "on_shortlist"]
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
    print(top[[c for c in JOIN_COLS if c in top]].to_string(index=False,
                                                     float_format=lambda v: f"{v:.3g}"))

print("\n" + "=" * 78)
print("the leading antigens on tumour cells, every field (log2FC x vs y):")
TOP_G = (PSHORT or SHORT)[:15]
print(TUM[TUM["gene"].isin(TOP_G)].pivot_table(index="gene", columns="field", values="log2fc")
      .reindex(index=TOP_G, columns=list(FIELDS)).round(2).to_string())
J.to_csv(ST.tab("strat_joined"), index=False)
print("\nwrote", ST.tab("strat_joined"))
```

```
==============================================================================
lesion: tumour/eryth vs patch/plaque | tumour cells from 31 vs 19 donors | tier 1
==============================================================================
  0 genes at q_study < 0.05 (1 at q_free); readings: {}
    gene  log2fc    pct_x    pct_y  auroc  q_study  q_free  log2fc_rea  q_study_rea     did  q_study_did reading  on_shortlist
  COL1A1  -0.415   0.0404   0.0629  0.489    0.881   0.885      -0.259        0.696  -0.156            1                 False
    DEF6   0.262    0.414    0.306  0.544    0.881   0.871       0.143        0.947   0.119            1                 False
   SRP68   0.575    0.214    0.109   0.55    0.881   0.406       0.361        0.747   0.214        0.941                 False
  ICOSLG   0.163   0.0101  0.00314  0.503    0.881   0.871      0.0508            1   0.112        0.941                 False
   MILR1  0.0518  0.00206 0.000332  0.501    0.881   0.885      0.0152        0.996  0.0366            1                 False
  OLFML3 -0.0554   0.0026  0.00383  0.499    0.881   0.885     -0.0175            1 -0.0379            1                 False
     GPI   0.319    0.506    0.396  0.552    0.881   0.871       0.252        0.865  0.0667            1                  True
   GRIN1   0.203  0.00714  0.00139  0.503    0.881   0.871      0.0101        0.785   0.193        0.754                 False
  CLEC5A   0.012 0.000559  2.2e-05    0.5    0.881   0.871     0.00724            1 0.00479            1                 False
   TACR2  0.0247  0.00136 0.000237  0.501    0.881   0.871     -0.0314            1  0.0562        0.872                 False
CACNA2D1  -0.104  0.00106  0.00493  0.498    0.881   0.871      0.0107            1  -0.115        0.941                 False
   SLIT3 -0.0998  0.00382  0.00557  0.499    0.881   0.948     -0.0236            1 -0.0762        0.741                 False

==============================================================================
disease: MF vs SS | tumour cells from 69 vs 7 donors | tier 1
==============================================================================
  0 genes at q_study < 0.05 (0 at q_free); readings: {}
  gene  log2fc   pct_x   pct_y  auroc  q_study  q_free  log2fc_rea  q_study_rea      did  q_study_did reading  on_shortlist
ADAM17   0.217   0.134   0.105  0.515        1   0.885       0.223            1 -0.00539            1                 False
 APLP2   0.434   0.219   0.164  0.529        1   0.571       -0.22            1    0.654        0.781                 False
ATP2B4   0.924   0.316   0.172  0.577        1   0.232        0.55            1    0.374        0.781                 False
  CANX   0.473   0.451   0.353  0.562        1   0.443      0.0371            1    0.436        0.781                 False
 CASP7    0.68  0.0493  0.0141  0.518        1   0.137     -0.0141            1    0.694        0.781                 False
 CD274   0.647  0.0474  0.0177  0.515        1   0.811      0.0136            1    0.634        0.781                 False
  CTSK   0.569  0.0283 0.00693  0.511        1   0.541       0.326            1    0.243        0.798                 False
 DIP2A   0.512   0.176   0.124  0.528        1   0.596       0.185            1    0.327        0.798                 False
 ERAP1    1.03   0.136  0.0585   0.54        1   0.232       0.464            1    0.563        0.781                 False
 ERBB2    0.25   0.011 0.00465  0.503        1   0.552      -0.104            1    0.354        0.781                 False
 FLOT1    1.05    0.19  0.0788  0.557        1    0.44       0.492            1    0.555        0.943                 False
 GALR2  -0.364 0.00233  0.0155  0.493        1  0.0964      -0.133            1   -0.231        0.943                 False

==============================================================================
stage class: advanced vs early | tumour cells from 56 vs 20 donors | tier 1
========
... [29793 more chars truncated]
```

```python  # cell 15
# ============================================================================
# §4c F4-F6  The antigen inside each stratum: lesion, stage, disease
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


forest("lesion", "F4_forest_lesion")
forest("stage class", "F5_forest_stage")
forest("disease", "F6_forest_disease")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F4_forest_lesion.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F4_forest_lesion.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F4_forest_lesion_source.csv
lesion:
        log2fc_x  log2fc_y      did  q_study  q_free
gene                                                
PTH2R        1.8      1.69    0.105        1       1
CPM          1.5      1.71    -0.21        1       1
CD70        1.37      1.54   -0.171        1       1
MRC2        1.27      1.39    -0.12        1       1
SGCE        1.55      1.03    0.521    0.768       1
NRN1        1.27     0.925    0.343        1       1
THY1       0.448      1.74     -1.3    0.911   0.942
COL6A3      1.05      1.12  -0.0752        1       1
EPCAM       1.24     0.791    0.452        1       1
CD9        0.878     0.977  -0.0996        1       1
PLD1       0.917     0.925 -0.00811        1       1
OPCML       1.41     0.147     1.26        1       1

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F5_forest_stage.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F5_forest_stage.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F5_forest_stage_source.csv
stage class:
        log2fc_x  log2fc_y    did  q_study  q_free
gene                                              
CD70        1.13      1.92 -0.797    0.708   0.267
CPM         1.54      1.46 0.0882        1       1
CD9          1.2      1.76 -0.554        1    0.79
MRC2        1.45      1.43 0.0152        1       1
COL6A3      1.15      1.61 -0.466    0.937   0.757
THY1        1.17      1.57 -0.397    0.993   0.947
PLD1        1.06      1.51 -0.443    0.591   0.652
NRN1        1.21      1.32 -0.112        1       1
PTH2R       1.72      0.53   1.19    0.606   0.281
SGCE        1.39     0.637   0.75    0.586    0.37
OPCML       1.14     0.484  0.651        1   0.995
EPCAM       1.23     0.387  0.844    0.743   0.712

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F6_forest_disease.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F6_forest_disease.pdf

<Figure size 500x300 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F6_forest_disease_source.csv
disease:
        log2fc_x  log2fc_y     did  q_study  q_free
gene                                               
PTH2R        1.5      2.26  -0.761        1   0.772
OPCML       1.06      1.87  -0.809        1    0.97
CPM         1.52      1.37   0.153        1   0.985
NRN1        1.21      1.47  -0.261        1   0.926
SGCE        1.24      1.24 0.00178        1       1
CD70        1.37     0.897   0.475    0.798   0.716
COL6A3      1.37     0.716   0.653        1   0.623
MRC2        1.53     0.266    1.26    0.798   0.161
THY1        1.38     0.142    1.24    0.798   0.444
PLD1        1.28     0.149    1.13        1   0.153
CD9         1.61    -0.525    2.13    0.798  0.0485
EPCAM       1.13   -0.0521    1.18    0.943   0.399
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
    "lesion": "lesion: tumour/erythroderma vs patch/plaque",
    "disease": "disease: MF vs SS",
    "stage class": "stage: advanced vs early",
    "stage IV vs I-III": "stage: IV vs I–III",
    "B stage": "blood stage: B2 vs B0–B1",
    "CD30": "CD30: positive vs negative",
    "large-cell transformation": "large-cell transformation: yes vs no",
    "folliculotropic": "folliculotropic MF: yes vs no",
    "any systemic tx at sampling": "systemic treatment at sampling: yes vs no",
    "retinoid": "retinoid at sampling: yes vs no",
    "ECP": "ECP at sampling: yes vs no",
    "brentuximab": "brentuximab at sampling: yes vs no",
    "treatment-naive": "treatment-naive: yes vs no",
    "sex": "sex: F vs M",
    f"age (median {AGE_MED:.0f})": f"age: ≥{AGE_MED:.0f} vs <{AGE_MED:.0f} y",
    f"burden (median {BUR_MED:.2f})": f"malignant share of CD4: ≥{BUR_MED:.2f} vs <{BUR_MED:.2f}",
    "cultured ex vivo": "cultured ex vivo: yes vs no",
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
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F7_strata_tumour_heat.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt48_F7_strata_tumour_heat.pdf

<Figure size 800x550 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt48_F7_strata_tumour_heat_source.csv
```

```python  # cell 17
# ============================================================================
# §4e  Sensitivity: the lesion contrast without the cultured donors and without li2024
# ============================================================================
# Two things could carry the lesion result on their own: the seven geskin-style vehicle wells
# that spent time in culture, and li2024, which contributes about a third of the patients. Both
# stay in the cohort, flagged; here they leave it.
EXCL_SETS = {
    "cultured ex vivo": PT.index[PT["is_cultured"]],
    "li2024": PT.index[PT["study"].eq("li2024")],
}
sens = []
for name in ("lesion", "disease"):
    lab, x, y_ = FIELDS[name]
    t1 = TUM[TUM["field"].eq(name)].set_index("gene")
    d1 = DID[DID["field"].eq(name)].set_index("gene")
    for ex_name, ex in EXCL_SETS.items():
        lab2 = lab.dropna().drop(ex, errors="ignore")
        if lab2.value_counts().min() < 3 or lab2.nunique() < 2:
            print(f"{name} without {ex_name}: a stratum falls below 3 donors -- not run")
            continue
        t2 = P.between(mal, lab2, x, y_, block=STUDY, n_perm=N_PERM, seed=400).set_index("gene")
        d2 = P.did(mal, rea, lab2, x, y_, block=STUDY, n_perm=N_PERM, seed=401).set_index("gene")
        s_all = set(t1.index[t1["q_study"] < Q])
        s_ex = set(t2.index[t2["q_study"] < Q]) if "q_study" in t2 else set()
        sens.append({"field": name, "excluded": ex_name, "n_excluded": int(len(ex)),
                     "n_mal_x/y": f"{int(t2['n_x'].iloc[0])}/{int(t2['n_y'].iloc[0])}",
                     "rho_tumour_lfc": stats.spearmanr(t1["log2fc"],
                                                       t2.loc[t1.index, "log2fc"]).statistic,
                     "rho_did": stats.spearmanr(d1["did"], d2.loc[d1.index, "did"],
                                                nan_policy="omit").statistic,
                     "tumour q<.05 all": len(s_all), "tumour q<.05 excl": len(s_ex),
                     "in both": len(s_all & s_ex)})
        _g = [g for g in (PSHORT or SHORT)[:12] if g in t2.index]
        print(f"\n{name}, excluding {ex_name} -- tumour-cell log2FC and DiD, "
              "all donors vs excluded:")
        print(pd.DataFrame({"tumour_all": t1.loc[_g, "log2fc"],
                            "tumour_excl": t2.loc[_g, "log2fc"],
                            "did_all": d1.loc[_g, "did"], "did_excl": d2.loc[_g, "did"]})
              .to_string(float_format=lambda v: f"{v:.3g}"))
sens = pd.DataFrame(sens)
print("\n" + sens.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
sens.to_csv(ST.tab("sensitivity_excl"), index=False)
print("\nwrote", ST.tab("sensitivity_excl"))
```

```
lesion, excluding cultured ex vivo -- tumour-cell log2FC and DiD, all donors vs excluded:
        tumour_all  tumour_excl  did_all  did_excl
gene                                              
PTH2R        0.415          0.5    0.105     0.272
CPM         0.0859       0.0817    -0.21    -0.167
MRC2        -0.115       -0.168    -0.12   -0.0549
CD9         -0.487       -0.519  -0.0996    -0.199
CD70        0.0217        0.016   -0.171    -0.173
THY1         -1.29        -1.26     -1.3     -1.28
COL6A3      -0.709       -0.637  -0.0752   -0.0281
SGCE         0.517        0.614    0.521      0.68
NRN1         0.385        0.474    0.343     0.459
PLD1       -0.0791       -0.186 -0.00811   -0.0804
OPCML         1.28        0.703     1.26     0.696
EPCAM        0.601        0.696    0.452     0.522


lesion, excluding li2024 -- tumour-cell log2FC and DiD, all donors vs excluded:
        tumour_all  tumour_excl  did_all  did_excl
gene                                              
PTH2R        0.415        0.338    0.105   0.00972
CPM         0.0859       0.0433    -0.21    -0.174
MRC2        -0.115       -0.325    -0.12    -0.589
CD9         -0.487       -0.492  -0.0996     -0.36
CD70        0.0217       -0.111   -0.171    -0.234
THY1         -1.29       -0.874     -1.3     -0.87
COL6A3      -0.709       -0.779  -0.0752   -0.0688
SGCE         0.517        0.313    0.521     0.351
NRN1         0.385        0.321    0.343     0.319
PLD1       -0.0791      -0.0883 -0.00811   -0.0171
OPCML         1.28         1.44     1.26      1.41
EPCAM        0.601         1.13    0.452     0.976
disease without cultured ex vivo: a stratum falls below 3 donors -- not run
disease without li2024: a stratum falls below 3 donors -- not run

 field         excluded  n_excluded n_mal_x/y  rho_tumour_lfc  rho_did  tumour q<.05 all  tumour q<.05 excl  in both
lesion cultured ex vivo           5     28/19           0.924    0.912                 0                  0        0
lesion           li2024          35     26/17           0.864    0.773                 0                  0        0

wrote /home/projects/nyosef/zvise/MF/tables/tgt48_sensitivity_excl.csv
```

## §5 — Lesion, disease, stage, sex and age together

The §4 contrasts take one field at a time, and the fields are correlated: tumour lesions are
advanced-stage by definition, and both are concentrated in particular studies.

The model is one OLS per gene on donor pseudobulk, with each donor's arm summarised as
log2(mean + 0.05):
- terms `is_tumour_lesion + is_MF + is_advanced + female + age_z`;
- HC3 standard errors, BH correction per term.

It is run on the malignant arm and, separately, on the reactive arm. Each term can then be read
as tumour-specific (malignant only), compartment-wide (both arms, same sign) or reactive-only.
For the lesion axis the compartment-wide reading is the one to expect, and the tumour-specific
rows are the interesting minority.

Study stays out of the design for the reason 47 gives: with this many studies it would absorb the
lesion term it is confounded with. It is reported instead as its own one-way ANOVA on the
malignant arm.

```python  # cell 19
# ============================================================================
# §5  Joint model on donor pseudobulk, malignant and reactive arms
# ============================================================================
X = pd.DataFrame(index=PT.index)
X["is_tumour_lesion"] = PT["lesion"].isin(["tumor", "erythroderma"]).astype(float)
X["is_MF"] = PT["dx2"].map({"MF": 1.0, "SS": 0.0})
X["is_advanced"] = PT["clin_stage_class"].map({"advanced": 1.0, "early": 0.0})
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
      f"(tumour lesion {int(Xm['is_tumour_lesion'].sum())}, MF {int(Xm['is_MF'].sum())}, "
      f"advanced {int(Xm['is_advanced'].sum())}, F {int(Xm['is_F'].sum())}), "
      f"{len(reg_m):,} genes detected in >= {MIN_DET_DONORS} of them; reactive arm: {len(Xr)}, "
      f"{len(reg_r):,} genes")
print("\nlesion x stage among the malignant-arm donors -- the collinearity the model carries:")
print(pd.crosstab(Xm["is_tumour_lesion"].map({1.0: "tumour/eryth", 0.0: "other"}),
                  Xm["is_advanced"].map({1.0: "advanced", 0.0: "early"})).to_string())
print("\nterm correlations:")
print(Xm.corr().round(2).to_string())

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
TOP_G_REG = [g for g in (PSHORT or SHORT)[:15] if g in set(reg["gene"])]
print(reg.set_index("gene").loc[TOP_G_REG, [f"b_{t}_mal" for t in TERMS]
                                + [f"q_{t}_mal" for t in TERMS]]
      .to_string(float_format=lambda v: f"{v:.3g}"))

Ym = pd.DataFrame(np.log2(mal.M + P.PSEUDO), index=mal.donors, columns=mal.genes)
Ym = Ym[reg_m["gene"]]
st = STUDY.reindex(Ym.index)
keep_st = st.value_counts()[lambda v: v >= 3].index
anv = pd.DataFrame({"gene": Ym.columns, "p_study": [
    stats.f_oneway(*[Ym.loc[st.eq(s), g] for s in keep_st]).pvalue for g in Ym.columns]})
anv["q_study"] = multipletests(anv["p_study"].fillna(1.0), method="fdr_bh")[1]
print(f"\nstudy, one-way ANOVA on the malignant arm over {len(keep_st)} studies with >= 3 "
      f"donors: {int((anv['q_study'] < Q).sum()):,} of {len(anv):,} genes at q < {Q} -- the "
      "size of the term the model does not carry.")
reg.merge(anv, on="gene", how="left").to_csv(ST.tab("joint_model"), index=False)
print("wrote", ST.tab("joint_model"))
```

```
malignant arm: 74 donors with every covariate (tumour lesion 27, MF 69, advanced 54, F 30), 1,957 genes detected in >= 10 of them; reactive arm: 77, 1,800 genes

lesion x stage among the malignant-arm donors -- the collinearity the model carries:
is_advanced       advanced  early
is_tumour_lesion                 
other                   27     20
tumour/eryth            27      0

term correlations:
                  is_tumour_lesion  is_MF  is_advanced  is_F  age_z
is_tumour_lesion              1.00  -0.24         0.46  0.06   0.18
is_MF                        -0.24   1.00        -0.16  0.00  -0.05
is_advanced                   0.46  -0.16         1.00 -0.06   0.05
is_F                          0.06   0.00        -0.06  1.00  -0.24
age_z                         0.18  -0.05         0.05 -0.24   1.00

genes at q < 0.05 per term, by reading:
                  is_tumour_lesion  is_MF  is_advanced  is_F  age_z
compartment-wide                 0     11            0     0      0
opposite                         0      1            0     0      0
reactive-only                    0    132            0     0      0
tumour-specific                  0     73            3     0      1

--- is_tumour_lesion: tumour-specific, strongest 10
  none

--- is_MF: tumour-specific, strongest 10
  gene  b_is_MF_mal  q_is_MF_mal  b_is_MF_rea  q_is_MF_rea  on_shortlist
 CD226         1.53     0.000119       -0.247        0.863         False
  CD96         1.49       0.0232       -0.115        0.893         False
 VAMP2         1.37      0.00516        0.516        0.153         False
  BIN1         -1.2       0.0323       -0.395        0.347         False
ADGRE5         1.14       0.0161      -0.0283        0.974         False
TGFBR2         1.12      0.00324        0.184        0.884         False
 FURIN         1.06       0.0492      -0.0357        0.988         False
  IL4R         1.01       0.0029        0.289        0.847         False
 TMED7            1       0.0346        0.704        0.161         False
  MRC2        0.979       0.0339       -0.195        0.725          True

--- is_advanced: tumour-specific, strongest 10
gene  b_is_advanced_mal  q_is_advanced_mal  b_is_advanced_rea  q_is_advanced_rea  on_shortlist
CCR7               1.71             0.0199                0.4              0.968         False
 FAS              -1.59           1.35e-05             0.0996              0.992         False
NEU1              0.784             0.0199              0.496              0.968         False

--- is_F: tumour-specific, strongest 10
  none

--- age_z: tumour-specific, strongest 10
  gene  b_age_z_mal  q_age_z_mal  b_age_z_rea  q_age_z_rea  on_shortlist
MGAT4A        0.485       0.0417       0.0385        0.929         False

the leading antigens, malignant-arm coefficients (log2 units per level / per SD of age):
         b_is_tumour_lesion_mal  b_is_MF_mal  b_is_advanced_mal  b_is_F_mal  b_age_z_mal  q_is_tumour_lesion_mal  q_is_MF_mal  q_is_advanced_mal  q_is_F_mal  q_age_z_mal
gene                                                                                                                                                                     
PTH2R                    0.0236         -0.3              0.964       0.233       -0.156                   0.991        0.805               0.71       0.976        0.871
CPM                        0.25       -0.353              0.237       0.645       -0.126                     0.9        0.738              0.892       0.976        0.896
MRC2                    -0.0392        0.979              0.406       0.373       -0.439                   0.991       0.0339              0.838       0.976        0.607
CD9                      -0.347        0.721            -0.0115      -0.238         0.13                   0.871        0.573              0.996       0.976        0.912
CD70                      0.405        0.264             -0.172      -0.053       0.0198                    0.86         0.81    
... [1965 more chars truncated]
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
kv("  of which on 42's shortlist", len(set(PSHORT) & set(SHORT)))
kv("42's shortlist missing from pooled", sorted(set(SHORT) - set(PSHORT)))
for g in [g for g in ("CCR4", "TNFRSF8", "KIR3DL2", "IL2RA", "CD7", "DPP4")
          if g in set(pool["gene"])]:
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
    f"{N_PAIRED} paired; the metadata gaps of 47 are unchanged.",
    "A clean lesion result: the lesion is confounded with stage and with study, and §4e is the "
    "only place that is probed directly.",
    "A treatment-effect study. Treatment at sampling is known for about half the cohort and is "
    "concentrated in particular studies; q_study is the honest p, and for several splits it has "
    "few arrangements to work with.",
    "A protein result: transcript only, and skin carries no CITE-seq.",
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
  independent donor keys                   109
pattern
paired            75
healthy           21
reactive only      7
no CD4 arm         3
malignant only     3
  patients in a pool                       85 (85 people)

==============================================================================
2 - POOLS
==============================================================================
  headline weighting                       donor
               n_donors  n_cells  kish_n_eff_cells  top_donor_share_cells        top_study  top_study_share_cells
arm                                                                                                              
CD4_malignant        78   187748              33.4                 0.0555           li2024                  0.476
CD4_reactive         82   147233              11.4                  0.261           li2024                  0.622
CD4_HC               21     4724              12.4                  0.166  chennareddy2025                  0.256

==============================================================================
3 - HEADLINE: POOLED MALIGNANT VS REACTIVE
==============================================================================
  genes                                    2355
  van Elteren q < 0.05                     525
  rho vs paired log2FC                     0.739
  pooled shortlist                         85
    of which on 42's shortlist             24
  42's shortlist missing from pooled       ['GPI', 'SURF4', 'TMEM30A', 'TNFRSF4']
    CCR4                                   log2FC +0.56 [+0.22, +0.89]  AUROC 0.554  q 0.97
    TNFRSF8                                log2FC +0.89 [+0.52, +1.22]  AUROC 0.526  q 0.0054
    KIR3DL2                                log2FC +1.63 [+1.16, +2.02]  AUROC 0.561  q 0.78
    IL2RA                                  log2FC -0.10 [-0.36, +0.16]  AUROC 0.505  q 0.35
    CD7                                    log2FC -1.73 [-2.29, -1.28]  AUROC 0.316  q 1.2e-11
    DPP4                                   log2FC -0.34 [-0.80, +0.04]  AUROC 0.491  q 0.00035

==============================================================================
4 - STRATA
==============================================================================
                                                   x vs y  mal n  rea n tier mal/rea/did  log10 cfg study  tumour q_study  tumour q_free  reactive q_study  reactive q_free  DiD q_study  DiD q_free
field                                                                                                                                                                                               
lesion                       tumour/eryth vs patch/plaque  31/19  34/22            1/1/1             4.28               0              1                 0                0            0           0
disease                                          MF vs SS   69/7   72/8            1/1/1             1.73               0              0                 0                0            0          34
stage class                             advanced vs early  56/20  59/20            1/1/1             8.67               0              3                 0                0            0           2
stage IV vs I-III                             IV vs I-III  20/56  22/57            1/1/1             8.73               0              0                 1                0            0           0
B stage                                       B2 vs B0-B1   9/45  10/49            1/1/1             4.18               0              1                 0                0            0           0
CD30                                           pos vs neg    9/8    9/8            1/1/1             1.38               0              0                 0                0            0           0
... [5243 more chars truncated]
```

```python  # cell 21
# ============================================================================
# §7  The complete numerical result, as one block to hand to another model
# ============================================================================
_ctx = f"""
Compartment: MF/CTCL **skin**, pooled rather than paired. Notebook 42 compares malignant and
reactive CD4 cells WITHIN each donor; this one pools all malignant CD4 cells against all reactive
CD4 cells, which lets donors with only one arm contribute and makes clinical strata comparable as
cell populations.

Cohort: {len(DON)} independent donor keys ({N_PAIRED} with both arms, {N_ONE_ARM} with one arm,
{N_HC} healthy-skin donors). Non-CD4 clones, diagnoses outside the MF/SS spectrum and repeat
biopsies of one patient were removed in 42 section 0b. Malignancy call `{man['malignancy_call']}`.

Weighting: donor-balanced (every donor contributes equal weight to its arm) is the headline;
cell-weighted is printed beside it as `*_cells`. Pooled means are of log1p CP10K.

Effect sizes: `log2fc` (log2 ratio of pooled means, pseudocount 0.05), `d_det` (difference in the
fraction of cells detected), `cohen_d`, and `auroc` (probability a random malignant cell exceeds a
random comparator cell; 0.5 = no separation, and it is capped by detection for sparse genes).
CIs are a {N_BOOT}-draw donor bootstrap. The headline p-value is van Elteren (`q_blocked`):
donor-level means ranked within study.

Stratified contrasts, per field: (a) malignant vs reactive within each stratum, (b) malignant
cells of one stratum vs the other, (c) the same for reactive cells, (d) difference-in-differences.
p-values come from permuting donor labels within study ({N_PERM:,} permutations, `q_study`), with
a free permutation printed beside them (`q_free`). A field is tier 1 (tested), 2 (described) or 3
(counted) according to how many donors each side has in each arm.

Skin-specific: the lesion that was biopsied (patch / plaque / tumour / erythroderma) is a
stratification the blood cohort does not have, and it is confounded with stage and with the
malignant burden. Everything is transcript; skin carries no CITE-seq.
"""

_sections = [
    {"name": "§0b Cohort by arm pattern",
     "what": "how many independent donors have both CD4 arms, one, or are healthy.",
     "how": "an arm counts when it holds >= 20 cells.",
     "table": pd.crosstab(DON["pattern"], DON["dx"], margins=True),
     "file": str(ST.tab("cohort").relative_to(NB_DIR))},
    {"name": "§1 Pool composition",
     "what": "who each pool is actually made of, under each weighting.",
     "how": "Kish effective donors (sum(w)^2/sum(w^2)); top_study_share_cells is the share of "
            "the cell-weighted pool held by its largest study.",
     "table": summ, "file": str(ST.tab("pool_composition").relative_to(NB_DIR))},
    {"name": "§2a Pooled shortlist",
     "what": "the genes that pass every pooled gate: clean in 42, log2FC > 0.5 with the "
             "bootstrap CI above zero, van Elteren q < 0.05, not a loss marker or lineage "
             "control.",
     "how": "donor-balanced pooled contrast, malignant vs reactive CD4.",
     "table": pool[gate].sort_values("log2fc", ascending=False)[POOL_COLS],
     "file": str(ST.tab("pooled_all_genes").relative_to(NB_DIR))},
    {"name": "§2a Controls",
     "what": "the antigens with a known answer, pooled.",
     "how": "same contrast. CCR4, TNFRSF8, KIR3DL2 and IL2RA should be up; CD7 and DPP4 down "
            "with AUROC below 0.5.",
     "table": pool.set_index("gene").loc[_ctl, ["log2fc", "log2fc_lo", "log2fc_hi", "d_det",
                                                "auroc", "auroc_lo", "auroc_hi",
                                                "log2fc_paired", "q_blocked"]]},
    {"name": "§2b Pooled vs paired agreement",
     "what": "whether pooling reproduces 42's paired ranking, and where it does not.",
     "how": "Spearman over all genes, then the largest absolute disagreements.",
     "table": {"spearman_pooled_vs_paired": round(float(RHO), 3),
               "spearman_donor_balanced_vs_cell_weighted": round(float(_rho_c), 3)}},
    {"name": "§2c Cell-level separation and leave-one-study-out",
     "what": "how well each leading antigen separates single cells, and whether any one study "
             "carries the effect.",
     "how": "pooled AUROC with a donor-bootstrap CI; then the pooled log2FC recomputed with "
            "each study dropped in turn.",
     "table": lo_sum, "file": str(ST.tab("loso").relative_to(NB_DIR))},
    {"name": "§3 Power table",
     "what": "donors per arm on each side of every candidate split, and the resulting tier.",
     "how": "tier 1 = tested, 2 = described, 3 = counted only.",
     "table": POWER[[c for c in POWER.columns if not c.startswith("studies")]],
     "file": str(ST.tab("power_table").relative_to(NB_DIR))},
    {"name": "§4a Stratified summary",
     "what": "how many genes differ between the two levels of each field, for tumour cells, "
             "reactive cells and the difference-in-differences.",
     "how": "within-study donor permutation (q_study) and free permutation (q_free), BH.",
     "table": SUMMARY, "file": str(ST.tab("strat_summary").relative_to(NB_DIR))},
    {"name": "§4b Leading antigens across every stratum",
     "what": "each leading antigen's tumour-cell log2FC between the levels of every field.",
     "how": "pooled, donor-balanced, first level over second.",
     "table": TUM[TUM["gene"].isin(TOP_G)].pivot_table(index="gene", columns="field",
                                                      values="log2fc"),
     "file": str(ST.tab("strat_joined").relative_to(NB_DIR))},
    {"name": "§4e Sensitivity",
     "what": "whether the lesion and disease contrasts survive dropping the cultured donors and "
             "dropping li2024.",
     "how": "the same contrast refitted on the reduced cohort; rho is against the full-cohort "
            "per-gene values.",
     "table": sens, "file": str(ST.tab("sensitivity_excl").relative_to(NB_DIR))},
    {"name": "§5 Joint model readings",
     "what": "for each donor-level term, how many genes are tumour-specific, compartment-wide, "
             "opposite or reactive-only.",
     "how": "OLS per gene on donor pseudobulk log2(mean + 0.05), HC3 errors, BH per term, run "
            "separately on the malignant and reactive arms.",
     "table": pd.DataFrame({t: reg[f"reading_{t}"].value_counts()
                            for t in TERMS}).drop(index="", errors="ignore").fillna(0).astype(int),
     "file": str(ST.tab("joint_model").relative_to(NB_DIR))},
    {"name": "§5 Leading antigens, malignant-arm coefficients",
     "what": "the size of each donor-level effect on the leading antigens.",
     "how": "log2 units per level, or per SD of age.",
     "table": reg.set_index("gene").loc[TOP_G_REG, [f"b_{t}_mal" for t in TERMS]
                                               + [f"q_{t}_mal" for t in TERMS]]},
]
TF.llm_export("48", "Skin, pooled and stratified", _ctx, _sections)
```

```
======================== COPY FROM HERE ========================

# 48 — Skin, pooled and stratified

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**, pooled rather than paired. Notebook 42 compares malignant and
reactive CD4 cells WITHIN each donor; this one pools all malignant CD4 cells against all reactive
CD4 cells, which lets donors with only one arm contribute and makes clinical strata comparable as
cell populations.

Cohort: 109 independent donor keys (75 with both arms, 10 with one arm,
21 healthy-skin donors). Non-CD4 clones, diagnoses outside the MF/SS spectrum and repeat
biopsies of one patient were removed in 42 section 0b. Malignancy call `mal_cluster`.

Weighting: donor-balanced (every donor contributes equal weight to its arm) is the headline;
cell-weighted is printed beside it as `*_cells`. Pooled means are of log1p CP10K.

Effect sizes: `log2fc` (log2 ratio of pooled means, pseudocount 0.05), `d_det` (difference in the
fraction of cells detected), `cohen_d`, and `auroc` (probability a random malignant cell exceeds a
random comparator cell; 0.5 = no separation, and it is capped by detection for sparse genes).
CIs are a 1000-draw donor bootstrap. The headline p-value is van Elteren (`q_blocked`):
donor-level means ranked within study.

Stratified contrasts, per field: (a) malignant vs reactive within each stratum, (b) malignant
cells of one stratum vs the other, (c) the same for reactive cells, (d) difference-in-differences.
p-values come from permuting donor labels within study (50,000 permutations, `q_study`), with
a free permutation printed beside them (`q_free`). A field is tier 1 (tested), 2 (described) or 3
(counted) according to how many donors each side has in each arm.

Skin-specific: the lesion that was biopsied (patch / plaque / tumour / erythroderma) is a
stratification the blood cohort does not have, and it is confounded with stage and with the
malignant burden. Everything is transcript; skin carries no CITE-seq.

## §0b Cohort by arm pattern

**What:** how many independent donors have both CD4 arms, one, or are healthy.

**How:** an arm counts when it holds >= 20 cells.

**Full table on disk:** `tables/tgt48_cohort.csv`

```csv
pattern,HC,MF,MF/SS,SS,All
healthy,21,0,0,0,21
malignant only,0,2,0,1,3
no CD4 arm,1,1,0,1,3
paired,0,67,2,6,75
reactive only,0,5,0,2,7
All,22,75,2,10,109
```

## §1 Pool composition

**What:** who each pool is actually made of, under each weighting.

**How:** Kish effective donors (sum(w)^2/sum(w^2)); top_study_share_cells is the share of the cell-weighted pool held by its largest study.

**Full table on disk:** `tables/tgt48_pool_composition.csv`

```csv
arm,n_donors,n_cells,kish_n_eff_cells,kish_n_eff_donor,top_donor,top_donor_share_cells,top_study_share_cells,top_study
CD4_malignant,78,187748,33.4,78,B5__PT09,0.0555,0.476,li2024
CD4_reactive,82,147233,11.4,82,Li2024_atlas__CTCL7,0.261,0.622,li2024
CD4_HC,21,4724,12.4,21,D13__SC374_HTO10,0.166,0.256,chennareddy2025
```

## §2a Pooled shortlist

**What:** the genes that pass every pooled gate: clean in 42, log2FC > 0.5 with the bootstrap CI above zero, van Elteren q < 0.05, not a loss marker or lineage control.

**How:** donor-balanced pooled contrast, malignant vs reactive CD4.

**Full table on disk:** `tables/tgt48_pooled_all_genes.csv`

```csv
index,gene,cd_name,log2fc,log2fc_lo,log2fc_hi,pct_a,pct_b,d_det,auroc,auroc_lo,cohen_d,log2fc_cells,log2fc_paired,q_blocked,log2fc_vs_hc,auroc_vs_hc,on_42_shortlist
1720,PTH2R,,1.61,1.16,2.04,
... [18315 more chars truncated]
```
