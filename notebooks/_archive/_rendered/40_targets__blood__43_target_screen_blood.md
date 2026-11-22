# 40_targets/blood/43_target_screen_blood.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/blood/43_target_screen_blood.ipynb` and is never run._

# 43 — Surface-target screen, **blood**: cohort fix, artifact tests, malignant vs reactive CD4

The blood twin of 42, rebuilt. The first pass ranked **TPO, MBP, ADGRE1, PTH2R, NRROS, EVC2 and
FGFR1 above CCR4**. TPO and MBP are not surface proteins; ADGRE1 is eosinophil-restricted in
humans; NRROS is myeloid. And `frac_donors_up` came out *inverted* against plausibility —
MBP 0.962, EVC2 0.936, CCR4 0.744 — which is the signature of a technical effect, because
something that scales with arm composition repeats in every patient and real inter-patient
biology does not.

Two things were wrong with that pass, and they are not the same thing.

**1. The cohort, which was wrong and is now fixed.** 54 of the 131 donor keys are `geskin2026`
lane×HTO wells — ex vivo 2-day cultures with five drug arms and one vehicle arm per lane — so
one patient was counted six times. Five more keys are **healthy donors carried under
`disease == "SS"`**, which put their CD4 cells in the *reactive* arm of a tumour-vs-reactive
comparison. Two patients appeared under two keys. Correlated wells inflate `frac_donors_up`
directly, which is the statistic the whole complaint rests on. §0b removes them: **52
independent paired donors**, diagnosis from `clin_dx`.

**2. The contamination hypothesis, which was tested and did not survive.** Blood in SS is ~92%
malignant CD4, so malignant×monocyte doublets should outnumber reactive×monocyte doublets by
about that ratio, and every minor population should preferentially contaminate the malignant
arm. §2 puts six tests to that, against an **empirical null** made of transcripts a CD4 T cell
cannot make. The null does not fire: `LYZ`, `S100A8`, `CD14`, `VCAN`, `PPBP`, `HBB` and `MS4A1`
are flat or *down* in the malignant arm, the myeloid score is **lower** there than in the
reactive arm, and the disputed genes survive trimming, winsorising, dropping lineage-positive
cells, and depth matching. §2d adds the two artifacts that are not contamination — confinement
to one cluster, and the malignancy call being a cluster label while the comparison is between
those same clusters — and rebuilds the whole screen with the arms taken from the **TCR clone**
instead. The effects hold.

So §2 excludes nothing, and says so loudly. The disputed genes are reproducible across all ten
blood studies, absent from healthy CD4 as well as from reactive CD4, and spread evenly over the
clusters. The objection to them is a judgement about **gene identity and surface topology** —
biology — and it belongs in the surfaceome plausibility layer in §7, not in a statistical
exclusion.

| § | what it does |
|---|---|
| §0b | the cohort fix: 131 donor keys → the independent set, diagnosis from `clin_dx` |
| §1 | input, on the rebuilt QC tables |
| **§2** | **the artifact tests** — six against an empirical contamination null, plus the TCR-call control |
| §3 | malignant vs reactive CD4 on the surviving genes, with per-donor figures |
| §4–§6 | off-tumour, trafficking, escape |
| §7 | the ranked table, the control check, and the surfaceome plausibility layer |
| §8 | skin ∩ blood |

**Inputs.** `jobs/run_target_screen_qc.sh --tissue blood`, which supersedes
`run_target_screen.sh` for this compartment: true-library CP10K, robust location statistics, a
contamination-filtered recompute, depth matching, per-cell couplings and the lineage-control
genes. §2d additionally needs the same job with
`--call mal_tcr --no-cluster --out-dir data/targets/by_call_mal_tcr`.

**Moved out of this notebook.** Stratification and heterogeneity are now `44`; the CITE-seq
protein-vs-transcript analysis is now `45`.

> The object is a **10,000-gene HVG subset**, so a gene absent here is absent from the screen,
> which is not the same as unexpressed. `CD4` and `TNFRSF8` (CD30) are both gone.

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

# BOTH modules, before any path is read. `malignancy` defaults to skin at import, so with only
# CM.use_tissue("blood") the arms would be built from the SKIN parquets reindexed onto blood
# cell ids -- every call undefined, every arm empty, and the first symptom is an unrelated
# KeyError five cells later. (Observed 2026-10-24.) The assert below is the guard, not the
# comment.
CM.use_tissue("blood")
MAL.use_tissue("blood")
TQ.use_tissue("blood")           # which lineages contaminate THIS compartment: 42 sets skin,
                                 # and the module keeps whatever the last notebook asked for
assert MAL.CLUSTER_PQ.name.startswith("blood_"), MAL.CLUSTER_PQ
assert "LYZ" in TQ.BAND_GENES and "KRT14" not in TQ.BAND_GENES, TQ.BAND_GENES

TISSUE = "blood"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt43_"
MALIG, REACT, HC_CD4 = "CD4_malignant", "CD4_reactive", "CD4_HC"
MIN_CELLS, MIN_DONORS, DETECT_FLOOR = 20, 5, 0.10

ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX
TF.rc()
pd.set_option("display.width", 240)

print("NB_DIR =", NB_DIR)
print("tissue:", CM.TISSUE, "| mal_cluster source:", MAL.CLUSTER_PQ.name,
      "| headline:", MAL.HEADLINE)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
tissue: blood | mal_cluster source: blood_T_malignancy_cluster_v1.parquet | headline: mal_cluster
```

## §0b — The cohort, before any statistic

`donor` in the target tables is not a patient. Three corrections, all read from the curated
`clin_*` columns (`docs/CLINICAL_METADATA.md`), none of them guessed here:

- **ex vivo drug arms.** `geskin2026` is six 2-day culture wells per lane — REGN668/dupilumab,
  anti-IL13, anti-IL4, anti-IL13Rα1, anti-IL13Rα2 and a vehicle control. The five drug wells are
  dropped; the vehicle well stays and is flagged `is_cultured`, because dropping it would remove
  the study entirely and because a surface result should be shown with and without cultured
  material.
- **mislabelled healthy donors.** `B4__PF46–PF50` carry `disease == "SS"` in the atlas and
  `clin_dx == "HC"` in the curation. The screen builds `CD4_HC` from the atlas column, so their
  cells were sitting in `CD4_reactive` — the benign compartment *beside a tumour* — in a
  tumour-vs-reactive test. Reassigned here.
- **one patient under two keys.** `B1__P5` is the non-malignant sort of `B1__P4`;
  `H__SS2` and `D13__SZ22` are one person sampled at two centres.

Diagnosis comes from `clin_dx` throughout. The atlas `disease` column disagrees with it for
every herrera donor, all of ren2023 and two buus donors, and is not read again in this notebook.

`burden` — the malignant share of a donor's CD4 compartment, computed from the arm cell counts
themselves — is attached here. It is the one donor covariate with complete coverage, and §2 uses
it as the mechanistic test for contamination.

```python  # cell 3
# ============================================================================
# §0b  Cohort: which donor keys are independent patients
# ============================================================================
don = TC.blood_donor_table()
don.to_csv(ST.tab("blood_cohort"), index=True)

_raw = pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet")
long = TC.filter_long(_raw, don)
bur = TC.burden(long)
don = don.join(bur, how="left")

print(TC.cohort_report(don, long))
print("\nwrote", ST.tab("blood_cohort"))

print("\n" + "=" * 78)
print("dropped keys, every one with its reason:")
print(don.loc[~don["cohort_keep"], ["study", "dx", "clin_ex_vivo", "clin_arm", "person",
                                    "n_cells", "drop_reason"]]
      .sort_values(["drop_reason", "study"]).to_string())

print("\nhealthy donors moved out of the reactive arm (clin_dx == HC, atlas disease != HC):")
_moved = don[don["is_hc"] & don["cohort_keep"] & don["disease"].ne("HC")]
print(_moved[["study", "disease", "dx", "n_cells"]].to_string()
      if len(_moved) else "  none")
```

```
blood donor keys in the object: 131
  -  54 dropped: ex vivo drug-culture arm
  -   1 dropped: same person as B1__P4
  -   1 dropped: same person as D13__SZ22
  =  75 independent donor keys (73 people)

dx x study, independent keys:
study                borcherding2019  borcherding2023  buus2025  dorando2026  gaydosik2022  geskin2026  harro2023  herrera2021  ren2023  rindler2021_fi
dx                                                                                                                                                     
HC                                 0                0         0            0             3           5          0            1        3               0
MF                                 0                0         2            0             0           0          3            1        4               1
MF/SS                              0                0         0            0             1           1          0            0        0               0
MF_or_SS_unresolved                0                0         0            6             0           0          0            0        0               0
SS                                 1                5        15            0             2           5          4            5        7               0

ex vivo 2-day culture (vehicle arm kept, flagged): 11 keys
shared dominant TRB clone (buus PT05/06/18):       3 keys

donors with BOTH a malignant and a reactive arm: 52 (52 people)
dx
SS                     37
MF                      8
MF_or_SS_unresolved     5
MF/SS                   2

wrote /home/projects/nyosef/zvise/MF/tables/tgt43_blood_cohort.csv

==============================================================================
dropped keys, every one with its reason:
                          study     dx  clin_ex_vivo  clin_arm           person  n_cells               drop_reason
donor                                                                                                             
B4__PF46_HTO10       geskin2026     HC  culture_drug  aIL13Ra2  pittsburgh:PF46     3336  ex vivo drug-culture arm
B4__PF46_HTO6        geskin2026     HC  culture_drug   REGN668  pittsburgh:PF46     3049  ex vivo drug-culture arm
B4__PF46_HTO7        geskin2026     HC  culture_drug     aIL13  pittsburgh:PF46     3185  ex vivo drug-culture arm
B4__PF46_HTO8        geskin2026     HC  culture_drug      aIL4  pittsburgh:PF46     4614  ex vivo drug-culture arm
B4__PF46_HTO9        geskin2026     HC  culture_drug  aIL13Ra1  pittsburgh:PF46     2625  ex vivo drug-culture arm
B4__PF47_HTO2        geskin2026     HC  culture_drug   REGN668  pittsburgh:PF47     3789  ex vivo drug-culture arm
B4__PF47_HTO3        geskin2026     HC  culture_drug     aIL13  pittsburgh:PF47     2739  ex vivo drug-culture arm
B4__PF47_HTO4        geskin2026     HC  culture_drug      aIL4  pittsburgh:PF47     2653  ex vivo drug-culture arm
B4__PF47_HTO5        geskin2026     HC  culture_drug  aIL13Ra1  pittsburgh:PF47     1730  ex vivo drug-culture arm
B4__PF47_HTO6        geskin2026     HC  culture_drug  aIL13Ra2  pittsburgh:PF47     3240  ex vivo drug-culture arm
B4__PF48_HTO2        geskin2026     HC  culture_drug   REGN668  pittsburgh:PF48     2447  ex vivo drug-culture arm
B4__PF48_HTO3        geskin2026     HC  culture_drug     aIL13  pittsburgh:PF48     1707  ex vivo drug-culture arm
B4__PF48_HTO4        geskin2026     HC  culture_drug      aIL4  pittsburgh:PF48     1855  ex vivo drug-culture arm
B4__PF48_HTO5        geskin2026     HC  culture_drug  aIL13Ra1  pittsburgh:PF48     1636  ex vivo drug-culture arm
B4__PF48_HTO6        geskin2026     HC  culture_drug  aIL13Ra2  pittsburgh:PF48     1717  ex vivo drug-culture arm
B4__PF49_HTO2        geskin2026     HC  culture_drug   REGN668  pittsburgh:PF49     5807  ex vivo drug-culture arm
B4__PF49_HTO3        geskin2026     HC  culture_drug     aIL13  pittsburgh:PF49     1853  ex vivo drug-culture arm
B4__PF49_HTO4        geskin2026     HC 
... [4858 more chars truncated]
```

## §1 — Input, and the four things that make blood different

1. **The object is a 10,000-gene HVG subset**, not the full 42,347. A rare-but-specific antigen
   is exactly what HVG selection discards, so a gene missing here is missing from the screen,
   which is not the same as unexpressed. `CD4` and `TNFRSF8` (CD30) are both gone, so the
   control check runs on three of four positive controls. 41 §3b lists the rest.
2. **Blood is ~92% malignant CD4 in SS.** The reactive arm barely exists in some donors — which
   is why the paired test runs on fewer donors than in skin, and why that abundance ratio is
   also the lever behind the contamination artifact §2 goes after.
3. **The CD8 arm is a CD8+NK mixture.** `nb31` labelled on `CD8A`, which NK cells also express
   as a CD8αα homodimer, so cytotoxic NK clusters came through as CD8 — the object holds only
   606 NK cells in 667,584, which is not a credible PBMC NK fraction. `32b` corrects the 12
   clusters the malignancy review actually looked at (`blood_T_celltype_corrections_v1.csv`,
   overlaid by `CM.read_obs`) and reports ~100 more with the same signature that are **not**
   corrected. The malignant-vs-reactive CD4 axis is unaffected; any statement about the CD8 or
   NK arm in §4 is a statement about a mixture. The fix belongs in `nb31`.
4. **The tables are the QC build.** `mean_all` is now log1p CP10K on the **true** library
   (`obs.total_counts`, median 4,127), not on the sum over the 10,000 HVG columns (median
   2,186). The HVG-capture ratio differs between a clean T cell and a myeloid doublet, which is
   the confound under test; `mean_all_hvg` keeps the old quantity so the change is auditable.

```python  # cell 5
# ============================================================================
# §1  Load the rest of the QC build
# ============================================================================
meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)
man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
cl_path = TARGETS / f"{TISSUE}_target_cluster.parquet"
cluster_long = TC.filter_long(pd.read_parquet(cl_path), don) if cl_path.exists() else None
qc_path = TARGETS / f"{TISSUE}_target_qc_cell.parquet"
qc_cell = pd.read_parquet(qc_path) if qc_path.exists() else None

assert man.get("builder") == "run_target_screen_qc.py", (
    "these tables came from the v1 builder, which has no robust statistics, no clean recompute "
    "and no lineage controls -- run jobs/run_target_screen_qc.sh --tissue blood first. "
    f"manifest says: {man.get('builder', 'run_target_screen.py')}")

print(f"source {Path(man['source']).name} | call {man['malignancy_call']} | "
      f"{man['n_genes_in_object']:,} genes in object -> "
      f"{man['gene_selection'].get('n_total', man['gene_selection'].get('n_screened')):,} "
      f"screened (+{man['gene_selection'].get('n_lineage_added', 0)} lineage controls, "
      f"+{man['gene_selection'].get('n_traffic_added', 0)} trafficking)")
if man["n_genes_in_object"] < 20000:
    print("*** HVG subset: absent is NOT unexpressed. Quote this next to every blood result. ***")
print("library:", man.get("library"))
print("clean rule:", json.dumps(man.get("clean_rule", {}), default=str))
print(f"depth matching: {man.get('ds_depth'):,} UMI on {man.get('frac_cells_ds', 0):.1%} "
      "of cells")

arms = (meta.groupby("arm").agg(donors=("donor", "nunique"), cells=("n_cells", "sum"),
                                studies=("study", "nunique"))
        .sort_values("cells", ascending=False))
print("\narms, independent cohort:")
print(arms.to_string())

paired = meta[meta["arm"].isin([MALIG, REACT])].groupby("donor")["arm"].nunique().eq(2)
PAIRED_DONORS = sorted(paired.index[paired])
print(f"\ndonors with both arms: {len(PAIRED_DONORS)}")
DONOR_STUDY = dict(zip(meta["donor"], meta["study"]))

print("\ncontamination load per arm (median over donors) -- the asymmetry the artifact needs:")
print(meta.groupby("arm")[["frac_lineage_pos", "mean_myeloid_score", "median_total_counts",
                           "median_doublet_score"]].median().round(4).to_string())
```

```
source blood_T_annotated.h5ad | call mal_cluster | 10,000 genes in object -> 1,098 screened (+14 lineage controls, +54 trafficking)
*** HVG subset: absent is NOT unexpressed. Quote this next to every blood result. ***
library: obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg
clean rule: {"strict_lineage_markers": ["CD14", "CD68", "CD79A", "FCN1", "HBB", "LYZ", "MNDA", "MS4A1", "PF4", "PPBP", "S100A8", "S100A9", "VCAN"], "strict_min_counts": 2, "strict_min_genes": 2, "frac_lineage_pos": 0.16322739909884, "doublet_quantile_per_study": 0.9, "frac_cells_clean": 0.7536190202281661}
depth matching: 2,000 UMI on 86.0% of cells

arms, independent cohort:
               donors   cells  studies
arm                                   
CD4_malignant      58  295656       10
CD4_reactive       57  111125       10
CD8                72   70525       10
CD4_HC             12   40400        4
NK                  6     756        3

donors with both arms: 52

contamination load per arm (median over donors) -- the asymmetry the artifact needs:
               frac_lineage_pos  mean_myeloid_score  median_total_counts  median_doublet_score
arm                                                                                           
CD4_HC                   0.0845              0.0845              4016.75                0.0610
CD4_malignant            0.0177              0.0140              4156.50                0.0702
CD4_reactive             0.0114              0.0333              4191.50                0.0420
CD8                      0.0189              0.1044              3682.50                0.0360
NK                       0.0750              0.4218              4148.75                0.0274
```

## §2 — Is the ranking contamination? A test that is allowed to fail

The charge against the first ranking is specific, so the test can be too.

**The mechanism.** Blood in SS is ~92% malignant CD4. Malignant×monocyte doublets therefore
outnumber reactive×monocyte doublets by roughly that ratio, so every minor population in the
sample preferentially contaminates the malignant arm — in every donor, which is why a technical
effect of this kind reproduces *better* than real inter-patient biology and why `frac_donors_up`
came out inverted against plausibility. Ambient RNA is shared by both arms and cannot do this.

**Six tests**, each with its own failure mode:

| test | what fires it |
|---|---|
| `fails_trim` | the effect collapses when the top 1% of cells is trimmed — carried by a handful of cells, not by the population |
| `fails_concentration` | more of the arm's counts sit in its top 1% of cells than for other genes *detected at the same rate* |
| `fails_lineage` | per-cell expression tracks the myeloid module score **inside** the malignant arm (the test is one module here and three in skin) |
| `fails_umi` | it tracks per-cell library size, or detection is confined to the top-UMI decile, beyond what its prevalence implies |
| `fails_clean` | it does not survive dropping lineage-positive and high-doublet-score cells |
| `fails_burden` | its per-donor effect grows with that donor's malignant:reactive abundance ratio |

Two failures ⇒ `artifact`, one ⇒ `suspect`. Only up-in-malignant genes can be flagged;
contamination entering the malignant arm cannot push a gene down.

**The null, and the escape hatch.** `LYZ`, `S100A8`, `S100A9`, `CD14`, `VCAN`, `FCN1`, `CST3`,
`LST1`, `AIF1`, `TYROBP`, `FCER1G`, `CD68`, `MNDA`, `PPBP`, `PF4`, `HBB`, `MS4A1`, `CD79A` are
not CD4 T-cell transcripts, so whatever they score *is* what contamination scores in this
cohort, and every threshold is read off them rather than chosen. That also makes the section
refutable: **if those genes are not up in the malignant arm, the mechanism is not operating
here, and §2b excludes nothing.** The calibration verdict prints before any flag is assigned —
read it first, because it decides whether the rest of the section means anything.

**Why the statistics are prevalence-matched.** `top1_share`, the top-UMI-decile odds ratio and
the myeloid correlation are all functions of how often a gene is detected at all: a gene seen
in two cells necessarily concentrates, and a gene seen in 80% of cells necessarily rises with
depth because detection saturates. Compared raw, the band would condemn CD7 and acquit a gene
seen once. Each is taken as a residual against genes of the same detection rate.

**§2d adds the two artifacts that are not contamination** — the malignancy call being a cluster
label while the comparison is between those same clusters, and the effect living in one or two
clusters rather than in the clone. The control for the first is the whole screen rebuilt with
the arms taken from the TCR clone, which is measured independently of the transcriptome.

```python  # cell 7
# ============================================================================
# §2a  The six artifact statistics, and the empirical contamination null
# ============================================================================
# The same paired test, four times, on four location estimators. If an effect is carried by a
# handful of cells it survives only the first.
sel = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
sel_trim = TS.paired_selectivity(long, MALIG, REACT, value="trim1_mean", min_donors=MIN_DONORS)
sel_wins = TS.paired_selectivity(long, MALIG, REACT, value="wins90_mean", min_donors=MIN_DONORS)
sel_clean = TS.paired_selectivity(long, MALIG, REACT, value="mean_all_clean",
                                  min_donors=MIN_DONORS)
sel_ds = TS.paired_selectivity(long, MALIG, REACT, value="mean_all_ds", min_donors=MIN_DONORS)
blk = TS.blocked_selectivity(long, MALIG, REACT, block=DONOR_STUDY, value="mean_all")
sel = sel.merge(blk, on="gene", how="left")

conc = TQ.concentration(long, MALIG)                       # top1/top5 share, detection
brn = TQ.burden_scaling(long, bur, MALIG, REACT)           # does the effect scale with burden
confine = (TQ.cluster_confinement(cluster_long, MALIG)
           if cluster_long is not None else None)

ev = TQ.assemble(sel, sel_trim, sel_clean, qc_cell, brn, conc, confine)
for name, s in (("wins", sel_wins), ("ds", sel_ds)):
    ev = ev.merge(s[["gene", "log2fc_paired"]].rename(
        columns={"log2fc_paired": f"log2fc_{name}"}), on="gene", how="left")
    ev[f"retained_{name}"] = TQ._ratio(ev[f"log2fc_{name}"], ev["log2fc_paired"])
ev = TQ.prevalence_residual(ev)

print(f"{len(sel):,} genes tested paired on {len(PAIRED_DONORS)} donors | "
      f"{int((sel['q'] < 0.05).sum()):,} at q < 0.05")

print("\n" + "=" * 78)
print("THE NULL: what pure contamination scores in this cohort")
print("=" * 78)
ok, msg = TQ.band_is_calibrated(ev, None)
print(msg)
band = TQ.contamination_band(ev)
print(f"\nlineage controls found: {band.attrs['n_genes']} "
      f"({', '.join(band.attrs['genes_found'])})")
print("\nper-gene, in full:")
_bcols = [c for c in ("gene", "log2fc_paired", "frac_donors_up", "median_pct_expr",
                      "retained_trim", "retained_clean", "top1_share_excess",
                      "r_myeloid_excess", "or_topdecile_excess", "rho_burden")
          if c in ev.columns]
print(ev[ev["gene"].isin(TQ.BAND_GENES)][_bcols]
      .sort_values("log2fc_paired", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print("\nquantiles of the band (thresholds are read from the p90 row):")
print(band.round(4).to_string())
```

```
1,098 genes tested paired on 52 donors | 478 at q < 0.05

==============================================================================
THE NULL: what pure contamination scores in this cohort
==============================================================================
18 lineage controls: median log2FC +0.00, median frac_donors_up 0.39 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.

lineage controls found: 18 (AIF1, CD14, CD68, CD79A, CST3, FCER1G, FCN1, HBB, LST1, LYZ, MNDA, MS4A1, PF4, PPBP, S100A8, S100A9, TYROBP, VCAN)

per-gene, in full:
  gene  log2fc_paired  frac_donors_up  median_pct_expr  retained_trim  retained_clean  top1_share_excess  r_myeloid_excess  or_topdecile_excess  rho_burden
 CD79A        0.04428          0.5962          0.01306              0           0.925           -0.06888         0.0008915              0.07311    -0.08802
  MNDA        0.02802          0.6538           0.0011              0          0.2981                  0            0.2185              -0.2826      -0.167
  CST3        0.01336          0.5769         0.007404              0          0.5111            -0.1437            0.3645              0.08985     -0.1351
  CD68       0.008893          0.5192         0.004255              0          0.7567                  0            0.2207                    0     -0.2198
   LYZ              0          0.3846        0.0005541            NaN             NaN                  0            0.2268              0.04645     0.05677
S100A8              0          0.4231         0.000361            NaN             NaN                  0            0.2407              -0.3631     0.06038
  VCAN              0          0.2308                0            NaN             NaN                  0           0.06143                 -0.2      0.3374
S100A9              0          0.3269         0.005198            NaN             NaN            -0.1568            0.2337              -0.2506    -0.01738
  PPBP              0          0.4615         0.000497            NaN             NaN                  0          0.002686              -0.2133      0.1319
  CD14              0          0.3269        0.0004134            NaN             NaN                  0           0.08893              -0.1527      0.3742
   PF4              0          0.4231         0.000172            NaN             NaN                  0          0.007363              -0.3684      0.2591
  FCN1              0          0.2308                0            NaN             NaN                  0           0.09358             0.002575      0.3364
   HBB              0          0.4808         0.006183            NaN             NaN                  0            -0.018              0.07561     -0.1841
 MS4A1      -0.005045          0.3846        0.0006448             -0              -0                  0          0.004488              -0.3508      0.3378
TYROBP        -0.1224          0.1538        0.0003326             -0          0.5494                  0            0.2336              -0.1902     -0.1315
  LST1        -0.1343          0.4038           0.0517          1.239           1.322           -0.03499             0.567             -0.05657    -0.05976
FCER1G        -0.2052          0.1346         0.001068             -0          0.9711                  0            0.1928              -0.3624     -0.1823
  AIF1         -1.613         0.03846         0.001394         0.9909           1.068                  0            0.1799               0.3253      0.2721

quantiles of the band (thresholds are read from the p90 row):
       log2fc_paired  frac_donors_up  retained_trim  retained_clean  top1_share  top1_share_excess  top5_share  r_myeloid  r_myeloid_excess   r_umi  r_umi_excess   r_dbl  or_topdecile  or_topdecile_excess  rho_burden  top_cluster_share
p50           0.0000          0.3942         0.0000 
... [655 more chars truncated]
```

```python  # cell 8
# ============================================================================
# §2b  The verdict -- which is only allowed to exclude anything if the null is calibrated
# ============================================================================
out = TQ.artifact_evidence(ev, band)
if not ok:
    # The lineage controls are the whole basis for calling a gene contamination. If they are not
    # up in the malignant arm, the mechanism is absent from this cohort and the six tests are
    # measuring something else. Nothing is excluded; the evidence is still printed, because the
    # columns are informative about every gene whether or not contamination is the explanation.
    out["artifact_flag"] = "clean"
    out["fail_reasons"] = "- (null not calibrated; no exclusion applied)"
    print("*" * 78)
    print("NO GENE IS EXCLUDED ON CONTAMINATION GROUNDS.")
    print(msg)
    print("*" * 78)

ev = out
print()
print(TQ.verdict_text(ev))

print("\n" + "=" * 78)
print("the six tests, per gene, for every gene with a positive effect at q < 0.05")
print("=" * 78)
_full = ev[(ev["q"] < 0.05) & (ev["log2fc_paired"] > 0)].sort_values("log2fc_paired",
                                                                    ascending=False)
_fc = [c for c in ("gene", "log2fc_paired", "frac_donors_up", "median_pct_expr",
                   "retained_trim", "retained_wins", "retained_clean", "retained_ds",
                   "top1_share_excess", "r_myeloid_excess", "r_umi_excess",
                   "or_topdecile_excess", "rho_burden", "top_cluster_share",
                   "n_fails", "fail_reasons") if c in ev.columns]
print(_full[_fc].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

ev.to_csv(ST.tab("artifact_evidence"), index=False)
band.to_csv(ST.tab("contamination_band"))
print("\nwrote", ST.tab("artifact_evidence"), "and", ST.tab("contamination_band"))
print("\nColumn key:  retained_* = effect kept after trimming the top 1% of cells / "
      "winsorising at the 90th percentile / dropping lineage-positive and high-doublet cells / "
      "binomially thinning to a common depth. 1.0 = unchanged, 0 = gone. *_excess = the "
      "statistic minus the cohort median for genes detected at the same rate. rho_burden = "
      "Spearman of the per-donor effect against that donor's malignant:reactive ratio.")
```

```
******************************************************************************
NO GENE IS EXCLUDED ON CONTAMINATION GROUNDS.
18 lineage controls: median log2FC +0.00, median frac_donors_up 0.39 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.
******************************************************************************

artifact_flag
clean    1098

the genes that prompted this, by name:
  gene  log2fc_paired  frac_donors_up  median_pct_expr  retained_trim  retained_clean  top1_share_excess  r_myeloid_excess  or_topdecile_excess  rho_burden  n_fails artifact_flag                                  fail_reasons
   TPO       1.851566        0.961538         0.122762       1.016870        0.987509           0.014844          0.000668            -0.375537    0.011270        1         clean - (null not calibrated; no exclusion applied)
 NRROS       1.200266        0.903846         0.156997       0.945380        1.018656          -0.029320          0.010041             0.103532   -0.075216        1         clean - (null not calibrated; no exclusion applied)
ADGRE1       1.151332        0.865385         0.095970       1.004172        1.041452          -0.025309          0.001156             0.142307   -0.038931        1         clean - (null not calibrated; no exclusion applied)
 FGFR1       1.147858        0.884615         0.096753       0.992124        1.105026           0.002837          0.005126            -0.012294   -0.082216        1         clean - (null not calibrated; no exclusion applied)
 PTH2R       0.970591        0.865385         0.040061       0.775923        0.813171           0.050630          0.010856            -0.148376   -0.131660        1         clean - (null not calibrated; no exclusion applied)
   MBP       0.842599        0.942308         0.727430       1.030848        1.004900           0.000746         -0.013604            -0.462787   -0.208657        1         clean - (null not calibrated; no exclusion applied)
  EVC2       0.841262        0.961538         0.050567       0.914620        0.982208          -0.010708         -0.002616            -0.218105   -0.151114        0         clean - (null not calibrated; no exclusion applied)

excluded from the headline (0 genes), every one with its reason:
Empty DataFrame
Columns: [gene, log2fc_paired, frac_donors_up, median_pct_expr, retained_trim, retained_clean, top1_share_excess, r_myeloid_excess, or_topdecile_excess, rho_burden, n_fails, artifact_flag, fail_reasons]
Index: []

==============================================================================
the six tests, per gene, for every gene with a positive effect at q < 0.05
==============================================================================
      gene  log2fc_paired  frac_donors_up  median_pct_expr  retained_trim  retained_wins  retained_clean  retained_ds  top1_share_excess  r_myeloid_excess  r_umi_excess  or_topdecile_excess  rho_burden  top_cluster_share  n_fails                                  fail_reasons
   KIR3DL2           2.79          0.9038           0.3672          1.046          1.247           1.074        1.033          -0.007758          -0.01091      -0.04351              0.06515     -0.1454           0.008361        0 - (null not calibrated; no exclusion applied)
      CD70          1.927          0.8846           0.2415          1.028          1.347          0.9889       0.8851            0.03709           0.02031       0.07066               0.4204     -0.2864            0.01198        2 - (null not calibrated; no exclusion applied)
       TPO          1.852          0.9615           0.1228          1.017         0.9733          0.9875       0.9401            0.01484         0.0006679      -0.02759              -0.3755     0.01127            0.01114        1 - (null not calibrated; no exclusion applied)
     TIGIT          1.653          0.8269    
... [71715 more chars truncated]
```

```python  # cell 9
# ============================================================================
# §2c FX  The null, and where the disputed genes actually sit
# ============================================================================
SETS = {"lineage controls\n(pure contamination)": TQ.BAND_GENES,
        "all screened genes": sorted(set(ev["gene"]) - set(TQ.BAND_GENES)),
        "disputed genes": TQ.SUSPECTS,
        "validated antigens": ST.POSITIVE_CONTROLS}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4),
                         gridspec_kw={"width_ratios": [1.05, 1.0]})

ax = axes[0]
rng = np.random.default_rng(0)
for i, (lab, gs) in enumerate(SETS.items()):
    v = ev.loc[ev["gene"].isin(gs), "log2fc_paired"].dropna()
    col = [TF.DOWN, "#bdbdbd", "#2e86c1", TF.UP][i]
    ax.scatter(np.full(len(v), i) + rng.normal(0, .07, len(v)), v, s=7 if i == 1 else 18,
               color=col, alpha=.35 if i == 1 else .9, linewidths=0)
    if len(v):
        ax.hlines(np.median(v), i - .25, i + .25, color="#111111", lw=1.4, zorder=5)
ax.axhline(0, lw=.5, c="k")
ax.set_xticks(range(len(SETS)))
ax.set_xticklabels(SETS, fontsize=6)
ax.set_ylabel("log2FC, malignant vs reactive CD4")
ax.set_title("Contamination null vs candidates", loc="left", fontsize=7.5)
TF._clean(ax)

# Right: the three-arm view. Contamination that rides into the malignant arm has no reason to
# spare healthy CD4 -- healthy blood carries monocytes too. A transcript that is absent in
# healthy and reactive CD4 and present in the clone is tumour-restricted, whatever one thinks
# of its gene name.
ax = axes[1]
show = TQ.SUSPECTS + ST.POSITIVE_CONTROLS + ["CD7"]
det = (long[long["gene"].isin(show) & long["arm"].isin([HC_CD4, REACT, MALIG])]
       .pivot_table(index="gene", columns="arm", values="pct_expr", aggfunc="median")
       .reindex([g for g in show if g in set(long["gene"])])
       .sort_values(MALIG))
for j, (arm, col, mk) in enumerate(((HC_CD4, TF.COMPARATOR, "o"), (REACT, TF.REACT, "o"),
                                    (MALIG, TF.MALIG, "o"))):
    if arm in det:
        ax.scatter(det[arm], np.arange(len(det)) + (j - 1) * 0.2, s=22, color=col,
                   marker=mk, linewidths=0, label=arm.replace("CD4_", ""))
for y, (_, r_) in enumerate(det.iterrows()):
    ax.plot([r_.min(), r_.max()], [y, y], color="#dddddd", lw=0.8, zorder=0)
ax.set_yticks(range(len(det)))
ax.set_yticklabels([TF.it(g) for g in det.index], fontsize=6.5)
ax.set_xlabel("median fraction of a donor's cells detected")
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_xlim(left=0)
ax.set_ylim(-0.7, len(det) - 0.3)
ax.legend(fontsize=5.5, frameon=False, loc="upper left", title="CD4 arm",
          title_fontsize=5.5)
ax.set_title("Detection across CD4 arms", loc="left", fontsize=7.5)
TF._clean(ax)

fig.tight_layout()
TF.save(fig, "FX_artifact_null", caption_text=(
    "Left: paired log2FC by gene set. Right: median detection per CD4 arm, symlog x."))
TF.save_source(det.reset_index(), "FX_three_arm_detection_source")
TF.logfig("FX_artifact_null", "contamination null vs candidates; three-arm detection",
          len(ev))

print("median detection by arm, the genes in question (and the controls):")
print(det.round(4).to_string())
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FX_artifact_null.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FX_artifact_null.pdf

<Figure size 920x340 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_FX_three_arm_detection_source.csv
median detection by arm, the genes in question (and the controls):
arm      CD4_HC  CD4_malignant  CD4_reactive
gene                                        
IL2RA    0.0747         0.0383        0.0636
PTH2R    0.0003         0.0401        0.0009
EVC2     0.0038         0.0506        0.0064
ADGRE1   0.0278         0.0960        0.0237
FGFR1    0.0115         0.0968        0.0166
CD7      0.6655         0.1011        0.6474
TPO      0.0014         0.1228        0.0052
NRROS    0.0545         0.1570        0.0478
CCR4     0.0700         0.2099        0.0989
KIR3DL2  0.0076         0.3672        0.0105
MBP      0.4463         0.7274        0.4486
```

```python  # cell 10
# ============================================================================
# §2d  Two more ways the effect could be an artifact, and neither is contamination
# ============================================================================
# (1) CLUSTER CIRCULARITY. The blood malignancy call is `mal_cluster` -- a *cluster* label. The
#     arms are therefore groups of Leiden clusters, and the comparison is between the same
#     clusters whose expression defined them. A gene confined to one or two malignant clusters
#     would be a cluster marker rediscovered as an antigen.
if confine is not None:
    print("share of the malignant arm's signal held by its single strongest cluster "
          f"({int(confine['n_clusters'].median())} clusters):")
    _q = confine["top_cluster_share"].describe(percentiles=[.5, .9, .99]).round(4)
    print("  all screened genes:", _q.to_dict())
    print("\n  the genes in question:")
    print(confine[confine["gene"].isin(TQ.SUSPECTS + ST.POSITIVE_CONTROLS)]
          .sort_values("top_cluster_share", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n  A gene spread evenly over N clusters scores 1/N. Confinement would show as a "
          "share far above that floor.")

# (2) THE CALL ITSELF. `mal_tcr` assigns malignancy from the expanded TCR clone, which is
#     measured independently of the transcriptome. If an effect holds under that call it cannot
#     be an artifact of how the clusters were drawn. Built by:
#         bash jobs/run_target_screen_qc.sh --tissue blood --call mal_tcr --no-cluster \
#              --out-dir data/targets/by_call_mal_tcr
TCR_DIR = TARGETS / "by_call_mal_tcr"
_keep = ["gene", "log2fc_paired", "frac_donors_up", "n_donors_paired"]
if (TCR_DIR / f"{TISSUE}_target_long.parquet").exists():
    long_tcr = TC.filter_long(pd.read_parquet(TCR_DIR / f"{TISSUE}_target_long.parquet"), don)
    sel_tcr = TS.paired_selectivity(long_tcr, MALIG, REACT, value="mean_all",
                                    min_donors=MIN_DONORS)
    both_calls = sel[_keep].merge(sel_tcr[_keep], on="gene", suffixes=("_cluster", "_tcr"))
    _r = stats.spearmanr(both_calls["log2fc_paired_cluster"], both_calls["log2fc_paired_tcr"])
    print("\n" + "=" * 78)
    print(f"the same screen with the arms built from the TCR clone instead of the cluster: "
          f"rho = {_r.statistic:.2f} over {len(both_calls):,} genes "
          f"({both_calls['n_donors_paired_tcr'].median():.0f} paired donors vs "
          f"{both_calls['n_donors_paired_cluster'].median():.0f})")
    print("=" * 78)
    print(both_calls[both_calls["gene"].isin(TQ.SUSPECTS + ST.POSITIVE_CONTROLS
                                             + ST.LOSS_MARKERS)]
          .sort_values("log2fc_paired_cluster", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    both_calls.to_csv(ST.tab("call_sensitivity"), index=False)
    print("\nAn effect that is an artifact of the clustering cannot survive a call made from "
          "the TCR. One that does survive is a property of the clone.")
    print("wrote", ST.tab("call_sensitivity"))
else:
    print(f"\n{TCR_DIR.name} not built -- the cluster-circularity control is MISSING. "
          "Run: bash jobs/run_target_screen_qc.sh --tissue blood --call mal_tcr --no-cluster "
          f"--out-dir {TCR_DIR}")
```

```
share of the malignant arm's signal held by its single strongest cluster (384 clusters):
  all screened genes: {'count': 1085.0, 'mean': 0.1697, 'std': 0.2005, 'min': 0.0032, '50%': 0.0956, '90%': 0.4496, '99%': 1.0, 'max': 1.0}

  the genes in question:
   gene  top_cluster_share  top2_cluster_share  n_clusters
  PTH2R            0.05251             0.09531         384
  IL2RA            0.04199             0.08221         384
   EVC2            0.02474             0.04244         384
  FGFR1             0.0146             0.02875         384
  NRROS            0.01321              0.0252         384
   CCR4            0.01299             0.02565         384
 ADGRE1            0.01213             0.02419         384
    TPO            0.01114             0.02135         384
KIR3DL2           0.008361              0.0167         384
    MBP           0.006322             0.01199         384

  A gene spread evenly over N clusters scores 1/N. Confinement would show as a share far above that floor.


==============================================================================
the same screen with the arms built from the TCR clone instead of the cluster: rho = 0.93 over 1,098 genes (53 paired donors vs 52)
==============================================================================
   gene  log2fc_paired_cluster  frac_donors_up_cluster  n_donors_paired_cluster  log2fc_paired_tcr  frac_donors_up_tcr  n_donors_paired_tcr
KIR3DL2                   2.79                   0.904                       52               2.06               0.906                   53
    TPO                   1.85                   0.962                       52               1.22               0.925                   53
  NRROS                    1.2                   0.904                       52              0.811               0.887                   53
 ADGRE1                   1.15                   0.865                       52              0.801               0.887                   53
  FGFR1                   1.15                   0.885                       52              0.766               0.849                   53
   CCR4                   1.13                   0.769                       52              0.803               0.717                   53
  PTH2R                  0.971                   0.865                       52              0.583               0.887                   53
    MBP                  0.843                   0.942                       52              0.673               0.925                   53
   EVC2                  0.841                   0.962                       52               0.57               0.887                   53
   CD27                  0.528                   0.827                       52              0.588               0.868                   53
    CD5                -0.0125                   0.481                       52             0.0553               0.509                   53
  IL2RA                  -0.19                   0.385                       52             -0.259               0.283                   53
    CD6                 -0.459                    0.25                       52             -0.329               0.264                   53
    CD2                 -0.504                   0.192                       52             -0.398                0.17                   53
   DPP4                  -1.91                  0.0192                       52              -1.69              0.0189                   53
    CD7                  -1.99                   0.115                       52              -1.84               0.151                   53

An effect that is an artifact of the clustering cannot survive a call made from the TCR. One that does survive is a property of the clone.
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_call_sensitivity.csv
```

## §3 — Malignant vs reactive CD4, on what survived

The comparison the whole screen exists for, now on an independent cohort and on genes that
passed §2. Five figures, in the style of `28_tme_degradome_mc`, because a cohort median hides
the thing that decides whether a target is usable: **what it does inside each patient.**

| figure | what it shows |
|---|---|
| **FA** | the leading antigens, one grey line per donor joining its reactive and malignant means |
| **FB** | the antigens whose answer is already known — read this one first |
| **FC** | the ranked candidate list as a dumbbell, with prevalence and off-tumour beside each row |
| **FD** | the donor × gene effect matrix, donors ordered by malignant burden |
| **FE** | the same comparison on *detection rate*, which is what a CAR is dosed against |

The star above each gene is the BH-adjusted paired Wilcoxon; `n↑/n↓` underneath it is the donor
vote, and the two percentages under each gene name are the median share of cells detected in
the malignant and reactive arms. Titles say what the panel is; the finding is in the table
printed under it.

FD is the figure to argue with. Donors run left to right by malignant share of their CD4
compartment, so a row that darkens towards the right is a row whose effect tracks arm
composition — the pattern §2 tests for formally with `fails_burden`, shown here so it can also
be seen.

```python  # cell 12
# ============================================================================
# §3  Malignant vs reactive CD4, on the genes that survived §2
# ============================================================================
prev = TS.prevalence(long, MALIG, detect_floor=DETECT_FLOOR)
off = TS.off_tumour(long, target_arms=[MALIG], healthy_t_arms=[HC_CD4], value="pct_expr")
off = off.rename(columns={"worst_offtumour": "worst_T_arm_pct",
                          "worst_offtumour_arm": "worst_T_arm"})
esc = TS.escape_risk(long, MALIG, cluster_long=cluster_long)

core = (sel.merge(prev, on="gene", how="left")
           .merge(off, on="gene", how="left")
           .merge(ev[["gene", "artifact_flag", "n_fails", "fail_reasons",
                      "retained_trim", "retained_clean", "retained_ds", "rho_burden"]],
                  on="gene", how="left"))
core = core.merge(ST.annotate(core["gene"])[["gene", "cd_name", "panel_class", "stage"]],
                  on="gene", how="left")

GATES = ((core["artifact_flag"].eq("clean")) & (core["q"] < 0.05)
         & (core["log2fc_paired"] > 0.5) & (core["frac_donors_positive"] >= 0.5)
         & (~core["gene"].isin(ST.LOSS_MARKERS)) & (~core["gene"].isin(TQ.BAND_GENES)))
up = core[GATES].sort_values("log2fc_paired", ascending=False)
SHORT_GENES = up["gene"].tolist()[:30]
TOP_GENES = SHORT_GENES[:12]

CAPTION = (f"Paired Wilcoxon within donor, BH. {len(PAIRED_DONORS)} donors, "
           f"{meta['study'].nunique()} studies. log1p CP10K.")
LINE_CAPTION = CAPTION + " Lines: red = up in malignant, blue = down."

print(f"{len(up)} genes pass every gate "
      "(clean, q<0.05, log2FC>0.5, detected in >=50% of donors):\n")
_c = ["gene", "log2fc_paired", "frac_donors_up", "frac_donors_positive", "median_pct_expr",
      "q", "z_blocked", "retained_trim", "retained_clean", "retained_ds", "rho_burden",
      "worst_T_arm", "cd_name", "stage"]
print(up[[c for c in _c if c in up]].to_string(index=False,
                                               float_format=lambda v: f"{v:.4g}"))

print("\nloss markers -- CD7 and DPP4/CD26 loss is the Sezary immunophenotype, and is the "
      "sanity check that the cleaning did not flatten real biology:")
print(core[core["gene"].isin(ST.LOSS_MARKERS)][["gene", "log2fc_paired", "frac_donors_up",
                                                "q", "artifact_flag"]]
      .sort_values("log2fc_paired").to_string(index=False,
                                              float_format=lambda v: f"{v:.4g}"))
```

```
41 genes pass every gate (clean, q<0.05, log2FC>0.5, detected in >=50% of donors):

    gene  log2fc_paired  frac_donors_up  frac_donors_positive  median_pct_expr         q  z_blocked  retained_trim  retained_clean  retained_ds  rho_burden  worst_T_arm       cd_name      stage
 KIR3DL2           2.79          0.9038                0.7759           0.3672 8.094e-08        6.7          1.046           1.074        1.033     -0.1454           NK        CD158k    phase 2
    CD70          1.927          0.8846                0.7759           0.2415 8.933e-08      6.528          1.028          0.9889       0.8851     -0.2864          CD8          CD70    phase 1
     TPO          1.852          0.9615                 0.569           0.1228 8.094e-08      7.764          1.017          0.9875       0.9401     0.01127 CD4_reactive                         
   TIGIT          1.653          0.8269                 0.931           0.4384 4.186e-07      6.457           1.03           1.068        1.009     -0.1992           NK                         
    STAM          1.643          0.8654                0.8621           0.4449 1.488e-07      5.787          1.057           1.048       0.9505      -0.122 CD4_reactive                         
   FCRL3          1.541          0.7115                0.6034           0.1954 3.687e-06       5.08          1.007           1.018       0.9446     -0.1463           NK                         
   GPR15           1.44          0.8269                0.5345           0.1801 9.642e-07      4.832           1.09            1.07       0.9452    -0.06668 CD4_reactive                         
   PDCD1          1.223          0.8654                0.5345           0.1076 2.289e-06      6.635         0.9657           1.038       0.9351     -0.2967          CD8          PD-1    phase 2
   NRROS            1.2          0.9038                0.6379            0.157 9.874e-08      6.192         0.9454           1.019        0.924    -0.07522           NK                         
    PI16          1.171          0.8462                0.6897           0.2539 1.444e-06      5.179          1.084           1.037         1.06     -0.1048 CD4_reactive                         
   ITGB1          1.158          0.9038                0.9828           0.8281 2.371e-07      7.167          1.015           1.024       0.9935     -0.2353           NK                         
    CCR4           1.13          0.7692                0.7241           0.2099   5.6e-05      3.783          1.035           1.019        1.005     -0.3069 CD4_reactive          CCR4   approved
    CD84          1.067          0.9808                0.9483           0.2466 8.094e-08      7.546          1.034           1.033        0.939   -0.003074 CD4_reactive CD84 / SLAMF5  phase 1/2
   EPHA4         0.9893          0.9423                0.8621           0.2017 8.094e-08      7.246          1.085           1.037        1.059     -0.2729 CD4_reactive                         
  TSPAN2         0.9381          0.9231                0.6379           0.1378 2.792e-07      6.289          1.061           1.064        1.059       -0.31           NK                         
   ABCC1         0.9072          0.9038                0.9138           0.2211 2.111e-07      5.417          1.048           1.059       0.9751    -0.04252 CD4_reactive                         
     MBP         0.8426          0.9423                     1           0.7274 8.094e-08      6.948          1.031           1.005         1.16     -0.2087           NK                         
  GPR146         0.8242          0.9038                0.5172           0.1104 8.841e-07      5.511         0.9841          0.9774        1.052    0.004952 CD4_reactive                         
    FUT8         0.7933          0.8269                0.8793            0.227 1.569e-06      5.724          1.036           1.147       0.8998      -0.116 CD4_reactive                         
  SLAMF6         0.7706          0.9
... [4794 more chars truncated]
```

```python  # cell 13
# ============================================================================
# §3 FA  The leading surviving antigens, one line per donor
# ============================================================================
recA = TF.t_paired(long, TOP_GENES, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
recA = recA.merge(ST.annotate(recA["gene"])[["gene", "cd_name", "stage"]], on="gene", how="left")

fig, ax = plt.subplots(figsize=(0.82 * len(recA) + 1.8, 3.6))
TF.paired_slopes(ax, recA)
ax.set_title(f"Top antigens, malignant vs reactive CD4 (n={len(PAIRED_DONORS)})",
             loc="left", fontsize=7.5)
TF.save(fig, "FA_top_candidates_paired", caption_text=LINE_CAPTION)
TF.save_source(recA, "FA_top_candidates_paired_source")
TF.logfig("FA_top_candidates_paired", "leading clean antigens, paired per donor", len(recA))

TF.show_tests(recA, cols=TF.SHOW_COLS + ["cd_name", "stage"])
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FA_top_candidates_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FA_top_candidates_paired.pdf

<Figure size 1164x360 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_FA_top_candidates_paired_source.csv
   gene  n_pairs  med_a    med_b  delta  log2fc  det_a    det_b  n_pos  n_neg         p         q direction reason cd_name    stage
KIR3DL2       52 0.5574  0.01433 0.5478    2.79 0.3672  0.01134     47      5 8.846e-10  4.99e-09        up         CD158k  phase 2
   CD70       52 0.3359  0.03729 0.2872   1.927 0.2415  0.02688     46      5 1.248e-09  4.99e-09        up           CD70  phase 1
    TPO       52 0.1607 0.005578 0.1523   1.852 0.1207 0.005358     50      2 4.175e-10  4.99e-09        up                        
  TIGIT       52 0.6402   0.1463 0.4205   1.653 0.4239   0.1014     43      9 1.639e-08 2.811e-08        up                        
   STAM       52  0.608   0.1561 0.4354   1.643 0.4449   0.1233     45      7 3.413e-09 8.191e-09        up                        
  FCRL3       52 0.2624  0.02889  0.186   1.541 0.1984  0.02155     37     12 3.257e-07 3.553e-07        up                        
  GPR15       52 0.2545  0.01976 0.1784    1.44 0.1816  0.02105     43      9 5.708e-08 8.562e-08        up                        
  PDCD1       52 0.1421  0.03289 0.1101   1.223 0.1128  0.03039     45      7 1.897e-07 2.276e-07        up           PD-1  phase 2
  NRROS       52 0.1866  0.05085 0.1413     1.2 0.1493  0.04705     47      5 1.956e-09 5.867e-09        up                        
   PI16       52 0.3237  0.07829 0.2061   1.171 0.2539  0.07505     44      8 9.468e-08 1.262e-07        up                        
  ITGB1       52  1.752   0.7302  1.074   1.158 0.8281   0.4423     47      5 7.343e-09 1.469e-08        up                        
   CCR4       52 0.2769   0.1131  0.191    1.13 0.2307  0.09565     40     12 8.823e-06 8.823e-06        up           CCR4 approved
```

```python  # cell 14
# ============================================================================
# §3 FB  Calibration: the antigens whose answer is already known
# ============================================================================
# Read this panel before FA. KIR3DL2 and CCR4 are the positive controls -- CCR4 is approved in
# CTCL, KIR3DL2 is lacutamab's target. CD7 and DPP4/CD26 loss IS the Sezary immunophenotype, so
# they must come out strongly negative. If this panel does not reproduce, nothing in FA is
# evidence of anything, and the cleaning in §2 is the first suspect.
CTRL_GENES = [g for g in (ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS)
              if g in set(long["gene"])]
recB = TF.t_paired(long, CTRL_GENES, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
recB = recB.merge(ev[["gene", "artifact_flag"]], on="gene", how="left")

fig, ax = plt.subplots(figsize=(1.0 * len(recB) + 1.6, 3.4))
TF.paired_slopes(ax, recB)
ax.set_title("Validated antigens and Sézary loss markers", loc="left", fontsize=7.5)
TF.save(fig, "FB_controls_paired", caption_text=LINE_CAPTION)
TF.save_source(recB, "FB_controls_paired_source")
TF.logfig("FB_controls_paired", "positive controls and loss markers, paired", len(recB))

TF.show_tests(recB, cols=TF.SHOW_COLS + ["artifact_flag"])
_missing = sorted(set(ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS) - set(long["gene"]))
if _missing:
    print(f"\nnot testable, absent from the 10k HVG var: {_missing}")
    print("TNFRSF8 is CD30, the brentuximab target, and CD26 is an alias of DPP4 which is "
          "present -- so the real loss is CD30 alone.")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FB_controls_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FB_controls_paired.pdf

<Figure size 1060x340 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_FB_controls_paired_source.csv
   gene  n_pairs    med_a   med_b    delta   log2fc    det_a   det_b  n_pos  n_neg         p         q direction reason artifact_flag
   CCR4       52   0.2769  0.1131    0.191     1.13   0.2307 0.09565     40     12 8.823e-06 1.323e-05        up                clean
KIR3DL2       52   0.5574 0.01433   0.5478     2.79   0.3672 0.01134     47      5 8.846e-10 3.981e-09        up                clean
  IL2RA       52  0.05406 0.08187 -0.01842  -0.1898   0.0383 0.06263     20     31    0.1386    0.1559        ns                clean
    CD7       52   0.2304    1.16  -0.8836   -1.994    0.148  0.6484      6     46 4.495e-09 1.349e-08      down                clean
   DPP4       52 0.001932  0.1759   -0.167   -1.907 0.002277  0.1364      1     51 3.715e-10 3.343e-09      down                clean
    CD5       52   0.5673  0.6569  -0.0059 -0.01247   0.4284  0.4549     25     27    0.7157    0.7157        ns                clean
    CD6       52   0.4901  0.7186  -0.1782  -0.4586   0.3747  0.4993     13     39 1.651e-05 2.122e-05      down                clean
   CD27       52    1.035   0.701   0.3604   0.5282   0.6295  0.4708     43      9 1.558e-07 3.505e-07        up                clean
    CD2       52   0.5731   1.027  -0.2382  -0.5044   0.4462  0.6515     10     42 1.267e-06  2.28e-06      down                clean

not testable, absent from the 10k HVG var: ['CD26', 'TNFRSF8']
TNFRSF8 is CD30, the brentuximab target, and CD26 is an alias of DPP4 which is present -- so the real loss is CD30 alone.
```

```python  # cell 15
# ============================================================================
# §3 FC  The surviving candidates as one ranked panel
# ============================================================================
recC = TF.t_paired(long, SHORT_GENES, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
recC = recC.merge(ST.annotate(recC["gene"])[["gene", "cd_name", "stage"]], on="gene", how="left")
recC = recC.merge(core[["gene", "frac_donors_positive", "median_pct_expr", "worst_T_arm",
                         "worst_T_arm_pct"]], on="gene", how="left")
recC = recC.merge(prev[["gene", "n_donors"]].rename(columns={"n_donors": "n_donors_malig"}),
                  on="gene", how="left")
recC["ylabel"] = [f"{TF.it(r['gene'])}" + (f" · {r['cd_name']}" if r["cd_name"] else "")
                  for _, r in recC.iterrows()]
recC = recC.sort_values("log2fc", ascending=False)

fig, _ = TF.shortlist_panel(recC, floor=DETECT_FLOOR,
                            title="Shortlist: effect, patients positive, off-tumour burden")
TF.save(fig, "FC_shortlist_dumbbell", caption_text=CAPTION)
TF.save_source(recC, "FC_shortlist_dumbbell_source")
TF.logfig("FC_shortlist_dumbbell", "ranked clean candidates, both arms", len(recC))

TF.show_tests(recC, cols=TF.SHOW_COLS + ["cd_name", "stage", "frac_donors_positive",
                                         "median_pct_expr", "worst_T_arm", "worst_T_arm_pct"])
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FC_shortlist_dumbbell.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FC_shortlist_dumbbell.pdf

<Figure size 900x870 with 3 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_FC_shortlist_dumbbell_source.csv
    gene  n_pairs  med_a    med_b   delta  log2fc  det_a    det_b  n_pos  n_neg         p         q direction reason       cd_name      stage  frac_donors_positive  median_pct_expr  worst_T_arm  worst_T_arm_pct
 KIR3DL2       52 0.5574  0.01433  0.5478    2.79 0.3672  0.01134     47      5 8.846e-10  4.25e-09        up               CD158k    phase 2                0.7759           0.3672           NK          0.08938
    CD70       52 0.3359  0.03729  0.2872   1.927 0.2415  0.02688     46      5 1.248e-09 4.678e-09        up                 CD70    phase 1                0.7759           0.2415          CD8          0.03532
     TPO       52 0.1607 0.005578  0.1523   1.852 0.1207 0.005358     50      2 4.175e-10  4.25e-09        up                                                 0.569           0.1228 CD4_reactive         0.005199
   TIGIT       52 0.6402   0.1463  0.4205   1.653 0.4239   0.1014     43      9 1.639e-08 3.074e-08        up                                                 0.931           0.4384           NK           0.2825
    STAM       52  0.608   0.1561  0.4354   1.643 0.4449   0.1233     45      7 3.413e-09 9.308e-09        up                                                0.8621           0.4449 CD4_reactive            0.122
   FCRL3       52 0.2624  0.02889   0.186   1.541 0.1984  0.02155     37     12 3.257e-07 3.908e-07        up                                                0.6034           0.1954           NK           0.1205
   GPR15       52 0.2545  0.01976  0.1784    1.44 0.1816  0.02105     43      9 5.708e-08 9.012e-08        up                                                0.5345           0.1801 CD4_reactive          0.02027
   PDCD1       52 0.1421  0.03289  0.1101   1.223 0.1128  0.03039     45      7 1.897e-07 2.371e-07        up                 PD-1    phase 2                0.5345           0.1076          CD8          0.04077
   NRROS       52 0.1866  0.05085  0.1413     1.2 0.1493  0.04705     47      5 1.956e-09 6.205e-09        up                                                0.6379            0.157           NK           0.1515
    PI16       52 0.3237  0.07829  0.2061   1.171 0.2539  0.07505     44      8 9.468e-08  1.42e-07        up                                                0.6897           0.2539 CD4_reactive           0.0766
   ITGB1       52  1.752   0.7302   1.074   1.158 0.8281   0.4423     47      5 7.343e-09 1.694e-08        up                                                0.9828           0.8281           NK           0.5229
    CCR4       52 0.2769   0.1131   0.191    1.13 0.2307  0.09565     40     12 8.823e-06 9.803e-06        up                 CCR4   approved                0.7241           0.2099 CD4_reactive          0.09893
    CD84       52 0.3287    0.121  0.1997   1.067 0.2428   0.1015     51      1 5.582e-10  4.25e-09        up        CD84 / SLAMF5  phase 1/2                0.9483           0.2466 CD4_reactive           0.1013
   EPHA4       52 0.2638  0.08086  0.1349  0.9893 0.1885  0.06046     49      3 7.031e-10  4.25e-09        up                                                0.8621           0.2017 CD4_reactive          0.06157
  TSPAN2       52 0.1703  0.05406  0.1205  0.9381 0.1318  0.05157     48      4 9.617e-09 2.061e-08        up                                                0.6379           0.1378           NK          0.09149
   ABCC1       52 0.2855   0.1476   0.155  0.9072 0.2211   0.1117     47      5 6.239e-09  1.56e-08        up                                                0.9138           0.2211 CD4_reactive           0.1073
     MBP       52  1.242   0.6643  0.5874  0.8426 0.7195   0.4424     49      3 8.353e-10  4.25e-09        up                  
... [2826 more chars truncated]
```

```python  # cell 16
# ============================================================================
# §3 FD  Leading markers inside each sample: the donor x gene effect matrix
# ============================================================================
M = TH.lfc_matrix(long, MALIG, REACT, genes=TOP_GENES, min_donors=MIN_DONORS)
M = M.loc[[g for g in TOP_GENES if g in M.index]]
# donors ordered by their own malignant burden, so the eye can check the one thing §2 warned
# about: an effect that grows left to right is an effect that grows with arm composition.
_b = bur["burden"].reindex(M.columns).dropna().sort_values()
M = M[list(_b.index)]

fig, ax = plt.subplots(figsize=(0.13 * M.shape[1] + 3.2, 0.19 * len(M) + 1.8))
TF.donor_gene_heat(ax, M, stars_by=dict(zip(core["gene"], core["q"])))
ax.set_xlabel(f"donor, ordered by malignant share of CD4 "
              f"({_b.iloc[0]:.0%} → {_b.iloc[-1]:.0%})")
ax.set_title("Per-donor log2FC, malignant vs reactive", loc="left", fontsize=7.5)
TF.save(fig, "FD_donor_gene_heat", caption_text=CAPTION)
TF.save_source(M.reset_index(), "FD_donor_gene_heat_source")
TF.logfig("FD_donor_gene_heat", "per-donor log2FC matrix, donors by burden", M.size)

print("per-donor log2FC, complete matrix (genes x donors):")
print(M.round(3).to_string())
print("\nper-gene summary of that matrix:")
_sum = pd.DataFrame({
    "n_donors": M.notna().sum(1),
    "median": M.median(1), "iqr": M.quantile(.75, axis=1) - M.quantile(.25, axis=1),
    "min": M.min(1), "max": M.max(1),
    "frac_up": (M > 0).sum(1) / M.notna().sum(1),
    "frac_up_1": (M > 1).sum(1) / M.notna().sum(1),
}).sort_values("median", ascending=False)
print(_sum.to_string(float_format=lambda v: f"{v:.3g}"))
print("\n`frac_up_1` -- donors where the antigen is at least doubled -- is the column that "
      "separates a cohort-wide antigen from one that is strong in a subset. 44 takes that "
      "apart formally.")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FD_donor_gene_heat.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FD_donor_gene_heat.pdf

<Figure size 996x408 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_FD_donor_gene_heat_source.csv
per-donor log2FC, complete matrix (genes x donors):
donor    D13__SZ16  B8__1181  B4__SZ32_HTO1  B5__PT22  B5__PT21  D13__SZ22  H__SS4  B5__PT23  D5__MFIVB  B2__CTCL6  H__MFIV1  B4__SZ39_HTO1  B2__CTCL4  H__SS5  H__SS6  H__SS1  B4__SZ30_HTO6  B6__P10  B6__P4  B6__P5  B2__CTCL5  B6__P2  H__SS3  B6__P11  B5__PT18  B6__P7  B8__999  B6__P3  B7__SS1  B5__PT15  B6__P9  B7__SS4  B5__PT14  B5__PT13  B8__1126  B5__PT09  B5__PT10  B6__P6  D13__SZ29  B5__PT16  B5__PT08  B5__PT07  B5__PT06  B5__PT05  B7__SS2  B6__P1  B8__1084  B1__P4  B4__SZ5_HTO1  B5__PT02  B8__1135  B4__SZ38_HTO1
gene                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
KIR3DL2      4.152     4.159          3.674     1.628     4.262      3.828   4.125     0.616      3.609      4.593     4.024         -0.201      2.175   3.204   2.195   2.094          0.424    3.354   3.518   2.465      3.607  -0.028   2.619    0.693     0.828  -0.326    3.984   3.032    0.487     3.088   3.405    1.957     0.373     2.645     2.550     2.256     3.870   2.556     -0.471     4.187     4.008     3.128     2.852     2.600   -0.156   3.948     2.899   4.263         2.642     2.727     3.412          0.875
CD70         1.682     2.430          3.010     2.486     3.464      2.564   1.462     2.385     -0.517      2.069     2.272          1.260      2.764   1.839   2.191   2.582          1.534    2.406   1.928   2.645      3.409  -0.886   1.615    1.369     1.728   2.312    2.003   2.175    1.952     0.000   1.093    1.142     0.807     1.928     2.132     1.136     3.386  -0.502     -0.133     0.605     2.673     2.657     1.646     1.421    2.055  -0.345     1.802   4.150         1.926     1.571     1.892          1.037
TPO          0.533     0.116         -0.038     3.813     1.169      3.116   0.951     2.948      1.215      4.256     1.020          1.296      5.040   2.179   3.962   0.889          0.125    3.577   2.063   1.256      3.742  -0.018   1.420    0.663     1.846   3.881    3.584   2.670    1.908     0.761   3.525    1.385     4.130     0.416     3.769     1.706     1.974   2.949      1.632     3.608     0.480     3.469     0.153     0.256    0.880   3.220     1.857   2.955         0.606     2.872     2.793          0.111
TIGIT       -0.490     2.821          1.015     2.346     2.438      1.408   2.251     2.727      2.763      2.134    -0.470         -0.828      4.335   2.178   1.612   2.975          0.779    1.083   2.745   1.828      1.832   1.135  -0.586    2.196    -1.220   2.156    1.667   1.906    1.414     2.295  -0.109   -0.030     0.702     2.307     1.639     1.969     1.071   1.392      0.840     2.825     2.573     1.668     0.276     1.906    2.982   2.623     1.379   1.363         0.738     1.294    -1.293         -0.075
STAM         0.788     1.713          1.736     2.064     1.111      2.467  -0.391     3.977     -0.393      2.218     1.433         -0.202      2.774   2.593   1.695   2.607          0.629    1.475   2.687   1.312      2.111   0.977   1.784    1.896    -1.409   1.607    2.239   1.823    1.524     1.588   0.549    1.408     2.081     1.971     1.678     2.243     1.430   0.796      1.329    -0.478     1.989     2.281     2.307     1.608   -0.338   0.278     1.765   1.818         2.269    -0.775     1.194          1.
... [4804 more chars truncated]
```

```python  # cell 17
# ============================================================================
# §3 FE  Detection, not expression: the axis a CAR is actually dosed against
# ============================================================================
recE = TF.t_paired(long, TOP_GENES, MALIG, REACT, value="pct_expr", min_donors=MIN_DONORS)
fig, ax = plt.subplots(figsize=(10.2, 3.4))
TF.paired_slopes(ax, recE, ylabel="fraction of a donor's cells detected", show_detection=False)
ax.set_title("Fraction of CD4 cells detected, per donor", loc="left", fontsize=7.5)
TF.save(fig, "FE_detection_paired", caption_text=LINE_CAPTION)
TF.save_source(recE, "FE_detection_paired_source")
TF.logfig("FE_detection_paired", "paired detection rate per donor", len(recE))

print("Detection is the quantity that decides whether a CAR sees the tumour; expression is the "
      "quantity the ranking is sorted on. They are not the same ordering, and where they "
      "disagree the detection column wins for a cell-therapy target.\n")
TF.show_tests(recE)

print("\n" + "=" * 78)
print("eligibility: in what fraction of patients would at least X% of the tumour be positive")
print("=" * 78)
elig = TH.eligibility(long, MALIG, genes=TOP_GENES, donors=PAIRED_DONORS)
print(elig.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
elig.to_csv(ST.tab("eligibility_blood"), index=False)
print("\nwrote", ST.tab("eligibility_blood"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FE_detection_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_FE_detection_paired.pdf

<Figure size 1020x340 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt43_FE_detection_paired_source.csv
Detection is the quantity that decides whether a CAR sees the tumour; expression is the quantity the ranking is sorted on. They are not the same ordering, and where they disagree the detection column wins for a cell-therapy target.

   gene  n_pairs  med_a    med_b   delta  log2fc  det_a    det_b  n_pos  n_neg         p         q direction reason
KIR3DL2       52 0.3672  0.01134  0.3441   2.476 0.3672  0.01134     47      5 8.353e-10  4.99e-09        up       
   CD70       52 0.2415  0.02688   0.202   1.612 0.2415  0.02688     46      5 1.248e-09  4.99e-09        up       
    TPO       52 0.1207 0.005358  0.1097   1.494 0.1207 0.005358     50      2 4.425e-10  4.99e-09        up       
  TIGIT       52 0.4239   0.1014  0.2583   1.549 0.4239   0.1014     45      7 5.909e-09 1.013e-08        up       
   STAM       52 0.4449   0.1233  0.3069   1.375 0.4449   0.1233     45      7 2.734e-09 6.937e-09        up       
  FCRL3       52 0.1984  0.02155  0.1305   1.353 0.1984  0.02155     39     10 1.815e-07  1.98e-07        up       
  GPR15       52 0.1816  0.02105  0.1077   1.158 0.1816  0.02105     42     10 6.995e-08 9.327e-08        up       
  PDCD1       52 0.1128  0.03039 0.08562   1.059 0.1128  0.03039     47      5 9.955e-08 1.195e-07        up       
  NRROS       52 0.1493  0.04705  0.1013   1.059 0.1493  0.04705     46      6 5.909e-09 1.013e-08        up       
   PI16       52 0.2539  0.07505  0.1478   1.001 0.2539  0.07505     44      8 5.153e-08  7.73e-08        up       
  ITGB1       52 0.8281   0.4423   0.376  0.7791 0.8281   0.4423     48      4  2.89e-09 6.937e-09        up       
   CCR4       52 0.2307  0.09565   0.139  0.8831 0.2307  0.09565     40     12 8.823e-06 8.823e-06        up       

==============================================================================
eligibility: in what fraction of patients would at least X% of the tumour be positive
==============================================================================
   gene  n_donors  median_pct_expr  frac_donors_ge10  frac_donors_ge25  frac_donors_ge50  frac_donors_ge75
  ITGB1        52            0.828             0.981             0.981             0.923             0.673
  TIGIT        52            0.424             0.923              0.75             0.327             0.154
   STAM        52            0.445             0.885             0.769             0.404            0.0192
   CD70        52            0.241             0.808             0.481            0.0577                 0
KIR3DL2        52            0.367             0.769             0.615             0.423             0.115
   CCR4        52            0.231             0.731             0.462             0.115                 0
   PI16        52            0.254             0.712               0.5              0.25            0.0192
  NRROS        52            0.149             0.615             0.308                 0                 0
  FCRL3        52            0.198             0.596             0.385             0.115                 0
  GPR15        52            0.182             0.558             0.385             0.115            0.0192
    TPO        52            0.121             0.558             0.365             0.173            0.0192
  PDCD1        52            0.113             0.519             0.192            0.0192                 0

wrote /home/projects/nyosef/zvise/MF/tables/tgt43_eligibility_blood.csv
```

## §4 — On-target off-tumour in blood

Blood is the compartment where an off-tumour hit is most consequential: a CAR that also binds
monocytes, NK cells or HSPCs causes cytopenia, not a rash. The non-T compartments come from
`31b_subtype_reannotation`; OneK1K (981 healthy donors) is the healthy-PBMC reference.

```python  # cell 19
# ============================================================================
# §4  Off-tumour
# ============================================================================
print("worst non-target T arm, for the surviving candidates:")
_o = (core[core["gene"].isin(SHORT_GENES)]
      [["gene", "cd_name", "worst_T_arm", "worst_T_arm_pct", "healthy_t_burden",
        "median_pct_expr", "frac_donors_positive"]]
      .sort_values("worst_T_arm_pct", ascending=False))
print(_o.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n`healthy_t_burden` is the aplasia cost: the share of HEALTHY CD4 cells that carry the "
      f"antigen, now measured on {meta.loc[meta['arm'].eq(HC_CD4), 'donor'].nunique()} healthy "
      "donors rather than 7, because the five geskin donors that the atlas labelled SS are "
      "healthy and have been moved into this arm.")

# What this axis does NOT cover, stated as a limit rather than a plan. The screen input is
# blood_T_annotated.h5ad, the T/NK compartment only; the non-T cells live in the joint object
# and would need their own pass of jobs/run_target_screen_qc.py to enter `long`.
SUBTYPES = NB_DIR / "data" / "atlas_joint" / "blood_subtypes.csv"
if SUBTYPES.exists():
    sub = pd.read_csv(SUBTYPES, usecols=["subtype_coarse"])
    print(f"\n{SUBTYPES.name}: {len(sub):,} non-T blood cells annotated by 31b "
          f"({sub['subtype_coarse'].nunique()} coarse types) -- but NOT in this axis.")
else:
    print(f"\n{SUBTYPES.name} NOT found: run 31b_subtype_reannotation first.")
print("*** The off-tumour axis above covers T and NK arms ONLY. Monocytes, DC, B, plasma, "
      "platelets and HSPC are not in `long`, so a myeloid or HSPC hit -- the kind that causes "
      "cytopenia, the dose-limiting toxicity for a blood-directed CAR -- is INVISIBLE here. "
      "Quote every blood candidate with this sentence attached. ***")

ONEK1K = NB_DIR / "data" / "external_ref" / "onek1k.h5ad"
print(f"\nhealthy PBMC reference {'present' if ONEK1K.exists() else 'ABSENT'}: {ONEK1K.name} "
      "(not yet folded in either)")
```

```
worst non-target T arm, for the surviving candidates:
    gene       cd_name  worst_T_arm  worst_T_arm_pct  healthy_t_burden  median_pct_expr  frac_donors_positive
     MBP                         NK            0.781             0.446            0.727                     1
   ITGB1                         NK            0.523             0.391            0.828                 0.983
   ANXA2                         NK            0.519             0.321            0.508                 0.983
TNFRSF1B         TNFR2           NK            0.461             0.205            0.278                 0.862
TNFRSF25               CD4_reactive            0.304             0.204            0.395                     1
   TIGIT                         NK            0.283             0.123            0.438                 0.931
    CD28               CD4_reactive            0.264             0.227            0.413                     1
    CD82               CD4_reactive            0.234             0.216            0.439                 0.931
   SIRPG               CD4_reactive            0.187             0.194            0.225                 0.828
  SLAMF6                         NK            0.176              0.11            0.259                 0.879
    IL6R               CD4_reactive            0.171             0.153            0.243                 0.983
     GRN                         NK            0.163            0.0906            0.155                 0.672
   NRROS                         NK            0.152            0.0545            0.157                 0.638
    STAM               CD4_reactive            0.122             0.103            0.445                 0.862
   FCRL3                         NK            0.121            0.0131            0.195                 0.603
    FUT8               CD4_reactive            0.117            0.0862            0.227                 0.879
   ABCC1               CD4_reactive            0.107            0.0699            0.221                 0.914
    CD84 CD84 / SLAMF5 CD4_reactive            0.101            0.0908            0.247                 0.948
    CCR4          CCR4 CD4_reactive           0.0989              0.07             0.21                 0.724
   CTLA4               CD4_reactive           0.0916            0.0847            0.163                 0.741
  TSPAN2                         NK           0.0915            0.0346            0.138                 0.638
 KIR3DL2        CD158k           NK           0.0894           0.00758            0.367                 0.776
    PI16               CD4_reactive           0.0766            0.0586            0.254                  0.69
   EPHA4               CD4_reactive           0.0616            0.0493            0.202                 0.862
    TFRC               CD4_reactive           0.0573            0.0387            0.126                 0.655
  GPR146               CD4_reactive           0.0463            0.0462             0.11                 0.517
   PDCD1          PD-1          CD8           0.0408            0.0289            0.108                 0.534
    CD70          CD70          CD8           0.0353            0.0168            0.241                 0.776
   GPR15               CD4_reactive           0.0203            0.0123             0.18                 0.534
     TPO               CD4_reactive           0.0052           0.00137            0.123                 0.569

`healthy_t_burden` is the aplasia cost: the share of HEALTHY CD4 cells that carry the antigen, now measured on 12 healthy donors rather than 7, because the five geskin donors that the atlas labelled SS are healthy and have been moved into this arm.


blood_subtypes.csv: 143,440 non-T blood cells annotated by 31b (15 coarse types) -- but NOT in this axis.
*** The off-tumour axis above covers T and NK arms ONLY. Monocytes, DC, B, plasma, platelets and HSPC are not in `long`, so a myeloid or HSPC hit -- the kind that causes cytopeni
... [209 more chars truncated]
```

```python  # cell 20
# ============================================================================
# §5-6  Surface plausibility (trafficking context) + escape risk
# ============================================================================
tr = long[long["gene"].isin(ST.TRAFFIC_GENES)].copy()
tr["module"] = tr["gene"].map(lambda g: ST.GENE_TO_TRAFFIC.get(g, [""])[0])
piv = tr.groupby(["arm", "donor", "module"])["mean_all"].mean().reset_index()
piv["z"] = piv.groupby("module")["mean_all"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
arm_mod = piv.pivot_table(index="module", columns="arm", values="z")
print("trafficking module z by arm -- the capacity to put a protein on the surface and keep "
      "it there:")
print(arm_mod[[c for c in (MALIG, REACT, HC_CD4) if c in arm_mod]].round(2).to_string())
print("\nThis is context, not a correction factor. 45 asks the sharper version of the same "
      "question on measured protein: is a marker's surface level better explained by these "
      "modules than by its own transcript.")

print("\nescape risk computed on",
      "32b clusters" if cluster_long is not None else "donors only")
print(esc[esc["gene"].isin(SHORT_GENES)]
      .sort_values("between_donor_sd", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n`between_donor_sd` is the spread of detection across patients and "
      "`within_donor_sd` the spread across a patient's own clusters -- a high second column is "
      "the antigen-negative subclone a single-target agent would select for. 44 takes the "
      "between-patient term apart properly.")
```

```
trafficking module z by arm -- the capacity to put a protein on the surface and keep it there:
arm                     CD4_malignant  CD4_reactive  CD4_HC
module                                                     
endocytosis                      0.02         -0.04   -0.30
er_folding_qc                   -0.09         -0.37   -0.14
er_to_golgi                      0.25         -0.44    0.05
er_translocation                 0.09         -0.26   -0.07
escrt_exosome                    1.03         -0.29   -0.40
glycosylation                    0.73         -0.32   -0.40
recycling                        0.50         -0.34   -0.65
retrograde                      -0.22         -0.16    0.01
shedding                         0.09          0.12   -0.08
snare_fusion                    -0.41         -0.40   -0.39
ubiquitin_proteostasis           0.49         -0.21   -0.62

This is context, not a correction factor. 45 asks the sharper version of the same question on measured protein: is a marker's surface level better explained by these modules than by its own transcript.

escape risk computed on 32b clusters
    gene  between_donor_sd  min_donor_pct  within_donor_sd
 KIR3DL2             0.288              0            0.156
TNFRSF1B             0.252         0.0241            0.101
     TPO             0.241              0            0.103
    PI16             0.238              0            0.118
   ANXA2             0.228         0.0638            0.189
   TIGIT             0.226         0.0298            0.139
    STAM             0.209          0.038            0.118
   GPR15             0.202       0.000355           0.0712
    CD82             0.197         0.0163            0.135
   FCRL3             0.194              0           0.0797
   ITGB1             0.193         0.0455            0.121
    CCR4             0.186        0.00256           0.0995
TNFRSF25             0.182          0.109            0.103
  TSPAN2              0.18       0.000542           0.0906
   SIRPG             0.178       0.000261            0.116
     MBP             0.173          0.269            0.134
   CTLA4             0.172        0.00778           0.0723
    CD70             0.168              0            0.107
    IL6R             0.162         0.0891           0.0917
    CD28             0.153          0.133            0.126
    CD84             0.153         0.0865            0.104
   ABCC1             0.143         0.0311           0.0829
  SLAMF6             0.143         0.0506           0.0999
   NRROS             0.141         0.0112           0.0811
    FUT8             0.136         0.0602            0.092
   PDCD1             0.128        0.00198           0.0589
   EPHA4              0.12         0.0395           0.0865
  GPR146             0.117        0.00163           0.0536
     GRN            0.0985         0.0206           0.0663
    TFRC            0.0824         0.0225           0.0561

`between_donor_sd` is the spread of detection across patients and `within_donor_sd` the spread across a patient's own clusters -- a high second column is the antigen-negative subclone a single-target agent would select for. 44 takes the between-patient term apart properly.
```

```python  # cell 21
# ============================================================================
# §7  Assemble, check the controls, shortlist
# ============================================================================
surf = ST.load_surfaceome(strict=False)
rank = TS.rank_table(prev, sel, off, esc, ST.annotate, surfaceome=surf)
rank = rank.merge(ev[["gene", "artifact_flag", "n_fails", "fail_reasons", "retained_trim",
                      "retained_clean", "retained_ds", "rho_burden", "top1_share"]],
                  on="gene", how="left")
if "top_cluster_share" in ev:
    rank = rank.merge(ev[["gene", "top_cluster_share"]], on="gene", how="left")

hit, missing, n_tested = TS.control_check(rank, ST.POSITIVE_CONTROLS)
print(f"positive controls of {n_tested:,} ranked genes:")
print(hit.to_string(index=False))
if missing:
    print(f"*** not testable: {missing} -- dropped by HVG selection, so the calibration runs "
          f"on {len(hit)} of {len(ST.POSITIVE_CONTROLS)} controls ***")

short = rank[rank["gene"].isin(SHORT_GENES)].copy()
show = ["gene", "cd_name", "frac_donors_positive", "median_pct_expr", "log2fc_paired", "q",
        "worst_T_arm", "healthy_t_burden", "between_donor_sd", "stage", "n_evidence",
        "surface_confident", "n_tm", "surfy_class", "artifact_flag"]
print(f"\n{len(short)} candidates pass the gates:")
print(short[[c for c in show if c in short]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

print("\n" + "=" * 78)
print("PLAUSIBILITY, which the statistics cannot decide")
print("=" * 78)
print("`surface_confident` / `n_tm` / `surfy_class` come from CSPA + Surfy, not from this "
      "cohort. A transcript can be reproducibly, tumour-restrictedly up and still be a hopeless "
      "antibody or CAR target because the protein is cytoplasmic, secreted, or expressed on a "
      "tissue this atlas cannot see. Genes with no surfaceome evidence are the ones to argue "
      "about before anyone orders a reagent:")
_noev = short[short["surface_confident"].isna() | ~short["surface_confident"].astype(bool)]
print(_noev[["gene", "cd_name", "log2fc_paired", "median_pct_expr", "n_evidence",
             "surfy_class", "topology"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:.3g}")
      if len(_noev) else "  every shortlisted gene has confident surfaceome evidence")

rank.to_csv(ST.tab("candidates_blood"), index=False)
short.to_csv(ST.tab("shortlist_blood"), index=False)
print("\nwrote", ST.tab("candidates_blood"), "and", ST.tab("shortlist_blood"))
```

```
positive controls of 1,098 ranked genes:
   gene  rank  log2fc_paired  frac_donors_positive
KIR3DL2     1       2.789584              0.775862
   CCR4    14       1.130046              0.724138
  IL2RA  1003      -0.189832              0.258621
*** not testable: ['TNFRSF8'] -- dropped by HVG selection, so the calibration runs on 3 of 4 controls ***

30 candidates pass the gates:
    gene       cd_name  frac_donors_positive  median_pct_expr  log2fc_paired         q  worst_T_arm  healthy_t_burden  between_donor_sd      stage  n_evidence surface_confident n_tm   surfy_class artifact_flag
 KIR3DL2        CD158k                0.7759           0.3672           2.79 8.094e-08           NK          0.007578            0.2883    phase 2           1              True  1.0     Receptors         clean
    CD70          CD70                0.7759           0.2415          1.927 8.933e-08          CD8           0.01684            0.1675    phase 1           2              True  1.0  Unclassified         clean
     TPO                               0.569           0.1228          1.852 8.094e-08 CD4_reactive          0.001367            0.2411                      2              True  1.0       Enzymes         clean
   TIGIT                               0.931           0.4384          1.653 4.186e-07           NK            0.1227            0.2262                      3              True  1.0 Miscellaneous         clean
    STAM                              0.8621           0.4449          1.643 1.488e-07 CD4_reactive            0.1029            0.2088                    NaN               NaN  NaN           NaN         clean
   FCRL3                              0.6034           0.1954          1.541 3.687e-06           NK           0.01309            0.1939                      3              True  1.0     Receptors         clean
   GPR15                              0.5345           0.1801           1.44 9.642e-07 CD4_reactive           0.01233            0.2019                      1              True  7.0     Receptors         clean
   PDCD1          PD-1                0.5345           0.1076          1.223 2.289e-06          CD8           0.02886            0.1282    phase 2           3              True  1.0  Unclassified         clean
   NRROS                              0.6379            0.157            1.2 9.874e-08           NK           0.05454            0.1405                      2              True  1.0 Miscellaneous         clean
    PI16                              0.6897           0.2539          1.171 1.444e-06 CD4_reactive           0.05855            0.2381                      2              True  1.0  Unclassified         clean
   ITGB1                              0.9828           0.8281          1.158 2.371e-07           NK            0.3906            0.1934                      4              True  1.0     Receptors         clean
    CCR4          CCR4                0.7241           0.2099           1.13   5.6e-05 CD4_reactive              0.07            0.1862   approved           4              True  7.0     Receptors         clean
    CD84 CD84 / SLAMF5                0.9483           0.2466          1.067 8.094e-08 CD4_reactive           0.09081             0.153  phase 1/2           4              True  1.0  Unclassified         clean
   EPHA4                              0.8621           0.2017         0.9893 8.094e-08 CD4_reactive           0.04934            0.1197                      3              True  1.0     Receptors         clean
  TSPAN2                              0.6379           0.1378         0.9381 2.792e-07           NK           0.03462            0.1801                      2              True  4.0 Miscellaneous         clean
   ABCC1                              0.9138           0.2211         0.9072 2.111e-07 CD4_reactive           0.06985            0.1428                      3              True 17.0  Transporters         clean
     MBP                                   1    
... [3856 more chars truncated]
```

## §8 — Skin ∩ blood

A target that separates malignant from reactive in **both** compartments is the headline result:
MF is a skin disease that becomes a blood disease, and an agent that works in one compartment
and not the other leaves the reservoir that repopulates it.

Discordance is informative too — a skin-only hit is a candidate for intralesional or topical
delivery, a blood-only hit for a systemic agent in SS.

> **Both sides have now had the same treatment.** 42 was rebuilt on this notebook's template:
> independent donors only, diagnosis from `clin_dx`, true-library normalisation, and the same
> six artifact tests against a skin-specific contamination null (keratin, collagen, melanocyte
> and myeloid transcripts in place of the monocyte and platelet ones). So the two columns below
> are comparable in a way they were not before.
>
> Two asymmetries remain and neither is fixable here. The blood object is a 10,000-gene HVG
> subset and the skin object is not, so a gene can be absent from the blood column for a reason
> that has nothing to do with biology. And the skin screen can see the whole tissue — its
> off-tumour axis covers keratinocytes, fibroblasts and myeloid cells — while the blood screen
> sees only the T and NK arms.

```python  # cell 23
# ============================================================================
# §8  Skin and blood side by side
# ============================================================================
skin_tab = ST.TAB_DIR / "tgt42_candidates_skin.csv"
if not skin_tab.exists():
    print(f"{skin_tab.name} not found -- run 42 first")
else:
    sk = pd.read_csv(skin_tab)
    both = sk.merge(rank, on="gene", suffixes=("_skin", "_blood"))
    both = both.dropna(subset=["log2fc_paired_skin", "log2fc_paired_blood"])
    print(f"{len(both):,} genes screened in both compartments "
          "(both sides cohort-fixed and artifact-tested; see the note above for what still "
          "differs)")

    sig = both[(both["q_skin"] < 0.05) & (both["q_blood"] < 0.05)]
    up_both = sig[(sig["log2fc_paired_skin"] > 0.5) & (sig["log2fc_paired_blood"] > 0.5)]
    print(f"\nselective in BOTH compartments: {len(up_both)}")
    cols = ["gene", "cd_name_skin", "log2fc_paired_skin", "log2fc_paired_blood",
            "frac_donors_positive_skin", "frac_donors_positive_blood", "stage_skin"]
    print(up_both[[c for c in cols if c in up_both]]
          .sort_values("log2fc_paired_blood", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.axhline(0, lw=.5, c="k"); ax.axvline(0, lw=.5, c="k")
    ax.scatter(both["log2fc_paired_skin"], both["log2fc_paired_blood"], s=4,
               c="#d5d8dc", linewidths=0)
    k = both["gene"].isin(up_both["gene"])
    ax.scatter(both.loc[k, "log2fc_paired_skin"], both.loc[k, "log2fc_paired_blood"], s=10,
               c="#2e86c1", linewidths=0)
    kc = both["gene"].isin(ST.POSITIVE_CONTROLS + TQ.SUSPECTS)
    ax.scatter(both.loc[kc, "log2fc_paired_skin"], both.loc[kc, "log2fc_paired_blood"], s=26,
               c=TF.DOWN, linewidths=0)
    for _, r_ in both[kc].iterrows():
        ax.annotate(r_["gene"], (r_["log2fc_paired_skin"], r_["log2fc_paired_blood"]),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("skin: log2FC malignant vs reactive")
    ax.set_ylabel("blood: log2FC malignant vs reactive")
    ax.set_title(f"{len(up_both)} antigens selective in skin and blood",
                 loc="left", fontsize=7.5)
    TF._clean(ax)
    fig.tight_layout()
    TF.save(fig, "skin_vs_blood", caption_text=(
        "Red: validated antigens and disputed genes. Both compartments artifact-tested."))
    up_both.to_csv(ST.tab("shared_skin_blood"), index=False)
    TF.logfig("skin_vs_blood", "skin vs blood selectivity", len(both))
    print("\nwrote", ST.tab("shared_skin_blood"))
```

```
1,091 genes screened in both compartments (both sides cohort-fixed and artifact-tested; see the note above for what still differs)

selective in BOTH compartments: 14
    gene cd_name_skin  log2fc_paired_skin  log2fc_paired_blood  frac_donors_positive_skin  frac_donors_positive_blood stage_skin
    CD70         CD70               1.102                1.927                     0.8077                      0.7759    phase 1
  ADGRE1          NaN              0.7917                1.151                     0.3077                      0.4828        NaN
     CPM          NaN               1.023                1.103                     0.5769                      0.2931        NaN
   PTH2R          NaN              0.5299               0.9706                     0.3077                      0.2931        NaN
  TSPAN2          NaN              0.5299               0.9381                     0.6538                      0.6379        NaN
    EVC2          NaN              0.6235               0.8413                     0.2308                      0.1552        NaN
    FUT8          NaN              0.6353               0.7933                     0.7436                      0.8793        NaN
    CD82          NaN               0.632               0.7649                          1                       0.931        NaN
    TFRC          NaN              0.7313               0.7456                     0.9487                      0.6552        NaN
    IL6R          NaN              0.6145               0.7376                     0.7692                      0.9828        NaN
     GRN          NaN              0.7777               0.6237                     0.8462                      0.6724        NaN
TNFRSF25          NaN              0.6977               0.6023                     0.8718                           1        NaN
  SLC5A3          NaN              0.6169               0.5352                     0.7692                      0.5345        NaN
 TNFRSF4          NaN              0.5419               0.5334                     0.8974                      0.6897        NaN

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_skin_vs_blood.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt43_skin_vs_blood.pdf

<Figure size 500x340 with 1 Axes>

wrote /home/projects/nyosef/zvise/MF/tables/tgt43_shared_skin_blood.csv
```

```python  # cell 24
# ============================================================================
# §9  Dossier
# ============================================================================
def SEC(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def kv(k, v): print(f"  {k:34s} {v}")

SEC("1 - INPUT")
kv("object", Path(man["source"]).name)
kv("builder", man.get("builder"))
kv("genes in object", f"{man['n_genes_in_object']:,}"
   + ("  *** HVG subset ***" if man["n_genes_in_object"] < 20000 else ""))
kv("malignancy call", man["malignancy_call"])
kv("library for CP10K", man.get("library"))

SEC("2 - COHORT")
kv("donor keys in the object", len(don))
kv("dropped: ex vivo drug arms", int(don["drop_reason"].eq("ex vivo drug-culture arm").sum()))
kv("dropped: duplicate persons",
   int(don["drop_reason"].fillna("").str.startswith("same person").sum()))
kv("independent keys", int(don["cohort_keep"].sum()))
kv("of those, 2-day cultured", int((don["cohort_keep"] & don["is_cultured"]).sum()))
kv("healthy donors moved to CD4_HC", int((don["is_hc"] & don["cohort_keep"]).sum()))
kv("donors with both CD4 arms", len(PAIRED_DONORS))
print()
print(pd.crosstab(don.loc[don["cohort_keep"], "dx"],
                  don.loc[don["cohort_keep"], "study"]).to_string())

SEC("3 - ARMS"); print(arms.to_string())

SEC("4 - ARTIFACT TESTS")
kv("lineage controls found", band.attrs["n_genes"])
kv("null calibrated", ok)
print("  " + msg)
print()
print(ev["artifact_flag"].value_counts().to_string())
kv("\n  genes excluded from the headline",
   int((ev["artifact_flag"].ne("clean") & (ev["log2fc_paired"] > 0)).sum()))

SEC("5 - CALIBRATION"); print(hit.to_string(index=False))
if missing:
    print(f"  not testable (HVG): {missing}")

SEC("6 - WHAT THIS IS NOT")
for line in [
    "A full-gene screen: the object is the 10k HVG subset, and CD4 and TNFRSF8 are not in it.",
    "A myeloid or HSPC safety statement: those compartments are not in `long` at all.",
    "A statement about the CD8 or NK arm: nb31 labelled on CD8A and that arm is a mixture.",
    "A target call. Nothing here has been seen as protein; 45 does that for 17 Sezary donors.",
    "A full-gene comparison with skin: the skin object carries all 42,347 genes and this "
    "one does not, so §8 compares only what both objects hold.",
]:
    print("  - " + line)

SEC("7 - OUTPUTS")
TF.write_index("figure_index")
for p in sorted(ST.TAB_DIR.glob(f"{PREFIX}*")) + sorted(ST.FIG_DIR.glob(f"{PREFIX}*")):
    print("  ", p.relative_to(NB_DIR))
```

```
==============================================================================
1 - INPUT
==============================================================================
  object                             blood_T_annotated.h5ad
  builder                            run_target_screen_qc.py
  genes in object                    10,000  *** HVG subset ***
  malignancy call                    mal_cluster
  library for CP10K                  obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg

==============================================================================
2 - COHORT
==============================================================================
  donor keys in the object           131
  dropped: ex vivo drug arms         54
  dropped: duplicate persons         2
  independent keys                   75
  of those, 2-day cultured           11
  healthy donors moved to CD4_HC     12
  donors with both CD4 arms          52

study                borcherding2019  borcherding2023  buus2025  dorando2026  gaydosik2022  geskin2026  harro2023  herrera2021  ren2023  rindler2021_fi
dx                                                                                                                                                     
HC                                 0                0         0            0             3           5          0            1        3               0
MF                                 0                0         2            0             0           0          3            1        4               1
MF/SS                              0                0         0            0             1           1          0            0        0               0
MF_or_SS_unresolved                0                0         0            6             0           0          0            0        0               0
SS                                 1                5        15            0             2           5          4            5        7               0

==============================================================================
3 - ARMS
==============================================================================
               donors   cells  studies
arm                                   
CD4_malignant      58  295656       10
CD4_reactive       57  111125       10
CD8                72   70525       10
CD4_HC             12   40400        4
NK                  6     756        3

==============================================================================
4 - ARTIFACT TESTS
==============================================================================
  lineage controls found             18
  null calibrated                    False
  18 lineage controls: median log2FC +0.00, median frac_donors_up 0.39 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.

artifact_flag
clean    1098
  
  genes excluded from the headline 0

==============================================================================
5 - CALIBRATION
==============================================================================
   gene  rank  log2fc_paired  frac_donors_positive
KIR3DL2     1       2.789584              0.775862
   CCR4    14       1.130046              0.724138
  IL2RA  1003      -0.189832              0.258621
  not testable (HVG): ['TNFRSF8']

==============================================================================
6 - WHAT THIS IS NOT
==============================================================================
  - A full-gene screen: the object is the 10k HVG subset, and CD4 and TNFRSF8 are not in it.
  - A myeloid or HSPC safety statement: those compartments are not in `long` at all.
  - A statement about the CD8 or NK arm: nb31 labelled on CD8A and that arm is a mixture.
  - A target call. Nothing here has been seen as protein; 45 does that for 17 Sezary donors.
  - A
... [1675 more chars truncated]
```

```python  # cell 25
# ============================================================================
# §10  The complete numerical result, as one block to hand to another model
# ============================================================================
_ctx = f"""
Compartment: MF/CTCL **blood**, from {Path(man['source']).name}
({man['n_genes_in_object']:,} genes in the object -- an HVG SUBSET, so a gene absent from the
screen is not necessarily unexpressed; CD4 and TNFRSF8/CD30 are both missing for this reason).

Cohort: {len(don)} donor keys reduced to {int(don['cohort_keep'].sum())} independent keys by
dropping the geskin2026 ex vivo drug-culture wells and duplicate persons, and by moving five
healthy donors that the atlas labelled SS out of the reactive arm. Diagnosis from the curated
`clin_dx`. {len(PAIRED_DONORS)} donors carry both a malignant and a reactive CD4 arm.

Arms: malignancy call `{man['malignancy_call']}`. Blood in Sezary syndrome is ~92% malignant
CD4, so the reactive arm is small in many donors. The CD8 arm is a CD8+NK mixture (nb31 labelled
on CD8A) and any statement about it is a statement about a mixture.

Headline test: per gene, the within-donor difference of mean log1p CP10K between the malignant
and reactive arms, paired Wilcoxon over donors, BH-adjusted (`q`). `log2fc_paired` is the median
per-donor log2 ratio; `frac_donors_up` the share of donors with a positive difference.

Artifact tests: six, against an empirical null made of transcripts a CD4 T cell cannot make
(LYZ, S100A8/9, CD14, VCAN, FCN1, CST3, LST1, AIF1, TYROBP, FCER1G, CD68, MNDA, PPBP, PF4, HBB,
MS4A1, CD79A). Thresholds are read off that null rather than chosen, and the section is allowed
to exclude nothing: the calibration verdict decides. Verdict here: {msg}

Known answers, for calibration: CCR4 (mogamulizumab) is approved in CTCL, KIR3DL2 is lacutamab's
target, IL2RA is registrational; TNFRSF8/CD30 is absent from this object. CD7 and DPP4/CD26 loss
is the Sezary immunophenotype and must come out negative.

Scope limits that apply to every number: the off-tumour axis covers T and NK arms ONLY -- the
non-T blood compartments are not in the screen table, so a myeloid or HSPC hit, the kind that
causes cytopenia, is invisible here. Nothing below has been seen as protein; notebook 45 does
that for 17 Sezary donors with CITE-seq.
"""

_sec = [
    {"name": "§0b Cohort", "what": "the independent donor set and its clinical fields.",
     "how": "curated clin_* columns only; no expression used.",
     "table": don[don["cohort_keep"]][[c for c in ("study", "dx", "dx2", "is_cultured",
                                                   "clone_shared", "sex", "age", "n_cells",
                                                   "burden") if c in don]],
     "file": "tables/tgt43_blood_cohort.csv", "max_rows": 100},
    {"name": "§0b Dropped keys", "what": "the keys removed, with the reason.",
     "how": "ex vivo drug-culture arm, or the same person under another key.",
     "table": don.loc[~don["cohort_keep"], ["study", "dx", "clin_ex_vivo", "person", "n_cells",
                                            "drop_reason"]], "max_rows": 60},
    {"name": "§1 Arms", "what": "donors, cells and studies per arm after the cohort fix.",
     "how": "groups of >= 20 cells.", "table": arms},
    {"name": "§1 Contamination load per arm",
     "what": "how dirty each arm is -- the asymmetry a contamination artifact would need.",
     "how": "median over the arm's donors.",
     "table": meta.groupby("arm")[["frac_lineage_pos", "mean_myeloid_score",
                                   "median_total_counts", "median_doublet_score"]].median()},
    {"name": "§2a Contamination null",
     "what": "what pure contamination scores in this cohort, per statistic.",
     "how": "the same tests applied to the lineage-control genes; thresholds come from the p90 "
            "row.", "table": band},
    {"name": "§2a Calibration verdict",
     "what": "whether the differential-contamination mechanism operates here at all.",
     "how": "the controls must themselves be up in the malignant arm; if they are not, no gene "
            "is excluded.", "table": {"null_calibrated": ok, "verdict": msg}},
    {"name": "§2b Artifact verdict, per gene",
     "what": "the six tests for every gene up at q < 0.05.",
     "how": "retained_* = share of the effect surviving trimming / winsorising / dropping dirty "
            "cells / depth matching; *_excess = prevalence-matched residual against the null; "
            "rho_burden = Spearman of the per-donor effect against that donor's "
            "malignant:reactive ratio. Two failures = artifact, one = suspect.",
     "table": ev[(ev["q"] < 0.05) & (ev["log2fc_paired"] > 0)].sort_values(
         "log2fc_paired", ascending=False)[[c for c in _fc if c in ev]],
     "file": "tables/tgt43_artifact_evidence.csv", "max_rows": 80},
    {"name": "§2c Three-arm detection",
     "what": "median detection of the disputed genes and the controls in healthy, reactive and "
             "malignant CD4.",
     "how": "median over donors of the share of cells with >= 1 UMI. Contamination has no "
            "reason to spare healthy CD4.", "table": det},
    {"name": "§3 Candidates passing every gate",
     "what": "the genes selectively up on malignant CD4 that survived §2.",
     "how": "clean, q < 0.05, paired log2FC > 0.5, detected in >= 50% of donors, not a loss "
            "marker or lineage control.",
     "table": up[[c for c in _c if c in up]]},
    {"name": "§3 Loss markers",
     "what": "the Sezary loss phenotype, as the check that cleaning did not flatten real "
             "biology.", "how": "same paired test.",
     "table": core[core["gene"].isin(ST.LOSS_MARKERS)][["gene", "log2fc_paired",
                                                        "frac_donors_up", "q",
                                                        "artifact_flag"]]},
    {"name": "§3 FD Per-gene summary of the donor x gene matrix",
     "what": "how consistent each leading antigen is across patients.",
     "how": "per-donor log2FC; frac_up_1 is the share of donors where the antigen is at least "
            "doubled.", "table": _sum},
    {"name": "§3 FE Eligibility",
     "what": "the fraction of patients in whom at least X% of malignant CD4 cells are positive.",
     "how": "transcript detection per donor, thresholded.", "table": elig,
     "file": "tables/tgt43_eligibility_blood.csv"},
    {"name": "§4 Off-tumour, T and NK arms only",
     "what": "the worst non-target T arm and the healthy-CD4 burden for each candidate.",
     "how": "share of cells detected. Monocytes, DC, B, plasma, platelets and HSPC are NOT in "
            "this table.", "table": _o},
    {"name": "§5 Trafficking modules per arm",
     "what": "the state of the machinery that puts a protein on the surface.",
     "how": "per-module z across donors, then the arm mean. Context, not a correction.",
     "table": arm_mod},
    {"name": "§6 Escape risk",
     "what": "how much each candidate varies between patients and within one patient's clusters.",
     "how": "SD of detection across donors, and across a donor's own clusters.",
     "table": esc[esc["gene"].isin(SHORT_GENES)]},
    {"name": "§7 Control check",
     "what": "where the antigens with a known answer sit in the ranking.",
     "how": "rank by paired log2FC over all ranked genes. TNFRSF8 is not testable: it is absent "
            "from the HVG object.", "table": hit},
    {"name": "§7 Shortlist with the plausibility layer",
     "what": "the final candidate table, with surfaceome evidence beside the statistics.",
     "how": "surface_confident / n_tm / surfy_class come from CSPA + Surfy, not from this "
            "cohort.", "table": short[[c for c in show if c in short]],
     "file": "tables/tgt43_shortlist_blood.csv"},
]
if "both_calls" in dir():
    _sec.append({"name": "§2d TCR-call control",
                 "what": "the same screen with the arms built from the expanded TCR clone "
                         "instead of the Leiden cluster.",
                 "how": "an effect that is an artifact of how the clusters were drawn cannot "
                        "survive a call made from the TCR.",
                 "table": both_calls[both_calls["gene"].isin(
                     TQ.SUSPECTS + ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS)],
                 "file": "tables/tgt43_call_sensitivity.csv"})
if "up_both" in dir():
    _sec.append({"name": "§8 Selective in skin AND blood",
                 "what": "antigens that separate tumour from reactive CD4 in both compartments.",
                 "how": "q < 0.05 and log2FC > 0.5 on both sides, after both screens were "
                        "cohort-fixed and artifact-tested.",
                 "table": up_both[[c for c in cols if c in up_both]],
                 "file": "tables/tgt43_shared_skin_blood.csv"})
TF.llm_export("43", "Surface-target screen, blood", _ctx, _sec)
```

```
======================== COPY FROM HERE ========================

# 43 — Surface-target screen, blood

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**, from blood_T_annotated.h5ad
(10,000 genes in the object -- an HVG SUBSET, so a gene absent from the
screen is not necessarily unexpressed; CD4 and TNFRSF8/CD30 are both missing for this reason).

Cohort: 131 donor keys reduced to 75 independent keys by
dropping the geskin2026 ex vivo drug-culture wells and duplicate persons, and by moving five
healthy donors that the atlas labelled SS out of the reactive arm. Diagnosis from the curated
`clin_dx`. 52 donors carry both a malignant and a reactive CD4 arm.

Arms: malignancy call `mal_cluster`. Blood in Sezary syndrome is ~92% malignant
CD4, so the reactive arm is small in many donors. The CD8 arm is a CD8+NK mixture (nb31 labelled
on CD8A) and any statement about it is a statement about a mixture.

Headline test: per gene, the within-donor difference of mean log1p CP10K between the malignant
and reactive arms, paired Wilcoxon over donors, BH-adjusted (`q`). `log2fc_paired` is the median
per-donor log2 ratio; `frac_donors_up` the share of donors with a positive difference.

Artifact tests: six, against an empirical null made of transcripts a CD4 T cell cannot make
(LYZ, S100A8/9, CD14, VCAN, FCN1, CST3, LST1, AIF1, TYROBP, FCER1G, CD68, MNDA, PPBP, PF4, HBB,
MS4A1, CD79A). Thresholds are read off that null rather than chosen, and the section is allowed
to exclude nothing: the calibration verdict decides. Verdict here: 18 lineage controls: median log2FC +0.00, median frac_donors_up 0.39 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.

Known answers, for calibration: CCR4 (mogamulizumab) is approved in CTCL, KIR3DL2 is lacutamab's
target, IL2RA is registrational; TNFRSF8/CD30 is absent from this object. CD7 and DPP4/CD26 loss
is the Sezary immunophenotype and must come out negative.

Scope limits that apply to every number: the off-tumour axis covers T and NK arms ONLY -- the
non-T blood compartments are not in the screen table, so a myeloid or HSPC hit, the kind that
causes cytopenia, is invisible here. Nothing below has been seen as protein; notebook 45 does
that for 17 Sezary donors with CITE-seq.

## §0b Cohort

**What:** the independent donor set and its clinical fields.

**How:** curated clin_* columns only; no expression used.

**Full table on disk:** `tables/tgt43_blood_cohort.csv`

```csv
donor,study,dx,dx2,is_cultured,clone_shared,sex,age,n_cells,burden
B1__P4,borcherding2019,SS,SS,False,False,M,61,6260,0.982
B2__CTCL2,borcherding2023,SS,SS,False,False,F,71,31441,
B2__CTCL3,borcherding2023,SS,SS,False,False,M,67,2015,
B2__CTCL4,borcherding2023,SS,SS,False,False,F,84,305,0.577
B2__CTCL5,borcherding2023,SS,SS,False,False,M,73,1040,0.739
B2__CTCL6,borcherding2023,SS,SS,False,False,F,69,1829,0.495
B4__PF46_HTO5,geskin2026,HC,HC,True,False,F,,49,
B4__PF47_HTO1,geskin2026,HC,HC,True,False,M,,3116,
B4__PF48_HTO1,geskin2026,HC,HC,True,False,M,,1922,
B4__PF49_HTO1,geskin2026,HC,HC,True,False,F,,2784,
B4__PF50_HTO1,geskin2026,HC,HC,True,False,M,,5711,
B4__SZ30_HTO6,geskin2026,MF/SS,unresolved,True,False,M,70,2578,0.646
B4__SZ32_HTO1,geskin2026,SS,SS,True,False,M,32,2582,0.142
B4__SZ33_HTO5,geskin2026,SS,SS,True,False,M,,261,
B4__SZ38_HTO1,geskin2026,SS,SS,True,False,M,,4882,0.994
B4__SZ39_HTO1,geskin2026,SS,SS,True,F
... [37572 more chars truncated]
```
