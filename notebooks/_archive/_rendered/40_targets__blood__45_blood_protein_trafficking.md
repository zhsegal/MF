# 40_targets/blood/45_blood_protein_trafficking.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/blood/45_blood_protein_trafficking.ipynb` and is never run._

# 45 — Blood CITE-seq: measured surface protein, and what explains it

The one thing no other part of this project can do: **check the transcript ranking against
measured surface protein on the same cells.** `blood_T_annotated.h5ad` carries a 128-marker
CITE-seq panel in `obsm["X_adt"]` — real counts, not imputation — on the 61,215 buus2025 cells.

Three questions, in order of how much they are worth.

**1. Which markers were actually measured?** This has to come first, and the previous version of
this analysis did not do it. `ADT_CD158k` — the KIR3DL2 clone, the lacutamab target, the top hit
of the whole blood screen — is positive in **0.01 % of cells**, the same rate as the `ADT_APC`
isotype control. Its protein delta came out as exactly 0.000 across 12 donors, and that was then
counted as protein *contradicting* the transcript. It is not a contradiction; it is a clone that
did not bind. Every marker is therefore gated against the four isotype controls before it enters
any agreement rate.

**2. Does protein reproduce the transcript direction?** On the markers that pass the gate. This
number is the honest estimate of what a transcript ranking is worth for surface biology, in this
disease, on this platform — and the markers that disagree are worth more than the ones that
agree, because they name the candidates that need flow before anyone believes them.

**3. When protein and transcript disagree, is the trafficking machinery the reason?** A
surface protein level is the end of a pipeline: transcription, translocation, folding,
glycosylation, export, endocytosis, recycling, shedding. If a marker's surface level is better
predicted by the cell's ER/Golgi/sheddase programme than by its own mRNA, the transcript was
never going to rank it correctly. The modules are `surface_targets.TRAFFIC_MODULES`; the model
is fitted on single cells with the donor carried as a random effect, and replicated on the
17-donor malignant-minus-reactive deltas.

**Scope, which applies to every number below.** One study (buus2025), 17 donors, **15 Sézary and
2 MF** by `clin_dx` — not all Sézary, as the earlier version of this section said. No healthy
donor and no skin sample carries ADT, so this validates malignant-vs-reactive *inside* CTCL
patients and says nothing about a healthy baseline.

**Input.** `bash jobs/run_adt_cells.sh`, which writes the per-cell table this notebook reads.

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
import target_figs as TF
import target_het as TH

TISSUE = "blood"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt45_"
MALIG, REACT = "CD4_malignant", "CD4_reactive"
MIN_DONORS = 5

ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX
TF.rc()
pd.set_option("display.width", 240)

adt_man = json.loads((TARGETS / "blood_adt_manifest.json").read_text())
cells = pd.read_parquet(TARGETS / "blood_adt_cells.parquet").set_index("cell_id")
A = pd.read_parquet(TARGETS / "blood_adt_counts.parquet").set_index("cell_id")
A = A.loc[cells.index]
PANEL = list(A.columns)
CTRL = [c for c in ST.ADT_CONTROLS if c in PANEL]
MARKERS = [c for c in PANEL if c not in CTRL]

don = TC.blood_donor_table()
long = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet"), don)
sel = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
prev = TS.prevalence(long, MALIG)

# The focus set for the trafficking model: antigens with a clinical story, the Sezary loss
# markers, and KLRD1/CD94 -- each paired below with the module its biology predicts.
FOCUS = ["CCR4", "KIR3DL2", "CD70", "IL2RA", "CD7", "DPP4", "CD5", "CD28", "CCR7",
         "TNFRSF1B", "KLRD1"]
MECHANISM = {
    "IL2RA": "shedding", "TNFRSF1B": "shedding", "CD27": "shedding",
    "CCR4": "endocytosis", "KIR3DL2": "endocytosis", "CCR7": "recycling",
    "CXCR4": "recycling", "SELPLG": "glycosylation", "CD7": "er_folding_qc",
}

print("NB_DIR =", NB_DIR)
print(f"{len(cells):,} ADT cells | {len(PANEL)} panel entries "
      f"({len(MARKERS)} markers + {len(CTRL)} isotype controls: {CTRL})")
print("\narm x diagnosis among the ADT cells:")
print(pd.crosstab(cells["arm"], cells["dx"]).to_string())
print(f"\ndonors: {cells['donor'].nunique()} "
      f"({don.loc[cells['donor'].unique(), 'dx'].value_counts().to_dict()})")
print("*** 15 Sezary and 2 MF by clin_dx -- the earlier version of this analysis called the "
      "ADT cohort 'all Sezary', which is the atlas disease label, not the curated one. ***")
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
61,215 ADT cells | 128 panel entries (124 markers + 4 isotype controls: ['ADT_PE', 'ADT_FITC', 'ADT_APC', 'ADT_Biotin'])

arm x diagnosis among the ADT cells:
dx               MF     SS
arm                       
CD4_malignant  3563  42430
CD4_reactive   2586   3069
CD8            1277   8290

donors: 17 ({'SS': 15, 'MF': 2})
*** 15 Sezary and 2 MF by clin_dx -- the earlier version of this analysis called the ADT cohort 'all Sezary', which is the atlas disease label, not the curated one. ***
```

## §1 — Which markers were measured at all

An antibody-derived tag is a count like any other: most of what a cell shows for a marker that
did not bind is background. The panel carries four isotype controls (`ADT_PE`, `ADT_FITC`,
`ADT_APC`, `ADT_Biotin`) whose whole purpose is to measure that background, so the gate is
written against them rather than against an arbitrary count threshold.

A marker passes if, in at least two thirds of donors, its mean `log1p` count exceeds that
donor's mean isotype `log1p` count by at least `MARGIN` — and it is detected above zero in at
least 1 % of cells. Both numbers are printed next to every marker, so the gate can be moved and
the consequence seen.

This matters more than it sounds. Any marker sitting at background has a protein delta of
almost exactly zero in every donor, and a near-zero delta next to a large transcript log2FC
reads as "protein contradicts transcript" when what it means is "no measurement".

```python  # cell 3
# ============================================================================
# §1  Isotype background, and the marker gate
# ============================================================================
MARGIN, MIN_FRAC_DONORS, MIN_DETECT = 0.10, 2 / 3, 0.01

L = np.log1p(A)
bg_cell = L[CTRL].mean(1) if CTRL else pd.Series(0.0, index=A.index)
donor = cells["donor"]

per_donor_marker = L.groupby(donor).mean()
per_donor_bg = bg_cell.groupby(donor).mean()
above = per_donor_marker.sub(per_donor_bg, axis=0) >= MARGIN

gate = pd.DataFrame({
    "frac_cells_pos": (A > 0).mean(),
    "mean_log1p": L.mean(),
    "mean_log1p_minus_bg": L.mean() - bg_cell.mean(),
    "frac_donors_above_bg": above.mean(),
    "is_control": [c in CTRL for c in A.columns],
})
gate["gene"] = [ST.ADT_TO_GENE.get(c) for c in gate.index]
gate["passes"] = (~gate["is_control"] & (gate["frac_donors_above_bg"] >= MIN_FRAC_DONORS)
                  & (gate["frac_cells_pos"] >= MIN_DETECT))
DETECTED = gate.index[gate["passes"]].tolist()

print(f"isotype controls, mean log1p per donor: "
      f"{per_donor_bg.min():.3f} - {per_donor_bg.max():.3f}")
print(f"\n{len(DETECTED)} of {len(MARKERS)} markers pass "
      f"(>= {MARGIN} above isotype in >= {MIN_FRAC_DONORS:.0%} of donors, "
      f"detected in >= {MIN_DETECT:.0%} of cells)")

print("\nFAILED markers -- these were previously counted as protein contradicting transcript:")
fail = gate[~gate["is_control"] & ~gate["passes"]].sort_values("frac_cells_pos")
print(fail[["gene", "frac_cells_pos", "mean_log1p", "mean_log1p_minus_bg",
            "frac_donors_above_bg"]]
      .to_string(float_format=lambda v: f"{v:.4g}"))

print("\nisotype controls, for scale:")
print(gate[gate["is_control"]][["frac_cells_pos", "mean_log1p"]]
      .to_string(float_format=lambda v: f"{v:.4g}"))

print("\nthe clinically validated antigens, and whether their clone worked:")
_pc = gate[gate["gene"].isin(ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS + FOCUS)]
print(_pc[["gene", "frac_cells_pos", "mean_log1p_minus_bg", "frac_donors_above_bg", "passes"]]
      .sort_values("frac_cells_pos", ascending=False)
      .to_string(float_format=lambda v: f"{v:.4g}"))
print("\nKIR3DL2 is the top transcript hit of the blood screen and the target of lacutamab. "
      "If ADT_CD158k is at isotype level here, this panel cannot speak about it either way.")

gate.to_csv(ST.tab("adt_gate"))
print("\nwrote", ST.tab("adt_gate"))
```

```
isotype controls, mean log1p per donor: 0.000 - 0.528

25 of 124 markers pass (>= 0.1 above isotype in >= 67% of donors, detected in >= 1% of cells)

FAILED markers -- these were previously counted as protein contradicting transcript:
                      gene  frac_cells_pos  mean_log1p  mean_log1p_minus_bg  frac_donors_above_bg
ADT_CD158k         KIR3DL2        0.000147   0.0002817              -0.0309                     0
ADT_CD217           IL17RA        0.002499    0.002316             -0.02887                     0
ADT_CD213a2        IL13RA2        0.002712    0.002744             -0.02844                     0
ADT_CD35               CR1        0.004754    0.004848             -0.02633                     0
ADT_CD213a1        IL13RA1        0.005946    0.005484              -0.0257                     0
ADT_CD8a              CD8A        0.006894      0.0883              0.05711                     0
ADT_CLA             SELPLG        0.007269    0.008257             -0.02293                     0
ADT_CD137L          TNFSF9        0.008544     0.01242             -0.01876                     0
ADT_CD14              CD14        0.008789     0.01057             -0.02062                     0
ADT_CD1c              CD1C        0.009589     0.01625             -0.01494                     0
ADT_CD23             FCER2        0.009654     0.01511             -0.01607                     0
ADT_CD360            IL21R         0.01075     0.02573             -0.00545                     0
ADT_CD193             CCR3         0.01235      0.0177             -0.01348               0.05882
ADT_CD1a              CD1A         0.01255     0.01632             -0.01486                     0
ADT_CD117              KIT         0.01446     0.01731             -0.01387                     0
ADT_CD84              CD84         0.01472     0.02045             -0.01074                     0
ADT_CD86              CD86         0.01477     0.02029             -0.01089                     0
ADT_CD56             NCAM1         0.01621     0.02746            -0.003727               0.05882
ADT_TCRgd             None         0.01723     0.02403            -0.007151               0.05882
ADT_CD274            CD274         0.01784     0.02143            -0.009748                     0
ADT_CD207            CD207         0.01944     0.02532            -0.005868               0.05882
ADT_CD294           PTGDR2         0.01964     0.02553             -0.00565               0.05882
ADT_CD11c            ITGAX         0.01998     0.02771            -0.003474               0.05882
ADT_CD152            CTLA4          0.0255     0.03348             0.002301                0.1176
ADT_LAP              TGFB1         0.02568      0.0493              0.01811                     0
ADT_KLRG1            KLRG1         0.02785     0.04304              0.01186                0.1176
ADT_CD11a_CD18        None         0.02821     0.04839              0.01721                     0
ADT_CD11b            ITGAM         0.02882     0.04022             0.009041                     0
ADT_CD185            CXCR5         0.02937     0.05191              0.02073               0.05882
ADT_CD163            CD163         0.03065     0.04446              0.01328               0.05882
ADT_CD20             MS4A1         0.03079     0.04018             0.008999                     0
ADT_CD99              CD99         0.03114      0.0359             0.004714                0.1176
ADT_CD138             SDC1         0.03115     0.04514              0.01396               0.05882
ADT_CD21               CR2         0.03288     0.04627              0.01509                     0
ADT_CD19              CD19         0.03333     0.04424              0.01305                     0
ADT_FceRIa          FCER1A         0.03584     0.05463              0.02345                0.1176
ADT_CD107a           LAMP1         0.03597      0.0454              0.01422               0.05882
ADT_CD101            CD101         0.0371
... [7792 more chars truncated]
```

## §2 — Normalisation, shown rather than assumed

Three ways to turn ADT counts into a value, because the choice changed the previous conclusion.

- **`clr`** — the standard: `log1p(x)` minus the cell's mean `log1p` over the non-control
  markers. Centred per cell, so it removes the cell's total tag capture.
- **`bg`** — `log1p(x)` minus the cell's mean `log1p` over the four isotype controls. Keeps the
  absolute scale and subtracts measured background instead of the panel mean.
- **`old`** — what the first version of this analysis used: `log1p(x / exp(mean(log1p(x))))`,
  a non-standard hybrid that divides counts by a geometric mean of `1 + counts` and then takes
  `log1p` again.

The three are compared on markers whose answer is known — CD7 and CD26/DPP4 must be **down** in
the Sézary clone, CD194/CCR4 **up** — and the primary is declared from that, not from taste.

The comparison is here because the first pass of this analysis reported malignant-minus-reactive
deltas of ~0.001 for most markers, and the normalisation was the obvious suspect. **It is not
the culprit**: all three give the same answer on the markers that were measured (CD7 ≈ −1.0,
CCR4 ≈ +0.8). The ~0.001 values belonged entirely to markers sitting at isotype background —
§1's problem, not §2's. CLR is kept as primary for being the standard and the most
interpretable, not because it rescued anything.

```python  # cell 5
# ============================================================================
# §2  Three normalisations, judged on the markers whose answer is known
# ============================================================================
Lm = L[MARKERS]
norms = {
    "clr": Lm.sub(Lm.mean(1), axis=0),
    "bg": Lm.sub(bg_cell, axis=0),
    "old": np.log1p(A[MARKERS].div(np.exp(np.log1p(A[MARKERS]).mean(1)), axis=0)),
}
arm = cells["arm"]
KNOWN = {"CD7": "down", "DPP4": "down", "CD2": "down", "CCR4": "up", "CD5": "down"}
g2a = {g: a for a, g in ST.ADT_TO_GENE.items() if g in KNOWN}

rows = []
for name, X in norms.items():
    d = X.assign(arm=arm.to_numpy(), donor=donor.to_numpy())
    pw = (d[d["arm"].isin([MALIG, REACT])]
          .groupby(["arm", "donor"]).mean(numeric_only=True))
    for gene, adt in g2a.items():
        if adt not in pw.columns:
            continue
        a_, b_ = pw.loc[MALIG, adt], pw.loc[REACT, adt]
        j = a_.index.intersection(b_.index)
        diff = (a_[j] - b_[j]).to_numpy()
        rows.append({"norm": name, "gene": gene, "expected": KNOWN[gene],
                     "delta": float(np.median(diff)), "n_donors": len(j),
                     "p": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
                     "correct_sign": np.sign(np.median(diff)) ==
                                     (1 if KNOWN[gene] == "up" else -1)})
cmpn = pd.DataFrame(rows)
print("malignant minus reactive, per normalisation, on markers whose direction is known:")
print(cmpn.pivot_table(index="gene", columns="norm", values="delta")
      .join(pd.Series(KNOWN, name="expected"))
      .to_string(float_format=lambda v: f"{v:.4g}"))
print("\ncorrect direction, out of", cmpn["gene"].nunique(), "markers:")
print(cmpn.groupby("norm")["correct_sign"].sum().to_string())
print("\nspread of the deltas, which is what makes a normalisation usable at all:")
print(cmpn.groupby("norm")["delta"].agg(["min", "max"]).round(4).to_string())

NORM = "clr"
P = norms[NORM]
print(f"\nprimary normalisation: {NORM}. Note that all three normalisations agree on the "
      "markers that were measured, so the ~0.001 deltas the first pass reported were not a "
      "normalisation problem -- they were markers at isotype background, which is what the "
      "gate above is for.")
cmpn.to_csv(ST.tab("normalisation_check"), index=False)
print("wrote", ST.tab("normalisation_check"))
```

```
malignant minus reactive, per normalisation, on markers whose direction is known:
          bg     clr     old expected
gene                                 
CCR4  0.8276  0.8234  0.7314       up
CD2  -0.7489 -0.7303 -0.7023     down
CD5   0.1906  0.2288  0.1972     down
CD7   -1.034  -1.028 -0.9373     down
DPP4 -0.6202 -0.6118 -0.5543     down

correct direction, out of 5 markers:
norm
bg     4
clr    4
old    4

spread of the deltas, which is what makes a normalisation usable at all:
         min     max
norm                
bg   -1.0337  0.8276
clr  -1.0276  0.8234
old  -0.9373  0.7314

primary normalisation: clr. Note that all three normalisations agree on the markers that were measured, so the ~0.001 deltas the first pass reported were not a normalisation problem -- they were markers at isotype background, which is what the gate above is for.
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_normalisation_check.csv
```

## §3 — Protein against transcript, on markers that were measured

Same paired design as the screen: for each marker, the malignant-minus-reactive difference
within each donor, median across donors. The protein side is CLR; the transcript side is the
same `log2fc_paired` the screen ranks on, recomputed here on the 17 ADT donors only so the two
columns describe the same patients.

Two numbers to read: the correlation across markers, and the list of markers that disagree.

```python  # cell 7
# ============================================================================
# §3  Paired protein delta beside the paired transcript delta
# ============================================================================
_pa = P.assign(arm=arm.to_numpy(), donor=donor.to_numpy())
pw = (_pa[_pa["arm"].isin([MALIG, REACT])]
      .groupby(["arm", "donor"]).mean(numeric_only=True))
ADT_DONORS = sorted(set(pw.loc[MALIG].index) & set(pw.loc[REACT].index))
print(f"{len(ADT_DONORS)} donors with both arms measured")

rows = []
for adt in MARKERS:
    a_, b_ = pw.loc[MALIG, adt].reindex(ADT_DONORS), pw.loc[REACT, adt].reindex(ADT_DONORS)
    diff = (a_ - b_).dropna().to_numpy()
    if len(diff) < 3:
        continue
    rows.append({"adt": adt, "gene": ST.ADT_TO_GENE.get(adt),
                 "prot_delta": float(np.median(diff)), "n_donors_prot": len(diff),
                 "p_prot": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
                 "detected": adt in DETECTED})
prot = pd.DataFrame(rows).dropna(subset=["gene"])
prot["q_prot"] = multipletests(prot["p_prot"], method="fdr_bh")[1]

# The transcript side on the SAME donors, so the two columns are about the same patients.
sel_adt = TS.paired_selectivity(long[long["donor"].isin(ADT_DONORS)], MALIG, REACT,
                                value="mean_all", min_donors=3)
cmp_ = (prot.merge(sel_adt[["gene", "log2fc_paired", "frac_donors_up", "q"]], on="gene")
            .merge(ST.annotate(prot["gene"])[["gene", "cd_name", "stage", "surface_caveat"]],
                   on="gene", how="left"))
print(f"{len(cmp_)} markers with both a protein and a transcript estimate; "
      f"{int(cmp_['detected'].sum())} of them above isotype background")

flat = cmp_["log2fc_paired"].eq(0)
print(f"{int(flat.sum())} have a transcript log2FC of exactly 0 (flat in the HVG object) and "
      "carry no direction to agree or disagree with")


def agreement(d, label):
    d = d[~d["log2fc_paired"].eq(0)]
    if len(d) < 4:
        print(f"{label}: too few markers"); return
    r = stats.spearmanr(d["prot_delta"], d["log2fc_paired"])
    ag = int(np.sign(d["prot_delta"]).eq(np.sign(d["log2fc_paired"])).sum())
    print(f"{label:38s} rho = {r.statistic:+.2f} (p = {r.pvalue:.1g})  "
          f"same direction {ag}/{len(d)} ({ag / len(d):.0%})")
    return r.statistic, ag, len(d)


print()
all_r = agreement(cmp_, "every mapped marker")
det_r = agreement(cmp_[cmp_["detected"]], "markers above isotype background")
und_r = agreement(cmp_[~cmp_["detected"]], "markers at background (meaningless)")
print("\nThe first line is the number the previous version of this notebook reported; the "
      "second is the one that means anything. An undetected clone has a delta of ~0 and a "
      "coin-flip sign, so including it drags the agreement rate towards 50% and makes the "
      "transcript look worse than it is.")

print("\ngate sensitivity -- the agreement rate at three strictnesses, so the gate is not "
      "doing the work on its own:")
for marg, frac in ((0.05, 0.5), (0.10, 2 / 3), (0.20, 0.8)):
    keep = set(gate.index[~gate["is_control"]
                          & (per_donor_marker.sub(per_donor_bg, axis=0) >= marg).mean()
                          .reindex(gate.index).ge(frac).fillna(False)
                          & (gate["frac_cells_pos"] >= MIN_DETECT)])
    agreement(cmp_[cmp_["adt"].isin(keep)],
              f"  margin {marg}, {frac:.0%} of donors ({len(keep)} markers)")

print("\n" + "=" * 78)
print("markers where MEASURED protein and transcript disagree -- the transcript misleads here")
print("=" * 78)
dis = cmp_[cmp_["detected"] & ~cmp_["log2fc_paired"].eq(0)]
dis = dis[np.sign(dis["prot_delta"]) != np.sign(dis["log2fc_paired"])]
dis = dis.reindex(dis["log2fc_paired"].abs().sort_values(ascending=False).index)
print(dis[["gene", "cd_name", "prot_delta", "q_prot", "log2fc_paired", "q",
           "n_donors_prot", "surface_caveat"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\nand the validated antigens, in full:")
ctrl = cmp_[cmp_["gene"].isin(ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS + FOCUS)]
print(ctrl[["gene", "cd_name", "detected", "prot_delta", "q_prot", "n_donors_prot",
            "log2fc_paired", "q", "stage"]]
      .sort_values("prot_delta", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
cmp_.to_csv(ST.tab("protein_vs_transcript"), index=False)
print("\nwrote", ST.tab("protein_vs_transcript"))
```

```
17 donors with both arms measured

100 markers with both a protein and a transcript estimate; 21 of them above isotype background
25 have a transcript log2FC of exactly 0 (flat in the HVG object) and carry no direction to agree or disagree with

every mapped marker                    rho = +0.61 (p = 5e-09)  same direction 50/75 (67%)
markers above isotype background       rho = +0.79 (p = 2e-05)  same direction 16/21 (76%)
markers at background (meaningless)    rho = +0.55 (p = 2e-05)  same direction 34/54 (63%)

The first line is the number the previous version of this notebook reported; the second is the one that means anything. An undetected clone has a delta of ~0 and a coin-flip sign, so including it drags the agreement rate towards 50% and makes the transcript look worse than it is.

gate sensitivity -- the agreement rate at three strictnesses, so the gate is not doing the work on its own:
  margin 0.05, 50% of donors (46 markers) rho = +0.75 (p = 8e-08)  same direction 26/38 (68%)
  margin 0.1, 67% of donors (25 markers) rho = +0.79 (p = 2e-05)  same direction 16/21 (76%)
  margin 0.2, 80% of donors (10 markers) rho = +0.85 (p = 0.004)  same direction 8/9 (89%)

==============================================================================
markers where MEASURED protein and transcript disagree -- the transcript misleads here
==============================================================================
 gene cd_name  prot_delta  q_prot  log2fc_paired      q  n_donors_prot surface_caveat
 CCR7             -0.0493  0.0368          0.295  0.857             17               
 IL4R             -0.0189    0.31          0.146  0.424             17               
ITGAL             -0.0244       1          0.123  0.424             14               
PTPRC              -0.214   0.212          0.104 0.0998             17               
  CD5     CD5       0.229   0.272        -0.0778  0.643             17               

and the validated antigens, in full:
    gene cd_name  detected  prot_delta   q_prot  n_donors_prot  log2fc_paired       q              stage
    CD27              True        0.95  0.00318             17          0.482  0.0195                   
    CCR4    CCR4      True       0.823   0.0585             17           1.19   0.121           approved
    CD28              True       0.655  0.00259             17          0.699 0.00638                   
     CD5     CD5      True       0.229    0.272             17        -0.0778   0.643          phase 1/2
TNFRSF1B   TNFR2     False      0.0698    0.192             14          0.847    0.31         phase 1/2a
    CD70    CD70     False        0.03    0.296             17           1.73  0.0203            phase 1
 KIR3DL2  CD158k     False      0.0284    0.192             12           2.73 0.00638            phase 2
   IL2RA    CD25     False      0.0221    0.192             17         0.0111   0.584 approved (2024-08)
    CCR7              True     -0.0493   0.0368             17          0.295   0.857                   
    DPP4              True      -0.612 0.000726             17          -1.87 0.00638                   
     CD2              True       -0.73   0.0282             17         -0.802   0.022                   
     CD7     CD7      True       -1.03   0.0071             17          -1.78  0.0251          phase 1/2

wrote /home/projects/nyosef/zvise/MF/tables/tgt45_protein_vs_transcript.csv
```

```python  # cell 8
# ============================================================================
# §3b F1  Protein against transcript
# ============================================================================
AGREE, DISAGREE = TF.LINE_DOWN, TF.LINE_UP
# Red: the protein/RNA sign disagreements from §3, plus markers where one of the two arms is a
# weak, non-significant shift -- both are "one arm ns", the transcript or the protein alone.
ONE_ARM_NS = ["CCR7", "TIGIT", "ITGB1", "PDCD1"]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.axhline(0, lw=.5, c="k"); ax.axvline(0, lw=.5, c="k")
und = cmp_[~cmp_["detected"]]
det = cmp_[cmp_["detected"]].copy()
det["off"] = det["adt"].isin(dis["adt"]) | det["gene"].isin(ONE_ARM_NS)
_dup = det["gene"].duplicated(keep=False)
det["lab"] = np.where(_dup, det["gene"] + " (" + det["adt"].str.replace("ADT_", "") + ")",
                      det["gene"])
ax.scatter(und["log2fc_paired"], und["prot_delta"], s=16, facecolor="none",
           edgecolor="#bdbdbd", linewidths=.6, label="at isotype background")
for off_, col, lab in ((False, AGREE, "protein agrees with RNA"),
                       (True, DISAGREE, f"one arm ns ({int(det['off'].sum())})")):
    d_ = det[det["off"].eq(off_)]
    ax.scatter(d_["log2fc_paired"], d_["prot_delta"], s=22, c=col, linewidths=0, label=lab,
               zorder=3)
_txt = [ax.text(r_["log2fc_paired"], r_["prot_delta"], r_["lab"], fontsize=5.5,
                color=DISAGREE if r_["off"] else "#222222")
        for _, r_ in det.iterrows()]
try:
    from adjustText import adjust_text
    adjust_text(_txt, ax=ax, expand=(1.2, 1.4),
                arrowprops=dict(arrowstyle="-", lw=0.3, color="#999999"))
except ImportError:
    pass
ax.set_xlabel(f"transcript log2FC, malignant vs reactive ({len(ADT_DONORS)} donors)")
ax.set_ylabel(f"CITE-seq protein delta ({NORM})")
ax.legend(fontsize=5.5, frameon=False, loc="lower right")
_r = det_r[0] if det_r else np.nan
ax.set_title(f"Protein vs transcript (ρ = {_r:+.2f}, {det_r[2] if det_r else 0} markers)",
             loc="left", fontsize=7.5)
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F1_protein_vs_transcript", caption_text=(
    f"{len(ADT_DONORS)} buus2025 donors. Open: at isotype background, excluded from ρ."))
TF.save_source(cmp_, "F1_protein_vs_transcript_source")
TF.logfig("F1_protein_vs_transcript", "protein vs transcript, gated", len(cmp_))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_F1_protein_vs_transcript.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_F1_protein_vs_transcript.pdf

<Figure size 540x380 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_F1_protein_vs_transcript_source.csv
```

## §4 — Is the surface level explained by the trafficking machinery?

A transcript is the first step of a long pipeline, and the question is how much of the rest the
cell's own machinery accounts for. For each measured marker, inside the malignant CD4 cells of
the 17 ADT donors:

```
protein_CLR ~ own mRNA + 11 trafficking module scores + RNA depth + total ADT depth [+ donor]
```

Two fits, because they answer slightly different questions.

- **Donor-demeaned OLS** (every variable centred inside its donor). This removes every
  between-patient difference by construction, so what is left is the within-patient,
  cell-to-cell question: among two malignant cells of the same patient, does the one with more
  ER/Golgi/sheddase programme carry more of this protein on its surface? It is also fast enough
  to run on the whole panel.
- **A mixed model** with donor as a random intercept, for the focus markers, as the formal
  version of the same thing.

The reported quantity is a **partial R²**: how much of the variance in surface protein the
trafficking modules explain *after* the marker's own mRNA and the library size have had their
turn. A marker where that exceeds the own-mRNA term is a marker whose transcript was never
going to rank it correctly.

**What this cannot show.** Module scores are averages over genes that are themselves in the
10,000-gene HVG subset — `endocytosis` survives with 3 genes, `er_to_golgi` with 5 — so a
module with few genes is a weak measurement, not an absent mechanism. The gene counts are
printed. And per-cell CLR is compositional: raising one marker lowers the rest a little, which
is why the library-size term is in every model.

```python  # cell 10
# ============================================================================
# §4a  The design, and the per-cell model on the whole measured panel
# ============================================================================
MODS = [c for c in cells.columns if c.startswith("traffic_")]
print("trafficking modules and the genes each kept after HVG selection:")
print(pd.Series(adt_man["traffic_module_genes_present"]).sort_values(ascending=False)
      .to_string())
print(f"\n{len(MODS)} module scores | "
      f"{sum(1 for m in MODS if adt_man['traffic_module_genes_present'].get(m[8:], 0) < 5)} "
      "of them rest on fewer than 5 genes and should be read as weak measurements")

use = cells["arm"].eq(MALIG)
print(f"\nmodel cells: {int(use.sum()):,} malignant CD4 cells from "
      f"{cells.loc[use, 'donor'].nunique()} donors")

# Two depth terms, not one. `log_umi` is the cell's RNA library; `log_adt` is its total tag
# capture. CLR centres each cell over the panel, so a cell that simply bound more antibody has
# every marker pulled towards its own mean -- without this term that compositional effect lands
# on whatever else varies with cell size, which is exactly the trafficking modules.
cells["log_adt"] = np.log1p(A[MARKERS].sum(1))
DEPTH = ["log_umi", "log_adt"]
base = pd.concat([cells.loc[use, MODS], cells.loc[use, DEPTH]], axis=1)
dcodes = cells.loc[use, "donor"]


def _demean(df, by):
    return df - df.groupby(by).transform("mean")


def partial_r2(y, Xa, Xb):
    """Share of the variance in `y` that `Xb` adds on top of `Xa`. Both already demeaned."""
    def _rss(X):
        X = np.column_stack([np.ones(len(X)), np.asarray(X, float)]) if X is not None \
            else np.ones((len(y), 1))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ beta) ** 2).sum())
    r0, ra = _rss(None), _rss(Xa)
    rab = _rss(np.column_stack([Xa, Xb])) if Xb is not None else ra
    return {"r2_a": 1 - ra / r0, "r2_ab": 1 - rab / r0,
            "partial_r2_b": (ra - rab) / ra if ra > 0 else np.nan}


rows = []
Xmod_d = _demean(base[MODS], dcodes).to_numpy(float)
Xumi_d = _demean(base[DEPTH], dcodes).to_numpy(float)
for adt in DETECTED:
    gene = ST.ADT_TO_GENE.get(adt)
    rna_col = f"rna_{gene}"
    if gene is None or rna_col not in cells.columns:
        continue
    y_raw = P.loc[use, adt]
    y = _demean(y_raw, dcodes).to_numpy(float)
    if not np.isfinite(y).all() or y.std() == 0:
        continue
    x_rna = _demean(cells.loc[use, [rna_col]], dcodes).to_numpy(float)
    # Incremental R2, each a share of the SAME total variance, so the three stack honestly:
    #   library size -> + the marker's own mRNA -> + the 11 trafficking modules.
    rna_step = partial_r2(y, Xumi_d, x_rna)
    tr = partial_r2(y, np.column_stack([Xumi_d, x_rna]), Xmod_d)
    r2_umi = rna_step["r2_a"]
    inc_rna = rna_step["r2_ab"] - rna_step["r2_a"]
    inc_tr = tr["r2_ab"] - tr["r2_a"]
    # between-donor share, from the variance the demeaning removed
    v_tot = float(np.var(y_raw.to_numpy(float)))
    v_within = float(np.var(y))
    rows.append({
        "adt": adt, "gene": gene,
        "r2_umi": r2_umi, "inc_r2_own_rna": inc_rna, "inc_r2_traffic": inc_tr,
        "r2_full": tr["r2_ab"],
        "donor_share": 1 - v_within / v_tot if v_tot > 0 else np.nan,
        "traffic_beats_own": inc_tr > inc_rna,
        "n_cells": int(use.sum()),
    })
vp = pd.DataFrame(rows).sort_values("inc_r2_traffic", ascending=False)
# Merge on the ADT, not the gene: PTPRC alone has three clones on this panel (CD45, CD45RA,
# CD45RO) and merging on the symbol would duplicate every one of its rows.
vp = vp.merge(cmp_[["adt", "prot_delta", "log2fc_paired", "detected"]], on="adt", how="left")
_dup = vp["gene"].duplicated(keep=False)
vp["label"] = np.where(_dup, vp["gene"] + " (" + vp["adt"].str.replace("ADT_", "") + ")",
                       vp["gene"])

print(f"\n{len(vp)} measured markers modelled. Trafficking explains more than the marker's "
      f"own mRNA for {int(vp['traffic_beats_own'].sum())} of them.\n")
_c = ["label", "r2_umi", "inc_r2_own_rna", "inc_r2_traffic", "r2_full",
      "donor_share", "traffic_beats_own", "prot_delta", "log2fc_paired"]
print(vp[_c].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
vp.to_csv(ST.tab("variance_partition"), index=False)
print("\nwrote", ST.tab("variance_partition"))
```

```
trafficking modules and the genes each kept after HVG selection:
glycosylation             14
shedding                  10
er_folding_qc              7
recycling                  6
er_translocation           6
er_to_golgi                5
retrograde                 5
snare_fusion               5
ubiquitin_proteostasis     4
escrt_exosome              4
endocytosis                3

11 module scores | 3 of them rest on fewer than 5 genes and should be read as weak measurements

model cells: 45,993 malignant CD4 cells from 17 donors


18 measured markers modelled. Trafficking explains more than the marker's own mRNA for 8 of them.

         label  r2_umi  inc_r2_own_rna  inc_r2_traffic  r2_full  donor_share  traffic_beats_own  prot_delta  log2fc_paired
         TIGIT  0.1028        0.001159         0.01006    0.114       0.1069               True     0.03213          1.969
          ICOS 0.07183        0.004531        0.009757  0.08611         0.15               True    -0.09279        -0.2108
          SELL  0.2558          0.1053         0.00837   0.3695       0.4765              False      0.3737         0.5218
          CCR4   0.171         0.03636        0.008124   0.2154       0.5513              False      0.8234          1.185
          IL4R  0.1222       7.718e-05        0.006385   0.1287      0.08912               True    -0.01894         0.1457
         PDCD1  0.1219        0.007574        0.006275   0.1358       0.3896              False      0.2693          1.216
          CCR7 0.09328       0.0001837        0.004348  0.09781      0.06363               True    -0.04927         0.2952
PTPRC (CD45RO) 0.09817        0.000745        0.003435   0.1024       0.3472               True       0.049         0.1038
          DPP4 0.05706        0.002561        0.003338  0.06296       0.1982               True     -0.6118         -1.869
          CD28   0.284        0.003172        0.002658   0.2898       0.4797              False      0.6546         0.6993
           CD5  0.1655        0.005606        0.001844    0.173       0.4421              False      0.2288       -0.07782
           CD2  0.1325         0.02073        0.001671   0.1549        0.766              False     -0.7303        -0.8019
          CD27  0.3584        0.009418        0.001598   0.3694       0.1821              False      0.9505         0.4818
           CD7  0.0416          0.1145        0.001446   0.1575       0.6222              False      -1.028         -1.777
PTPRC (CD45RA) 0.00744       0.0001307        0.001238 0.008809       0.2536               True      -0.214         0.1038
       KIR2DL3  0.1131        0.003074         0.00116   0.1174      0.02122              False     0.02135        0.03163
          IL7R  0.2203         0.01404        0.001085   0.2354       0.3383              False     -0.2282         -0.404
          CD3E  0.1638       8.965e-07        0.000686   0.1645       0.2424               True     -0.4019        -0.1733

wrote /home/projects/nyosef/zvise/MF/tables/tgt45_variance_partition.csv
```

```python  # cell 11
# ============================================================================
# §4b  Which module, for the focus markers, and does it match the expected mechanism
# ============================================================================
rows = []
for adt in DETECTED:
    gene = ST.ADT_TO_GENE.get(adt)
    rna_col = f"rna_{gene}"
    if gene is None or gene not in FOCUS or rna_col not in cells.columns:
        continue
    y = _demean(P.loc[use, adt], dcodes).to_numpy(float)
    x_rna = _demean(cells.loc[use, [rna_col]], dcodes).to_numpy(float)
    Xa = np.column_stack([Xumi_d, x_rna])
    for i, m in enumerate(MODS):
        pr = partial_r2(y, Xa, Xmod_d[:, [i]])
        rows.append({"gene": gene, "module": m[8:],
                     "partial_r2": pr["r2_ab"] - pr["r2_a"],
                     "n_module_genes": adt_man["traffic_module_genes_present"].get(m[8:], 0),
                     "expected": MECHANISM.get(gene) == m[8:]})
mod_tab = pd.DataFrame(rows)
if len(mod_tab):
    piv = mod_tab.pivot_table(index="gene", columns="module", values="partial_r2")
    print("incremental R2 of each module, added on top of the marker's own mRNA and "
          "library size:")
    print(piv.round(4).to_string())
    print("\nbest module per focus marker, and whether it is the one its biology predicts:")
    best = (mod_tab.sort_values("partial_r2", ascending=False)
            .groupby("gene").head(1)
            .merge(pd.Series(MECHANISM, name="predicted").rename_axis("gene").reset_index(),
                   on="gene", how="left"))
    best = best.merge(vp.drop_duplicates("gene")[["gene", "inc_r2_own_rna",
                                                  "traffic_beats_own"]],
                      on="gene", how="left")
    print(best[["gene", "module", "partial_r2", "n_module_genes", "predicted",
                "inc_r2_own_rna", "traffic_beats_own"]]
          .sort_values("partial_r2", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if best["module"].value_counts().iloc[0] >= 3:
        _dom = best["module"].value_counts().index[0]
        print(f"\n*** `{_dom}` wins for {int(best['module'].value_counts().iloc[0])} of "
              f"{len(best)} focus markers. The 11 module scores are averages over overlapping "
              "biology and are strongly correlated with each other and with how much of "
              "anything a cell is making, so a single module winning repeatedly is evidence of "
              "one shared biosynthetic-activity axis, not of that specific pathway. Read the "
              "trafficking column as 'the cell's machinery' and not as a named mechanism. ***")
    print("\nThe `predicted` column is a prior, not a result: IL2RA and TNFRSF1B are shed "
          "(serum sIL-2R tracks CTCL burden), CCR4 and KIR3DL2 internalise on ligand binding "
          "-- which is what mogamulizumab and lacutamab depend on -- and CCR7 recycles. A "
          "marker whose best module matches its prior is the interesting case; one whose best "
          "module is a 3-gene score is probably reading noise.")
    mod_tab.to_csv(ST.tab("module_partial_r2"), index=False)
    print("\nwrote", ST.tab("module_partial_r2"))
else:
    print("no focus marker passed the gate with its gene present in the HVG var")
```

```
incremental R2 of each module, added on top of the marker's own mRNA and library size:
module  endocytosis  er_folding_qc  er_to_golgi  er_translocation  escrt_exosome  glycosylation  recycling  retrograde  shedding  snare_fusion  ubiquitin_proteostasis
gene                                                                                                                                                                  
CCR4         0.0008         0.0003       0.0000            0.0016         0.0001         0.0011     0.0001      0.0001    0.0002        0.0015                  0.0046
CCR7         0.0000         0.0000       0.0000            0.0006         0.0002         0.0011     0.0001      0.0001    0.0002        0.0007                  0.0025
CD28         0.0000         0.0002       0.0000            0.0002         0.0002         0.0005     0.0000      0.0002    0.0001        0.0003                  0.0013
CD5          0.0004         0.0005       0.0001            0.0000         0.0003         0.0002     0.0004      0.0000    0.0000        0.0000                  0.0000
CD7          0.0001         0.0000       0.0000            0.0000         0.0004         0.0000     0.0001      0.0002    0.0002        0.0000                  0.0003
DPP4         0.0000         0.0001       0.0000            0.0006         0.0003         0.0002     0.0001      0.0001    0.0001        0.0007                  0.0019

best module per focus marker, and whether it is the one its biology predicts:
gene                 module  partial_r2  n_module_genes     predicted  inc_r2_own_rna  traffic_beats_own
CCR4 ubiquitin_proteostasis    0.004629               4   endocytosis         0.03636              False
CCR7 ubiquitin_proteostasis    0.002528               4     recycling       0.0001837               True
DPP4 ubiquitin_proteostasis    0.001893               4           NaN        0.002561               True
CD28 ubiquitin_proteostasis    0.001283               4           NaN        0.003172              False
 CD5          er_folding_qc   0.0004946               7           NaN        0.005606              False
 CD7          escrt_exosome   0.0004252               4 er_folding_qc          0.1145              False

*** `ubiquitin_proteostasis` wins for 4 of 6 focus markers. The 11 module scores are averages over overlapping biology and are strongly correlated with each other and with how much of anything a cell is making, so a single module winning repeatedly is evidence of one shared biosynthetic-activity axis, not of that specific pathway. Read the trafficking column as 'the cell's machinery' and not as a named mechanism. ***

The `predicted` column is a prior, not a result: IL2RA and TNFRSF1B are shed (serum sIL-2R tracks CTCL burden), CCR4 and KIR3DL2 internalise on ligand binding -- which is what mogamulizumab and lacutamab depend on -- and CCR7 recycles. A marker whose best module matches its prior is the interesting case; one whose best module is a 3-gene score is probably reading noise.

wrote /home/projects/nyosef/zvise/MF/tables/tgt45_module_partial_r2.csv
```

```python  # cell 12
# ============================================================================
# §4c  The formal version: donor as a random effect, for the focus markers
# ============================================================================
# Donor-demeaning above is a fixed-effect model and throws the between-patient variance away.
# This is the same question with the donor carried as a random intercept, which keeps it. Only
# the focus markers, because each fit is a few seconds on 46k cells.
import statsmodels.formula.api as smf

rows = []
for adt in DETECTED:
    gene = ST.ADT_TO_GENE.get(adt)
    rna_col = f"rna_{gene}"
    if gene is None or gene not in FOCUS or rna_col not in cells.columns:
        continue
    d = pd.DataFrame({"y": P.loc[use, adt].to_numpy(),
                      "rna": cells.loc[use, rna_col].to_numpy(),
                      "umi": cells.loc[use, "log_umi"].to_numpy(),
                      "adt": cells.loc[use, "log_adt"].to_numpy(),
                      "donor": dcodes.to_numpy()})
    for m in MODS:
        d[m[8:]] = cells.loc[use, m].to_numpy()
    terms = " + ".join(["rna", "umi", "adt"] + [m[8:] for m in MODS])
    try:
        fit = smf.mixedlm(f"y ~ {terms}", d, groups=d["donor"]).fit(method="lbfgs")
    except Exception as exc:                                       # noqa: BLE001
        print(f"  {gene}: mixed model did not converge ({exc})"); continue
    rec = {"gene": gene, "n_cells": len(d),
           "b_rna": fit.params.get("rna"), "p_rna": fit.pvalues.get("rna"),
           "donor_var": float(fit.cov_re.iloc[0, 0]), "resid_var": float(fit.scale)}
    tops = (fit.params[[m[8:] for m in MODS]].abs().sort_values(ascending=False).head(3))
    rec["top_modules"] = ", ".join(
        f"{k} {fit.params[k]:+.3f}{'*' if fit.pvalues[k] < 0.05 else ''}" for k in tops.index)
    rows.append(rec)
mix = pd.DataFrame(rows)
if len(mix):
    print("mixed model, protein ~ own mRNA + trafficking + depth, donor random intercept:")
    print(mix.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    mix.to_csv(ST.tab("mixedlm_focus"), index=False)
    print("\nwrote", ST.tab("mixedlm_focus"))
else:
    print("no focus marker could be fitted")
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2237: ConvergenceWarning: The MLE may be on the boundary of the parameter space.
  warnings.warn(msg, ConvergenceWarning)
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2261: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
  warnings.warn(msg, ConvergenceWarning)
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2237: ConvergenceWarning: The MLE may be on the boundary of the parameter space.
  warnings.warn(msg, ConvergenceWarning)
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2261: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
  warnings.warn(msg, ConvergenceWarning)

mixed model, protein ~ own mRNA + trafficking + depth, donor random intercept:
gene  n_cells     b_rna     p_rna  donor_var  resid_var                                                                 top_modules
 CD5    45993    0.0636 1.335e-67     0.4011     0.3787             glycosylation +0.073*, recycling +0.061*, er_folding_qc -0.050*
 CD7    45993     0.338         0     0.7513     0.3944                 shedding -0.109*, escrt_exosome -0.053*, retrograde -0.045*
DPP4    45993    0.2742 7.391e-28          0     0.1614       snare_fusion +0.059*, ubiquitin_proteostasis +0.048*, shedding +0.029
CD28    45993   0.07963 1.914e-49          0     0.7479 glycosylation -0.154*, ubiquitin_proteostasis -0.102*, snare_fusion -0.088*
CCR4    45993    0.2335         0     0.9718     0.4861 snare_fusion -0.155*, ubiquitin_proteostasis -0.140*, glycosylation -0.118*
CCR7    45993 -0.005709 0.0003325          0    0.08838 glycosylation +0.065*, snare_fusion +0.042*, ubiquitin_proteostasis +0.041*

wrote /home/projects/nyosef/zvise/MF/tables/tgt45_mixedlm_focus.csv

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2237: ConvergenceWarning: The MLE may be on the boundary of the parameter space.
  warnings.warn(msg, ConvergenceWarning)
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2261: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
  warnings.warn(msg, ConvergenceWarning)
```

```python  # cell 13
# ============================================================================
# §4d  Donor-level replication of the variance partition
# ============================================================================
# The donor-level version of the same question, which is the quantity the screen ranks on
# rather than the cell-to-cell one.
md = (cells.loc[cells["arm"].isin([MALIG, REACT])]
      .groupby(["arm", "donor"])[MODS].mean())
pd_ = pw.copy()
rows = []
for adt in DETECTED:
    gene = ST.ADT_TO_GENE.get(adt)
    rna_col = f"rna_{gene}"
    if gene is None or rna_col not in cells.columns:
        continue
    rna_d = (cells.loc[cells["arm"].isin([MALIG, REACT])]
             .groupby(["arm", "donor"])[rna_col].mean())
    j = ADT_DONORS
    y = (pd_.loc[MALIG, adt].reindex(j) - pd_.loc[REACT, adt].reindex(j))
    x = (rna_d.loc[MALIG].reindex(j) - rna_d.loc[REACT].reindex(j))
    Tm = (md.loc[MALIG].reindex(j) - md.loc[REACT].reindex(j))
    ok = y.notna() & x.notna() & Tm.notna().all(1)
    if ok.sum() < 8 or y[ok].std() == 0:
        continue
    yv = y[ok].to_numpy(float)
    r_own = stats.spearmanr(yv, x[ok].to_numpy(float))
    pr = partial_r2(yv - yv.mean(),
                    (x[ok].to_numpy(float) - x[ok].mean()).reshape(-1, 1),
                    (Tm[ok].to_numpy(float) - Tm[ok].to_numpy(float).mean(0)))
    rows.append({"gene": gene, "n_donors": int(ok.sum()), "rho_own_rna": r_own.statistic,
                 "p_own_rna": r_own.pvalue, "r2_own_rna": pr["r2_a"],
                 "inc_r2_traffic": pr["r2_ab"] - pr["r2_a"]})
dl = pd.DataFrame(rows).sort_values("inc_r2_traffic", ascending=False)
print("\ndonor-level replication -- the malignant-minus-reactive delta, 17 points per marker:")
print(dl.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print("\nWith 17 donors and 11 modules this model is saturated: the trafficking column is an "
      "upper bound and the ordering, not the value, is what survives. The cell-level fit above "
      "is the one with power; this is here to show the same ranking appears in the quantity "
      "the screen actually uses.")
dl.to_csv(ST.tab("donor_level_traffic"), index=False)
print("wrote", ST.tab("donor_level_traffic"))
```

```
donor-level replication -- the malignant-minus-reactive delta, 17 points per marker:
   gene  n_donors  rho_own_rna  p_own_rna  r2_own_rna  inc_r2_traffic
   DPP4        17       0.2819     0.2731    0.001541          0.9057
  ITGB1        14       0.3582     0.2085      0.1074          0.8666
  PTPRC        17      0.09314     0.7222    0.003338           0.825
  TIGIT        17       0.2304     0.3737     0.06505           0.806
  ITGAL        14       0.5209    0.05615      0.1817          0.7815
KIR2DL3        17      0.09337     0.7215      0.1236          0.6885
   CCR7        17      -0.1985      0.445     0.01349          0.6845
   IL4R        17    -0.009804     0.9702      0.2824          0.6698
   CD3E        17       0.1054     0.6873    0.001558           0.656
   CD52        14       0.4374     0.1178      0.1956          0.6379
  PTPRC        17       0.3015     0.2396      0.2195          0.5763
   IL7R        17       0.6814   0.002597      0.3876          0.5499
   CD27        17       0.5613    0.01906      0.4089          0.4977
   ICOS        17       0.5833    0.01397       0.472          0.4534
  PDCD1        17       0.6544   0.004368       0.541          0.3754
    CD2        17       0.7696  0.0003032      0.6962          0.2859
   CD28        17       0.7206   0.001102      0.6649          0.2006
    CD5        17        0.826  4.407e-05       0.715          0.1986
    CD7        17       0.6176   0.008242      0.5597          0.1913
   CCR4        17       0.8603  9.427e-06      0.8048          0.1216
   SELL        17       0.9118  3.506e-07      0.8996          0.0869

With 17 donors and 11 modules this model is saturated: the trafficking column is an upper bound and the ordering, not the value, is what survives. The cell-level fit above is the one with power; this is here to show the same ranking appears in the quantity the screen actually uses.
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_donor_level_traffic.csv
```

```python  # cell 14
# ============================================================================
# §4e F2  A test the machinery can fail: held-out donors, against random programs
# ============================================================================
# The in-sample R2 in §4a has no null: 11 extra covariates on 46k cells always explain something.
# Two controls here.
#  (1) Out of sample. Fit on 16 donors, score the 17th. The donor is the replicate, so the
#      Wilcoxon below is over 17 numbers, and 11 df cannot overfit what they are scored on.
#  (2) Specificity. The same held-out gain from 11 random programs, each the mean of as many
#      measured transcripts (`rna_*`, the marker's own excluded) as the real module has genes.
#      Machinery that does no better than random programs is "the cell's transcriptional
#      state", not trafficking. The pool is surface-marker transcripts, which co-vary with
#      T-cell state, so this null is conservative.
# Everything is demeaned within donor (as in §4a), so the fits carry no intercept and the
# per-donor Gram matrices give every leave-one-donor-out fit exactly, without refitting.
N_NULL = 200
DON_U = np.asarray(sorted(dcodes.unique()))
IDX = [np.flatnonzero(dcodes.to_numpy() == d) for d in DON_U]
MOD_SIZES = [max(1, int(adt_man["traffic_module_genes_present"].get(m[8:], 1))) for m in MODS]
RNA_POOL = [c for c in cells.columns if c.startswith("rna_")]
R_d = _demean(cells.loc[use, RNA_POOL], dcodes).to_numpy(float)


def lodo_gain(y, Xa, Xb):
    """Per held-out donor: R2(Xa + Xb) - R2(Xa) on that donor, coefficients fitted on the rest."""
    Z = np.column_stack([Xa, Xb, y])
    G = np.stack([Z[ix].T @ Z[ix] for ix in IDX])
    Gt, p, na = G.sum(0), Z.shape[1] - 1, Xa.shape[1]
    out = np.full(len(IDX), np.nan)
    for i, te in enumerate(G):
        tr = Gt - te
        if te[p, p] <= 0:
            continue
        sse = []
        for cols in (np.arange(na), np.arange(p)):
            b = np.linalg.lstsq(tr[np.ix_(cols, cols)], tr[cols, p], rcond=None)[0]
            sse.append(te[p, p] - 2 * b @ te[cols, p] + b @ te[np.ix_(cols, cols)] @ b)
        out[i] = (sse[0] - sse[1]) / te[p, p]
    return out


# observed
design, rows, gains = {}, [], {}
for _, r_ in vp.iterrows():
    rna_col = f"rna_{r_['gene']}"
    y = _demean(P.loc[use, r_["adt"]], dcodes).to_numpy(float)
    x_rna = _demean(cells.loc[use, [rna_col]], dcodes).to_numpy(float)
    Xa = np.column_stack([Xumi_d, x_rna])
    g_rna, g_tr = lodo_gain(y, Xumi_d, x_rna), lodo_gain(y, Xa, Xmod_d)
    design[r_["label"]] = (y, Xa, [RNA_POOL.index(c) for c in RNA_POOL if c != rna_col])
    gains[r_["label"]] = g_tr
    rows.append({"label": r_["label"], "gene": r_["gene"], "adt": r_["adt"],
                 "gain_rna": np.nanmedian(g_rna), "gain_traffic": np.nanmedian(g_tr),
                 "gain_traffic_q25": np.nanpercentile(g_tr, 25),
                 "gain_traffic_q75": np.nanpercentile(g_tr, 75),
                 "frac_donors_gain": float(np.nanmean(g_tr > 0)),
                 "p_donor": float(stats.wilcoxon(g_tr[np.isfinite(g_tr)],
                                                 alternative="greater").pvalue)})
mt = pd.DataFrame(rows)

# null: the same markers, 11 random size-matched programs per draw
rng = np.random.default_rng(0)
null = {k: np.empty(N_NULL) for k in design}
for b_ in range(N_NULL):
    picks = [rng.permutation(len(RNA_POOL)) for _ in MOD_SIZES]
    for k, (y, Xa, pool) in design.items():
        pool_set = set(pool)
        Xn = np.column_stack([R_d[:, [j for j in pk if j in pool_set][:s]].mean(1)
                              for pk, s in zip(picks, MOD_SIZES)])
        null[k][b_] = np.nanmedian(lodo_gain(y, Xa, Xn))
mt["null_q05"] = [np.percentile(null[k], 5) for k in mt["label"]]
mt["null_median"] = [np.median(null[k]) for k in mt["label"]]
mt["null_q95"] = [np.percentile(null[k], 95) for k in mt["label"]]
mt["p_null"] = [(1 + (null[k] >= o).sum()) / (N_NULL + 1)
                for k, o in zip(mt["label"], mt["gain_traffic"])]
mt["q_donor"] = multipletests(mt["p_donor"], method="fdr_bh")[1]
mt["q_null"] = multipletests(mt["p_null"], method="fdr_bh")[1]
mt["beats_random"] = (mt["q_donor"] < 0.05) & (mt["q_null"] < 0.05)
mt = mt.sort_values("gain_traffic").reset_index(drop=True)

print(f"held-out-donor gain in R2, {len(DON_U)} donors, {N_NULL} random-program draws:")
print(mt.drop(columns=["adt"]).iloc[::-1]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\n{int(mt['beats_random'].sum())} of {len(mt)} markers: the machinery adds to held-out "
      "donors AND beats size-matched random programs (both BH q < 0.05).")
mt.to_csv(ST.tab("machinery_test"), index=False)
print("wrote", ST.tab("machinery_test"))

# F2 -- one row per marker
fig, ax = plt.subplots(figsize=(5.6, 0.22 * len(mt) + 1.4))
for yi, r_ in mt.iterrows():
    ax.add_patch(plt.Rectangle((r_["null_q05"], yi - .32), r_["null_q95"] - r_["null_q05"], .64,
                               facecolor="#e6e6e6", edgecolor="none", zorder=1))
    ax.vlines(r_["null_median"], yi - .32, yi + .32, color="#9e9e9e", lw=.8, zorder=2)
    g = gains[r_["label"]]
    ax.scatter(g, np.full(len(g), yi) + np.random.default_rng(yi).normal(0, .06, len(g)), s=4,
               color=TF.MALIG, alpha=.25, linewidths=0, zorder=2)
    ax.hlines(yi, r_["gain_traffic_q25"], r_["gain_traffic_q75"], color=TF.MALIG, lw=1.4,
              zorder=3)
    ax.scatter(r_["gain_traffic"], yi, s=26, color=TF.MALIG, zorder=4, linewidths=0)
    ax.scatter(r_["gain_rna"], yi, s=22, marker="D", facecolor="white", edgecolor=TF.REACT,
               linewidths=1, zorder=4)
ax.axvline(0, lw=.5, c="k")
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_yticks(range(len(mt)))
ax.set_yticklabels([TF.it(l) + ("  ★" if b else "") for l, b in zip(mt["label"],
                                                                    mt["beats_random"])])
for t, b in zip(ax.get_yticklabels(), mt["beats_random"]):
    t.set_fontweight("bold" if b else "normal")
ax.set_ylim(-.6, len(mt) - .4)
ax.set_xlabel("held-out-donor gain in R² (symlog)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=TF.MALIG, markersize=4,
                          label="+ machinery (median, IQR, donors)"),
                   Line2D([], [], marker="D", ls="", markerfacecolor="white",
                          markeredgecolor=TF.REACT, markersize=4, label="+ own mRNA"),
                   Patch(color="#e6e6e6", label="random programs, 5–95%")],
          fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_title(f"Machinery beats random programs for {int(mt['beats_random'].sum())}/{len(mt)} "
             "markers", loc="left", fontsize=7.5)
TF._clean(ax)
fig.tight_layout()
TF.save(fig, "F2_machinery_test", caption_text=(
    f"Malignant CD4, {len(DON_U)} donors, within-donor. Gain over depth + own mRNA; "
    f"★ q < 0.05 vs 0 and vs {N_NULL} random programs."))
TF.save_source(mt, "F2_machinery_test_source")
TF.logfig("F2_machinery_test", "held-out machinery gain vs random programs", len(mt))
```

```
held-out-donor gain in R2, 17 donors, 200 random-program draws:
         label    gene   gain_rna  gain_traffic  gain_traffic_q25  gain_traffic_q75  frac_donors_gain  p_donor  null_q05  null_median  null_q95  p_null  q_donor  q_null  beats_random
          SELL    SELL    0.06642      0.004782         -0.004535             0.012            0.7059   0.1645 -0.001972     0.002355  0.009506  0.2587    0.996  0.9235         False
         PDCD1   PDCD1   0.007219      0.001547         -0.009747          0.007059            0.5294   0.7415 -0.004891    -0.001249  0.002188  0.1144    0.996  0.7761         False
          DPP4    DPP4  0.0004775     0.0002973         -0.005584          0.001809            0.5882   0.7105 -0.001372     0.001038  0.005176  0.6517    0.996  0.9235         False
          CD3E    CD3E -7.624e-05     0.0002432         -0.001436         0.0009984            0.5882   0.5732 -0.001219   -2.731e-05  0.001115  0.3483    0.996  0.9235         False
           CD5     CD5    0.00317     3.453e-05         -0.001397          0.002721            0.5294   0.5184 -0.002511    0.0001671  0.003087  0.5572    0.996  0.9235         False
          CD27    CD27    0.01116    -0.0004054         -0.002139          0.001002            0.3529   0.7707 -0.003016   -0.0002649  0.003226  0.5373    0.996  0.9235         False
           CD2     CD2    0.01281    -0.0005118         -0.002785          0.001618            0.4118   0.7707 -0.002878    0.0001599  0.002984  0.6318    0.996  0.9235         False
          ICOS    ICOS   0.004213     -0.001009         -0.006842          0.009792            0.4118   0.5912 -0.004165   -0.0004947  0.003334  0.6468    0.996  0.9235         False
          IL7R    IL7R   0.004627     -0.001157         -0.002132        -9.832e-05            0.2353   0.9747 -0.002275   -0.0004268  0.002653  0.7711    0.996  0.9235         False
       KIR2DL3 KIR2DL3 -0.0009215     -0.001331         -0.002787        -0.0001292            0.2353   0.9399 -0.002431   -0.0007615  0.001148  0.7413    0.996  0.9235         False
          CCR4    CCR4    0.04009     -0.001898         -0.009781          0.007427            0.4706   0.6267 -0.005786    2.947e-05  0.006377  0.7264    0.996  0.9235         False
           CD7     CD7     0.1019     -0.002014          -0.01325         0.0006152            0.3529   0.9884  -0.01333    -0.005532 -0.001682 0.06965    0.996  0.7761         False
PTPRC (CD45RA)   PTPRC  0.0002461     -0.002261         -0.003222        -4.708e-06            0.2353   0.9508  -0.01801    -0.005971 -0.000809  0.1294    0.996  0.7761         False
          IL4R    IL4R  7.315e-05     -0.002293          -0.01872          0.001672            0.4118   0.9557 -0.001692     0.001091  0.004462    0.99    0.996    0.99         False
          CD28    CD28   0.002541     -0.002939         -0.004216         0.0007225            0.2941   0.8877 -0.006268    -0.002164 0.0009593  0.6816    0.996  0.9235         False
         TIGIT   TIGIT -0.0007248      -0.00342          -0.01224            0.0061            0.4118   0.8355  -0.00525    -0.001403  0.002677  0.8209    0.996  0.9235         False
          CCR7    CCR7  0.0001516     -0.003704          -0.01926         0.0009444            0.2941   0.9899 -0.003306   -4.918e-06  0.002609  0.9701    0.996    0.99         False
PTPRC (CD45RO)   PTPRC  -0.000838     -0.007789          -0.01789        -6.445e-05            0.2353    0.996  -0.01296     -0.00509 -0.001099  0.7861    0.996  0.9235         False

0 of 18 markers: the machinery adds to held-out donors AND beats size-matched random programs (both BH q < 0.05).
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_machinery_test.csv

wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_F2_machinery_test.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_F2_machinery_test.pdf

<Figure size 560x536 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_F2_machinery_test_source.csv
```

```python  # cell 15
# ============================================================================
# §6  Gene cards
# ============================================================================
# One row per antigen: the paired effect, how many patients carry it, where else it is found,
# and what is known about the protein. LRRC33 is the old symbol of NRROS.
CARD_GENES = {"FCRL3": "FCRL3", "GPR15": "GPR15", "NRROS": "NRROS (LRRC33)"}
off = TS.off_tumour(long, target_arms=[MALIG], healthy_t_arms=["CD4_HC"], value="pct_expr")
surf_all = ST.load_surfaceome(confident_only=False, strict=False)
_ann = ST.annotate(list(CARD_GENES)).set_index("gene")
_sel, _prev, _off = (t.set_index("gene") for t in (sel, prev, off))
_surf = surf_all.set_index("gene") if surf_all is not None else pd.DataFrame()


def _surface(g):
    if g not in _surf.index:
        return "not in reference"
    s_ = _surf.loc[g]
    return (f"{'confident' if s_['surface_confident'] else 'weak'} · "
            f"{s_['surfy_class'] or '?'} · {int(float(s_['n_tm'])) if pd.notna(s_['n_tm']) else '?'} TM")


for gene, label in CARD_GENES.items():
    a_ = _ann.loc[gene]
    cd = a_["cd_name"] or (_surf.loc[gene, "cd_number"] if gene in _surf.index else "") or "—"
    pc = cmp_[cmp_["gene"].eq(gene)]
    facts = {
        "CD name": cd,
        "clinical stage": a_["stage"] or "no precedent",
        "surface": _surface(gene),
        "median log2FC": (f"{_sel.loc[gene, 'log2fc_paired']:+.2f} "
                          f"(q = {_sel.loc[gene, 'q']:.2g})") if gene in _sel.index else "—",
        "malignant +": (f"{_prev.loc[gene, 'median_pct_expr']:.0%} median"
                              if gene in _prev.index else "—"),
        "healthy CD4 +": (f"{_off.loc[gene, 'healthy_t_burden']:.0%} median"
                                if gene in _off.index else "—"),
        "worst off-tumour": (f"{TF.ARM_SHORT.get(_off.loc[gene, 'worst_offtumour_arm'])} "
                             f"{_off.loc[gene, 'worst_offtumour']:.0%}"
                             if gene in _off.index else "—"),
        "CITE protein": (f"Δ {pc['prot_delta'].iloc[0]:+.2f} {NORM}"
                         + ("" if pc["detected"].iloc[0] else " (at background)")
                         if len(pc) else "not on panel"),
    }
    fig, rec, burden = TF.gene_card(long, gene, label=label, min_donors=MIN_DONORS)
    TF.save(fig, f"card_{gene}", caption_text=(
        f"{int(rec['n_pairs'].iloc[0])} paired donors. % positive = share of a donor's cells "
        "in that arm with ≥1 UMI of the transcript."))
    TF.save_source(burden, f"card_{gene}_source")
    TF.logfig(f"card_{gene}", f"gene card, {label}", int(rec["n_pairs"].iloc[0]))
    print(label, facts, "\n")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_card_FCRL3.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_card_FCRL3.pdf

<Figure size 740x260 with 3 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_card_FCRL3_source.csv
FCRL3 {'CD name': 'CD307c', 'clinical stage': 'no precedent', 'surface': 'confident · Receptors · 1 TM', 'median log2FC': '+1.54 (q = 3.7e-06)', 'malignant +': '20% median', 'healthy CD4 +': '1% median', 'worst off-tumour': 'NK 12%', 'CITE protein': 'Δ +0.05 clr (at background)'} 


wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_card_GPR15.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_card_GPR15.pdf

<Figure size 740x260 with 3 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_card_GPR15_source.csv
GPR15 {'CD name': '—', 'clinical stage': 'no precedent', 'surface': 'confident · Receptors · 7 TM', 'median log2FC': '+1.44 (q = 9.6e-07)', 'malignant +': '18% median', 'healthy CD4 +': '1% median', 'worst off-tumour': 'reactive 2%', 'CITE protein': 'not on panel'} 


wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_card_NRROS.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt45_card_NRROS.pdf

<Figure size 740x260 with 3 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_card_NRROS_source.csv
NRROS (LRRC33) {'CD name': '—', 'clinical stage': 'no precedent', 'surface': 'confident · Miscellaneous · 1 TM', 'median log2FC': '+1.20 (q = 9.9e-08)', 'malignant +': '16% median', 'healthy CD4 +': '5% median', 'worst off-tumour': 'NK 15%', 'CITE protein': 'not on panel'}
```

```python  # cell 16
# ============================================================================
# §5  Dossier
# ============================================================================
def SEC(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def kv(k, v): print(f"  {k:36s} {v}")

SEC("1 - SCOPE")
kv("source", Path(adt_man["source"]).name)
kv("cells with measured ADT", f"{adt_man['n_cells']:,}")
kv("donors", cells["donor"].nunique())
kv("diagnoses", don.loc[cells['donor'].unique(), "dx"].value_counts().to_dict())
kv("panel", f"{len(MARKERS)} markers + {len(CTRL)} isotype controls")
kv("markers above background", f"{len(DETECTED)}")
kv("normalisation", NORM)

SEC("2 - PROTEIN vs TRANSCRIPT")
for lab, res in (("every mapped marker", all_r),
                 ("markers above background", det_r),
                 ("markers at background", und_r)):
    if res:
        kv(lab, f"rho {res[0]:+.2f}, same direction {res[1]}/{res[2]} "
                f"({res[1] / res[2]:.0%})")
kv("markers that disagree", len(dis))

SEC("3 - TRAFFICKING")
kv("markers modelled", len(vp))
kv("trafficking beats own mRNA", int(vp["traffic_beats_own"].sum()))
kv("median incremental R2, trafficking", f"{vp['inc_r2_traffic'].median():.4f}")
kv("median incremental R2, own mRNA", f"{vp['inc_r2_own_rna'].median():.4f}")
kv("beats random programs, held out", f"{int(mt['beats_random'].sum())}/{len(mt)}")
print()
print(pd.Series(adt_man["traffic_module_genes_present"]).sort_values(ascending=False)
      .rename("HVG genes per module").to_string())

SEC("4 - WHAT THIS IS NOT")
for line in [
    "A healthy baseline: no HC donor carries ADT.",
    "A skin result: no skin sample carries ADT.",
    "A statement about KIR3DL2 protein: that clone sits at isotype level in this panel.",
    "A causal claim about trafficking. The modules are correlated with each other and with "
    "cell state; an incremental R2 is an association inside malignant cells, not a mechanism, "
    "and the modules behave largely as one shared activity axis rather than as 11 pathways.",
    "A full trafficking measurement: the modules are averaged over HVG survivors only, and "
    "three of them rest on fewer than five genes.",
]:
    print("  - " + line)

SEC("5 - OUTPUTS")
TF.write_index("figure_index")
for p in sorted(ST.TAB_DIR.glob(f"{PREFIX}*")) + sorted(ST.FIG_DIR.glob(f"{PREFIX}*")):
    print("  ", p.relative_to(NB_DIR))
```

```
==============================================================================
1 - SCOPE
==============================================================================
  source                               blood_T_annotated.h5ad
  cells with measured ADT              61,215
  donors                               17
  diagnoses                            {'SS': 15, 'MF': 2}
  panel                                124 markers + 4 isotype controls
  markers above background             25
  normalisation                        clr

==============================================================================
2 - PROTEIN vs TRANSCRIPT
==============================================================================
  every mapped marker                  rho +0.61, same direction 50/75 (67%)
  markers above background             rho +0.79, same direction 16/21 (76%)
  markers at background                rho +0.55, same direction 34/54 (63%)
  markers that disagree                5

==============================================================================
3 - TRAFFICKING
==============================================================================
  markers modelled                     18
  trafficking beats own mRNA           8
  median incremental R2, trafficking   0.0030
  median incremental R2, own mRNA      0.0039
  beats random programs, held out      0/18

glycosylation             14
shedding                  10
er_folding_qc              7
recycling                  6
er_translocation           6
er_to_golgi                5
retrograde                 5
snare_fusion               5
ubiquitin_proteostasis     4
escrt_exosome              4
endocytosis                3

==============================================================================
4 - WHAT THIS IS NOT
==============================================================================
  - A healthy baseline: no HC donor carries ADT.
  - A skin result: no skin sample carries ADT.
  - A statement about KIR3DL2 protein: that clone sits at isotype level in this panel.
  - A causal claim about trafficking. The modules are correlated with each other and with cell state; an incremental R2 is an association inside malignant cells, not a mechanism, and the modules behave largely as one shared activity axis rather than as 11 pathways.
  - A full trafficking measurement: the modules are averaged over HVG survivors only, and three of them rest on fewer than five genes.

==============================================================================
5 - OUTPUTS
==============================================================================
wrote /home/projects/nyosef/zvise/MF/tables/tgt45_figure_index.csv
   tables/tgt45_F1_protein_vs_transcript_source.csv
   tables/tgt45_F2_machinery_test_source.csv
   tables/tgt45_adt_gate.csv
   tables/tgt45_card_FCRL3_source.csv
   tables/tgt45_card_GPR15_source.csv
   tables/tgt45_card_NRROS_source.csv
   tables/tgt45_donor_level_traffic.csv
   tables/tgt45_figure_index.csv
   tables/tgt45_machinery_test.csv
   tables/tgt45_mixedlm_focus.csv
   tables/tgt45_module_partial_r2.csv
   tables/tgt45_normalisation_check.csv
   tables/tgt45_protein_vs_transcript.csv
   tables/tgt45_variance_partition.csv
   figures/targets/tgt45_F1_protein_vs_transcript.pdf
   figures/targets/tgt45_F1_protein_vs_transcript.png
   figures/targets/tgt45_F2_machinery_test.pdf
   figures/targets/tgt45_F2_machinery_test.png
   figures/targets/tgt45_card_FCRL3.pdf
   figures/targets/tgt45_card_FCRL3.png
   figures/targets/tgt45_card_GPR15.pdf
   figures/targets/tgt45_card_GPR15.png
   figures/targets/tgt45_card_NRROS.pdf
   figures/targets/tgt45_card_NRROS.png
```

```python  # cell 17
# ============================================================================
# §7  The complete numerical result, as one block to hand to another model
# ============================================================================
_ctx = f"""
Compartment: MF/CTCL **blood**, CITE-seq. This is the only part of the project that checks the
transcript ranking against MEASURED surface protein on the same cells:
{adt_man['n_cells']:,} buus2025 cells carry a {len(MARKERS)}-marker antibody-derived-tag panel
plus {len(CTRL)} isotype controls, over {cells['donor'].nunique()} donors (15 Sezary and 2 MF by
the curated clin_dx -- not all Sezary). No healthy donor and no skin sample carries ADT.

Marker gate: a marker counts as measured only if its mean log1p count exceeds that donor's mean
isotype count by >= 0.10 in at least two thirds of donors, and it is detected in >= 1% of cells.
{len(DETECTED)} of {len(MARKERS)} markers pass. This matters: ADT_CD158k (the KIR3DL2 clone, the
lacutamab target and the top transcript hit of the blood screen) sits at isotype level, so its
near-zero protein delta is a clone that did not bind, NOT protein contradicting transcript.

Normalisation: CLR (log1p count minus the cell's mean log1p over the non-control markers). Two
alternatives were compared on markers whose direction is known and all three agree, so the
choice is not carrying the result.

Protein delta: malignant minus reactive CD4, per donor, median over the {len(ADT_DONORS)} donors
with both arms. The transcript side is recomputed on the SAME donors so the two columns describe
the same patients.

Trafficking model: per marker, inside malignant CD4 cells, protein_CLR ~ own mRNA + 11
trafficking module scores + RNA depth + total ADT depth, donor-demeaned (and as a mixed model
with donor random intercept for the focus markers). The reported quantity is an INCREMENTAL R2:
what the machinery explains after the marker's own mRNA and library size have had their turn.
§4e tests it out of sample -- fit on 16 donors, score the 17th -- against 200 draws of
size-matched random gene programs, because 11 extra covariates on 46k cells always explain
something in sample.

Caveats that apply throughout: the modules are averaged over HVG survivors only and three rest on
fewer than five genes; they are strongly correlated with each other and with how much of anything
a cell is making, so read the trafficking column as "the cell's machinery", not as a named
pathway.
"""
_sec = [
    {"name": "§1 Marker gate",
     "what": "which panel markers were measured above isotype background, and which were not.",
     "how": "per-donor mean log1p against that donor's mean isotype log1p; `passes` is the gate.",
     "table": gate, "file": "tables/tgt45_adt_gate.csv", "max_rows": 140},
    {"name": "§2 Normalisation check",
     "what": "the three candidate normalisations judged on markers whose direction is known "
             "(CD7, DPP4, CD2, CD5 down; CCR4 up).",
     "how": "malignant-minus-reactive delta per normalisation; correct_sign is the test.",
     "table": cmpn, "file": "tables/tgt45_normalisation_check.csv"},
    {"name": "§3 Protein against transcript, every mapped marker",
     "what": "the measured protein delta beside the transcript log2FC for the same donors.",
     "how": "paired Wilcoxon on the protein side (q_prot); `detected` marks the markers above "
            "isotype background -- the ones at background carry no direction and must be "
            "excluded from any agreement rate.",
     "table": cmp_, "file": "tables/tgt45_protein_vs_transcript.csv", "max_rows": 140},
    {"name": "§3 Agreement rates",
     "what": "how often protein and transcript agree in direction.",
     "how": "Spearman and sign agreement, computed three ways. The middle row is the honest "
            "number; the first is what a version of this analysis that ignored the gate "
            "reported.",
     "table": pd.DataFrame(
         [{"set": lab, "rho": (r[0] if r else np.nan),
           "same_direction": (r[1] if r else np.nan), "n_markers": (r[2] if r else np.nan)}
          for lab, r in (("every mapped marker", all_r),
                         ("markers above isotype background", det_r),
                         ("markers at background (meaningless)", und_r))])},
    {"name": "§3 Markers where measured protein and transcript disagree",
     "what": "the candidates a transcript ranking would get wrong.",
     "how": "sign of the protein delta against sign of the transcript log2FC, measured markers "
            "only.", "table": dis[["gene", "cd_name", "prot_delta", "q_prot", "log2fc_paired",
                                   "q", "n_donors_prot", "surface_caveat"]]},
    {"name": "§4a Variance partition, every measured marker",
     "what": "how much of the surface-protein variance the trafficking machinery explains after "
             "the marker's own mRNA.",
     "how": "incremental R2 on donor-demeaned single cells; `traffic_beats_own` is the flag that "
            "the transcript was never going to rank that marker correctly.",
     "table": vp, "file": "tables/tgt45_variance_partition.csv", "max_rows": 140},
    {"name": "§4e Held-out test against random programs",
     "what": "whether the machinery still helps on a donor it was not fitted on, and whether it "
             "beats size-matched random gene programs.",
     "how": "leave-one-donor-out gain in R2 over depth + own mRNA; Wilcoxon over donors "
            "(q_donor) and against 200 random-program draws (q_null). `beats_random` requires "
            "both.", "table": mt, "file": "tables/tgt45_machinery_test.csv", "max_rows": 140},
    {"name": "§4d Donor-level replication",
     "what": "the same question on the 17 donor-level deltas, which is the quantity the screen "
             "ranks on.",
     "how": "saturated with 11 modules and 17 points -- the ordering survives, the value is an "
            "upper bound.", "table": dl, "file": "tables/tgt45_donor_level_traffic.csv"},
]
if "best" in dir():
    _sec.insert(-1, {"name": "§4b Best module per focus marker",
                     "what": "which trafficking module explains most of each focus marker, and "
                             "whether it is the one its biology predicts.",
                     "how": "incremental R2 per module; `predicted` is a prior (IL2RA and "
                            "TNFRSF1B are shed, CCR4 and KIR3DL2 internalise, CCR7 recycles), "
                            "not a result.",
                     "table": best, "file": "tables/tgt45_module_partial_r2.csv"})
if "mix" in dir() and len(mix):
    _sec.insert(-1, {"name": "§4c Mixed model, focus markers",
                     "what": "the formal version with donor as a random intercept.",
                     "how": "protein ~ own mRNA + trafficking + depth, donor random intercept.",
                     "table": mix, "file": "tables/tgt45_mixedlm_focus.csv"})
TF.llm_export("45", "Blood CITE-seq: measured protein and what explains it", _ctx, _sec)
```

```
======================== COPY FROM HERE ========================

# 45 — Blood CITE-seq: measured protein and what explains it

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**, CITE-seq. This is the only part of the project that checks the
transcript ranking against MEASURED surface protein on the same cells:
61,215 buus2025 cells carry a 124-marker antibody-derived-tag panel
plus 4 isotype controls, over 17 donors (15 Sezary and 2 MF by
the curated clin_dx -- not all Sezary). No healthy donor and no skin sample carries ADT.

Marker gate: a marker counts as measured only if its mean log1p count exceeds that donor's mean
isotype count by >= 0.10 in at least two thirds of donors, and it is detected in >= 1% of cells.
25 of 124 markers pass. This matters: ADT_CD158k (the KIR3DL2 clone, the
lacutamab target and the top transcript hit of the blood screen) sits at isotype level, so its
near-zero protein delta is a clone that did not bind, NOT protein contradicting transcript.

Normalisation: CLR (log1p count minus the cell's mean log1p over the non-control markers). Two
alternatives were compared on markers whose direction is known and all three agree, so the
choice is not carrying the result.

Protein delta: malignant minus reactive CD4, per donor, median over the 17 donors
with both arms. The transcript side is recomputed on the SAME donors so the two columns describe
the same patients.

Trafficking model: per marker, inside malignant CD4 cells, protein_CLR ~ own mRNA + 11
trafficking module scores + RNA depth + total ADT depth, donor-demeaned (and as a mixed model
with donor random intercept for the focus markers). The reported quantity is an INCREMENTAL R2:
what the machinery explains after the marker's own mRNA and library size have had their turn.
§4e tests it out of sample -- fit on 16 donors, score the 17th -- against 200 draws of
size-matched random gene programs, because 11 extra covariates on 46k cells always explain
something in sample.

Caveats that apply throughout: the modules are averaged over HVG survivors only and three rest on
fewer than five genes; they are strongly correlated with each other and with how much of anything
a cell is making, so read the trafficking column as "the cell's machinery", not as a named
pathway.

## §1 Marker gate

**What:** which panel markers were measured above isotype background, and which were not.

**How:** per-donor mean log1p against that donor's mean isotype log1p; `passes` is the gate.

**Full table on disk:** `tables/tgt45_adt_gate.csv`

```csv
index,frac_cells_pos,mean_log1p,mean_log1p_minus_bg,frac_donors_above_bg,is_control,gene,passes
ADT_CD1a,0.0125,0.0163,-0.0149,0,False,CD1A,False
ADT_CD1c,0.00959,0.0162,-0.0149,0,False,CD1C,False
ADT_CD1d,0.0498,0.0735,0.0423,0.0588,False,CD1D,False
ADT_CD2,0.741,1.8,1.77,0.941,False,CD2,True
ADT_CD3,0.679,1.34,1.3,0.941,False,CD3E,True
ADT_CD4,0.84,2.22,2.18,1,False,CD4,True
ADT_CD5,0.673,1.11,1.08,1,False,CD5,True
ADT_CD7,0.488,1.03,1,0.941,False,CD7,True
ADT_CD8A,0.125,0.321,0.29,0.529,False,CD8A,False
ADT_CD11a,0.459,0.705,0.674,0.765,False,ITGAL,True
ADT_CD11a_CD18,0.0282,0.0484,0.0172,0,False,,False
ADT_CD11b,0.0288,0.0402,0.00904,0,False,ITGAM,False
ADT_CD11c,0.02,0.0277,-0.00347,0.0588,False,ITGAX,False
ADT_CD14,0.00879,0.0106,-0.0206,0,False,CD14,False
ADT_CD16,0.0769,0.14,0.109,0.0588,False,FCGR3A,False
ADT_CD18,0.34,0.44,0.409,0.647,False,ITGB2,False
ADT_CD19,0.0333,0.0442,0.0131,0,False,CD19,False
ADT_CD20,0.0308,0.0402,0.009,0,False,MS
... [23234 more chars truncated]
```
