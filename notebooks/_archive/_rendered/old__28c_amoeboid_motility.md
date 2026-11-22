# old/28c_amoeboid_motility.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28c_amoeboid_motility.ipynb` and is never run._

# 28c — amoeboid motility machinery in malignant T cells (atlas **v2**)

Does the malignant clone carry more of the non-proteolytic (amoeboid) migration machinery than
benign CD4 T cells? The degradome notebooks (28, 28b) ask the proteolytic question; this one asks
the other. Structure follows 28b: one figure per program, **program score first, then one violin
slot per gene**, three comparisons as rows.

**RNA presence ≠ migration.** Everything here measures machinery, not motility.

| row | comparison | replicate | test |
|---|---|---|---|
| A | malignant vs benign CD4, same MF biopsy | biopsy (`sample_id`) | paired Wilcoxon signed-rank |
| B | MF malignant \| MF benign \| healthy-donor CD4 | donor | donor-level Mann-Whitney, 3 pairs |
| C | epidermis malignant vs benign, dermis malignant vs benign, epidermis vs dermis malignant | donor × layer | paired Wilcoxon signed-rank within donor |

## Arms

- **Call.** `mal_tcr`, as in 28b (`malignancy.load_malignancy` + `cd4_level`): clonal vs
  non-clonal CD4 with a recovered CDR3. CDR3-less CD4 is excluded, not counted benign. Tregs
  are not CD4 here.
- **A, B** are MF skin only. **Healthy** = all atlas HC CD4 (`disease == HC`), its own arm.
- **C** uses the layer-split donors — any donor with ≥ `MIN_CELLS` malignant and benign CD4 in
  both Epidermis and Dermis (li2024 MF, buus2025 SS). Layer = the `tissue` of the sample.

## Programs

| program | genes | score |
|---|---|---|
| osmo_engine | WNK1 OXSR1 STK39 SLC12A2 AQP3 AQP1 AQP9 | mean of genes |
| actomyosin | RHOA ROCK1 ROCK2 MYH9 MYH10 PLCB1–4 | mean of genes |
| actin_piston | WAS WASL ACTR2 ACTR3 ARPC1B ARPC2–5 | mean of genes |
| nuclear_repair | CHMP4B CHMP2B VPS4A VPS4B | mean of genes |
| mechanosensor | PIEZO1 PIEZO2 YAP1 WWTR1 | mean of genes |
| nuclear_lamina | LMNA LMNB1 LMNB2 | log2 LMNA / (LMNB1 + LMNB2); lower = more deformable |

## Statistics

- **Values.** Per cell log1p CP10K (library size over all 42,347 genes). A program score is the
  plain per-cell mean of its genes (28b's family-score convention). The lamina score is computed
  per unit from mean CP10K, so it has dots and no per-cell violin.
- **Replicate.** Unit mean over ≥ `cfg.MIN_CELLS` cells: biopsy in A, donor in B, donor × layer
  in C.
- **Not testable (n.t.).** < `cfg.MIN_DETECTION` of cells detected in both arms, or
  < `cfg.MIN_DONORS_PER_ARM` units (paired: nonzero paired deltas).
- **Correction.** BH over every test drawn in one program's figure.
- **Colour of a star.** Teal = left arm higher, rust = right arm higher, black = n.s.

## Guardrails

- Intact amoeboid machinery is the T-cell baseline, not a malignant gain. Where malignant ≈
  benign, that is the result.
- Components only. S1PR1, CCR7, CD69, PPP3C\*, NFATC\*, TEAD\* and the IEG set are asserted
  absent from every program.

## Compute

§2 reads a subset of the 12 GB skin T object. Run on a GPU/compute kernel, never the login node.

## Outputs

`figures/amoeboid/amb28c_<program>.{png,pdf}`, `tables/amb28c_tests.csv`,
`tables/amb28c_unit_means.csv`, `tables/amb28c_arms.csv`.

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory, offset_copy
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests


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

import degradome_data as cfg                        # 28/28b gates: MIN_CELLS, MIN_DETECTION, ...
import malignancy as MAL                            # mal_tcr, as in 28b
from tcr_signaling_helpers import DISSOC_LABILE     # the project's IEG / dissociation set

DATA = NB_DIR / "data"
T_OBJ = DATA / "atlas_joint" / "skin_T_tcr_annotated_v4.h5ad"     # v2 skin T, all genes
FIG_DIR = NB_DIR / "figures" / "amoeboid"
TAB_DIR = NB_DIR / "tables"
PREFIX = "amb28c_"

PROGRAMS = {
    "osmo_engine": ["WNK1", "OXSR1", "STK39", "SLC12A2", "AQP3", "AQP1", "AQP9"],
    "actomyosin": ["RHOA", "ROCK1", "ROCK2", "MYH9", "MYH10", "PLCB1", "PLCB2", "PLCB3", "PLCB4"],
    "actin_piston": ["WAS", "WASL", "ACTR2", "ACTR3", "ARPC1B", "ARPC2", "ARPC3", "ARPC4",
                     "ARPC5"],
    "nuclear_repair": ["CHMP4B", "CHMP2B", "VPS4A", "VPS4B"],
    "mechanosensor": ["PIEZO1", "PIEZO2", "YAP1", "WWTR1"],
    "nuclear_lamina": ["LMNA", "LMNB1", "LMNB2"],
}
LAMINA = "nuclear_lamina"
GENES = [g for gs in PROGRAMS.values() for g in gs]

# Egress axis (S1PR1/CCR7/CD69), TCR-driven calcineurin-NFAT, TEAD, and the IEG set would split
# arms for the wrong reason. Components only: none of them may sit in a program.
_bad = [g for g in GENES
        if g in {"S1PR1", "CCR7", "CD69", "FOS", "JUN", "DUSP1", "ZFP36"} | DISSOC_LABILE
        or g.startswith(("PPP3C", "NFATC", "TEAD", "EGR", "NR4A"))]
assert not _bad, f"excluded genes in a program: {_bad}"

MIN_CELLS = cfg.MIN_CELLS            # cells per replicate unit (25)
MIN_DET = cfg.MIN_DETECTION          # below this in both arms -> not testable (1%)
MIN_N = cfg.MIN_DONORS_PER_ARM       # units per arm / nonzero paired deltas (4)
MAX_CELLS = 20_000                   # per-arm subsample for the violin KDE only
LAM_EPS = 0.1                        # CP10K; keeps a unit with no LMNB1/LMNB2 finite
SEED = 0
LAYERS = ["Epidermis", "Dermis"]
MALIG_LV, REACT_LV, HC_LV = "CD4_malignant", "CD4_reactive", "CD4_HC"

# Rows of every program figure: the arms drawn in each slot (left to right) and the tests.
# Every test is (left arm, right arm), so a star's colour always reads "left arm higher".
ROWS = {"A": ["mal", "ben"],
        "B": ["mal", "ben", "hc"],
        "C": ["epi_mal", "epi_ben", "derm_mal", "derm_ben"]}
KEY = {"A": "sample_id", "B": "donor", "C": "donor"}
TESTS = {"A": [("mal", "ben", "paired")],
         "B": [("mal", "ben", "mwu"), ("ben", "hc", "mwu"), ("mal", "hc", "mwu")],
         "C": [("epi_mal", "epi_ben", "paired"), ("derm_mal", "derm_ben", "paired"),
               ("epi_mal", "derm_mal", "paired")]}

MALIG, REACT, HC_COL = "#7b3294", "#5e81ac", "#444444"          # 28b's arm colours
MALIG_L, REACT_L = "#c2a5cf", "#a9bfd8"                          # dermis shades in row C
UP, DOWN, NS = "#2a9d8f", "#b5542a", "#bdbdbd"
ARM_COL = {"mal": MALIG, "ben": REACT, "hc": HC_COL, "epi_mal": MALIG, "epi_ben": REACT,
           "derm_mal": MALIG_L, "derm_ben": REACT_L}
ARM_SHORT = {"mal": "Malignant", "ben": "Benign", "hc": "Healthy", "epi_mal": "Epi malig.",
             "epi_ben": "Epi benign", "derm_mal": "Derm malig.", "derm_ben": "Derm benign"}

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": ["Carlito", "Calibri", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 400, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def tab28c(name):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    return TAB_DIR / f"{PREFIX}{name}.csv"


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{PREFIX}{name}.{ext}")
    plt.show()
    plt.close(fig)
    print(f"  -> {FIG_DIR.name}/{PREFIX}{name}.png/.pdf")


print(f"NB_DIR {NB_DIR}")
print(f"{len(PROGRAMS)} programs, {len(GENES)} genes | gates: {MIN_CELLS} cells/unit, "
      f"{MIN_DET:.0%} detection, {MIN_N} units")
```

```
NB_DIR /home/projects/nyosef/zvise/MF
6 programs, 36 genes | gates: 25 cells/unit, 1% detection, 4 units
```

```python  # cell 2
# ============================================================================
# §1  Arms — mal_tcr as in 28b, MF skin for A/B, healthy CD4, layer-split donors for C
# ============================================================================
# obs only here; the matrix is read in §2. v1/v2 guard: the v2 skin T object is 42,347 genes.
tb = sc.read_h5ad(T_OBJ, backed="r")
assert tb.n_vars == 42_347, f"{T_OBJ.name} has {tb.n_vars} genes: not the v2 object"
OBS_COLS = ["cell_id", "donor", "sample_id", "study", "disease", "tissue", "tech", "cell_type_T"]
obs = tb.obs[OBS_COLS].astype(str).copy()
HAS_TCR = tb.obs["has_tcr"].astype(str).str.lower().eq("true").to_numpy()
print(f"{T_OBJ.name}: {tb.n_obs:,} cells x {tb.n_vars:,} genes")

# 28b §1: clonal / non-clonal on mal_tcr (CDR3-less CD4 -> CD4_unassessed, excluded), and
# healthy CD4 as an arm of its own rather than folded into "benign".
mal = MAL.load_malignancy(obs["cell_id"], has_tcr=HAS_TCR, verbose=False)
LV = pd.Series(np.asarray(MAL.cd4_level(obs["cell_type_T"], mal, call="mal_tcr")).astype(str),
               index=obs.index)
LV[(obs["cell_type_T"] == "CD4") & (obs["disease"] == "HC")] = HC_LV
obs["arm_level"] = LV
obs["layer"] = obs["tissue"].where(obs["tissue"].isin(LAYERS), "")

IS_MF = obs["disease"].eq("MF")
IS_MAL, IS_BEN = LV.eq(MALIG_LV), LV.eq(REACT_LV)

# C: donors with >= MIN_CELLS malignant AND benign CD4 in BOTH layers, any disease.
_c = obs[(IS_MAL | IS_BEN) & (obs["layer"] != "")]
_ct = (pd.crosstab(_c["donor"], [_c["layer"], _c["arm_level"]])
       .reindex(columns=pd.MultiIndex.from_product([LAYERS, [MALIG_LV, REACT_LV]]), fill_value=0))
LAYER_DONORS = sorted(_ct.index[(_ct >= MIN_CELLS).all(axis=1)])
IN_C = obs["donor"].isin(LAYER_DONORS)

ARM_MASK = {
    "mal": IS_MF & IS_MAL,
    "ben": IS_MF & IS_BEN,
    "hc": LV.eq(HC_LV),
    "epi_mal": IN_C & obs["layer"].eq("Epidermis") & IS_MAL,
    "epi_ben": IN_C & obs["layer"].eq("Epidermis") & IS_BEN,
    "derm_mal": IN_C & obs["layer"].eq("Dermis") & IS_MAL,
    "derm_ben": IN_C & obs["layer"].eq("Dermis") & IS_BEN,
}
LOAD = np.logical_or.reduce([m.to_numpy() for m in ARM_MASK.values()])

ARMS = pd.DataFrame([{"arm": a, "n_cells": int(m.sum()), "n_donors": obs.loc[m, "donor"].nunique(),
                      "n_samples": obs.loc[m, "sample_id"].nunique(),
                      "tech": ", ".join(f"{t} {n:,}" for t, n in
                                        obs.loc[m, "tech"].value_counts().items())}
                     for a, m in ARM_MASK.items()])
ARMS.to_csv(tab28c("arms"), index=False)
print(f"\nCD4 on mal_tcr: malignant {int(IS_MAL.sum()):,} | benign {int(IS_BEN.sum()):,} | "
      f"unassessed (no CDR3, excluded) {int(LV.eq('CD4_unassessed').sum()):,} | "
      f"healthy {int(LV.eq(HC_LV).sum()):,}")
print(ARMS.to_string(index=False))

_ct.columns = [f"{l[:4]}|{'mal' if a == MALIG_LV else 'ben'}" for l, a in _ct.columns]
_ct["disease"] = _c.groupby("donor")["disease"].first()
print(f"\nlayer-split donors (C): {len(LAYER_DONORS)} "
      f"({', '.join(f'{d} {n}' for d, n in _ct.loc[LAYER_DONORS, 'disease'].value_counts().items())})")
print(_ct.assign(in_C=_ct.index.isin(LAYER_DONORS)).to_string())
```

```
skin_T_tcr_annotated_v4.h5ad: 539,916 cells x 42,347 genes

CD4 on mal_tcr: malignant 80,328 | benign 71,420 | unassessed (no CDR3, excluded) 286,492 | healthy 4,742
     arm  n_cells  n_donors  n_samples                                tech
     mal    68843        33         33           10x_5p 68,685, ECCITE 158
     ben    59220        38         38            10x_5p 59,186, ECCITE 34
      hc     4742        22         20 10x_5p 4,169, 10x_3p 555, ECCITE 18
 epi_mal    19827        10         14                       10x_5p 19,827
 epi_ben     5541        10         14                        10x_5p 5,541
derm_mal    22478        10         14                       10x_5p 22,478
derm_ben    42312        10         14                       10x_5p 42,312

layer-split donors (C): 10 (MF 8, SS 2)
                     Epid|mal  Epid|ben  Derm|mal  Derm|ben disease   in_C
donor                                                                     
B5__PT09                 2873      2063      2094       513      SS   True
B5__PT22                    0      3027         0      1972      SS  False
B5__PT23                  129       291      3409       885      SS   True
Li2024_atlas__CTCL1      2023        95      4705       269      MF   True
Li2024_atlas__CTCL2       858        50       299      1796      MF   True
Li2024_atlas__CTCL3      2713        28      4743      1618      MF   True
Li2024_atlas__CTCL4       692       740        44      5252      MF   True
Li2024_atlas__CTCL5      4384       411      1963      3044      MF   True
Li2024_atlas__CTCL6      2837       422       267      7299      MF   True
Li2024_atlas__CTCL7      2786      1367       290     19871      MF   True
Li2024_atlas__CTCL8       532        74      4664      1765      MF   True
```

```python  # cell 3
# ============================================================================
# §2  Expression — per-cell log1p CP10K for the program genes, and the program scores
# ============================================================================
# One read from the backed object: every cell any arm uses. The library size is summed over all
# 42,347 genes before the program genes are sliced out; the full matrix is then dropped.
ad = tb[LOAD].to_memory()
tb.file.close()
X = sp.csr_matrix(ad.layers["raw_counts"])
VAR = pd.Index(ad.var_names.astype(str))
assert not [g for g in GENES if g not in VAR], [g for g in GENES if g not in VAR]
lib = np.asarray(X.sum(axis=1)).ravel()
RAW = X[:, VAR.get_indexer(GENES)].toarray().astype(float)
CELLS = ad.obs_names
del ad, X

o = obs.loc[CELLS]
MASK = {a: m.loc[CELLS].to_numpy() for a, m in ARM_MASK.items()}
CP10K = pd.DataFrame(RAW / np.maximum(lib, 1)[:, None] * 1e4, index=CELLS, columns=GENES)
E = np.log1p(CP10K)
DET = pd.DataFrame(RAW > 0, index=CELLS, columns=GENES)
# Program score = plain per-cell mean of its genes (28b family-score convention). The lamina
# score is a ratio of unit means (§3), so it has no per-cell value.
for p, gs in PROGRAMS.items():
    DET[f"score_{p}"] = DET[gs].any(axis=1)
    if p != LAMINA:
        E[f"score_{p}"] = E[gs].mean(axis=1)
del RAW

# Fixed per-arm subsample for the violin KDE; every feature of an arm draws the same cells.
_rng = np.random.default_rng(SEED)
SUB = {}
for a, m in MASK.items():
    idx = np.flatnonzero(m)
    SUB[a] = np.sort(_rng.choice(idx, MAX_CELLS, replace=False)) if len(idx) > MAX_CELLS else idx

print(f"loaded {len(CELLS):,} cells | median library size {np.median(lib):,.0f}")
print("% cells detected per gene, MF malignant | MF benign | healthy:")
_d = pd.DataFrame({a: DET.loc[MASK[a], GENES].mean() for a in ("mal", "ben", "hc")})
for p, gs in PROGRAMS.items():
    print(f"  {p:<15} " + "  ".join(f"{g} {r.mal:.0%}|{r.ben:.0%}|{r.hc:.0%}"
                                    for g, r in _d.loc[gs].iterrows()))
```

```
loaded 145,062 cells | median library size 2,442
% cells detected per gene, MF malignant | MF benign | healthy:
  osmo_engine     WNK1 32%|31%|20%  OXSR1 21%|14%|10%  STK39 6%|3%|2%  SLC12A2 7%|3%|4%  AQP3 32%|18%|24%  AQP1 0%|0%|1%  AQP9 0%|0%|0%
  actomyosin      RHOA 87%|71%|55%  ROCK1 45%|27%|21%  ROCK2 8%|5%|4%  MYH9 64%|52%|39%  MYH10 4%|1%|1%  PLCB1 5%|1%|2%  PLCB2 7%|3%|4%  PLCB3 2%|1%|1%  PLCB4 0%|0%|0%
  actin_piston    WAS 48%|26%|19%  WASL 7%|4%|3%  ACTR2 51%|30%|25%  ACTR3 64%|38%|35%  ARPC1B 82%|53%|34%  ARPC2 88%|68%|54%  ARPC3 85%|65%|46%  ARPC4 53%|31%|22%  ARPC5 62%|33%|26%
  nuclear_repair  CHMP4B 23%|11%|7%  CHMP2B 30%|15%|9%  VPS4A 24%|13%|8%  VPS4B 35%|17%|11%
  mechanosensor   PIEZO1 11%|7%|5%  PIEZO2 0%|0%|0%  YAP1 0%|0%|0%  WWTR1 4%|1%|2%
  nuclear_lamina  LMNA 68%|33%|50%  LMNB1 25%|13%|7%  LMNB2 9%|4%|2%
```

```python  # cell 4
# ============================================================================
# §3  Tests — unit means, then A (paired per biopsy), B (donor MWU), C (paired per donor)
# ============================================================================
_UM = {}


def unit_means(arm, key):
    """(unit x feature) mean over >= MIN_CELLS cells of one arm, and the arm's detection.

    Unit = `key` (sample_id or donor). The lamina score is log2 of the ratio of unit-mean
    CP10K, LMNA / (LMNB1 + LMNB2), so it is built here rather than per cell.
    """
    k = (arm, key)
    if k not in _UM:
        m = MASK[arm]
        units = o[key].to_numpy()[m]
        n = pd.Series(units).value_counts()
        mean = E[m].groupby(units).mean()
        lam = CP10K.loc[m, PROGRAMS[LAMINA]].groupby(units).mean()
        mean[f"score_{LAMINA}"] = np.log2((lam["LMNA"] + LAM_EPS)
                                          / (lam["LMNB1"] + lam["LMNB2"] + LAM_EPS))
        _UM[k] = (mean[n.reindex(mean.index).to_numpy() >= MIN_CELLS], DET[m].mean())
    return _UM[k]


def run_test(feature, a, b, kind, key):
    ma, da = unit_means(a, key)
    mb, db = unit_means(b, key)
    rec = {"feature": feature, "arm_a": a, "arm_b": b, "unit": key,
           "test": "paired signed-rank" if kind == "paired" else "MWU",
           "det_a": da[feature], "det_b": db[feature], "p": np.nan, "reason": "",
           "n_pos": np.nan, "n_neg": np.nan}
    if kind == "paired":
        both = ma.index.intersection(mb.index)
        x, y = ma.loc[both, feature].to_numpy(), mb.loc[both, feature].to_numpy()
        d = x - y
        nz = d[d != 0]
        rec.update(n_a=len(x), n_b=len(y), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                   delta=np.median(d) if len(d) else np.nan)
        rec["sign"] = np.sign(rec["n_pos"] - rec["n_neg"])
        short = len(nz) < MIN_N
    else:
        x, y = ma[feature].to_numpy(), mb[feature].to_numpy()
        rec.update(n_a=len(x), n_b=len(y),
                   delta=(np.median(x) - np.median(y)) if len(x) and len(y) else np.nan)
        rec["sign"] = np.sign(rec["delta"])
        short = min(len(x), len(y)) < MIN_N
    rec["med_a"] = np.median(x) if len(x) else np.nan
    rec["med_b"] = np.median(y) if len(y) else np.nan
    if not np.nanmax([rec["det_a"], rec["det_b"]]) >= MIN_DET:
        rec["reason"] = f"below {MIN_DET:.0%} detection in both arms"
    elif short:
        rec["reason"] = f"too few units ({rec['n_a']}, {rec['n_b']}; need {MIN_N})"
    elif np.ptp(np.r_[x, y]) == 0:
        rec["reason"] = "no variance"
    elif kind == "paired":
        rec["p"] = float(wilcoxon(nz).pvalue)
    else:
        rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    rec["vals_a"], rec["vals_b"] = x, y
    return rec


def bh(df):
    """BH over every test in one program's figure; direction = a vs b where q < 0.05."""
    df = df.reset_index(drop=True)
    m = df["p"].notna()
    df["q"] = np.nan
    if m.any():
        df.loc[m, "q"] = multipletests(df.loc[m, "p"], method="fdr_bh")[1]
    df["direction"] = np.select([~m, df["q"] >= 0.05, df["sign"] > 0],
                                ["not tested", "ns", "up"], "down")
    return df


def features(p):
    return [f"score_{p}"] + PROGRAMS[p]


RES = pd.concat([bh(pd.DataFrame([dict(run_test(f, a, b, kind, KEY[row]), program=p, row=row)
                                  for f in features(p) for row, ts in TESTS.items()
                                  for a, b, kind in ts]))
                 for p in PROGRAMS], ignore_index=True)
_cols = ["program", "row", "feature", "arm_a", "arm_b", "unit", "test", "n_a", "n_b", "n_pos",
         "n_neg", "med_a", "med_b", "delta", "det_a", "det_b", "p", "q", "direction", "reason"]
RES[_cols].to_csv(tab28c("tests"), index=False)
pd.concat([m.rename_axis("unit").reset_index().melt(id_vars="unit", var_name="feature")
           .assign(arm=a, key=k) for (a, k), (m, _) in _UM.items()]).to_csv(
    tab28c("unit_means"), index=False)

_fmt = {"med_a": "{:.3f}".format, "med_b": "{:.3f}".format, "delta": "{:+.3f}".format,
        "q": "{:.3g}".format}
print("program scores (BH within each program's figure):")
print(RES.loc[RES["feature"].str.startswith("score_"),
              ["program", "row", "arm_a", "arm_b", "n_a", "n_b", "med_a", "med_b", "delta", "q",
               "direction"]].to_string(index=False, formatters=_fmt))
print("\ngenes with q < 0.05:")
_s = RES[~RES["feature"].str.startswith("score_") & (RES["q"] < 0.05)]
print(_s[["program", "row", "feature", "arm_a", "arm_b", "n_a", "med_a", "med_b", "delta", "q",
          "direction"]].to_string(index=False, formatters=_fmt) if len(_s) else "  none")
```

```
program scores (BH within each program's figure):
       program row    arm_a    arm_b  n_a  n_b med_a med_b  delta        q direction
   osmo_engine   A      mal      ben   33   33 0.204 0.193 -0.000    0.908        ns
   osmo_engine   B      mal      ben   33   38 0.204 0.193 +0.011     0.84        ns
   osmo_engine   B      ben       hc   38   21 0.193 0.151 +0.042   0.0105        up
   osmo_engine   B      mal       hc   33   21 0.204 0.151 +0.053   0.0118        up
   osmo_engine   C  epi_mal  epi_ben   10   10 0.179 0.195 -0.033    0.457        ns
   osmo_engine   C derm_mal derm_ben   10   10 0.194 0.195 +0.012    0.743        ns
   osmo_engine   C  epi_mal derm_mal   10   10 0.179 0.194 -0.019   0.0172      down
    actomyosin   A      mal      ben   33   33 0.456 0.402 +0.048 4.29e-05        up
    actomyosin   B      mal      ben   33   38 0.456 0.402 +0.055   0.0116        up
    actomyosin   B      ben       hc   38   21 0.402 0.245 +0.157 2.01e-05        up
    actomyosin   B      mal       hc   33   21 0.456 0.245 +0.211 1.03e-06        up
    actomyosin   C  epi_mal  epi_ben   10   10 0.492 0.448 +0.059   0.0667        ns
    actomyosin   C derm_mal derm_ben   10   10 0.488 0.417 +0.062   0.0271        up
    actomyosin   C  epi_mal derm_mal   10   10 0.492 0.488 +0.000    0.889        ns
  actin_piston   A      mal      ben   33   33 1.099 0.948 +0.153 4.22e-07        up
  actin_piston   B      mal      ben   33   38 1.099 0.935 +0.165  0.00156        up
  actin_piston   B      ben       hc   38   21 0.935 0.508 +0.426 3.75e-07        up
  actin_piston   B      mal       hc   33   21 1.099 0.508 +0.591 1.34e-07        up
  actin_piston   C  epi_mal  epi_ben   10   10 1.102 0.933 +0.137   0.0037        up
  actin_piston   C derm_mal derm_ben   10   10 0.922 0.791 +0.216   0.0037        up
  actin_piston   C  epi_mal derm_mal   10   10 1.102 0.922 +0.102    0.262        ns
nuclear_repair   A      mal      ben   33   33 0.327 0.238 +0.102 4.07e-08        up
nuclear_repair   B      mal      ben   33   38 0.327 0.237 +0.089 7.32e-05        up
nuclear_repair   B      ben       hc   38   21 0.237 0.115 +0.123 3.28e-06        up
nuclear_repair   B      mal       hc   33   21 0.327 0.115 +0.212 6.14e-08        up
nuclear_repair   C  epi_mal  epi_ben   10   10 0.449 0.317 +0.059   0.0036        up
nuclear_repair   C derm_mal derm_ben   10   10 0.387 0.244 +0.073   0.0149        up
nuclear_repair   C  epi_mal derm_mal   10   10 0.449 0.387 +0.060   0.0263        up
 mechanosensor   A      mal      ben   33   33 0.047 0.034 +0.017  0.00356        up
 mechanosensor   B      mal      ben   33   38 0.047 0.037 +0.010    0.247        ns
 mechanosensor   B      ben       hc   38   21 0.037 0.018 +0.019 0.000736        up
 mechanosensor   B      mal       hc   33   21 0.047 0.018 +0.029 0.000338        up
 mechanosensor   C  epi_mal  epi_ben   10   10 0.044 0.036 +0.007    0.423        ns
 mechanosensor   C derm_mal derm_ben   10   10 0.041 0.029 +0.016   0.0854        ns
 mechanosensor   C  epi_mal derm_mal   10   10 0.044 0.041 -0.002    0.574        ns
nuclear_lamina   A      mal      ben   33   33 3.005 2.019 +0.376   0.0253        up
nuclear_lamina   B      mal      ben   33   38 3.005 2.272 +0.734    0.237        ns
nuclear_lamina   B      ben       hc   38   21 2.272 3.921 -1.649   0.0246      down
nuclear_lamina   B      mal       hc   33   21 3.005 3.921 -0.916   0.0581        ns
nuclear_lamina   C  epi_mal  epi_ben   10   10 3.180 3.158 +0.146    0.648        ns
nuclear_lamina   C derm_mal derm_ben   10   10 2.687 1.423 +1.011   0.0273        up
nuclear_lamina   C  epi_mal derm_mal   10   10 3.180 2.687 +0.523    0.214        ns

genes with q < 0.05:
       program row feature    arm_a    arm_b  n_a med_a med_b  delta        q direction
   osmo_engine   B    WNK1      ben       hc   38 0.502 0.295 +0.207 0.000703        up
   osmo_engine   B    WNK1      mal       hc   33 0.500 0.295 +0.204  0.00707        up
   osmo_e
... [10374 more chars truncated]
```

```python  # cell 5
# ============================================================================
# §4  Drawing — 28b's conventions: violin = cells (zeros included), dots = unit means, tick = median
# ============================================================================
OFFSETS = {2: [-0.2, 0.2], 3: [-0.27, 0.0, 0.27], 4: [-0.33, -0.11, 0.11, 0.33]}
VW = {2: 0.36, 3: 0.25, 4: 0.2}                    # violin width per slot, by arm count
LEVELS = {"B": [0, 1, 2], "C": [0, 0, 1]}          # bracket stacking; C's two within-layer pairs
TXT = {"up": UP, "down": DOWN}                     # left arm higher | right arm higher
PRETTY = {p: p.replace("_", " ") for p in PROGRAMS}
SCORE_YLAB = {LAMINA: "log2 LMNA /\n(LMNB1 + LMNB2)"}


def _stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def sig(q):
    return "n.t." if not np.isfinite(q) else (_stars(q) or "n.s.")


def pct(x):
    if not np.isfinite(x):
        return "–"
    return f"{x:.0%}" if x >= 0.01 else f"{x:.1%}" if x >= 0.001 else "<0.1%"


def _clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=2, pad=1.5)


def _violin(ax, v, pos, color, width):
    v = np.asarray(v)[np.isfinite(v)]
    if len(v) and np.ptp(v) > 0:
        b = ax.violinplot([v], positions=[pos], widths=width, showextrema=False)["bodies"][0]
        b.set_facecolor(color)
        b.set_alpha(0.3)
        b.set_edgecolor("none")


def _room(ax, lo, hi, frac=0.72):
    """Data in the bottom `frac` of the axes; stars and brackets go in the rest."""
    span = (hi - lo) or 1.0
    bottom = lo - 0.04 * span
    ax.set_ylim(bottom, bottom + (hi - bottom) / frac)


def slot(ax, x, row, feature, R):
    """One feature in one row at x: per-arm violin + unit-mean dots + median tick, then the
    row's tests -- A: star and biopsy votes above the slot; B/C: one bracket per test.
    Returns the (lo, hi) the slot drew, for the shared y limit."""
    arms = ROWS[row]
    w = VW[len(arms)]
    pos = {a: x + d for a, d in zip(arms, OFFSETS[len(arms)])}
    t = R[(R["row"] == row) & (R["feature"] == feature)]
    rng = np.random.default_rng(SEED)
    per_cell = feature in E.columns
    lo, hi = ([0.0] if per_cell else []), []
    if row == "A":                                  # the paired biopsies, one line each
        r = t.iloc[0]
        vals = {"mal": r["vals_a"], "ben": r["vals_b"]}
        for va, vb in zip(r["vals_a"], r["vals_b"]):
            ax.plot([pos["mal"], pos["ben"]], [va, vb], color="#9e9e9e", linewidth=0.4,
                    alpha=0.5, zorder=2)
    else:
        vals = {a: unit_means(a, KEY[row])[0][feature].dropna().to_numpy() for a in arms}
    for a in arms:
        if per_cell:
            c = E[feature].to_numpy()[SUB[a]]
            _violin(ax, c, pos[a], ARM_COL[a], w)
            if (c > 0).sum() >= 20:
                hi.append(float(np.quantile(c[c > 0], 0.98)))
        v = np.asarray(vals[a], float)
        jit = np.zeros(len(v)) if row == "A" else rng.uniform(-0.2 * w, 0.2 * w, len(v))
        ax.scatter(pos[a] + jit, v, s=5, color=ARM_COL[a], edgecolor="white", linewidth=0.2,
                   zorder=3)
        if len(v):
            ax.hlines(np.median(v), pos[a] - 0.35 * w, pos[a] + 0.35 * w, color="#111111",
                      linewidth=1.1, zorder=4)
            lo.append(float(v.min()))
            hi.append(float(v.max()))

    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for k, r in enumerate(t.itertuples()):
        col = TXT.get(r.direction, "#333333")
        if row == "A":
            ax.text(x, 0.97, sig(r.q), transform=tr, ha="center", va="top", fontsize=6.5,
                    color=col)
            if np.isfinite(r.q):
                ax.text(x, 0.86, f"{r.n_pos:.0f}↑/{r.n_neg:.0f}↓", transform=tr, ha="center",
                        va="top", fontsize=5, color="#555555")
            continue
        x0, x1 = pos[r.arm_a], pos[r.arm_b]
        y = 0.76 + 0.08 * LEVELS[row][k]
        ax.plot([x0, x0, x1, x1], [y - 0.02, y, y, y - 0.02], transform=tr, color="#555555",
                linewidth=0.5, clip_on=False)
        ax.text((x0 + x1) / 2, y, sig(r.q), transform=tr, ha="center", va="bottom", fontsize=5,
                color=col)
    return (min(lo) if lo else 0.0), (max(hi) if hi else 1.0)


def det_ticks(ax, R, genes):
    """% of cells detected under each gene tick: malignant (purple) | benign (blue), MF."""
    r = R[R["row"] == "A"].set_index("feature")
    for j, g in enumerate(genes):
        for k, (v, col) in enumerate(((r.loc[g, "det_a"], MALIG), (r.loc[g, "det_b"], REACT))):
            ax.text(j, 0, pct(v), ha="center", va="top", fontsize=5, color=col,
                    transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                                          y=-13 - 7 * k, units="points"))
    ax.text(-0.55, 0, "% cells\ndetected", ha="right", va="top", fontsize=5, color="#555555",
            transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-13,
                                  units="points"))


def row_title(row, R):
    s = R[(R["row"] == row) & R["feature"].str.startswith("score_")]
    if row == "A":
        return (f"A  malignant vs benign CD4, paired within MF biopsy "
                f"({int(s['n_a'].iloc[0])} biopsies)")
    if row == "B":
        n = {a: len(unit_means(a, "donor")[0]) for a in ROWS["B"]}
        return (f"B  MF malignant | MF benign | healthy-donor CD4, donor means "
                f"({n['mal']} | {n['ben']} | {n['hc']} donors)")
    dis = (o.loc[o["donor"].isin(LAYER_DONORS)].groupby("donor")["disease"].first()
           .value_counts())
    return (f"C  epidermis vs dermis, paired within donor ({len(LAYER_DONORS)} layer-split "
            f"donors: {', '.join(f'{n} {d}' for d, n in dis.items())})")


def finding(p, R):
    """Suptitle from the program-score tests: the finding, not the figure's name."""
    s = R[R["feature"] == f"score_{p}"].set_index(["row", "arm_a", "arm_b"])["direction"]

    def rel(key, left, right):
        d = s.get(key, "not tested")
        return {"up": f"{left} > {right}", "down": f"{left} < {right}",
                "ns": f"{left} ≈ {right}"}.get(d, f"{left} vs {right} n.t.")

    return (f"{PRETTY[p]}: " + " · ".join([
        rel(("A", "mal", "ben"), "malignant", "benign CD4 (same biopsy)"),
        rel(("B", "mal", "hc"), "malignant", "healthy CD4"),
        rel(("C", "epi_mal", "derm_mal"), "epidermal", "dermal malignant")]))


def caption(fig, text):
    """Wrapped caption under everything already drawn (28b)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight() - 0.01,
             textwrap.fill(text, int(fig.get_figwidth() * 21)), fontsize=5.8, color="#555555",
             ha="left", va="top")


NOTE = (f"Violin = cells (≤ {MAX_CELLS // 1000}k per arm, zeros included); dots = unit means "
        f"over ≥ {MIN_CELLS} cells (A: MF biopsy, a line joins the two arms of one biopsy; B: "
        f"donor; C: donor × layer); tick = median. A, C: Wilcoxon signed-rank on the paired unit "
        f"differences (A: n↑/n↓ = biopsies higher in malignant / in benign); B: donor-level "
        f"Mann-Whitney. BH over every test in this figure (* < 0.05, ** < 0.01, *** < 0.001); "
        f"teal = left arm higher, rust = right arm higher; n.t. = < {MIN_DET:.0%} of cells "
        f"detected in both arms or < {MIN_N} units. Call: mal_tcr; CDR3-less CD4 excluded. RNA "
        f"presence, not measured motility.")


def program_figure(p):
    R = RES[RES["program"] == p]
    genes = PROGRAMS[p]
    G = len(genes)
    fig = plt.figure(figsize=(1.8 + 0.95 * (G + 1), 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, G], wspace=2.0 / (G + 1), hspace=0.8,
                          left=0.1, right=0.98, top=0.9, bottom=0.1)
    for i, row in enumerate(ROWS):
        ax_s, ax_g = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        _room(ax_s, *slot(ax_s, 0, row, f"score_{p}", R))
        ext = [slot(ax_g, j, row, g, R) for j, g in enumerate(genes)]
        _room(ax_g, min(e[0] for e in ext), max(e[1] for e in ext))
        ax_s.set_xticks(OFFSETS[len(ROWS[row])])
        ax_s.set_xticklabels([ARM_SHORT[a] for a in ROWS[row]], rotation=90, fontsize=5.5)
        ax_s.set_xlim(-0.5, 0.5)
        ax_s.set_ylabel(SCORE_YLAB.get(p, "program score\nmean log1p CP10K"), fontsize=6)
        ax_g.set_xticks(range(G))
        ax_g.set_xticklabels(genes, style="italic", fontsize=6.5)
        ax_g.set_xlim(-0.5, G - 0.5)
        ax_g.set_ylabel("log1p CP10K", fontsize=6)
        if row == "A":
            det_ticks(ax_g, R, genes)
        ax_s.text(0, 1.1, row_title(row, R), transform=ax_s.transAxes, ha="left", va="bottom",
                  fontsize=7, fontweight="bold")
        for ax in (ax_s, ax_g):
            _clean(ax)
    fig.suptitle(finding(p, R), fontsize=9, x=0.01, ha="left", y=0.985)
    caption(fig, NOTE + (" Lamina score = log2 of the ratio of unit-mean CP10K; lower = more "
                         "deformable nucleus." if p == LAMINA else ""))
    save_fig(fig, p)


print("drawing utilities ready")
```

```
drawing utilities ready
```

```python  # cell 6
# ============================================================================
# F  One figure per program — score first, then every gene; rows A, B, C
# ============================================================================
for p in PROGRAMS:
    program_figure(p)
```

```
<Figure size 940x820 with 6 Axes>
  -> amoeboid/amb28c_osmo_engine.png/.pdf

<Figure size 1130x820 with 6 Axes>
  -> amoeboid/amb28c_actomyosin.png/.pdf

<Figure size 1130x820 with 6 Axes>
  -> amoeboid/amb28c_actin_piston.png/.pdf

<Figure size 655x820 with 6 Axes>
  -> amoeboid/amb28c_nuclear_repair.png/.pdf

<Figure size 655x820 with 6 Axes>
  -> amoeboid/amb28c_mechanosensor.png/.pdf

<Figure size 560x820 with 6 Axes>
  -> amoeboid/amb28c_nuclear_lamina.png/.pdf
```

```python  # cell 7
# ============================================================================
# §R  What was written
# ============================================================================
print("tables ->", TAB_DIR)
for n in ("arms", "tests", "unit_means"):
    print("  ", tab28c(n).name)
print("figures ->", FIG_DIR)
for p in PROGRAMS:
    print(f"   {PREFIX}{p}.png/.pdf")
print("\nRNA presence != migration: every number above is machinery, not measured motility.")
```

```
tables -> /home/projects/nyosef/zvise/MF/tables
   amb28c_arms.csv
   amb28c_tests.csv
   amb28c_unit_means.csv
figures -> /home/projects/nyosef/zvise/MF/figures/amoeboid
   amb28c_osmo_engine.png/.pdf
   amb28c_actomyosin.png/.pdf
   amb28c_actin_piston.png/.pdf
   amb28c_nuclear_repair.png/.pdf
   amb28c_mechanosensor.png/.pdf
   amb28c_nuclear_lamina.png/.pdf

RNA presence != migration: every number above is machinery, not measured motility.
```

```python  # cell 8
# ============================================================================
# §LLM  Plain-text report — the design and every number the figures draw, for an external reader
# ============================================================================
ARM_LABEL = {"mal": "MF malignant CD4", "ben": "MF benign CD4", "hc": "healthy-donor CD4",
             "epi_mal": "epidermal malignant CD4", "epi_ben": "epidermal benign CD4",
             "derm_mal": "dermal malignant CD4", "derm_ben": "dermal benign CD4"}
ROW_WHAT = {
    "A": "malignant vs benign CD4 in the SAME MF biopsy; unit = biopsy; Wilcoxon signed-rank on "
         "the per-biopsy difference (holds donor, study, chemistry and site fixed)",
    "B": "MF malignant | MF benign | healthy-donor CD4 across donors; unit = donor; two-sided "
         "Mann-Whitney on donor means, 3 pairs",
    "C": "by skin layer in layer-split donors; unit = donor x layer; Wilcoxon signed-rank paired "
         "within donor: epi malignant vs epi benign, derm malignant vs derm benign, epi "
         "malignant vs derm malignant",
}
LIB = pd.Series(lib, index=CELLS)


def _g(x, fmt=".3f"):
    return "NA" if x is None or not np.isfinite(x) else format(x, fmt)


out = []
P = out.append
P("=" * 100)
P("NOTEBOOK 28c — AMOEBOID (NON-PROTEOLYTIC) MOTILITY MACHINERY IN MALIGNANT vs BENIGN CD4 T CELLS,")
P("CUTANEOUS T-CELL LYMPHOMA (CTCL) SKIN — single-cell RNA-seq, atlas v2")
P("=" * 100)
P("\nQUESTION: does the malignant T-cell clone carry more of the amoeboid (non-proteolytic) "
  "migration machinery than benign CD4 T cells from the same biopsy, than healthy-skin CD4, and "
  "does that differ between epidermis and dermis? RNA presence of machinery components is "
  "measured, NOT motility. Intact amoeboid machinery is the normal T-cell baseline, so "
  "malignant ~= benign is a valid result, not a failure.")
P(f"\nDATA: skin T-cell object {T_OBJ.name} ({len(obs):,} cells, 42,347 genes); "
  f"{len(CELLS):,} cells used.")

P("\nARMS")
P("- Malignancy call = mal_tcr (ALICE TCR clonality): a CD4 cell is malignant if its CDR3 recurs "
  "in the patient's founder clone, benign if it has a recovered CDR3 outside the clone. CD4 "
  f"without a CDR3 ({int(LV.eq('CD4_unassessed').sum()):,} cells) is untested and EXCLUDED. "
  "Tregs are a separate cell type and excluded.")
P("- Healthy = every CD4 cell from healthy-donor skin (disease == HC), no TCR requirement.")
P("- Rows A and B use MF skin only. Row C uses layer-split donors: >= "
  f"{MIN_CELLS} malignant and >= {MIN_CELLS} benign CD4 in BOTH epidermis and dermis (any "
  "CTCL subtype).")
P("- Arm codes: " + "; ".join(f"{k} = {v}" for k, v in ARM_LABEL.items()))
_arms = ARMS.copy()
_arms["median_library_size"] = [int(LIB[MASK[a]].median()) for a in _arms["arm"]]
P(_arms.to_string(index=False))
P("\nLayer-split donors (cells per layer|class; in_C = used in row C):")
P(_ct.assign(in_C=_ct.index.isin(LAYER_DONORS)).to_string())

P("\nPROGRAMS (gene sets; components only — egress genes S1PR1/CCR7/CD69, calcineurin-NFAT, TEAD "
  "and dissociation immediate-early genes are excluded by assertion)")
for p, gs in PROGRAMS.items():
    how = ("score = log2( mean CP10K LMNA / (mean CP10K LMNB1 + LMNB2) ) per unit, pseudocount "
           f"{LAM_EPS} CP10K; lower = more deformable nucleus" if p == LAMINA
           else "score = per-cell mean of the genes' log1p CP10K")
    P(f"  {p}: {', '.join(gs)} | {how}")

P("\nMETHODS")
P("- Expression per cell: log1p(CP10K), library size summed over all 42,347 genes.")
P(f"- Replicate value: mean over the unit's cells (unit needs >= {MIN_CELLS} cells in that arm).")
for row, txt in ROW_WHAT.items():
    P(f"- Row {row}: {txt}.")
P(f"- Not testable: < {MIN_DET:.0%} of cells detected in both arms, fewer than {MIN_N} units per "
  f"arm, fewer than {MIN_N} nonzero paired differences, or no variance.")
P("- Multiple testing: Benjamini-Hochberg over all tests of ONE program (its score + genes, rows "
  "A-C) = one figure. Not corrected across programs.")
P("- Columns: med_a / med_b = median over units of the unit means for arm_a / arm_b (in paired "
  "rows only the paired units). delta = median of per-unit differences a - b (paired) or "
  "med_a - med_b (MWU). n_pos / n_neg = paired units with a > b / a < b. det_a / det_b = "
  "fraction of the arm's cells with >= 1 count (for a score: >= 1 gene of the program). "
  "direction: up = arm_a higher, down = arm_b higher (q < 0.05), ns = q >= 0.05, not tested = "
  "see reason.")
P("- Units: genes and non-lamina scores in log1p CP10K; nuclear_lamina score in log2 ratio.")

P("\nCAVEATS FOR INTERPRETATION")
P("- Row A is the confound-free contrast. Row B's healthy arm differs from MF arms in donor, study "
  "and chemistry (" + ARMS.set_index("arm").loc["hc", "tech"] + "), so MF-vs-healthy "
  "shifts may include batch effects.")
P("- Many genes are zero-inflated; detection fractions are printed next to every test. Malignant "
  "cells can differ in library size (median_library_size above): per-cell detection rises with "
  "depth even after CP10K normalization.")
P(f"- Row C has {len(LAYER_DONORS)} donors; li2024 donors contribute one biopsy split into "
  "epidermis and dermis, buus2025 donors several biopsies per layer.")

P("\n" + "=" * 100)
P("PROGRAM-SCORE OVERVIEW (delta, BH q, direction; rows = programs, columns = row:arm_a-vs-arm_b)")
P("=" * 100)
_sc = RES[RES["feature"].str.startswith("score_")].copy()
_sc["comparison"] = _sc["row"] + ":" + _sc["arm_a"] + "-vs-" + _sc["arm_b"]
_sc["cell"] = [f"{_g(r.delta, '+.3f')} q={_g(r.q, '.2g')} {r.direction}" for r in _sc.itertuples()]
with pd.option_context("display.width", 400, "display.max_columns", None):
    P(_sc.pivot(index="program", columns="comparison", values="cell")
      .reindex(index=list(PROGRAMS), columns=list(dict.fromkeys(_sc["comparison"])))
      .to_string())

P("\n" + "=" * 100)
P("DETECTION: % of cells with >= 1 count, per arm")
P("=" * 100)
_det = pd.DataFrame({a: DET.loc[MASK[a], GENES + [f"score_{p}" for p in PROGRAMS]].mean()
                     for a in ARM_MASK}).mul(100).round(1)
with pd.option_context("display.width", 400, "display.max_rows", None):
    P(_det.to_string())

P("\n" + "=" * 100)
P("ALL TESTS, PER PROGRAM FIGURE (figure file, title, then every test drawn)")
P("=" * 100)
_cols = ["row", "feature", "arm_a", "arm_b", "test", "n_a", "n_b", "n_pos", "n_neg", "med_a",
         "med_b", "delta", "det_a", "det_b", "p", "q", "direction", "reason"]
_fmt = {"med_a": "{:.3f}".format, "med_b": "{:.3f}".format, "delta": "{:+.3f}".format,
        "det_a": "{:.3f}".format, "det_b": "{:.3f}".format, "p": "{:.3g}".format,
        "q": "{:.3g}".format, "n_pos": lambda x: _g(x, ".0f"), "n_neg": lambda x: _g(x, ".0f")}
for p in PROGRAMS:
    R = RES[RES["program"] == p]
    P(f"\n--- {p}: {', '.join(PROGRAMS[p])} | figure {PREFIX}{p}.png")
    P(f"title: {finding(p, R)}")
    with pd.option_context("display.width", 400, "display.max_rows", None,
                           "display.max_colwidth", 60):
        P(R[_cols].to_string(index=False, formatters=_fmt))

P("\n" + "=" * 100)
P("SIGNIFICANT RESULTS IN WORDS (q < 0.05, BH within program)")
P("=" * 100)
for r in RES[RES["q"] < 0.05].itertuples():
    what = "PROGRAM SCORE" if r.feature.startswith("score_") else f"gene {r.feature}"
    hi_, lo_ = (r.arm_a, r.arm_b) if r.direction == "up" else (r.arm_b, r.arm_a)
    votes = (f"; {r.n_pos:.0f}/{r.n_a} units a>b, {r.n_neg:.0f} a<b"
             if r.test.startswith("paired") else f"; n = {r.n_a} vs {r.n_b} donors")
    P(f"  [{r.program}] {what}, row {r.row}: {ARM_LABEL[hi_]} > {ARM_LABEL[lo_]} "
      f"(a={r.arm_a} median {_g(r.med_a)}, b={r.arm_b} median {_g(r.med_b)}, "
      f"delta {_g(r.delta, '+.3f')}{votes}; q = {_g(r.q, '.2g')}; detected "
      f"{r.det_a:.0%} vs {r.det_b:.0%})")

_nt = RES[RES["direction"] == "not tested"]
P(f"\nNOT TESTABLE: {len(_nt)} of {len(RES)} tests")
for (f, reason), d in _nt.groupby(["feature", "reason"]):
    P(f"  {f}: {reason} [{', '.join(d['row'] + ':' + d['arm_a'] + '-' + d['arm_b'])}]")
print("\n".join(out))
```

```
====================================================================================================
NOTEBOOK 28c — AMOEBOID (NON-PROTEOLYTIC) MOTILITY MACHINERY IN MALIGNANT vs BENIGN CD4 T CELLS,
CUTANEOUS T-CELL LYMPHOMA (CTCL) SKIN — single-cell RNA-seq, atlas v2
====================================================================================================

QUESTION: does the malignant T-cell clone carry more of the amoeboid (non-proteolytic) migration machinery than benign CD4 T cells from the same biopsy, than healthy-skin CD4, and does that differ between epidermis and dermis? RNA presence of machinery components is measured, NOT motility. Intact amoeboid machinery is the normal T-cell baseline, so malignant ~= benign is a valid result, not a failure.

DATA: skin T-cell object skin_T_tcr_annotated_v4.h5ad (539,916 cells, 42,347 genes); 145,062 cells used.

ARMS
- Malignancy call = mal_tcr (ALICE TCR clonality): a CD4 cell is malignant if its CDR3 recurs in the patient's founder clone, benign if it has a recovered CDR3 outside the clone. CD4 without a CDR3 (286,492 cells) is untested and EXCLUDED. Tregs are a separate cell type and excluded.
- Healthy = every CD4 cell from healthy-donor skin (disease == HC), no TCR requirement.
- Rows A and B use MF skin only. Row C uses layer-split donors: >= 25 malignant and >= 25 benign CD4 in BOTH epidermis and dermis (any CTCL subtype).
- Arm codes: mal = MF malignant CD4; ben = MF benign CD4; hc = healthy-donor CD4; epi_mal = epidermal malignant CD4; epi_ben = epidermal benign CD4; derm_mal = dermal malignant CD4; derm_ben = dermal benign CD4
     arm  n_cells  n_donors  n_samples                                tech  median_library_size
     mal    68843        33         33           10x_5p 68,685, ECCITE 158                 3633
     ben    59220        38         38            10x_5p 59,186, ECCITE 34                 1503
      hc     4742        22         20 10x_5p 4,169, 10x_3p 555, ECCITE 18                 2096
 epi_mal    19827        10         14                       10x_5p 19,827                 3692
 epi_ben     5541        10         14                        10x_5p 5,541                 2085
derm_mal    22478        10         14                       10x_5p 22,478                 3092
derm_ben    42312        10         14                       10x_5p 42,312                 1359

Layer-split donors (cells per layer|class; in_C = used in row C):
                     Epid|mal  Epid|ben  Derm|mal  Derm|ben disease   in_C
donor                                                                     
B5__PT09                 2873      2063      2094       513      SS   True
B5__PT22                    0      3027         0      1972      SS  False
B5__PT23                  129       291      3409       885      SS   True
Li2024_atlas__CTCL1      2023        95      4705       269      MF   True
Li2024_atlas__CTCL2       858        50       299      1796      MF   True
Li2024_atlas__CTCL3      2713        28      4743      1618      MF   True
Li2024_atlas__CTCL4       692       740        44      5252      MF   True
Li2024_atlas__CTCL5      4384       411      1963      3044      MF   True
Li2024_atlas__CTCL6      2837       422       267      7299      MF   True
Li2024_atlas__CTCL7      2786      1367       290     19871      MF   True
Li2024_atlas__CTCL8       532        74      4664      1765      MF   True

PROGRAMS (gene sets; components only — egress genes S1PR1/CCR7/CD69, calcineurin-NFAT, TEAD and dissociation immediate-early genes are excluded by assertion)
  osmo_engine: WNK1, OXSR1, STK39, SLC12A2, AQP3, AQP1, AQP9 | score = per-cell mean of the genes' log1p CP10K
  actomyosin: RHOA, ROCK1, ROCK2, MYH9, MYH10, PLCB1, PLCB2, PLCB3, PLCB4 | score = per-cell mean of the genes' log1p CP10K
  actin_piston: WAS, WASL, ACTR2, ACTR3, ARPC1B, ARPC2, ARPC3, ARPC4, ARPC5 | score = per-cell mean of the genes' log1p CP10K
  nuclear_repair: CHMP4B, CHMP2B, VPS4A, VPS4B | 
... [86022 more chars truncated]
```
