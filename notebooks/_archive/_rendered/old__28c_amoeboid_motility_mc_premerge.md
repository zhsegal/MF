# old/28c_amoeboid_motility_mc_premerge.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28c_amoeboid_motility_mc_premerge.ipynb` and is never run._

# 28c_mc — amoeboid motility machinery in malignant T cells, on the cluster-level call

A fresh copy of `28c_amoeboid_motility` with one change of input: malignant vs benign CD4 come
from **`mal_cluster`** (`26b_cluster_malignancy`) instead of `mal_tcr`. 28c is left untouched for
a later comparison. The outputs are `amb28cmc_*`.

Does the malignant clone carry more of the non-proteolytic (amoeboid) migration machinery than
benign CD4 T cells? The degradome notebooks (28, 28b) ask the proteolytic question; this one asks
the other. One figure per program: **program score first, then one violin slot per gene**, with
three comparisons as rows.

**RNA presence ≠ migration.** Everything here measures machinery, not motility.

| row | comparison | replicate | test |
|---|---|---|---|
| A | malignant vs benign CD4, same MF biopsy | biopsy (`sample_id`) | paired Wilcoxon signed-rank |
| B | MF malignant \| MF benign \| healthy-donor CD4 | donor | donor-level Mann-Whitney, 3 pairs |
| C | epidermis malignant vs benign, dermis malignant vs benign, epidermis vs dermis malignant | donor × layer | paired Wilcoxon signed-rank within donor |
| A-dm | row A on counts **thinned to one common depth** (sensitivity, not drawn) | biopsy | paired Wilcoxon signed-rank |
| A·early / A·adv | `<program>_stage` figure: row A inside early / advanced MF biopsies (`stage_class`) | biopsy | paired Wilcoxon signed-rank |
| B·stage | malignant and benign early vs advanced (pooled across studies), each malignant stage vs healthy; the malignant − benign gap advanced vs early | donor | donor-level Mann-Whitney |

## Arms

- **Call:** `mal_cluster` (`malignancy.load_malignancy` + `cd4_level(call="mal_cluster")`). Each
  sample's T cells are Leiden-clustered and each cluster is called; a cell takes its cluster's call.
  CD4 with no call are excluded. Tregs are not CD4 here.
- **Call provenance:** in the arms table, per arm (tcr = ALICE-decided, auto = program + V skew,
  user = manual).
- **Circularity:** none of the motility genes is in the program that calls `auto` clusters.
- **A, B** are MF skin only. **Healthy** = all atlas HC CD4 (`disease == HC`), its own arm.
- **C** uses every donor with ≥ `MIN_CELLS` malignant and benign CD4 in both Epidermis and Dermis.
  The cluster call reaches donors without V(D)J, so C can include more donors than 28c had.

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

- **Values:** per-cell log1p CP10K (library size over all 42,347 genes). A program score is the
  plain per-cell mean of its genes. The lamina score is computed per unit from mean CP10K.
- **Replicate:** unit mean over ≥ `cfg.MIN_CELLS` cells — biopsy in A, donor in B, donor × layer in C.
- **Not testable (n.t.):** < `cfg.MIN_DETECTION` of cells detected in both arms, or
  < `cfg.MIN_DONORS_PER_ARM` units (for paired tests: nonzero paired deltas).
- **Correction:** BH over every test drawn in one program's figure.
- **Stage (`*_stage`):** `stage_class` (early IA–IIA, advanced IIB+), MF only; samples of unknown
  stage are left out. Early vs advanced is pooled across studies. Row C is not split (too few
  advanced MF layer-split donors; counts in §3c).
- **Depth control (A-dm):** 28c found malignant libraries ~2.4× deeper than benign, and detection
  rises with depth even after CP10K.
  - Every MF malignant and benign cell is binomially thinned to one common depth (cells below it
    are dropped), re-scored, and row A is re-tested.
  - A program whose A result survives thinning is not a depth artefact.

## Guardrails

- Intact amoeboid machinery is the T-cell baseline, not a malignant gain. Where malignant ≈ benign,
  that is the result.
- Components only. S1PR1, CCR7, CD69, PPP3C\*, NFATC\*, TEAD\* and the IEG set are asserted absent.

## Compute

§2 reads a subset of the 12 GB skin T object. Run on a compute node (`jobs/run_notebook.sh`),
never the login node.

## Outputs

- `figures/amoeboid/amb28cmc_<program>.{png,pdf}`, `amb28cmc_<program>_stage.{png,pdf}`
- `tables/amb28cmc_tests.csv`, `amb28cmc_tests_depthmatched.csv`
- `tables/amb28cmc_unit_means.csv`, `amb28cmc_arms.csv`
- `tables/amb28cmc_tests_stage.csv`, `amb28cmc_arms_stage.csv`

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
import malignancy as MAL                            # mal_cluster
import cluster_malig_helpers as CM                  # call provenance + the calling program
from tcr_signaling_helpers import DISSOC_LABILE     # the project's IEG / dissociation set

DATA = NB_DIR / "data"
T_OBJ = DATA / "atlas_joint" / "skin_T_tcr_annotated_v4.h5ad"     # v2 skin T, all genes
FIG_DIR = NB_DIR / "figures" / "amoeboid"
TAB_DIR = NB_DIR / "tables"
PREFIX = "amb28cmc_"                                # never over 28c's amb28c_*
ARM_CALL = "mal_cluster"

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
# A gene the cluster call was drawn on would make its malignant-vs-benign difference circular.
IN_CALL_PROGRAM = sorted(set(GENES) & set(pd.read_csv(CM.PROGRAM_CSV)["gene"]))

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

# Stage companions (<program>_stage figures): rows A and B split by stage_class (early IA-IIA |
# advanced IIB+), MF only; samples of unknown stage are left out. Row C is not split (§3c).
STAGES = ["early", "advanced"]
STAGE_COL = {"early": "#e8a33d", "advanced": "#c0392b"}
ROWS_S = {"Ae": ["mal_early", "ben_early"],
          "Aa": ["mal_adv", "ben_adv"],
          "Bs": ["mal_early", "mal_adv", "ben_early", "ben_adv", "hc"]}
KEY_S = {"Ae": "sample_id", "Aa": "sample_id", "Bs": "donor"}
TESTS_S = {"Ae": [("mal_early", "ben_early", "paired")],
           "Aa": [("mal_adv", "ben_adv", "paired")],
           "Bs": [("mal_early", "mal_adv", "mwu"), ("ben_early", "ben_adv", "mwu"),
                  ("mal_early", "hc", "mwu"), ("mal_adv", "hc", "mwu")]}
PAIRED_ROWS = {"A", "Ae", "Aa"}
ARM_STAGE = {"mal_early": "early", "ben_early": "early", "mal_adv": "advanced",
             "ben_adv": "advanced"}
ARM_COL.update({"mal_early": MALIG_L, "mal_adv": MALIG, "ben_early": REACT_L,
                "ben_adv": REACT})
ARM_SHORT.update({"mal_early": "Malig. early", "mal_adv": "Malig. adv.",
                  "ben_early": "Benign early", "ben_adv": "Benign adv."})

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
      f"{MIN_DET:.0%} detection, {MIN_N} units | call {ARM_CALL} | program genes in the "
      f"calling program: {IN_CALL_PROGRAM or 'none'}")
```

```
NB_DIR /home/projects/nyosef/zvise/MF
6 programs, 36 genes | gates: 25 cells/unit, 1% detection, 4 units | call mal_cluster | program genes in the calling program: none
```

```python  # cell 2
# ============================================================================
# §1  Arms — mal_cluster, MF skin for A/B, healthy CD4, layer-split donors for C
# ============================================================================
# obs only here; the matrix is read in §2. v1/v2 guard: the v2 skin T object is 42,347 genes.
tb = sc.read_h5ad(T_OBJ, backed="r")
assert tb.n_vars == 42_347, f"{T_OBJ.name} has {tb.n_vars} genes: not the v2 object"
OBS_COLS = ["cell_id", "donor", "sample_id", "study", "disease", "tissue", "tech", "cell_type_T",
            "stage_class"]
obs = tb.obs[OBS_COLS].astype(str).copy()
HAS_TCR = tb.obs["has_tcr"].astype(str).str.lower().eq("true").to_numpy()
print(f"{T_OBJ.name}: {tb.n_obs:,} cells x {tb.n_vars:,} genes")

# malignant / benign CD4 on mal_cluster (no call -> CD4_unassessed, excluded), and healthy CD4 as
# an arm of its own rather than folded into "benign" (HC is benign by construction in the call).
mal = MAL.load_malignancy(obs["cell_id"], has_tcr=HAS_TCR, verbose=False)
LV = pd.Series(np.asarray(MAL.cd4_level(obs["cell_type_T"], mal, call=ARM_CALL)).astype(str),
               index=obs.index)
LV[(obs["cell_type_T"] == "CD4") & (obs["disease"] == "HC")] = HC_LV
obs["arm_level"] = LV
obs["layer"] = obs["tissue"].where(obs["tissue"].isin(LAYERS), "")
_src = pd.read_parquet(CM.CALLS_PQ, columns=["cell_id", "call_source"]).set_index("cell_id")
obs["call_source"] = _src["call_source"].reindex(obs["cell_id"].to_numpy()).fillna("none").to_numpy()

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
                      "call_source": ", ".join(f"{s} {n / m.sum():.0%}" for s, n in
                                               obs.loc[m, "call_source"].value_counts().items()),
                      "tech": ", ".join(f"{t} {n:,}" for t, n in
                                        obs.loc[m, "tech"].value_counts().items())}
                     for a, m in ARM_MASK.items()])
ARMS.to_csv(tab28c("arms"), index=False)
print(f"\nCD4 on {ARM_CALL}: malignant {int(IS_MAL.sum()):,} | benign {int(IS_BEN.sum()):,} | "
      f"no call (excluded) {int(LV.eq('CD4_unassessed').sum()):,} | "
      f"healthy {int(LV.eq(HC_LV).sum()):,}")
print(ARMS.to_string(index=False))

_ct.columns = [f"{l[:4]}|{'mal' if a == MALIG_LV else 'ben'}" for l, a in _ct.columns]
_ct["disease"] = _c.groupby("donor")["disease"].first()
print(f"\nlayer-split donors (C): {len(LAYER_DONORS)} "
      f"({', '.join(f'{d} {n}' for d, n in _ct.loc[LAYER_DONORS, 'disease'].value_counts().items())})")
print(_ct.assign(in_C=_ct.index.isin(LAYER_DONORS)).to_string())

# Stage arms (<program>_stage): rows A and B split by stage_class, MF only. Subsets of mal /
# ben, so LOAD already holds them; kept out of ARM_MASK so arms.csv is unchanged.
STAGE_ARM_MASK = {f"{a}_{s}": IS_MF & m & obs["stage_class"].eq(st)
                  for st, s in (("early", "early"), ("advanced", "adv"))
                  for a, m in (("mal", IS_MAL), ("ben", IS_BEN))}
_smp_stage = obs.loc[IS_MF & (IS_MAL | IS_BEN)].groupby("sample_id")["stage_class"].first()
N_NOSTAGE = int((~_smp_stage.isin(STAGES)).sum())
ARMS_S = pd.DataFrame([{
    "arm": a, "n_cells": int(m.sum()), "n_donors": obs.loc[m, "donor"].nunique(),
    "n_samples": obs.loc[m, "sample_id"].nunique(),
    "study": ", ".join(f"{s} {n}" for s, n in
                       obs.loc[m].groupby("sample_id")["study"].first().value_counts().items())}
    for a, m in STAGE_ARM_MASK.items()])
ARMS_S.to_csv(tab28c("arms_stage"), index=False)
print(f"\nstage arms (stage_class, MF; {N_NOSTAGE} MF samples with a call and unknown stage "
      "left out):")
print(ARMS_S.to_string(index=False))
```

```
skin_T_tcr_annotated_v4.h5ad: 539,916 cells x 42,347 genes

CD4 on mal_cluster: malignant 221,388 | benign 177,160 | no call (excluded) 39,692 | healthy 4,742
     arm  n_cells  n_donors  n_samples                call_source                                                      tech
     mal   185643        84         92 auto 53%, tcr 45%, user 2% 10x_5p 161,364, 10x_flex 20,613, 10x_3p 3,334, ECCITE 332
     ben   137664        86         98 tcr 64%, auto 31%, user 5%  10x_5p 127,139, 10x_3p 6,086, 10x_flex 4,137, ECCITE 302
      hc     4742        22         20           hc 81%, none 19%                       10x_5p 4,169, 10x_3p 555, ECCITE 18
 epi_mal    25254        10         14  tcr 96%, auto 3%, user 1%                                             10x_5p 25,254
 epi_ben     7921        10         14 tcr 82%, user 10%, auto 8%                                              10x_5p 7,921
derm_mal    33147        10         14  tcr 96%, user 3%, auto 0%                                             10x_5p 33,147
derm_ben    66085        10         14           tcr 98%, auto 2%                                             10x_5p 66,085

layer-split donors (C): 10 (MF 8, SS 2)
                     Epid|mal  Epid|ben  Derm|mal  Derm|ben disease   in_C
donor                                                                     
B5__PT09                 4867      3591      5560      1682      SS   True
B5__PT22                   10      4275         0      3362      SS  False
B5__PT23                  165       427      6206       820      SS   True
Li2024_atlas__CTCL1      2513       105      5163       247      MF   True
Li2024_atlas__CTCL2       968        46       389      2154      MF   True
Li2024_atlas__CTCL3      2776        31      6047      1747      MF   True
Li2024_atlas__CTCL4      1142       917        99      6989      MF   True
Li2024_atlas__CTCL5      5094       407      2438      3666      MF   True
Li2024_atlas__CTCL6      3462       550       475      9748      MF   True
Li2024_atlas__CTCL7      3088      1721       570     36711      MF   True
Li2024_atlas__CTCL8      1179       126      6200      2321      MF   True

stage arms (stage_class, MF; 16 MF samples with a call and unknown stage left out):
      arm  n_cells  n_donors  n_samples                                                                                                                         study
mal_early    38329        19         19                                                                                      li2024 15, rindler2021_mc 2, alkon2024 2
ben_early    71186        19         19                                                                                      li2024 15, rindler2021_mc 2, alkon2024 2
  mal_adv   130459        50         58  li2024 18, brentuximab2026 12, rindler2021_mc 7, chennareddy2025 7, herrera2021 5, gaydosik2019 5, gaydosik2022 3, lyp2026 1
  ben_adv    53700        51         63 li2024 19, brentuximab2026 13, rindler2021_mc 10, chennareddy2025 7, herrera2021 5, gaydosik2019 5, gaydosik2022 3, lyp2026 1
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
# stage arms last, so the violin subsamples of the original arms below are drawn unchanged
MASK = {a: m.loc[CELLS].to_numpy() for a, m in {**ARM_MASK, **STAGE_ARM_MASK}.items()}
CP10K = pd.DataFrame(RAW / np.maximum(lib, 1)[:, None] * 1e4, index=CELLS, columns=GENES)
E = np.log1p(CP10K)
DET = pd.DataFrame(RAW > 0, index=CELLS, columns=GENES)
# Program score = plain per-cell mean of its genes (28b family-score convention). The lamina
# score is a ratio of unit means (§3), so it has no per-cell value.
for p, gs in PROGRAMS.items():
    DET[f"score_{p}"] = DET[gs].any(axis=1)
    if p != LAMINA:
        E[f"score_{p}"] = E[gs].mean(axis=1)

# Depth-matched copy for the row-A sensitivity. Malignant libraries run deeper than benign ones,
# and detection rises with depth even after CP10K. Every MF malignant and benign cell is thinned
# binomially to one common depth (cells below it are dropped), so both arms are scored on the
# same sampling effort.
_A = MASK["mal"] | MASK["ben"]
DM_DEPTH = int(max(500, np.quantile(lib[_A], 0.10)))
DM_KEEP = _A & (lib >= DM_DEPTH)
_thin = np.random.default_rng(SEED).binomial(RAW[DM_KEEP].astype(np.int64),
                                             (DM_DEPTH / lib[DM_KEEP])[:, None])
CP10K_DM = pd.DataFrame(_thin / DM_DEPTH * 1e4, index=CELLS[DM_KEEP], columns=GENES)
E_DM = np.log1p(CP10K_DM)
for p, gs in PROGRAMS.items():
    if p != LAMINA:
        E_DM[f"score_{p}"] = E_DM[gs].mean(axis=1)
del RAW, _thin

# Fixed per-arm subsample for the violin KDE; every feature of an arm draws the same cells.
_rng = np.random.default_rng(SEED)
SUB = {}
for a, m in MASK.items():
    idx = np.flatnonzero(m)
    SUB[a] = np.sort(_rng.choice(idx, MAX_CELLS, replace=False)) if len(idx) > MAX_CELLS else idx

print(f"loaded {len(CELLS):,} cells | median library size {np.median(lib):,.0f} | "
      f"malignant {np.median(lib[MASK['mal']]):,.0f} vs benign {np.median(lib[MASK['ben']]):,.0f}")
print(f"depth-matched copy: {int(DM_KEEP.sum()):,} of {int(_A.sum()):,} MF CD4 cells thinned to "
      f"{DM_DEPTH:,} UMIs")
print("% cells detected per gene, MF malignant | MF benign | healthy:")
_d = pd.DataFrame({a: DET.loc[MASK[a], GENES].mean() for a in ("mal", "ben", "hc")})
for p, gs in PROGRAMS.items():
    print(f"  {p:<15} " + "  ".join(f"{g} {r.mal:.0%}|{r.ben:.0%}|{r.hc:.0%}"
                                    for g, r in _d.loc[gs].iterrows()))
```

```
loaded 351,367 cells | median library size 2,300 | malignant 3,209 vs benign 1,517
depth-matched copy: 290,983 of 323,307 MF CD4 cells thinned to 837 UMIs
% cells detected per gene, MF malignant | MF benign | healthy:
  osmo_engine     WNK1 40%|30%|20%  OXSR1 20%|12%|10%  STK39 13%|5%|2%  SLC12A2 11%|4%|4%  AQP3 25%|18%|24%  AQP1 0%|0%|1%  AQP9 0%|0%|0%
  actomyosin      RHOA 75%|65%|55%  ROCK1 42%|26%|21%  ROCK2 10%|4%|4%  MYH9 59%|49%|39%  MYH10 4%|1%|1%  PLCB1 4%|2%|2%  PLCB2 12%|5%|4%  PLCB3 2%|1%|1%  PLCB4 0%|0%|0%
  actin_piston    WAS 41%|25%|19%  WASL 9%|5%|3%  ACTR2 45%|30%|25%  ACTR3 56%|39%|35%  ARPC1B 75%|50%|34%  ARPC2 77%|64%|54%  ARPC3 70%|60%|46%  ARPC4 48%|29%|22%  ARPC5 53%|33%|26%
  nuclear_repair  CHMP4B 19%|11%|7%  CHMP2B 23%|13%|9%  VPS4A 22%|12%|8%  VPS4B 30%|16%|11%
  mechanosensor   PIEZO1 17%|7%|5%  PIEZO2 0%|0%|0%  YAP1 0%|0%|0%  WWTR1 5%|1%|2%
  nuclear_lamina  LMNA 66%|34%|50%  LMNB1 29%|12%|7%  LMNB2 13%|4%|2%
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
   osmo_engine   A      mal      ben   88   88 0.207 0.194 +0.016   0.0055        up
   osmo_engine   B      mal      ben   83   86 0.205 0.192 +0.013    0.539        ns
   osmo_engine   B      ben       hc   86   21 0.192 0.151 +0.041  0.00823        up
   osmo_engine   B      mal       hc   83   21 0.205 0.151 +0.054  0.00528        up
   osmo_engine   C  epi_mal  epi_ben   10   10 0.177 0.198 +0.004        1        ns
   osmo_engine   C derm_mal derm_ben   10   10 0.195 0.187 +0.025    0.656        ns
   osmo_engine   C  epi_mal derm_mal   10   10 0.177 0.195 -0.022   0.0123      down
    actomyosin   A      mal      ben   88   88 0.404 0.361 +0.051 1.72e-08        up
    actomyosin   B      mal      ben   83   86 0.402 0.363 +0.039   0.0143        up
    actomyosin   B      ben       hc   86   21 0.363 0.245 +0.118 9.79e-05        up
    actomyosin   B      mal       hc   83   21 0.402 0.245 +0.156 2.29e-06        up
    actomyosin   C  epi_mal  epi_ben   10   10 0.466 0.410 +0.066    0.117        ns
    actomyosin   C derm_mal derm_ben   10   10 0.471 0.402 +0.061    0.176        ns
    actomyosin   C  epi_mal derm_mal   10   10 0.466 0.471 -0.012    0.954        ns
  actin_piston   A      mal      ben   88   88 0.926 0.789 +0.130 4.92e-10        up
  actin_piston   B      mal      ben   83   86 0.917 0.797 +0.120   0.0111        up
  actin_piston   B      ben       hc   86   21 0.797 0.508 +0.289 2.88e-05        up
  actin_piston   B      mal       hc   83   21 0.917 0.508 +0.409 6.05e-07        up
  actin_piston   C  epi_mal  epi_ben   10   10 0.970 0.930 +0.087   0.0902        ns
  actin_piston   C derm_mal derm_ben   10   10 0.915 0.762 +0.147   0.0185        up
  actin_piston   C  epi_mal derm_mal   10   10 0.970 0.915 +0.111    0.164        ns
nuclear_repair   A      mal      ben   88   88 0.263 0.193 +0.056 3.22e-12        up
nuclear_repair   B      mal      ben   83   86 0.270 0.219 +0.051  0.00139        up
nuclear_repair   B      ben       hc   86   21 0.219 0.115 +0.105 2.14e-05        up
nuclear_repair   B      mal       hc   83   21 0.270 0.115 +0.156 1.06e-07        up
nuclear_repair   C  epi_mal  epi_ben   10   10 0.414 0.296 +0.095   0.0383        up
nuclear_repair   C derm_mal derm_ben   10   10 0.391 0.227 +0.083    0.018        up
nuclear_repair   C  epi_mal derm_mal   10   10 0.414 0.391 +0.036   0.0218        up
 mechanosensor   A      mal      ben   88   88 0.051 0.033 +0.016 1.11e-07        up
 mechanosensor   B      mal      ben   83   86 0.050 0.037 +0.014  0.00809        up
 mechanosensor   B      ben       hc   86   21 0.037 0.018 +0.018 0.000555        up
 mechanosensor   B      mal       hc   83   21 0.050 0.018 +0.032  7.3e-06        up
 mechanosensor   C  epi_mal  epi_ben   10   10 0.045 0.033 +0.009    0.148        ns
 mechanosensor   C derm_mal derm_ben   10   10 0.041 0.028 +0.016   0.0342        up
 mechanosensor   C  epi_mal derm_mal   10   10 0.045 0.041 -0.009    0.584        ns
nuclear_lamina   A      mal      ben   88   88 2.932 2.538 +0.132    0.256        ns
nuclear_lamina   B      mal      ben   83   86 2.928 2.642 +0.286    0.695        ns
nuclear_lamina   B      ben       hc   86   21 2.642 3.921 -1.280   0.0125      down
nuclear_lamina   B      mal       hc   83   21 2.928 3.921 -0.994   0.0309      down
nuclear_lamina   C  epi_mal  epi_ben   10   10 3.208 2.895 +0.326    0.721        ns
nuclear_lamina   C derm_mal derm_ben   10   10 2.604 1.218 +1.210   0.0125        up
nuclear_lamina   C  epi_mal derm_mal   10   10 3.208 2.604 +0.425   0.0182        up

genes with q < 0.05:
       program row feature    arm_a    arm_b  n_a med_a med_b  delta        q direction
   osmo_engine   A    WNK1      mal      ben   88 0.500 0.501 +0.044   0.0265        up
   osmo_engine   B    WNK1      ben       hc   86 0.515 0.295 +0.220 0.000689        up
   osmo_e
... [10110 more chars truncated]
```

```python  # cell 5
# ============================================================================
# §3b  Row A on the depth-matched copy — does each result survive equal sampling depth?
# ============================================================================
# Same unit (biopsy), same gates, same paired signed-rank and same BH-within-program as row A;
# only the counts differ. Not drawn: this is the check that row A is not a depth artefact.
_dm_sample = o.loc[CP10K_DM.index, "sample_id"].to_numpy()
_dm_arm = np.where(MASK["mal"][DM_KEEP], "mal", "ben")


def _dm_unit_means(arm):
    m = _dm_arm == arm
    units = _dm_sample[m]
    n = pd.Series(units).value_counts()
    mean = E_DM[m].groupby(units).mean()
    lam = CP10K_DM.loc[m, PROGRAMS[LAMINA]].groupby(units).mean()
    mean[f"score_{LAMINA}"] = np.log2((lam["LMNA"] + LAM_EPS)
                                      / (lam["LMNB1"] + lam["LMNB2"] + LAM_EPS))
    return mean[n.reindex(mean.index).to_numpy() >= MIN_CELLS]


_um_mal, _um_ben = _dm_unit_means("mal"), _dm_unit_means("ben")
_both = _um_mal.index.intersection(_um_ben.index)
_dm = []
for p in PROGRAMS:
    recs = []
    for f in features(p):
        d = (_um_mal.loc[_both, f] - _um_ben.loc[_both, f]).to_numpy()
        nz = d[d != 0]
        recs.append({"program": p, "feature": f, "n_pairs": len(d),
                     "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                     "delta": np.median(d) if len(d) else np.nan,
                     "sign": np.sign((d > 0).sum() - (d < 0).sum()),
                     "p": float(wilcoxon(nz).pvalue) if len(nz) >= MIN_N else np.nan})
    _dm.append(bh(pd.DataFrame(recs)))
DM = pd.concat(_dm, ignore_index=True).assign(depth=DM_DEPTH)

_a = RES[RES["row"] == "A"].set_index("feature")[["delta", "q", "direction"]]
DM = DM.join(_a.add_suffix("_full_depth"), on="feature")
DM["survives"] = (DM["direction_full_depth"].isin(["up", "down"])
                  & (DM["direction"] == DM["direction_full_depth"]))
DM[["program", "feature", "depth", "n_pairs", "n_pos", "n_neg", "delta", "p", "q", "direction",
    "delta_full_depth", "q_full_depth", "direction_full_depth", "survives"]].to_csv(
    tab28c("tests_depthmatched"), index=False)

print(f"row A at {DM_DEPTH:,} UMIs per cell ({len(_both)} biopsies), program scores:")
print(DM.loc[DM["feature"].str.startswith("score_"),
             ["program", "n_pairs", "delta", "q", "direction", "delta_full_depth",
              "direction_full_depth", "survives"]].to_string(index=False, formatters=_fmt))
_sig_full = DM["direction_full_depth"].isin(["up", "down"])
print(f"\nrow-A results that survive depth matching: {int(DM['survives'].sum())} of "
      f"{int(_sig_full.sum())} significant at full depth -> {tab28c('tests_depthmatched').name}")
```

```
row A at 837 UMIs per cell (88 biopsies), program scores:
       program  n_pairs  delta       q direction  delta_full_depth direction_full_depth  survives
   osmo_engine       88 -0.013  0.0363      down          0.016068                   up     False
    actomyosin       88 +0.010  0.0482        up          0.051391                   up      True
  actin_piston       88 +0.037 0.00262        up          0.130210                   up      True
nuclear_repair       88 +0.012 0.00364        up          0.056215                   up      True
 mechanosensor       88 +0.004 0.00422        up          0.016353                   up      True
nuclear_lamina       88 +0.062   0.274        ns          0.131607                   ns     False

row-A results that survive depth matching: 17 of 34 significant at full depth -> amb28cmc_tests_depthmatched.csv
```

```python  # cell 6
# ============================================================================
# §3c  Stage tests — rows A and B split by stage_class, and the gap advanced vs early
# ============================================================================
# Ae / Aa: row A's paired signed-rank inside early and inside advanced biopsies. Bs: donor means,
# malignant and benign early vs advanced (pooled across studies) and each malignant stage vs
# healthy. Ad, written on the Aa row rather than drawn: the per-biopsy malignant - benign
# difference averaged per donor, advanced vs early, donor-level Mann-Whitney -- study and
# chemistry cancel inside the difference. BH over every test in one program's stage figure.
S2D = o.groupby("sample_id")["donor"].first()
LAYER_STAGE = (o.loc[o["donor"].isin(LAYER_DONORS) & o["disease"].eq("MF")]
               .groupby("donor")["stage_class"].first().value_counts())


def run_delta_test(feature):
    """Row Ad: advanced vs early on the donor-mean malignant - benign gap (a = advanced)."""
    gap, det = {}, []
    for st, (m, b) in {"adv": ("mal_adv", "ben_adv"), "early": ("mal_early", "ben_early")}.items():
        ma, da = unit_means(m, "sample_id")
        mb, db = unit_means(b, "sample_id")
        both = ma.index.intersection(mb.index)
        d = ma.loc[both, feature] - mb.loc[both, feature]
        gap[st] = d.groupby(S2D.reindex(both).to_numpy()).mean().dropna().to_numpy()
        det += [da[feature], db[feature]]
    x, y = gap["adv"], gap["early"]
    det = np.nanmax(det)
    rec = {"feature": feature, "arm_a": "gap_adv", "arm_b": "gap_early", "unit": "donor",
           "test": "MWU on gap", "det_a": det, "det_b": det, "p": np.nan, "reason": "",
           "n_pos": np.nan, "n_neg": np.nan, "n_a": len(x), "n_b": len(y),
           "med_a": np.median(x) if len(x) else np.nan,
           "med_b": np.median(y) if len(y) else np.nan}
    rec["delta"] = rec["med_a"] - rec["med_b"]
    rec["sign"] = np.sign(rec["delta"])
    if not det >= MIN_DET:
        rec["reason"] = f"below {MIN_DET:.0%} detection in every arm"
    elif min(len(x), len(y)) < MIN_N:
        rec["reason"] = f"too few units ({len(x)}, {len(y)}; need {MIN_N})"
    elif np.ptp(np.r_[x, y]) == 0:
        rec["reason"] = "no variance"
    else:
        rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    rec["vals_a"], rec["vals_b"] = x, y
    return rec


RES_S = pd.concat([bh(pd.DataFrame(
    [dict(run_test(f, a, b, kind, KEY_S[row]), program=p, row=row)
     for f in features(p) for row, ts in TESTS_S.items() for a, b, kind in ts]
    + [dict(run_delta_test(f), program=p, row="Ad") for f in features(p)]))
    for p in PROGRAMS], ignore_index=True)
RES_S[_cols].to_csv(tab28c("tests_stage"), index=False)

print("program scores by stage (BH within each program's stage figure):")
print(RES_S.loc[RES_S["feature"].str.startswith("score_"),
                ["program", "row", "arm_a", "arm_b", "n_a", "n_b", "med_a", "med_b", "delta", "q",
                 "direction"]].to_string(index=False, formatters=_fmt))
_s = RES_S[~RES_S["feature"].str.startswith("score_") & (RES_S["q"] < 0.05)]
print("\ngenes with q < 0.05:")
print(_s[["program", "row", "feature", "arm_a", "arm_b", "n_a", "n_b", "delta", "q",
          "direction"]].to_string(index=False, formatters=_fmt) if len(_s) else "  none")
_c_adv = int(LAYER_STAGE.get("advanced", 0))
print(f"\nrow C not split by stage; its MF layer-split donors: "
      f"{', '.join(f'{n} {s}' for s, n in LAYER_STAGE.items())}"
      + (f" (advanced {_c_adv} < {MIN_N})" if _c_adv < MIN_N else ""))
```

```
program scores by stage (BH within each program's stage figure):
       program row     arm_a     arm_b  n_a  n_b med_a med_b  delta        q direction
   osmo_engine  Ae mal_early ben_early   17   17 0.165 0.179 +0.008    0.206        ns
   osmo_engine  Aa   mal_adv   ben_adv   56   56 0.207 0.198 +0.019   0.0619        ns
   osmo_engine  Bs mal_early   mal_adv   19   49 0.210 0.193 +0.017    0.645        ns
   osmo_engine  Bs ben_early   ben_adv   19   51 0.191 0.188 +0.004    0.968        ns
   osmo_engine  Bs mal_early        hc   19   21 0.210 0.151 +0.059   0.0619        ns
   osmo_engine  Bs   mal_adv        hc   49   21 0.193 0.151 +0.042   0.0151        up
   osmo_engine  Ad   gap_adv gap_early   49   17 0.013 0.008 +0.005    0.968        ns
    actomyosin  Ae mal_early ben_early   17   17 0.418 0.385 +0.072  0.00546        up
    actomyosin  Aa   mal_adv   ben_adv   56   56 0.417 0.363 +0.045 3.97e-05        up
    actomyosin  Bs mal_early   mal_adv   19   49 0.404 0.425 -0.021    0.879        ns
    actomyosin  Bs ben_early   ben_adv   19   51 0.387 0.366 +0.021    0.913        ns
    actomyosin  Bs mal_early        hc   19   21 0.404 0.245 +0.159 0.000267        up
    actomyosin  Bs   mal_adv        hc   49   21 0.425 0.245 +0.180 1.45e-05        up
    actomyosin  Ad   gap_adv gap_early   49   17 0.048 0.072 -0.024    0.453        ns
  actin_piston  Ae mal_early ben_early   17   17 0.908 0.737 +0.165   0.0023        up
  actin_piston  Aa   mal_adv   ben_adv   56   56 1.016 0.822 +0.129 7.73e-07        up
  actin_piston  Bs mal_early   mal_adv   19   49 0.876 1.012 -0.135   0.0958        ns
  actin_piston  Bs ben_early   ben_adv   19   51 0.805 0.840 -0.035    0.352        ns
  actin_piston  Bs mal_early        hc   19   21 0.876 0.508 +0.368  0.00235        up
  actin_piston  Bs   mal_adv        hc   49   21 1.012 0.508 +0.503 1.61e-07        up
  actin_piston  Ad   gap_adv gap_early   49   17 0.138 0.165 -0.027    0.638        ns
nuclear_repair  Ae mal_early ben_early   17   17 0.313 0.259 +0.085  0.00267        up
nuclear_repair  Aa   mal_adv   ben_adv   56   56 0.262 0.189 +0.062 3.32e-08        up
nuclear_repair  Bs mal_early   mal_adv   19   49 0.313 0.270 +0.043    0.206        ns
nuclear_repair  Bs ben_early   ben_adv   19   51 0.269 0.205 +0.064    0.109        ns
nuclear_repair  Bs mal_early        hc   19   21 0.313 0.115 +0.199    5e-05        up
nuclear_repair  Bs   mal_adv        hc   49   21 0.270 0.115 +0.156 1.29e-06        up
nuclear_repair  Ad   gap_adv gap_early   49   17 0.062 0.085 -0.024    0.828        ns
 mechanosensor  Ae mal_early ben_early   17   17 0.033 0.029 +0.009     0.12        ns
 mechanosensor  Aa   mal_adv   ben_adv   56   56 0.054 0.038 +0.019 2.62e-05        up
 mechanosensor  Bs mal_early   mal_adv   19   49 0.040 0.054 -0.013    0.572        ns
 mechanosensor  Bs ben_early   ben_adv   19   51 0.032 0.038 -0.006    0.812        ns
 mechanosensor  Bs mal_early        hc   19   21 0.040 0.018 +0.022  0.00719        up
 mechanosensor  Bs   mal_adv        hc   49   21 0.054 0.018 +0.035 2.62e-05        up
 mechanosensor  Ad   gap_adv gap_early   49   17 0.021 0.009 +0.011    0.477        ns
nuclear_lamina  Ae mal_early ben_early   17   17 2.424 1.757 +0.197    0.611        ns
nuclear_lamina  Aa   mal_adv   ben_adv   56   56 3.028 2.333 +0.227    0.356        ns
nuclear_lamina  Bs mal_early   mal_adv   19   49 2.424 2.928 -0.503    0.226        ns
nuclear_lamina  Bs ben_early   ben_adv   19   51 1.982 2.489 -0.507     0.45        ns
nuclear_lamina  Bs mal_early        hc   19   21 2.424 3.921 -1.497   0.0344      down
nuclear_lamina  Bs   mal_adv        hc   49   21 2.928 3.921 -0.994    0.175        ns
nuclear_lamina  Ad   gap_adv gap_early   49   17 0.222 0.197 +0.025    0.725        ns

genes with q < 0.05:
       program row feature     arm_a     arm_b  n_a  n_b  delta        q direction
   osmo_engine  Aa    WNK1   mal_adv   ben_adv   56   56 +0.068   0.0115        up
   osm
... [8467 more chars truncated]
```

```python  # cell 7
# ============================================================================
# §4  Drawing — 28b's conventions: violin = cells (zeros included), dots = unit means, tick = median
# ============================================================================
OFFSETS = {2: [-0.2, 0.2], 3: [-0.27, 0.0, 0.27], 4: [-0.33, -0.11, 0.11, 0.33],
           5: [-0.36, -0.18, 0.0, 0.18, 0.36]}
VW = {2: 0.36, 3: 0.25, 4: 0.2, 5: 0.15}           # violin width per slot, by arm count
LEVELS = {"B": [0, 1, 2], "C": [0, 0, 1],          # bracket stacking; C's two within-layer pairs
          "Bs": [0, 0, 1, 2]}
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
    arms = {**ROWS, **ROWS_S}[row]
    w = VW[len(arms)]
    pos = {a: x + d for a, d in zip(arms, OFFSETS[len(arms)])}
    t = R[(R["row"] == row) & (R["feature"] == feature)]
    rng = np.random.default_rng(SEED)
    per_cell = feature in E.columns
    lo, hi = ([0.0] if per_cell else []), []
    if row in PAIRED_ROWS:                          # the paired biopsies, one line each
        r = t.iloc[0]
        vals = {arms[0]: r["vals_a"], arms[1]: r["vals_b"]}
        for va, vb in zip(r["vals_a"], r["vals_b"]):
            ax.plot([pos[arms[0]], pos[arms[1]]], [va, vb], color="#9e9e9e", linewidth=0.4,
                    alpha=0.5, zorder=2)
    else:
        vals = {a: unit_means(a, {**KEY, **KEY_S}[row])[0][feature].dropna().to_numpy()
                for a in arms}
    for a in arms:
        if per_cell:
            c = E[feature].to_numpy()[SUB[a]]
            _violin(ax, c, pos[a], ARM_COL[a], w)
            if (c > 0).sum() >= 20:
                hi.append(float(np.quantile(c[c > 0], 0.98)))
        v = np.asarray(vals[a], float)
        jit = np.zeros(len(v)) if row in PAIRED_ROWS else rng.uniform(-0.2 * w, 0.2 * w, len(v))
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
        if row in PAIRED_ROWS:
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


def det_ticks(ax, R, genes, row="A"):
    """% of cells detected under each gene tick: malignant (purple) | benign (blue), MF."""
    r = R[R["row"] == row].set_index("feature")
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
    """One-line summary of the program-score tests, for the §LLM report."""
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
        f"detected in both arms or < {MIN_N} units. Call: mal_cluster (per-sample cluster call); CD4 "
        f"with no call excluded. RNA "
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
    fig.suptitle(f"{PRETTY[p]} program: score and genes in malignant vs benign CD4", fontsize=9,
                 x=0.01, ha="left", y=0.985)
    caption(fig, NOTE + (" Lamina score = log2 of the ratio of unit-mean CP10K; lower = more "
                         "deformable nucleus." if p == LAMINA else ""))
    save_fig(fig, p)


print("drawing utilities ready")
```

```
drawing utilities ready
```

```python  # cell 8
# ============================================================================
# F  One figure per program — score first, then every gene; rows A, B, C
# ============================================================================
for p in PROGRAMS:
    program_figure(p)
```

```
<Figure size 940x820 with 6 Axes>
  -> amoeboid/amb28cmc_osmo_engine.png/.pdf

<Figure size 1130x820 with 6 Axes>
  -> amoeboid/amb28cmc_actomyosin.png/.pdf

<Figure size 1130x820 with 6 Axes>
  -> amoeboid/amb28cmc_actin_piston.png/.pdf

<Figure size 655x820 with 6 Axes>
  -> amoeboid/amb28cmc_nuclear_repair.png/.pdf

<Figure size 655x820 with 6 Axes>
  -> amoeboid/amb28cmc_mechanosensor.png/.pdf

<Figure size 560x820 with 6 Axes>
  -> amoeboid/amb28cmc_nuclear_lamina.png/.pdf
```

```python  # cell 9
# ============================================================================
# F_stage  One stage figure per program — rows A and B split by stage
# ============================================================================
# Ae / Aa = row A inside early / advanced biopsies; on Aa, under each star, the gap test (Ad):
# malignant - benign, advanced vs early (↑ = larger gap in advanced). Bs = donor means by stage.
def row_title_s(row, R):
    s = R[(R["row"] == row) & R["feature"].str.startswith("score_")]
    if row == "Ae":
        return (f"A · early  malignant vs benign CD4, paired within early MF biopsy "
                f"({int(s['n_a'].iloc[0])} biopsies)")
    if row == "Aa":
        g = R[(R["row"] == "Ad") & R["feature"].str.startswith("score_")].iloc[0]
        return (f"A · advanced  paired within advanced MF biopsy ({int(s['n_a'].iloc[0])} "
                f"biopsies); 'gap' = malignant - benign, advanced vs early "
                f"({int(g['n_a'])} advanced | {int(g['n_b'])} early donors)")
    n = {a: len(unit_means(a, "donor")[0]) for a in ROWS_S["Bs"]}
    return ("B · stage  donor means, early vs advanced and vs healthy-donor CD4 ("
            + " | ".join(f"{ARM_SHORT[a]} {n[a]}" for a in ROWS_S["Bs"]) + " donors)")


def finding_s(p, R):
    """One-line summary of the stage figure's program-score tests, for the §LLM report."""
    s = R[R["feature"] == f"score_{p}"].set_index(["row", "arm_a", "arm_b"])["direction"]

    def rel(key, left, right):
        d = s.get(key, "not tested")
        return {"up": f"{left} > {right}", "down": f"{left} < {right}",
                "ns": f"{left} ≈ {right}"}.get(d, f"{left} vs {right} n.t.")

    return (f"{PRETTY[p]} by stage: " + " · ".join([
        rel(("Ae", "mal_early", "ben_early"), "malignant", "benign (early)"),
        rel(("Aa", "mal_adv", "ben_adv"), "malignant", "benign (advanced)"),
        rel(("Ad", "gap_adv", "gap_early"), "gap in advanced", "early"),
        rel(("Bs", "mal_early", "mal_adv"), "early", "advanced malignant")]))


def gap_annot(ax, R, feats):
    """Row Ad under the Aa stars: 'gap' + star + ↑ (larger in advanced) / ↓ (smaller)."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    d = R[R["row"] == "Ad"].set_index("feature")
    for x, f in enumerate(feats):
        r = d.loc[f]
        arrow = {"up": "↑", "down": "↓"}.get(r["direction"], "")
        ax.text(x, 0.77, f"gap {sig(r['q'])}{arrow}", transform=tr, ha="center", va="top",
                fontsize=5, color={"up": STAGE_COL["advanced"],
                                   "down": STAGE_COL["early"]}.get(r["direction"], "#555555"))


_mal_s = ARMS_S[ARMS_S["arm"].str.startswith("mal")]
NOTE_S = (NOTE + f" Stage = stage_class (early IA-IIA, advanced IIB+), MF only; {N_NOSTAGE} MF "
          "samples of unknown stage left out; "
          + "; ".join(f"{r.arm} {r.n_samples} samples / {r.n_donors} donors ({r.study})"
                      for r in _mal_s.itertuples())
          + ". Early vs advanced is pooled across studies (donor-level Mann-Whitney); gap = the "
          "per-biopsy malignant - benign difference averaged per donor. Row C is not split by "
          "stage (MF layer-split donors: "
          + ", ".join(f"{n} {s}" for s, n in LAYER_STAGE.items()) + ").")


def program_figure_stage(p):
    R = RES_S[RES_S["program"] == p]
    genes = PROGRAMS[p]
    G = len(genes)
    fig = plt.figure(figsize=(1.8 + 0.95 * (G + 1), 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, G], wspace=2.0 / (G + 1), hspace=0.8,
                          left=0.1, right=0.98, top=0.9, bottom=0.1)
    for i, row in enumerate(ROWS_S):
        arms = ROWS_S[row]
        ax_s, ax_g = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        _room(ax_s, *slot(ax_s, 0, row, f"score_{p}", R))
        ext = [slot(ax_g, j, row, g, R) for j, g in enumerate(genes)]
        _room(ax_g, min(e[0] for e in ext), max(e[1] for e in ext))
        ax_s.set_xticks(OFFSETS[len(arms)])
        ax_s.set_xticklabels([ARM_SHORT[a] for a in arms], rotation=90, fontsize=5.5)
        for t, a in zip(ax_s.get_xticklabels(), arms):
            t.set_color(STAGE_COL.get(ARM_STAGE.get(a), "#111111"))
        ax_s.set_xlim(-0.5, 0.5)
        ax_s.set_ylabel(SCORE_YLAB.get(p, "program score\nmean log1p CP10K"), fontsize=6)
        ax_g.set_xticks(range(G))
        ax_g.set_xticklabels(genes, style="italic", fontsize=6.5)
        ax_g.set_xlim(-0.5, G - 0.5)
        ax_g.set_ylabel("log1p CP10K", fontsize=6)
        if row in PAIRED_ROWS:
            det_ticks(ax_g, R, genes, row=row)
        if row == "Aa":
            gap_annot(ax_s, R, [f"score_{p}"])
            gap_annot(ax_g, R, genes)
        ax_s.text(0, 1.1, row_title_s(row, R), transform=ax_s.transAxes, ha="left", va="bottom",
                  fontsize=7, fontweight="bold",
                  color=STAGE_COL.get({"Ae": "early", "Aa": "advanced"}.get(row), "#111111"))
        for ax in (ax_s, ax_g):
            _clean(ax)
    fig.suptitle(f"{PRETTY[p]} program: score and genes in malignant vs benign CD4, early and "
                 "advanced MF", fontsize=9, x=0.01, ha="left", y=0.985)
    caption(fig, NOTE_S + (" Lamina score = log2 of the ratio of unit-mean CP10K; lower = more "
                           "deformable nucleus." if p == LAMINA else ""))
    save_fig(fig, f"{p}_stage")


for p in PROGRAMS:
    program_figure_stage(p)
```

```
<Figure size 940x820 with 6 Axes>
  -> amoeboid/amb28cmc_osmo_engine_stage.png/.pdf

<Figure size 1130x820 with 6 Axes>
  -> amoeboid/amb28cmc_actomyosin_stage.png/.pdf

<Figure size 1130x820 with 6 Axes>
  -> amoeboid/amb28cmc_actin_piston_stage.png/.pdf

<Figure size 655x820 with 6 Axes>
  -> amoeboid/amb28cmc_nuclear_repair_stage.png/.pdf

<Figure size 655x820 with 6 Axes>
  -> amoeboid/amb28cmc_mechanosensor_stage.png/.pdf

<Figure size 560x820 with 6 Axes>
  -> amoeboid/amb28cmc_nuclear_lamina_stage.png/.pdf
```

```python  # cell 10
# ============================================================================
# §R  What was written
# ============================================================================
print("tables ->", TAB_DIR)
for n in ("arms", "tests", "tests_depthmatched", "unit_means", "arms_stage", "tests_stage"):
    print("  ", tab28c(n).name)
print("figures ->", FIG_DIR)
for p in PROGRAMS:
    print(f"   {PREFIX}{p}.png/.pdf, {PREFIX}{p}_stage.png/.pdf")
print("\nRNA presence != migration: every number above is machinery, not measured motility.")
```

```
tables -> /home/projects/nyosef/zvise/MF/tables
   amb28cmc_arms.csv
   amb28cmc_tests.csv
   amb28cmc_tests_depthmatched.csv
   amb28cmc_unit_means.csv
   amb28cmc_arms_stage.csv
   amb28cmc_tests_stage.csv
figures -> /home/projects/nyosef/zvise/MF/figures/amoeboid
   amb28cmc_osmo_engine.png/.pdf, amb28cmc_osmo_engine_stage.png/.pdf
   amb28cmc_actomyosin.png/.pdf, amb28cmc_actomyosin_stage.png/.pdf
   amb28cmc_actin_piston.png/.pdf, amb28cmc_actin_piston_stage.png/.pdf
   amb28cmc_nuclear_repair.png/.pdf, amb28cmc_nuclear_repair_stage.png/.pdf
   amb28cmc_mechanosensor.png/.pdf, amb28cmc_mechanosensor_stage.png/.pdf
   amb28cmc_nuclear_lamina.png/.pdf, amb28cmc_nuclear_lamina_stage.png/.pdf

RNA presence != migration: every number above is machinery, not measured motility.
```

```python  # cell 11
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
P("NOTEBOOK 28c_mc — AMOEBOID (NON-PROTEOLYTIC) MOTILITY MACHINERY IN MALIGNANT vs BENIGN CD4 T CELLS,")
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
P("- Malignancy call = mal_cluster: each sample's T cells are Leiden-clustered and every cluster "
  "is called malignant or benign -- by ALICE TCR clonality where the sample has TCR (tcr), else "
  "by a gene program + V-gene skew learned on the ALICE clusters (auto), else manually (user). "
  f"A cell takes its cluster's call. CD4 with no call ({int(LV.eq('CD4_unassessed').sum()):,} "
  "cells) is EXCLUDED. Tregs are a separate cell type and excluded. Call provenance per arm is "
  "in the arms table (call_source). Motility genes in the calling program: "
  f"{IN_CALL_PROGRAM or 'none'}.")
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
P(f"- Depth control: row A was re-run with every MF CD4 cell thinned to {DM_DEPTH:,} UMIs; "
  f"{int(DM['survives'].sum())} of {int(DM['direction_full_depth'].isin(['up', 'down']).sum())} "
  "significant row-A results keep their direction at q < 0.05 (amb28cmc_tests_depthmatched).")
P(f"- Row C has {len(LAYER_DONORS)} donors ("
  + ", ".join(f"{s} {n}" for s, n in o.loc[o["donor"].isin(LAYER_DONORS)]
              .groupby("donor")["study"].first().value_counts().items())
  + "); a donor contributes one or several biopsies per layer.")

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
P("ALL TESTS, PER PROGRAM FIGURE (figure file, score summary, then every test drawn)")
P("=" * 100)
_cols = ["row", "feature", "arm_a", "arm_b", "test", "n_a", "n_b", "n_pos", "n_neg", "med_a",
         "med_b", "delta", "det_a", "det_b", "p", "q", "direction", "reason"]
_fmt = {"med_a": "{:.3f}".format, "med_b": "{:.3f}".format, "delta": "{:+.3f}".format,
        "det_a": "{:.3f}".format, "det_b": "{:.3f}".format, "p": "{:.3g}".format,
        "q": "{:.3g}".format, "n_pos": lambda x: _g(x, ".0f"), "n_neg": lambda x: _g(x, ".0f")}
for p in PROGRAMS:
    R = RES[RES["program"] == p]
    P(f"\n--- {p}: {', '.join(PROGRAMS[p])} | figure {PREFIX}{p}.png")
    P(f"summary: {finding(p, R)}")
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

P("\n" + "=" * 100)
P(f"ROW A AT EQUAL SEQUENCING DEPTH ({DM_DEPTH:,} UMIs per cell; sensitivity, not drawn)")
P("=" * 100)
with pd.option_context("display.width", 400, "display.max_rows", None):
    P(DM[["program", "feature", "n_pairs", "n_pos", "n_neg", "delta", "q", "direction",
          "delta_full_depth", "q_full_depth", "direction_full_depth", "survives"]]
      .to_string(index=False, formatters=_fmt))

P("\n" + "=" * 100)
P(f"STAGE-STRATIFIED (figures {PREFIX}<program>_stage.png; tests {tab28c('tests_stage').name})")
P("=" * 100)
ARM_LABEL_S = {**ARM_LABEL, "mal_early": "early-stage MF malignant CD4",
               "mal_adv": "advanced-stage MF malignant CD4",
               "ben_early": "early-stage MF benign CD4", "ben_adv": "advanced-stage MF benign CD4",
               "gap_adv": "advanced-stage malignant - benign gap",
               "gap_early": "early-stage malignant - benign gap"}
P(f"- Stage = stage_class: early = IA-IIA, advanced = IIB+ (MF only). {N_NOSTAGE} MF samples with "
  "a call and unknown stage are left out.")
P("- Ae / Aa: row A inside early / inside advanced biopsies (paired signed-rank). Bs: donor means; "
  "malignant and benign early vs advanced (Mann-Whitney, POOLED ACROSS STUDIES) and each "
  "malignant stage vs healthy. Ad: per-biopsy malignant - benign difference averaged per donor, "
  "a = advanced vs b = early, Mann-Whitney (study and chemistry cancel inside the difference). "
  "BH over all tests of one program's stage figure.")
P("- Row C is not split by stage; its MF layer-split donors: "
  + ", ".join(f"{n} {s}" for s, n in LAYER_STAGE.items()) + ".")
P(ARMS_S.to_string(index=False))
for p in PROGRAMS:
    R = RES_S[RES_S["program"] == p]
    P(f"\n--- {p} | figure {PREFIX}{p}_stage.png")
    P(f"summary: {finding_s(p, R)}")
    with pd.option_context("display.width", 400, "display.max_rows", None,
                           "display.max_colwidth", 60):
        P(R[_cols].to_string(index=False, formatters=_fmt))
P("\nSTAGE-STRATIFIED SIGNIFICANT RESULTS (q < 0.05, BH within program's stage figure)")
for r in RES_S[RES_S["q"] < 0.05].itertuples():
    what = "PROGRAM SCORE" if r.feature.startswith("score_") else f"gene {r.feature}"
    hi_, lo_ = (r.arm_a, r.arm_b) if r.direction == "up" else (r.arm_b, r.arm_a)
    votes = (f"; {r.n_pos:.0f}/{r.n_a} units a>b, {r.n_neg:.0f} a<b"
             if r.test.startswith("paired") else f"; n = {r.n_a} vs {r.n_b} donors")
    P(f"  [{r.program}] {what}, row {r.row}: {ARM_LABEL_S[hi_]} > {ARM_LABEL_S[lo_]} "
      f"(a={r.arm_a} median {_g(r.med_a)}, b={r.arm_b} median {_g(r.med_b)}, "
      f"delta {_g(r.delta, '+.3f')}{votes}; q = {_g(r.q, '.2g')})")

_nt = RES[RES["direction"] == "not tested"]
P(f"\nNOT TESTABLE: {len(_nt)} of {len(RES)} tests")
for (f, reason), d in _nt.groupby(["feature", "reason"]):
    P(f"  {f}: {reason} [{', '.join(d['row'] + ':' + d['arm_a'] + '-' + d['arm_b'])}]")
print("\n".join(out))
```

```
====================================================================================================
NOTEBOOK 28c_mc — AMOEBOID (NON-PROTEOLYTIC) MOTILITY MACHINERY IN MALIGNANT vs BENIGN CD4 T CELLS,
CUTANEOUS T-CELL LYMPHOMA (CTCL) SKIN — single-cell RNA-seq, atlas v2
====================================================================================================

QUESTION: does the malignant T-cell clone carry more of the amoeboid (non-proteolytic) migration machinery than benign CD4 T cells from the same biopsy, than healthy-skin CD4, and does that differ between epidermis and dermis? RNA presence of machinery components is measured, NOT motility. Intact amoeboid machinery is the normal T-cell baseline, so malignant ~= benign is a valid result, not a failure.

DATA: skin T-cell object skin_T_tcr_annotated_v4.h5ad (539,916 cells, 42,347 genes); 351,367 cells used.

ARMS
- Malignancy call = mal_cluster: each sample's T cells are Leiden-clustered and every cluster is called malignant or benign -- by ALICE TCR clonality where the sample has TCR (tcr), else by a gene program + V-gene skew learned on the ALICE clusters (auto), else manually (user). A cell takes its cluster's call. CD4 with no call (39,692 cells) is EXCLUDED. Tregs are a separate cell type and excluded. Call provenance per arm is in the arms table (call_source). Motility genes in the calling program: none.
- Healthy = every CD4 cell from healthy-donor skin (disease == HC), no TCR requirement.
- Rows A and B use MF skin only. Row C uses layer-split donors: >= 25 malignant and >= 25 benign CD4 in BOTH epidermis and dermis (any CTCL subtype).
- Arm codes: mal = MF malignant CD4; ben = MF benign CD4; hc = healthy-donor CD4; epi_mal = epidermal malignant CD4; epi_ben = epidermal benign CD4; derm_mal = dermal malignant CD4; derm_ben = dermal benign CD4
     arm  n_cells  n_donors  n_samples                call_source                                                      tech  median_library_size
     mal   185643        84         92 auto 53%, tcr 45%, user 2% 10x_5p 161,364, 10x_flex 20,613, 10x_3p 3,334, ECCITE 332                 3209
     ben   137664        86         98 tcr 64%, auto 31%, user 5%  10x_5p 127,139, 10x_3p 6,086, 10x_flex 4,137, ECCITE 302                 1517
      hc     4742        22         20           hc 81%, none 19%                       10x_5p 4,169, 10x_3p 555, ECCITE 18                 2096
 epi_mal    25254        10         14  tcr 96%, auto 3%, user 1%                                             10x_5p 25,254                 3302
 epi_ben     7921        10         14 tcr 82%, user 10%, auto 8%                                              10x_5p 7,921                 2011
derm_mal    33147        10         14  tcr 96%, user 3%, auto 0%                                             10x_5p 33,147                 2729
derm_ben    66085        10         14           tcr 98%, auto 2%                                             10x_5p 66,085                 1246

Layer-split donors (cells per layer|class; in_C = used in row C):
                     Epid|mal  Epid|ben  Derm|mal  Derm|ben disease   in_C
donor                                                                     
B5__PT09                 4867      3591      5560      1682      SS   True
B5__PT22                   10      4275         0      3362      SS  False
B5__PT23                  165       427      6206       820      SS   True
Li2024_atlas__CTCL1      2513       105      5163       247      MF   True
Li2024_atlas__CTCL2       968        46       389      2154      MF   True
Li2024_atlas__CTCL3      2776        31      6047      1747      MF   True
Li2024_atlas__CTCL4      1142       917        99      6989      MF   True
Li2024_atlas__CTCL5      5094       407      2438      3666      MF   True
Li2024_atlas__CTCL6      3462       550       475      9748      MF   True
Li2024_atlas__CTCL7      3088      1721       570     36711      MF   True
Li2024_atlas__CTCL8      11
... [168185 more chars truncated]
```
